"""
Réglages de génération contre les boucles : mesure objective, sans réentraîner le modèle.

Chaque méthode continue les mêmes débuts de textes (jeux de test, jamais vus à l'entraînement), avec les mêmes
graines. Mesures (sens souhaité entre parenthèses) :
  - SER@4 : part des positions où les 4 derniers tokens répètent un passage déjà écrit (plus bas) ;
  - 4-grammes répétés : part des 4-grammes déjà vus dans la suite (plus bas) ;
  - distinct-2 / distinct-3 : part de 2-/3-grammes différents (plus haut) ;
  - perplexité jugée : vraisemblance de la suite selon un modèle plus grand (CroissantLLM 1,3B). Trop basse =
    texte qui boucle (facile à prédire), trop haute = texte incohérent ; on la compare au texte humain.
Méthodes : glouton, échantillonnage simple, top-p (Holtzman 2020), min-p (Nguyen et al. 2025), pénalité de
répétition (CTRL, Keskar 2019), interdiction des 3-grammes répétés, DRY (Weidmann et al. 2026).

Usage : python eval_generation.py ../out/etude_hq-ddp/hq-ddp_768.pth --device cpu
"""

import argparse
import json
import math
import os
import random
import sys

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

import torch
import torch.nn.functional as F
from transformers import AutoTokenizer

from model.model_minimind import MiniMindConfig, MiniMindForCausalLM

TEST_DIR = os.path.join(os.path.dirname(__file__), "..", "dataset", "fr_en", "test")

METHODES = {
    "glouton": dict(temperature=0),
    "tirage T=0,8": dict(temperature=0.8),
    "top-p 0,9": dict(temperature=0.8, top_p=0.9),
    "min-p 0,05": dict(temperature=0.8, min_p=0.05),
    "pénalité 1,2 (glouton)": dict(temperature=0, repetition_penalty=1.2),
    "sans 3-gramme répété (glouton)": dict(temperature=0, no_repeat_ngram=3),
    "min-p 0,05 + DRY": dict(temperature=0.8, min_p=0.05, dry=(0.8, 1.75, 2)),
}


def dry_penalty(seq, logits, multiplier, base, allowed):
    """DRY : pour chaque token candidat qui prolongerait une répétition d'au moins `allowed` tokens déjà vue,
    pénalité multiplier · base^(longueur - allowed). Version directe en O(n²), suffisante pour 100 tokens."""
    n = len(seq)
    best = {}
    for i in range(
        n - 1
    ):  # le token seq[i+1] a suivi le contexte se terminant en seq[i]
        length = 0
        while length <= i and seq[i - length] == seq[n - 1 - length] and length < 50:
            length += 1
        if length >= allowed:
            nxt = seq[i + 1]
            best[nxt] = max(best.get(nxt, 0), length)
    for tok, length in best.items():
        logits[tok] -= multiplier * base ** (length - allowed)
    return logits


@torch.no_grad()
def generate(
    model,
    prompt,
    length,
    temperature=0.0,
    top_p=None,
    min_p=None,
    repetition_penalty=None,
    no_repeat_ngram=None,
    dry=None,
    seed=0,
):
    g = torch.Generator(device=prompt.device).manual_seed(seed)
    out = model(prompt, use_cache=True)
    past, logits = out.past_key_values, out.logits[0, -1].float()
    seq = prompt[0].tolist()
    for _ in range(length):
        if repetition_penalty:
            seen = torch.tensor(sorted(set(seq)), device=logits.device)
            vals = logits[seen]
            logits[seen] = torch.where(
                vals > 0, vals / repetition_penalty, vals * repetition_penalty
            )
        if no_repeat_ngram and len(seq) >= no_repeat_ngram:
            prefix = tuple(seq[-(no_repeat_ngram - 1) :])
            for i in range(len(seq) - no_repeat_ngram + 1):
                if tuple(seq[i : i + no_repeat_ngram - 1]) == prefix:
                    logits[seq[i + no_repeat_ngram - 1]] = -math.inf
        if dry:
            logits = dry_penalty(seq, logits, *dry)
        if temperature == 0:
            nxt = int(logits.argmax())
        else:
            probs = (logits / temperature).softmax(-1)
            if min_p:
                probs = torch.where(probs >= min_p * probs.max(), probs, 0)
            if top_p:
                sp, si = probs.sort(descending=True)
                drop = sp.cumsum(-1) - sp > top_p
                probs[si[drop]] = 0
            nxt = int(torch.multinomial(probs / probs.sum(), 1, generator=g))
        seq.append(nxt)
        out = model(
            torch.tensor([[nxt]], device=prompt.device),
            past_key_values=past,
            use_cache=True,
        )
        past, logits = out.past_key_values, out.logits[0, -1].float()
    return seq[prompt.size(1) :]


def metrics(ids):
    n = len(ids)
    grams = lambda k: [tuple(ids[i : i + k]) for i in range(n - k + 1)]
    g4 = grams(4)
    seen, rep4 = set(), 0
    for g in g4:
        rep4 += g in seen
        seen.add(g)
    ser = sum(
        1
        for i in range(4, n + 1)
        if any(ids[i - 4 : i] == ids[j - 4 : j] for j in range(4, i))
    ) / max(1, n - 3)
    distinct = lambda k: len(set(grams(k))) / max(1, len(grams(k)))
    return dict(
        ser4=ser,
        rep4=rep4 / max(1, len(g4)),
        distinct2=distinct(2),
        distinct3=distinct(3),
    )


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("poids")
    parser.add_argument(
        "--tokenizer_path", default="../dataset/fr_en/tokenizer_fr65_16k"
    )
    parser.add_argument(
        "--juge",
        default="croissantllm/CroissantLLMBase",
        help="modèle Hugging Face qui note la plausibilité ('' = aucun)",
    )
    parser.add_argument(
        "--jeux", default="wikipedia_fr,web_fr_selectionne,wikisource_fr"
    )
    parser.add_argument("--n", type=int, default=20)
    parser.add_argument("--debut", type=int, default=32)
    parser.add_argument("--longueur", type=int, default=96)
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--mlflow_experiment", default="minimind-eval-generation")
    args = parser.parse_args()

    tok = AutoTokenizer.from_pretrained(args.tokenizer_path)
    model = MiniMindForCausalLM(MiniMindConfig(vocab_size=len(tok)))
    model.load_state_dict(
        {k: v.float() for k, v in torch.load(args.poids, map_location="cpu").items()}
    )
    model = model.to(args.device).eval()

    rng, cases = random.Random(0), []
    for name in args.jeux.split(","):
        docs = [
            json.loads(line)["text"]
            for line in open(os.path.join(TEST_DIR, f"{name}.jsonl"), encoding="utf-8")
        ]
        for text in rng.sample(docs, args.n * 3):
            ids = tok(text, add_special_tokens=False).input_ids
            if len(ids) >= args.debut + args.longueur:
                cases.append(
                    (ids[: args.debut], ids[args.debut : args.debut + args.longueur])
                )
            if len([c for c in cases if True]) >= args.n * (
                args.jeux.split(",").index(name) + 1
            ):
                break

    texts = {"texte humain (repère)": [(p, h) for p, h in cases]}
    for name, kw in METHODES.items():
        texts[name] = []
        for i, (p, _) in enumerate(cases):
            prompt = torch.tensor([[tok.bos_token_id] + p], device=args.device)
            texts[name].append(
                (p, generate(model, prompt, args.longueur, seed=i, **kw))
            )
        print(f"généré : {name}", flush=True)

    judge = judge_tok = None
    if args.juge:
        from transformers import AutoModelForCausalLM

        judge_tok = AutoTokenizer.from_pretrained(args.juge)
        judge = (
            AutoModelForCausalLM.from_pretrained(args.juge, dtype=torch.float32)
            .to(args.device)
            .eval()
        )

    rows = {}
    for name, items in texts.items():
        ms = [metrics(c) for _, c in items]
        row = {k: sum(m[k] for m in ms) / len(ms) for k in ms[0]}
        if judge is not None:
            nll, count = 0.0, 0
            for p, c in items:
                ctx, cont = tok.decode(p), tok.decode(c)
                a = judge_tok(ctx, return_tensors="pt").input_ids.to(args.device)
                b = judge_tok(
                    cont, add_special_tokens=False, return_tensors="pt"
                ).input_ids.to(args.device)
                with torch.no_grad():
                    logits = (
                        judge(torch.cat([a, b], 1))
                        .logits[0, a.size(1) - 1 : -1]
                        .float()
                    )
                nll += F.cross_entropy(logits, b[0], reduction="sum").item()
                count += b.size(1)
            row["perplexite_jugee"] = math.exp(nll / count)
        rows[name] = row
        print(
            f"{name:32} " + "  ".join(f"{k} {v:.3f}" for k, v in row.items()),
            flush=True,
        )

    example = cases[0][0]
    print("\nDébut imposé :", tok.decode(example).replace("\n", " ⏎ "))
    for name, items in texts.items():
        print(f"\n[{name}] {tok.decode(items[0][1])[:300]}".replace("\n", " ⏎ "))

    os.makedirs("../out/evals", exist_ok=True)
    with open("../out/evals/generation.json", "w") as f:
        json.dump(
            {"poids": args.poids, "cas": len(cases), "resultats": rows},
            f,
            indent=1,
            ensure_ascii=False,
        )
    if args.mlflow_experiment:
        import mlflow

        mlflow.set_tracking_uri(
            os.environ.get("MINIMIND_MLFLOW_URI", "https://mlflow.tgu.ovh")
        )
        mlflow.set_experiment(args.mlflow_experiment)
        for name, row in rows.items():
            with mlflow.start_run(run_name=name):
                mlflow.set_tag("poids", args.poids)
                mlflow.log_metrics(row)
