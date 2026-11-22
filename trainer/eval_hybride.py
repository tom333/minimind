"""
Mesure de la génération hybride (brouillon par diffusion, relecture et finalisation en mot à mot) face au
mot à mot seul, sur des débuts de textes tirés des jeux de test (jamais vus à l'entraînement).

Pour chaque début de texte, les deux méthodes écrivent la même longueur. On mesure :
  - acceptation : part des mots du brouillon gardés par le mot à mot (plus haut = la diffusion travaille) ;
    on la donne aussi pour le PREMIER bloc seul, avant qu'une éventuelle boucle de répétition ne la gonfle ;
  - passes par mot : nombre d'appels au modèle divisé par le nombre de mots (mot à mot seul = 1 ; plus bas =
    plus rapide, à coût de passe égal) ;
  - qualité : loss de la suite écrite, jugée par un modèle de référence fixe (plus bas = texte plus plausible),
    et part de 3-grammes répétés (plus bas = moins de boucles) ;
  - accord : part de mots identiques entre la sortie hybride et la sortie mot à mot.

Usage : python eval_hybride.py poids_mixte.pth --juge ../out/etude_fr-en-full/fr-en-full-t0_768.pth
"""

import argparse
import json
import os
import random
import sys
import time

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

import torch
import torch.nn.functional as F
from transformers import AutoTokenizer

from model.model_minimind import MiniMindConfig, MiniMindForCausalLM
from trainer.diffusion import ar_generate, hybrid_generate, set_causal

TEST_DIR = os.path.join(os.path.dirname(__file__), "..", "dataset", "fr_en", "test")


def load(path, vocab, device):
    model = MiniMindForCausalLM(MiniMindConfig(vocab_size=vocab))
    model.load_state_dict(
        {k: v.float() for k, v in torch.load(path, map_location="cpu").items()}
    )
    return model.to(device).eval()


def repetition(ids):
    grams = [tuple(ids[i : i + 3]) for i in range(len(ids) - 2)]
    return 1 - len(set(grams)) / max(1, len(grams))


@torch.no_grad()
def judge_loss(judge, seq, start):
    logits = judge(seq).logits[0, start - 1 : -1].float()
    return F.cross_entropy(logits, seq[0, start:]).item()


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "mixte", help="poids d'un modèle entraîné aux deux objectifs (--objective both)"
    )
    parser.add_argument(
        "--juge",
        required=True,
        help="modèle mot à mot de référence qui note la qualité",
    )
    parser.add_argument(
        "--tokenizer_path", default="../dataset/fr_en/tokenizer_fr65_16k"
    )
    parser.add_argument(
        "--jeux", default="wikipedia_fr,web_fr_selectionne,web_en_educatif"
    )
    parser.add_argument("--n", type=int, default=20, help="débuts de textes par jeu")
    parser.add_argument("--debut", type=int, default=24, help="tokens du début imposé")
    parser.add_argument("--longueur", type=int, default=32)
    parser.add_argument("--block", type=int, default=8)
    parser.add_argument(
        "--steps",
        type=str,
        default="2,4,8",
        help="tours de démasquage par bloc (plusieurs valeurs = plusieurs réglages)",
    )
    parser.add_argument("--accept", type=float, default=0.1)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--mlflow_experiment", default="minimind-eval-hybride")
    args = parser.parse_args()

    tok = AutoTokenizer.from_pretrained(args.tokenizer_path)
    model, judge = (
        load(args.mixte, len(tok), args.device),
        load(args.juge, len(tok), args.device),
    )
    rng, prompts = random.Random(0), []
    for name in args.jeux.split(","):
        docs = [
            json.loads(line)["text"]
            for line in open(os.path.join(TEST_DIR, f"{name}.jsonl"), encoding="utf-8")
        ]
        for text in rng.sample(docs, args.n):
            ids = tok(text, add_special_tokens=False).input_ids[: args.debut]
            if len(ids) == args.debut:
                prompts.append(
                    (name, torch.tensor([[tok.bos_token_id] + ids], device=args.device))
                )

    rows = {}
    configs = [("mot à mot", None)] + [
        (f"hybride, {s} tours/bloc", int(s)) for s in args.steps.split(",")
    ]
    for label, steps in configs:
        acc, first, passes, loss, rep, agree, secs = [], [], [], [], [], [], 0.0
        for _, prompt in prompts:
            start = prompt.size(1)
            t = time.perf_counter()
            if steps is None:
                out = ar_generate(model, prompt, args.longueur)
                n_pass = args.longueur
            else:
                out, hist, rate, n_pass = hybrid_generate(
                    model,
                    prompt,
                    args.longueur,
                    steps=steps,
                    block=args.block,
                    accept=args.accept,
                )
                acc.append(rate)
                first.append(hist[0][1] / hist[0][2])
                agree.append(
                    (
                        out[0, start:]
                        == ar_generate(model, prompt, args.longueur)[0, start:]
                    )
                    .float()
                    .mean()
                    .item()
                )
            secs += time.perf_counter() - t
            set_causal(model, True)
            passes.append(n_pass / args.longueur)
            loss.append(judge_loss(judge, out, start))
            rep.append(repetition(out[0, start:].tolist()))
        mean = lambda v: sum(v) / len(v) if v else float("nan")
        rows[label] = dict(
            acceptation=mean(acc),
            acceptation_1er_bloc=mean(first),
            passes_par_mot=mean(passes),
            loss_juge=mean(loss),
            repetition_3g=mean(rep),
            accord_avec_mot_a_mot=mean(agree),
            secondes=secs,
        )
        print(
            f"{label:24} " + "  ".join(f"{k} {v:.3f}" for k, v in rows[label].items()),
            flush=True,
        )

    os.makedirs("../out/evals", exist_ok=True)
    with open("../out/evals/hybride.json", "w") as f:
        json.dump(
            {
                "mixte": args.mixte,
                "juge": args.juge,
                "prompts": len(prompts),
                "resultats": rows,
            },
            f,
            indent=1,
            ensure_ascii=False,
        )
    if args.mlflow_experiment:
        import mlflow

        from trainer.provenance import git_commit

        mlflow.set_tracking_uri(
            os.environ.get("MINIMIND_MLFLOW_URI", "https://mlflow.tgu.ovh")
        )
        mlflow.set_experiment(args.mlflow_experiment)
        for label, metrics in rows.items():
            with mlflow.start_run(run_name=label):
                mlflow.set_tags(
                    {"git_commit": git_commit(), "mixte": args.mixte, "juge": args.juge}
                )
                mlflow.log_metrics({k: v for k, v in metrics.items() if v == v})
