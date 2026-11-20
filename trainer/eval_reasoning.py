"""
Banc d'essai de raisonnement pour un modèle de conversation (après SFT).

Problèmes d'arithmétique générés par programme, donc à réponse certaine, classés par NOMBRE D'ÉTAPES
(1, 2 ou 3 opérations enchaînées) et par langue. Un problème à 2 étapes oblige à réutiliser le résultat
de la première : c'est exactement ce qu'un petit modèle ne sait pas faire.
  « Paul a 5 pommes. Il en mange 2. Il en achète 4. Combien de pommes a-t-il maintenant ? »  → 7
Score : le dernier nombre de la réponse doit être le bon (décodage déterministe, sans tirage).
S'y ajoutent des questions pièges, affichées telles quelles pour lecture humaine (pas de score fiable).

Usage : cd trainer && python eval_reasoning.py ../out/etude_sft-full/sft-full-t0_768_best.pth --langs zh,en
"""

import argparse
import os
import random
import re
import sys

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

import torch

# Gabarits par langue : (début, retrait, ajout, question). {n} = prénom, {a}/{x} = nombres, {o} = objet.
TEMPLATES = {
    "fr": dict(
        names=["Paul", "Marc", "Léo", "Hugo"],
        objects=["pommes", "billes", "livres", "pièces"],
        start="{n} a {a} {o}.",
        minus=["Il en perd {x}.", "Il en donne {x}.", "Il en vend {x}."],
        plus=["Il en achète {x}.", "Il en reçoit {x}.", "Il en trouve {x}."],
        ask="Combien de {o} a-t-il maintenant ?",
    ),
    "en": dict(
        names=["Paul", "Mark", "Leo", "Tom"],
        objects=["apples", "marbles", "books", "coins"],
        start="{n} has {a} {o}.",
        minus=[
            "He loses {x} of them.",
            "He gives away {x} of them.",
            "He sells {x} of them.",
        ],
        plus=["He buys {x} more.", "He receives {x} more.", "He finds {x} more."],
        ask="How many {o} does he have now?",
    ),
    "zh": dict(
        names=["小明", "小华", "小强", "小军"],
        objects=["苹果", "弹珠", "书", "硬币"],
        start="{n}有{a}个{o}。",
        minus=["他丢了{x}个。", "他送出了{x}个。", "他卖了{x}个。"],
        plus=["他又买了{x}个。", "他又得到了{x}个。", "他又找到了{x}个。"],
        ask="他现在有多少个{o}？",
    ),
}
# Questions pièges : la bonne réponse contredit le réflexe (ici, il faut y aller EN VOITURE : c'est elle qu'on lave).
TRAPS = {
    "fr": [
        "Je dois laver ma voiture à la station de lavage, qui est à 50 mètres. J'y vais à pied ou en voiture ?"
    ],
    "en": [
        "I need to wash my car at the car wash, which is 50 meters away. Should I walk there or drive there?"
    ],
    "zh": ["我要去洗车，洗车店离我家只有50米。我应该走路去还是开车去？"],
}


def make_problems(lang, n_ops, count, seed=0):
    """Retourne `count` couples (énoncé, réponse) à n_ops opérations ; les résultats restent positifs."""
    rng, t, problems = random.Random(f"{lang}-{n_ops}-{seed}"), TEMPLATES[lang], []
    for _ in range(count):
        obj, total = rng.choice(t["objects"]), rng.randint(6, 20)
        parts = [t["start"].format(n=rng.choice(t["names"]), a=total, o=obj)]
        for _ in range(n_ops):
            if total >= 3 and rng.random() < 0.5:
                x = rng.randint(1, min(total - 1, 9))
                total -= x
                parts.append(rng.choice(t["minus"]).format(x=x))
            else:
                x = rng.randint(1, 9)
                total += x
                parts.append(rng.choice(t["plus"]).format(x=x))
        parts.append(t["ask"].format(o=obj))
        problems.append(((" " if lang != "zh" else "").join(parts), total))
    return problems


@torch.no_grad()
def answer(model, tokenizer, question, max_new_tokens=120):
    text = tokenizer.apply_chat_template(
        [{"role": "user", "content": question}],
        tokenize=False,
        add_generation_prompt=True,
        open_thinking=False,
    )
    ids = tokenizer(text, return_tensors="pt").input_ids.to(
        next(model.parameters()).device
    )
    out = model.generate(
        ids,
        max_new_tokens=max_new_tokens,
        do_sample=False,
        eos_token_id=tokenizer.eos_token_id,
    )
    return tokenizer.decode(out[0, ids.shape[1] :], skip_special_tokens=True)


def reasoning_eval(model, tokenizer, langs=("fr", "en"), count=30, max_ops=3):
    """Retourne ({nom: taux de réussite}, {langue: réponses aux questions pièges})."""
    was_training = model.training
    model.eval()
    metrics, traps = {}, {}
    for lang in langs:
        for n_ops in range(1, max_ops + 1):
            ok = 0
            for question, expected in make_problems(lang, n_ops, count):
                numbers = re.findall(
                    r"\d+", answer(model, tokenizer, question).split("</think>")[-1]
                )
                ok += bool(numbers) and int(numbers[-1]) == expected
            metrics[f"raisonnement_{lang}_{n_ops}op"] = ok / count
        traps[lang] = [(q, answer(model, tokenizer, q, 150)) for q in TRAPS[lang]]
    model.train(was_training)
    return metrics, traps


if __name__ == "__main__":
    from transformers import AutoTokenizer

    from model.model_minimind import MiniMindConfig, MiniMindForCausalLM

    parser = argparse.ArgumentParser()
    parser.add_argument("checkpoint")
    parser.add_argument("--tokenizer_path", default="../model")
    parser.add_argument("--langs", default="fr,en")
    parser.add_argument("--count", type=int, default=30)
    args = parser.parse_args()

    tok = AutoTokenizer.from_pretrained(args.tokenizer_path)
    model = MiniMindForCausalLM(MiniMindConfig(vocab_size=len(tok)))
    model.load_state_dict(
        {
            k: v.float()
            for k, v in torch.load(args.checkpoint, map_location="cpu").items()
        }
    )
    metrics, traps = reasoning_eval(
        model.cuda(), tok, args.langs.split(","), args.count
    )
    for name, value in metrics.items():
        print(f"{name:26} {value:.0%}")
    for lang, items in traps.items():
        for q, a in items:
            print(f"\n[piège {lang}] {q}\n→ {a[:400]}")
    for lang in args.langs.split(","):
        print(f"\nexemple {lang} : {make_problems(lang, 2, 1)[0]}")
