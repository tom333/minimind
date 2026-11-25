"""
Corpus de pré-entraînement avec du raisonnement écrit : notre corpus sélectionné (français et anglais), mélangé à des
problèmes de mathématiques résolus pas à pas.

Source des problèmes : MetaMathQA (Yu et al. 2023, licence MIT), 395 000 problèmes réécrits à partir des jeux
d'entraînement de GSM8K et MATH, chacun avec sa solution détaillée, en anglais. Les jeux de TEST de GSM8K n'y figurent
pas ; mais les problèmes en sont proches : un score sur GSM8K mesurera surtout cet apprentissage, pas un raisonnement
général. La compréhension d'un texte en français (Belebele) reste, elle, hors de ces données.
Format : « Question: … / Answer: … », celui des consignes de GSM8K. Champ lang = "math" (perte suivie à part).
Part visée : environ un quart des tokens. Le reste : un tirage aléatoire (graine fixe) des documents du corpus sélectionné.

Usage : python -m corpus.melange_raisonnement --part_web 0.23
"""

import argparse
import json
import os
import random

from huggingface_hub import hf_hub_download

ROOT = os.path.join(os.path.dirname(__file__), "..", "dataset", "fr_en")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--web", default=os.path.join(ROOT, "pretrain_fr65hq.jsonl"))
    parser.add_argument("--part_web", type=float, default=0.23, help="part des documents du corpus sélectionné gardés")
    parser.add_argument("--sortie", default=os.path.join(ROOT, "pretrain_raisonnement_q3.jsonl"))
    args = parser.parse_args()

    rng = random.Random(0)
    problems = json.load(open(hf_hub_download("meta-math/MetaMathQA", "MetaMathQA-395K.json", repo_type="dataset")))
    chars = {"web": 0, "math": 0}
    with open(args.sortie, "w", encoding="utf-8") as out:
        with open(args.web, encoding="utf-8") as f:
            for line in f:
                if rng.random() < args.part_web:
                    doc = json.loads(line)
                    out.write(json.dumps({"text": doc["text"], "lang": doc.get("lang")}, ensure_ascii=False) + "\n")
                    chars["web"] += len(doc["text"])
        for p in problems:
            text = f"Question: {p['query'].strip()}\nAnswer: {p['response'].strip()}"
            out.write(json.dumps({"text": text, "lang": "math"}, ensure_ascii=False) + "\n")  # mêmes champs partout (chargement)
            chars["math"] += len(text)
    total = sum(chars.values())
    print(f"{args.sortie} : {total / 1e9:.2f} Md de caractères, dont mathématiques {chars['math'] / total:.1%} "
          f"({len(problems)} problèmes)")
