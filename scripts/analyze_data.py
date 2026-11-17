"""
Analyse qualité d'un corpus de pretrain jsonl ({"text": ...}) : doublons, structure, signaux de qualité.
Tourne sur CPU (peut s'exécuter pendant un entraînement GPU). Usage :
  python scripts/analyze_data.py dataset/pretrain_t2t_mini.jsonl [--sample 100000] [--procs 4]
"""

import argparse
import hashlib
import json
import random
from collections import Counter
from multiprocessing import Pool

import numpy as np
from datasketch import MinHash, MinHashLSH

parser = argparse.ArgumentParser()
parser.add_argument("path")
parser.add_argument(
    "--sample",
    type=int,
    default=100_000,
    help="Nb de documents pour la détection de quasi-doublons",
)
parser.add_argument("--procs", type=int, default=4)
parser.add_argument(
    "--chunk",
    type=int,
    default=50_000,
    help="Taille des segments pour cartographier les sources",
)
args = parser.parse_args()

NUM_PERM, NGRAM, THRESHOLD = 64, 5, 0.8


def ascii_ratio(t):
    return sum(c.isascii() for c in t) / max(len(t), 1)


def repetition_ratio(t):
    """Part du texte couverte par le n-gramme de 10 caractères le plus fréquent (boucles, copier-coller)."""
    if len(t) < 50:
        return 0.0
    grams = Counter(t[i : i + 10] for i in range(0, len(t) - 10, 5))
    g, n = grams.most_common(1)[0]
    return n * 10 / len(t) if n > 1 else 0.0


def minhash(text):
    m = MinHash(num_perm=NUM_PERM)
    t = text.replace(" ", "")
    for i in range(
        max(len(t) - NGRAM + 1, 1)
    ):  # n-grammes de caractères : marche pour chinois et anglais
        m.update(t[i : i + NGRAM].encode("utf-8"))
    return m.hashvalues


texts = [json.loads(line)["text"] for line in open(args.path, encoding="utf-8")]
n = len(texts)
print(f"Corpus : {n:,} documents\n")

# ---- 1. doublons exacts
hashes = Counter(hashlib.md5(t.encode("utf-8")).hexdigest() for t in texts)
exact_dups = sum(c - 1 for c in hashes.values() if c > 1)
print(f"1. Doublons exacts : {exact_dups:,} ({exact_dups / n:.2%})")

# ---- 2. quasi-doublons (MinHash LSH, Jaccard >= THRESHOLD sur n-grammes de caractères)
random.seed(0)
idx = random.sample(range(n), min(args.sample, n))
with Pool(args.procs) as pool:
    sigs = pool.map(minhash, [texts[i] for i in idx], chunksize=500)
lsh = MinHashLSH(threshold=THRESHOLD, num_perm=NUM_PERM)
near_dups = 0
examples = []
for k, (i, hv) in enumerate(zip(idx, sigs)):
    m = MinHash(num_perm=NUM_PERM, hashvalues=hv)
    found = lsh.query(m)
    if found:
        near_dups += 1
        if len(examples) < 3:
            examples.append((texts[i][:80], texts[found[0]][:80]))
    else:
        lsh.insert(i, m)
print(
    f"2. Quasi-doublons (Jaccard ≥ {THRESHOLD}) sur un échantillon de {len(idx):,} : {near_dups:,} ({near_dups / len(idx):.2%})"
)
print(
    "   (sur l'échantillon seulement : sur le corpus entier, le taux réel est plus élevé, car chaque document a plus de 'jumeaux' potentiels)"
)
for a, b in examples:
    print(f"   ex. « {a} »\n       « {b} »")

# ---- 3. structure du fichier par segment
print(f"\n3. Structure par segments de {args.chunk:,} documents (ordre du fichier) :")
print(
    f"   {'segment':>17}  {'% ASCII':>7}  {'longueur médiane':>16}  début du 1er document"
)
for s in range(0, n, args.chunk):
    seg = texts[s : s + args.chunk]
    sample = seg[:: max(len(seg) // 2000, 1)]
    asc = np.mean([ascii_ratio(t) for t in sample])
    med = int(np.median([len(t) for t in sample]))
    print(
        f"   {s:>8,}-{min(s + args.chunk, n):>8,}  {asc:>6.0%}  {med:>16,}  {seg[0][:40]!r}"
    )

# ---- 4. signaux de qualité simples
lens = np.array([len(t) for t in texts])
sample = [texts[i] for i in idx[:20000]]
rep = np.array([repetition_ratio(t) for t in sample])
print("\n4. Signaux de qualité :")
print(
    f"   longueur (caractères) : médiane {int(np.median(lens)):,}, p1 {int(np.percentile(lens, 1))}, p99 {int(np.percentile(lens, 99)):,}"
)
print(
    f"   documents très courts (< 50 caractères) : {(lens < 50).sum():,} ({(lens < 50).mean():.2%})"
)
print(
    f"   documents très répétitifs (> 30 % du texte = un même motif répété) : {(rep > 0.3).mean():.2%} (échantillon de {len(sample):,})"
)
worst = sorted(zip(rep, sample), reverse=True)[:2]
for r, t in worst:
    print(f"   ex. répétition {r:.0%} : {t[:100]!r}")
