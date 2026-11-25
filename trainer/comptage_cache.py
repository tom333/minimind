"""
Comptage pendant l'entraînement : pour chaque token du sous-ensemble d'entraînement, la probabilité que lui donne
un prédicteur par comptage (suites de 1 à N tokens dans tout le corpus), calculée UNE fois et rangée sur disque.

L'entraînement s'en sert pour une loss de mélange, −log(μ·p_comptage + (1−μ)·p_modèle) : le modèle n'a plus à
apprendre ce que le comptage sait déjà (cf. trainer/diag_comptage.py, même estimateur).
Point essentiel : chaque token d'entraînement figure lui-même dans les comptes. On retire donc sa propre
occurrence (« leave-one-out ») : sinon toute suite rare semblerait parfaitement prévisible, le modèle apprendrait
à faire confiance au comptage, et cette confiance s'effondrerait sur des textes nouveaux.
Sorties, [séquences, longueur − 1] (positions 1 à 339) : logp.npy (float16, log p_comptage du bon token) et
famille.npy (uint8 : ordre du plus long contexte déjà vu × 16 + log2 de son nombre d'occurrences), puis meta.json.

Usage : python comptage_cache.py --donnees ../dataset/fr_en/pretrain_fr65.jsonl --sortie ../out/comptage/fr65-sub50M
"""

import argparse
import json
import os
import sys
import time
from multiprocessing import Pool

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

import numpy as np

from trainer.diag_comptage import MULT, ONE, next_order

KAPPA = {2: 512, 3: 512, 4: 128, 5: 512, 6: 128, 7: 512, 8: 512, 9: 16, 10: 1}  # réglés par diag_comptage sur fr65
STATE = {}


def count_range(job):
    """Occurrences des m-grammes demandés parmi ceux qui FINISSENT dans [a, b), par morceaux de 10M tokens."""
    a, b, m = job
    train, queries = STATE["train"], STATE["queries"]
    counts = np.zeros(len(queries), dtype=np.int32)
    for lo in range(a, b, 10_000_000):
        hi = min(b, lo + 10_000_000)
        start = max(0, lo - (m - 1))
        tok = train[start:hi].astype(np.uint64) + ONE
        h = tok
        for k in range(2, m + 1):
            h = next_order(tok, h, k)
        part = h[max(0, lo - start - (m - 1)) :]
        idx = np.searchsorted(queries, part)
        idx[idx == len(queries)] = 0
        hit = queries[idx] == part
        counts += np.bincount(idx[hit], minlength=len(queries)).astype(np.int32)
    return counts


def block_counts(blocks, train, orders, workers, log):
    """cnt[m][s, i] : occurrences dans le corpus du m-gramme du bloc s qui finit à la position i (0 s'il déborde du bloc)."""
    tok = blocks.astype(np.uint64) + ONE
    cnt = {1: np.bincount(train, minlength=int(blocks.max()) + 1)[blocks].astype(np.int64)}
    h = tok
    total = len(train)
    for m in range(2, orders + 1):
        h = next_order(tok, h, m)
        queries = np.unique(h)
        STATE.update(train=train, queries=queries)
        step = -(-total // workers)
        with Pool(workers) as pool:  # processus fils créés après coup : ils voient corpus et requêtes sans copie
            counts = sum(pool.map(count_range, [(a, min(total, a + step), m) for a in range(0, total, step)]))
        full = np.zeros(blocks.shape, dtype=np.int64)
        full[:, m - 1 :] = counts[np.searchsorted(queries, h)]
        cnt[m] = full
        log(f"ordre {m} : {len(queries) / 1e6:.1f}M suites distinctes")
    return cnt


def count_logp(blocks, cnt, orders, vocab, total, loo):
    """log p_comptage du token i sachant les précédents (positions 1 à L−1), et la famille du contexte."""
    d = 1 if loo else 0  # occurrence propre retirée des comptes (tokens d'entraînement)
    target = blocks[:, 1:]
    p = (cnt[1][:, 1:] - d + 1.0) / (total - d + vocab)
    longest = np.ones(target.shape, dtype=np.int64)
    seen = np.zeros(target.shape)
    pos = np.arange(1, blocks.shape[1])[None, :]
    for n in range(2, orders + 1):
        ok = pos >= n - 1
        full = np.where(ok, cnt[n][:, 1:] - d, 0).astype(np.float64)
        ctx = np.where(ok, cnt[n - 1][:, :-1] - d, 0).astype(np.float64)
        lam = ctx / (ctx + KAPPA[n])
        p = lam * full / np.maximum(ctx, 1.0) + (1 - lam) * p
        longest, seen = np.where(ctx > 0, n, longest), np.where(ctx > 0, ctx, seen)
    family = longest * 16 + np.minimum(np.log2(np.maximum(seen, 1.0)).astype(np.int64), 15)
    return np.log(p).astype(np.float16), family.astype(np.uint8)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--donnees", required=True, help="corpus .jsonl dont les tokens sont déjà préparés")
    parser.add_argument("--subset_tokens", type=int, default=50_000_000, help="même valeur que --train_subset_tokens")
    parser.add_argument("--seq", type=int, default=340)
    parser.add_argument("--val_docs", type=int, default=2000)
    parser.add_argument("--vocab", type=int, default=16000)
    parser.add_argument("--ordres", type=int, default=10)
    parser.add_argument("--workers", type=int, default=8)
    parser.add_argument("--sortie", required=True)
    args = parser.parse_args()

    from dataset.lm_dataset import PackedPretrainDataset

    start = time.time()
    log = lambda msg: print(f"[{(time.time() - start) / 60:5.1f} min] {msg}", flush=True)
    cache = f"{args.donnees}.train_rand{args.val_docs}.npy"
    ds = PackedPretrainDataset(None, None, max_length=args.seq, cache_path=cache)
    ds.restrict(args.subset_tokens)  # même tirage que l'entraînement
    train = np.asarray(ds.tokens)
    blocks = np.stack([train[i * args.seq : (i + 1) * args.seq] for i in ds.index]).astype(np.int64)
    log(f"{len(blocks)} séquences d'entraînement, corpus de {len(train) / 1e6:.0f}M tokens")
    cnt = block_counts(blocks, train, args.ordres, args.workers, log)
    logp, family = count_logp(blocks, cnt, args.ordres, args.vocab, len(train), loo=True)
    os.makedirs(args.sortie, exist_ok=True)
    np.save(os.path.join(args.sortie, "logp.npy"), logp)
    np.save(os.path.join(args.sortie, "famille.npy"), family)
    meta = dict(data=os.path.basename(args.donnees), subset_tokens=args.subset_tokens, seq=args.seq, sequences=len(blocks),
                orders=args.ordres, kappa=KAPPA, minutes=(time.time() - start) / 60,
                logp_moyen=float(logp.astype(np.float32).mean()), part_quasi_certaine=float((logp > -0.01).mean()))
    json.dump(meta, open(os.path.join(args.sortie, "meta.json"), "w"), indent=1)
    log(f"loss du comptage seul sur ces tokens (occurrence propre retirée) : {-meta['logp_moyen']:.3f} ; "
        f"positions jugées quasi certaines : {meta['part_quasi_certaine']:.1%} → {args.sortie}")
