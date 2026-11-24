"""
Quelle part d'un long entraînement n'est que du comptage ?

Deux modèles de même forme : A entraîné peu (50M tokens, 27 min), B entraîné longtemps (1 Md tokens, 8 h). On donne
à A un prédicteur PAR COMPTAGE construit sur tout le corpus d'entraînement (n-grammes jusqu'à l'ordre N : une seule
passe, sans carte graphique), et on mesure si le mélange « A + comptage » rattrape B.
  - s'il le rattrape, les heures d'entraînement en plus n'apprennent que ce qu'une passe de comptage sait déjà ;
  - sinon, elles apprennent autre chose (de la composition), que le comptage ne remplace pas.
Le prédicteur par comptage : P_n(y | contexte) = λ·(fréquence observée) + (1-λ)·P_{n-1}, avec λ = c/(c+κ_n) où c est
le nombre d'occurrences du contexte. Le mélange avec le modèle : μ·P_comptage + (1-μ)·P_modèle, μ ne dépendant que
du contexte (ordre le plus long déjà vu, nombre d'occurrences). κ et μ sont réglés sur une moitié des blocs de
validation, toutes les mesures sont faites sur l'autre moitié. Comme pour le modèle, le contexte s'arrête au début
du bloc de 340 tokens.

Usage : python diag_comptage.py --a ../out/etude_fr-en-court/fr-en-court-t0_768.pth --b ../out/etude_fr-en-full/fr-en-full-t0_768.pth
"""

import argparse
import os
import sys
from multiprocessing import Pool

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

import numpy as np

SEQ = 340
MULT = np.uint64(0x9E3779B97F4A7C15)
ONE = np.uint64(1)
QUERIES = {}


def next_order(tok, h, m):
    """h : hachage des (m-1)-grammes ; retourne celui des m-grammes (l'indice j désigne le m-gramme finissant en j+m-1)."""
    return tok[..., m - 1 :] + MULT * h[..., :-1]


def init_worker(queries):
    QUERIES.update(queries)


def count_chunk(job):
    """Occurrences, dans train[a:b], des n-grammes demandés (ceux qui FINISSENT dans [a, b) : aucun double compte)."""
    path, a, b, orders, vocab = job
    train = np.load(path, mmap_mode="r")
    start = max(0, a - (orders - 1))
    tok = train[start:b].astype(np.uint64) + ONE
    res, h = {1: np.bincount(train[a:b], minlength=vocab)}, tok
    for m in range(2, orders + 1):
        h = next_order(tok, h, m)
        part, q = h[max(0, a - start - (m - 1)) :], QUERIES[m]
        idx = np.searchsorted(q, part)
        idx[idx == len(q)] = 0
        res[m] = np.bincount(idx[q[idx] == part], minlength=len(q))
    return res


def count_all(path, total, orders, vocab, queries, chunk, workers):
    jobs = [(path, a, min(total, a + chunk), orders, vocab) for a in range(0, total, chunk)]
    counts = None
    with Pool(workers, initializer=init_worker, initargs=(queries,)) as pool:
        for res in pool.imap_unordered(count_chunk, jobs):
            counts = res if counts is None else {m: counts[m] + res[m] for m in res}
    return counts


def selftest():
    """Le comptage par morceaux doit donner les mêmes nombres qu'un comptage direct, y compris aux jonctions."""
    rng = np.random.default_rng(0)
    data = rng.integers(0, 5, size=20000).astype(np.uint16)
    np.save("/tmp/diag_comptage_test.npy", data)
    tok = data.astype(np.uint64) + ONE
    queries, h = {}, tok
    for m in range(2, 5):
        h = next_order(tok, h, m)
        queries[m] = np.unique(h[:200])
    counts = count_all("/tmp/diag_comptage_test.npy", len(data), 4, 5, queries, chunk=997, workers=2)
    assert (counts[1] == np.bincount(data, minlength=5)).all()
    h = tok
    for m in range(2, 5):
        h = next_order(tok, h, m)
        values, direct = np.unique(h, return_counts=True)
        assert (counts[m] == direct[np.searchsorted(values, queries[m])]).all(), f"ordre {m}"


def neural_logp(path, blocks, vocab):
    import torch

    from model.model_minimind import MiniMindConfig, MiniMindForCausalLM

    cache = f"/tmp/diag_comptage_{os.path.basename(path)}_{len(blocks)}.npy"
    if os.path.exists(cache):
        return np.load(cache)
    model = MiniMindForCausalLM(MiniMindConfig(vocab_size=vocab))
    model.load_state_dict({k: v.float() for k, v in torch.load(path, map_location="cpu").items()})
    model.eval()
    out = np.zeros(blocks.shape, dtype=np.float64)
    with torch.no_grad():
        for s in range(0, len(blocks), 16):
            b = torch.from_numpy(blocks[s : s + 16].astype(np.int64))
            lp = model(b).logits[:, :-1].float().log_softmax(-1)
            out[s : s + 16, 1:] = lp.gather(-1, b[:, 1:, None]).squeeze(-1).numpy()
    np.save(cache, out)
    return out


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--a", required=True, help="poids du modèle entraîné peu")
    parser.add_argument("--b", required=True, help="poids du modèle entraîné longtemps")
    parser.add_argument("--donnees", default="../dataset/fr_en/pretrain_fr65.jsonl")
    parser.add_argument("--vocab", type=int, default=16000)
    parser.add_argument("--ordres", type=int, default=10, help="ordre maximal des n-grammes comptés")
    parser.add_argument("--pas", type=int, default=5, help="un bloc de validation sur `pas`")
    parser.add_argument("--workers", type=int, default=10)
    args = parser.parse_args()
    selftest()

    val = np.load(f"{args.donnees}.val_rand2000.npy")
    blocks = val[: len(val) // SEQ * SEQ].reshape(-1, SEQ)[:: args.pas]
    train_path = f"{args.donnees}.train_rand2000.npy"
    total = len(np.load(train_path, mmap_mode="r"))
    N, tok = args.ordres, blocks.astype(np.uint64) + ONE
    H, h = {1: tok}, tok  # H[m][bloc, i] : hachage du m-gramme finissant à la position i (0 si i < m-1)
    for m in range(2, N + 1):
        h = next_order(tok, h, m)
        H[m] = np.zeros(tok.shape, dtype=np.uint64)
        H[m][:, m - 1 :] = h
    queries = {m: np.unique(H[m][:, m - 1 :]) for m in range(2, N + 1)}
    counts = count_all(train_path, total, N, args.vocab, queries, chunk=10_000_000, workers=args.workers)
    print(f"comptage : {total / 1e6:.0f}M tokens, ordres 1 à {N}, {blocks.size / 1e3:.0f}k tokens de validation", flush=True)

    def lookup(m, hashes):
        return counts[1][blocks[:, : hashes.shape[1]]] if m == 1 else counts[m][np.searchsorted(queries[m], hashes)]

    pos = np.arange(SEQ)[None, :]
    ctx, full = {}, {}  # pour l'ordre n : occurrences du contexte (n-1 tokens) et du contexte suivi du bon token
    for n in range(2, N + 1):
        ok = pos >= n - 1
        c = np.zeros(blocks.shape)
        c[:, 1:] = (counts[1][blocks[:, :-1]] if n == 2 else lookup(n - 1, H[n - 1][:, :-1]))
        ctx[n] = np.where(ok, c, 0.0)
        full[n] = np.where(ok, lookup(n, H[n]), 0.0)
    unigram = (counts[1][blocks] + 1.0) / (total + args.vocab)
    dev = (np.arange(len(blocks)) % 2 == 0)[:, None] & (pos >= 1)
    test = (np.arange(len(blocks)) % 2 == 1)[:, None] & (pos >= 1)

    def p_count(kappa):
        p = unigram.copy()
        for n in range(2, N + 1):
            lam = ctx[n] / (ctx[n] + kappa[n])
            p = lam * full[n] / np.maximum(ctx[n], 1.0) + (1 - lam) * p
        return p

    kappa = {n: 4.0 for n in range(2, N + 1)}
    grid = [0.125, 0.25, 0.5, 1, 2, 4, 8, 16, 32, 64, 128, 512]
    for _ in range(2):
        for n in range(2, N + 1):
            kappa[n] = min(grid, key=lambda k: -np.log(p_count({**kappa, n: k})[dev]).mean())
    pc = p_count(kappa)
    print("κ par ordre :", {n: kappa[n] for n in kappa}, flush=True)

    longest = np.ones(blocks.shape, dtype=np.int64)  # ordre du plus long contexte déjà vu dans le corpus
    seen = np.zeros(blocks.shape)
    for n in range(2, N + 1):
        longest, seen = np.where(ctx[n] > 0, n, longest), np.where(ctx[n] > 0, ctx[n], seen)
    bucket = longest * 16 + np.minimum(np.log2(np.maximum(seen, 1.0)).astype(np.int64), 15)

    def mix(pm):
        """μ par famille de contextes, réglé sur dev ; retourne les probabilités mélangées partout."""
        mus = np.linspace(0, 0.98, 50)
        best = lambda mask: mus[np.argmax([np.log(mu * pc[mask] + (1 - mu) * pm[mask]).sum() for mu in mus])]
        default, mu = best(dev), np.zeros(blocks.shape)
        for b in np.unique(bucket):
            mask = (bucket == b) & dev
            mu[bucket == b] = best(mask) if mask.sum() >= 200 else default
        return mu * pc + (1 - mu) * pm

    import torch

    torch.set_num_threads(os.cpu_count())
    pa, pb = (np.exp(neural_logp(p, blocks, args.vocab)) for p in (args.a, args.b))
    rows = {"comptage seul": pc, "A (entraîné peu)": pa, "B (entraîné longtemps)": pb, "A + comptage": mix(pa), "B + comptage": mix(pb)}
    loss = lambda p, mask: -np.log(p[mask]).mean()
    print(f"\n{'prédicteur':26}{'loss (moitié test)':>20}")
    for name, p in rows.items():
        print(f"{name:26}{loss(p, test):20.3f}")
    print(f"\n{'contexte déjà vu (tokens)':26}{'part':>6}" + "".join(f"{n:>24}" for n in rows))
    for lo, hi in [(1, 1), (2, 2), (3, 3), (4, 4), (5, 6), (7, 9), (10, N)]:
        mask = test & (longest >= lo) & (longest <= hi)
        if mask.sum():
            label = f"{lo - 1} ou plus" if hi == N else f"{lo - 1}" if lo == hi else f"{lo - 1} à {hi - 1}"
            print(f"{label:26}{mask.sum() / test.sum():6.1%}" + "".join(f"{loss(p, mask):24.3f}" for p in rows.values()))
