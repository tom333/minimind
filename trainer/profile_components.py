"""
Où part le calcul d'un pas d'entraînement, composant par composant ?

Méthode par retrait : on mesure un pas complet (avant + arrière + optimiseur), puis on neutralise un
composant à la fois (attention, FFN, couche de sortie + loss) et on mesure ce que son retrait fait gagner
en temps et en mémoire. Les parts ne s'additionnent pas exactement (la compilation fusionne des
opérations voisines), mais elles donnent l'ordre de grandeur de chaque poste.
Usage : cd trainer && python profile_components.py [--vocab_size 16000] [--use_compile 1]
"""

import argparse
import os
import sys
import time

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

import torch
import torch.nn.functional as F

from model.model_minimind import MiniMindConfig, MiniMindForCausalLM

parser = argparse.ArgumentParser()
parser.add_argument("--vocab_size", type=int, default=16000)
parser.add_argument("--batch_size", type=int, default=64)
parser.add_argument("--seq", type=int, default=340)
parser.add_argument("--use_compile", type=int, default=1)
args = parser.parse_args()
STEPS, WARMUP = 20, 8


class Zero(torch.nn.Module):
    """Remplace un sous-bloc : renvoie zéro, donc le résiduel laisse passer l'entrée inchangée."""

    def __init__(self, returns_cache):
        super().__init__()
        self.returns_cache = returns_cache

    def forward(self, x, *a, **k):
        out = torch.zeros_like(x)
        return (out, None) if self.returns_cache else out


def measure(remove=None, optimizer=True):
    torch.manual_seed(0)
    model = MiniMindForCausalLM(MiniMindConfig(vocab_size=args.vocab_size)).cuda()
    for block in model.model.layers:
        if remove == "attention":
            block.self_attn = Zero(returns_cache=True)
        if remove == "ffn":
            block.mlp = Zero(returns_cache=False)
    muon = [
        p
        for n, p in model.named_parameters()
        if p.dim() == 2 and "embed_tokens" not in n
    ]
    ids = {id(p) for p in muon}
    opts = [
        torch.optim.AdamW(
            [p for p in model.parameters() if id(p) not in ids],
            lr=7e-4,
            weight_decay=0.1,
        )
    ]
    if muon:
        opts.append(torch.optim.Muon(muon, lr=0.01, weight_decay=0.1))

    def loss_fn(x):
        hidden = model.model(x)[0]
        if remove == "sortie":  # ni projection vers le vocabulaire, ni cross-entropy
            return hidden.float().pow(2).mean()
        logits = model.lm_head(hidden)
        return F.cross_entropy(
            logits[:, :-1].reshape(-1, logits.size(-1)).float(), x[:, 1:].reshape(-1)
        )

    fn = torch.compile(loss_fn) if args.use_compile else loss_fn
    x = torch.randint(0, args.vocab_size, (args.batch_size, args.seq), device="cuda")
    torch.cuda.reset_peak_memory_stats()
    for i in range(WARMUP + STEPS):
        if i == WARMUP:
            torch.cuda.synchronize()
            t0 = time.perf_counter()
        with torch.autocast("cuda", dtype=torch.bfloat16):
            loss = fn(x)
        loss.backward()
        if optimizer:
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            for o in opts:
                o.step()
        for o in opts:
            o.zero_grad(set_to_none=True)
    torch.cuda.synchronize()
    ms = (time.perf_counter() - t0) / STEPS * 1000
    mem = torch.cuda.max_memory_allocated() / 2**30
    n = sum(p.numel() for p in model.parameters()) / 1e6
    model = opts = None  # libère la mémoire GPU avant la mesure suivante
    torch.cuda.empty_cache()
    return ms, mem, n


if __name__ == "__main__":
    full_ms, full_mem, n = measure()
    tokens = args.batch_size * args.seq
    print(
        f"Modèle {n:.1f}M paramètres, vocabulaire {args.vocab_size}, batch {args.batch_size} x {args.seq}, compile={args.use_compile}"
    )
    print(
        f"pas complet : {full_ms:.0f} ms, {tokens / full_ms:.1f}k tokens/s, mémoire max {full_mem:.1f} Go\n"
    )
    print(
        f"{'poste retiré':32} {'temps gagné':>12} {'part du pas':>12} {'mémoire gagnée':>15}"
    )
    for name, kw in [
        ("FFN (8 blocs)", dict(remove="ffn")),
        ("attention (8 blocs)", dict(remove="attention")),
        ("sortie : projection + loss", dict(remove="sortie")),
        ("optimiseur (Muon + AdamW)", dict(optimizer=False)),
    ]:
        ms, mem, _ = measure(**kw)
        print(
            f"{name:32} {full_ms - ms:9.0f} ms {(full_ms - ms) / full_ms:11.0%} {full_mem - mem:12.1f} Go"
        )
