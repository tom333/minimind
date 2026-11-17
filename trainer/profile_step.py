"""
Profilage d'un step d'entraînement dans la configuration actuelle (compile, bf16, Muon+AdamW, packing).
1. Temps par phase (données / forward / backward / optimiseur), mesuré avec synchronisation GPU.
2. Répartition du temps GPU par famille de kernels (torch.profiler).
Usage : cd trainer && python profile_step.py [--use_compile 0|1] [--optimizer muon|adamw]
"""

import argparse
import os
import sys
import time
from collections import defaultdict

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

import numpy as np
import torch
from torch.profiler import ProfilerActivity, profile
from torch.utils.data import DataLoader

from dataset.lm_dataset import PackedPretrainDataset
from model.model_minimind import MiniMindConfig, MiniMindForCausalLM

parser = argparse.ArgumentParser()
parser.add_argument("--use_compile", type=int, default=1)
parser.add_argument("--optimizer", default="muon", choices=["muon", "adamw"])
parser.add_argument("--batch_size", type=int, default=32)
parser.add_argument("--accumulation_steps", type=int, default=2)
args = parser.parse_args()
SEQ = 340

# ---- données réelles (cache packé déjà construit par train_pretrain.py)
ds = PackedPretrainDataset(
    None, None, SEQ, cache_path="../dataset/pretrain_t2t_mini.jsonl.train_rand2000.npy"
)
loader = iter(
    DataLoader(
        ds, batch_size=args.batch_size, shuffle=True, num_workers=8, pin_memory=True
    )
)

# ---- modèle + optimiseurs comme dans train_pretrain.py
model = MiniMindForCausalLM(MiniMindConfig()).cuda()
muon_params = [
    p
    for n, p in model.named_parameters()
    if args.optimizer == "muon" and p.dim() == 2 and "embed_tokens" not in n
]
muon_ids = {id(p) for p in muon_params}
others = [p for p in model.parameters() if id(p) not in muon_ids]
optimizers = [torch.optim.AdamW(others, lr=7e-4, weight_decay=0.1)]
if muon_params:
    optimizers.append(torch.optim.Muon(muon_params, lr=0.01, weight_decay=0.1))
fwd = torch.compile(model) if args.use_compile else model

phase_time = defaultdict(float)


def timed(name, fn):
    torch.cuda.synchronize()
    t0 = time.perf_counter()
    out = fn()
    torch.cuda.synchronize()
    phase_time[name] += time.perf_counter() - t0
    return out


def train_step(measure):
    run = timed if measure else (lambda name, fn: fn())
    for _ in range(args.accumulation_steps):
        x, y = run(
            "données (attente + copie GPU)",
            lambda: [t.cuda(non_blocking=True) for t in next(loader)],
        )
        with torch.autocast("cuda", dtype=torch.bfloat16):
            loss = run(
                "forward", lambda: fwd(x, labels=y).loss / args.accumulation_steps
            )
        run("backward", lambda: loss.backward())

    def opt_step():
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        for o in optimizers:
            o.step()
            o.zero_grad(set_to_none=True)

    run("optimiseur (clip + step)", opt_step)


# ---- 1. temps par phase
for _ in range(10):  # warmup : compilation, autotuning, workers
    train_step(measure=False)
N = 20
for _ in range(N):
    train_step(measure=True)
total = sum(phase_time.values())
tokens = N * args.accumulation_steps * args.batch_size * SEQ
print(
    f"\n=== Temps par phase ({N} steps, compile={args.use_compile}, optimiseur={args.optimizer}) ==="
)
for k, v in sorted(phase_time.items(), key=lambda kv: -kv[1]):
    print(f"  {k:32} {v / N * 1000:7.1f} ms/step  {v / total:6.1%}")
print(f"  débit (avec synchronisations)    {tokens / total / 1e3:7.1f}k tokens/s")


# ---- 2. répartition du temps GPU par famille de kernels
def family(name):
    n = name.lower()
    if any(k in n for k in ("flash", "fmha", "attention", "efficient_attention")):
        return "attention (Flash Attention)"
    if any(
        k in n
        for k in (
            "gemm",
            "matmul",
            "cutlass",
            "xmma",
            "sm80_",
            "sm86_",
            "ampere_",
            "cublas",
            "addmm",
            "bmm",
        )
    ):
        return "multiplications de matrices"
    if "triton" in n:
        return "opérations fusionnées (Triton : normes, SiLU, RoPE...)"
    if any(k in n for k in ("adam", "foreach", "multi_tensor")):
        return "mise à jour AdamW (élément par élément)"
    if any(k in n for k in ("memcpy", "memset")):
        return "copies / mises à zéro mémoire"
    if any(k in n for k in ("nll", "cross_entropy", "softmax", "log_softmax")):
        return "loss (cross-entropy)"
    if any(k in n for k in ("reduce", "norm")):
        return "réductions (norme du gradient...)"
    return "autres kernels élémentaires"


with profile(activities=[ProfilerActivity.CUDA]) as prof:
    for _ in range(5):
        train_step(measure=False)
    torch.cuda.synchronize()
fam = defaultdict(float)
top = []
for e in prof.key_averages():
    t = getattr(e, "self_device_time_total", 0) or getattr(e, "self_cuda_time_total", 0)
    if t > 0:
        fam[family(e.key)] += t
        top.append((t, e.key))
gpu_total = sum(fam.values())
print("\n=== Temps GPU par famille de kernels ===")
for k, v in sorted(fam.items(), key=lambda kv: -kv[1]):
    print(f"  {k:55} {v / gpu_total:6.1%}")
print("\n=== 8 kernels les plus coûteux ===")
for t, k in sorted(top, reverse=True)[:8]:
    print(f"  {t / gpu_total:6.1%}  {k[:90]}")
print(f"\nMémoire GPU max : {torch.cuda.max_memory_allocated() / 2**30:.1f} Go")
