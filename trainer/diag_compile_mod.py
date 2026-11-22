"""
Diagnostic : combien de temps prend la COMPILATION (torch.compile) du premier pas d'entraînement selon le
nombre de tokens retenus par Mixture-of-Depths ? Constat à expliquer : le premier run à 25 % de capacité
(85 tokens sur 340) d'une étude perd ~15 minutes au démarrage, pas ceux à 12,5 % (42) ou 50 % (170).
Chaque mesure tourne dans un processus neuf avec un cache de compilation vide.
Usage : python diag_compile_mod.py [--tokens 0,42,84,85,86,170] [--dtype bfloat16]
"""

import argparse
import os
import subprocess
import sys
import tempfile
import time

ONE = r"""
import sys, time, torch
sys.path.insert(0, "..")
from model.model_minimind import MiniMindConfig, MiniMindForCausalLM
k, dtype = int(sys.argv[1]), getattr(torch, sys.argv[2])
cap = (k + 0.5) / 340 if k else 0.0
model = MiniMindForCausalLM(MiniMindConfig(vocab_size=16000, mod_capacity=cap)).cuda()
fn = torch.compile(model)
x = torch.randint(0, 16000, (64, 340), device="cuda")
times = []
for i in range(4):
    torch.cuda.synchronize(); t = time.perf_counter()
    with torch.autocast("cuda", dtype=dtype):
        out = fn(x, labels=x)
    (out.loss + out.aux_loss).backward()
    model.zero_grad(set_to_none=True)
    torch.cuda.synchronize(); times.append(time.perf_counter() - t)
kept = [int(340 * l.mod_capacity) for l in model.model.layers if l.router is not None][:1]
print(f"RESULT tokens={kept[0] if kept else 340} premier_pas={times[0]:.0f}s deuxieme={times[1]:.1f}s regime={times[-1]*1000:.0f}ms", flush=True)
"""

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--tokens", default="0,42,84,85,86,170")
    parser.add_argument("--dtype", default="bfloat16")
    args = parser.parse_args()
    for k in args.tokens.split(","):
        env = dict(os.environ, TORCHINDUCTOR_CACHE_DIR=tempfile.mkdtemp(prefix="inductor_"))
        start = time.time()
        r = subprocess.run([sys.executable, "-c", ONE, k, args.dtype], env=env, capture_output=True, text=True)
        line = [l for l in r.stdout.splitlines() if l.startswith("RESULT")]
        print(line[0] if line else f"tokens={k} ÉCHEC : {r.stderr.strip().splitlines()[-1][:160]}", f"| total {time.time() - start:.0f}s", flush=True)
