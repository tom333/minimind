"""
Benchmark : kernels Liger (Triton fusionnés) vs PyTorch, sur notre modèle, sans modifier le code du modèle.
Remplacements testés (monkeypatch sur une instance) :
  - RMSNorm            → LigerRMSNorm
  - SiLU(gate) * up    → LigerSiLUMulFunction (un seul kernel au lieu de deux)
  - lm_head + CE       → LigerFusedLinearCrossEntropyLoss (ne matérialise pas les logits [B, T, vocab])
Mesure : débit (tokens/s) forward+backward+step, mémoire GPU max, et égalité de la loss.
Usage : cd trainer && python bench_liger.py
"""

import os
import sys
import time

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

import torch
import torch.nn.functional as F
from liger_kernel.ops.swiglu import LigerSiLUMulFunction
from liger_kernel.transformers import LigerFusedLinearCrossEntropyLoss, LigerRMSNorm

from model.feedforward import FeedForward
from model.model_minimind import MiniMindConfig, MiniMindForCausalLM
from model.norm import RMSNorm

BATCH, SEQ, STEPS, WARMUP = 32, 340, 30, 10


def patch_liger(model):
    """Remplace les RMSNorm et le SwiGLU de l'instance par les kernels Liger (poids conservés)."""
    for module in list(model.modules()):
        for name, child in list(module.named_children()):
            if isinstance(child, RMSNorm):
                new = LigerRMSNorm(child.weight.numel(), eps=child.eps).to(
                    child.weight.device
                )
                new.weight.data.copy_(child.weight.data)
                setattr(module, name, new)
            elif isinstance(child, FeedForward):
                child.forward = lambda x, m=child: m.down_proj(
                    LigerSiLUMulFunction.apply(m.gate_proj(x), m.up_proj(x))
                )
    return model


def make_loss_fn(model, fused_ce):
    flce = LigerFusedLinearCrossEntropyLoss()

    def loss_fn(x):
        hidden = model.model(x)[0]
        h, y = hidden[:, :-1].reshape(-1, hidden.size(-1)), x[:, 1:].reshape(-1)
        if fused_ce:
            return flce(model.lm_head.weight, h, y)
        return F.cross_entropy(model.lm_head(h).float(), y)

    return loss_fn


def run(liger, compile_, x):
    torch.manual_seed(0)
    model = MiniMindForCausalLM(MiniMindConfig()).cuda()
    if liger:
        patch_liger(model)
    loss_fn = make_loss_fn(model, fused_ce=liger)
    if compile_:
        loss_fn = torch.compile(loss_fn)
    opt = torch.optim.AdamW(model.parameters(), lr=1e-4)
    torch.cuda.reset_peak_memory_stats()
    first_loss = None
    for i in range(WARMUP + STEPS):
        if i == WARMUP:
            torch.cuda.synchronize()
            t0 = time.time()
        with torch.autocast("cuda", dtype=torch.bfloat16):
            loss = loss_fn(x)
        if first_loss is None:
            first_loss = loss.item()
        loss.backward()
        opt.step()
        opt.zero_grad(set_to_none=True)
    torch.cuda.synchronize()
    tok_s = STEPS * BATCH * SEQ / (time.time() - t0)
    return tok_s, torch.cuda.max_memory_allocated() / 2**30, first_loss


if __name__ == "__main__":
    x = torch.randint(0, 6400, (BATCH, SEQ), device="cuda")
    results = {}
    for compile_ in (False, True):
        for liger in (False, True):
            tok_s, mem, loss0 = run(liger, compile_, x)
            results[(liger, compile_)] = tok_s
            print(
                f"compile={compile_!s:5} liger={liger!s:5} : {tok_s / 1e3:5.1f}k tokens/s, mémoire max {mem:4.1f} Go, loss initiale {loss0:.4f}"
            )
    for compile_ in (False, True):
        print(
            f"gain Liger (compile={compile_}) : {results[(True, compile_)] / results[(False, compile_)] - 1:+.0%}"
        )
