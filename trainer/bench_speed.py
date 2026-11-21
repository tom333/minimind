"""
Banc de débit : que gagne-t-on, en temps par pas, avec des réglages SYSTÈME qui ne changent pas le modèle ?

Chaque variante est mesurée sur le même pas d'entraînement que train_pretrain.py (modèle bilingue de 71M,
lot de 64 × 340 tokens, bf16, Muon + AdamW, clip du gradient), sur des données aléatoires : seul le temps
compte ici. Les variantes marquées « (math) » modifient légèrement le calcul : un gain de temps ne vaut
alors que si la loss ne se dégrade pas, à vérifier ensuite par un run court.
Repère : le pas de référence dure ~594 ms sur une RTX 3060 (36 600 tokens/s).
Usage : cd trainer && python bench_speed.py [--only base,tf32] [--mlflow_experiment minimind-bench-debit]
"""

import argparse
import os
import sys
import time

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

import torch

from model.model_minimind import MiniMindConfig, MiniMindForCausalLM

VOCAB, BATCH, SEQ, STEPS, WARMUP = 16000, 64, 340, 30, 10

# nom → réglages. Clés : tf32, fused, clip, ns_steps, compile_mode, arch (surcharges du modèle), batch, seq
VARIANTS = {
    "base": {},
    "tf32": dict(
        tf32=True
    ),  # matmul fp32 sur les cœurs tenseurs (Newton-Schulz de Muon, loss)
    "adamw-fused": dict(fused=True),  # AdamW en un seul noyau CUDA
    "sans-clip": dict(clip=False),  # coût du clip du gradient (diagnostic)
    "muon-ns4 (math)": dict(ns_steps=4),  # 4 itérations de Newton-Schulz au lieu de 5
    "muon-ns3 (math)": dict(ns_steps=3),
    "compile-max-autotune": dict(compile_mode="max-autotune-no-cudagraphs"),
    "tf32+fused": dict(tf32=True, fused=True),
    "relu2-meme-largeur (math)": dict(
        arch=dict(ffn_glu=False, hidden_act="relu2")
    ),  # repère structurel
    "mod-50 (math)": dict(arch=dict(mod_capacity=0.5)),  # Mixture-of-Depths, un bloc sur deux
    "mod-25 (math)": dict(arch=dict(mod_capacity=0.25)),
    "mod-12 (math)": dict(arch=dict(mod_capacity=0.125)),
    # prochaine marche : ~150M paramètres (largeur 1024, 10 blocs). Temps par pas et mémoire réels.
    "150M": dict(arch=dict(hidden_size=1024, num_hidden_layers=10)),
    "150M lot-32": dict(arch=dict(hidden_size=1024, num_hidden_layers=10), batch=32),
    "150M mod-50": dict(arch=dict(hidden_size=1024, num_hidden_layers=10, mod_capacity=0.5)),
    "150M mod-25": dict(arch=dict(hidden_size=1024, num_hidden_layers=10, mod_capacity=0.25)),
    "150M mod-25 tf32+fused": dict(arch=dict(hidden_size=1024, num_hidden_layers=10, mod_capacity=0.25), tf32=True, fused=True),
    "lot-128 (math)": dict(
        batch=128
    ),  # débit par token d'un lot double (la loss par token en pâtit, étude 04)
}


def measure(
    tf32=False,
    fused=False,
    clip=True,
    ns_steps=5,
    compile_mode=None,
    arch=None,
    batch=BATCH,
    seq=SEQ,
):
    torch._dynamo.reset()
    torch.backends.cuda.matmul.allow_tf32 = tf32
    torch.backends.cudnn.allow_tf32 = tf32
    torch.set_float32_matmul_precision("high" if tf32 else "highest")
    torch.manual_seed(0)
    model = MiniMindForCausalLM(MiniMindConfig(vocab_size=VOCAB, **(arch or {}))).cuda()
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
            fused=fused,
        ),
        torch.optim.Muon(muon, lr=0.01, weight_decay=0.1, ns_steps=ns_steps),
    ]
    fn = (
        torch.compile(model, mode=compile_mode)
        if compile_mode
        else torch.compile(model)
    )
    x = torch.randint(0, VOCAB, (batch, seq), device="cuda")
    torch.cuda.reset_peak_memory_stats()
    for i in range(WARMUP + STEPS):
        if i == WARMUP:
            torch.cuda.synchronize()
            t0 = time.perf_counter()
        with torch.autocast("cuda", dtype=torch.bfloat16):
            out = fn(x, labels=x)
            loss = out.loss + out.aux_loss
        loss.backward()
        if clip:
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        for o in opts:
            o.step()
            o.zero_grad(set_to_none=True)
    torch.cuda.synchronize()
    ms = (time.perf_counter() - t0) / STEPS * 1000
    mem = torch.cuda.max_memory_allocated() / 2**30
    del model, opts, fn
    torch.cuda.empty_cache()
    return ms, mem, batch * seq / ms


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--only",
        default="",
        help="noms de variantes séparés par des virgules (préfixes acceptés)",
    )
    parser.add_argument("--mlflow_experiment", default="")
    args = parser.parse_args()
    wanted = [w for w in args.only.split(",") if w]
    results = {}
    print(
        f"{'variante':28} {'ms/pas':>8} {'k tokens/s':>11} {'vs base':>8} {'mémoire':>9}",
        flush=True,
    )
    for name, kw in VARIANTS.items():
        if wanted and not any(name.startswith(w) for w in wanted) and name != "base":
            continue
        try:
            ms, mem, ktps = measure(**kw)
        except Exception as e:  # une variante qui plante (mode de compilation, mémoire) n'arrête pas le banc
            print(f"{name:28} échec : {type(e).__name__}: {str(e)[:120]}", flush=True)
            torch.cuda.empty_cache()
            continue
        results[name] = (ms, mem, ktps)
        base = results["base"][2]
        print(
            f"{name:28} {ms:8.0f} {ktps:11.1f} {ktps / base - 1:+8.1%} {mem:7.1f} Go",
            flush=True,
        )
    if args.mlflow_experiment:
        import mlflow

        mlflow.set_tracking_uri(
            os.environ.get("MINIMIND_MLFLOW_URI", "https://mlflow.tgu.ovh")
        )
        mlflow.set_experiment(args.mlflow_experiment)
        for name, (ms, mem, ktps) in results.items():
            with mlflow.start_run(run_name=name):
                mlflow.log_params(
                    {"variante": name, **{k: str(v) for k, v in VARIANTS[name].items()}}
                )
                mlflow.log_metrics(
                    {"ms_par_pas": ms, "k_tokens_par_s": ktps, "gpu_mem_go": mem}
                )
