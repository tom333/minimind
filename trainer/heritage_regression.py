"""
Héritage par régression : remplir chaque bloc de l'enfant pour qu'il reproduise, à lui seul, ce que font plusieurs
blocs du parent.

Le parent (28 blocs) est découpé en tranches consécutives, une par bloc de l'enfant (8). Sur de vrais textes, on
connaît l'état interne du parent à l'entrée et à la sortie de chaque tranche : la cible de chaque bloc de l'enfant
est donc connue, et remplir ses matrices devient une régression.
  1. Moindres carrés (quelques passes de lecture, aucune descente de gradient). Le bloc de l'enfant part du premier
     bloc de sa tranche ; on recalcule ses deux matrices de sortie (o_proj pour l'attention, down_proj pour le FFN)
     pour expliquer au mieux la mise à jour de toute la tranche. Les blocs sont traités dans l'ordre, chacun
     recevant le flux réellement produit par les blocs déjà remplis.
  2. Ajustement local (option --tokens_locaux). Chaque bloc est entraîné SEUL à transformer l'état d'entrée de sa
     tranche en son état de sortie : environ mille nombres à reproduire par position, au lieu d'un token à
     deviner. Aucun gradient ne traverse plus d'un bloc ; tous les blocs avancent en même temps.
Mesure : loss de prédiction de l'enfant assemblé, sur des blocs de validation, sans entraînement de bout en bout.
Repères (parent Qwen3-0.6B, 8 blocs) : 9,40 pour l'enfant simplement taillé ; 3,48 après 5 min de réparation
classique, 3,28 après 11 min (étude heritage).

Usage : python heritage_regression.py Qwen/Qwen3-0.6B-Base --tokenizer ../dataset/fr_en/tokenizer_q3_24k --nom q3-06-r8
"""

import argparse
import copy
import os
import sys
import time

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

import numpy as np
import torch

from trainer.heritage import DATA_DIR, SEQ, load_parent, loss_on, save_child


def random_blocks(path, n_seq, seed):
    """n_seq séquences de 340 tokens tirées au hasard dans les tokens déjà préparés par l'entraînement."""
    tokens = np.load(path, mmap_mode="r")
    idx = np.sort(np.random.default_rng(seed).choice(len(tokens) // SEQ, n_seq, replace=False))
    return torch.from_numpy(np.stack([tokens[i * SEQ : (i + 1) * SEQ] for i in idx]).astype(np.int64))


@torch.no_grad()
def parent_states(model, batch, wanted):
    """Flux résiduel du parent après k blocs, pour chaque k demandé (0 = sortie de l'embedding)."""
    core, cuda = model.model, batch.is_cuda
    pos = (core.freqs_cos[: batch.size(1)], core.freqs_sin[: batch.size(1)])
    x, out = core.embed_tokens(batch), {}
    if 0 in wanted:
        out[0] = x
    with torch.autocast("cuda" if cuda else "cpu", dtype=torch.bfloat16, enabled=cuda):
        for i, layer in enumerate(core.layers[: max(wanted)]):
            x, _ = layer(x, pos)
            if i + 1 in wanted:
                out[i + 1] = x
    return {k: v.float() for k, v in out.items()}, pos


@torch.no_grad()
def fill_by_least_squares(parent, child_layers, bounds, blocks, ridge=1e-3, rounds=2, batch_size=32):
    """Recalcule les deux matrices de sortie de chaque bloc de l'enfant, o_proj (attention) et down_proj (FFN) :
    W = (FᵀF + λ)⁻¹ FᵀR, où F sont les entrées de ces matrices (ce que le bloc a calculé) et R ce qui manque pour
    atteindre l'état de sortie de la tranche. Les deux sont résolues ensemble ; comme les entrées du FFN dépendent
    d'o_proj, on recommence `rounds` fois, puis un dernier tour règle down_proj seule, o_proj étant figée."""
    device = next(parent.parameters()).device
    for k, layer in enumerate(child_layers):
        o_proj, down_proj, box = layer.self_attn.o_proj, layer.mlp.down_proj, {}
        hooks = [
            module.register_forward_pre_hook(lambda _, inputs, name=name: box.__setitem__(name, inputs[0]))
            for name, module in (("attention", o_proj), ("ffn", down_proj))
        ]
        na, nf, width = o_proj.in_features, down_proj.in_features, o_proj.out_features
        for turn in range(rounds + 1):
            joint = turn < rounds
            n = na + nf if joint else nf
            gram = torch.zeros(n, n, dtype=torch.float64, device=device)
            cross = torch.zeros(n, width, dtype=torch.float64, device=device)
            for batch in blocks.split(batch_size):
                states, pos = parent_states(parent, batch.to(device), {0, bounds[k + 1]})
                x = states[0]
                for done in child_layers[:k]:
                    x, _ = done(x, pos)
                mid = x + layer.self_attn(layer.input_layernorm(x), pos)[0]
                layer.mlp(layer.post_attention_layernorm(mid))
                attention, ffn = box["attention"].reshape(-1, na).float(), box["ffn"].reshape(-1, nf).float()
                feats = torch.cat([attention, ffn], dim=1) if joint else ffn
                missing = (states[bounds[k + 1]] - (x if joint else mid)).reshape(-1, width)
                gram += (feats.T @ feats).double()
                cross += (feats.T @ missing).double()
            gram.diagonal().mul_(1 + ridge)  # régularisation relative à l'échelle de chaque entrée
            weights = torch.linalg.solve(gram, cross).T.float()
            if joint:
                o_proj.weight.copy_(weights[:, :na])
                down_proj.weight.copy_(weights[:, na:])
            else:
                down_proj.weight.copy_(weights)
        for hook in hooks:
            hook.remove()


def local_training(parent, child_layers, bounds, blocks, muon_lr, adam_lr, batch_size=16, log=None):
    """Chaque bloc apprend seul : entrée = état du parent au début de sa tranche, cible = état à la fin."""
    device = next(parent.parameters()).device
    params = [p for layer in child_layers for p in layer.parameters()]
    for p in params:
        p.requires_grad_(True)
    opts = [
        torch.optim.Muon([p for p in params if p.dim() == 2], lr=muon_lr, weight_decay=0.0),
        torch.optim.AdamW([p for p in params if p.dim() < 2], lr=adam_lr, weight_decay=0.0),
    ]
    peaks, steps = [muon_lr, adam_lr], len(blocks) // batch_size
    for step in range(steps):
        scale = min(1.0, (step + 1) / max(1, steps // 50)) * 0.5 * (1 + np.cos(np.pi * step / steps))
        for opt, peak in zip(opts, peaks):
            for group in opt.param_groups:
                group["lr"] = peak * max(scale, 0.02)
        batch = blocks[step * batch_size : (step + 1) * batch_size].to(device)
        states, pos = parent_states(parent, batch, set(bounds))
        loss = 0.0
        with torch.autocast("cuda" if batch.is_cuda else "cpu", dtype=torch.bfloat16, enabled=batch.is_cuda):
            for k, layer in enumerate(child_layers):
                out, _ = layer(states[bounds[k]], pos)
                target = states[bounds[k + 1]]
                loss = loss + (out.float() - target).pow(2).mean() / target.pow(2).mean()
        loss.backward()
        for opt in opts:
            opt.step()
            opt.zero_grad(set_to_none=True)
        if log and (step % 100 == 0 or step == steps - 1):
            log(step, steps, loss.item() / len(child_layers))
    for p in params:
        p.requires_grad_(False)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("parent")
    parser.add_argument("--tokenizer", required=True, help="dossier produit par heritage.py vocabulaire")
    parser.add_argument("--blocs", type=int, default=8)
    parser.add_argument("--donnees", default="fr_en/pretrain_fr65hq_q3.jsonl", help="corpus dont les tokens sont déjà préparés")
    parser.add_argument("--tokens_mc", type=int, default=500_000, help="tokens lus pour les moindres carrés")
    parser.add_argument("--tokens_locaux", type=int, default=0, help="tokens de l'ajustement local (0 = aucun)")
    parser.add_argument("--muon_lr", type=float, default=0.003)
    parser.add_argument("--adam_lr", type=float, default=2e-4)
    parser.add_argument("--sans_mc", action="store_true", help="saute les moindres carrés (ajustement local seul)")
    parser.add_argument("--sortie", default="../out/heritage")
    parser.add_argument("--nom", required=True)
    parser.add_argument("--device", default="cuda")
    args = parser.parse_args()

    parent, tok, shape, _ = load_parent(args.parent, args.tokenizer, args.device)
    data = os.path.join(DATA_DIR, args.donnees)
    val = np.load(f"{data}.val_rand2000.npy")
    control = torch.from_numpy(val[: len(val) // SEQ * SEQ].reshape(-1, SEQ)[::32].astype(np.int64))  # jamais vus
    parent_layers = list(parent.model.layers)
    bounds = [int(b) for b in np.linspace(0, len(parent_layers), args.blocs + 1).round()]
    child_layers = [copy.deepcopy(parent_layers[b]) for b in bounds[:-1]]
    fiche = dict(parent=args.parent, tranches=bounds, tokenizer=os.path.relpath(args.tokenizer, DATA_DIR))

    def child_loss():
        parent.model.layers = torch.nn.ModuleList(child_layers)
        loss = loss_on(parent, control, args.device)
        parent.model.layers = torch.nn.ModuleList(parent_layers)
        return loss

    print(f"parent : {len(parent_layers)} blocs, loss {loss_on(parent, control, args.device):.3f} sur {control.numel() / 1e3:.0f}k tokens de validation")
    print(f"tranches du parent par bloc de l'enfant : {[(a, b - 1) for a, b in zip(bounds, bounds[1:])]}")
    print(f"enfant = premier bloc de chaque tranche, tel quel : loss {child_loss():.3f} (hasard : {np.log(len(tok)):.2f})", flush=True)
    start = time.time()
    if not args.sans_mc:
        fill_by_least_squares(parent, child_layers, bounds, random_blocks(f"{data}.train_rand2000.npy", args.tokens_mc // SEQ, seed=0))
        print(f"après moindres carrés ({args.tokens_mc / 1e6:.1f}M tokens, {(time.time() - start) / 60:.1f} min) : loss {child_loss():.3f}", flush=True)
        if args.tokens_locaux:
            save_child(parent, child_layers, shape, args.sortie, f"{args.nom}-mc", fiche)
    if args.tokens_locaux:
        start = time.time()
        log = lambda step, steps, err: print(f"  pas {step}/{steps} : erreur relative moyenne {err:.4f}, {(time.time() - start) / 60:.1f} min", flush=True)
        local_training(parent, child_layers, bounds, random_blocks(f"{data}.train_rand2000.npy", args.tokens_locaux // SEQ, seed=1), args.muon_lr, args.adam_lr, log=log)
        print(f"après ajustement local ({args.tokens_locaux / 1e6:.1f}M tokens, {(time.time() - start) / 60:.1f} min) : loss {child_loss():.3f}", flush=True)
    save_child(parent, child_layers, shape, args.sortie, args.nom, fiche)
