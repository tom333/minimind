"""
Héritage avec réduction de LARGEUR : tailler un enfant étroit dans un parent plus large (ex. Qwen3-1.7B, largeur
2 048 → 1 024), pas seulement moins profond. C'est la coupe qu'il faudra pour hériter d'un parent beaucoup plus gros.

Méthode (SliceGPT, Ashkboos et al. 2024 ; Minitron, Muralidharan et al. 2024) :
  1. les poids des normalisations sont fondus dans les matrices qui les suivent : la normalisation devient alors
     insensible à une rotation du flux résiduel ;
  2. on fait tourner tout le flux résiduel vers ses directions principales (analyse en composantes principales des
     états du parent sur de vrais textes) et on ne garde que les premières : l'embedding, les entrées et les sorties
     des blocs sont projetés ; l'échelle des normalisations compense l'énergie perdue ;
  3. dans chaque FFN, on garde les neurones les plus utiles (activation moyenne × norme de leur sortie) ;
  4. les blocs gardés sont donnés (--blocs_gardes) ; l'attention garde toutes ses têtes.
Avec les réglages par défaut, l'enfant a exactement la forme de l'enfant taillé dans Qwen3-0.6B (q3-06-b8) :
la comparaison ne dépend que du parent.

Usage : python heritage_largeur.py Qwen/Qwen3-1.7B-Base --tokenizer ../dataset/fr_en/tokenizer_q3_24k --nom q3-17-l8
"""

import argparse
import json
import os
import sys

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

import numpy as np
import torch

from trainer.heritage import DATA_DIR, calibration, load_parent, loss_on


@torch.no_grad()
def collect(model, blocks, keep, device):
    """Covariance non centrée du flux résiduel à l'entrée de chaque bloc gardé et en sortie du dernier bloc, et
    importance des neurones des FFN des blocs gardés."""
    layers, width = model.model.layers, model.config.hidden_size
    points = keep + [len(layers)]
    cov = {p: torch.zeros(width, width, dtype=torch.float64) for p in points}
    importance = {b: 0.0 for b in keep}
    hooks = []
    for p in points:
        if p < len(layers):
            hooks.append(layers[p].register_forward_pre_hook(lambda m, a, p=p: add_state(cov[p], a[0])))
        else:
            hooks.append(model.model.norm.register_forward_pre_hook(lambda m, a, p=p: add_state(cov[p], a[0])))
    for b in keep:
        down = layers[b].mlp.down_proj
        hooks.append(down.register_forward_pre_hook(lambda m, a, b=b: importance.__setitem__(b, importance[b] + a[0].float().abs().reshape(-1, a[0].size(-1)).sum(0).cpu())))
    for batch in blocks.split(8):
        model(batch.to(device))
    for h in hooks:
        h.remove()
    return cov, {b: importance[b] * layers[b].mlp.down_proj.weight.float().norm(dim=0).cpu() for b in keep}


def add_state(acc, x):
    """Ajoute xᵀx à l'accumulateur. Ne renvoie rien : un crochet qui renvoie une valeur remplacerait l'entrée du bloc."""
    x = x.float().reshape(-1, x.size(-1))
    acc.add_((x.T @ x).double().cpu())


def carve_width(model, keep, new, ffn, calib, device):
    """État de l'enfant (blocs `keep`, largeur `new`, `ffn` neurones par FFN) et part d'énergie gardée par point."""
    old = model.config.hidden_size
    cov, importance = collect(model, calib, keep, device)

    total = sum(c / c.trace() for c in cov.values())  # chaque point pèse autant, quelle que soit la taille de ses états
    values, vectors = torch.linalg.eigh(total)
    Q = vectors[:, values.argsort(descending=True)[:new]].float()  # [ancienne largeur, nouvelle largeur]
    kept_energy = {p: float((Q.double().T @ c @ Q.double()).trace() / c.trace()) for p, c in cov.items()}
    print("énergie du flux résiduel gardée par la projection :", {p: round(e, 3) for p, e in kept_energy.items()})

    st = {k: v.float().cpu() for k, v in model.state_dict().items()}
    child = {"model.embed_tokens.weight": st["model.embed_tokens.weight"] @ Q}
    child["lm_head.weight"] = child["model.embed_tokens.weight"]
    for i, b in enumerate(keep):
        pre, out = f"model.layers.{b}.", f"model.layers.{i}."
        scale = (old / new * kept_energy[b]) ** 0.5  # la norme ne voit plus que l'énergie gardée, sur moins de dimensions
        g1, g2 = st[pre + "input_layernorm.weight"], st[pre + "post_attention_layernorm.weight"]
        for name in ("q_proj", "k_proj", "v_proj"):
            child[out + f"self_attn.{name}.weight"] = (st[pre + f"self_attn.{name}.weight"] * g1) @ Q
        child[out + "self_attn.o_proj.weight"] = Q.T @ st[pre + "self_attn.o_proj.weight"]
        for name in ("q_norm", "k_norm"):
            child[out + f"self_attn.{name}.weight"] = st[pre + f"self_attn.{name}.weight"]
        sel = importance[b].argsort(descending=True)[: ffn].sort().values
        for name in ("gate_proj", "up_proj"):
            child[out + f"mlp.{name}.weight"] = (st[pre + f"mlp.{name}.weight"][sel] * g2) @ Q
        child[out + "mlp.down_proj.weight"] = Q.T @ st[pre + "mlp.down_proj.weight"][:, sel]
        child[out + "input_layernorm.weight"] = torch.full((new,), scale)
        child[out + "post_attention_layernorm.weight"] = torch.full((new,), scale)
    final = len(model.model.layers)
    child["model.norm.weight"] = (old / new * kept_energy[final]) ** 0.5 * torch.diagonal(Q.T @ torch.diag(st["model.norm.weight"]) @ Q)

    return child, kept_energy


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("parent")
    parser.add_argument("--tokenizer", required=True, help="vocabulaire réduit d'un parent de la même famille")
    parser.add_argument("--blocs_gardes", default="0,2,3,18,20,21,23,25", help="blocs gardés (défaut : ceux de q3-06-b8)")
    parser.add_argument("--largeur", type=int, default=1024)
    parser.add_argument("--ffn", type=int, default=3072, help="neurones gardés par FFN")
    parser.add_argument("--fiche_forme", default="../out/heritage/q3-06-b8.json", help="forme de l'enfant (doit concorder)")
    parser.add_argument("--corpus", default="fr_en/pretrain_fr65hq.jsonl")
    parser.add_argument("--sequences", type=int, default=64)
    parser.add_argument("--controle", type=int, default=96, help="séquences de contrôle pour mesurer la loss")
    parser.add_argument("--sortie", default="../out/heritage")
    parser.add_argument("--nom", required=True)
    parser.add_argument("--device", default="cpu")
    args = parser.parse_args()

    from trainer.heritage import save_child

    model, tok, shape, _ = load_parent(args.parent, args.tokenizer, args.device, check_parent=False)
    keep = [int(b) for b in args.blocs_gardes.split(",")]
    corpus = os.path.join(DATA_DIR, args.corpus)
    calib = calibration(tok, corpus, args.sequences, 5000, decalage=2)
    control = calibration(tok, corpus, args.controle, 5000, decalage=3)
    old, new = shape["hidden_size"], args.largeur
    print(f"parent : {shape['num_hidden_layers']} blocs, largeur {old}, FFN {shape['intermediate_size']} ; loss {loss_on(model, control, args.device):.3f}", flush=True)
    child, kept_energy = carve_width(model, keep, new, args.ffn, calib, args.device)

    from model.model_minimind import MiniMindConfig, MiniMindForCausalLM

    form = json.load(open(args.fiche_forme))["config"]
    small = MiniMindForCausalLM(MiniMindConfig(vocab_size=len(tok), **form))
    missing, unexpected = small.load_state_dict(child, strict=False)
    assert not missing and not unexpected, f"forme de l'enfant différente de {args.fiche_forme} : {missing} {unexpected}"
    small = small.to(args.device).eval()
    print(f"enfant (largeur {new}, {len(keep)} blocs, FFN {args.ffn}) avant entraînement : loss {loss_on(small, control, args.device):.3f} (hasard {np.log(len(tok)):.2f})")
    fiche = dict(parent=args.parent, blocs_gardes=keep, largeur=new, ffn=args.ffn, tokenizer=os.path.relpath(args.tokenizer, DATA_DIR))
    save_child(small, list(small.model.layers), form, args.sortie, args.nom, fiche)
