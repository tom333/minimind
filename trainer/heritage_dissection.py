"""
Dissection de l'héritage : qu'est-ce qui, dans une matrice héritée, porte la valeur ?

On part de l'enfant taillé (trainer/heritage.py) et on abîme ses blocs d'une façon précise avant l'entraînement.
Chaque variante est ensuite réparée avec la même recette et le même budget : celle qui apprend aussi vite que
l'enfant intact montre que la propriété détruite ne comptait pas.
  signes        chaque poids des blocs devient −a, 0 ou +a (a = moyenne des valeurs absolues de sa ligne) :
                il ne reste que le schéma de câblage, 1,58 bit par poids
  spectre       mêmes valeurs singulières, directions tirées au hasard : il ne reste que l'« échelle » des matrices
  directions    mêmes directions, valeurs singulières toutes égales (norme conservée)
  attention     attention héritée, FFN aléatoires          ffn        FFN hérités, attention aléatoire
  embedding     embedding hérité, blocs aléatoires         blocs      blocs hérités, embedding aléatoire
  desordre      blocs hérités, rangés dans l'ordre inverse
Les parties « aléatoires » viennent de l'initialisation ordinaire du modèle (même graine pour toutes).

Usage : python heritage_dissection.py ../out/heritage/q3-06-b8   (écrit q3-06-b8-<variante>_1024.pth à côté)
"""

import argparse
import json
import os
import sys

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

import torch

from model.model_minimind import MiniMindConfig, MiniMindForCausalLM

MATRICES = ("q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj")


def is_block_matrix(name):
    return ".layers." in name and name.split(".")[-2] in MATRICES


def signs(w):
    scale = w.abs().mean(dim=1, keepdim=True)
    return (w / scale).round().clamp(-1, 1) * scale


def spectrum_only(w, generator):
    s = torch.linalg.svdvals(w)
    u, _ = torch.linalg.qr(torch.randn(w.size(0), len(s), generator=generator))
    v, _ = torch.linalg.qr(torch.randn(w.size(1), len(s), generator=generator))
    return (u * s) @ v.T


def directions_only(w):
    u, s, vh = torch.linalg.svd(w, full_matrices=False)
    return u @ vh * (s.pow(2).mean().sqrt())  # même norme de Frobenius, spectre plat


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("base", help="chemin sans suffixe de l'enfant taillé, ex. ../out/heritage/q3-06-b8")
    args = parser.parse_args()
    fiche = json.load(open(f"{args.base}.json"))
    width = fiche["config"]["hidden_size"]
    child = {k: v.float() for k, v in torch.load(f"{args.base}_{width}.pth", map_location="cpu").items()}
    torch.manual_seed(0)
    vocab = child["model.embed_tokens.weight"].size(0)
    fresh = {k: v.float() for k, v in MiniMindForCausalLM(MiniMindConfig(vocab_size=vocab, **fiche["config"])).state_dict().items()}
    generator = torch.Generator().manual_seed(0)
    n_layers = fiche["config"]["num_hidden_layers"]
    in_attention = lambda k: ".self_attn." in k or ".input_layernorm." in k
    in_ffn = lambda k: ".mlp." in k or ".post_attention_layernorm." in k
    in_blocks = lambda k: ".layers." in k

    def reversed_blocks():
        out = dict(child)
        for k, v in child.items():
            if in_blocks(k):
                parts = k.split(".")
                parts[2] = str(n_layers - 1 - int(parts[2]))
                out[".".join(parts)] = v
        return out

    variants = {
        "signes": {k: signs(v) if is_block_matrix(k) else v for k, v in child.items()},
        "spectre": {k: spectrum_only(v, generator) if is_block_matrix(k) else v for k, v in child.items()},
        "directions": {k: directions_only(v) if is_block_matrix(k) else v for k, v in child.items()},
        "attention": {k: fresh[k] if in_ffn(k) else v for k, v in child.items()},
        "ffn": {k: fresh[k] if in_attention(k) else v for k, v in child.items()},
        "embedding": {k: fresh[k] if in_blocks(k) else v for k, v in child.items()},
        "blocs": {k: v if in_blocks(k) else fresh[k] for k, v in child.items()},
        "desordre": reversed_blocks(),
    }
    for name, state in variants.items():
        assert state.keys() == child.keys()
        changed = sum(v.numel() for k, v in state.items() if not torch.equal(v, child[k])) / 1e6
        path = f"{args.base}-{name}_{width}.pth"
        torch.save({k: v.half() for k, v in state.items()}, path)
        json.dump({**fiche, "variante": name}, open(f"{args.base}-{name}.json", "w"), indent=1)
        print(f"{name:11} {changed:6.1f}M de poids modifiés → {path}", flush=True)
