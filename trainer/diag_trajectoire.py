"""
La fin d'un entraînement se laisse-t-elle prévoir ?

Un essai garde une copie de ses poids tous les 250 pas (--snapshot_interval). On regarde alors le chemin parcouru
par les poids :
  1. est-il droit ? cosinus entre deux déplacements successifs (1 = même direction), et rapport entre la distance
     à vol d'oiseau et le chemin réellement parcouru ;
  2. peut-on sauter ? à partir de deux copies (pas s0 et s1), on prolonge le mouvement jusqu'à la fin de l'essai,
     en tenant compte de la baisse du taux d'apprentissage, et on mesure la loss du point d'arrivée prévu, pour
     une fraction α du saut (α = 0 : pas de saut ; α = 1 : saut complet) ;
  3. vaut-il mieux moyenner ? loss de la moyenne de plusieurs copies successives.
Repères : la loss réelle au pas s1 (ce qu'on a sans rien faire) et la loss réelle en fin d'essai (ce qu'on vise).

Usage : python diag_trajectoire.py ../out/etude_trajectoire/trajectoire-t0_768
"""

import argparse
import glob
import os
import re
import sys

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

import numpy as np
import torch
import torch.nn.functional as F

from model.model_minimind import MiniMindConfig, MiniMindForCausalLM
from trainer.trainer_utils import get_lr

SEQ = 340


def load(path):
    return {k: v.float() for k, v in torch.load(path, map_location="cpu").items()}


def combine(states, weights):
    return {k: sum(w * s[k] for s, w in zip(states, weights)) for k in states[0]}


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("prefixe", help="chemin des copies sans le suffixe _pas<N>.pth")
    parser.add_argument("--donnees", default="../dataset/fr_en/pretrain_fr65.jsonl")
    parser.add_argument("--vocab", type=int, default=16000)
    parser.add_argument("--warmup", type=float, default=0.02)
    parser.add_argument("--pas", type=int, default=16, help="un bloc de validation sur `pas`")
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    args = parser.parse_args()

    files = {int(re.search(r"_pas(\d+)\.pth$", f).group(1)): f for f in glob.glob(f"{args.prefixe}_pas*.pth")}
    steps = sorted(files)
    states = {s: load(files[s]) for s in steps}
    total, keys = steps[-1], [k for k in states[steps[0]] if k != "lm_head.weight"]
    flat = lambda s: torch.cat([states[s][k].flatten() for k in keys])
    val = np.load(f"{args.donnees}.val_rand2000.npy")
    blocks = torch.from_numpy(val[: len(val) // SEQ * SEQ].reshape(-1, SEQ)[:: args.pas].astype(np.int64))
    model = MiniMindForCausalLM(MiniMindConfig(vocab_size=args.vocab)).to(args.device).eval()

    @torch.no_grad()
    def loss_of(state):
        model.load_state_dict(state)
        nll = 0.0
        for b in blocks.split(32):
            b = b.to(args.device)
            with torch.autocast(args.device, dtype=torch.bfloat16, enabled=args.device == "cuda"):
                logits = model(b).logits[:, :-1].float()
            nll += F.cross_entropy(logits.reshape(-1, logits.size(-1)), b[:, 1:].reshape(-1), reduction="sum").item()
        return nll / blocks[:, 1:].numel()

    real = {s: loss_of(states[s]) for s in steps}
    print(f"{len(steps)} copies, pas {steps[0]} à {total} ; loss sur {blocks.numel() / 1e3:.0f}k tokens de validation")
    print("\n1. Le chemin est-il droit ?")
    print(f"{'de':>6}{'à':>6}{'loss réelle':>13}{'déplacement':>13}{'cosinus avec le précédent':>27}")
    moves, path = [], 0.0
    for a, b in zip(steps, steps[1:]):
        move = flat(b) - flat(a)
        cos = F.cosine_similarity(move, moves[-1], dim=0).item() if moves else float("nan")
        moves.append(move)
        path += move.norm().item()
        print(f"{a:6d}{b:6d}{real[b]:13.3f}{move.norm().item():13.1f}{cos:27.2f}")
    print(f"distance à vol d'oiseau / chemin parcouru : {(flat(total) - flat(steps[0])).norm().item() / path:.2f} (1 = ligne droite)")

    warm = int(args.warmup * total)
    lr = np.array([s / warm if s < warm else get_lr(s - warm, total - warm, 1.0) for s in range(total + 1)])
    tau = np.cumsum(lr)  # « temps » mesuré en taux d'apprentissage cumulé : les pas de fin comptent moins
    print("\n2. Peut-on sauter jusqu'à la fin ?")
    print(f"{'copies':>12}{'sans saut':>11}{'α=0,25':>9}{'α=0,5':>8}{'α=1':>7}{'fin réelle':>12}")
    for s0, s1 in [(steps[2], steps[4]), (steps[4], steps[6]), (steps[5], steps[7])]:
        reach = (tau[total] - tau[s1]) / (tau[s1] - tau[s0])
        jumped = [loss_of(combine([states[s1], states[s0]], [1 + a * reach, -a * reach])) for a in (0.25, 0.5, 1.0)]
        print(f"{f'{s0}→{s1}':>12}{real[s1]:11.3f}" + "".join(f"{v:{w}.3f}" for v, w in zip(jumped, (9, 8, 7))) + f"{real[total]:12.3f}")

    print("\n3. Vaut-il mieux moyenner ?")
    for group in (steps[4:7], steps[5:8], steps[-4:], steps[-2:]):
        avg = loss_of(combine([states[s] for s in group], [1 / len(group)] * len(group)))
        print(f"moyenne des copies {group} : {avg:.3f} (dernière du groupe seule : {real[group[-1]]:.3f} ; fin réelle : {real[total]:.3f})")
