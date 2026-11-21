"""Poids ternaires (−1, 0, +1) à la BitNet b1.58 : le modèle calcule avec des poids arrondis."""
import torch
import torch.nn.functional as F
from torch import nn


# -------------------------------------------------------------------------------------
# TernaryLinear : une couche linéaire dont les poids UTILISÉS ne valent que −s, 0 ou +s
# -------------------------------------------------------------------------------------
# (Ma et al. 2024, « The Era of 1-bit LLMs », BitNet b1.58 : log2(3) = 1,58 bit par poids)
# - self.weight reste en haute précision : ce sont les poids "latents", ceux que l'optimiseur ajuste.
# - À chaque passage, on calcule une échelle s = moyenne des |poids| (par matrice, ou par ligne =
#   par neurone), puis on arrondit poids/s à −1, 0 ou +1.
# - L'arrondi a une dérivée nulle presque partout → astuce STE (straight-through estimator) :
#   au forward on utilise la valeur arrondie, au backward on fait comme si l'arrondi était l'identité.
#   C'est la même astuce que dans le routeur du MoE (feedforward.py).
# Le stockage final ne garde que les signes (−1/0/+1) et les échelles : ~16× plus petit qu'en bf16.
# Variantes reprises de Ternary Bonsai 2 (PrismML, 2026) :
# - échelle par GROUPE de 128 poids consécutifs d'une ligne (plus fine que par ligne) ;
# - rotation de Walsh-Hadamard de l'entrée, par blocs : f(x) = W·(R·x), R orthogonale fixe (signes
#   aléatoires × Hadamard normalisée). Elle étale les valeurs extrêmes sur tout le bloc, ce qui rend
#   l'arrondi à trois valeurs moins destructeur (QuIP#, SpinQuant).
GROUP = 128


def hadamard_block(n):
    """Plus grande puissance de 2 (≤ 512) qui divise n : taille des blocs de la rotation."""
    b = 1
    while n % (b * 2) == 0 and b * 2 <= 512:
        b *= 2
    return b


class TernaryLinear(nn.Linear):
    scale_mode = "matrix"  # matrix | row | group ; réglé par convert_to_ternary
    hadamard = False

    def ternary_weight(self):
        w = self.weight
        if self.scale_mode == "group":
            w = w.view(w.size(0), -1, GROUP)
        dims = {"matrix": None, "row": 1, "group": 2}[self.scale_mode]
        scale = w.abs().mean(dim=dims, keepdim=True).clamp(min=1e-5)
        w_q = (w / scale).round().clamp(-1, 1) * scale
        return (w + (w_q - w).detach()).view_as(self.weight)  # forward : w_q ; backward : gradient de w

    def rotate(self, x):
        b = self.rot.size(0)
        return (x.view(*x.shape[:-1], -1, b) @ self.rot.to(x.dtype)).view_as(x)

    def forward(self, x):
        if self.hadamard:
            x = self.rotate(x)
        return F.linear(x, self.ternary_weight().to(x.dtype), self.bias)


def convert_to_ternary(module: nn.Module, scale_mode: str = "matrix", hadamard: bool = False):
    """Transforme sur place toutes les nn.Linear de `module` en TernaryLinear. Les noms de paramètres
    ne changent pas : un checkpoint haute précision se charge tel quel (arrondi après coup ou QAT)."""
    for m in module.modules():
        if type(m) is nn.Linear:
            m.__class__ = TernaryLinear
            m.scale_mode = scale_mode
            assert scale_mode != "group" or m.in_features % GROUP == 0
            m.hadamard = hadamard
            if hadamard:
                b = hadamard_block(m.in_features)
                h = torch.ones(1, 1)
                while h.size(0) < b:  # construction de Sylvester : H_2n = [[H, H], [H, −H]]
                    h = torch.cat([torch.cat([h, h], 1), torch.cat([h, -h], 1)], 0)
                g = torch.Generator().manual_seed(0)
                signs = torch.randint(0, 2, (b,), generator=g).float() * 2 - 1
                rot = torch.diag(signs) @ h / b**0.5
                m.register_buffer("rot", rot.to(m.weight.device), persistent=False)
    return module
