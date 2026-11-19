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
class TernaryLinear(nn.Linear):
    scale_per_row = False  # réglé par convert_to_ternary

    def ternary_weight(self):
        w = self.weight
        dims = 1 if self.scale_per_row else None
        scale = w.abs().mean(dim=dims, keepdim=True).clamp(min=1e-5)
        w_q = (w / scale).round().clamp(-1, 1) * scale
        return w + (w_q - w).detach()  # forward : w_q ; backward : gradient de w

    def forward(self, x):
        return F.linear(x, self.ternary_weight().to(x.dtype), self.bias)


def convert_to_ternary(module: nn.Module, scale_per_row: bool = False):
    """Transforme sur place toutes les nn.Linear de `module` en TernaryLinear. Les noms de paramètres
    ne changent pas : un checkpoint haute précision se charge tel quel (arrondi après coup ou QAT)."""
    for m in module.modules():
        if type(m) is nn.Linear:
            m.__class__ = TernaryLinear
            m.scale_per_row = scale_per_row
    return module
