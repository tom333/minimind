"""RMSNorm : normalisation utilisée avant chaque sous-bloc, en sortie, et sur q/k."""
import torch
from torch import nn


# -------------------------------------------------------------------------------------
# RMSNorm : garder les valeurs sous contrôle
# -------------------------------------------------------------------------------------
# Problème : à travers les couches, les valeurs peuvent exploser ou s'écraser → l'entraînement
# diverge. Solution : ramener chaque vecteur à une "taille" (RMS = Root Mean Square) de 1.
#   RMS(x) = sqrt(mean(x²)) ; norm(x) = x / RMS(x)     (rsqrt = 1/sqrt ; eps évite /0)
#   ex : [3,4,0,0] → [1.2,1.6,0,0]  et  [300,400,0,0] → même résultat (seule la direction compte)
# self.weight : gain appris par dimension (init à 1), pour ré-amplifier si utile.
# Différence avec LayerNorm (GPT-2) : pas de soustraction de la moyenne → plus simple, aussi bon.
# Calcul en float32 (x.float()) même si le modèle tourne en fp16, pour la précision.
# Utilisé : avant attention, avant MLP, en sortie finale, et sur q/k (spécificité Qwen3).
class RMSNorm(torch.nn.Module):
    def __init__(self, dim: int, eps: float = 1e-5):
        super().__init__()
        self.eps = eps
        self.weight = nn.Parameter(torch.ones(dim))

    def norm(self, x):
        return x * torch.rsqrt(x.pow(2).mean(-1, keepdim=True) + self.eps)

    def forward(self, x):
        return (self.weight * self.norm(x.float())).type_as(x)
