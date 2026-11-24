"""Grilles 3D : une mémoire que chaque position lit à un endroit, à la place des matrices du FFN."""
import torch
import torch.nn.functional as F
from torch import nn

from .config import MiniMindConfig


# -------------------------------------------------------------------------------------
# GridFFN : le savoir du bloc est rangé dans des grilles 3D, pas dans des matrices
# -------------------------------------------------------------------------------------
# Idée reprise de l'image de synthèse (Instant-NGP, Müller et al. 2022) : au lieu de faire traverser de grandes
# matrices à chaque exemple, on lui fait LIRE une grille de nombres à sa position. Chaque exemple ne touche que
# les 8 cases voisines : elles se remplissent vite, sans déranger le reste.
# Le texte n'a pas de position 3D naturelle : le modèle la calcule. Pour chaque grille, une projection apprise
# transforme l'état du token (768 nombres) en 3 coordonnées entre −1 et 1 ; on lit la grille à cet endroit, par
# interpolation entre les 8 cases voisines. Plusieurs grilles (plusieurs « points de vue ») et plusieurs finesses
# (4, 8, 16, 32 cases par côté) sont lues, mises bout à bout, puis ramenées à la largeur du modèle.
# Coût : quelques lectures par token, au lieu de trois grandes multiplications de matrices.
class GridFFN(nn.Module):
    def __init__(self, config: MiniMindConfig):
        super().__init__()
        self.count, self.features, self.sizes = config.grid_count, config.grid_features, tuple(config.grid_sizes)
        self.coords = nn.Linear(config.hidden_size, 3 * self.count, bias=False)
        # noms en "mem_" : paramètres à accès épars, hors Muon et sans weight decay (comme la couche mémoire)
        self.mem_grids = nn.ParameterList(
            [nn.Parameter(torch.randn(self.count, self.features, r, r, r) * 0.01) for r in self.sizes]
        )
        self.out = nn.Linear(self.count * self.features * len(self.sizes), config.hidden_size, bias=False)

    def forward(self, x):
        batch, seq, _ = x.shape
        where = torch.tanh(self.coords(x)).float().view(batch * seq, self.count, 3).transpose(0, 1)
        where = where[:, :, None, None, :]  # [grilles, tokens, 1, 1, 3]
        read = [
            F.grid_sample(grid.float(), where, mode="bilinear", padding_mode="border", align_corners=True).flatten(2)
            for grid in self.mem_grids
        ]  # par finesse : [grilles, nombres par case, tokens]
        read = torch.cat(read, dim=1).permute(2, 0, 1).reshape(batch, seq, -1)
        return self.out(read.to(x.dtype))
