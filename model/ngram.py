"""Tables de n-grammes à l'entrée : beaucoup de paramètres, presque aucun calcul par token."""
import torch
import torch.nn.functional as F
from torch import nn

from .config import MiniMindConfig


# -------------------------------------------------------------------------------------
# NgramEmbedding : chaque position reçoit un vecteur qui dépend de ses n derniers tokens
# -------------------------------------------------------------------------------------
# Idée : l'embedding classique ne connaît que le token courant ; les premiers blocs dépensent leur
# capacité à reconstruire le contexte immédiat (« de la », « il y a »). Ici ce contexte est LU dans une
# table : pour chaque ordre n (2, 3, 4…), les n derniers tokens sont hachés en un numéro de ligne.
#   → la table ajoute ordres × lignes × dim paramètres, pour le coût d'une lecture par ordre.
# Deux n-grammes différents peuvent tomber sur la même ligne (collision) : le modèle fait avec.
# Causal par construction : la position t ne lit que les tokens t, t-1, … (jamais le futur).
# Parenté : embeddings hachés (fastText), N-Grammer (Roy et al. 2022).
class NgramEmbedding(nn.Module):
    def __init__(self, config: MiniMindConfig):
        super().__init__()
        self.orders, self.rows = tuple(config.ngram_orders), config.ngram_rows
        # nom en "mem_" : paramètre exclu de Muon et du weight decay (table à accès épars)
        self.mem_ngram = nn.Parameter(torch.randn(len(self.orders) * self.rows, config.ngram_dim) * 0.02)
        self.proj = nn.Linear(len(self.orders) * config.ngram_dim, config.hidden_size, bias=False)

    def forward(self, input_ids):
        h, shifted, parts = input_ids.long(), input_ids.long(), []
        for n in range(2, max(self.orders) + 1):
            shifted = F.pad(shifted[:, :-1], (1, 0))  # token de la position t-(n-1) ; 0 avant le début
            h = (h * 1000003 + shifted * 7919 + n) % 2147483647  # hachage roulant des n derniers tokens
            if n in self.orders:
                parts.append(F.embedding(h % self.rows + self.orders.index(n) * self.rows, self.mem_ngram))
        return self.proj(torch.cat(parts, dim=-1))
