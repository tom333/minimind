"""Couche mémoire à clés produit : beaucoup de paramètres, très peu de calcul par token."""
import torch
import torch.nn.functional as F
from torch import nn

from .config import MiniMindConfig


# -------------------------------------------------------------------------------------
# MemoryLayer : une grande table de "souvenirs" consultée par chaque token
# -------------------------------------------------------------------------------------
# (Lample et al. 2019 « Large Memory Layers with Product Keys » ; Berges et al. 2024 « Memory Layers at Scale »)
# Idée : au lieu de faire traverser tous les poids par chaque token (FFN), le token calcule une
# REQUÊTE, retrouve les k lignes les plus proches dans une table de n² lignes, et ne lit que celles-là.
#   → la table ajoute n²·hidden paramètres (ici 65 536 × 768 ≈ 50M) pour un coût de calcul minime.
# Astuce des CLÉS PRODUIT, pour ne pas comparer la requête aux n² lignes :
#   la requête est coupée en deux moitiés, chacune comparée à n sous-clés (2·n comparaisons) ;
#   la ligne (i, j) de la table a pour score  score1[i] + score2[j]. On garde les k meilleures
#   de chaque moitié (k² candidats), puis les k meilleures paires.
# Sortie = moyenne des k lignes lues, pondérée par softmax(scores). Plusieurs têtes = plusieurs requêtes.
class MemoryLayer(nn.Module):
    def __init__(self, config: MiniMindConfig):
        super().__init__()
        self.n, self.k, self.heads = config.memory_keys, config.memory_k, config.memory_heads
        self.half = config.memory_query_dim // 2
        self.query = nn.Linear(config.hidden_size, self.heads * 2 * self.half, bias=False)
        # normalise les requêtes sur le lot : sans elle, quelques lignes captent tous les accès
        self.query_norm = nn.BatchNorm1d(self.heads * 2 * self.half)
        # noms en "mem_" : paramètres exclus de Muon (table à accès épars, comme un embedding)
        self.mem_keys = nn.Parameter(torch.randn(self.heads, 2, self.n, self.half) / self.half**0.5)
        self.mem_values = nn.Parameter(torch.randn(self.n * self.n, config.hidden_size) * 0.02)

    def lookup(self, x):
        """Retourne (indices [N, heads·k] dans la table, poids [N, heads·k])."""
        q = self.query_norm(self.query(x).float()).view(-1, self.heads, 2, self.half)  # [N, H, 2, half]
        scores = torch.einsum("nhcd,hckd->nhck", q, self.mem_keys.float())  # [N, H, 2, n]
        s, i = scores.topk(self.k, dim=-1)  # k meilleures sous-clés de chaque moitié
        pair_scores = (s[:, :, 0, :, None] + s[:, :, 1, None, :]).flatten(2)  # [N, H, k²]
        pair_idx = (i[:, :, 0, :, None] * self.n + i[:, :, 1, None, :]).flatten(2)
        best, pos = pair_scores.topk(self.k, dim=-1)  # k meilleures paires
        idx = pair_idx.gather(-1, pos)
        return idx.flatten(1), F.softmax(best, dim=-1).flatten(1)

    def forward(self, x):
        lead = x.shape[:-1]
        idx, w = self.lookup(x.reshape(-1, x.size(-1)))
        # embedding_bag : somme pondérée des lignes lues, sans matérialiser [N, heads·k, hidden]
        out = F.embedding_bag(idx, self.mem_values, per_sample_weights=w.to(self.mem_values.dtype), mode="sum")
        return out.view(*lead, -1).to(x.dtype)
