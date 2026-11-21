"""FFN SwiGLU (dense ou à matrices structurées BTT) et Mixture of Experts."""

import torch
import torch.nn.functional as F
from torch import nn
from transformers.activations import ACT2FN

from .config import MiniMindConfig
from .norm import RMSNorm


# -------------------------------------------------------------------------------------
# BTTLinear : matrice STRUCTURÉE (Block Tensor-Train à 2 noyaux, Qiu et al. 2024 ; rang 1 = Monarch)
# -------------------------------------------------------------------------------------
# Remplace une matrice pleine  in → out  par deux étages de PETITES matrices :
#   l'entrée (in = a·b) est vue comme une grille [a, b], la sortie (out = e·c) comme [e, c]
#   étage 1 : pour chacune des a lignes, une matrice  b → r·c          (a blocs indépendants)
#   étage 2 : pour chacune des c colonnes, une matrice  a·r → e        (c blocs indépendants)
# Coût (paramètres ET calcul par token) = r·c·a·(b + e)   au lieu de   a·b·e·c pour la matrice pleine.
# → à coût égal, on peut avoir une sortie beaucoup plus LARGE (ex. 768 → 8192 au prix de 768 → 2432).
# Les noyaux sont stockés [bloc, sortie, entrée] (comme nn.Linear) pour l'optimiseur Muon par bloc.
class BTTLinear(nn.Module):
    def __init__(self, in_dims, out_dims, rank, init_std=0.02):
        super().__init__()
        (self.a, self.b), (self.e, self.c), self.r = in_dims, out_dims, rank
        # chaque poids "équivalent" est une somme de r produits core1·core2 → même écart-type qu'une
        # matrice pleine initialisée à init_std
        std = (init_std**2 / rank) ** 0.25
        self.core1 = nn.Parameter(torch.randn(self.a, rank * self.c, self.b) * std)
        self.core2 = nn.Parameter(torch.randn(self.c, self.e, self.a * rank) * std)

    def forward(self, x):
        lead = x.shape[:-1]
        x = x.reshape(-1, self.a, self.b)
        z = torch.einsum("nab,akb->nak", x, self.core1.to(x.dtype))  # [N, a, r·c]
        z = z.reshape(-1, self.a, self.r, self.c).permute(0, 3, 1, 2).reshape(-1, self.c, self.a * self.r)
        y = torch.einsum("ncj,cej->nec", z, self.core2.to(x.dtype))  # [N, e, c]
        return y.reshape(*lead, self.e * self.c)


# -------------------------------------------------------------------------------------
# FeedForward SwiGLU : la "mémoire" du modèle
# -------------------------------------------------------------------------------------
#   x [768] ─┬─ gate_proj → [2432] → SiLU ─┐
#            └─ up_proj   → [2432] ────────┴─ × → down_proj → [768]
# - Traite chaque token ISOLÉMENT (pas de communication, c'est le rôle de l'attention).
# - Vu comme une mémoire clé-valeur : chaque neurone détecte un motif et ajoute une info.
#   5.6M des 7.4M params du bloc → l'essentiel des connaissances factuelles.
# - SiLU(x) = x·sigmoid(x) : non-linéarité (ReLU "arrondi"). Sans elle, empiler des Linear
#   = une seule multiplication de matrices.
# - GLU (gate × up) : la branche gate agit comme un "robinet" neurone par neurone.
#   Empiriquement meilleur qu'un FFN classique down(relu(up(x))). Standard depuis PaLM/Llama.
class FeedForward(nn.Module):
    def __init__(self, config: MiniMindConfig, intermediate_size: int = None):
        super().__init__()
        intermediate_size = intermediate_size or config.intermediate_size
        if getattr(config, "ffn_btt_rank", 0):
            # FFN structuré : même SwiGLU, mais les 3 matrices sont des BTT → largeur interne m1·m2
            (h1, h2), (m1, m2), r = config.ffn_btt_hidden, config.ffn_btt_mid, config.ffn_btt_rank
            assert h1 * h2 == config.hidden_size, "ffn_btt_hidden doit factoriser hidden_size"
            # FFN plus large → sortie plus grande à l'init ; on réduit l'écart-type pour retrouver
            # l'échelle de sortie du FFN plein (3 matrices en cascade → puissance 1/6)
            std = 0.02 * (config.intermediate_size / (m1 * m2)) ** (1 / 6)
            self.gate_proj = BTTLinear((h1, h2), (m1, m2), r, std)
            self.up_proj = BTTLinear((h1, h2), (m1, m2), r, std)
            self.down_proj = BTTLinear((m2, m1), (h2, h1), r, std)
        else:
            if config.ffn_glu:
                self.gate_proj = nn.Linear(config.hidden_size, intermediate_size, bias=False)
            self.down_proj = nn.Linear(intermediate_size, config.hidden_size, bias=False)
            self.up_proj = nn.Linear(config.hidden_size, intermediate_size, bias=False)
        self.act_fn = ACT2FN[config.hidden_act]
        self.glu = config.ffn_glu
        self.sub_norm = RMSNorm(intermediate_size, eps=config.rms_norm_eps) if config.subln else nn.Identity()

    def forward(self, x):
        h = self.act_fn(self.gate_proj(x)) * self.up_proj(x) if self.glu else self.act_fn(self.up_proj(x))
        return self.down_proj(self.sub_norm(h))


# -------------------------------------------------------------------------------------
# MoE (Mixture of Experts) : utilisé seulement si use_moe=True
# -------------------------------------------------------------------------------------
# 4 FFN "experts" + un routeur (self.gate) qui choisit, pour CHAQUE token, le meilleur expert.
# - Chaque token ne passe que par 1 expert sur 4 → 4× plus de paramètres stockés (198M)
#   mais calcul par token ≈ modèle 64M ("198M-A64M", A = Active).
# - aux_loss : pénalité si le routeur envoie tout au même expert (sinon 1 expert fait tout,
#   les autres ne servent à rien). Ajoutée à la loss principale.
# - Architecture de DeepSeek-V3, Mixtral, Qwen3-MoE.
class MOEFeedForward(nn.Module):
    def __init__(self, config: MiniMindConfig):
        super().__init__()
        self.config = config
        self.gate = nn.Linear(config.hidden_size, config.num_experts, bias=False)
        self.experts = nn.ModuleList(
            [
                FeedForward(config, intermediate_size=config.moe_intermediate_size)
                for _ in range(config.num_experts)
            ]
        )
        self.act_fn = ACT2FN[config.hidden_act]

    def forward(self, x):
        batch_size, seq_len, hidden_dim = x.shape
        x_flat = x.view(-1, hidden_dim)
        scores = F.softmax(
            self.gate(x_flat), dim=-1
        )  # le routeur note les experts pour chaque token
        topk_weight, topk_idx = torch.topk(
            scores, k=self.config.num_experts_per_tok, dim=-1, sorted=False
        )  # garde le(s) meilleur(s)
        if self.config.norm_topk_prob:
            if self.config.num_experts_per_tok > 1:
                topk_weight = topk_weight / (
                    topk_weight.sum(dim=-1, keepdim=True) + 1e-20
                )
            else:
                top1 = torch.topk(
                    F.softmax(self.gate(x_flat.detach()), dim=-1),
                    k=1,
                    dim=-1,
                    sorted=False,
                )[0]
                topk_weight = (
                    top1 - top1.detach() + 1.0
                )  # k=1: 1.0 forward on purpose, gradient via straight-through
        y = torch.zeros_like(x_flat)
        # chaque expert traite uniquement les tokens qui lui ont été routés
        for i, expert in enumerate(self.experts):
            mask = topk_idx == i
            if mask.any():
                token_idx = mask.any(dim=-1).nonzero().flatten()
                weight = topk_weight[mask].view(-1, 1)
                y.index_add_(
                    0, token_idx, (expert(x_flat[token_idx]) * weight).to(y.dtype)
                )
            elif self.training:
                y[0, 0] += 0 * sum(
                    p.sum() for p in expert.parameters()
                )  # garde l'expert inutilisé dans le graphe (DDP)
        # aux_loss d'équilibrage : charge réelle × proba moyenne, minimale si répartition uniforme
        if self.training and self.config.router_aux_loss_coef > 0:
            load = F.one_hot(topk_idx, self.config.num_experts).float().mean(0)
            self.aux_loss = (
                (load * scores.mean(0)).sum()
                * self.config.num_experts
                * self.config.router_aux_loss_coef
            )
        else:
            self.aux_loss = scores.new_zeros(1).squeeze()
        return y.view(batch_size, seq_len, hidden_dim)
