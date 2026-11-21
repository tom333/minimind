"""Attention multi-têtes avec GQA, RoPE, QK-norm, masque causal et KV-cache."""

import math

import torch
import torch.nn.functional as F
from torch import nn

from .config import MiniMindConfig
from .norm import RMSNorm
from .rope import apply_rotary_pos_emb


# GQA : duplique les têtes K/V pour qu'elles correspondent au nombre de têtes Q.
# ex n_rep=2 : têtes kv [0,1,2,3] → [0,0,1,1,2,2,3,3] (chaque K/V partagée par 2 têtes Q)
def repeat_kv(x: torch.Tensor, n_rep: int) -> torch.Tensor:
    bs, slen, num_key_value_heads, head_dim = x.shape
    if n_rep == 1:
        return x
    return (
        x[:, :, :, None, :]
        .expand(bs, slen, num_key_value_heads, n_rep, head_dim)
        .reshape(bs, slen, num_key_value_heads * n_rep, head_dim)
    )


# -------------------------------------------------------------------------------------
# Attention : les tokens échangent de l'information
# -------------------------------------------------------------------------------------
# Chaque token produit 3 vecteurs (analogie moteur de recherche) :
#   Q (Query) = "ce que je cherche"   K (Key) = "mon étiquette"   V (Value) = "mon contenu"
#   score(i,j) = Qi·Kj / sqrt(head_dim)     (/sqrt évite la saturation du softmax)
#   poids(i,·) = softmax(scores ligne i)    (somme = 1)
#   sortie(i)  = Σj poids(i,j)·Vj           (moyenne pondérée des contenus)
#
# Multi-têtes : 8 attentions indépendantes de 96 dims (8×96=768) au lieu d'une de 768.
#   Chaque tête peut se spécialiser (syntaxe, token précédent, ponctuation...).
#   o_proj recombine les points de vue.
#
# GQA (Grouped Query Attention) : 8 têtes Q mais seulement 4 têtes K/V
#   → k_proj/v_proj sortent 384 au lieu de 768 → KV-cache divisé par 2
#     (32k tokens fp16 : 0.81 Go → 0.40 Go), perte de qualité négligeable.
#
# Masque causal : un token ne voit que le PASSÉ (triangle de -inf au-dessus de la diagonale,
#   -inf → 0 après softmax). Empêche de "lire la réponse" à l'entraînement, et permet
#   d'entraîner toutes les positions d'une séquence en UNE passe.
#
# KV-cache : en génération, K et V des tokens passés ne changent pas (grâce au masque causal)
#   → on les garde et on ne calcule que ceux du nouveau token.
class Attention(nn.Module):
    def __init__(self, config: MiniMindConfig):
        super().__init__()
        self.num_key_value_heads = (
            config.num_attention_heads
            if config.num_key_value_heads is None
            else config.num_key_value_heads
        )
        self.n_local_heads = config.num_attention_heads
        self.n_local_kv_heads = self.num_key_value_heads
        self.n_rep = (
            self.n_local_heads // self.n_local_kv_heads
        )  # 8/4 = 2 têtes Q par tête K/V
        self.head_dim = config.head_dim
        self.is_causal = True
        self.q_proj = nn.Linear(
            config.hidden_size, config.num_attention_heads * self.head_dim, bias=False
        )  # 768 → 768
        self.k_proj = nn.Linear(
            config.hidden_size, self.num_key_value_heads * self.head_dim, bias=False
        )  # 768 → 384 (GQA)
        self.v_proj = nn.Linear(
            config.hidden_size, self.num_key_value_heads * self.head_dim, bias=False
        )  # 768 → 384 (GQA)
        self.o_proj = nn.Linear(
            config.num_attention_heads * self.head_dim, config.hidden_size, bias=False
        )  # 768 → 768
        self.q_norm = RMSNorm(
            self.head_dim, eps=config.rms_norm_eps
        )  # QK-norm (Qwen3) : stabilise l'attention
        self.k_norm = RMSNorm(self.head_dim, eps=config.rms_norm_eps)
        self.sub_norm = (
            RMSNorm(config.num_attention_heads * self.head_dim, eps=config.rms_norm_eps)
            if config.subln
            else nn.Identity()
        )  # SubLN (BitNet), cf. config.subln
        self.attn_dropout = nn.Dropout(config.dropout)
        self.resid_dropout = nn.Dropout(config.dropout)
        self.dropout = config.dropout
        self.flash = (
            hasattr(torch.nn.functional, "scaled_dot_product_attention")
            and config.flash_attn
        )

    def forward(
        self,
        x,
        position_embeddings,
        past_key_value=None,
        use_cache=False,
        attention_mask=None,
    ):
        # Shapes indiquées pour batch=1, seq_len=10
        bsz, seq_len, _ = x.shape  # x : [1, 10, 768]
        xq, xk, xv = (
            self.q_proj(x),
            self.k_proj(x),
            self.v_proj(x),
        )  # [1,10,768] [1,10,384] [1,10,384]
        xq = xq.view(bsz, seq_len, self.n_local_heads, self.head_dim)  # [1, 10, 8, 96]
        xk = xk.view(
            bsz, seq_len, self.n_local_kv_heads, self.head_dim
        )  # [1, 10, 4, 96]
        xv = xv.view(
            bsz, seq_len, self.n_local_kv_heads, self.head_dim
        )  # [1, 10, 4, 96]
        xq, xk = self.q_norm(xq), self.k_norm(xk)
        cos, sin = position_embeddings
        xq, xk = apply_rotary_pos_emb(xq, xk, cos, sin)  # position via rotation
        # KV-cache : on concatène K/V du passé avec ceux des nouveaux tokens
        if past_key_value is not None:
            xk = torch.cat([past_key_value[0], xk], dim=1)
            xv = torch.cat([past_key_value[1], xv], dim=1)
        past_kv = (xk, xv) if use_cache else None
        # repeat_kv 4→8 têtes, puis transpose → [batch, têtes, seq, head_dim] = [1, 8, 10, 96]
        # .contiguous() : avec 1 seule tête K/V (n_rep=8), l'expand de repeat_kv donne une disposition
        # mémoire que le backward compilé de Flash Attention rejette (bug torch.compile). Mêmes valeurs.
        xq, xk, xv = (
            xq.transpose(1, 2),
            repeat_kv(xk, self.n_rep).transpose(1, 2).contiguous(),
            repeat_kv(xv, self.n_rep).transpose(1, 2).contiguous(),
        )
        # Chemin rapide : Flash Attention (kernel GPU optimisé, même calcul, ne matérialise
        # pas la matrice seq×seq). Utilisé si pas de cache ni de padding.
        if (
            self.flash
            and (seq_len > 1)
            and (not self.is_causal or past_key_value is None)
            and (attention_mask is None or torch.all(attention_mask == 1))
        ):
            output = F.scaled_dot_product_attention(
                xq,
                xk,
                xv,
                dropout_p=self.dropout if self.training else 0.0,
                is_causal=self.is_causal,
            )
        else:
            # Chemin "manuel" lisible : la formule de l'attention en clair
            scores = (xq @ xk.transpose(-2, -1)).float() / math.sqrt(
                self.head_dim
            )  # Q·Kᵀ/√d
            # masque causal : -inf au-dessus de la diagonale (interdit de voir le futur)
            if self.is_causal:
                scores[:, :, :, -seq_len:] += torch.full(
                    (seq_len, seq_len), float("-inf"), device=scores.device
                ).triu(1)
            # masque de padding : ignore les tokens de remplissage
            if attention_mask is not None:
                scores += (1.0 - attention_mask.unsqueeze(1).unsqueeze(2)) * -1e9
            output = (
                self.attn_dropout(F.softmax(scores, dim=-1).type_as(xq)) @ xv
            )  # softmax · V
        output = output.transpose(1, 2).reshape(
            bsz, seq_len, -1
        )  # recolle les têtes → [1, 10, 768]
        output = self.resid_dropout(self.o_proj(self.sub_norm(output)))  # mélange final des têtes
        return output, past_kv
