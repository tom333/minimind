"""RoPE : encodage des positions par rotation de q et k (+ extension YaRN)."""
import math

import torch


# -------------------------------------------------------------------------------------
# RoPE (Rotary Position Embedding) : dire au modèle OÙ sont les tokens
# -------------------------------------------------------------------------------------
# Problème : l'attention est aveugle à l'ordre ("chien mord homme" = "homme mord chien").
# Idée : TOURNER q et k d'un angle proportionnel à leur position.
#   - Le vecteur de 96 dims = 48 paires = 48 "aiguilles" sur un cadran 2D.
#   - Position p → aiguille tournée de p·θ.
#   - Produit scalaire q(pos m)·k(pos n) ne dépend que de l'angle relatif (m-n)·θ
#     → le modèle perçoit des DISTANCES RELATIVES, pas des positions absolues.
#     (vérifié : score(q@5,k@3) == score(q@105,k@103) == score(q@1505,k@1503))
#   - Les 48 aiguilles tournent à des vitesses différentes (comme secondes/minutes/heures) :
#     rapides = distinguer les voisins, lentes = distinguer les longues distances.
#     freqs_i = 1 / rope_base^(2i/dim)
# cos/sin sont PRÉCALCULÉS pour toutes les positions (0..32767) et stockés en buffer (pas appris).
#
# YaRN (bloc if rope_scaling) : bonus d'inférence pour lire plus long que vu à l'entraînement
# (2048 → ×16). Ralentit les aiguilles lentes pour que les longues distances retombent dans
# des angles déjà connus. f'(i) = f(i)·((1-γ) + γ/s), γ = rampe linéaire entre 0 et 1.
def precompute_freqs_cis(
    dim: int,
    end: int = int(32 * 1024),
    rope_base: float = 1e6,
    rope_scaling: dict = None,
):
    freqs, attn_factor = (
        1.0 / (rope_base ** (torch.arange(0, dim, 2)[: (dim // 2)].float() / dim)),
        1.0,
    )
    if (
        rope_scaling is not None
    ):  # YaRN: f'(i) = f(i)((1-γ) + γ/s), where γ∈[0,1] is linear ramp
        orig_max, factor, beta_fast, beta_slow, attn_factor = (
            rope_scaling.get("original_max_position_embeddings", 2048),
            rope_scaling.get("factor", 16),
            rope_scaling.get("beta_fast", 32.0),
            rope_scaling.get("beta_slow", 1.0),
            rope_scaling.get("attention_factor", 1.0),
        )
        if end / orig_max > 1.0:
            inv_dim = lambda b: (
                (dim * math.log(orig_max / (b * 2 * math.pi)))
                / (2 * math.log(rope_base))
            )
            low, high = (
                max(math.floor(inv_dim(beta_fast)), 0),
                min(math.ceil(inv_dim(beta_slow)), dim // 2 - 1),
            )
            ramp = torch.clamp(
                (torch.arange(dim // 2, device=freqs.device).float() - low)
                / max(high - low, 0.001),
                0,
                1,
            )
            freqs = freqs * (1 - ramp + ramp / factor)
    t = torch.arange(end, device=freqs.device)
    freqs = torch.outer(
        t, freqs
    ).float()  # angle[position, paire] = position × fréquence
    freqs_cos = torch.cat([torch.cos(freqs), torch.cos(freqs)], dim=-1) * attn_factor
    freqs_sin = torch.cat([torch.sin(freqs), torch.sin(freqs)], dim=-1) * attn_factor
    return freqs_cos, freqs_sin


# Applique la rotation 2D : (a,b) → (a·cosθ − b·sinθ, b·cosθ + a·sinθ)
# rotate_half fait (a,b) → (-b,a) sur toutes les paires d'un coup.
# Seuls q et k sont tournés (pas v) : la position sert à décider QUI regarde QUI,
# pas CE QU'ON transmet.
def apply_rotary_pos_emb(q, k, cos, sin, unsqueeze_dim=1):
    def rotate_half(x):
        return torch.cat(
            (-x[..., x.shape[-1] // 2 :], x[..., : x.shape[-1] // 2]), dim=-1
        )

    q_embed = (
        (q * cos.unsqueeze(unsqueeze_dim))
        + (rotate_half(q) * sin.unsqueeze(unsqueeze_dim))
    ).to(q.dtype)
    k_embed = (
        (k * cos.unsqueeze(unsqueeze_dim))
        + (rotate_half(k) * sin.unsqueeze(unsqueeze_dim))
    ).to(k.dtype)
    return q_embed, k_embed
