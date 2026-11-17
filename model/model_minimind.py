import math, torch, torch.nn.functional as F
from torch import nn
from transformers.activations import ACT2FN
from transformers import PreTrainedModel, GenerationMixin, PretrainedConfig
from transformers.modeling_outputs import MoeCausalLMOutputWithPast

# =====================================================================================
# GUIDE DE LECTURE (annotations pédagogiques)
# -------------------------------------------------------------------------------------
# Ce fichier DÉFINIT l'architecture (le "plan"). Instancier MiniMindForCausalLM(config)
# donne un modèle aux poids ALÉATOIRES ; les poids utiles viennent de l'entraînement
# (trainer/*.py) ou d'un fichier .pth / .safetensors.
#
# Flux complet (≈ architecture Qwen3 / Llama 3) :
#
#   ids → Embedding → [ RMSNorm → Attention(Q,K,V + RoPE + masque causal + GQA) → +résiduel
#                       RMSNorm → FFN SwiGLU (ou MoE)                          → +résiduel ] × 8
#       → RMSNorm → lm_head → logits (un score par token du vocabulaire)
#   entraînement : cross_entropy(logits décalés d'une position, labels)
#   inférence    : boucle generate() + KV-cache + sampling (température, top-k, top-p)
#
# Répartition des 63.9M paramètres (config par défaut) :
#   embedding 4.9M (6400 × 768, partagé avec lm_head)
#   1 bloc    7.4M = attention 1.8M + MLP 5.6M   → × 8 blocs ≈ 59M
# =====================================================================================


# 🌏🌎🌍🌏🌎🌍🌏🌎🌍🌏🌎🌍🌏🌎🌍🌏🌎🌍🌏🌎🌍🌏🌎🌍🌏🌎🌍🌏🌎🌍🌏🌎🌍🌏🌎🌍🌏🌎🌍🌏🌎🌍🌏🌎🌍🌏🌎🌍🌏🌎🌍🌏
#                                     MiniMind Config
# 🌏🌎🌍🌏🌎🌍🌏🌎🌍🌏🌎🌍🌏🌎🌍🌏🌎🌍🌏🌎🌍🌏🌎🌍🌏🌎🌍🌏🌎🌍🌏🌎🌍🌏🌎🌍🌏🌎🌍🌏🌎🌍🌏🌎🌍🌏🌎🌍🌏🌎🌍🌏
# Toutes les dimensions du modèle. Les plus importantes :
#   hidden_size=768        "largeur" : chaque token = vecteur de 768 nombres
#   num_hidden_layers=8    "profondeur" : 8 blocs empilés
#   vocab_size=6400        taille du vocabulaire du tokenizer (BPE ByteLevel)
#   num_attention_heads=8  têtes Q ; num_key_value_heads=4 têtes K/V (GQA, cf. Attention)
#   head_dim = 768/8 = 96  dimension de chaque tête
#   intermediate_size      largeur interne du FFN = ceil(768·π/64)·64 = 2432
#                          (≈3.14× hidden, arrondi multiple de 64 : plus efficace sur GPU)
#   rope_theta=1e6         base des fréquences RoPE (grande = aiguilles lentes = long contexte)
#   tie_word_embeddings    même matrice pour embedding d'entrée et lm_head de sortie
class MiniMindConfig(PretrainedConfig):
    model_type = "minimind"

    def __init__(self, hidden_size=768, num_hidden_layers=8, use_moe=False, **kwargs):
        super().__init__(**kwargs)
        self.hidden_size = hidden_size
        self.num_hidden_layers = num_hidden_layers
        self.use_moe = use_moe
        self.dropout = kwargs.get("dropout", 0.0)
        self.vocab_size = kwargs.get("vocab_size", 6400)
        self.bos_token_id = kwargs.get("bos_token_id", 1)
        self.eos_token_id = kwargs.get("eos_token_id", 2)
        self.flash_attn = kwargs.get("flash_attn", True)
        self.num_attention_heads = kwargs.get("num_attention_heads", 8)
        self.num_key_value_heads = kwargs.get("num_key_value_heads", 4)
        self.head_dim = kwargs.get(
            "head_dim", self.hidden_size // self.num_attention_heads
        )
        self.hidden_act = kwargs.get("hidden_act", "silu")
        self.intermediate_size = kwargs.get(
            "intermediate_size", math.ceil(hidden_size * math.pi / 64) * 64
        )
        self.max_position_embeddings = kwargs.get("max_position_embeddings", 32768)
        self.rms_norm_eps = kwargs.get("rms_norm_eps", 1e-6)
        self.rope_theta = kwargs.get("rope_theta", 1e6)
        self.tie_word_embeddings = kwargs.get("tie_word_embeddings", True)
        self.inference_rope_scaling = kwargs.get("inference_rope_scaling", False)
        # YaRN (extrapolation de contexte à l'inférence), désactivé par défaut
        self.rope_scaling = (
            {
                "beta_fast": 32,
                "beta_slow": 1,
                "factor": 16,
                "original_max_position_embeddings": 2048,
                "attention_factor": 1.0,
                "type": "yarn",
            }
            if self.inference_rope_scaling
            else None
        )
        ### MoE specific configs (ignored if use_moe = False)
        self.num_experts = kwargs.get("num_experts", 4)
        self.num_experts_per_tok = kwargs.get("num_experts_per_tok", 1)
        self.moe_intermediate_size = kwargs.get(
            "moe_intermediate_size", self.intermediate_size
        )
        self.norm_topk_prob = kwargs.get("norm_topk_prob", True)
        self.router_aux_loss_coef = kwargs.get("router_aux_loss_coef", 5e-4)


# 🌏🌎🌍🌏🌎🌍🌏🌎🌍🌏🌎🌍🌏🌎🌍🌏🌎🌍🌏🌎🌍🌏🌎🌍🌏🌎🌍🌏🌎🌍🌏🌎🌍🌏🌎🌍🌏🌎🌍🌏🌎🌍🌏🌎🌍🌏🌎🌍🌏🌎🌍🌏
#                                     MiniMind Model
# 🌏🌎🌍🌏🌎🌍🌏🌎🌍🌏🌎🌍🌏🌎🌍🌏🌎🌍🌏🌎🌍🌏🌎🌍🌏🌎🌍🌏🌎🌍🌏🌎🌍🌏🌎🌍🌏🌎🌍🌏🌎🌍🌏🌎🌍🌏🌎🌍🌏🌎🌍🌏


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
        xq, xk, xv = (
            xq.transpose(1, 2),
            repeat_kv(xk, self.n_rep).transpose(1, 2),
            repeat_kv(xv, self.n_rep).transpose(1, 2),
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
        output = self.resid_dropout(self.o_proj(output))  # mélange final des têtes
        return output, past_kv


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
        self.gate_proj = nn.Linear(config.hidden_size, intermediate_size, bias=False)
        self.down_proj = nn.Linear(intermediate_size, config.hidden_size, bias=False)
        self.up_proj = nn.Linear(config.hidden_size, intermediate_size, bias=False)
        self.act_fn = ACT2FN[config.hidden_act]

    def forward(self, x):
        return self.down_proj(self.act_fn(self.gate_proj(x)) * self.up_proj(x))


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


# -------------------------------------------------------------------------------------
# Bloc Transformer : la brique répétée 8 fois
# -------------------------------------------------------------------------------------
#   h = h + Attention(RMSNorm(h))   → "communiquer" entre tokens
#   h = h + MLP(RMSNorm(h))         → "réfléchir" token par token
# - Connexion résiduelle (+ h) : chaque bloc AJOUTE une correction au lieu de remplacer
#   le vecteur → le signal (et le gradient) traverse les couches sans se dégrader.
# - Pre-norm (normaliser AVANT chaque sous-bloc) : standard moderne, plus stable que post-norm.
class MiniMindBlock(nn.Module):
    def __init__(self, layer_id: int, config: MiniMindConfig):
        super().__init__()
        self.self_attn = Attention(config)
        self.input_layernorm = RMSNorm(config.hidden_size, eps=config.rms_norm_eps)
        self.post_attention_layernorm = RMSNorm(
            config.hidden_size, eps=config.rms_norm_eps
        )
        self.mlp = FeedForward(config) if not config.use_moe else MOEFeedForward(config)

    def forward(
        self,
        hidden_states,
        position_embeddings,
        past_key_value=None,
        use_cache=False,
        attention_mask=None,
    ):
        residual = hidden_states
        hidden_states, present_key_value = self.self_attn(
            self.input_layernorm(hidden_states),
            position_embeddings,
            past_key_value,
            use_cache,
            attention_mask,
        )
        hidden_states += residual  # résiduel après attention
        hidden_states = hidden_states + self.mlp(
            self.post_attention_layernorm(hidden_states)
        )  # résiduel après MLP
        return hidden_states, present_key_value


# -------------------------------------------------------------------------------------
# MiniMindModel : le "tronc" (embedding → 8 blocs → norme finale), sans tête de prédiction
# -------------------------------------------------------------------------------------
class MiniMindModel(nn.Module):
    def __init__(self, config: MiniMindConfig):
        super().__init__()
        self.config = config
        self.vocab_size, self.num_hidden_layers = (
            config.vocab_size,
            config.num_hidden_layers,
        )
        # Embedding = table 6400 lignes × 768 colonnes : le token 500 → ligne 500.
        # Aléatoire au départ ; l'entraînement rapproche les vecteurs des tokens de sens proche.
        self.embed_tokens = nn.Embedding(config.vocab_size, config.hidden_size)
        self.dropout = nn.Dropout(config.dropout)
        self.layers = nn.ModuleList(
            [MiniMindBlock(l, config) for l in range(self.num_hidden_layers)]
        )
        self.norm = RMSNorm(config.hidden_size, eps=config.rms_norm_eps)
        # cos/sin RoPE précalculés : buffers (déplacés sur GPU avec le modèle) mais pas des poids
        # appris, et non sauvegardés (persistent=False)
        freqs_cos, freqs_sin = precompute_freqs_cis(
            dim=config.head_dim,
            end=config.max_position_embeddings,
            rope_base=config.rope_theta,
            rope_scaling=config.rope_scaling,
        )
        self.register_buffer("freqs_cos", freqs_cos, persistent=False)
        self.register_buffer("freqs_sin", freqs_sin, persistent=False)

    def forward(
        self,
        input_ids,
        attention_mask=None,
        past_key_values=None,
        use_cache=False,
        **kwargs,
    ):
        batch_size, seq_length = input_ids.shape
        if hasattr(past_key_values, "layers"):
            past_key_values = None
        past_key_values = past_key_values or [None] * len(self.layers)
        # start_pos = nb de tokens déjà dans le KV-cache → les nouveaux tokens reçoivent
        # la bonne position RoPE (et pas 0)
        start_pos = (
            past_key_values[0][0].shape[1] if past_key_values[0] is not None else 0
        )
        hidden_states = self.dropout(
            self.embed_tokens(input_ids)
        )  # ids → vecteurs [B, T, 768]
        # Recompute RoPE buffers lost during meta-device init (transformers>=5.x)
        if self.freqs_cos[0, 0] == 0:
            freqs_cos, freqs_sin = precompute_freqs_cis(
                dim=self.config.head_dim,
                end=self.config.max_position_embeddings,
                rope_base=self.config.rope_theta,
                rope_scaling=self.config.rope_scaling,
            )
            self.freqs_cos, self.freqs_sin = (
                freqs_cos.to(hidden_states.device),
                freqs_sin.to(hidden_states.device),
            )
        position_embeddings = (
            self.freqs_cos[start_pos : start_pos + seq_length],
            self.freqs_sin[start_pos : start_pos + seq_length],
        )
        presents = []
        for layer, past_key_value in zip(self.layers, past_key_values):
            hidden_states, present = layer(
                hidden_states,
                position_embeddings,
                past_key_value=past_key_value,
                use_cache=use_cache,
                attention_mask=attention_mask,
            )
            presents.append(present)
        hidden_states = self.norm(hidden_states)
        aux_loss = sum(
            [l.mlp.aux_loss for l in self.layers if isinstance(l.mlp, MOEFeedForward)],
            hidden_states.new_zeros(1).squeeze(),
        )
        return hidden_states, presents, aux_loss


# -------------------------------------------------------------------------------------
# MiniMindForCausalLM : le modèle complet (tronc + lm_head + loss + génération)
# -------------------------------------------------------------------------------------
# C'est la classe qu'on instancie. Hérite de PreTrainedModel/GenerationMixin pour être
# compatible transformers (from_pretrained, save_pretrained...).
class MiniMindForCausalLM(PreTrainedModel, GenerationMixin):
    config_class = MiniMindConfig
    _tied_weights_keys = {"lm_head.weight": "model.embed_tokens.weight"}

    def __init__(self, config: MiniMindConfig = None):
        self.config = config or MiniMindConfig()
        super().__init__(self.config)
        self.model = MiniMindModel(self.config)
        # lm_head : vecteur 768 → 6400 scores (logits), un par token du vocabulaire
        self.lm_head = nn.Linear(
            self.config.hidden_size, self.config.vocab_size, bias=False
        )
        # Weight tying : même matrice pour embedding et lm_head ("reconnaître" et "prédire" un
        # token sont liés) → économise 4.9M params (8% du modèle)
        if self.config.tie_word_embeddings:
            self.model.embed_tokens.weight = self.lm_head.weight
        self.post_init()

    def forward(
        self,
        input_ids,
        attention_mask=None,
        past_key_values=None,
        use_cache=False,
        logits_to_keep=0,
        labels=None,
        **kwargs,
    ):
        hidden_states, past_key_values, aux_loss = self.model(
            input_ids, attention_mask, past_key_values, use_cache, **kwargs
        )
        slice_indices = (
            slice(-logits_to_keep, None)
            if isinstance(logits_to_keep, int)
            else logits_to_keep
        )
        logits = self.lm_head(hidden_states[:, slice_indices, :])
        loss = None
        # LOSS = cœur de l'entraînement : prédire le token SUIVANT (décalage d'une position).
        #   phrase [Le, chat, dort] : sortie@"Le" doit prédire "chat", sortie@"chat" → "dort".
        #   cross_entropy mesure l'erreur. ignore_index=-100 : positions non comptées
        #   (ex. en SFT, la question de l'utilisateur et le padding sont mis à -100).
        if labels is not None:
            x, y = logits[..., :-1, :].contiguous(), labels[..., 1:].contiguous()
            loss = F.cross_entropy(
                x.view(-1, x.size(-1)), y.view(-1), ignore_index=-100
            )
        return MoeCausalLMOutputWithPast(
            loss=loss,
            aux_loss=aux_loss,
            logits=logits,
            past_key_values=past_key_values,
            hidden_states=hidden_states,
        )

    # ---------------------------------------------------------------------------------
    # generate : le modèle ne produit que 6400 scores ; on CHOISIT un token, on l'ajoute,
    # on recommence. Boucle :
    #   1. forward → logits du dernier token (grâce au KV-cache, seuls les nouveaux tokens
    #      passent dans le modèle : input_ids[:, past_len:])
    #   2. modifier la distribution :
    #      - temperature : logits/T. T<1 = prudent/répétitif, T>1 = créatif puis incohérent,
    #        T→0 ≈ greedy (do_sample=False → argmax)
    #      - repetition_penalty : réduit le score des tokens déjà présents (anti-boucles)
    #      - top_k : ne garder que les k meilleurs (garde-fou contre la longue traîne)
    #      - top_p (nucleus) : garder le plus petit ensemble de tokens dont la proba cumulée
    #        atteint p. Adaptatif : 1 candidat si le modèle est sûr, beaucoup s'il hésite.
    #   3. tirer un token au hasard selon les probas (multinomial)
    #   4. l'ajouter, garder le KV-cache
    #   5. stop quand toutes les séquences ont produit EOS (fin de réponse)
    # ---------------------------------------------------------------------------------
    # https://github.com/jingyaogong/minimind/discussions/611
    @torch.inference_mode()
    def generate(
        self,
        inputs=None,
        attention_mask=None,
        max_new_tokens=8192,
        temperature=0.85,
        top_p=0.85,
        top_k=50,
        eos_token_id=2,
        streamer=None,
        use_cache=True,
        num_return_sequences=1,
        do_sample=True,
        repetition_penalty=1.0,
        **kwargs,
    ):
        input_ids = kwargs.pop("input_ids", inputs).repeat(num_return_sequences, 1)
        attention_mask = (
            attention_mask.repeat(num_return_sequences, 1)
            if attention_mask is not None
            else None
        )
        past_key_values = kwargs.pop("past_key_values", None)
        finished = torch.zeros(
            input_ids.shape[0], dtype=torch.bool, device=input_ids.device
        )
        if streamer:
            streamer.put(input_ids.cpu())
        for _ in range(max_new_tokens):
            past_len = past_key_values[0][0].shape[1] if past_key_values else 0
            outputs = self.forward(
                input_ids[:, past_len:],
                attention_mask,
                past_key_values,
                use_cache=use_cache,
                **kwargs,
            )
            attention_mask = (
                torch.cat(
                    [
                        attention_mask,
                        attention_mask.new_ones(attention_mask.shape[0], 1),
                    ],
                    -1,
                )
                if attention_mask is not None
                else None
            )
            logits = outputs.logits[:, -1, :] / temperature  # température
            if repetition_penalty != 1.0:
                for i in range(input_ids.shape[0]):
                    seen = torch.unique(input_ids[i])
                    score = logits[i, seen]
                    logits[i, seen] = torch.where(
                        score > 0,
                        score / repetition_penalty,
                        score * repetition_penalty,
                    )
            if top_k > 0:  # top-k
                logits[logits < torch.topk(logits, top_k)[0][..., -1, None]] = -float(
                    "inf"
                )
            if top_p < 1.0:  # top-p (nucleus)
                sorted_logits, sorted_indices = torch.sort(logits, descending=True)
                mask = (
                    torch.cumsum(torch.softmax(sorted_logits, dim=-1), dim=-1) > top_p
                )
                mask[..., 1:], mask[..., 0] = mask[..., :-1].clone(), 0
                logits[mask.scatter(1, sorted_indices, mask)] = -float("inf")
            # tirage aléatoire selon les probas (sampling) ou argmax (greedy)
            next_token = (
                torch.multinomial(torch.softmax(logits, dim=-1), num_samples=1)
                if do_sample
                else torch.argmax(logits, dim=-1, keepdim=True)
            )
            if eos_token_id is not None:
                next_token = torch.where(
                    finished.unsqueeze(-1),
                    next_token.new_full((next_token.shape[0], 1), eos_token_id),
                    next_token,
                )
            input_ids = torch.cat([input_ids, next_token], dim=-1)
            past_key_values = outputs.past_key_values if use_cache else None
            if streamer:
                streamer.put(next_token.cpu())
            if eos_token_id is not None:
                finished |= next_token.squeeze(-1).eq(eos_token_id)
                if finished.all():
                    break
        if streamer:
            streamer.end()
        if kwargs.get("return_kv"):
            return {"generated_ids": input_ids, "past_kv": past_key_values}
        return input_ids
