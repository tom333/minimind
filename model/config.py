"""Hyperparamètres du modèle (dimensions, têtes, RoPE, MoE)."""
import math

from transformers import PretrainedConfig


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
        # Logit softcap (Gemma 2, speedrun) : logits = cap·tanh(logits/cap), borne les scores de sortie.
        # 0 = désactivé (comportement d'origine)
        self.logit_softcap = kwargs.get("logit_softcap", 0.0)
        # FFN à matrices structurées BTT (cf. feedforward.BTTLinear). rang 0 = matrices pleines (origine).
        # hidden_size = h1*h2 et largeur interne du FFN = m1*m2.
        self.ffn_btt_rank = kwargs.get("ffn_btt_rank", 0)
        self.ffn_btt_hidden = kwargs.get("ffn_btt_hidden", (32, 24))
        self.ffn_btt_mid = kwargs.get("ffn_btt_mid", (128, 64))
        # Couche mémoire à clés produit (cf. memory.MemoryLayer) ajoutée dans les blocs listés.
        # () = aucune (origine). Table de memory_keys² lignes × hidden_size.
        self.memory_layers = tuple(kwargs.get("memory_layers", ()))
        self.memory_keys = kwargs.get("memory_keys", 256)
        self.memory_k = kwargs.get("memory_k", 8)
        self.memory_heads = kwargs.get("memory_heads", 2)
        self.memory_query_dim = kwargs.get("memory_query_dim", 256)
        # Poids ternaires (cf. ternary.TernaryLinear) pour les matrices des blocs :
        # "" = désactivé (origine), "matrix" = une échelle par matrice, "row" = une échelle par neurone.
        self.ternary = kwargs.get("ternary", "")  # "" | matrix | row | group (groupes de 128)
        self.ternary_hadamard = kwargs.get("ternary_hadamard", False)  # rotation de l'entrée (Bonsai 2)
        # FFN sans porte (2 matrices au lieu de 3) : down(act(up(x))), avec hidden_act="relu2" (Primer,
        # BitNet b1.58 2B4T, speedrun nanoGPT). ffn_glu=True = SwiGLU d'origine.
        self.ffn_glu = kwargs.get("ffn_glu", True)
        # SubLN (BitNet) : RMSNorm supplémentaire sur l'entrée des projections de sortie (o_proj, down_proj),
        # pour stabiliser l'échelle des activations quand les poids sont ternaires.
        self.subln = kwargs.get("subln", False)
        # Un seul FFN partagé par tous les blocs (Apple, « One Wide FFN Is All You Need », 2023) :
        # même calcul par token, mais les paramètres du FFN ne sont stockés qu'une fois.
        self.ffn_shared = kwargs.get("ffn_shared", False)
        # Mixture-of-Depths (Raposo et al. 2024) : dans un bloc sur deux, seule une fraction mod_capacity
        # des tokens (choisis par un routeur) traverse le bloc ; les autres le contournent. 0 = désactivé.
        self.mod_capacity = kwargs.get("mod_capacity", 0.0)
        self.mod_aux_coef = kwargs.get("mod_aux_coef", 0.01)
        self.mod_sort = kwargs.get("mod_sort", False)  # diagnostic : ordonner les tokens retenus par un tri
