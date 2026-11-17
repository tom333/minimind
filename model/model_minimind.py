import torch
import torch.nn.functional as F
from torch import nn
from transformers import GenerationMixin, PreTrainedModel
from transformers.modeling_outputs import MoeCausalLMOutputWithPast

from .attention import Attention, repeat_kv  # noqa: F401  (ré-exportés)
from .config import MiniMindConfig
from .feedforward import FeedForward, MOEFeedForward
from .norm import RMSNorm
from .rope import apply_rotary_pos_emb, precompute_freqs_cis  # noqa: F401

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
#
# Organisation du package model/ (dans l'ordre de lecture conseillé) :
#   config.py       MiniMindConfig : toutes les dimensions
#   norm.py         RMSNorm
#   rope.py         RoPE (+ YaRN) : positions par rotation de q/k
#   attention.py    Attention : Q/K/V, GQA, masque causal, KV-cache
#   feedforward.py  FeedForward SwiGLU + MOEFeedForward
#   model_minimind.py (ce fichier) : bloc Transformer, tronc, modèle complet, loss, generate
# Les autres scripts n'importent que ce fichier (MiniMindConfig, MiniMindForCausalLM) : il ré-exporte tout.
# =====================================================================================


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
        # softcap : tanh écrase en douceur les scores extrêmes vers ±cap (aucun effet sur les petits)
        cap = getattr(self.config, "logit_softcap", 0.0)
        if cap:
            logits = cap * torch.tanh(logits / cap)
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
