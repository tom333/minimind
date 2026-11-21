"""
Diffusion masquée pour le texte (MDLM, Sahoo et al. 2024 ; LLaDA, Nie et al. 2025).

Le « bruit » est le masquage : on tire un niveau t entre 0 et 1, on remplace chaque token par [MASQUE]
avec la probabilité t, et le modèle, qui voit toute la séquence (attention NON causale), doit retrouver
les tokens masqués. La loss pondérée par 1/t est une borne supérieure de la log-vraisemblance négative
par token : elle se compare donc, à son désavantage, à la loss d'un modèle mot à mot.
La génération part d'une séquence entièrement masquée et fixe à chaque tour les tokens les plus sûrs.

Usage (échantillons) : cd trainer && python diffusion.py poids.pth --prompt "Le chat" --length 48
"""

import math

import torch
import torch.nn.functional as F

MASK_ID = 27  # <|buffer1|> : token réservé du tokenizer, absent des corpus


def set_causal(model, causal):
    """Bascule l'attention de tous les blocs : causale (mot à mot) ou bidirectionnelle (diffusion)."""
    for m in model.modules():
        if hasattr(m, "is_causal"):
            m.is_causal = causal


def mdm_loss(model, x, generator=None, eps=1e-3):
    """Borne (nats par token) : E_t [ 1/t · somme des cross-entropies sur les positions masquées ] / L."""
    b, length = x.shape
    # niveaux de bruit stratifiés sur le lot (un par séquence) : variance plus faible qu'un tirage indépendant
    u = torch.rand(1, device=x.device, generator=generator)
    t = ((u + torch.arange(b, device=x.device)) / b).clamp(min=eps)
    mask = torch.rand(b, length, device=x.device, generator=generator) < t[:, None]
    logits = model(torch.where(mask, MASK_ID, x)).logits
    ce = F.cross_entropy(logits[mask].float(), x[mask], reduction="none")
    weight = (1.0 / t)[:, None].expand(b, length)[mask]
    return (ce * weight).sum() / (b * length)


@torch.no_grad()
def mdm_generate(model, prompt, length, steps=32, temperature=0.0):
    """Démasquage progressif : à chaque tour, on fixe les prédictions les plus sûres parmi les masques."""
    x = torch.cat([prompt, torch.full((1, length), MASK_ID, device=prompt.device)], 1)
    history = []
    for step in range(steps):
        masked = x == MASK_ID
        remaining = int(masked.sum())
        if remaining == 0:
            break
        logits = model(x).logits.float()
        logits[..., MASK_ID] = -float("inf")
        probs = logits.softmax(-1)
        pred = torch.multinomial(probs[0] ** (1 / temperature), 1).T if temperature > 0 else probs.argmax(-1)
        conf = probs.gather(-1, pred[..., None])[..., 0].masked_fill(~masked, -1.0)
        keep = conf[0].topk(math.ceil(remaining / (steps - step))).indices
        x[0, keep] = pred[0, keep]
        history.append(x.clone())
    return x, history


if __name__ == "__main__":
    import argparse
    import os
    import sys

    sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
    from transformers import AutoTokenizer

    from model.model_minimind import MiniMindConfig, MiniMindForCausalLM

    parser = argparse.ArgumentParser()
    parser.add_argument("checkpoint")
    parser.add_argument("--tokenizer_path", default="../dataset/fr_en/tokenizer_fr65_16k")
    parser.add_argument("--prompt", default="La photosynthèse est")
    parser.add_argument("--length", type=int, default=48)
    parser.add_argument("--steps", type=int, default=24)
    parser.add_argument("--device", default="cuda")
    args = parser.parse_args()
    tok = AutoTokenizer.from_pretrained(args.tokenizer_path)
    model = MiniMindForCausalLM(MiniMindConfig(vocab_size=len(tok)))
    model.load_state_dict({k: v.float() for k, v in torch.load(args.checkpoint, map_location="cpu").items()})
    model = model.to(args.device).eval()
    set_causal(model, False)
    prompt = torch.tensor([[tok.bos_token_id] + tok(args.prompt, add_special_tokens=False).input_ids], device=args.device)
    out, history = mdm_generate(model, prompt, args.length, args.steps)
    for i in sorted({0, len(history) // 3, 2 * len(history) // 3, len(history) - 1}):
        print(f"tour {i + 1:2d} : " + tok.decode(history[i][0, 1:]).replace("<|buffer1|>", "▒"))
