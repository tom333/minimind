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


def _predict(model, x, temperature):
    logits = model(x).logits.float()
    logits[..., MASK_ID] = -float("inf")
    probs = logits.softmax(-1)
    pred = torch.multinomial((probs[0] ** (1 / temperature)).clamp_min(1e-12), 1).T if temperature > 0 else probs.argmax(-1)
    return pred, probs.gather(-1, pred[..., None])[..., 0]


@torch.no_grad()
def mdm_generate(model, prompt, length, steps=32, temperature=0.0, block=0, refine=0, refine_frac=0.2):
    """Démasquage progressif : à chaque tour, on fixe les prédictions les plus sûres parmi les masques.

    block > 0 : diffusion par blocs (LLaDA « semi-autoregressive », BD3-LM) — les blocs de `block` tokens sont
      produits de gauche à droite, chacun par démasquage ; évite que le modèle remplisse d'abord la fin.
    refine > 0 : tours de RAFFINAGE après le brouillon complet, à la façon d'un modèle d'images — on re-masque
      la part `refine_frac` des tokens les moins sûrs et on les prédit à nouveau, avec tout le contexte.
    """
    p = prompt.size(1)
    x = torch.cat([prompt, torch.full((1, length), MASK_ID, device=prompt.device)], 1)
    history = []
    spans = [(s, min(s + block, p + length)) for s in range(p, p + length, block)] if block else [(p, p + length)]
    per_span = max(1, steps // len(spans))
    for start, end in spans:
        for step in range(per_span):
            masked = x == MASK_ID
            masked[:, :start] = False
            masked[:, end:] = False
            remaining = int(masked.sum())
            if remaining == 0:
                break
            pred, conf = _predict(model, x, temperature)
            conf = conf.masked_fill(~masked, -1.0)
            keep = conf[0].topk(math.ceil(remaining / (per_span - step))).indices
            x[0, keep] = pred[0, keep]
            history.append(("brouillon", x.clone()))
    for _ in range(refine):
        pred, conf = _predict(model, x, temperature)
        conf[:, :p] = float("inf")  # le début imposé n'est jamais remis en cause
        worst = conf[0].topk(max(1, int(length * refine_frac)), largest=False).indices
        x[0, worst] = MASK_ID
        history.append(("re-masqué", x.clone()))
        pred, _ = _predict(model, x, temperature)
        x[0, worst] = pred[0, worst]
        history.append(("raffiné", x.clone()))
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
    parser.add_argument("--block", type=int, default=0, help="taille des blocs (0 = toute la réponse d'un coup)")
    parser.add_argument("--refine", type=int, default=0, help="tours de raffinage après le brouillon")
    parser.add_argument("--temperature", type=float, default=0.0)
    args = parser.parse_args()
    tok = AutoTokenizer.from_pretrained(args.tokenizer_path)
    model = MiniMindForCausalLM(MiniMindConfig(vocab_size=len(tok)))
    model.load_state_dict({k: v.float() for k, v in torch.load(args.checkpoint, map_location="cpu").items()})
    model = model.to(args.device).eval()
    set_causal(model, False)
    prompt = torch.tensor([[tok.bos_token_id] + tok(args.prompt, add_special_tokens=False).input_ids], device=args.device)
    torch.manual_seed(0)
    out, history = mdm_generate(model, prompt, args.length, args.steps, args.temperature, args.block, args.refine)
    drafts = [i for i, (kind, _) in enumerate(history) if kind == "brouillon"]
    shown = sorted({drafts[0], drafts[len(drafts) // 3], drafts[2 * len(drafts) // 3], drafts[-1]} | set(range(drafts[-1] + 1, len(history))))
    for i in shown:
        kind, x = history[i]
        text = tok.decode(x[0, 1:]).replace("<|buffer1|>", "▒").replace("\n", " ⏎ ")
        print(f"{kind:9} tour {i + 1:2d} : {text}")
