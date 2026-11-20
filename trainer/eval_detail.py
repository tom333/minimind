"""
Évaluation détaillée : va au-delà de la val_loss moyenne, qui est dominée par les tokens fréquents.
  - loss par SOURCE du corpus (position du document dans le fichier, cf. scripts/analyze_data.py)
  - loss par FRÉQUENCE du token cible (fréquents / moyens / rares)
  - précision top-1
  - avec un modèle de référence : divergence KL et part des prédictions qui changent ("flips")
Utilisable en module (detailed_eval, appelé en fin d'entraînement) ou en ligne de commande :
  cd trainer && python eval_detail.py ../out/full/pretrain_768.pth [--ref autre.pth]
"""

import argparse
import os
import random
import sys

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

import numpy as np
import torch
import torch.nn.functional as F

DATA = os.path.join(
    os.environ.get("MINIMIND_DATA_DIR", os.path.join(os.path.dirname(__file__), "..", "dataset")),
    "pretrain_t2t_mini.jsonl",
)
N_DOCS, VAL_DOCS, SEQ, BOS = 1_270_238, 2000, 340, 1
# Sources repérées par scripts/analyze_data.py : (dernier index de document exclu, nom)
SOURCES = [
    (550_000, "zh_instructions_courtes"),
    (650_000, "multilingue_divers"),
    (1_050_000, "zh_reponses_longues"),
    (1_150_000, "zh_textes_longs"),
    (1_200_000, "maths_anglais"),
    (N_DOCS, "anglais_instructions"),
]


def _val_tokens_and_sources(data=DATA, doc_groups=None):
    """Blocs de validation + n° de groupe de chaque token. `doc_groups` : nom du groupe (langue, source…)
    de chaque document de validation, dans l'ordre du fichier ; None = sources du corpus chinois d'origine."""
    val = np.load(f"{data}.val_rand{VAL_DOCS}.npy")
    val = torch.from_numpy(val[: len(val) // SEQ * SEQ].astype(np.int64))
    if doc_groups is None:
        # même tirage que train_pretrain.py : retrouve l'index d'origine de chaque document de validation
        doc_idx = sorted(random.Random(0).sample(range(N_DOCS), VAL_DOCS))
        names = [name for _, name in SOURCES]
        doc_source = torch.bucketize(torch.tensor(doc_idx), torch.tensor([b for b, _ in SOURCES]), right=True)
    else:
        names = sorted(set(doc_groups))
        doc_source = torch.tensor([names.index(g) for g in doc_groups])
    token_doc = torch.cumsum(val == BOS, 0) - 1  # n° du document de chaque token
    return val.view(-1, SEQ), doc_source[token_doc].view(-1, SEQ), names


def _freq_buckets(data=DATA, vocab_size=6400):
    """Classe chaque token du vocabulaire : 0 = fréquent (50 % des occurrences), 1 = moyen (40 %), 2 = rare (10 %)."""
    train = np.load(f"{data}.train_rand{VAL_DOCS}.npy", mmap_mode="r")
    step = max(1, len(train) // 50_000_000)  # échantillon régulier : le corpus peut être rangé par source
    counts = torch.from_numpy(np.bincount(train[::step], minlength=vocab_size)).float()
    order = counts.argsort(descending=True)
    mass = counts[order].cumsum(0) / counts.sum()
    bucket = torch.empty(vocab_size, dtype=torch.long)
    bucket[order] = (mass > 0.5).long() + (mass > 0.9).long()
    return bucket


@torch.no_grad()
def detailed_eval(model, device="cuda", ref_model=None, batch_size=32, data=DATA, doc_groups=None):
    """Retourne un dict de métriques {nom: valeur}. `model(x).logits` doit prédire le token suivant."""
    blocks, sources, names = _val_tokens_and_sources(data, doc_groups)
    bucket = _freq_buckets(data, model.config.vocab_size)
    was_training = model.training
    model.eval()
    losses, correct, kls, flips = [], [], [], []
    for b in blocks.split(batch_size):
        b = b.to(device)
        with torch.autocast(
            "cuda", dtype=torch.bfloat16, enabled="cuda" in str(device)
        ):
            logits = model(b).logits[:, :-1].float()
            ref_logits = (
                ref_model(b).logits[:, :-1].float() if ref_model is not None else None
            )
        y = b[:, 1:]
        losses.append(
            F.cross_entropy(
                logits.reshape(-1, logits.size(-1)), y.reshape(-1), reduction="none"
            )
            .view_as(y)
            .cpu()
        )
        correct.append((logits.argmax(-1) == y).cpu())
        if ref_logits is not None:
            logp, ref_logp = logits.log_softmax(-1), ref_logits.log_softmax(-1)
            kls.append((ref_logp.exp() * (ref_logp - logp)).sum(-1).cpu())
            flips.append((logits.argmax(-1) != ref_logits.argmax(-1)).cpu())
    model.train(was_training)
    loss, correct = torch.cat(losses), torch.cat(correct).float()
    target, src = blocks[:, 1:], sources[:, 1:]
    out = {"val_loss": loss.mean().item(), "top1": correct.mean().item()}
    for i, name in enumerate(names):
        out[f"loss_src_{name}"] = loss[src == i].mean().item()
    for i, name in enumerate(["frequents", "moyens", "rares"]):
        out[f"loss_tok_{name}"] = loss[bucket[target] == i].mean().item()
    if kls:
        out["kl_vs_ref"] = torch.cat(kls).mean().item()
        out["flips_vs_ref"] = torch.cat(flips).float().mean().item()
    return out


def format_metrics(m):
    lines = [f"val_loss {m['val_loss']:.4f}   précision top-1 {m['top1']:.1%}"]
    lines.append(
        "par source   : "
        + "  ".join(
            f"{k[9:]} {v:.3f}" for k, v in m.items() if k.startswith("loss_src_")
        )
    )
    lines.append(
        "par fréquence: "
        + "  ".join(
            f"{k[9:]} {v:.3f}" for k, v in m.items() if k.startswith("loss_tok_")
        )
    )
    if "kl_vs_ref" in m:
        lines.append(
            f"vs référence : KL {m['kl_vs_ref']:.4f}   prédictions changées {m['flips_vs_ref']:.1%}"
        )
    return "\n".join(lines)


if __name__ == "__main__":
    from model.model_minimind import MiniMindConfig, MiniMindForCausalLM

    parser = argparse.ArgumentParser()
    parser.add_argument("checkpoint")
    parser.add_argument(
        "--ref",
        default=None,
        help="checkpoint de référence (même architecture) pour KL / flips",
    )
    parser.add_argument("--hidden_size", type=int, default=768)
    parser.add_argument("--num_hidden_layers", type=int, default=8)
    args = parser.parse_args()

    def load(path):
        m = MiniMindForCausalLM(
            MiniMindConfig(
                hidden_size=args.hidden_size, num_hidden_layers=args.num_hidden_layers
            )
        )
        m.load_state_dict(
            {k: v.float() for k, v in torch.load(path, map_location="cpu").items()}
        )
        return m.cuda().eval()

    print(
        format_metrics(
            detailed_eval(
                load(args.checkpoint), ref_model=load(args.ref) if args.ref else None
            )
        )
    )
