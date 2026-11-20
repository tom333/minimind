"""
Mélange des sources brutes en un corpus de pretrain au ratio français / anglais voulu, puis
apprentissage du tokenizer commun.

  mix(fr_share)        → <données>/fr_en/pretrain_fr65.jsonl   (lignes {"text", "lang", "source"})
  train_tokenizer(...) → <données>/fr_en/tokenizer_fr65_16k/   (tokenizer.json + configuration)
L'ordre des documents dans le fichier n'a pas d'importance : l'entraînement découpe le corpus en
blocs et les tire au hasard. Le sous-échantillonnage est déterministe (basé sur le contenu).
"""
import hashlib
import json
import os
import sys

from .pretrain_sources import DATA_DIR, RAW_DIR, SOURCES

OUT_DIR = os.path.join(DATA_DIR, "fr_en")


def corpus_path(fr_share):
    return os.path.join(OUT_DIR, f"pretrain_fr{round(fr_share * 100)}.jsonl")


def mix(fr_share=0.65):
    """Garde le plus de texte possible en respectant fr_share (part du français, en caractères)."""
    size = {"fr": 0, "en": 0}
    for name, src in SOURCES.items():
        size[src["lang"]] += os.path.getsize(os.path.join(RAW_DIR, f"{name}.jsonl"))
    total = min(size["fr"] / fr_share, size["en"] / (1 - fr_share))
    keep = {"fr": fr_share * total / size["fr"], "en": (1 - fr_share) * total / size["en"]}  # part gardée par langue
    stats = {name: dict(documents=0, caracteres=0) for name in SOURCES}
    with open(corpus_path(fr_share) + ".tmp", "w", encoding="utf-8") as out:
        for name, src in SOURCES.items():
            threshold = int(keep[src["lang"]] * 2**32)
            with open(os.path.join(RAW_DIR, f"{name}.jsonl"), encoding="utf-8") as f:
                for line in f:
                    # tirage déterministe : le même document est toujours gardé ou écarté
                    if int.from_bytes(hashlib.md5(line[:200].encode("utf-8")).digest()[:4], "big") >= threshold:
                        continue
                    out.write(line)
                    stats[name]["documents"] += 1
                    stats[name]["caracteres"] += len(line)
    os.replace(corpus_path(fr_share) + ".tmp", corpus_path(fr_share))
    chars = {lang: sum(s["caracteres"] for n, s in stats.items() if SOURCES[n]["lang"] == lang) for lang in size}
    return {
        "chemin": corpus_path(fr_share),
        "documents": sum(s["documents"] for s in stats.values()),
        "go": round(sum(chars.values()) / 1e9, 2),
        "part_francais": round(chars["fr"] / sum(chars.values()), 3),
        "par_source": {n: {"documents": s["documents"], "go": round(s["caracteres"] / 1e9, 2)} for n, s in stats.items()},
    }


def tokenizer_dir(fr_share, vocab_size):
    return os.path.join(OUT_DIR, f"tokenizer_fr{round(fr_share * 100)}_{vocab_size // 1000}k")


def train_tokenizer(fr_share=0.65, vocab_size=16000, sample_go=0.4):
    """Apprend le BPE sur un échantillon régulier du corpus mélangé (1 document sur k), puis mesure
    sa compression (caractères par token) dans chaque langue sur des documents non vus."""
    sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "trainer"))
    from train_tokenizer import train_tokenizer as train_bpe  # script d'origine du projet
    from transformers import AutoTokenizer

    src = corpus_path(fr_share)
    stride = max(1, round(os.path.getsize(src) / (sample_go * 1e9)))
    sample, held_out = os.path.join(OUT_DIR, "tokenizer_sample.jsonl"), {"fr": [], "en": []}
    with open(src, encoding="utf-8") as f, open(sample, "w", encoding="utf-8") as out:
        for i, line in enumerate(f):
            if i % stride == 0:
                out.write(line)
            elif i % stride == 1 and i % (stride * 200) == 1:  # documents témoins, hors échantillon
                d = json.loads(line)
                held_out[d["lang"]].append(d["text"])
    out_dir = tokenizer_dir(fr_share, vocab_size)
    train_bpe(sample, out_dir, vocab_size)
    os.remove(sample)
    tok = AutoTokenizer.from_pretrained(out_dir)
    stats = {"chemin": out_dir, "vocabulaire": len(tok), "documents_temoins": {k: len(v) for k, v in held_out.items()}}
    for lang, docs in held_out.items():
        n_chars = sum(len(t) for t in docs)
        n_tokens = sum(len(ids) for ids in tok(docs, add_special_tokens=False).input_ids)
        n_words = sum(len(t.split()) for t in docs)
        stats[f"caracteres_par_token_{lang}"] = round(n_chars / n_tokens, 2)
        stats[f"tokens_par_mot_{lang}"] = round(n_tokens / n_words, 2)
    return stats


if __name__ == "__main__":
    print(json.dumps(mix(), ensure_ascii=False, indent=1))
