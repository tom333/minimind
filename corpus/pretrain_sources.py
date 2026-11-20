"""
Sources de texte brut pour le pretrain bilingue français / anglais.

Chaque source est lue en flux depuis Hugging Face (on ne télécharge que ce qu'on garde), nettoyée,
dédupliquée (doublons exacts) et écrite dans  <données>/fr_en/raw/<nom>.jsonl , une ligne par document :
  {"text": ..., "lang": "fr" | "en", "source": <nom>}
Usage direct : python -m corpus.pretrain_sources wikipedia_fr [--go 0.01]
"""
import argparse
import hashlib
import json
import os
import re

DATA_DIR = os.environ.get("MINIMIND_DATA_DIR", os.path.join(os.path.dirname(__file__), "..", "dataset"))
RAW_DIR = os.path.join(DATA_DIR, "fr_en", "raw")
HF = "hf://datasets/"

# go = volume de texte gardé (Go de caractères UTF-8). Ratio visé : 65 % français / 35 % anglais.
SOURCES = {
    "wikipedia_fr": dict(
        lang="fr", go=1.2, files=HF + "wikimedia/wikipedia/20231101.fr/*.parquet", licence="CC BY-SA 3.0",
        role="faits, style encyclopédique",
    ),
    "fineweb2_fr": dict(
        lang="fr", go=1.5, files=HF + "HuggingFaceFW/fineweb-2/data/fra_Latn/train/*.parquet", licence="ODC-By",
        role="web varié, déjà filtré et dédupliqué", keep=lambda r: r.get("language_score", 1.0) >= 0.9,
    ),
    "fineweb_edu_en": dict(
        lang="en", go=1.4, files=HF + "HuggingFaceFW/fineweb-edu/sample/10BT/*.parquet", licence="ODC-By",
        role="web filtré sur sa valeur éducative", keep=lambda r: r.get("score", 5.0) >= 3.0,
    ),
}

MIN_CHARS, MAX_CHARS = 200, 20_000  # trop court = peu d'information ; trop long = un document domine
# Wikipédia : tout ce qui suit ces titres est une liste de liens ou de références, pas de la prose
WIKI_TAIL = re.compile(r"\n(Notes et références|Références|Voir aussi|Liens externes|Bibliographie|Articles connexes|Annexes)\s*\n")
SPACES = re.compile(r"[ \t ]+")
BLANK_LINES = re.compile(r"\n{3,}")


def clean(text, source):
    if source.startswith("wikipedia"):
        m = WIKI_TAIL.search(text)
        if m:
            text = text[: m.start()]
    text = BLANK_LINES.sub("\n\n", SPACES.sub(" ", text)).strip()
    return text[:MAX_CHARS]


def build(name, go=None):
    """Écrit la source `name` et retourne ses statistiques."""
    from datasets import load_dataset

    src = SOURCES[name]
    target = int((go if go is not None else src["go"]) * 1e9)
    keep = src.get("keep", lambda r: True)
    os.makedirs(RAW_DIR, exist_ok=True)
    path = os.path.join(RAW_DIR, f"{name}.jsonl")
    seen, stats = set(), dict(lus=0, gardes=0, filtres=0, courts=0, doublons=0, caracteres=0)
    stream = load_dataset("parquet", data_files=src["files"], split="train", streaming=True)
    with open(path + ".tmp", "w", encoding="utf-8") as f:
        for row in stream:
            stats["lus"] += 1
            if not keep(row):
                stats["filtres"] += 1
                continue
            text = clean(row["text"], name)
            if len(text) < MIN_CHARS:
                stats["courts"] += 1
                continue
            digest = hashlib.md5(text.encode("utf-8")).digest()
            if digest in seen:
                stats["doublons"] += 1
                continue
            seen.add(digest)
            f.write(json.dumps({"text": text, "lang": src["lang"], "source": name}, ensure_ascii=False) + "\n")
            stats["gardes"] += 1
            stats["caracteres"] += len(text)
            if stats["caracteres"] >= target:
                break
    os.replace(path + ".tmp", path)  # le fichier n'apparaît que complet
    return {**stats, "chemin": path, "go": round(stats["caracteres"] / 1e9, 3), "licence": src["licence"]}


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("source", choices=list(SOURCES))
    parser.add_argument("--go", type=float, default=None, help="volume de texte à garder (défaut : celui de la source)")
    args = parser.parse_args()
    print(build(args.source, args.go))
