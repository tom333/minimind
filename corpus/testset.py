"""
Jeux de test communs, tenus à l'écart de tous les corpus d'entraînement.

Ils servent à comparer des modèles entraînés sur des corpus différents (dont la loss de validation
n'est pas comparable) et, plus tard, des modèles extérieurs. Chaque jeu vient d'un fichier que la
construction des corpus ne lit jamais : le DERNIER fichier de chaque source (les corpus lisent les
premiers), un découpage de test officiel, ou une source absente des corpus (Wikisource).
Sortie : <données>/fr_en/test/<nom>.jsonl , lignes {"text": ...}.
Usage : python -m corpus.testset
"""
import json
import os

from .pretrain_sources import DATA_DIR, HF, MAX_CHARS, MIN_CHARS, clean

TEST_DIR = os.path.join(DATA_DIR, "fr_en", "test")
N_DOCS = 1000

# nom → (motif de fichiers, lequel prendre, filtre, colonnes à lire)
TESTS = {
    "wikipedia_fr": ("wikimedia/wikipedia/20231101.fr/*.parquet", -1, None, None),
    "web_fr_courant": ("HuggingFaceFW/fineweb-2/data/fra_Latn/test/*.parquet", 0, None, None),
    "web_fr_selectionne": ("epfml/FineWeb2-HQ/fra_Latn/*.parquet", -1, lambda r: r["quality_score"] >= 0.5, ["text", "quality_score"]),
    "web_en_educatif": ("HuggingFaceFW/fineweb-edu/sample/10BT/*.parquet", -1, lambda r: r["score"] >= 3.0, None),
    "wikisource_fr": ("wikimedia/wikisource/20231201.fr/*.parquet", 0, None, None),
}


def build(name):
    from datasets import load_dataset
    from huggingface_hub import HfFileSystem

    pattern, which, keep, columns = TESTS[name]
    files = sorted(HfFileSystem().glob("datasets/" + pattern))
    extra = {"columns": columns} if columns else {}
    stream = load_dataset("parquet", data_files="hf://" + files[which], split="train", streaming=True, **extra)
    os.makedirs(TEST_DIR, exist_ok=True)
    docs = []
    for i, row in enumerate(stream):
        if keep is not None and not keep(row):
            continue
        text = clean(row["text"], name)
        if len(text) >= MIN_CHARS:
            docs.append(text[: MAX_CHARS // 4])  # documents plafonnés à 5 000 caractères : aucun ne domine
        if len(docs) >= N_DOCS:
            break
    with open(os.path.join(TEST_DIR, f"{name}.jsonl"), "w", encoding="utf-8") as f:
        for text in docs:
            f.write(json.dumps({"text": text}, ensure_ascii=False) + "\n")
    return {"documents": len(docs), "fichier_source": files[which], "octets": sum(len(t.encode("utf-8")) for t in docs)}


if __name__ == "__main__":
    for name in TESTS:
        if not os.path.exists(os.path.join(TEST_DIR, f"{name}.jsonl")):
            print(name, build(name), flush=True)
