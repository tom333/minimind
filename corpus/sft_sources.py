"""
Conversations pour l'ajustement (SFT) bilingue français / anglais.

Chaque source est lue en flux, convertie au format du projet et filtrée, puis écrite dans
<données>/fr_en/sft/<nom>.jsonl , une ligne par conversation :
  {"conversations": [{"role": "user" | "assistant" | "system", "content": ...}, ...], "lang": ..., "source": ...}
`assemble()` réunit les sources en  fr_en/sft_fr_en.jsonl  (conversations seules, format attendu par
SFTDataset) et  fr_en/sft_fr_en.meta.jsonl  (langue et source de chaque ligne, dans le même ordre).
Usage : python -m corpus.sft_sources [nom]
"""
import hashlib
import json
import os
import sys

from .pretrain_sources import DATA_DIR, HF

SFT_DIR = os.path.join(DATA_DIR, "fr_en", "sft")
MAX_CHARS = 3200  # ~768 tokens : une conversation plus longue serait coupée en pleine réponse
ROLES = {"user": "user", "human": "user", "assistant": "assistant", "gpt": "assistant", "system": "system"}

# Détection de langue minimale pour le jeu multilingue (allemand, espagnol, français, italien, portugais)
STOPWORDS = {
    "fr": (" le ", " les ", " des ", " est ", " une ", " pour ", " dans ", " vous "),
    "de": (" der ", " die ", " und ", " ist ", " das ", " nicht ", " sie ", " mit "),
    "es": (" el ", " los ", " una ", " para ", " es ", " por ", " las ", " con "),
    "it": (" il ", " che ", " per ", " una ", " della ", " sono ", " non ", " gli "),
    "pt": (" o ", " os ", " uma ", " para ", " não ", " com ", " você ", " são "),
}


def is_french(text):
    t = " " + text.lower() + " "
    scores = {lang: sum(t.count(w) for w in words) for lang, words in STOPWORDS.items()}
    best = max(scores, key=scores.get)
    return best == "fr" and scores["fr"] >= 3 and scores["fr"] > 1.5 * max(v for k, v in scores.items() if k != "fr")


def from_messages(key, content_key="content"):
    return lambda r: [(m["role"], m[content_key]) for m in r[key]]


def oasst_threads(lang):
    """OpenAssistant est un arbre de messages : on remonte de chaque réponse retenue jusqu'à la racine."""

    def expand(stream):
        msgs = {m["message_id"]: m for m in stream}
        for m in msgs.values():
            # réponse validée par les relecteurs et classée première parmi ses concurrentes
            if m["role"] != "assistant" or m["lang"] != lang or m["deleted"] or not m["review_result"] or m["rank"]:
                continue
            chain = []
            while m is not None:
                chain.append(("user" if m["role"] == "prompter" else "assistant", m["text"]))
                m = msgs.get(m["parent_id"])
            yield {"turns": chain[::-1]}

    return expand


SOURCES = {
    # écrits par des humains (les autres sources sont produites par des modèles) : peu nombreux mais précieux
    "oasst_fr": dict(
        lang="fr", limit=10_000, hub="OpenAssistant/oasst2", licence="Apache 2.0",
        expand=oasst_threads("fr"), convert=lambda r: r["turns"],
    ),
    "oasst_en": dict(
        lang="en", limit=40_000, hub="OpenAssistant/oasst2", licence="Apache 2.0",
        expand=oasst_threads("en"), convert=lambda r: r["turns"],
    ),
    "dolly_en": dict(
        lang="en", limit=15_000, hub="databricks/databricks-dolly-15k", licence="CC BY-SA 3.0",
        convert=lambda r: [("user", (r["instruction"] + "\n" + r["context"]).strip()), ("assistant", r["response"])],
    ),
    "smoltalk_en": dict(
        lang="en", limit=250_000, files=HF + "HuggingFaceTB/smoltalk/data/all/train-*.parquet", licence="Apache 2.0",
        convert=from_messages("messages"),
        # écartés : appels de fonctions (hors sujet ici) et documents longs (dépassent le contexte)
        keep=lambda r: r["source"] not in ("apigen-80k", "longalign"),
    ),
    "smoltalk_fr": dict(
        lang="fr", limit=250_000, licence="Apache 2.0",
        files=HF + "HuggingFaceTB/smoltalk2/SFT/smoltalk_multilingual_8languages_lang_5_no_think-*.parquet",
        convert=from_messages("messages"),
        keep=lambda r: is_french(" ".join(m["content"] for m in r["messages"])),
    ),
    "openhermes_fr": dict(
        lang="fr", limit=200_000, files=HF + "legmlai/openhermes-fr/data/train-*.parquet", licence="ODC-By",
        convert=lambda r: [("user", r["prompt"]), ("assistant", r["accepted_completion"])],
        keep=lambda r: not r["bad_entry"],
    ),
    "french_instruct": dict(
        lang="fr", limit=200_000, files=HF + "angeluriot/french_instruct/dataset.parquet", licence="MIT",
        convert=lambda r: ([("system", r["context"])] if r["context"] else []) + [(m["role"], m["text"]) for m in r["conversation"]],
    ),
    "french_alpaca": dict(
        lang="fr", limit=110_000, kind="json", licence="Apache 2.0",
        files=HF + "jpacifico/French-Alpaca-dataset-Instruct-110K/Acquiesce_data_110k_instructions.json",
        convert=lambda r: [("user", (r["instruction"] + "\n" + r["input"]).strip()), ("assistant", r["output"])],
    ),
}


def normalize(turns):
    """Valide une conversation : rôles connus, alternance utilisateur / assistant, fin sur l'assistant."""
    conv = [{"role": ROLES.get(role), "content": (content or "").strip()} for role, content in turns]
    if any(m["role"] is None or not m["content"] for m in conv):
        return None
    body = conv[1:] if conv[0]["role"] == "system" else conv
    if len(body) < 2 or len(body) % 2 or any(m["role"] != ("user", "assistant")[i % 2] for i, m in enumerate(body)):
        return None
    if sum(len(m["content"]) for m in conv) > MAX_CHARS:
        return None
    return conv


def build(name, limit=None):
    from datasets import load_dataset

    src = SOURCES[name]
    limit = limit or src["limit"]
    keep = src.get("keep", lambda r: True)
    os.makedirs(SFT_DIR, exist_ok=True)
    path = os.path.join(SFT_DIR, f"{name}.jsonl")
    seen, stats = set(), dict(lus=0, gardes=0, filtres=0, invalides=0, doublons=0, caracteres=0)
    if "hub" in src:
        stream = load_dataset(src["hub"], split="train", streaming=True)
    else:
        stream = load_dataset(src.get("kind", "parquet"), data_files=src["files"], split="train", streaming=True)
    stream = src["expand"](stream) if "expand" in src else stream
    with open(path + ".tmp", "w", encoding="utf-8") as f:
        for row in stream:
            stats["lus"] += 1
            if not keep(row):
                stats["filtres"] += 1
                continue
            conv = normalize(src["convert"](row))
            if conv is None:
                stats["invalides"] += 1
                continue
            digest = hashlib.md5(json.dumps(conv, ensure_ascii=False).encode("utf-8")).digest()
            if digest in seen:
                stats["doublons"] += 1
                continue
            seen.add(digest)
            f.write(json.dumps({"conversations": conv, "lang": src["lang"], "source": name}, ensure_ascii=False) + "\n")
            stats["gardes"] += 1
            stats["caracteres"] += sum(len(m["content"]) for m in conv)
            if stats["gardes"] >= limit:
                break
    os.replace(path + ".tmp", path)
    return {**stats, "chemin": path, "licence": src["licence"]}


def assemble():
    """Réunit les sources présentes en un fichier d'entraînement et son fichier de métadonnées."""
    out = os.path.join(DATA_DIR, "fr_en", "sft_fr_en.jsonl")
    stats = {}
    with open(out, "w", encoding="utf-8") as f, open(out[:-6] + ".meta.jsonl", "w", encoding="utf-8") as meta:
        for name in SOURCES:
            path = os.path.join(SFT_DIR, f"{name}.jsonl")
            if not os.path.exists(path):
                continue
            for line in open(path, encoding="utf-8"):
                d = json.loads(line)
                f.write(json.dumps({"conversations": d["conversations"]}, ensure_ascii=False) + "\n")
                meta.write(json.dumps({"lang": d["lang"], "source": d["source"]}) + "\n")
                s = stats.setdefault(d["lang"], dict(conversations=0, caracteres=0))
                s["conversations"] += 1
                s["caracteres"] += sum(len(m["content"]) for m in d["conversations"])
    return {"chemin": out, **stats}


if __name__ == "__main__":
    for name in sys.argv[1:] or list(SOURCES):
        print(name, build(name), flush=True)
    print("assemblage", assemble(), flush=True)
