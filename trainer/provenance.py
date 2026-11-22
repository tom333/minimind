"""
Traçabilité d'un run : avec quel code, quelle image et quelles données a-t-il tourné ?

Ces informations sont posées en étiquettes (tags) sur chaque run MLflow, pour qu'un résultat puisse être
rattaché à un commit et à un fichier de données précis, et qu'un corpus régénéré sous le même nom ne passe
pas inaperçu. Dans l'image, le commit vient de la variable MINIMIND_GIT_COMMIT posée à la construction
(scripts/publier-code-dagster.sh) ; en local, il est lu dans le dépôt git.
"""

import hashlib
import json
import os
import platform
import subprocess

import torch

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))


def git_commit():
    commit = os.environ.get("MINIMIND_GIT_COMMIT")
    if commit:
        return commit
    try:
        run = lambda *a: subprocess.run(["git", "-C", ROOT, *a], capture_output=True, text=True, timeout=10)
        head = run("rev-parse", "--short=12", "HEAD").stdout.strip()
        dirty = run("status", "--porcelain", "--untracked-files=no").stdout.strip()
        return (head + ("-dirty" if dirty else "")) or "inconnu"
    except (OSError, subprocess.SubprocessError):
        return "inconnu"


def fingerprint(path):
    """Empreinte SHA-256 (16 premiers caractères) d'un fichier, mise en cache à côté de lui tant que sa taille
    et sa date ne changent pas : un fichier de 2 Go n'est relu qu'une fois."""
    if not path or not os.path.isfile(path):
        return None
    stat, cache = os.stat(path), path + ".sha256"
    key = f"{stat.st_size}:{int(stat.st_mtime)}"
    try:
        with open(cache) as f:
            saved = json.load(f)
        if saved["key"] == key:
            return saved["sha256"]
    except (OSError, ValueError, KeyError):
        pass
    digest = hashlib.sha256()
    with open(path, "rb") as f:
        for block in iter(lambda: f.read(1 << 24), b""):
            digest.update(block)
    value = digest.hexdigest()[:16]
    try:
        with open(cache, "w") as f:
            json.dump({"key": key, "sha256": value}, f)
    except OSError:
        pass  # dossier en lecture seule (jeu de données monté)
    return value


def run_tags(data_path, tokenizer_path, token_cache=None):
    """Étiquettes de traçabilité. Les données sont identifiées par le fichier réellement lu : le cache de
    tokens s'il existe, sinon le corpus."""
    data_file = token_cache if token_cache and os.path.isfile(token_cache) else data_path
    tags = {
        "git_commit": git_commit(),
        "image": os.environ.get("MINIMIND_IMAGE", "local"),
        "data_file": os.path.basename(data_file),
        "data_sha256": fingerprint(data_file),
        "tokenizer_sha256": fingerprint(os.path.join(tokenizer_path, "tokenizer.json")),
        "torch": torch.__version__,
        "host": platform.node(),
        "gpu": torch.cuda.get_device_name(0) if torch.cuda.is_available() else "cpu",
    }
    return {k: v for k, v in tags.items() if v is not None}
