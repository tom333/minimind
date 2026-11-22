"""Outils communs des tests : un tout petit corpus synthétique et un lanceur de l'entraînement sur processeur."""

import json
import os
import random
import re
import subprocess
import sys

import pytest

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
sys.path.insert(0, ROOT)

WORDS = "le la un une chat chien maison jardin mange dort court bleu rouge grand petit the a cat dog house runs eats big small".split()


@pytest.fixture(scope="session")
def corpus(tmp_path_factory):
    """300 documents de mots tirés au hasard, avec un champ `lang`, au format du corpus de pré-entraînement."""
    rng, path = random.Random(0), tmp_path_factory.mktemp("data") / "mini.jsonl"
    with open(path, "w", encoding="utf-8") as f:
        for i in range(300):
            text = " ".join(rng.choice(WORDS) for _ in range(rng.randint(40, 120)))
            f.write(json.dumps({"text": text, "lang": "fr" if i % 3 else "en"}) + "\n")
    return str(path)


def train(corpus, out, *extra, steps=24):
    """Lance trainer/train_pretrain.py sur un modèle minuscule ; retourne (sortie, {pas: loss}, val_loss finale)."""
    cmd = [sys.executable, "train_pretrain.py", "--device", "cpu", "--dtype", "float32", "--data_path", corpus,
           "--tokenizer_path", "../model", "--packing", "1", "--max_seq_len", "64", "--max_steps", str(steps),
           "--eval_interval", str(steps), "--val_docs", "20", "--batch_size", "4", "--hidden_size", "64",
           "--num_hidden_layers", "2", "--use_compile", "0", "--optimizer", "muon", "--accumulation_steps", "1",
           "--log_interval", "1", "--num_workers", "0", "--save_interval", "8", "--save_dir", str(out),
           "--resume_dir", str(out), "--save_weight", "t", *extra]
    env = dict(os.environ, HF_DATASETS_CACHE=str(out) + "_hf", PYTHONPATH=ROOT)
    run = subprocess.run(cmd, cwd=os.path.join(ROOT, "trainer"), env=env, capture_output=True, text=True, timeout=600)
    assert run.returncode == 0, run.stdout[-2000:] + run.stderr[-2000:]
    losses = {int(s): float(v) for s, v in re.findall(r"\((\d+)/\d+\), loss: ([\d.]+)", run.stdout)}
    val = re.findall(r"step \d+: val_loss ([\d.]+)", run.stdout)
    return run.stdout, losses, float(val[-1]) if val else None
