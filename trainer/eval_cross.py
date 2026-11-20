"""
Évaluation croisée : plusieurs modèles sur les MÊMES jeux de test (corpus/testset.py).

Deux mesures par jeu, plus bas = mieux :
  - loss par token, comparable entre modèles qui partagent le même tokenizer ;
  - bits par octet, comparable entre modèles aux tokenizers différents (dont les modèles extérieurs).
Usage : cd trainer && python eval_cross.py nom1=chemin1.pth nom2=chemin2.pth --tokenizer_path ...
"""
import argparse
import json
import math
import os
import sys

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

import torch
import torch.nn.functional as F

TEST_DIR = os.path.join(os.environ.get("MINIMIND_DATA_DIR", os.path.join(os.path.dirname(__file__), "..", "dataset")), "fr_en", "test")
SEQ = 340


@torch.no_grad()
def score(model, tokenizer, texts, batch_size=64):
    """Retourne (loss par token, bits par octet) sur une liste de documents, lus par blocs de SEQ tokens."""
    bos, eos = tokenizer.bos_token_id, tokenizer.eos_token_id
    ids = [t for doc in tokenizer(texts, add_special_tokens=False).input_ids for t in [bos] + doc + [eos]]
    blocks = torch.tensor(ids[: len(ids) // SEQ * SEQ]).view(-1, SEQ)
    n_bytes = sum(len(t.encode("utf-8")) for t in texts) * (blocks.numel() / len(ids))  # part réellement évaluée
    nll, count = 0.0, 0
    for b in blocks.split(batch_size):
        b = b.cuda()
        with torch.autocast("cuda", dtype=torch.bfloat16):
            logits = model(b).logits[:, :-1].float()
        nll += F.cross_entropy(logits.reshape(-1, logits.size(-1)), b[:, 1:].reshape(-1), reduction="sum").item()
        count += b[:, 1:].numel()
    return nll / count, nll / math.log(2) / n_bytes


if __name__ == "__main__":
    from transformers import AutoTokenizer

    from model.model_minimind import MiniMindConfig, MiniMindForCausalLM

    parser = argparse.ArgumentParser()
    parser.add_argument("models", nargs="+", help="nom=chemin_des_poids")
    parser.add_argument("--tokenizer_path", default="../dataset/fr_en/tokenizer_fr65_16k")
    args = parser.parse_args()

    tok = AutoTokenizer.from_pretrained(args.tokenizer_path)
    tests = {f[:-6]: [json.loads(line)["text"] for line in open(os.path.join(TEST_DIR, f), encoding="utf-8")] for f in sorted(os.listdir(TEST_DIR))}
    results = {}
    for spec in args.models:
        name, path = spec.split("=", 1)
        model = MiniMindForCausalLM(MiniMindConfig(vocab_size=len(tok)))
        model.load_state_dict({k: v.float() for k, v in torch.load(path, map_location="cpu").items()})
        model = model.cuda().eval()
        results[name] = {t: score(model, tok, texts) for t, texts in tests.items()}
        del model
    names = list(results)
    print(f"{'jeu de test':22}" + "".join(f"{n + ' (loss)':>20}{n + ' (bits/octet)':>24}" for n in names))
    for t in tests:
        print(f"{t:22}" + "".join(f"{results[n][t][0]:20.3f}{results[n][t][1]:24.3f}" for n in names))
