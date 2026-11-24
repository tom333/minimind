"""
Évaluation croisée : plusieurs modèles sur les MÊMES jeux de test (corpus/testset.py).

Deux mesures par jeu, plus bas = mieux :
  - loss par token, comparable entre modèles qui partagent le même tokenizer ;
  - bits par octet, comparable entre modèles aux tokenizers différents (dont les modèles extérieurs).
Un modèle extérieur se désigne par  nom=hf:dépôt  (ex. smollm2=hf:HuggingFaceTB/SmolLM2-135M) : il est lu
avec son propre tokenizer. Les blocs font SEQ tokens de CHAQUE tokenizer, donc un peu plus ou moins de
texte selon le modèle ; les bits par octet restent comparables.
Un modèle hérité (trainer/heritage.py) se désigne par  nom=poids.pth@fiche.json  (forme et tokenizer dans la fiche).
Usage : cd trainer && python eval_cross.py nom1=chemin1.pth nom2=hf:dépôt --tokenizer_path ...
"""
import argparse
import json
import math
import os
import sys

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

import torch
import torch.nn.functional as F

DATA_DIR = os.environ.get("MINIMIND_DATA_DIR", os.path.join(os.path.dirname(__file__), "..", "dataset"))
TEST_DIR = os.path.join(DATA_DIR, "fr_en", "test")
SEQ = 340


@torch.no_grad()
def score(model, tokenizer, texts, batch_size=64, device="cuda"):
    """Retourne (loss par token, bits par octet) sur une liste de documents, lus par blocs de SEQ tokens."""
    bos, eos = tokenizer.bos_token_id, tokenizer.eos_token_id
    bos = eos if bos is None else bos  # certains tokenizers n'ont pas de token de début : le séparateur suffit
    ids = [t for doc in tokenizer(texts, add_special_tokens=False).input_ids for t in [bos] + doc + [eos]]
    blocks = torch.tensor(ids[: len(ids) // SEQ * SEQ]).view(-1, SEQ)
    n_bytes = sum(len(t.encode("utf-8")) for t in texts) * (blocks.numel() / len(ids))  # part réellement évaluée
    nll, count = 0.0, 0
    for b in blocks.split(batch_size):
        b = b.to(device)
        with torch.autocast(device, dtype=torch.bfloat16):
            logits = model(b).logits[:, :-1].float()
        nll += F.cross_entropy(logits.reshape(-1, logits.size(-1)), b[:, 1:].reshape(-1), reduction="sum").item()
        count += b[:, 1:].numel()
    return nll / count, nll / math.log(2) / n_bytes


def log_mlflow(experiment, name, path, scores):
    """Un run par modèle évalué : {jeu: (loss, bits par octet)} → métriques loss_<jeu> et bpb_<jeu>."""
    import mlflow

    mlflow.set_tracking_uri(os.environ.get("MINIMIND_MLFLOW_URI", "https://mlflow.tgu.ovh"))
    mlflow.set_experiment(experiment)
    with mlflow.start_run(run_name=name):
        from trainer.provenance import git_commit

        mlflow.set_tag("git_commit", git_commit())
        mlflow.log_params({"modele": path, "seq": SEQ})
        for test, (loss, bpb) in scores.items():
            mlflow.log_metrics({f"loss_{test}": loss, f"bpb_{test}": bpb})


if __name__ == "__main__":
    from transformers import AutoModelForCausalLM, AutoTokenizer

    from model.model_minimind import MiniMindConfig, MiniMindForCausalLM

    parser = argparse.ArgumentParser()
    parser.add_argument("models", nargs="+", help="nom=chemin_des_poids")
    parser.add_argument("--mlflow_experiment", default="", help="si renseigné : un run MLflow par modèle (bpb_<jeu>, loss_<jeu>)")
    parser.add_argument("--device", default="cuda", help="cpu : mesure possible pendant qu'un entraînement occupe le GPU")
    parser.add_argument("--tokenizer_path", default="../dataset/fr_en/tokenizer_fr65_16k")
    args = parser.parse_args()

    tok = AutoTokenizer.from_pretrained(args.tokenizer_path)
    tests = {f[:-6]: [json.loads(line)["text"] for line in open(os.path.join(TEST_DIR, f), encoding="utf-8")] for f in sorted(os.listdir(TEST_DIR))}
    results = {}
    for spec in args.models:
        name, path = spec.split("=", 1)
        if path.startswith("hf:"):
            model_tok = AutoTokenizer.from_pretrained(path[3:])
            model = AutoModelForCausalLM.from_pretrained(path[3:], dtype=torch.bfloat16)
        else:
            # modèle hérité : poids.pth@fiche.json, la fiche (trainer/heritage.py) donne sa forme et son tokenizer
            path, _, fiche = path.partition("@")
            meta = json.load(open(fiche)) if fiche else {}
            model_tok = AutoTokenizer.from_pretrained(os.path.join(DATA_DIR, meta["tokenizer"])) if fiche else tok
            model = MiniMindForCausalLM(MiniMindConfig(vocab_size=len(model_tok), **meta.get("config", {})))
            model.load_state_dict({k: v.float() for k, v in torch.load(path, map_location="cpu").items()})
        print(f"{name} : {sum(p.numel() for p in model.parameters()) / 1e6:.0f}M paramètres", flush=True)
        model = model.to(args.device).eval()
        # petits lots : les logits d'un grand vocabulaire (65k pour LFM3) saturent vite les 12 Go
        results[name] = {t: score(model, model_tok, texts, batch_size=4, device=args.device) for t, texts in tests.items()}
        print(f"{name} : " + "  ".join(f"{t} {loss:.3f} / {bpb:.3f}" for t, (loss, bpb) in results[name].items()), flush=True)
        if args.mlflow_experiment:
            log_mlflow(args.mlflow_experiment, name, path, results[name])
        del model
        torch.cuda.empty_cache()
    names = list(results)
    print(f"{'jeu de test':22}" + "".join(f"{n + ' (loss)':>20}{n + ' (bits/octet)':>24}" for n in names))
    for t in tests:
        print(f"{t:22}" + "".join(f"{results[n][t][0]:20.3f}{results[n][t][1]:24.3f}" for n in names))
