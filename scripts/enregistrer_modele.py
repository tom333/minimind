"""
Inscrit un modèle entraîné dans le registre de modèles MLflow.

Le modèle est exporté au format Hugging Face (config + poids safetensors + tokenizer), envoyé comme artefact,
puis enregistré sous un nom et une version, avec un alias (« reference », « candidat »…) et le lien vers le run
d'entraînement : ses mesures, le commit du code et l'empreinte des données. Les poids sont découpés en parties
de moins de 100 Mo, taille maximale acceptée par le serveur.

Usage :
  python scripts/enregistrer_modele.py --nom minimind-fr-en-base --alias reference \\
      --poids out/etude_fr-en-full/fr-en-full-t0_768.pth --run minimind-fr-en-full/fr-en-full-t0
Recharger un modèle inscrit :  mlflow.artifacts.download_artifacts("models:/minimind-fr-en-base@reference")
puis MiniMindForCausalLM.from_pretrained(dossier).
"""

import argparse
import hashlib
import json
import os
import sys
import tempfile

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

import mlflow
import torch
from mlflow.tracking import MlflowClient
from transformers import AutoTokenizer

from model.model_minimind import MiniMindConfig, MiniMindForCausalLM
from trainer.provenance import git_commit

ARCH_KEYS = {"hidden_size": int, "num_hidden_layers": int, "num_attention_heads": int, "num_key_value_heads": int,
             "intermediate_size": int, "hidden_act": str, "ffn_glu": lambda v: v == "1", "mod_capacity": float,
             "ternary": str, "subln": lambda v: v == "1"}

parser = argparse.ArgumentParser()
parser.add_argument("--nom", required=True)
parser.add_argument("--poids", required=True)
parser.add_argument("--run", required=True, help="expérience/nom_du_run d'entraînement")
parser.add_argument("--tokenizer", default="dataset/fr_en/tokenizer_fr65_16k")
parser.add_argument("--alias", default="candidat")
parser.add_argument("--description", default="")
parser.add_argument("--mlflow_uri", default=os.environ.get("MINIMIND_MLFLOW_URI", "https://mlflow.tgu.ovh"))
args = parser.parse_args()

mlflow.set_tracking_uri(args.mlflow_uri)
client = MlflowClient()

# le run d'entraînement : architecture, mesures et traçabilité
experiment, run_name = args.run.split("/")
source = client.search_runs([client.get_experiment_by_name(experiment).experiment_id], f"tags.mlflow.runName = '{run_name}'")[0]
params = source.data.params
arch = {k: cast(params[k]) for k, cast in ARCH_KEYS.items() if params.get(k) not in (None, "None", "")}

tokenizer = AutoTokenizer.from_pretrained(args.tokenizer)
model = MiniMindForCausalLM(MiniMindConfig(vocab_size=len(tokenizer), **arch))
state = torch.load(args.poids, map_location="cpu")
model.load_state_dict({k: v.float() for k, v in state.items()})
with open(args.poids, "rb") as f:
    sha = hashlib.file_digest(f, "sha256").hexdigest()

# expérience dont les artefacts passent par le serveur (aucun identifiant de stockage côté client)
name = "minimind-modeles"
if client.get_experiment_by_name(name) is None:
    client.create_experiment(name, artifact_location="mlflow-artifacts:/minimind-modeles")
mlflow.set_experiment(name)

metrics = {k: v for k, v in source.data.metrics.items() if k.startswith("final_") or k == "val_loss"}
with tempfile.TemporaryDirectory() as tmp, mlflow.start_run(run_name=f"{args.nom}-{run_name}") as run:
    model.half().save_pretrained(tmp, max_shard_size="90MB")
    tokenizer.save_pretrained(tmp)
    manifest = {"run_entrainement": source.info.run_id, "experience": experiment, "poids_origine": args.poids, "poids_sha256": sha,
                "parametres_M": round(sum(p.numel() for p in model.parameters()) / 1e6, 2), "architecture": arch,
                "mesures": metrics, "tracabilite_entrainement": {k: v for k, v in source.data.tags.items() if not k.startswith("mlflow.")},
                "commit_export": git_commit()}
    with open(os.path.join(tmp, "manifeste.json"), "w") as f:
        json.dump(manifest, f, indent=1, ensure_ascii=False)
    mlflow.log_artifacts(tmp, "model")
    mlflow.log_metrics(metrics)
    mlflow.set_tags({"run_entrainement": source.info.run_id, "poids_sha256": sha[:16], "git_commit": git_commit()})
    if not client.search_registered_models(f"name = '{args.nom}'"):
        client.create_registered_model(args.nom, description="MiniMind bilingue français / anglais (https://github.com/tom333/minimind)")
    version = client.create_model_version(args.nom, source=f"{run.info.artifact_uri}/model", run_id=run.info.run_id)

client.set_registered_model_alias(args.nom, args.alias, version.version)
client.update_model_version(args.nom, version.version, description=args.description or f"Issu du run {args.run} ; val_loss {metrics.get('val_loss')}")
for key in ("val_loss", "final_top1", "final_loss_src_fr", "final_loss_src_en", "final_val_loss_fr", "final_val_loss_en"):
    if key in metrics:
        client.set_model_version_tag(args.nom, version.version, key, f"{metrics[key]:.4f}")
print(f"{args.nom} version {version.version} (alias {args.alias}) : {manifest['parametres_M']}M paramètres, mesures {metrics}")
