"""
Code Dagster de minimind : lance les études Optuna (trainer/study.py) sur le GPU du cluster.

Un run = un pod Kubernetes qui demande le GPU avec la priorité `gpu-training`. Cette priorité est
supérieure à celle de LocalAI : le pod LocalAI est évincé le temps du run, puis replanifié
automatiquement par Kubernetes quand le GPU se libère (rien à remettre en route à la main).
"""
import os
import sys

import dagster as dg
import yaml

REPO = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
sys.path.insert(0, os.path.join(REPO, "trainer"))
HOST_DIR = "/data/projets/perso/minimind"  # données et sorties restent sur le disque du nœud pc

# Configuration Kubernetes du pod de run : GPU exclusif, nœud pc, volumes de données et de sorties
GPU_RUN = {
    "dagster-k8s/config": {
        "container_config": {
            "resources": {
                "requests": {"cpu": "2", "memory": "12Gi", "nvidia.com/gpu": "1"},
                "limits": {"memory": "28Gi", "nvidia.com/gpu": "1"},
            },
            "volume_mounts": [
                {"name": "corpus", "mount_path": "/data/corpus"},
                {"name": "sorties", "mount_path": "/data/out"},
                {"name": "shm", "mount_path": "/dev/shm"},  # mémoire partagée des workers du DataLoader
            ],
        },
        "pod_spec_config": {
            "priority_class_name": "gpu-training",
            "node_selector": {"kubernetes.io/hostname": "pc"},
            "volumes": [
                {"name": "corpus", "host_path": {"path": f"{HOST_DIR}/dataset"}},
                {"name": "sorties", "host_path": {"path": f"{HOST_DIR}/out"}},
                {"name": "shm", "empty_dir": {"medium": "Memory", "size_limit": "8Gi"}},
            ],
        },
        "job_spec_config": {"ttl_seconds_after_finished": 86400},
    }
}


class EtudeConfig(dg.Config):
    fichier: str = "muon_lr.yaml"  # fichier de trainer/etudes/ (versionné avec le code, donc avec l'image)
    surcharges: dict[str, str] = {}  # remplace des arguments de `base`, ex. {"max_tokens": "12500000"}


@dg.op
def etude_optuna(context: dg.OpExecutionContext, config: EtudeConfig) -> None:
    from study import run_study, summary

    with open(os.path.join(REPO, "trainer", "etudes", config.fichier)) as f:
        cfg = yaml.safe_load(f)
    context.log.info(f"Étude {cfg['name']} ({cfg.get('sampler', 'grid')}), surcharges : {config.surcharges}")
    study = run_study(cfg, config.surcharges)
    done = [t for t in study.trials if t.value is not None]
    context.add_output_metadata(
        {
            "trials": dg.MetadataValue.md(summary(study)),
            "meilleure_val_loss": study.best_value if done else None,
            "meilleurs_parametres": dg.MetadataValue.json(study.best_params if done else {}),
            "mlflow": dg.MetadataValue.url(f"https://mlflow.tgu.ovh/#/experiments?searchFilter=minimind-{cfg['name']}"),
        }
    )
    if not done:
        raise dg.Failure("aucun trial n'a abouti")


@dg.job(tags=GPU_RUN, description="Lance ou reprend une étude Optuna d'entraînement sur le GPU.")
def etude():
    etude_optuna()


defs = dg.Definitions(jobs=[etude])
