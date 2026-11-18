#!/usr/bin/env bash
# Construit l'image minimind (entraînement + code Dagster) et la pousse dans le registre du cluster.
set -euo pipefail
cd "$(dirname "$0")/.."
image=localhost:32000/minimind-dagster:latest
docker build -t "$image" .
docker push "$image"
echo "Image publiée. Recharger le code : kubectl -n dagster rollout restart deploy/dagster-code-minimind"
