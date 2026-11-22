#!/usr/bin/env bash
# Construit l'image minimind (entraînement + code Dagster) et la pousse dans le registre du cluster.
set -euo pipefail
cd "$(dirname "$0")/.."
image=localhost:32000/minimind-dagster:latest
commit=$(git rev-parse --short=12 HEAD)$(git diff --quiet HEAD -- . ":(exclude)out" || echo -dirty)
docker build --build-arg GIT_COMMIT="$commit" -t "$image" .
docker push "$image"
echo "Image publiée. Recharger le code : kubectl -n dagster rollout restart deploy/dagster-code-minimind"
