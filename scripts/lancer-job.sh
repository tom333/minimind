#!/usr/bin/env bash
# Lance un job Dagster de l'emplacement minimind et affiche l'identifiant du run.
# Usage : scripts/lancer-job.sh corpus_brut            (job d'assets, sans configuration)
#         scripts/lancer-job.sh etude '{"ops":{"etude_optuna":{"config":{"fichier":"btt.yaml"}}}}'
set -euo pipefail
job=$1
config=${2:-"{}"}
kubectl -n dagster port-forward svc/dagster-dagster-webserver 13000:80 >/dev/null 2>&1 &
pf=$!; trap 'kill $pf' EXIT; sleep 4
gql() { command curl -s -X POST localhost:13000/graphql -H 'Content-Type: application/json' -d "$1"; }
gql '{"query":"mutation { reloadRepositoryLocation(repositoryLocationName: \"minimind\") { __typename } }"}' >/dev/null
sleep 3
gql '{"query":"mutation($p: ExecutionParams!){ launchRun(executionParams:$p){ __typename ... on LaunchRunSuccess { run { runId } } ... on PythonError { message } ... on RunConfigValidationInvalid { errors { message } } } }","variables":{"p":{"selector":{"repositoryLocationName":"minimind","repositoryName":"__repository__","jobName":"'"$job"'"},"runConfigData":'"$config"'}}}' \
  | python3 -c "import json,sys; d=json.load(sys.stdin)['data']['launchRun']; print(d['run']['runId'] if 'run' in d else d)"
