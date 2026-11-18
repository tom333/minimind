#!/usr/bin/env bash
# Lance une étude Optuna (fichier de trainer/etudes/) comme run Dagster sur le GPU du cluster.
# Usage : scripts/lancer-etude.sh btt.yaml        → affiche l'identifiant du run
set -euo pipefail
fichier=$1
kubectl -n dagster port-forward svc/dagster-dagster-webserver 13000:80 >/dev/null 2>&1 &
pf=$!; trap 'kill $pf' EXIT; sleep 4
gql() { command curl -s -X POST localhost:13000/graphql -H 'Content-Type: application/json' -d "$1"; }
gql '{"query":"mutation { reloadRepositoryLocation(repositoryLocationName: \"minimind\") { __typename } }"}' >/dev/null
sleep 3
gql '{"query":"mutation($p: ExecutionParams!){ launchRun(executionParams:$p){ __typename ... on LaunchRunSuccess { run { runId } } ... on PythonError { message } } }","variables":{"p":{"selector":{"repositoryLocationName":"minimind","repositoryName":"__repository__","jobName":"etude"},"runConfigData":{"ops":{"etude_optuna":{"config":{"fichier":"'"$fichier"'"}}}}}}}' \
  | python3 -c "import json,sys; d=json.load(sys.stdin)['data']['launchRun']; print(d['run']['runId'] if 'run' in d else d)"
