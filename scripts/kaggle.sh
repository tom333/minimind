#!/usr/bin/env bash
# Pilotage de Kaggle (GPU gratuit) depuis ce PC. Prérequis : jeton d'API dans ~/.kaggle/kaggle.json (chmod 600).
#   scripts/kaggle.sh donnees                     crée ou met à jour le jeu de données privé (tokens en cache + tokenizer, ~2 Go)
#   scripts/kaggle.sh lancer diagnostic|bench     mesure la machine avec notre banc de débit
#   scripts/kaggle.sh lancer train NOM "ARGS"     entraînement (ARGS = options de train_pretrain.py) ; relancer la
#                                                 même commande après une session coupée : le run reprend où il s'était arrêté
#   scripts/kaggle.sh etat | sorties              état du dernier lancement / téléchargement de out/ dans out/kaggle/
set -euo pipefail
cd "$(dirname "$0")/.."
KAGGLE="uv run --quiet kaggle"
USER_K=$(python3 -c "import json,os;print(json.load(open(os.path.expanduser('~/.kaggle/kaggle.json')))['username'])")
DATASET="$USER_K/minimind-fr-en"; KERNEL="$USER_K/minimind-run"; BUILD=$PWD/out/kaggle_build   # sur le même disque que le corpus : liens physiques, pas de copie
CORPUS=${MINIMIND_KAGGLE_CORPUS:-pretrain_fr65}

case "${1:-}" in
  donnees)
    mkdir -p "$BUILD/data" && find "$BUILD/data" -mindepth 1 -delete; mkdir -p "$BUILD/data/tokenizer_fr65_16k"
    for f in train_rand2000.npy val_rand2000.npy val_rand2000.langs.json; do ln -f "$PWD/dataset/fr_en/$CORPUS.jsonl.$f" "$BUILD/data/"; done
    cp -f dataset/fr_en/tokenizer_fr65_16k/* "$BUILD/data/tokenizer_fr65_16k/"
    printf '{"title": "minimind-fr-en", "id": "%s", "licenses": [{"name": "other"}]}\n' "$DATASET" > "$BUILD/data/dataset-metadata.json"
    if $KAGGLE datasets status "$DATASET" >/dev/null 2>&1; then $KAGGLE datasets version -p "$BUILD/data" -m "mise à jour" --dir-mode zip
    else $KAGGLE datasets create -p "$BUILD/data" --dir-mode zip; fi ;;   # privé par défaut
  lancer)
    mode=$2; name=${3:-$mode}; args=${4:-}; mkdir -p "$BUILD/kernel"
    # la sortie de la session précédente n'est attachée (pour la reprise) que si le notebook existe déjà
    EXISTE=non; $KAGGLE kernels status "$KERNEL" >/dev/null 2>&1 && EXISTE=oui
    python3 - "$mode" "$name" "$args" <<PY
import json, sys
mode, name, args = sys.argv[1:4]
config = {"mode": mode, "name": name, "args": args.split(), "ref": "experiments", "repo": "https://github.com/tom333/minimind",
          "max_hours": 8, "mlflow_uri": "https://mlflow.tgu.ovh"}
src = open("cloud/kaggle/run.py").read().replace("__CONFIG__", repr(config))
open("$BUILD/kernel/run.py", "w").write(src)
meta = {"id": "$KERNEL", "title": "minimind-run", "code_file": "run.py", "language": "python", "kernel_type": "script",
        "is_private": True, "enable_gpu": True, "enable_internet": True,
        "dataset_sources": ["$DATASET"], "kernel_sources": ["$KERNEL"] if mode == "train" and "$EXISTE" == "oui" else [], "competition_sources": []}
json.dump(meta, open("$BUILD/kernel/kernel-metadata.json", "w"), indent=1)
PY
    $KAGGLE kernels push -p "$BUILD/kernel" ;;
  etat) $KAGGLE kernels status "$KERNEL" ;;
  sorties) mkdir -p out/kaggle && $KAGGLE kernels output "$KERNEL" -p out/kaggle && ls -la out/kaggle ;;
  *) sed -n 2,8p "$0" ;;
esac
