#!/usr/bin/env bash
# Pilotage de Google Colab depuis ce PC (google-colab-cli). Prérequis : `uv tool install google-colab-cli`, puis
# une première commande `colab sessions` lancée à la main pour s'authentifier (URL à ouvrir, code à coller).
#   scripts/colab.sh session [T4|L4|A100|H100]     ouvre la machine (session « minimind »)
#   scripts/colab.sh donnees                       envoie les tokens en cache et le tokenizer (~2 Go) dans /content/data
#   scripts/colab.sh lancer diagnostic|bench       mesure la machine avec notre banc de débit
#   scripts/colab.sh lancer train NOM "ARGS"       entraînement détaché (ARGS = options de train_pretrain.py) ; relancer
#                                                  la même commande après une coupure : le run reprend où il s'était arrêté
#   scripts/colab.sh etat                          fin du journal, GPU, processus (sert aussi à garder la session active)
#   scripts/colab.sh garder                        `etat` toutes les 5 minutes jusqu'à la fin du travail
#   scripts/colab.sh sorties                       rapatrie journaux, bancs et poids dans out/colab/
#   scripts/colab.sh stop                          libère la machine
# Sorties sur la machine : /content/out, ou Google Drive si MINIMIND_COLAB_OUT=/content/drive/MyDrive/minimind/out
# (après `colab drivemount -s minimind`), ce qui fait survivre le point de reprise à la perte de la machine.
set -euo pipefail
cd "$(dirname "$0")/.."
S=minimind; OUT=${MINIMIND_COLAB_OUT:-/content/out}; CORPUS=${MINIMIND_KAGGLE_CORPUS:-pretrain_fr65}
py() { colab exec -s $S --timeout "${2:-120}" <<<"$1"; }   # exécute du Python dans le noyau distant

case "${1:-}" in
  session) colab new -s $S --gpu "${2:-T4}"; py 'import subprocess; print(subprocess.run("nvidia-smi --query-gpu=name,memory.total,driver_version --format=csv,noheader", shell=True, capture_output=True, text=True).stdout)' ;;
  donnees)
    py 'import os; os.makedirs("/content/data/tokenizer_fr65_16k", exist_ok=True)'
    for f in dataset/fr_en/tokenizer_fr65_16k/*; do colab upload -s $S "$f" "/content/data/tokenizer_fr65_16k/$(basename "$f")"; done
    for f in val_rand2000.langs.json val_rand2000.npy; do colab upload -s $S "dataset/fr_en/$CORPUS.jsonl.$f" "/content/data/$CORPUS.jsonl.$f"; done
    # le gros fichier de tokens (~2 Go) : Colab refuse les envois de plus de ~50 Mo, donc morceaux de 48 Mo,
    # réessayés en cas d'échec, puis reconstitution et contrôle de l'empreinte sur la machine
    big="dataset/fr_en/$CORPUS.jsonl.train_rand2000.npy"; parts=out/kaggle_build/chunks
    mkdir -p $parts && find $parts -type f -delete; split -b 48M -d -a 3 "$big" $parts/train.part.
    for f in $parts/train.part.*; do
      for essai in 1 2 3 4 5; do colab upload -s $S "$f" "/content/data/$(basename "$f")" >/dev/null 2>&1 && break; sleep 10; done
      echo "envoyé : $(basename "$f")"
    done
    sum=$(sha256sum "$big" | cut -d' ' -f1)
    py "import subprocess; print(subprocess.run('cd /content/data && cat train.part.* > $CORPUS.jsonl.train_rand2000.npy && rm -f train.part.* chunk_test.bin && sha256sum $CORPUS.jsonl.train_rand2000.npy', shell=True, capture_output=True, text=True).stdout)" 600 | grep -q "$sum" && echo "tokens reconstitués, empreinte conforme" || echo "ERREUR : empreinte différente"
    py 'import subprocess; print(subprocess.run("ls -la /content/data /content/data/tokenizer_fr65_16k", shell=True, capture_output=True, text=True).stdout)' ;;
  lancer)
    mode=$2; name=${3:-$mode}; args=${4:-}; tmp=$(mktemp --suffix=.py)
    python3 - "$mode" "$name" "$args" "$OUT" > "$tmp" <<'PY'
import sys
mode, name, args, out = sys.argv[1:5]
config = {"mode": mode, "name": name, "args": args.split(), "ref": "experiments", "repo": "https://github.com/tom333/minimind",
          "max_hours": 10, "mlflow_uri": "https://mlflow.tgu.ovh", "input": "/content/data", "out": out}
print(open("cloud/run.py").read().replace("__CONFIG__", repr(config)))
PY
    colab upload -s $S "$tmp" /content/run.py; rm -f "$tmp"
    # processus détaché : il survit à la fin de la commande, et son journal va dans la sortie
    py "import os, subprocess; os.makedirs('$OUT', exist_ok=True); subprocess.Popen('nohup python /content/run.py > $OUT/run.log 2>&1 &', shell=True); print('lancé : $mode $name')" ;;
  etat) py "import subprocess; print(subprocess.run('tail -n ${2:-15} $OUT/run.log | cut -c1-220; echo; nvidia-smi --query-gpu=utilization.gpu,memory.used --format=csv,noheader; pgrep -f \"run.py|train_pretrain|bench_speed\" >/dev/null && echo EN_COURS || echo TERMINE', shell=True, capture_output=True, text=True).stdout)" ;;
  garder) until "$0" etat 3 | tee /dev/stderr | grep -q TERMINE; do sleep 300; done ;;
  sorties)
    mkdir -p out/colab
    for f in $(py "import glob, os; print(' '.join(p for p in glob.glob('$OUT/*') if os.path.isfile(p)))" | tail -1); do colab download -s $S "$f" "out/colab/$(basename "$f")" && echo "reçu : $(basename "$f")"; done ;;
  stop) colab stop -s $S ;;
  *) sed -n 2,15p "$0" ;;
esac
