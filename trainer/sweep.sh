#!/usr/bin/env bash
# Sweep générique : lance chaque run décrit dans sweeps/<nom>.txt, à budget fixe, loggé dans MLflow.
# Format d'une ligne : <nom_du_run> <arguments supplémentaires...>   (lignes vides et # ignorées)
# Base commune : LR 7e-4, warmup 2 %, wd 0.1, 50M tokens (surchargeable par les arguments de la ligne).
# Usage : cd trainer && bash sweep.sh batch
set -euo pipefail
cd "$(dirname "$0")"
name=$1
out=../out/sweep_$name
mkdir -p "$out"

grep -vE '^\s*(#|$)' "sweeps/$name.txt" | while read -r run extra; do
  echo "=== $(date +%H:%M) $run"
  # shellcheck disable=SC2086
  ../.venv/bin/python train_pretrain.py \
    --learning_rate 7e-4 --warmup_ratio 0.02 --weight_decay 0.1 --max_tokens 50000000 \
    --packing 1 --use_compile 1 --save_resume 0 --eval_interval 500 --log_interval 100 \
    --save_dir "$out" --save_weight "pretrain_$run" \
    --use_mlflow --mlflow_experiment "minimind-$name" --run_name "$run" \
    $extra > "$out/$run.log" 2>&1 < /dev/null
  grep "val_loss" "$out/$run.log" | tail -1
done
