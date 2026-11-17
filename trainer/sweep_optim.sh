#!/usr/bin/env bash
# Sweep d'optimisation à budget fixe (50M tokens), comparé à la référence tokens-50M (val_loss 2.95).
# Chaque ligne : "nom_du_run | arguments supplémentaires"
# Usage : cd trainer && bash sweep_optim.sh
set -euo pipefail
cd "$(dirname "$0")"

RUNS=(
  "ref-seed2       | --seed 2"
  "warmup2         | --warmup_ratio 0.02"
  "wd0.1           | --weight_decay 0.1"
  "warmup2-lr1.5e-3 | --warmup_ratio 0.02 --learning_rate 1.5e-3"
  "warmup2-lr2e-3   | --warmup_ratio 0.02 --learning_rate 2e-3"
)
mkdir -p ../out/sweep_optim

for run in "${RUNS[@]}"; do
  name=$(echo "${run%%|*}" | xargs)
  extra=${run#*|}
  echo "=== $(date +%H:%M) $name"
  # shellcheck disable=SC2086
  ../.venv/bin/python train_pretrain.py \
    --learning_rate 7e-4 --max_tokens 50000000 \
    --packing 1 --use_compile 1 --save_resume 0 --eval_interval 500 --log_interval 100 \
    --save_dir ../out/sweep_optim --save_weight "pretrain_$name" \
    --use_mlflow --mlflow_experiment minimind-optim --run_name "$name" \
    $extra > "../out/sweep_optim/$name.log" 2>&1
  grep "val_loss" "../out/sweep_optim/$name.log" | tail -1
done
