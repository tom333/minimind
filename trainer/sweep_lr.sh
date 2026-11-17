#!/usr/bin/env bash
# Sweep du learning rate : même budget de tokens pour chaque run, comparaison sur val_loss dans MLflow.
# Usage : cd trainer && bash sweep_lr.sh
set -euo pipefail
cd "$(dirname "$0")"

LRS=${LRS:-"1e-4 3e-4 5e-4 1e-3 2e-3"}
MAX_TOKENS=${MAX_TOKENS:-50000000}
mkdir -p ../out/sweep_lr

for lr in $LRS; do
  echo "=== $(date +%H:%M) lr=$lr"
  ../.venv/bin/python train_pretrain.py \
    --learning_rate "$lr" --max_tokens "$MAX_TOKENS" \
    --packing 1 --use_compile 1 --save_resume 0 --eval_interval 500 --log_interval 100 \
    --save_dir ../out/sweep_lr --save_weight "pretrain_lr$lr" \
    --use_mlflow --mlflow_experiment minimind-pretrain --run_name "lr-$lr" \
    > "../out/sweep_lr/lr$lr.log" 2>&1
  grep "val_loss" "../out/sweep_lr/lr$lr.log" | tail -1
done
