#!/usr/bin/env bash
# Scaling law en données : même modèle/LR, budgets de tokens croissants (chaque run a son propre cosinus complet).
# Usage : cd trainer && bash sweep_tokens.sh
set -euo pipefail
cd "$(dirname "$0")"

BUDGETS=${BUDGETS:-"12500000 25000000 50000000 100000000"}
LR=${LR:-7e-4}
mkdir -p ../out/sweep_tokens

for tok in $BUDGETS; do
  echo "=== $(date +%H:%M) tokens=$tok"
  ../.venv/bin/python train_pretrain.py \
    --learning_rate "$LR" --max_tokens "$tok" \
    --packing 1 --use_compile 1 --save_resume 0 --eval_interval 500 --log_interval 100 \
    --save_dir ../out/sweep_tokens --save_weight "pretrain_tok$tok" \
    --use_mlflow --mlflow_experiment minimind-scaling --run_name "tokens-$((tok / 1000000))M" \
    > "../out/sweep_tokens/tok$tok.log" 2>&1
  grep "val_loss" "../out/sweep_tokens/tok$tok.log" | tail -1
done
