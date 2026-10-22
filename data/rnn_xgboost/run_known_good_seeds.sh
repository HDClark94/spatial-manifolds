#!/bin/bash
set -e
cd /Users/harryclark/Documents/spatial-manifolds
PY=/opt/anaconda3/envs/sm/bin/python
TRAIN=scripts/figures/figure_validations/train_multiscale.py
VAL=scripts/figures/figure_validations/validate_multiscale.py

run_one () {
  SEED=$1; NAME=$2
  DIR=data/rnn_xgboost/$NAME
  mkdir -p "$DIR"
  $PY $TRAIN --rnn_seed $SEED --pc_seed 8001 --save_dir "$DIR" \
    --steps 25000 --Ng 1024 --pc_rf "0.12" --sequence_length 20 --irnn_scale 0.99 \
    > "$DIR/train.log" 2>&1
  $PY $VAL --dir "$DIR" --pc_rf "0.12" --pc_seed 8001 --sequence_length 20 \
    > "$DIR/validate.log" 2>&1
  echo "=== $NAME (seed $SEED) ==="; grep "decode err" "$DIR/validate.log"
}

run_one 4444 "merge_source_net6seed_Ng1024"
run_one 5555 "merge_source_net7seed_Ng1024"
echo "KNOWN_GOOD_SEEDS_DONE"
