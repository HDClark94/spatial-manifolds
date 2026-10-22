#!/bin/bash
set -e
cd /Users/harryclark/Documents/spatial-manifolds
PY=/opt/anaconda3/envs/sm/bin/python
TRAIN=scripts/figures/figure_validations/train_multiscale.py
VAL=scripts/figures/figure_validations/validate_multiscale.py
STEPS=10000

run_screen () {
  SEED=$1; PCRF=$2
  DIR=data/rnn_xgboost/seedscreen_${PCRF}_s${SEED}
  mkdir -p "$DIR"
  $PY $TRAIN --rnn_seed $SEED --pc_seed 8001 --save_dir "$DIR" \
    --steps $STEPS --Ng 1024 --pc_rf "$PCRF" --sequence_length 20 --irnn_scale 0.99 \
    > "$DIR/train.log" 2>&1
  $PY $VAL --dir "$DIR" --pc_rf "$PCRF" --pc_seed 8001 --sequence_length 20 --n_avg 20 \
    > "$DIR/validate.log" 2>&1
  RESULT=$(grep "decode err" "$DIR/validate.log")
  echo "seed=$SEED pc_rf=$PCRF -> $RESULT"
}

for SEED in 9601 9602 9603 9604; do
  run_screen $SEED "0.12"
done
for SEED in 9701 9702 9703 9704; do
  run_screen $SEED "0.1944"
done

echo "SEED_SCREEN_DONE"
