#!/bin/bash
set -e
cd /Users/harryclark/Documents/spatial-manifolds
PY=/opt/anaconda3/envs/sm/bin/python

echo "=== RUN 1: combined (mixed widths + long sequence + encoder mask) ==="
DIR1=data/rnn_xgboost/multiscale_combo_mask_Ng1024
mkdir -p "$DIR1"
$PY scripts/figures/figure_validations/train_multiscale.py \
  --rnn_seed 6001 --pc_seed 6001 --save_dir "$DIR1" \
  --steps 50000 --pc_rf "0.06,0.18" --sequence_length 100 --encoder_mask \
  > "$DIR1/train.log" 2>&1
$PY scripts/figures/figure_validations/validate_multiscale.py \
  --dir "$DIR1" --pc_rf "0.06,0.18" --pc_seed 6001 --sequence_length 100 \
  > "$DIR1/validate.log" 2>&1
echo "=== RUN 1 DONE ==="
tail -20 "$DIR1/validate.log"

echo "=== RUN 2: ablation (mixed widths + long sequence, NO encoder mask) ==="
DIR2=data/rnn_xgboost/multiscale_combo_nomask_Ng1024
mkdir -p "$DIR2"
$PY scripts/figures/figure_validations/train_multiscale.py \
  --rnn_seed 6002 --pc_seed 6002 --save_dir "$DIR2" \
  --steps 50000 --pc_rf "0.06,0.18" --sequence_length 100 \
  > "$DIR2/train.log" 2>&1
$PY scripts/figures/figure_validations/validate_multiscale.py \
  --dir "$DIR2" --pc_rf "0.06,0.18" --pc_seed 6002 --sequence_length 100 \
  > "$DIR2/validate.log" 2>&1
echo "=== RUN 2 DONE ==="
tail -20 "$DIR2/validate.log"

echo "OVERNIGHT_PIPELINE_DONE"
