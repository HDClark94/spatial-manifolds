#!/bin/bash
set -e
cd /Users/harryclark/Documents/spatial-manifolds
PY=/opt/anaconda3/envs/sm/bin/python
TRAIN=scripts/figures/figure_validations/train_multiscale.py
VAL=scripts/figures/figure_validations/validate_multiscale.py
STEPS=25000

run_one () {
  NAME=$1; SEED=$2; PCRF=$3; SEQARG=$4; VALSEQ=$5
  DIR=data/rnn_xgboost/$NAME
  mkdir -p "$DIR"
  echo "=== $NAME: START $(date) ==="
  $PY $TRAIN --rnn_seed $SEED --pc_seed $SEED --save_dir "$DIR" \
    --steps $STEPS --pc_rf "$PCRF" $SEQARG --irnn_scale 0.99 \
    > "$DIR/train.log" 2>&1
  $PY $VAL --dir "$DIR" --pc_rf "$PCRF" --pc_seed $SEED --sequence_length $VALSEQ \
    > "$DIR/validate.log" 2>&1
  echo "=== $NAME: DONE $(date) ==="
  tail -25 "$DIR/validate.log"
}

# Controls: single place-cell width, FIXED sequence length -- does scale track horizon?
run_one "multiscale_seq20_control_Ng1024"  8001 "0.12" "--sequence_length 20"  20
run_one "multiscale_seq100_control_Ng1024" 8002 "0.12" "--sequence_length 100" 100

# Primary: mixed widths AND mixed sequence lengths together
run_one "multiscale_combo2_Ng1024" 8003 "0.06,0.18" "--sequence_length_mix 20,100" 60

echo "OVERNIGHT_PIPELINE2_DONE"
