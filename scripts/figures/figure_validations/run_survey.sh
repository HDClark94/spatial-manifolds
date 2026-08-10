#!/bin/bash
# 10-seed survey: train 8 new Ng=1024 nets (reuse the 2 diffPC nets), then remake the survey figure.
DIR=/Users/harryclark/Documents/spatial-manifolds/scripts/figures/figure_validations
DATA=/Users/harryclark/Documents/spatial-manifolds/data/rnn_xgboost
LOG=$DATA/survey.log
cd /Users/harryclark/Documents/spatial-manifolds/GRID-PATTERN-FORMATION
: > "$LOG"
run(){ echo "[survey $(date +%H:%M)] $1" >> "$LOG"; }

# new nets: "rnn_seed pc_seed name"
SEEDS=("1111 11 survey_s3" "2222 22 survey_s4" "3333 33 survey_s5" "4444 44 survey_s6" \
       "5555 55 survey_s7" "6666 66 survey_s8" "7777 77 survey_s9" "8888 88 survey_s10")

for entry in "${SEEDS[@]}"; do
  set -- $entry; rnn=$1; pc=$2; name=$3
  if [ -f "$DATA/${name}_Ng1024/ckpt.pth" ] && python3 -c "import torch,sys; sys.exit(0 if torch.load('$DATA/${name}_Ng1024/ckpt.pth',map_location='cpu')['history']['step']>=50000 else 1)" 2>/dev/null; then
    run "=== $name already at 50k, skipping ==="; continue
  fi
  run "=== training $name (rnn $rnn / pc $pc) ==="
  python3 "$DIR/train_diffPC.py" --rnn_seed $rnn --pc_seed $pc --save_dir "$DATA/${name}_Ng1024" --steps 50000 >> "$LOG" 2>/dev/null
  [ -f "$DATA/${name}_Ng1024/ckpt.pth" ] || run "WARNING: $name ckpt missing"
done

run "=== all training done — making survey figure ==="
python3 "$DIR/survey_validation_figure.py" >> "$LOG" 2>/dev/null
run "===== SURVEY DONE ====="
