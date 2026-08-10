#!/bin/bash
# Queue: wait for net710 to finish -> train HD integrator (50k) -> build net7+HD ΔpR2 cache.
REPO=/Users/harryclark/Documents/spatial-manifolds/GRID-PATTERN-FORMATION
SCR=/Users/harryclark/Documents/spatial-manifolds/scripts/figures/figure_validations
DATA=/Users/harryclark/Documents/spatial-manifolds/data/rnn_xgboost
LOG=$DATA/hd_pipeline.log
cd "$REPO" || exit 1

echo "QUEUED $(date '+%H:%M') — waiting for net710 to finish..." > "$LOG"
while pgrep -f build_two_net_cache_710.py >/dev/null; do sleep 60; done
echo "net710 finished $(date '+%H:%M') — training HD integrator (50k)..." >> "$LOG"

python3 "$SCR/train_hd_net.py" --save_dir "$DATA/hd_net_Ng1024" --steps 50000 --rnn_seed 7010 >> "$LOG" 2>&1
if ! grep -q "HD_TRAIN DONE" "$LOG"; then echo "HD TRAINING FAILED — aborting" >> "$LOG"; exit 1; fi

echo "HD net trained $(date '+%H:%M') — building net7+HD cache..." >> "$LOG"
python3 "$SCR/build_two_net_cache_hd7.py" >> "$LOG" 2>&1
if grep -q "HD7 DONE" "$LOG"; then echo "HD PIPELINE COMPLETE $(date '+%H:%M')" >> "$LOG"
else echo "HD BUILD FAILED $(date '+%H:%M')" >> "$LOG"; fi
