#!/bin/bash
# Queue: wait for the HD pipeline to finish -> build net7(GRID-selected)+net10(random) ΔpR2 cache.
REPO=/Users/harryclark/Documents/spatial-manifolds/GRID-PATTERN-FORMATION
SCR=/Users/harryclark/Documents/spatial-manifolds/scripts/figures/figure_validations
DATA=/Users/harryclark/Documents/spatial-manifolds/data/rnn_xgboost
LOG=$DATA/net710grid.log
cd "$REPO" || exit 1

echo "QUEUED $(date '+%H:%M') — waiting for HD pipeline to finish..." > "$LOG"
while pgrep -f run_hd_pipeline.sh >/dev/null; do sleep 60; done
echo "HD pipeline finished $(date '+%H:%M') — building net7(grid)+net10 cache..." >> "$LOG"

python3 "$SCR/build_two_net_cache_710grid.py" >> "$LOG" 2>&1
if grep -q "NET710GRID DONE" "$LOG"; then echo "NET710GRID COMPLETE $(date '+%H:%M')" >> "$LOG"
else echo "NET710GRID FAILED $(date '+%H:%M')" >> "$LOG"; fi
