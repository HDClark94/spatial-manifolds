#!/bin/bash
# Disconnected control on nets 6 + 7 (grid-forming 1024 seeds): cache -> figures -> shift-null
DIR=/Users/harryclark/Documents/spatial-manifolds/scripts/figures/figure_validations
DATA=/Users/harryclark/Documents/spatial-manifolds/data/rnn_xgboost
LOG=$DATA/two_net_67_pipeline.log
cd /Users/harryclark/Documents/spatial-manifolds/GRID-PATTERN-FORMATION
: > "$LOG"
run(){ echo "[67 $(date +%H:%M)] $1" >> "$LOG"; }

run "=== building nets 6+7 xgboost cache ==="
python3 "$DIR/build_two_net_cache_67.py" >> "$LOG" 2>/dev/null
grep -q "^DONE$" "$LOG" || { run "cache did not report DONE — aborting"; exit 1; }

run "=== generating figures ==="
python3 "$DIR/make_figures_67.py" >> "$LOG" 2>/dev/null

run "=== shift-null ==="
python3 "$DIR/make_shift_null_67.py" >> "$LOG" 2>/dev/null

run "===== 67 DONE ====="
