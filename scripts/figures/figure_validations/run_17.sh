#!/bin/bash
# Disconnected control on nets 1 + 7 (two recurrence-driven seeds). Waits for the 6+7
# pipeline to finish first so the two XGBoost caches don't split the CPU.
DIR=/Users/harryclark/Documents/spatial-manifolds/scripts/figures/figure_validations
DATA=/Users/harryclark/Documents/spatial-manifolds/data/rnn_xgboost
LOG67=$DATA/two_net_67_pipeline.log
LOG=$DATA/two_net_17_pipeline.log
cd /Users/harryclark/Documents/spatial-manifolds/GRID-PATTERN-FORMATION
: > "$LOG"
run(){ echo "[17 $(date +%H:%M)] $1" >> "$LOG"; }

run "=== waiting for 6+7 pipeline to finish (avoid CPU contention) ==="
for w in $(seq 1 720); do grep -q "67 DONE" "$LOG67" 2>/dev/null && break; sleep 60; done
grep -q "67 DONE" "$LOG67" 2>/dev/null || run "WARNING: 6+7 not done after wait window — proceeding anyway"

run "=== building nets 1+7 xgboost cache ==="
python3 "$DIR/build_two_net_cache_17.py" >> "$LOG" 2>/dev/null
grep -q "^DONE$" "$LOG" || { run "cache did not report DONE — aborting"; exit 1; }

run "=== generating figures ==="
python3 "$DIR/make_figures_17.py" >> "$LOG" 2>/dev/null

run "=== shift-null ==="
python3 "$DIR/make_shift_null_17.py" >> "$LOG" 2>/dev/null

run "===== 17 DONE ====="
