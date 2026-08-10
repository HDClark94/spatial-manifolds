#!/bin/bash
# Autonomous DIFFERENT-place-cell pipeline: train A -> train B -> validate -> cache -> figures
DIR=/Users/harryclark/Documents/spatial-manifolds/scripts/figures/figure_validations
DATA=/Users/harryclark/Documents/spatial-manifolds/data/rnn_xgboost
LOG=$DATA/two_net_diffPC_pipeline.log
cd /Users/harryclark/Documents/spatial-manifolds/GRID-PATTERN-FORMATION
: > "$LOG"
run(){ echo "[orch $(date +%H:%M)] $1" >> "$LOG"; }

run "=== training net A (rnn_seed 1234, pc_seed 101) ==="
python3 "$DIR/train_diffPC.py" --rnn_seed 1234 --pc_seed 101 --save_dir "$DATA/diffPC_A_Ng1024" --steps 50000 >> "$LOG" 2>/dev/null
[ -f "$DATA/diffPC_A_Ng1024/ckpt.pth" ] || { run "netA ckpt missing — abort"; exit 1; }

run "=== training net B (rnn_seed 5678, pc_seed 202) ==="
python3 "$DIR/train_diffPC.py" --rnn_seed 5678 --pc_seed 202 --save_dir "$DATA/diffPC_B_Ng1024" --steps 50000 >> "$LOG" 2>/dev/null
[ -f "$DATA/diffPC_B_Ng1024/ckpt.pth" ] || { run "netB ckpt missing — abort"; exit 1; }

run "=== validating both nets ==="
python3 "$DIR/validate_both_diffPC.py" >> "$LOG" 2>/dev/null
grep -q VALIDATION_DONE "$LOG" || run "WARNING: validation did not finish cleanly (continuing)"

run "=== building diffPC xgboost cache ==="
python3 "$DIR/build_two_net_cache_diffPC.py" >> "$LOG" 2>/dev/null
grep -q "^DONE$" "$LOG" || { run "cache did not report DONE — aborting figures"; exit 1; }

run "=== generating figures ==="
python3 "$DIR/make_figures_diffPC.py" >> "$LOG" 2>/dev/null

run "===== ALL DONE ====="
