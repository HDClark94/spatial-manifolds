#!/bin/bash
# Autonomous two-network (1024+1024) pipeline: wait for training -> validate -> cache -> figures
DIR=/Users/harryclark/Documents/spatial-manifolds/scripts/figures/figure_validations
LOG=/Users/harryclark/Documents/spatial-manifolds/data/rnn_xgboost/two_net_pipeline.log
TRAIN_OUT=/private/tmp/claude-501/-Users-harryclark-Documents-spatial-manifolds/3b20463b-9f4d-451d-bd1d-edaed06c3152/tasks/byxfc17vj.output
cd /Users/harryclark/Documents/spatial-manifolds/GRID-PATTERN-FORMATION
: > "$LOG"

echo "[orch $(date +%H:%M)] waiting for seed5678 training to finish..." >> "$LOG"
for i in $(seq 1 400); do
  grep -q "DONE step=50000" "$TRAIN_OUT" 2>/dev/null && break
  sleep 30
done
if ! grep -q "DONE step=50000" "$TRAIN_OUT" 2>/dev/null; then
  echo "[orch $(date +%H:%M)] TRAINING NOT DONE after wait window — aborting" >> "$LOG"; exit 1
fi
echo "[orch $(date +%H:%M)] training done:" >> "$LOG"; grep "DONE" "$TRAIN_OUT" >> "$LOG"

echo "[orch $(date +%H:%M)] === validating both nets ===" >> "$LOG"
python3 "$DIR/validate_both_nets.py" >> "$LOG" 2>/dev/null
grep -q VALIDATION_DONE "$LOG" || echo "[orch] WARNING: validation did not finish cleanly" >> "$LOG"

echo "[orch $(date +%H:%M)] === building two-net xgboost cache ===" >> "$LOG"
python3 "$DIR/build_two_net_cache.py" >> "$LOG" 2>/dev/null
if ! grep -q "^DONE$" "$LOG"; then
  echo "[orch $(date +%H:%M)] cache build did not report DONE — aborting figures" >> "$LOG"; exit 1
fi

echo "[orch $(date +%H:%M)] === generating figures ===" >> "$LOG"
python3 "$DIR/make_two_net_figures.py" >> "$LOG" 2>/dev/null

echo "[orch $(date +%H:%M)] ===== ALL DONE =====" >> "$LOG"
