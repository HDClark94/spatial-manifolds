#!/bin/bash
# Overnight: wait for tanh-4096 training (+ 710grid to free the CPU) -> IF it learned, run net7+tanh xgboost.
REPO=/Users/harryclark/Documents/spatial-manifolds/GRID-PATTERN-FORMATION
SCR=/Users/harryclark/Documents/spatial-manifolds/scripts/figures/figure_validations
DATA=/Users/harryclark/Documents/spatial-manifolds/data/rnn_xgboost
LOG=$DATA/tanh_pipeline.log; TL=/tmp/tanh4096.log
cd "$REPO" || exit 1

echo "QUEUED $(date '+%H:%M') — waiting for tanh training + 710grid to finish..." > "$LOG"
while pgrep -f "Ng 4096 --activation tanh" >/dev/null; do sleep 60; done
while pgrep -f build_two_net_cache_710grid.py >/dev/null; do sleep 60; done

minloss=$(grep -E "^step" "$TL" | sed -E 's/.*loss ([0-9.]+) err.*/\1/' | sort -n | head -1)
minerr=$(grep -E "^step" "$TL" | sed -E 's/.*err +([0-9.]+) cm.*/\1/' | sort -n | head -1)
echo "tanh training ended: min loss=${minloss} (chance=6.238), min err=${minerr} cm  $(date '+%H:%M')" >> "$LOG"

if awk "BEGIN{exit !(${minloss:-9} < 6.10)}"; then
  echo "TANH PROMISING (loss dropped below chance) — running net7+tanh xgboost $(date '+%H:%M')" >> "$LOG"
  python3 "$SCR/build_two_net_cache_tanh7.py" >> "$LOG" 2>&1
  if grep -q "TANH7 DONE" "$LOG"; then echo "TANH PIPELINE COMPLETE $(date '+%H:%M')" >> "$LOG"
  else echo "TANH BUILD FAILED $(date '+%H:%M')" >> "$LOG"; fi
else
  echo "TANH NOT PROMISING (min loss ${minloss} >= 6.10, ~chance) — SKIPPING xgboost $(date '+%H:%M')" >> "$LOG"
fi
