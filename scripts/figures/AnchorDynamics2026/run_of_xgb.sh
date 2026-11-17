#!/bin/zsh
# Overnight run of the open-field XGBoost decomposition (of_xgb_decomp.py).
# Usage: run_of_xgb.sh [n_workers]
# Resumable: finished sessions are skipped, so rerunning continues where it stopped.
set -u
NW=${1:-8}
cd "$(dirname "$0")"
LOG=../../../data/of_xgb/logs
mkdir -p $LOG
for w in $(seq 0 $((NW - 1))); do
  python3 of_xgb_decomp.py $w $NW > $LOG/w$w.log 2>&1 &
done
wait
python3 of_xgb_decomp.py > $LOG/collect.log 2>&1
python3 build_of_xgb_examples.py > $LOG/examples.log 2>&1
python3 fig1_supp_of_decomp.py > $LOG/figS4.log 2>&1
echo done > $LOG/DONE
