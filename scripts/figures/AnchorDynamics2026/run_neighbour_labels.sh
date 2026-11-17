#!/bin/zsh
# Unfiltered whole-track and pre-zone labels (build_neighbour_labels.py), then the
# neighbour test (neighbour_test.py) for Figure S20. ~15 min on 7 workers.
# Usage: run_neighbour_labels.sh [n_workers]
# Resumable: finished sessions are skipped. Delete data/population_state/neighbour_test/
# to rebuild from changed labels or cell classes.
set -u
NW=${1:-7}
cd "$(dirname "$0")"
LOG=../../../data/population_state/neighbour_test/logs
mkdir -p $LOG
for w in $(seq 0 $((NW - 1))); do
  python3 build_neighbour_labels.py $w $NW > $LOG/w$w.log 2>&1 &
done
wait
python3 neighbour_test.py > $LOG/test.log 2>&1
