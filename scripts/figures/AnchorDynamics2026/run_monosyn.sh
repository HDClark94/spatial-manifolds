#!/bin/zsh
# Monosynaptic detection over every labelled session (monosyn.py batch).
# Usage: run_monosyn.sh [n_workers]
# Resumable: finished sessions are skipped. Figure 5 and S19 cache tables built
# from these (pairs_by_identity, subnetwork_*, connection_direction); delete
# them when the connections are rebuilt.
set -u
NW=${1:-3}
cd "$(dirname "$0")"
LOG=../../../data/monosyn/logs
mkdir -p $LOG
for w in $(seq 0 $((NW - 1))); do
  python3 monosyn.py batch $w $NW > $LOG/w$w.log 2>&1 &
done
wait
