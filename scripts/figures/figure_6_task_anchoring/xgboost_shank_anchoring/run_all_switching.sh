#!/usr/bin/env bash
# Run the shank-reconstruction assay for every MULTI-SHANK anchoring-switching session
# (ENTm cells on >=2 shanks — single-shank sessions only give a same-shank prediction).
# 1) identify_switching_sessions.py must have written switching_sessions.csv first.
# 2) each session is independent -> ideal for an eddie array job (one task per row).
#
# Usage:
#   ./run_all_switching.sh                      # all multi-shank switchers, sequential
#   ./run_all_switching.sh /path/to/cache       # custom cache dir
set -euo pipefail

PY="${PY:-/opt/anaconda3/envs/sm/bin/python}"
HERE="$(cd "$(dirname "$0")" && pwd)"
DATA_PATH="${1:-/Users/harryclark/Documents/spatial-manifolds/data/xgboost_shank_anchoring}"
SOURCE_PATH="${SOURCE_PATH:-/Users/harryclark/Downloads/COHORT12/}"
CSV="$DATA_PATH/switching_sessions.csv"
LOCS="${SOURCE_PATH}all_cluster_brain_locations_chris.csv"

if [[ ! -f "$CSV" ]]; then
  echo "Missing $CSV — run identify_switching_sessions.py first."; exit 1
fi

# passing sessions with ENTm on >=2 shanks
"$PY" - "$CSV" "$LOCS" <<'PYEOF' | while read -r mouse day; do
import csv, sys, pandas as pd
loc = pd.read_csv(sys.argv[2])
loc = loc[loc['brain_region'].astype(str).str.startswith('ENTm')]
with open(sys.argv[1]) as f:
    for r in csv.DictReader(f):
        if str(r.get('session_passes')).strip().lower() != 'true':
            continue
        m, d = int(r['mouse']), int(r['day'])
        n_shanks = loc[(loc['mouse'] == m) & (loc['day'] == d)]['shank_id'].dropna().nunique()
        if n_shanks >= 2:
            print(m, d)
PYEOF
  PKL="$DATA_PATH/M${mouse}D${day}_shank_reconstruction.pkl"
  if [[ -f "$PKL" ]]; then
    echo "=== M${mouse} D${day} — already cached, skipping ==="
    continue
  fi
  echo "=== M${mouse} D${day} ==="
  "$PY" "$HERE/run_shank_reconstruction.py" --mouse "$mouse" --day "$day" \
        --data_path "$DATA_PATH" --n_cov 16 --n_cv 10 --n_filters 5 --history_length 1000
done

echo "All switching sessions done."
