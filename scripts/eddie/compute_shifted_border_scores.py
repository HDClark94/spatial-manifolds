"""
compute_shifted_border_scores.py
─────────────────────────────────
Compute shifted border scores for one OF session (OF1 or OF2) for a single
mouse × day. Outputs a flat parquet file with one row per cluster × travel lag.

Usage on Eddie:
    uv run compute_shifted_border_scores.py \
        --mouse 29 --day 23 --session OF1 \
        --source_path /exports/eddie/scratch/hclark3/COHORT12/ \
        --output_path /exports/eddie/scratch/hclark3/data/border_scores_of/

Output schema (one row per cluster × travel):
    cluster_id, brain_region, border_score, travel, sig,
    null_border_score  (list[float] at travel==0, else None)
"""

import os
import warnings
import numpy as np
import pandas as pd
import pynapple as nap
from argparse import ArgumentParser

warnings.filterwarnings('ignore')

from spatial_manifolds.predictive_grid import compute_travel_projected
from spatial_manifolds.detect_grids import curate_clusters
from spatial_manifolds.tuning_scores.border_score import (
    border_score_from_rate_map, compute_rate_map,
)

# ── Constants matching the notebook ───────────────────────────────────────────
TRAVEL_VALUES = np.arange(-50, 50, 1)   # -50 … +49 cm, 100 values
N_SHUFFLES    = 200
N_BINS        = 40
SIGMA         = 2.0
ARENA_MIN     = 0.0
ARENA_MAX     = 100.0
ALPHA         = 0.05                    # → 95th percentile of null

# ── CLI ───────────────────────────────────────────────────────────────────────
parser = ArgumentParser()
parser.add_argument('--mouse',       type=int,  required=True)
parser.add_argument('--day',         type=int,  required=True)
parser.add_argument('--session',     type=str,  required=True,
                    choices=['OF1', 'OF2'])
parser.add_argument('--source_path', type=str,  required=True,
                    help='Root of COHORT12 directory on Eddie scratch')
parser.add_argument('--output_path', type=str,  required=True,
                    help='Flat directory where the parquet will be saved')
args = parser.parse_args()

mouse       = args.mouse
day         = args.day
session     = args.session
source_path = args.source_path.rstrip('/') + '/'
output_path = args.output_path.rstrip('/') + '/'

os.makedirs(output_path, exist_ok=True)

out_file = output_path + f'shifted_border_score_M{mouse}_D{day:02}_{session}.parquet'

if os.path.exists(out_file):
    print(f'SKIP  M{mouse} D{day} {session} — output already exists: {out_file}')
    raise SystemExit(0)

# ── Paths ─────────────────────────────────────────────────────────────────────
session_folder = f'{source_path}M{mouse}/D{day:02}/{session}/'
spikes_path    = session_folder + f'sub-{mouse}_day-{day:02}_ses-{session}_srt-kilosort4_clusters.npz'
beh_path       = session_folder + f'sub-{mouse}_day-{day:02}_ses-{session}_beh.nwb'
brain_loc_path = source_path + 'all_cluster_brain_locations_chris.csv'

for p in [spikes_path, beh_path]:
    if not os.path.exists(p):
        print(f'MISS  M{mouse} D{day} {session} — file not found: {p}')
        raise SystemExit(1)

# ── Brain region lookup ───────────────────────────────────────────────────────
brain_locations = pd.read_csv(brain_loc_path) if os.path.exists(brain_loc_path) else None

def get_brain_region(cid):
    if brain_locations is None:
        return 'unknown'
    row = brain_locations[
        (brain_locations['mouse']      == mouse) &
        (brain_locations['day']        == day)   &
        (brain_locations['cluster_id'] == cid)
    ]
    return row['brain_region'].iloc[0] if len(row) else 'unknown'

# ── Load data ─────────────────────────────────────────────────────────────────
print(f'RUN   M{mouse} D{day} {session}')
beh      = nap.load_file(beh_path)
clusters = curate_clusters(nap.load_file(spikes_path))

pos_times = np.array(beh['P_x'].times())
pos_x_raw = np.array(beh['P_x'])
pos_y_raw = np.array(beh['P_y'])
position  = np.stack([beh['P_x'], beh['P_y']], axis=1)  # TsdFrame for travel projection

last_bin  = clusters[clusters.index[0]].count(bin_size=10, time_units='ms').index[-1]
ep        = nap.IntervalSet(start=0, end=last_bin, time_units='s')

cluster_ids = list(clusters.index)
print(f'  {len(cluster_ids)} cells')

rows = []

for cid in cluster_ids:
    spike_times  = np.array(clusters[cid].restrict(ep).t)
    brain_region = get_brain_region(cid)

    # ── Null distribution at travel = 0 (200 circular-shift shuffles) ─────────
    null_scores = []
    for _ in range(N_SHUFFLES):
        shuffled = np.array(
            nap.shift_timestamps(clusters[cid].restrict(ep), min_shift=20.0).t
        )
        rm_sh = compute_rate_map(
            shuffled, pos_x_raw, pos_y_raw, pos_times,
            n_bins=N_BINS, arena_min=ARENA_MIN, arena_max=ARENA_MAX, sigma=SIGMA,
        )
        null_scores.append(border_score_from_rate_map(rm_sh))

    null_arr   = np.array(null_scores, dtype=float)
    sig_thresh = np.nanpercentile(null_arr, 100 * (1 - ALPHA))

    # ── Border score at each travel lag ───────────────────────────────────────
    for travel in TRAVEL_VALUES:
        if travel == 0:
            px_t = pos_x_raw
            py_t = pos_y_raw
            pt   = pos_times
        else:
            beh_lag   = compute_travel_projected(['P_x', 'P_y'], position, position, float(travel))
            px_t      = np.array(beh_lag['P_x'])
            py_t      = np.array(beh_lag['P_y'])
            pt        = np.array(beh_lag['P_x'].times())

        rm = compute_rate_map(
            spike_times, px_t, py_t, pt,
            n_bins=N_BINS, arena_min=ARENA_MIN, arena_max=ARENA_MAX, sigma=SIGMA,
        )
        bs = border_score_from_rate_map(rm)

        rows.append(dict(
            cluster_id        = int(cid),
            brain_region      = brain_region,
            border_score      = float(bs) if np.isfinite(bs) else float('nan'),
            travel            = int(travel),
            sig               = bool(np.isfinite(bs) and bs > sig_thresh),
            null_border_score = null_arr.tolist() if travel == 0 else None,
        ))

df = pd.DataFrame(rows)
df.to_parquet(out_file, index=False)

n_sig = df[(df['travel'] == 0) & df['sig']]['cluster_id'].nunique()
print(f'  Saved: {out_file}')
print(f'  Border cells at lag 0: {n_sig}/{len(cluster_ids)}')
