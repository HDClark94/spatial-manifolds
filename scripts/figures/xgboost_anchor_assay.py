import numpy as np
import pandas as pd
import pynapple as nap
from scipy.ndimage import gaussian_filter
from spatial_manifolds.mlencoding import *
from spatial_manifolds.detect_grids import *
from argparse import ArgumentParser
import warnings
import os

warnings.filterwarnings('ignore')

"""
XGBoost Anchor Assay — anchoring-mode conditional ΔpR² (batched cell runs)
---------------------------------------------------------------------------
For every target cell (GC or NGS) in a given session, this script computes
conditional pseudo-R² separately for trials in which the cell is task-anchored
versus non-anchored, using two covariate conditions:

    pos              — position alone (baseline)
    pos+ngs_shank    — position + all NGS cells on a given shank

The primary output metric is:
    ΔpR² = pR²(pos+ngs_shank) − pR²(pos)

evaluated separately within anchored and non-anchored trial subsets.

Anchored/non-anchored labels are derived from the spectral analysis of the
target cell's own trial-by-trial tuning curve:
    1. Compute spectrogram via spectral_analysis.
    2. Find peak-frequency bins at task-locked harmonics [12, 28, 44, 60, 76].
    3. Assign binary trial labels via get_kmeans_spatial_labels.
    4. Map trial labels to time bins.

Resulting DataFrame columns:
    mouse, day                — session identifiers
    target_cluster_id         — cluster ID of the target cell
    target_shank_id           — shank of the target cell (0–3)
    target_cell_type          — 'GC' or 'NG'
    cov_shank_id              — shank of the NGS covariate pool
    n_ngs_cells               — number of NGS cells used as covariates
    covariate_type            — 'pos+ngs_shank'
    anchor_mode               — 'anchored' or 'non_anchored'
    prop_anchor_mode          — fraction of time bins in this mode
    pR2_cond                  — conditional pR² (pos+ngs_shank model)
    pR2_pos_cond              — conditional pR² (pos-only baseline)
    delta_pR2                 — pR2_cond − pR2_pos_cond

Run this script multiple times with different --cell_start/--cell_end ranges
to cover all target cells in a session in parallel on the cluster.
"""

# ── Default values (overridden by argparse on HPC) ───────────────────────────
use_parser = True

source_path      = '/Users/harryclark/Downloads/COHORT12/'
data_path        = '/Users/harryclark/Documents/spatial-manifolds/data/xgboost_anchor_assay/'
cell_class_path  = '/Users/harryclark/Documents/spatial-manifolds/data/cell_classifications.csv'
mouse      = 25
day        = 24
cell_start = 0
cell_end   = 20

if use_parser:
    parser = ArgumentParser()
    parser.add_argument('--mouse',          type=int, required=True,  help='Mouse ID')
    parser.add_argument('--day',            type=int, required=True,  help='Day of recording')
    parser.add_argument('--data_path',      type=str, required=True,  help='Output directory for results')
    parser.add_argument('--cell_class_path',type=str, required=True,  help='Path to cell_classifications.csv')
    parser.add_argument('--cell_start',     type=int, required=True,  help='Start index of target cells (inclusive)')
    parser.add_argument('--cell_end',       type=int, required=True,  help='End index of target cells (exclusive)')
    args = parser.parse_args()

    mouse           = args.mouse
    day             = args.day
    data_path       = args.data_path
    cell_class_path = args.cell_class_path
    cell_start      = args.cell_start
    cell_end        = args.cell_end
    source_path     = '/exports/eddie/scratch/hclark3/COHORT12/'

print(f"Running Anchor Assay for Mouse {mouse} Day {day:02d} on cells {cell_start}–{cell_end - 1}")

# ── Config ────────────────────────────────────────────────────────────────────
MIN_ANCHOR_TRIALS = 10        # minimum time-bins required per anchoring mode
PEAK_INDICES      = [12, 28, 44, 60, 76]  # task-locked harmonic indices
nfilters          = 5
history_length    = 1000      # ms

# ── Output paths ──────────────────────────────────────────────────────────────
os.makedirs(data_path, exist_ok=True)
anchor_cache_dir = os.path.join(data_path, 'gc_ngs_shank_xgboost_anchor_cache')
os.makedirs(anchor_cache_dir, exist_ok=True)

# ── Load cell classifications ─────────────────────────────────────────────────
cell_classifications = pd.read_csv(cell_class_path)

_anchor_types = ['GC', 'NG']

# ── Build valid sessions (at least 1 GC and 1 NGS cell) ──────────────────────
gc_sessions  = cell_classifications[cell_classifications['cell_type'] == 'GC'][['mouse', 'day']].drop_duplicates()
ngs_sessions = cell_classifications[cell_classifications['cell_type'] == 'NG'][['mouse', 'day']].drop_duplicates()
valid_sessions = gc_sessions.merge(ngs_sessions, on=['mouse', 'day'])

if not ((valid_sessions['mouse'] == mouse) & (valid_sessions['day'] == day)).any():
    print(f"M{mouse} D{day:02d} is not a valid session (requires ≥1 GC and ≥1 NGS cell). Exiting.")
    exit(0)

# ── XGBoost model ─────────────────────────────────────────────────────────────
xgb_history = MLencoding(
    tunemodel='xgboost',
    cov_history=True,
    spike_history=False,
    window=time_bs,
    n_filters=nfilters,
    max_time=history_length,
)

# ── Anchoring-label helper ────────────────────────────────────────────────────
def _compute_anchored_labels(target_id, tcs, last_ephys_bin):
    """Return per-trial task-anchored labels (1=anchored, 0=non-anchored)
    computed from the target cell's own trial-by-trial tuning curve."""
    if target_id not in tcs:
        return None
    tc = gaussian_filter(np.nan_to_num(tcs[target_id]).astype(np.float64), sigma=2.5)
    tc = tc[:last_ephys_bin]
    tcs_to_use = {target_id: tc, 1000: tc}
    results    = spectral_analysis(tcs_to_use, tl, bs=bs)
    S          = results[3].mean(0)
    max_peaks  = np.argmax(S, axis=0)
    labels_idx = np.isin(max_peaks, PEAK_INDICES).astype(int)
    return get_kmeans_spatial_labels(tc, labels_idx, bs=bs, tl=tl)

# ── Target cells for this session ────────────────────────────────────────────
sess_cells = cell_classifications[
    (cell_classifications['mouse']     == mouse) &
    (cell_classifications['day']       == day) &
    (cell_classifications['cell_type'].isin(_anchor_types))
].copy()

if len(sess_cells) == 0:
    print(f"No GC/NG target cells found for M{mouse} D{day:02d}. Exiting.")
    exit(0)

sess_cells['cluster_id'] = sess_cells['cluster_id'].astype(int)
sess_cells = reconstruct_shank_id(sess_cells, mouse, colname='probe_x')

# ── Apply batch slicing ───────────────────────────────────────────────────────
all_target_ids   = sess_cells['cluster_id'].values
target_ids_batch = all_target_ids[cell_start:cell_end]
sess_cells_batch = sess_cells[sess_cells['cluster_id'].isin(target_ids_batch)].copy()

if len(sess_cells_batch) == 0:
    print(f"No cells in batch [{cell_start}, {cell_end}). Exiting.")
    exit(0)

# ── Per-cell cache paths ──────────────────────────────────────────────────────
cache_files = {
    int(r['cluster_id']): os.path.join(
        anchor_cache_dir,
        f'M{mouse}_D{day:02d}_C{int(r["cluster_id"])}_anchor.csv'
    )
    for _, r in sess_cells_batch.iterrows()
}

# ── Session-level cache shortcut ──────────────────────────────────────────────
anchor_rows = []
if all(os.path.exists(f) for f in cache_files.values()):
    for cf in cache_files.values():
        anchor_rows.extend(pd.read_csv(cf).to_dict('records'))
    print(f"M{mouse} D{day:02d}: all batch targets cached — loading from disk.")
else:
    # ── Load VR spike trains and behaviour ────────────────────────────────────
    try:
        tcs, tcs_time, _, last_ephys_bin, beh, clusters = compute_vr_tcs(
            mouse, day,
            apply_zscore=False, apply_guassian_filter=False,
            source_path=source_path,
        )
    except Exception as e:
        print(f"Session load failed for M{mouse} D{day:02d}: {e}")
        exit(1)

    last_ephys_time_bin = clusters[clusters.index[0]].count(
        bin_size=time_bs, time_units='ms'
    ).index[-1]
    ep = nap.IntervalSet(start=0, end=last_ephys_time_bin, time_units='s')

    dt_in_time = np.array(
        beh['travel'].bin_average(bin_size=time_bs, time_units='ms', ep=ep)
        - ((beh['trial_number'][0] - 1) * tl)
    )
    pos_in_time_anchor = dt_in_time % tl
    if np.any(np.isnan(pos_in_time_anchor)):
        pos_in_time_anchor = pd.Series(dt_in_time).ffill().bfill().values % tl

    trial_number_in_time = (dt_in_time // tl) + beh['trial_number'][0]
    first_trial = int(beh['trial_number'][0])

    # ── NGS cells grouped by shank ────────────────────────────────────────────
    sess_ngs = cell_classifications[
        (cell_classifications['mouse']     == mouse) &
        (cell_classifications['day']       == day) &
        (cell_classifications['cell_type'] == 'NG')
    ].copy()
    sess_ngs['cluster_id'] = sess_ngs['cluster_id'].astype(int)
    sess_ngs = reconstruct_shank_id(sess_ngs, mouse, colname='probe_x')
    ngs_by_shank = {
        sh: grp['cluster_id'].values.astype(int)
        for sh, grp in sess_ngs.groupby('shank_id')
        if any(c in tcs_time for c in grp['cluster_id'].astype(int))
    }

    # ── Per-target loop ───────────────────────────────────────────────────────
    for _, target_row in sess_cells_batch.iterrows():
        target_id    = int(target_row['cluster_id'])
        target_shank = int(target_row['shank_id'])
        target_type  = str(target_row['cell_type'])
        cache_file   = cache_files[target_id]

        if os.path.exists(cache_file):
            anchor_rows.extend(pd.read_csv(cache_file).to_dict('records'))
            continue

        if target_id not in tcs_time:
            continue

        # ── Compute anchored labels ───────────────────────────────────────────
        try:
            task_anchored_labels = _compute_anchored_labels(target_id, tcs, last_ephys_bin)
        except Exception as e:
            print(f"  Anchoring failed for {target_id}: {e}")
            continue
        if task_anchored_labels is None:
            continue

        # Map trial labels → time bins
        n_label_trials = len(task_anchored_labels)
        tl_in_time = np.zeros(len(trial_number_in_time), dtype=int)
        for ti, tn in enumerate(trial_number_in_time.astype(int)):
            idx = tn - first_trial
            if 0 <= idx < n_label_trials:
                tl_in_time[ti] = int(task_anchored_labels[idx])

        y   = np.array(tcs_time[target_id])
        T   = len(y)
        pos = pos_in_time_anchor[:T]
        if len(pos) < T:
            pos = np.pad(pos, (0, T - len(pos)), mode='edge')
        tl_t = tl_in_time[:T]

        # ── Fit position-only baseline ────────────────────────────────────────
        Y_hat_pos, _ = xgb_history.fit_cv(pos[:, None], y, verbose=0, continuous_folds=True)

        cell_anchor_rows = []

        # ── One pos+NGS model per available shank ─────────────────────────────
        for cov_shank, ngs_cids in ngs_by_shank.items():
            valid_ngs = [c for c in ngs_cids if c in tcs_time and c != target_id]
            if not valid_ngs:
                continue

            ngs_mat = np.vstack([
                np.pad(
                    np.array(tcs_time[c])[:T],
                    (0, max(0, T - len(np.array(tcs_time[c])[:T]))),
                    mode='constant',
                )
                for c in valid_ngs
            ]).T   # shape (T, n_ngs)

            x_ngs    = np.column_stack([pos, ngs_mat])
            Y_hat_ngs, _ = xgb_history.fit_cv(x_ngs, y, verbose=0, continuous_folds=True)

            # ── Conditional pR² per anchoring mode ───────────────────────────
            for anchor_name, label_val in [('anchored', 1), ('non_anchored', 0)]:
                mask = tl_t == label_val
                prop = float(mask.sum()) / T

                if mask.sum() >= MIN_ANCHOR_TRIALS:
                    pR2_pos_cond = float(poisson_pseudoR2(
                        y[mask], Y_hat_pos[mask], np.nanmean(y[mask])
                    ))
                    pR2_ngs_cond = float(poisson_pseudoR2(
                        y[mask], Y_hat_ngs[mask], np.nanmean(y[mask])
                    ))
                else:
                    pR2_pos_cond = np.nan
                    pR2_ngs_cond = np.nan

                delta = (
                    pR2_ngs_cond - pR2_pos_cond
                    if not (np.isnan(pR2_ngs_cond) or np.isnan(pR2_pos_cond))
                    else np.nan
                )
                row = dict(
                    mouse=mouse, day=day,
                    target_cluster_id=target_id,
                    target_shank_id=target_shank,
                    target_cell_type=target_type,
                    cov_shank_id=int(cov_shank),
                    n_ngs_cells=len(valid_ngs),
                    covariate_type='pos+ngs_shank',
                    anchor_mode=anchor_name,
                    prop_anchor_mode=prop,
                    pR2_cond=pR2_ngs_cond,
                    pR2_pos_cond=pR2_pos_cond,
                    delta_pR2=delta,
                )
                cell_anchor_rows.append(row)
                anchor_rows.append(row)

        pd.DataFrame(cell_anchor_rows).to_csv(cache_file, index=False)
        print(f"  {target_type} {target_id} (sh{target_shank}): done")

# ── Save combined results for this batch ──────────────────────────────────────
anchor_df = pd.DataFrame(anchor_rows)
suffix    = f'_{cell_start}_{cell_end}'
csv_path  = os.path.join(data_path, f'xgboost_anchor_assay_M{mouse}_D{day:02d}{suffix}.csv')
anchor_df.to_csv(csv_path, index=False)
print(f"Saved {len(anchor_df)} rows → {csv_path}")
