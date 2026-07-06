"""
xgboost_parameter_validation.py
────────────────────────────────
Parameter validation script for XGBoost coordination assay.
Runs on a single representative session specified in run_eddie_master_script.py.

Two tests:

  Test 1 — GC vs GC sweep
    For each grid cell as target, all other GCs as covariates + position.
    Sweeps history_length × n_filters × {VR, OF1}.
    Also includes a no-history condition (cov_history=False).
    Output: xgboost_validation_gc_gc_{session}_M{mouse}_D{day}.csv

  Test 2 — Baseline sweep
    All behavioural baselines × history_length × n_filters.
    Averaged over GC target cells.
    Output: xgboost_validation_baselines_{session}_M{mouse}_D{day}.csv

Both outputs are plain CSVs designed for easy loading into the
history_length_assay notebook for sanity-check validation plots.
"""

import numpy as np
import pandas as pd
import pynapple as nap
from argparse import ArgumentParser
import warnings
warnings.filterwarnings('ignore')

from spatial_manifolds.detect_grids import *
from spatial_manifolds.mlencoding import MLencoding

# ── Parameter grids ───────────────────────────────────────────────────────────
HISTORY_LENGTHS = [50, 100, 500, 1000]   # ms
NFILTERS_RANGE  = list(range(1, 21))     # 1 → 20
N_CV            = 5

# ── Defaults (overridden by CLI) ──────────────────────────────────────────────
use_parser   = True
source_path  = '/Users/harryclark/Downloads/COHORT12/'
data_path    = '/exports/eddie/scratch/hclark3/data/xgboost_validation/'
mouse        = 29
day          = 23
session_type = 'VR'   # 'VR' or 'OF1'

if use_parser:
    parser = ArgumentParser()
    parser.add_argument('--mouse',        type=int, required=True)
    parser.add_argument('--day',          type=int, required=True)
    parser.add_argument('--data_path',    type=str, required=True)
    parser.add_argument('--session_type', type=str, default='VR',
                        choices=['VR', 'OF1'])
    args = parser.parse_args()
    mouse        = args.mouse
    day          = args.day
    data_path    = args.data_path
    session_type = args.session_type
    source_path  = '/exports/eddie/scratch/hclark3/COHORT12/'

print(f'Parameter validation  M{mouse} D{day}  session={session_type}')
print(f'History lengths: {HISTORY_LENGTHS} ms + no-history')
print(f'n_filters sweep: {NFILTERS_RANGE}')

# ── Load session data ─────────────────────────────────────────────────────────
gcs, ngs, all_cells = classify_cells_both_sessions(
    mouse, day, percentile_threshold=95, source_path=source_path)
gc_ids = gcs.cluster_id.values.astype(int)
print(f'GC cells: {len(gc_ids)}')

fig_path = '/exports/eddie/scratch/hclark3/data/'   # unused, just for reference

if session_type == 'VR':
    tcs, tcs_time, _, last_ephys_bin, beh, clusters = compute_vr_tcs(
        mouse, day, apply_zscore=False, apply_guassian_filter=False,
        source_path=source_path)
    last_t = clusters[clusters.index[0]].count(
        bin_size=time_bs, time_units='ms').index[-1]
    ep = nap.IntervalSet(start=0, end=last_t, time_units='s')

    # Position and speed
    dt  = np.array(beh['travel'].bin_average(bin_size=time_bs, time_units='ms', ep=ep)
                   - ((beh['trial_number'][0] - 1) * tl))
    if np.any(np.isnan(dt)):
        dt = pd.Series(dt).ffill().bfill().values
    pos_vr = (dt % tl)
    spd_vr = np.array(pd.Series(
        np.array(beh['S'].bin_average(bin_size=time_bs, time_units='ms', ep=ep))
    ).ffill().bfill())

    # LFP for target cell (loaded per cell inside loop)
    def get_lfp(cid):
        try:
            return np.array(get_theta_trace(
                mouse=mouse, day=day, cluster_id=cid,
                time_bs=50, resample_bs=time_bs,
                session_type='VR', source_path=source_path))
        except Exception:
            return None

    VR_BASELINES = {
        'null':          lambda T, pos, spd, lfp: np.zeros((T, 1)),
        'pos':           lambda T, pos, spd, lfp: pos[:T, None],
        'speed':         lambda T, pos, spd, lfp: spd[:T, None],
        'lfp':           lambda T, pos, spd, lfp: (lfp[:T, None] if lfp is not None
                                                   else np.zeros((T, 1))),
        'pos_speed':     lambda T, pos, spd, lfp: np.column_stack([pos[:T], spd[:T]]),
        'pos_lfp':       lambda T, pos, spd, lfp: np.column_stack(
                             [pos[:T], lfp[:T] if lfp is not None else np.zeros(T)]),
        'speed_lfp':     lambda T, pos, spd, lfp: np.column_stack(
                             [spd[:T], lfp[:T] if lfp is not None else np.zeros(T)]),
        'pos_speed_lfp': lambda T, pos, spd, lfp: np.column_stack(
                             [pos[:T], spd[:T],
                              lfp[:T] if lfp is not None else np.zeros(T)]),
    }
    beh_signals = (pos_vr, spd_vr)

else:  # OF1
    tcs, tcs_time, beh_of, clusters_of, ep = compute_of_tcs(
        mouse, day, apply_zscore=False, apply_guassian_filter=False,
        source_path=source_path, session='OF1')
    last_t = clusters_of[clusters_of.index[0]].count(
        bin_size=time_bs, time_units='ms').index[-1]

    def _b(key):
        a = np.array(beh_of[key].bin_average(bin_size=time_bs, time_units='ms', ep=ep))
        return pd.Series(a).ffill().bfill().values

    px_of = _b('head_x'); py_of = _b('head_y')
    spd_of = _b('S');     hd_of = _b('H');    hing_of = _b('Hing')

    def get_lfp(cid):
        try:
            return np.array(get_theta_trace(
                mouse=mouse, day=day, cluster_id=cid,
                time_bs=50, resample_bs=time_bs,
                session_type='OF1', source_path=source_path))
        except Exception:
            return None

    OF_BASELINES = {
        'null':                 lambda T, lfp: np.zeros((T, 1)),
        'pos':                  lambda T, lfp: np.column_stack([px_of[:T], py_of[:T]]),
        'speed':                lambda T, lfp: spd_of[:T, None],
        'hd':                   lambda T, lfp: hd_of[:T, None],
        'hing':                 lambda T, lfp: hing_of[:T, None],
        'lfp':                  lambda T, lfp: (lfp[:T, None] if lfp is not None
                                                else np.zeros((T, 1))),
        'pos_speed':            lambda T, lfp: np.column_stack([px_of[:T], py_of[:T],
                                                                 spd_of[:T]]),
        'pos_hd':               lambda T, lfp: np.column_stack([px_of[:T], py_of[:T],
                                                                 hd_of[:T]]),
        'pos_speed_hd_hing_lfp': lambda T, lfp: np.column_stack([
            px_of[:T], py_of[:T], spd_of[:T], hd_of[:T], hing_of[:T],
            lfp[:T] if lfp is not None else np.zeros(T)]),
    }

BASELINES = VR_BASELINES if session_type == 'VR' else OF_BASELINES

# ── Filter GC IDs to those present in tcs_time ───────────────────────────────
gc_ids = [cid for cid in gc_ids if cid in tcs_time]
print(f'GC cells with spike data: {len(gc_ids)}')

# ── Helper: pad array to length T ─────────────────────────────────────────────
def _pad(arr, T):
    arr = np.array(arr)[:T]
    return np.pad(arr, (0, max(0, T - len(arr))))


# ════════════════════════════════════════════════════════════════════════════════
# TEST 1: GC vs GC parameter sweep
# ════════════════════════════════════════════════════════════════════════════════
print('\n── Test 1: GC vs GC parameter sweep ──')
rows_gc = []

# No-history condition
xgb_nohist = MLencoding(tunemodel='xgboost', cov_history=False, spike_history=False,
                         window=time_bs, n_filters=1, max_time=time_bs)

for ti, target_id in enumerate(gc_ids):
    y = np.array(tcs_time[target_id])
    T = len(y)
    lfp = get_lfp(target_id)
    if lfp is not None:
        lfp = _pad(lfp, T)

    # Covariate matrix: all other GC cells ONLY (no position)
    # Baseline is null (zeros) — measures raw spike-timing coordination above chance
    cov_ids = [c for c in gc_ids if c != target_id]
    cov_mat = np.vstack([_pad(np.array(tcs_time[c]), T) for c in cov_ids]).T
    x_null  = np.zeros((T, 1))
    x_full  = cov_mat  # GC spike histories only

    print(f'  Target GC {target_id} ({ti+1}/{len(gc_ids)})  '
          f'n_cov_gc={len(cov_ids)}  T={T}', flush=True)

    # ── No-history condition (current time bin only) ──────────────────────────
    _, pr2_null_nh = xgb_nohist.fit_cv(x_null, y, verbose=0,
                                        continuous_folds=True, n_cv=N_CV)
    _, pr2_full_nh = xgb_nohist.fit_cv(x_full, y, verbose=0,
                                        continuous_folds=True, n_cv=N_CV)
    rows_gc.append(dict(
        mouse=mouse, day=day, session=session_type,
        target_id=target_id, history_ms=0, n_filters=1,
        cov_history=False,
        pr2_null=float(np.nanmean(pr2_null_nh)),
        pr2_gc=float(np.nanmean(pr2_full_nh)),
        delta_pr2=float(np.nanmean(pr2_full_nh)) - float(np.nanmean(pr2_null_nh)),
    ))

    # ── History × n_filters sweep ─────────────────────────────────────────────
    for hl in HISTORY_LENGTHS:
        # Fit position-only baseline once per (target, history, nf_max)
        # Use nf=NFILTERS_RANGE[-1] for baseline (doesn't matter since no cells)
        for nf in NFILTERS_RANGE:
            xgb = MLencoding(tunemodel='xgboost', cov_history=True, spike_history=False,
                              window=time_bs, n_filters=nf, max_time=hl)

            _, pr2_null = xgb.fit_cv(x_null, y, verbose=0,
                                     continuous_folds=True, n_cv=N_CV)
            _, pr2_gc   = xgb.fit_cv(x_full, y, verbose=0,
                                     continuous_folds=True, n_cv=N_CV)

            rows_gc.append(dict(
                mouse=mouse, day=day, session=session_type,
                target_id=target_id, history_ms=hl, n_filters=nf,
                cov_history=True,
                pr2_null=float(np.nanmean(pr2_null)),
                pr2_gc=float(np.nanmean(pr2_gc)),
                delta_pr2=float(np.nanmean(pr2_gc)) - float(np.nanmean(pr2_null)),
            ))

df_gc = pd.DataFrame(rows_gc)
out_gc = (f'{data_path}xgboost_validation_gc_gc_{session_type}'
          f'_M{mouse}_D{day}.csv')
df_gc.to_csv(out_gc, index=False)
print(f'Saved Test 1 → {out_gc}  ({len(df_gc)} rows)')


# ════════════════════════════════════════════════════════════════════════════════
# TEST 2: Baseline sweep × parameter grid
# ════════════════════════════════════════════════════════════════════════════════
print('\n── Test 2: Baseline sweep ──')
rows_bl = []

# Use all GC cells as targets (or cap at first 10 for speed)
MAX_TARGETS_BL = min(len(gc_ids), 10)
bl_targets = gc_ids[:MAX_TARGETS_BL]

for ti, target_id in enumerate(bl_targets):
    y   = np.array(tcs_time[target_id])
    T   = len(y)
    lfp = get_lfp(target_id)
    if lfp is not None:
        lfp = _pad(lfp, T)

    print(f'  Baseline target {target_id} ({ti+1}/{MAX_TARGETS_BL})', flush=True)

    # No-history condition across all baselines
    xgb_nh = MLencoding(tunemodel='xgboost', cov_history=False, spike_history=False,
                         window=time_bs, n_filters=1, max_time=time_bs)
    for bl_name, bl_fn in BASELINES.items():
        if session_type == 'VR':
            x_b = bl_fn(T, pos_vr, spd_vr, lfp)
        else:
            x_b = bl_fn(T, lfp)
        _, pr2 = xgb_nh.fit_cv(x_b, y, verbose=0, continuous_folds=True, n_cv=N_CV)
        rows_bl.append(dict(
            mouse=mouse, day=day, session=session_type,
            target_id=target_id, baseline=bl_name,
            history_ms=0, n_filters=1, cov_history=False,
            pr2=float(np.nanmean(pr2)),
        ))

    # History × n_filters × baselines
    for hl in HISTORY_LENGTHS:
        for nf in NFILTERS_RANGE:
            xgb = MLencoding(tunemodel='xgboost', cov_history=True, spike_history=False,
                              window=time_bs, n_filters=nf, max_time=hl)
            for bl_name, bl_fn in BASELINES.items():
                if session_type == 'VR':
                    x_b = bl_fn(T, pos_vr, spd_vr, lfp)
                else:
                    x_b = bl_fn(T, lfp)
                _, pr2 = xgb.fit_cv(x_b, y, verbose=0,
                                    continuous_folds=True, n_cv=N_CV)
                rows_bl.append(dict(
                    mouse=mouse, day=day, session=session_type,
                    target_id=target_id, baseline=bl_name,
                    history_ms=hl, n_filters=nf, cov_history=True,
                    pr2=float(np.nanmean(pr2)),
                ))

df_bl = pd.DataFrame(rows_bl)
out_bl = (f'{data_path}xgboost_validation_baselines_{session_type}'
          f'_M{mouse}_D{day}.csv')
df_bl.to_csv(out_bl, index=False)
print(f'Saved Test 2 → {out_bl}  ({len(df_bl)} rows)')

print('\nDone.')
