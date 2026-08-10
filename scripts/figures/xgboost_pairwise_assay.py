import os
import numpy as np
import pandas as pd
import pynapple as nap
from spatial_manifolds.mlencoding import *
from spatial_manifolds.detect_grids import *
from argparse import ArgumentParser
import warnings
warnings.filterwarnings('ignore')


"""
XGBoost Pairwise Cell Assay — VR and OF1
------------------------------------------------------------------
For each target cell, tests every other cell individually as a single
covariate on top of all behavioral baselines (plus a 'null' baseline
with no behavioral information). Handles both VR and OF1 sessions via
--session_type.

The null baseline uses a constant column (ones) so pR² reflects only
what the single covariate cell contributes above noise.

Output columns:
    mouse, day, session_type,
    target_cluster_id, target_type,
    covariate_cluster_id, covariate_type,
    baseline, pR2_cv
"""

use_parser = True

source_path = '/Users/harryclark/Downloads/COHORT12/'
data_path = '/Users/harryclark/Documents/data/'
fig_path = '/Users/harryclark/Documents/figs/FIGURE1/'
mouse = 25
day = 24
cell_start = 0
cell_end = 5
session_type = 'VR'

history_length = 1000
nfilters = int(history_length / time_bs)

if use_parser:
    parser = ArgumentParser()
    parser.add_argument('--mouse', type=int, required=True)
    parser.add_argument('--day', type=int, required=True)
    parser.add_argument('--data_path', type=str, required=True)
    parser.add_argument('--cell_start', type=int, required=True)
    parser.add_argument('--cell_end', type=int, required=True)
    parser.add_argument('--history_length', type=int, default=1000)
    parser.add_argument('--nfilters', type=int, default=None)
    parser.add_argument('--session_type', type=str, default='VR', choices=['VR', 'OF1'])
    args = parser.parse_args()

    mouse = args.mouse
    day = args.day
    data_path = args.data_path
    cell_start = args.cell_start
    cell_end = args.cell_end
    history_length = args.history_length
    nfilters = args.nfilters
    if nfilters is None:
        nfilters = int(history_length / time_bs)
    session_type = args.session_type
    source_path = '/exports/eddie/scratch/hclark3/COHORT12/'

print(f"Running XGBoost Pairwise Assay ({session_type}) for Mouse {mouse} Day {day} "
      f"cells {cell_start}-{cell_end-1} (history={history_length}ms, nfilters={nfilters})")

gcs, ngs, all = classify_cells_both_sessions(mouse, day, percentile_threshold=95, source_path=source_path)
g_m_ids, g_m_cluster_ids, _ = HDBSCAN_grid_modules(
    gcs, all, mouse, day, min_cluster_size=3, cluster_selection_epsilon=3,
    figpath=fig_path, curate_with_vr=True, curate_with_brain_region=True, source_path=source_path
)
g_m_cluster_ids = sorted(g_m_cluster_ids, key=len, reverse=True)


def get_cell_type(cell_id, gcs, ngs, g_m_cluster_ids):
    if cell_id in gcs.cluster_id.values:
        in_module = any(cell_id in module for module in g_m_cluster_ids)
        return 'cmGC' if in_module else 'ncmGC'
    elif cell_id in ngs.cluster_id.values:
        return 'NGS'
    return 'NS'


# ── Load session data ────────────────────────────────────────────────────────

if session_type == 'VR':
    tcs, tcs_time, _, last_ephys_bin, beh, clusters = compute_vr_tcs(
        mouse, day, apply_zscore=False, apply_guassian_filter=False, source_path=source_path
    )
    last_ephys_time_bin = clusters[clusters.index[0]].count(bin_size=time_bs, time_units='ms').index[-1]
    ep = nap.IntervalSet(start=0, end=last_ephys_time_bin, time_units='s')

    speed_in_time = np.array(beh['S'].bin_average(bin_size=time_bs, time_units='ms', ep=ep))
    dt_in_time = np.array(beh['travel'].bin_average(bin_size=time_bs, time_units='ms', ep=ep)
                          - ((beh['trial_number'][0] - 1) * tl))
    pos_in_time = dt_in_time % tl

    if np.any(np.isnan(pos_in_time)):
        dt_in_time = np.array(pd.Series(dt_in_time).ffill().bfill())
        pos_in_time = dt_in_time % tl
    if np.any(np.isnan(speed_in_time)):
        speed_in_time = np.array(pd.Series(speed_in_time).ffill().bfill())

    def make_baselines(T, theta):
        pos   = pos_in_time[:T]
        speed = speed_in_time[:T]
        if len(pos)   < T: pos   = np.pad(pos,   (0, T - len(pos)),   mode='constant')
        if len(speed) < T: speed = np.pad(speed, (0, T - len(speed)), mode='constant')
        names = ['null', 'pos', 'speed', 'lfp', 'pos_speed', 'pos_lfp', 'speed_lfp', 'pos_speed_lfp']
        xs = [
            np.zeros((T, 1)),
            pos[:, None],
            speed[:, None],
            theta[:, None],
            np.column_stack((pos, speed)),
            np.column_stack((pos, theta)),
            np.column_stack((speed, theta)),
            np.column_stack((pos, speed, theta)),
        ]
        return names, xs

else:  # OF1
    tcs, tcs_time, beh_OF, clusters_OF, ep = compute_of_tcs(
        mouse, day, apply_zscore=False, apply_guassian_filter=False,
        source_path=source_path, session='OF1'
    )
    last_ephys_time_bin = clusters_OF[clusters_OF.index[0]].count(bin_size=time_bs, time_units='ms').index[-1]
    ep = nap.IntervalSet(start=0, end=last_ephys_time_bin, time_units='s')

    pos_x_in_time = np.array(beh_OF['P_x'].bin_average(bin_size=time_bs, time_units='ms', ep=ep))
    pos_y_in_time = np.array(beh_OF['P_y'].bin_average(bin_size=time_bs, time_units='ms', ep=ep))
    speed_in_time = np.array(beh_OF['S'].bin_average(bin_size=time_bs, time_units='ms', ep=ep))
    hd_in_time    = np.array(beh_OF['H'].bin_average(bin_size=time_bs, time_units='ms', ep=ep))
    hing_in_time  = np.array(beh_OF['Hing'].bin_average(bin_size=time_bs, time_units='ms', ep=ep))

    for arr_name, arr in [('pos_x', pos_x_in_time), ('pos_y', pos_y_in_time),
                           ('speed', speed_in_time), ('hd', hd_in_time), ('hing', hing_in_time)]:
        if np.any(np.isnan(arr)):
            arr[:] = np.array(pd.Series(arr).ffill().bfill())

    def make_baselines(T, theta):
        px = pos_x_in_time[:T]; py = pos_y_in_time[:T]
        sp = speed_in_time[:T]; hd = hd_in_time[:T]; hing = hing_in_time[:T]
        for a in [px, py, sp, hd, hing]:
            if len(a) < T: a = np.pad(a, (0, T - len(a)), mode='constant')
        names = [
            'null', 'pos', 'speed', 'hd', 'hing', 'lfp',
            'pos_speed', 'pos_hd', 'pos_hing', 'pos_lfp',
            'pos_speed_hd', 'pos_speed_hing', 'pos_speed_lfp',
            'pos_hd_hing', 'pos_speed_hd_hing', 'pos_speed_hd_hing_lfp',
        ]
        xs = [
            np.zeros((T, 1)),
            np.column_stack((px, py)),
            sp[:, None],
            hd[:, None],
            hing[:, None],
            theta[:, None],
            np.column_stack((px, py, sp)),
            np.column_stack((px, py, hd)),
            np.column_stack((px, py, hing)),
            np.column_stack((px, py, theta)),
            np.column_stack((px, py, sp, hd)),
            np.column_stack((px, py, sp, hing)),
            np.column_stack((px, py, sp, theta)),
            np.column_stack((px, py, hd, hing)),
            np.column_stack((px, py, sp, hd, hing)),
            np.column_stack((px, py, sp, hd, hing, theta)),
        ]
        return names, xs


xgb_history = MLencoding(tunemodel='xgboost', cov_history=True, spike_history=False,
                         window=time_bs, n_filters=nfilters, max_time=history_length)

all_target_cell_ids = np.array(all.cluster_id.values)
if cell_end is None:
    cell_end = len(all_target_cell_ids)
target_cells_batch = all_target_cell_ids[cell_start:cell_end]

def _cell_out(tid):
    return f'{data_path}/xgboost_pairwise_M{mouse}_D{day}_h{history_length}_{session_type}_C{int(tid)}.csv'

for idx, target_id in enumerate(target_cells_batch):
    out_path = _cell_out(target_id)
    if os.path.exists(out_path):
        print(f'Target cell {target_id} already done — skipping ({out_path})')
        continue
    print(f'Processing target cell {target_id} ({cell_start + idx + 1}/{cell_end})')
    results_rows = []

    y = np.array(tcs_time[target_id])
    T = len(y)
    target_type = get_cell_type(target_id, gcs, ngs, g_m_cluster_ids)

    try:
        theta = get_theta_trace(
            mouse=mouse, day=day, cluster_id=target_id,
            time_bs=50, resample_bs=time_bs,
            session_type=session_type, source_path=source_path,
        )
        theta = np.array(theta)
        if len(theta) < T:
            theta = np.pad(theta, (0, T - len(theta)), mode='constant')
        else:
            theta = theta[:T]
    except Exception as e:
        print(f"  Could not load theta for {target_id}: {e}. Using zeros.")
        theta = np.zeros(T)

    baseline_names, baseline_xs = make_baselines(T, theta)

    # ── Baseline-only fits (no covariate cell) ───────────────────────────────
    for b_name, x_b in zip(baseline_names, baseline_xs):
        _, pR2_cv = xgb_history.fit_cv(x_b, y, verbose=0, continuous_folds=True)
        pR2 = float(np.nanmean(pR2_cv))
        print(f'  baseline [{b_name}] pR2={pR2:.4f}')
        results_rows.append(dict(
            mouse=mouse, day=day, session_type=session_type,
            target_cluster_id=int(target_id), target_type=target_type,
            covariate_cluster_id=-1, covariate_type='none',
            baseline=b_name, pR2_cv=pR2,
        ))

    # ── Pairwise: each other cell as a single covariate ──────────────────────
    other_cell_ids = all_target_cell_ids[all_target_cell_ids != target_id]

    for cov_id in other_cell_ids:
        if cov_id not in tcs_time:
            continue

        cov_spike = np.array(tcs_time[cov_id])
        cov_spike = cov_spike[:T]
        if len(cov_spike) < T:
            cov_spike = np.pad(cov_spike, (0, T - len(cov_spike)), mode='constant')

        cov_type = get_cell_type(int(cov_id), gcs, ngs, g_m_cluster_ids)

        for b_name, x_b in zip(baseline_names, baseline_xs):
            x = np.column_stack((x_b, cov_spike[:, None]))
            _, pR2_cv = xgb_history.fit_cv(x, y, verbose=0, continuous_folds=True)
            pR2 = float(np.nanmean(pR2_cv))
            print(f'  [{b_name}] cov={cov_id}({cov_type}) pR2={pR2:.4f}')
            results_rows.append(dict(
                mouse=mouse, day=day, session_type=session_type,
                target_cluster_id=int(target_id), target_type=target_type,
                covariate_cluster_id=int(cov_id), covariate_type=cov_type,
                baseline=b_name, pR2_cv=pR2,
            ))

    _tmp = out_path + '.tmp'
    pd.DataFrame(results_rows).to_csv(_tmp, index=False)
    os.replace(_tmp, out_path)   # atomic: file only appears when fully written
    print(f'  saved {len(results_rows)} rows -> {out_path}')
print(f'Saved {len(results_df)} rows to {csv_path}')
