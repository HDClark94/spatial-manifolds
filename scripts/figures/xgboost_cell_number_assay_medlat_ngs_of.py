import numpy as np
import pandas as pd
import pynapple as nap
from matplotlib import pyplot as plt
from spatial_manifolds.mlencoding import *
from spatial_manifolds.detect_grids import *
from argparse import ArgumentParser
import warnings
warnings.filterwarnings('ignore')


"""
XGBoost Cell Number Assay — Medial vs Lateral NGS Covariates, OF1
------------------------------------------------------------------
Fits XGBoost models to predict firing of target cells in Open Field 1,
using medial NGS cells (coord_SCs_x < 3400 um) and lateral NGS cells
(coord_SCs_x >= 3400 um) as separate covariate populations.
Position is 2D (x, y). Baselines use speed (S), head direction (H),
and head-in-goal angle (Hing).
"""

use_parser = True

source_path = '/Users/harryclark/Downloads/COHORT12/'
data_path = '/Users/harryclark/Documents/data/'
fig_path = '/Users/harryclark/Documents/figs/FIGURE1/'
mouse = 25
day = 24
cell_start = 60
cell_end = 80
medial_lateral_boundary = 3400
# xgboost parameters
nfilters = 5
history_length = 1000
max_cells = 10

if use_parser:
    parser = ArgumentParser()
    parser.add_argument('--mouse', type=int, required=True, help='Mouse ID')
    parser.add_argument('--day', type=int, required=True, help='Day of recording')
    parser.add_argument('--data_path', type=str, required=True, help='Path to data directory')
    parser.add_argument('--cell_start', type=int, required=True, help='Start index of target cells (inclusive)')
    parser.add_argument('--cell_end', type=int, required=True, help='End index of target cells (exclusive)')
    parser.add_argument('--history_length', type=int, default=1000, help='History length in ms')
    parser.add_argument('--nfilters', type=int, default=5, help='Number of history filters per covariate')
    parser.add_argument('--max_cells', type=int, default=10, help='Max number of covariate cells to add')
    args = parser.parse_args()

    mouse = args.mouse
    day = args.day
    data_path = args.data_path
    cell_start = args.cell_start
    cell_end = args.cell_end
    nfilters = args.nfilters
    history_length = args.history_length
    max_cells = args.max_cells
    source_path = '/exports/eddie/scratch/hclark3/COHORT12/'

print(f"Running XGBoost Medial/Lateral NGS Assay (OF1) for Mouse {mouse} Day {day} on cells {cell_start} to {cell_end-1} (history={history_length}ms, nfilters={nfilters})")

gcs, ngs, all = classify_cells_both_sessions(mouse, day, percentile_threshold=95, source_path=source_path)
g_m_ids, g_m_cluster_ids, _ = HDBSCAN_grid_modules(gcs, all, mouse, day, min_cluster_size=3, cluster_selection_epsilon=3,
                                                figpath=fig_path, curate_with_vr=True, curate_with_brain_region=True, source_path=source_path)

g_m_cluster_ids = sorted(g_m_cluster_ids, key=len, reverse=True)

# load brain locations for medial/lateral split
locations = pd.read_csv(f'{source_path}/all_cluster_brain_locations_chris.csv')
locations = locations[(locations['mouse'] == mouse) & (locations['day'] == day)]
locations['coord_SCs_x'] = locations['coord_SCs_x'] * -1

# split NGS cells into medial and lateral
ngs_cluster_ids = ngs.cluster_id.values.astype(int)
ent_locations = locations[locations['brain_region'].str.contains('ENT')]

medial_ngs_ids = ngs_cluster_ids[np.isin(ngs_cluster_ids,
    ent_locations[ent_locations['coord_SCs_x'] < medial_lateral_boundary]['cluster_id'].values.astype(int))]
lateral_ngs_ids = ngs_cluster_ids[np.isin(ngs_cluster_ids,
    ent_locations[ent_locations['coord_SCs_x'] >= medial_lateral_boundary]['cluster_id'].values.astype(int))]

covariate_sets = [
    ('medial', medial_ngs_ids),
    ('lateral', lateral_ngs_ids),
]

print(f"Number of medial NGS covariate cells: {len(medial_ngs_ids)}")
print(f"Number of lateral NGS covariate cells: {len(lateral_ngs_ids)}")

# load the open field behaviour data
tcs, tcs_time, beh_OF, clusters_OF, ep = compute_of_tcs(mouse, day, apply_zscore=False, apply_guassian_filter=False, source_path=source_path, session='OF1')
last_ephys_time_bin = clusters_OF[clusters_OF.index[0]].count(bin_size=time_bs, time_units='ms').index[-1]

ep = nap.IntervalSet(start=0, end=last_ephys_time_bin, time_units='s')

# 2D position (x, y)
pos_x_in_time = np.array(beh_OF['P_x'].bin_average(bin_size=time_bs, time_units='ms', ep=ep))
pos_y_in_time = np.array(beh_OF['P_y'].bin_average(bin_size=time_bs, time_units='ms', ep=ep))

if np.any(np.isnan(pos_x_in_time)):
    pos_x_in_time = np.array(pd.Series(pos_x_in_time).ffill().bfill())
if np.any(np.isnan(pos_y_in_time)):
    pos_y_in_time = np.array(pd.Series(pos_y_in_time).ffill().bfill())

# speed, head direction, head-in-goal angle
speed_in_time = np.array(beh_OF['S'].bin_average(bin_size=time_bs, time_units='ms', ep=ep))
hd_in_time = np.array(beh_OF['H'].bin_average(bin_size=time_bs, time_units='ms', ep=ep))
hing_in_time = np.array(beh_OF['Hing'].bin_average(bin_size=time_bs, time_units='ms', ep=ep))

if np.any(np.isnan(speed_in_time)):
    speed_in_time = np.array(pd.Series(speed_in_time).ffill().bfill())
if np.any(np.isnan(hd_in_time)):
    hd_in_time = np.array(pd.Series(hd_in_time).ffill().bfill())
if np.any(np.isnan(hing_in_time)):
    hing_in_time = np.array(pd.Series(hing_in_time).ffill().bfill())

grid_module_population_cluster_ids = np.array(g_m_cluster_ids[0].copy()) if len(g_m_cluster_ids) > 0 else np.array([])

xgb_history = MLencoding(tunemodel='xgboost', cov_history=True, spike_history=False,
                         window=time_bs, n_filters=nfilters, max_time=history_length)

results_rows = []

def get_target_type(target_id, gcs, ngs, g_m_cluster_ids):
    if target_id in gcs.cluster_id.values:
        in_module = any(target_id in module for module in g_m_cluster_ids)
        return 'cmGC' if in_module else 'ncmGC'
    elif target_id in ngs.cluster_id.values:
        return 'NGS'
    return 'NS'

all_target_cell_ids = np.array(all.cluster_id.values)

if cell_end is None:
    cell_end = len(all_target_cell_ids)
target_cells_batch = all_target_cell_ids[cell_start:cell_end]

for idx, id in enumerate(target_cells_batch):
    print(f'Processing reference cell {id} ({cell_start+idx+1}/{cell_end})')

    y = np.array(tcs_time[id])
    T = len(y)

    try:
        theta = get_theta_trace(
            mouse=mouse, day=day, cluster_id=id,
            time_bs=50, resample_bs=time_bs, vr_type='OF1', source_path=source_path,
        )
        theta = np.array(theta)
        if len(theta) < T:
            theta = np.pad(theta, (0, T - len(theta)), mode='constant')
        else:
            theta = theta[:T]
    except Exception as e:
        print(f"    Could not load theta for cluster {id}: {e}. Using zeros.")
        theta = np.zeros(T)

    # Align position, speed, hd, hing to length T
    pos_x = pos_x_in_time[:T]
    pos_y = pos_y_in_time[:T]
    speed = speed_in_time[:T]
    hd    = hd_in_time[:T]
    hing  = hing_in_time[:T]
    if len(pos_x) < T:
        pos_x = np.pad(pos_x, (0, T - len(pos_x)), mode='constant')
    if len(pos_y) < T:
        pos_y = np.pad(pos_y, (0, T - len(pos_y)), mode='constant')
    if len(speed) < T:
        speed = np.pad(speed, (0, T - len(speed)), mode='constant')
    if len(hd) < T:
        hd = np.pad(hd, (0, T - len(hd)), mode='constant')
    if len(hing) < T:
        hing = np.pad(hing, (0, T - len(hing)), mode='constant')

    BASELINE_COV_NAMES = [
        'pos', 'speed', 'hd', 'hing', 'lfp',
        'pos_speed', 'pos_hd', 'pos_hing', 'pos_lfp',
        'pos_speed_hd', 'pos_speed_hing', 'pos_speed_lfp',
        'pos_hd_hing', 'pos_speed_hd_hing',
        'pos_speed_hd_hing_lfp',
    ]
    baseline_xs = [
        np.column_stack((pos_x, pos_y)),
        speed[:, None],
        hd[:, None],
        hing[:, None],
        theta[:, None],
        np.column_stack((pos_x, pos_y, speed)),
        np.column_stack((pos_x, pos_y, hd)),
        np.column_stack((pos_x, pos_y, hing)),
        np.column_stack((pos_x, pos_y, theta)),
        np.column_stack((pos_x, pos_y, speed, hd)),
        np.column_stack((pos_x, pos_y, speed, hing)),
        np.column_stack((pos_x, pos_y, speed, theta)),
        np.column_stack((pos_x, pos_y, hd, hing)),
        np.column_stack((pos_x, pos_y, speed, hd, hing)),
        np.column_stack((pos_x, pos_y, speed, hd, hing, theta)),
    ]

    target_type = get_target_type(id, gcs, ngs, g_m_cluster_ids)

    # baseline fits (same for both covariate locations)
    for b_idx, x_b in enumerate(baseline_xs):
        Y_hat_b, pR2_cv_b = xgb_history.fit_cv(x_b, y, verbose=0, continuous_folds=True)
        print(f'  baseline [{BASELINE_COV_NAMES[b_idx]}] pR2 = {np.nanmean(pR2_cv_b):.4f} target_type={target_type}')
        for cov_location, _ in covariate_sets:
            results_rows.append(dict(
                mouse=mouse, day=day, target_cluster_id=id,
                target_type=target_type,
                baseline=BASELINE_COV_NAMES[b_idx],
                n_covariate_cells=0,
                covariate_location=cov_location,
                pR2_cv=float(np.nanmean(pR2_cv_b)),
            ))

    # cell models: baseline + n covariate cells from each location
    SC_x_ref = all[all.cluster_id == id].SC_x.values[0]
    SC_y_ref = all[all.cluster_id == id].SC_y.values[0]

    for cov_location, cov_population_ids in covariate_sets:
        cov_cluster_ids = cov_population_ids.copy()
        if id in cov_cluster_ids:
            cov_cluster_ids = np.setdiff1d(cov_cluster_ids, id)

        cov_clusters_df = all[all.cluster_id.isin(cov_cluster_ids)].copy()
        cov_clusters_df['SC_diff'] = np.sqrt(
            (cov_clusters_df.SC_x - SC_x_ref)**2 + (cov_clusters_df.SC_y - SC_y_ref)**2
        )
        cov_clusters_df = cov_clusters_df.sample(frac=1).reset_index(drop=True)

        cov_tcs_time = {
            cluster_id: tcs_time[cluster_id]
            for cluster_id in cov_clusters_df.cluster_id
            if cluster_id in tcs_time
        }
        if len(cov_tcs_time) == 0:
            print(f'    No {cov_location} NGS covariate cells available, skipping.')
            continue
        all_x = np.vstack(list(cov_tcs_time.values())).T[:T]

        n_neurons_nonzero = np.arange(1, min(max_cells+1, len(cov_cluster_ids)+1))

        for b_idx, x_b in enumerate(baseline_xs):
            for j, n in enumerate(n_neurons_nonzero):
                np.random.seed(j)
                x = np.column_stack((x_b, all_x[:, :n]))
                Y_hat, pR2_cv = xgb_history.fit_cv(x, y, verbose=0, continuous_folds=True)
                print(f'  {cov_location} baseline [{BASELINE_COV_NAMES[b_idx]}] n_cells={n} pR2 = {np.nanmean(pR2_cv):.4f}')
                results_rows.append(dict(
                    mouse=mouse, day=day, target_cluster_id=id,
                    target_type=target_type,
                    baseline=BASELINE_COV_NAMES[b_idx],
                    n_covariate_cells=int(n),
                    covariate_location=cov_location,
                    pR2_cv=float(np.nanmean(pR2_cv)),
                ))

results_df = pd.DataFrame(results_rows)
suffix = f'_{cell_start}_{cell_end}'
csv_path = f'{data_path}/xgboost_medlat_ngs_of_M{mouse}_D{day}_h{history_length}{suffix}.csv'
results_df.to_csv(csv_path, index=False)
print(f'Saved results DataFrame to {csv_path}')
