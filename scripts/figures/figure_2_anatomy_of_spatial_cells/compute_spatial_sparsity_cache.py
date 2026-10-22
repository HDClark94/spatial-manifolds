"""
Compute Skaggs spatial sparsity (mean(rate_map)^2 / mean(rate_map^2)) for every
ENTm cell in data/cell_classifications.csv from its OF1 rate map, and cache the
result to data/spatial_sparsity_cache.csv (mirrors data/spatial_stability_cache.csv).
"""
import numpy as np
import pandas as pd
import pynapple as nap
from pathlib import Path

from spatial_manifolds.data.binning import get_bin_config
from spatial_manifolds.util import gaussian_filter_nan

SOURCE = Path('/Users/harryclark/Downloads/COHORT12/')
OUT_PATH = Path('/Users/harryclark/Documents/spatial-manifolds/data/spatial_sparsity_cache.csv')

cc = pd.read_csv('/Users/harryclark/Documents/spatial-manifolds/data/cell_classifications.csv')
cc = cc[cc['mouse'] != 22]
cc = cc[cc['brain_region'].str.contains('ENTm', case=False, na=False)].copy()

bin_config = get_bin_config('OF1')[('P_x', 'P_y')]

results = []
sessions = cc[['mouse', 'day']].drop_duplicates().values
for mouse, day in sessions:
    mouse, day = int(mouse), int(day)
    of1_folder = SOURCE / f'M{mouse}' / f'D{day:02d}' / 'OF1'
    beh_file = of1_folder / f'sub-{mouse}_day-{day:02d}_ses-OF1_beh.nwb'
    clusters_file = of1_folder / f'sub-{mouse}_day-{day:02d}_ses-OF1_srt-kilosort4_clusters.npz'

    if not beh_file.exists() or not clusters_file.exists():
        print(f'M{mouse}D{day}: missing OF1 data, skipping')
        continue

    session = nap.load_file(str(beh_file), lazy_loading=False)
    clusters = nap.load_file(str(clusters_file))
    positions = np.stack([session['P_x'], session['P_y']], axis=1)

    cluster_ids = cc[(cc['mouse'] == mouse) & (cc['day'] == day)]['cluster_id'].unique()
    for cluster_id in cluster_ids:
        cluster_id = int(cluster_id)
        if cluster_id not in clusters.index:
            continue
        spikes = clusters[cluster_id]
        with np.errstate(invalid='ignore', divide='ignore'):
            tc = nap.compute_2d_tuning_curves(
                nap.TsGroup([spikes]),
                positions,
                nb_bins=bin_config['num_bins'],
                minmax=bin_config['bounds'],
                ep=session['moving'],
            )
            rate_map = gaussian_filter_nan(
                np.stack(list(tc[0].values())),
                sigma=(0, bin_config['smooth_sigma'], bin_config['smooth_sigma']),
            )[0]

        r = rate_map[~np.isnan(rate_map)]
        sparsity = (r.mean() ** 2) / (np.mean(r ** 2) + 1e-9) if len(r) > 0 else np.nan

        results.append({
            'mouse': mouse, 'day': day, 'cluster_id': cluster_id, 'sparsity': sparsity,
        })

    print(f'M{mouse}D{day}: {len(cluster_ids)} cells processed')

out = pd.DataFrame(results)
out.to_csv(OUT_PATH, index=False)
print(f'Wrote {len(out)} rows to {OUT_PATH}')
