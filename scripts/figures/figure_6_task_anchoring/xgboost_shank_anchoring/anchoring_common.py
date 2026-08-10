"""
Shared helpers for the MEC shank-reconstruction / task-anchoring assay.

Pipeline (see run_shank_reconstruction.py):
  * identify anchoring-switching sessions with the task_anchoring_catalogue criteria
    (PC1 of the cell x trial anchoring matrix),
  * for each ENTm ('MEC') cell as a target, fit an xgboost model per shank using that
    shank's ENTm cells as covariates (target excluded from its own shank),
  * bin the cross-validated predicted spikes into a trial x position rate map,
  * recompute anchoring labels on the predicted map with the SAME criteria.

MEC = brain_region starting with 'ENTm' (strict medial entorhinal).
Covariates = the shank's ENTm cells' time-binned activity only (+ xgboost history filters).
"""
import sys
sys.path.insert(0, '/Users/harryclark/Documents/spatial-manifolds/src')

import numpy as np
import pandas as pd
import pynapple as nap
from scipy.ndimage import gaussian_filter

from spatial_manifolds.detect_grids import compute_vr_tcs, get_kmeans_spatial_labels
from spatial_manifolds.toroidal import spectral_analysis
from spatial_manifolds.anaylsis_parameters import bs, tl, time_bs

# ---------------------------------------------------------------- config
SOURCE_PATH  = '/Users/harryclark/Downloads/COHORT12/'
CELLS_CSV    = '/Users/harryclark/Documents/spatial-manifolds/data/cell_classifications_no_regions.csv'
LOCS_CSV     = SOURCE_PATH + 'all_cluster_brain_locations_chris.csv'

bpt          = int(tl / bs)                       # position bins per trial (100)
TARGET_FREQS = np.arange(1, 6) * bs / tl          # anchoring spectral peaks
EDGE_TRIALS  = int(1600 / (2 * (tl / bs)))        # 8 edge trials masked (no valid spectral window)
MEC_PREFIX   = 'ENTm'                             # 'MEC' definition (targets)
PARA_PREFIX  = 'PAR'                              # parasubiculum (alternative covariate pool)

COLOR_ANCHOR     = '#9d5391'
COLOR_NON_ANCHOR = '#e7d7e8'


# ---------------------------------------------------------------- anchoring labels
def compute_anchoring_labels(tc):
    """Task-anchoring labels for a flattened trial x position tuning curve `tc`
    (length n_trials * bpt).  Identical criteria to task_anchoring_catalogue:
    spectral peak membership -> k-means spatial labels -> mask 8 edge trials.
    Returns a float array of length n_trials (0 / 1 / nan) or None on failure."""
    if tc is None or len(tc) == 0:
        return None
    try:
        results  = spectral_analysis({0: tc, 100000: tc}, tl, bs)
        fvalid   = results[5]
        freq_idx = [int(np.argmin(np.abs(fvalid - f))) for f in TARGET_FREQS]
        lbl_win  = np.isin(np.argmax(results[3][0], axis=0), freq_idx).astype(int)
        labels   = get_kmeans_spatial_labels(tc, lbl_win, bs=bs, tl=tl).astype(float)
        labels[:EDGE_TRIALS]  = np.nan
        labels[-EDGE_TRIALS:] = np.nan
        return labels
    except Exception:
        return None


def pca_cell_agreement(labels, n_pcs=5):
    """PCA on the smoothed cell x trial anchoring matrix.
    Returns (trial_score, cell_corr, var_expl, enrichment) exactly as the
    task_anchoring_catalogue uses to score switching sessions."""
    from sklearn.decomposition import PCA
    if not labels:
        raise ValueError('pca_cell_agreement received an empty label list')
    n_cells    = len(labels)
    max_trials = max(len(l) for l in labels)

    smoothed = np.full((n_cells, max_trials), 0.5)
    for i, lab in enumerate(labels):
        s = pd.Series(lab).rolling(5, center=True, min_periods=1).mean()
        smoothed[i, :len(lab)] = s.fillna(0.5).values

    k      = min(n_pcs, n_cells, max_trials)
    pca    = PCA(n_components=k)
    scores = pca.fit_transform(smoothed.T)
    trial_score = scores[:, 0]
    enrichment  = pca.explained_variance_ratio_ * n_cells
    var_expl    = float(pca.explained_variance_ratio_[0])
    cell_corr   = np.array([np.corrcoef(smoothed[i], trial_score)[0, 1] for i in range(n_cells)])
    if (cell_corr > 0).sum() < n_cells / 2:
        trial_score = -trial_score
        cell_corr   = -cell_corr
    return trial_score, cell_corr, var_expl, enrichment


def session_switching_stats(labels):
    """Return a dict of the switching-session statistics + the pass flag
    (agree_frac>0.3 & pc1_enrichment>1.5 & pc1_dominance>2)."""
    n_cells = len(labels)
    trial_score, cell_corr, var_expl, enrichment = pca_cell_agreement(labels)
    agree_frac     = float((cell_corr > 0.3).sum() / n_cells)
    pc1_enrichment = float(enrichment[0])
    pc1_dominance  = float(enrichment[0] / enrichment[1]) if len(enrichment) > 1 else np.nan
    session_passes = bool(agree_frac > 0.3 and pc1_enrichment > 1.5 and pc1_dominance > 2)
    return dict(n_cells=n_cells, agree_frac=agree_frac, var_expl=var_expl,
                pc1_enrichment=pc1_enrichment, pc1_dominance=pc1_dominance,
                session_passes=session_passes, cell_corr=cell_corr)


# ---------------------------------------------------------------- MEC cells + shanks
def get_mec_cells(mouse, day, valid_ids=None, source_path=SOURCE_PATH):
    """DataFrame of ENTm ('MEC') cells for a session: cluster_id, shank_id, brain_region.
    If `valid_ids` is given, keep only those (e.g. cells with tuning curves)."""
    loc = pd.read_csv(source_path + 'all_cluster_brain_locations_chris.csv')
    loc = loc[(loc['mouse'] == mouse) & (loc['day'] == day)].copy()
    loc = loc[loc['brain_region'].astype(str).str.startswith(MEC_PREFIX)]
    loc = loc[['cluster_id', 'shank_id', 'brain_region']].dropna(subset=['shank_id'])
    loc['cluster_id'] = loc['cluster_id'].astype(int)
    loc['shank_id']   = loc['shank_id'].astype(int)
    if valid_ids is not None:
        loc = loc[loc['cluster_id'].isin(set(int(c) for c in valid_ids))]
    return loc.drop_duplicates('cluster_id').reset_index(drop=True)


def get_shank_pools(mouse, day, valid_ids=None, source_path=SOURCE_PATH):
    """Per-shank covariate pool = both ENTm and PAR cells on that shank (a representative
    sample of the two is drawn at fit time). Returns
    {shank_id: {'ent_ids': [...], 'par_ids': [...], 'n_ent': int, 'n_par': int}}
    for every shank carrying at least one ENTm or PAR cell."""
    loc = pd.read_csv(source_path + 'all_cluster_brain_locations_chris.csv')
    loc = loc[(loc['mouse'] == mouse) & (loc['day'] == day)].copy()
    loc = loc.dropna(subset=['shank_id'])
    loc['cluster_id'] = loc['cluster_id'].astype(int)
    loc['shank_id']   = loc['shank_id'].astype(int)
    if valid_ids is not None:
        loc = loc[loc['cluster_id'].isin(set(int(c) for c in valid_ids))]
    reg = loc['brain_region'].astype(str)
    loc = loc.assign(_ent=reg.str.startswith(MEC_PREFIX), _par=reg.str.startswith(PARA_PREFIX))

    pools = {}
    for sh, g in loc.groupby('shank_id'):
        ent = sorted(g.loc[g['_ent'], 'cluster_id'].unique().tolist())
        par = sorted(g.loc[g['_par'], 'cluster_id'].unique().tolist())
        if len(ent) == 0 and len(par) == 0:
            continue
        pools[int(sh)] = dict(ent_ids=ent, par_ids=par, n_ent=len(ent), n_par=len(par))
    return pools


def stratified_sample(ent_ids, par_ids, n_cov, rng):
    """Representative sample of n_cov cells across ENTm and PAR, proportional to their
    counts. Returns (chosen_ids, n_ent_used, n_par_used). Uses all cells if the pool
    is already <= n_cov."""
    ent, par = list(ent_ids), list(par_ids)
    total = len(ent) + len(par)
    if n_cov is None or total <= n_cov:
        return sorted(ent + par), len(ent), len(par)
    n_e = int(round(n_cov * len(ent) / total))
    n_p = n_cov - n_e
    if n_e > len(ent):
        n_e, n_p = len(ent), n_cov - len(ent)
    if n_p > len(par):
        n_p, n_e = len(par), n_cov - len(par)
    chosen = rng.sample(ent, n_e) + rng.sample(par, n_p)
    return sorted(chosen), n_e, n_p


# ---------------------------------------------------------------- time variables
def compute_time_vars(beh, clusters, source_path=SOURCE_PATH):
    """Per 10 ms time-bin behavioural variables + a moving mask, all aligned to
    the spike-count bins used for xgboost covariates/targets."""
    tns = beh['trial_number']
    dt_beh   = beh['travel'] - ((tns[0] - 1) * tl)
    max_bound = int(((np.ceil(np.nanmax(dt_beh)) // tl) + 1) * tl)

    last_ephys_time_bin = clusters[clusters.index[0]].count(
        bin_size=time_bs, time_units='ms').index[-1]
    ep = nap.IntervalSet(start=0, end=last_ephys_time_bin, time_units='s')

    ref_counts = clusters[clusters.index[0]].count(bin_size=time_bs, time_units='ms', ep=ep)
    times = np.asarray(ref_counts.index)

    dt_in_time = np.asarray(beh['travel'].bin_average(bin_size=time_bs, time_units='ms', ep=ep)) \
                 - ((tns[0] - 1) * tl)
    if np.any(np.isnan(dt_in_time)):
        dt_in_time = pd.Series(dt_in_time).ffill().bfill().values
    pos_in_time          = dt_in_time % tl
    trial_number_in_time = (dt_in_time // tl) + tns[0]

    # moving mask: which 10 ms bins fall inside beh['moving'] (matches observed rate maps)
    n = min(len(times), len(dt_in_time))
    times, dt_in_time = times[:n], dt_in_time[:n]
    pos_in_time, trial_number_in_time = pos_in_time[:n], trial_number_in_time[:n]
    idx_tsd = nap.Tsd(t=times, d=np.arange(n).astype(float))
    kept    = idx_tsd.restrict(beh['moving']).values.astype(int)
    moving_mask = np.zeros(n, dtype=bool)
    moving_mask[kept] = True

    return dict(ep=ep, times=times, dt_in_time=dt_in_time, pos_in_time=pos_in_time,
                trial_number_in_time=trial_number_in_time, moving_mask=moving_mask,
                max_bound=max_bound)


def predicted_spikes_to_tc(y_hat, dt_in_time, moving_mask, max_bound, last_ephys_bin):
    """Bin cross-validated predicted spike counts (per 10 ms bin) into a distance
    (trial x position) firing-rate map, over moving epochs only — the same
    construction as the observed compute_vr_tcs rate maps."""
    edges = np.arange(0, max_bound + bs, bs)
    m = moving_mask & np.isfinite(y_hat) & np.isfinite(dt_in_time)
    num, _ = np.histogram(dt_in_time[m], bins=edges, weights=np.clip(y_hat[m], 0, None))
    occ, _ = np.histogram(dt_in_time[m], bins=edges)
    with np.errstate(invalid='ignore', divide='ignore'):
        rate = np.where(occ > 0, num / (occ * (time_bs / 1000.0)), 0.0)
    return rate[:last_ephys_bin]


def labels_to_time(target_labels, trial_number_in_time):
    """Broadcast per-trial anchoring labels onto the 10 ms time-bin grid, so pR2
    can be split by the target cell's anchored / non-anchored epochs."""
    tn = trial_number_in_time.astype(int)
    lab_in_time = np.full(len(tn), np.nan)
    first_tn = int(tn[0])
    for i, t in enumerate(tn):
        j = t - first_tn
        if 0 <= j < len(target_labels):
            lab_in_time[i] = target_labels[j]
    return lab_in_time
