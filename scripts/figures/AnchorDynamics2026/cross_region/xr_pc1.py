"""Per-cell agreement with the MEC anchoring axis, recomputed.

data/population_state/pc1_by_region.csv covers 46 sessions, but 9 of the 34
MEC+VIS sessions this analysis needs are not among them (including M25 D24), and
the script that produced it no longer exists. This reimplements it so every
session in the coupling run has a PC1 rank, and validates against the cached
values wherever both exist -- if the reimplementation disagrees, the cached
figures and the new ranks would be on different scales and could not be plotted
together.

The axis is defined by ENTm GC/NGS cells, as everywhere else in this project.
For a cell inside that population, PC1 is recomputed WITHOUT that cell before
correlating: a cell contributes to the component it is being scored against, and
with a few dozen cells that self-contribution is not small.
"""
import json, sys, warnings
import numpy as np, pandas as pd
warnings.filterwarnings('ignore')
sys.path.insert(0, '/Users/harryclark/Documents/spatial-manifolds/src')
import pynapple as nap

_g = {}
for _c in json.load(open('/Users/harryclark/Documents/spatial-manifolds/scripts/'
                         'figures/AnchorDynamics2026/lick_raster_by_trial_type.ipynb'))['cells']:
    if _c['cell_type'] != 'code':
        continue
    _s = ''.join(_c['source'])
    if _s.startswith('MOUSE, DAY') or 'INV = pd.DataFrame' in _s:
        continue
    exec(compile(_s, '<nb>', 'exec'), _g)
globals().update({k: v for k, v in _g.items() if not k.startswith('__')})
from spatial_manifolds.anchoring import trial_cluster_labels


def pc1_following(mouse, day, rng=None, n_shift=100):
    """r between each cell's per-trial anchoring labels and the population PC1.

    Returns one row per labelled cell: r, a circular-shift null r_null, the
    share of label variance PC1 carries, and the cell's anchored fraction.
    """
    rng = rng or np.random.default_rng(0)
    bp, cp = vr_paths(mouse, day)
    beh = nap.load_file(bp); clusters = nap.load_file(cp)
    trials_all = beh['trials'].as_dataframe()
    trials, orig = clip_trials(trials_all, clusters)
    keep = np.isin(trials_all.number.values.astype(int), orig)
    tn, trav = beh['trial_number'], beh['travel']
    dt = trav - (float(np.asarray(tn.values)[0]) - 1) * TL
    moving = beh['S'].threshold(3.0, method='above').time_support
    n_tr = len(trials_all)

    # every curated cell gets a label vector; the AXIS uses only ENTm GC/NGS
    pop_ids = set(spatial_cell_ids(mouse, day, clusters))
    all_ids = spatial_cell_ids(mouse, day, clusters, region=None,
                               classes=('GC', 'NGS', 'NS'))
    if len(pop_ids) < 5 or not all_ids:
        return None
    tc = nap.compute_1d_tuning_curves(clusters[all_ids], dt, nb_bins=n_tr * NBIN,
                                      minmax=[0, n_tr * TL], ep=moving)
    lab, ids = {}, []
    for c in all_ids:
        M = np.asarray(tc[c]).reshape(n_tr, NBIN)[keep]
        v, _, _, _ = trial_cluster_labels(M)
        if not np.isnan(v).all():
            lab[c] = v; ids.append(c)
    pop = [c for c in ids if c in pop_ids]
    if len(pop) < 5:
        return None
    L = np.array([lab[c] for c in pop])
    frac = np.nanmean(L, axis=0)

    def axis_from(M):
        Mc = np.nan_to_num(M - np.nanmean(M, axis=1, keepdims=True))
        U, sv, Vt = np.linalg.svd(Mc, full_matrices=False)
        pc = Vt[0]
        if np.corrcoef(pc, np.nan_to_num(frac))[0, 1] < 0:   # orient by the state
            pc = -pc
        return pc, sv[0] ** 2 / np.sum(sv ** 2)

    pc1_all, var = axis_from(L)
    rows = []
    for c in ids:
        v = np.nan_to_num(lab[c])
        if c in pop_ids and len(pop) > 5:
            j = pop.index(c)
            pc, _ = axis_from(np.delete(L, j, axis=0))       # leave-one-out
        else:
            pc = pc1_all
        if np.std(v) == 0:
            continue
        r = float(np.corrcoef(v, pc)[0, 1])
        null = [abs(np.corrcoef(np.roll(v, k), pc)[0, 1])
                for k in rng.integers(5, len(v) - 5, n_shift)]
        rows.append(dict(mouse=mouse, day=day, cluster_id=int(c), r=r,
                         r_null=float(np.nanmean(null)), pc1_var=float(var),
                         frac_anch=float(np.nanmean(v)), is_mec=c in pop_ids))
    return pd.DataFrame(rows)


if __name__ == '__main__':
    from scipy.stats import pearsonr
    cache = pd.read_csv('/Users/harryclark/Documents/spatial-manifolds/data/'
                        'population_state/pc1_by_region.csv')
    for mo, dy in ((25, 23), (21, 22)):
        new = pc1_following(mo, dy)
        old = cache[(cache.mouse == mo) & (cache.day == dy)]
        m = new.merge(old[['cluster_id', 'r']], on='cluster_id', suffixes=('', '_old'))
        if len(m):
            rr = pearsonr(m.r, m.r_old)
            print(f'M{mo}D{dy}: {len(new)} cells, {len(m)} matched to cache | '
                  f'r vs cached r = {rr[0]:.4f}, median |diff| = '
                  f'{(m.r - m.r_old).abs().median():.4f}')
        else:
            print(f'M{mo}D{dy}: {len(new)} cells, not in cache')
