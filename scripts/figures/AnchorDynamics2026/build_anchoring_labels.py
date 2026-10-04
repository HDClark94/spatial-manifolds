"""Build the canonical anchoring labels once, so every figure reads the same file.

Usage:  python3 build_anchoring_labels.py [worker n_workers]

WHY THIS EXISTS. Until now each notebook recomputed anchoring labels from the
raw rate maps, which made every figure a few minutes of compute, made results
depend on whichever copy of the classifier a notebook happened to import, and
meant a change to the classifier silently invalidated cached products elsewhere
(`eye_anchoring_trials.csv`, `theta_frequency.csv`, `pc1_by_region.csv`) with no
record of which labels they were built from. The gated classifier makes this
worse, because the per-cell null adds a shuffle to every cell.

So the labels are computed once, here, and written to files the notebooks load.

WHAT IT WRITES, all under data/population_state/

  labels/M{mouse}D{day}.npz     per session:
        labels      (n_cells, n_trials) 1 anchored / 0 non / NaN unclassified
        cluster_id  (n_cells,)
        trial       (n_trials,) trial numbers, already clipped to the ephys window
        frac_anch   (n_trials,) population state: fraction of cells anchored
        pc1         (n_trials,) PC1 of the cell x trial matrix, oriented to
                    correlate positively with frac_anch
        pc1_load    (n_cells,)  each cell's loading on that axis
        pc1_var     share of label variance carried by PC1
        null_thr    (n_cells,)  the cell's own chance threshold
        gate        (n_cells,)  'split kept' / 'all anchored' / 'all non-anchored'

  anchoring_cells.csv           one row per cell, with region, gate action,
                                null threshold and anchored fraction
  anchoring_trials.csv          one row per trial: frac_anch, pc1, n_cells
  null_thresholds.npz           the shuffle cache, so nothing reshuffles again

The per-cell labels use ALL curated cells (GC/NGS/NS, any region). The
population state and PC1 are computed over ALL curated ENTm cells -- grid,
non-grid spatial AND narrow-spiking. Restricting the population to spatial cells
only changes the state (r = 0.63 between the two definitions) and, because
fewer cells means fewer sessions carrying both states, changes which sessions
survive the four-cell inclusion rule.
"""
import json, os, sys, time, warnings
import numpy as np, pandas as pd
warnings.filterwarnings('ignore')
sys.path.insert(0, '/Users/harryclark/Documents/spatial-manifolds/src')
import pynapple as nap
from spatial_manifolds.anchoring import trial_cluster_labels, null_mean_corr, save_null_cache

NB_PATH = ('/Users/harryclark/Documents/spatial-manifolds/scripts/figures/'
           'AnchorDynamics2026/lick_raster_by_trial_type.ipynb')
_g = {}
for _c in json.load(open(NB_PATH))['cells']:
    if _c['cell_type'] != 'code':
        continue
    _s = ''.join(_c['source'])
    if _s.startswith('MOUSE, DAY') or 'INV = pd.DataFrame' in _s:
        continue
    exec(compile(_s, '<nb>', 'exec'), _g)
globals().update({k: v for k, v in _g.items() if not k.startswith('__')})

OUT = '/Users/harryclark/Documents/spatial-manifolds/data/population_state'
os.makedirs(f'{OUT}/labels', exist_ok=True)


def gate_action(mc, thr):
    """Which branch the absolute gate took, for the record."""
    from sklearn.cluster import KMeans
    v = np.isfinite(mc)
    if v.sum() < 6:
        return 'unclassified'
    km = KMeans(2, n_init=10, random_state=0).fit(mc[v].reshape(-1, 1))
    lo, hi = np.sort(km.cluster_centers_.ravel())
    return ('all non-anchored' if hi < thr else
            'all anchored' if lo >= thr else 'split kept')


def build(mouse, day):
    bp, cp = vr_paths(mouse, day)
    if not (os.path.exists(bp) and os.path.exists(cp)):
        return None
    beh = nap.load_file(bp); clusters = nap.load_file(cp)
    trials_all = beh['trials'].as_dataframe()
    trials, orig = clip_trials(trials_all, clusters)
    keep = np.isin(trials_all.number.values.astype(int), orig)
    tn, trav = beh['trial_number'], beh['travel']
    dt = trav - (float(np.asarray(tn.values)[0]) - 1) * TL
    moving = beh['S'].threshold(3.0, method='above').time_support
    n_tr = len(trials_all)

    pop_ids = set(spatial_cell_ids(mouse, day, clusters,
                                   classes=('GC', 'NGS', 'NS')))   # all ENTm
    all_ids = spatial_cell_ids(mouse, day, clusters, region=None,
                               classes=('GC', 'NGS', 'NS'))
    if len(pop_ids) < 5 or not all_ids:
        return None
    tc = nap.compute_1d_tuning_curves(clusters[all_ids], dt, nb_bins=n_tr * NBIN,
                                      minmax=[0, n_tr * TL], ep=moving)
    L, ids, thrs, acts = [], [], [], []
    for c in all_ids:
        M = np.asarray(tc[c]).reshape(n_tr, NBIN)[keep]
        lab, mc, sm, _ = trial_cluster_labels(M)
        v = np.isfinite(mc)
        thr = null_mean_corr(sm[v]) if v.sum() >= 6 else np.nan
        L.append(lab); ids.append(int(c)); thrs.append(thr)
        acts.append(gate_action(mc, thr) if np.isfinite(thr) else 'unclassified')
    L = np.array(L); ids = np.array(ids)

    in_pop = np.array([i in pop_ids for i in ids])
    P = L[in_pop]
    usable = ~np.isnan(P).all(axis=1)
    P = P[usable]
    if P.shape[0] < 5:
        return None
    frac = np.nanmean(P, axis=0)
    Pc = np.nan_to_num(P - np.nanmean(P, axis=1, keepdims=True))
    U, sv, Vt = np.linalg.svd(Pc, full_matrices=False)
    pc1, load = Vt[0], U[:, 0]
    if np.corrcoef(pc1, np.nan_to_num(frac))[0, 1] < 0:      # sign is arbitrary
        pc1, load = -pc1, -load
    var = float(sv[0] ** 2 / np.sum(sv ** 2))

    full_load = np.full(len(ids), np.nan)
    full_load[np.where(in_pop)[0][usable]] = load
    np.savez_compressed(f'{OUT}/labels/M{mouse}D{day}.npz',
                        labels=L, cluster_id=ids,
                        trial=trials.number.values.astype(int),
                        frac_anch=frac, pc1=pc1, pc1_load=full_load,
                        pc1_var=var, null_thr=np.array(thrs),
                        gate=np.array(acts))
    cells = pd.DataFrame(dict(mouse=mouse, day=day, cluster_id=ids,
                              in_population=in_pop, null_thr=thrs, gate=acts,
                              pc1_load=full_load,
                              frac_anch=np.nanmean(L, axis=1)))
    tr = pd.DataFrame(dict(mouse=mouse, day=day,
                           trial=trials.number.values.astype(int),
                           frac_anch=frac, pc1=pc1, n_cells=P.shape[0],
                           pc1_var=var))
    return cells, tr


if __name__ == '__main__':
    w, nw = (int(sys.argv[1]), int(sys.argv[2])) if len(sys.argv) > 2 else (0, 1)
    sess = sorted({(int(m), int(d)) for m, d in
                   CLS[['mouse', 'day']].drop_duplicates().values})
    mine = [s for i, s in enumerate(sess) if i % nw == w]
    print(f'worker {w}/{nw}: {len(mine)} of {len(sess)} sessions', flush=True)
    C, T, t0 = [], [], time.time()
    for k, (mo, dy) in enumerate(mine, 1):
        if os.path.exists(f'{OUT}/labels/M{mo}D{dy}.npz'):
            z = np.load(f'{OUT}/labels/M{mo}D{dy}.npz', allow_pickle=True)
            C.append(pd.DataFrame(dict(mouse=mo, day=dy, cluster_id=z['cluster_id'],
                                       in_population=np.isfinite(z['pc1_load']),
                                       null_thr=z['null_thr'], gate=z['gate'],
                                       pc1_load=z['pc1_load'],
                                       frac_anch=np.nanmean(z['labels'], axis=1))))
            T.append(pd.DataFrame(dict(mouse=mo, day=dy, trial=z['trial'],
                                       frac_anch=z['frac_anch'], pc1=z['pc1'],
                                       n_cells=int(np.isfinite(z['pc1_load']).sum()),
                                       pc1_var=float(z['pc1_var']))))
            continue
        try:
            got = build(mo, dy)
        except Exception as e:
            print(f'  ! M{mo}D{dy}: {type(e).__name__}: {e}', flush=True); continue
        if got is None:
            continue
        C.append(got[0]); T.append(got[1])
        save_null_cache()
        print(f'[w{w} {k}/{len(mine)}] M{mo}D{dy}: {len(got[0])} cells, '
              f'{time.time()-t0:.0f}s', flush=True)
    if C:
        pd.concat(C, ignore_index=True).to_csv(f'{OUT}/anchoring_cells_w{w}.csv',
                                               index=False)
        pd.concat(T, ignore_index=True).to_csv(f'{OUT}/anchoring_trials_w{w}.csv',
                                               index=False)
    print(f'worker {w} done in {time.time()-t0:.0f}s', flush=True)
