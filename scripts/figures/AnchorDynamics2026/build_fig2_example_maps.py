"""Example rate maps for Figure 2D-F, as trial x position maps from -30 to 200 cm
(the last 30 cm of the previous trial first, as the black-box classifier sees
it).

The cells shown are the clearest of those that load most on the session's
whole-track state: of the N_CAND highest PC1 loadings, the N_CELLS whose maps
are most reliable on the trials they are labelled anchored (split-half
correlation of the odd- and even-trial mean maps over 0-200 cm), among cells
firing at least MIN_RATE Hz on the track. Until 2026-10-31 they were simply the
three highest loadings, which can be sparse cells with no visible field.

The maps are built exactly as build_neighbour_labels.py builds the segments it
classifies (moving epochs, 2 cm bins), and kept on the same trials, so each map
lines up row for row with the label rasters beneath it.

The sessions are chosen by pick_fig2_examples.py; Figure 2 reads them from the
output, so EXAMPLES is the only place they are set.

Writes data/population_state/fig2_example_maps.npz.
"""
import importlib.util
import sys

import numpy as np
import pynapple as nap

HERE = '/Users/harryclark/Documents/spatial-manifolds/scripts/figures/AnchorDynamics2026'
sys.path.insert(0, HERE)
_spec = importlib.util.spec_from_file_location('bnl', f'{HERE}/build_neighbour_labels.py')
B = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(B)

EXAMPLES = {'MEC': (25, 23), 'SUB': (26, 19), 'VIS': (28, 20)}   # chosen from pick_fig2_examples.pdf
N_CELLS, N_CAND = 3, 12
MIN_RATE, MIN_ANCH = 1.0, 10          # Hz on the track; anchored trials per half-split
OUT = f'{B.PS}/fig2_example_maps.npz'


def pc1_loading(X):
    """First-component loadings of a cells x trials label matrix, signed so that
    the component's score rises with the fraction of cells anchored."""
    Xc = np.nan_to_num(X - np.nanmean(X, 1, keepdims=True))
    U, _, Vt = np.linalg.svd(Xc, full_matrices=False)
    sign = np.sign(np.corrcoef(Vt[0], np.nanmean(X, 0))[0, 1])
    return U[:, 0] * sign


def reliability(M, lab):
    """Split-half correlation of the mean map over the cell's anchored trials."""
    tr = np.where(lab == 1)[0]
    if len(tr) < 2 * MIN_ANCH:
        return np.nan
    on = M[:, 15:]                                  # 0-200 cm (2 cm bins from -30)
    a, b = np.nanmean(on[tr[::2]], 0), np.nanmean(on[tr[1::2]], 0)
    ok = np.isfinite(a) & np.isfinite(b)
    return np.corrcoef(a[ok], b[ok])[0, 1] if ok.sum() > 10 else np.nan


def session_maps(region, mouse, day, n_cand=N_CAND):
    """Maps of the n_cand cells loading most on the whole-track state, with their
    ids, loadings, mean rates and anchored-trial reliability."""
    z = np.load(f'{B.OUT}/M{mouse}D{day}.npz', allow_pickle=True)
    m = z['region'] == region
    L = z['labels_full'][m]
    ids = z['cluster_id'][m].astype(int)
    load = pc1_loading(L)
    top = np.argsort(load)[::-1][:n_cand]
    bp, cp = B.vr_paths(mouse, day)
    beh, clusters = nap.load_file(bp), nap.load_file(cp)
    trials_all = beh['trials'].as_dataframe()
    trials, orig = B.clip_trials(trials_all, clusters)
    keep = np.isin(trials_all.number.values.astype(int), orig)
    assert np.array_equal(trials.number.values.astype(int), z['trial'].astype(int))
    tn, trav = beh['trial_number'], beh['travel']
    dt = trav - (float(np.asarray(tn.values)[0]) - 1) * B.TL
    moving = beh['S'].threshold(3.0, method='above').time_support
    n_tr = len(trials_all)
    tc = nap.compute_1d_tuning_curves(clusters[list(ids[top])], dt, nb_bins=n_tr * B.NBIN,
                                      minmax=[0, n_tr * B.TL], ep=moving)
    maps = np.array([B.segment(np.asarray(tc[c]), n_tr, -30, 200)[keep] for c in ids[top]])
    rate = np.nanmean(maps[:, :, 15:], axis=(1, 2))
    rel = np.array([reliability(M_, L[i]) for M_, i in zip(maps, top)])
    return dict(maps=maps, ids=ids[top], load=load[top], rate=rate, rel=rel)


def choose_cells(s, n=N_CELLS):
    """The n most reliable maps among cells above the rate floor, in loading order."""
    ok = np.where((s['rate'] >= MIN_RATE) & np.isfinite(s['rel']))[0]
    best = ok[np.argsort(s['rel'][ok])[::-1][:n]]
    if len(best) < n:                               # too few: fill by loading
        best = np.r_[best, [i for i in range(len(s['ids'])) if i not in best][:n - len(best)]]
    return np.sort(best)


if __name__ == '__main__':
    out = {}
    for region, (mouse, day) in EXAMPLES.items():
        s = session_maps(region, mouse, day)
        pick = choose_cells(s)
        out[f'{region}_maps'] = s['maps'][pick]
        out[f'{region}_cells'] = s['ids'][pick]
        out[f'{region}_session'] = np.array([mouse, day])
        print(f'{region} M{mouse} D{day}: cells {list(s["ids"][pick])}, '
              f'reliability {np.round(s["rel"][pick], 2)}, rate {np.round(s["rate"][pick], 1)} Hz')
    np.savez_compressed(OUT, **out)
    print(f'wrote {OUT}')
