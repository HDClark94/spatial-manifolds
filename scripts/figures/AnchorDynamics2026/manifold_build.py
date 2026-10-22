"""Per-session trial x position x cell maps for the manifold figure.

Usage:  python manifold_build.py [worker n_workers]

For every session with anchoring labels: the MEC spatial cells that define the
population state (ENTm, open-field GC or NGS, as lick_raster_by_trial_type's
spatial_cell_ids), and for every clipped trial their running-only rate maps on
the 200 cm track at 2 cm, smoothed (NaN-aware, sigma 1.5 bins as the labels)
and averaged into 4 cm bins. Alongside, per trial: the population state (PC1 of
the labels > 0, anchoring_trials.csv), trial type, outcome, the first stop on
the track outside the black boxes, and the spike total, so silent trials can be
dropped downstream rather than read as states.

Writes data/manifold/M{mo}D{dy}.npz
"""
import json
import os
import sys
import time
import warnings

import numpy as np
import pandas as pd
from scipy.ndimage import gaussian_filter1d

warnings.filterwarnings('ignore')
sys.path.insert(0, '/Users/harryclark/Documents/spatial-manifolds/src')
import pynapple as nap
from spatial_manifolds.anchoring import load_session_labels, smooth_nanaware

ROOT = '/Users/harryclark/Documents/spatial-manifolds'
FIG = f'{ROOT}/scripts/figures/AnchorDynamics2026'
PS = f'{ROOT}/data/population_state'
OUTD = f'{ROOT}/data/manifold'
STOP_SPD = 3.0            # cm/s, the task's stop threshold
STOP_MIN_S = 0.1          # a stop must last this long
TRACK_LO, TRACK_HI = 30.0, 170.0     # outside the 30 cm black boxes

_g = {}
for _c in json.load(open(f'{FIG}/lick_raster_by_trial_type.ipynb'))['cells']:
    if _c['cell_type'] != 'code':
        continue
    _s = ''.join(_c['source'])
    if _s.startswith('MOUSE, DAY') or 'INV = pd.DataFrame' in _s:
        continue
    exec(compile(_s, '<nb>', 'exec'), _g)
vr_paths, clip_trials, spatial_cell_ids = _g['vr_paths'], _g['clip_trials'], _g['spatial_cell_ids']
TL, NBIN, CLS = _g['TL'], _g['NBIN'], _g['CLS']
T_ALL = pd.read_csv(f'{PS}/anchoring_trials.csv')


def first_stop(P, S, t0, t1):
    """Track position of the first stop of >= STOP_MIN_S inside [t0, t1]."""
    i0, i1 = np.searchsorted(P.index, [t0, t1])
    p, s, t = np.asarray(P.values)[i0:i1], np.asarray(S.values)[i0:i1], np.asarray(P.index)[i0:i1]
    # the 60 Hz speed trace is noisy sample to sample (it goes negative), so
    # it is smoothed over ~100 ms before the threshold is applied
    s = gaussian_filter1d(s.astype(float), 6)
    still = (s < STOP_SPD) & (p > TRACK_LO) & (p < TRACK_HI)
    if not still.any():
        return np.nan
    d = np.diff(np.r_[0, still.astype(int), 0])
    for a, b in zip(np.flatnonzero(d == 1), np.flatnonzero(d == -1)):
        if t[b - 1] - t[a] >= STOP_MIN_S:
            return float(p[a])
    return np.nan


def session(mo, dy):
    z = load_session_labels(mo, dy)
    if z is None:
        return None
    bp, cp = vr_paths(mo, dy)
    if not (os.path.exists(bp) and os.path.exists(cp)):
        return None
    beh = nap.load_file(bp); clusters = nap.load_file(cp)
    trials_all = beh['trials'].as_dataframe()
    trials, orig = clip_trials(trials_all, clusters)
    keep = np.isin(trials_all.number.values.astype(int), orig)
    ids = spatial_cell_ids(mo, dy, clusters)
    if len(ids) < 15:
        return None
    tn, trav = beh['trial_number'], beh['travel']
    dtrav = trav - (float(np.asarray(tn.values)[0]) - 1) * TL
    moving = beh['S'].threshold(STOP_SPD, method='above').time_support
    n_all = len(trials_all)
    tc = nap.compute_1d_tuning_curves(clusters[ids], dtrav, nb_bins=n_all * NBIN,
                                      minmax=[0, n_all * TL], ep=moving)
    occ = nap.compute_1d_tuning_curves(
        nap.TsGroup({0: nap.Ts(np.asarray(dtrav.restrict(moving).index))}), dtrav,
        nb_bins=n_all * NBIN, minmax=[0, n_all * TL], ep=moving)
    n_tr = int(keep.sum())
    M = np.empty((n_tr, NBIN // 2, len(ids)), np.float32)
    nspk = np.zeros(n_tr)
    for j, c in enumerate(ids):
        R = np.asarray(tc[c]).reshape(n_all, NBIN)[keep]
        Sm = np.array([smooth_nanaware(r) for r in R])
        M[:, :, j] = np.nanmean(Sm.reshape(n_tr, NBIN // 2, 2), axis=2)
        nspk += np.nan_to_num(R).sum(1)
    valid = np.isfinite(np.asarray(occ[0]).reshape(n_all, NBIN)[keep]).mean(1)
    # per-trial behaviour, on the clipped numbering
    st = T_ALL[(T_ALL.mouse == mo) & (T_ALL.day == dy)].set_index('trial')
    tr = trials.reset_index(drop=True)
    state = np.array([st.pc1.get(int(k), np.nan) for k in tr.number], float)
    P, S = beh['P'], beh['S']
    stops = np.array([first_stop(P, S, a, b) for a, b in zip(tr.start, tr.end)])
    cl = CLS[(CLS.mouse == mo) & (CLS.day == dy)].set_index('cluster_id')
    np.savez_compressed(
        f'{OUTD}/M{mo}D{dy}.npz', maps=M.astype(np.float16), ids=np.array(ids),
        cls=np.array([str(cl.cell_class_of1.get(c, '')) for c in ids]),
        region=np.array([str(cl.brain_region.get(c, '')) for c in ids]),
        trial=tr.number.values.astype(int), pc1=state, ttype=tr.type.values.astype(str),
        perf=tr.performance.values.astype(str), stop=stops, nspk=nspk, valid=valid,
        t_start=tr.start.values, t_end=tr.end.values)
    return n_tr, len(ids)


if __name__ == '__main__':
    w, nw = (int(sys.argv[1]), int(sys.argv[2])) if len(sys.argv) > 2 else (0, 1)
    sess = sorted({(int(f.split('M')[1].split('D')[0]), int(f.split('D')[1].split('.')[0]))
                   for f in os.listdir(f'{PS}/labels') if f.endswith('.npz')})
    mine = [s for i, s in enumerate(sess) if i % nw == w]
    t0 = time.time()
    for mo, dy in mine:
        if os.path.exists(f'{OUTD}/M{mo}D{dy}.npz'):
            continue
        try:
            r = session(mo, dy)
        except Exception as e:
            print(f'  ! M{mo}D{dy}: {type(e).__name__}: {e}', flush=True); continue
        if r:
            print(f'  M{mo}D{dy}: {r[0]} trials, {r[1]} cells, {time.time() - t0:.0f}s', flush=True)
    print(f'worker {w} done {time.time() - t0:.0f}s', flush=True)
