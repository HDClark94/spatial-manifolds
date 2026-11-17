"""Time-binned activity of every MEC cell, for the manifold supplement.

Usage:  python manifold_supp_build.py [worker n_workers]

For every session with anchoring labels: all curated MEC clusters (brain_region
ENTm*, any open-field class -- grid, non-grid spatial and non-spatial alike),
spike counts in DT bins, smoothed with a Gaussian of SMOOTH_S, square-rooted
(variance-stabilising) and z-scored per cell. Only bins in which the animal
runs (>= 3 cm/s, the task's own threshold) are kept, so stopping -- which
changes every cell's rate -- cannot itself make a structure. Each bin carries
its track position, lap (clipped numbering, as the labels), running speed and
the lap's population state (PC1 of the labels > 0).

Writes data/manifold_supp/M{mo}D{dy}.npz
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

ROOT = '/Users/harryclark/Documents/spatial-manifolds'
FIG = f'{ROOT}/scripts/figures/AnchorDynamics2026'
PS = f'{ROOT}/data/population_state'
OUTD = f'{ROOT}/data/manifold_supp'
DT = 0.1                   # s
SMOOTH_S = 0.15            # s, Gaussian sd
MIN_SPD = 3.0              # cm/s
MIN_CELLS = 20
MIN_RATE = 0.2             # Hz over the running bins

_g = {}
for _c in json.load(open(f'{FIG}/lick_raster_by_trial_type.ipynb'))['cells']:
    if _c['cell_type'] != 'code':
        continue
    _s = ''.join(_c['source'])
    if _s.startswith('MOUSE, DAY') or 'INV = pd.DataFrame' in _s:
        continue
    exec(compile(_s, '<nb>', 'exec'), _g)
vr_paths, clip_trials, CLS = _g['vr_paths'], _g['clip_trials'], _g['CLS']
T_ALL = pd.read_csv(f'{PS}/anchoring_trials.csv')


def session(mo, dy):
    bp, cp = vr_paths(mo, dy)
    if not (os.path.exists(bp) and os.path.exists(cp)):
        return None
    st = T_ALL[(T_ALL.mouse == mo) & (T_ALL.day == dy)]
    if not len(st):
        return None
    beh = nap.load_file(bp); clusters = nap.load_file(cp)
    trials_all = beh['trials'].as_dataframe()
    trials, orig = clip_trials(trials_all, clusters)
    cl = CLS[(CLS.mouse == mo) & (CLS.day == dy)]
    ids = [int(c) for c in cl[cl.brain_region.astype(str).str.startswith('ENTm')].cluster_id
           if int(c) in clusters.index]
    if len(ids) < MIN_CELLS:
        return None
    t0, t1 = float(trials.start.min()), float(trials.end.max())
    edges = np.arange(t0, t1, DT); ctr = edges[:-1] + DT / 2

    def at(tsd):
        i = np.searchsorted(np.asarray(tsd.index), ctr).clip(0, len(tsd) - 1)
        return np.asarray(tsd.values)[i]
    pos, spd, trn = at(beh['P']), at(beh['S']), at(beh['trial_number']).astype(int)
    spd = gaussian_filter1d(spd.astype(float), SMOOTH_S / DT)
    remap = {int(o): int(n) for o, n in zip(orig, trials.number)}
    tr = np.array([remap.get(t, -1) for t in trn])
    state = dict(zip(st.trial.astype(int), st.pc1.values))
    pc1 = np.array([state.get(int(t), np.nan) for t in tr])
    keep = (spd >= MIN_SPD) & (tr > 0) & np.isfinite(pos) & np.isfinite(pc1)
    C = np.zeros((len(ctr), len(ids)), np.float32)
    for j, c in enumerate(ids):
        C[:, j] = np.histogram(np.asarray(clusters[c].index), bins=edges)[0]
    rate = C[keep].mean(0) / DT
    ok = rate >= MIN_RATE
    if ok.sum() < MIN_CELLS:
        return None
    X = np.sqrt(gaussian_filter1d(C[:, ok], SMOOTH_S / DT, axis=0))[keep]
    X = (X - X.mean(0)) / (X.std(0) + 1e-9)
    cls = cl.set_index('cluster_id').cell_class_of1
    np.savez_compressed(f'{OUTD}/M{mo}D{dy}.npz', X=X.astype(np.float32),
                        pos=pos[keep].astype(np.float32), trial=tr[keep], speed=spd[keep],
                        pc1=pc1[keep], t=ctr[keep], ids=np.array(ids)[ok],
                        cls=np.array([str(cls.get(c, '')) for c in np.array(ids)[ok]]))
    return int(keep.sum()), int(ok.sum())


if __name__ == '__main__':
    w, nw = (int(sys.argv[1]), int(sys.argv[2])) if len(sys.argv) > 2 else (0, 1)
    sess = sorted({(int(f.split('M')[1].split('D')[0]), int(f.split('D')[1].split('.')[0]))
                   for f in os.listdir(f'{PS}/labels') if f.endswith('.npz')})
    os.makedirs(OUTD, exist_ok=True)
    t0 = time.time()
    for mo, dy in [s for i, s in enumerate(sess) if i % nw == w]:
        if os.path.exists(f'{OUTD}/M{mo}D{dy}.npz'):
            continue
        try:
            r = session(mo, dy)
        except Exception as e:
            print(f'  ! M{mo}D{dy}: {type(e).__name__}: {e}', flush=True); continue
        if r:
            print(f'  M{mo}D{dy}: {r[0]} running bins, {r[1]} MEC cells, {time.time() - t0:.0f}s',
                  flush=True)
