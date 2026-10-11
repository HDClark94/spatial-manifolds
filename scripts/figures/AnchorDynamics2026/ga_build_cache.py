"""Build the data the graphical abstract needs, so the abstract itself can be
rebuilt without the lab volume mounted.

Writes ga_cache.npz with:
  gc_map     trial x position firing rate for one real grid cell
  gc_anch    per-trial anchoring label for the same trials
  gc_meta    mouse, day, cluster id, grid score
  stops_b / stops_nb   (trial, position) stop coordinates, cued and uncued
  trials_b / trials_nb the trial numbers those rasters span

Run this once with /Volumes/datasetsSD (or the lab mount) available.
"""
import json
import os
import sys
import warnings

import numpy as np
import pandas as pd

warnings.filterwarnings('ignore')
sys.path.insert(0, '/Users/harryclark/Documents/spatial-manifolds/src')
from spatial_manifolds.anchoring import smooth_nanaware           # noqa: E402

FIG = os.path.dirname(os.path.abspath(__file__))
ROOT = '/Users/harryclark/Documents/spatial-manifolds'
NB = f'{FIG}/lick_raster_by_trial_type.ipynb'
OUT = f'{FIG}/ga_cache.npz'

MOUSE, DAY = 26, 18
CELL = 109          # grid score 0.95, highest PC1 loading in the session,
                    # and 52% anchored trials so both states are well sampled
NBIN, TL, SIGMA = 50, 200.0, 2.0

_g = {}
for _c in json.load(open(NB))['cells']:
    if _c['cell_type'] != 'code':
        continue
    _s = ''.join(_c['source'])
    if _s.startswith('MOUSE, DAY') or 'INV = pd.DataFrame' in _s:
        continue
    exec(compile(_s, '<nb>', 'exec'), _g)
globals().update({k: v for k, v in _g.items() if not k.startswith('__')})

# ---- the grid cell's trial x position map, as fig2_supp builds its examples --
bp, cp = vr_paths(MOUSE, DAY)                                   # noqa: F821
beh = nap.load_file(bp)                                         # noqa: F821
clusters = nap.load_file(cp)                                    # noqa: F821
trials_all = beh['trials'].as_dataframe()
_, orig = clip_trials(trials_all, clusters)                     # noqa: F821
keep = np.isin(trials_all.number.values.astype(int), orig)
tn, trav = beh['trial_number'], beh['travel']
dt = trav - (float(np.asarray(tn.values)[0]) - 1) * TL
moving = beh['S'].threshold(3.0, method='above').time_support
n_all = len(trials_all)
tc = nap.compute_1d_tuning_curves(clusters[[CELL]], dt,          # noqa: F821
                                  nb_bins=n_all * NBIN,
                                  minmax=[0, n_all * TL], ep=moving)
M = np.asarray(tc[CELL]).reshape(n_all, NBIN)[keep]
gc_map = np.array([smooth_nanaware(r, sigma=SIGMA) for r in M], dtype=np.float32)

z = np.load(f'{ROOT}/data/population_state/labels/M{MOUSE}D{DAY}.npz',
            allow_pickle=True)
i = int(np.where(z['cluster_id'] == CELL)[0][0])
gc_anch = np.asarray(z['labels'][i], dtype=np.float32)
assert len(gc_anch) == gc_map.shape[0], (len(gc_anch), gc_map.shape)

C = pd.read_csv(f'{ROOT}/data/behavioural_influence_cache/'
                f'M{MOUSE}_D{DAY}.csv')
gs = float(C.loc[C.cluster_id == CELL, 'grid_score'].iloc[0])
print(f'M{MOUSE}D{DAY} cell {CELL}: grid score {gs:.2f}, '
      f'{gc_map.shape[0]} trials, {np.nanmean(gc_anch):.0%} anchored')

# ---- the cued and uncued stop rasters ---------------------------------------
beh_, trials, stops = load_vr(MOUSE, DAY)                        # noqa: F821
out = dict(gc_map=gc_map, gc_anch=gc_anch,
           gc_meta=np.array([MOUSE, DAY, CELL, gs], dtype=float))
for tt in ('b', 'nb'):
    tr = trials[trials.type == tt].number.astype(int).values
    sel = stops[np.isin(stops.trial.astype(int), tr)]
    out[f'stops_{tt}'] = np.c_[sel.trial.values.astype(float),
                               sel.pos.values.astype(float)]
    print(f'  {tt}: {len(tr)} trials, {len(sel)} stops')
out['n_trials'] = np.array([int(trials.number.max())], dtype=float)
# ---- one pupil image per state, from the cached Figure 5 frames -------------
# Seven frames per state at MATCHED track positions. Position 0 is used here:
# it is where the pupil is largest (Figure 5I peaks at the start of the track)
# and so where a size difference is most legible. It is not the only position
# that shows it -- the anchored radius is smaller at all seven
# (10.3 11.1 8.5 7.8 7.9 7.8 7.7 against 12.8 12.0 9.7 9.5 8.6 8.8 8.9 px) --
# and the panel prints the two radii so the image is not asked to carry the
# claim on its own.
EYE = f'{FIG}/eye_frames_M21D19_trials.npz'
ez = np.load(EYE, allow_pickle=True)
ks = sorted(int(k) for k in ez.files)
PUP_R = dict(anch=10.3, non=12.8)
for lab, k in (('anch', ks[0]), ('non', ks[7])):
    g = ez[str(k)][:, :, ::-1].mean(2)
    ys, xs = np.nonzero(g < np.percentile(g, 3))      # the pupil is the darkest
    cy, cx, h = int(ys.mean()), int(xs.mean()), 46
    out[f'pupil_{lab}'] = ez[str(k)][max(0, cy - h):cy + h,
                                     max(0, cx - h):cx + h, ::-1]
    out[f'pupil_{lab}_r'] = np.array([PUP_R[lab]])
    print(f'  pupil {lab}: frame {k}, radius {PUP_R[lab]} px')

np.savez_compressed(OUT, **out)
print(f'wrote {OUT}')
