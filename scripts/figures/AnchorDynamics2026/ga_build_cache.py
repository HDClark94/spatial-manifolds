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
# ---- one pupil image per state, same mouse, same place on the track --------
# Both frames come from M21 D19 -- the same mouse and the same session -- and
# from the two trials Figure 6 uses as its anchored and non-anchored examples.
# They are matched on track position: 20 cm, early on the track and clear of
# the teleport at 0. Figure 6's own frame grid is [0, 30, 60, 90, 120, 150,
# 170], so 20 cm is grabbed fresh here rather than taken from the cache.
import importlib.util                                          # noqa: E402
_spec = importlib.util.spec_from_file_location(
    'f5', f'{FIG}/fig5_pupil_arousal.py')
f5 = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(f5)          # guarded by __main__, so nothing is built

EMO, EDY, PUP_POS, TOL = 21, 19, 20.0, 5.0
EYE = f'{FIG}/eye_frames_M{EMO}D{EDY}_trials.npz'
vid = f5.find_video(EMO, EDY)


def crop_pupil(fr):
    """Square crop centred on the darkest region, which is the pupil."""
    g = fr.mean(2)
    ys, xs = np.nonzero(g < np.percentile(g, 3))
    cy, cx, h = int(ys.mean()), int(xs.mean()), 46
    return fr[max(0, cy - h):cy + h, max(0, cx - h):cx + h]


if vid is not None:
    # the volume is up: take the frames at the position actually wanted
    import cv2                                                 # noqa: E402
    S5 = f5.load_session(EMO, EDY)
    blocks = f5.state_blocks(EMO, EDY, S5, f5.population(EMO, EDY))
    cap = cv2.VideoCapture(vid)
    # the full-frame videos need the eye crop from all_eye_crops.csv before
    # anything is looked for inside them; without it the darkest-region search
    # finds a dark corner of the rig rather than the pupil
    if 'eye_zone' in os.path.basename(vid):
        cx0 = cy0 = 0
        cw = ch = None
    else:
        _r = f5.crops()
        _r = _r[(_r.mouse == EMO) & (_r.day == EDY) &
                (_r.session.astype(str).str.lower() == 'vr')].iloc[0]
        cx0, cy0, cw, ch = int(_r.x), int(_r.y), int(_r.w), int(_r.h)
    for lab in ('anch', 'non'):
        a, b, _, _ = blocks[lab]
        u, _ = f5.pick_trial(S5, a, b)
        m = np.where((S5['tr'] == u) & S5['good'] & (S5['spd'] >= 3.0) &
                     (np.abs(S5['pos'] - PUP_POS) <= TOL))[0]
        i = int(m[np.argmin(np.abs(S5['pos'][m] - PUP_POS))])
        cap.set(cv2.CAP_PROP_POS_FRAMES, i)
        ok, fr = cap.read()
        if not ok:
            raise SystemExit(f'{lab}: frame {i} unreadable')
        fr = fr if cw is None else fr[cy0:cy0 + ch, cx0:cx0 + cw]
        out[f'pupil_{lab}'] = crop_pupil(fr[:, :, ::-1])
        out[f'pupil_{lab}_r'] = np.array([float(S5['rad'][i])])
        out[f'pupil_{lab}_pos'] = np.array([float(S5['pos'][i])])
        print(f'  pupil {lab}: M{EMO}D{EDY} trial {u}, frame {i}, '
              f'{S5["pos"][i]:.0f} cm, radius {S5["rad"][i]:.1f} px')
    cap.release()
else:
    # The eye videos live on the lab volume, not on the microSD copy, so when
    # only the SD is mounted the nearest CACHED matched position is used.
    # Figure 6 grabs its frames at [0, 30, 60, 90, 120, 150, 170] cm, so 30 cm
    # is the nearest to the 20 cm wanted, and it is on the track proper rather
    # than at the teleport. Both frames are the same mouse, the same session
    # and the same two trials Figure 6 uses, matched on position.
    SHOW_POS = [0, 30, 60, 90, 120, 150, 170]
    RADII = {'anch': [10.3, 11.1, 8.5, 7.8, 7.9, 7.8, 7.7],
             'non': [12.8, 12.0, 9.7, 9.5, 8.6, 8.8, 8.9]}
    j = int(np.argmin(np.abs(np.array(SHOW_POS) - PUP_POS)))
    ez = np.load(EYE, allow_pickle=True)
    ks = sorted(int(k) for k in ez.files)
    for lab, k in (('anch', ks[j]), ('non', ks[7 + j])):
        out[f'pupil_{lab}'] = crop_pupil(ez[str(k)][:, :, ::-1])
        out[f'pupil_{lab}_r'] = np.array([RADII[lab][j]])
        out[f'pupil_{lab}_pos'] = np.array([float(SHOW_POS[j])])
        print(f'  pupil {lab}: M{EMO}D{EDY} cached frame {k}, '
              f'{SHOW_POS[j]} cm, radius {RADII[lab][j]} px '
              f'(lab volume down; {PUP_POS:.0f} cm not cached)')

np.savez_compressed(OUT, **out)
print(f'wrote {OUT}')
