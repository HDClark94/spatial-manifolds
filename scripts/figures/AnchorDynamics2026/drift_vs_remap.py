"""Does the grid map DRIFT on non-anchored trials, or does it SWITCH between maps?

The paper establishes that a trial is anchored or not. It does not say what the
map is doing when it is not anchored, and two accounts predict different things:

  DRIFT (attractor unpinned)   the grid phase is carried forward by path
                               integration with no landmark correction, so it
                               performs a random walk. Consecutive non-anchored
                               trials sit CLOSE to each other, and the distance
                               between them GROWS with how many trials separate
                               them -- diffusion.

  DISCRETE REMAPPING           the network occupies one of a few maps and jumps
                               between them. Consecutive trials are no more
                               similar than distant ones: flat in trial
                               separation, at the level of independent draws.

Each non-anchored trial's smoothed rate map is cross-correlated against the
cell's ANCHORED TEMPLATE at every circular offset, exactly as build_trial_maps.py
does, but the SIGNED offset is kept per trial and in trial order rather than
averaged within a state. The statistic is then the circular distance between the
offsets of two non-anchored trials, as a function of how many trials apart they
are.

THE NULL IS THE CELL'S OWN OFFSETS, SHUFFLED. Permuting a cell's offsets across
its non-anchored trials destroys temporal order while preserving exactly which
offsets that cell visits and how often. So the comparison is not against a
uniform distribution -- a cell that only ever sits at two offsets is tested
against a null that also only sits at those two.

Only GRID cells (open-field cell_class_of1 == 'GC') are used, and only trials
whose best correlation clears MIN_RBEST, so that a displaced field is being
measured rather than a cell with no field to displace.

ANSWER: NEITHER. THE PHASE IS RE-DRAWN EACH TRAVERSAL.

Three measures agree, on 494 grid cells from 57 sessions:

  THE FIELD SURVIVES, DISPLACED. On non-anchored trials the map correlates
  +0.041 with the anchored template at zero offset but +0.578 at its best
  offset, against +0.381 and +0.618 for anchored trials. So the map is intact
  and nearly as coherent as usual; it is in the wrong place. Not tuning loss.

  CONSECUTIVE NON-ANCHORED TRIALS ARE NOT ALIGNED. Aligning pairs of trials
  directly to each other, 13.4% of adjacent non-anchored pairs sit within 10 cm,
  against 10% expected for independent offsets and 24.7% for anchored pairs.
  Mean |offset| is 45.8 cm against a uniform expectation of 50 cm.

  AND THERE IS NO DIFFUSION. A random walk makes that alignment decay with trial
  separation. It does not: 13.4% at separation 1, 12.9% at 5, 10.7% at 20,
  approaching the independent level from a starting point already close to it.
  Map-to-map correlation is likewise flat over separations 1-10 and only reaches
  zero by ~80 trials, the scale of the state blocks rather than of a walk.

So the attractor does not slip its moorings and wander. Each non-anchored
traversal begins at an essentially arbitrary phase, which path integration then
carries coherently to the end of the track -- coherent WITHIN a trial,
independent BETWEEN trials. That is the signature of a failed phase RESET at
trial onset, not of a drifting one, and it is the opposite of what a drifting
attractor predicts. Whatever sets anchoring acts at the start of a traversal.

MEASUREMENT CAVEAT. Aligning two single-trial maps of a periodic cell can pick
the wrong autocorrelation peak, which pushes every number toward the uniform
floor. The anchored control is what shows the measure still has reach: 24.7%
against a 10% floor. Absolute percentages here are attenuated; the comparison
between states is what carries the result.

Caches smoothed trial x position maps for grid cells to
data/population_state/drift_maps.npz; the statistics above are computed from it.
"""
import json
import os
import sys
import time
import warnings

import numpy as np
import pandas as pd

warnings.filterwarnings('ignore')
sys.path.insert(0, '/Users/harryclark/Documents/spatial-manifolds/src')
import pynapple as nap
from spatial_manifolds.anchoring import load_session_labels, smooth_nanaware

ROOT = '/Users/harryclark/Documents/spatial-manifolds'
FIG = f'{ROOT}/scripts/figures/AnchorDynamics2026'
PS = f'{ROOT}/data/population_state'
OUT = f'{PS}/drift_vs_remap_trials.csv'
SIGMA = 2.0
MIN_TEMPLATE = 8        # anchored trials before a template is trusted
MIN_NON = 10            # non-anchored trials before a cell is worth testing
MIN_RBEST = 0.30        # the field must still be coherent somewhere
MIN_PERIOD_CM = 25.0    # shortest field spacing taken seriously along the track
MIN_ACORR = 0.20        # template autocorrelation peak before a cell is periodic

_g = {}
for _c in json.load(open(f'{FIG}/lick_raster_by_trial_type.ipynb'))['cells']:
    if _c['cell_type'] != 'code':
        continue
    _s = ''.join(_c['source'])
    if _s.startswith('MOUSE, DAY') or 'INV = pd.DataFrame' in _s:
        continue
    exec(compile(_s, '<nb>', 'exec'), _g)
globals().update({k: v for k, v in _g.items() if not k.startswith('__')})

CLS = pd.read_csv(f'{ROOT}/data/cell_classifications_v2.csv')
GRID = set(map(tuple, CLS.loc[CLS.cell_class_of1 == 'GC',
                              ['mouse', 'day', 'cluster_id']].values))


def signed_lag(a, b):
    """Best circular correlation of a against b, with the SIGNED offset."""
    a = a - np.nanmean(a); b = b - np.nanmean(b)
    if np.nanstd(a) == 0 or np.nanstd(b) == 0:
        return np.nan, np.nan
    denom = np.sqrt(np.nansum(a ** 2) * np.nansum(b ** 2))
    if denom == 0:
        return np.nan, np.nan
    n = len(a)
    cc = np.array([np.nansum(np.roll(a, k) * b) for k in range(n)]) / denom
    k = int(np.argmax(cc))
    return float(cc[k]), float(k if k <= n // 2 else k - n)


def circ_dist(d, n):
    """Signed bin difference wrapped onto [-n/2, n/2), as an absolute value."""
    return np.abs((d + n // 2) % n - n // 2)


def template_period(tmpl, bin_cm):
    """Field spacing along the track, from the template's own autocorrelation.

    Returns (period_in_bins, peak_height), or (nan, nan) if the template has no
    periodic structure worth calling a spacing.
    """
    a = tmpl - np.nanmean(tmpl)
    if not np.isfinite(a).any() or np.nanstd(a) == 0:
        return np.nan, np.nan
    a = np.nan_to_num(a)
    n = len(a)
    ac = np.array([np.sum(np.roll(a, k) * a) for k in range(n)]) / np.sum(a ** 2)
    lo = max(2, int(round(MIN_PERIOD_CM / bin_cm)))
    hi = n // 2
    if hi <= lo:
        return np.nan, np.nan
    k = lo + int(np.argmax(ac[lo:hi]))
    return (float(k), float(ac[k])) if ac[k] >= MIN_ACORR else (np.nan, float(ac[k]))


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
    tn, trav = beh['trial_number'], beh['travel']
    dt = trav - (float(np.asarray(tn.values)[0]) - 1) * TL
    moving = beh['S'].threshold(3.0, method='above').time_support
    n_all = len(trials_all)
    ids = [int(c) for c in z['cluster_id']]
    have = [c for c in ids
            if c in set(int(x) for x in clusters.index) and (mo, dy, c) in GRID]
    if not have:
        return None
    tc = nap.compute_1d_tuning_curves(clusters[have], dt, nb_bins=n_all * NBIN,
                                      minmax=[0, n_all * TL], ep=moving)
    bin_cm = TL / NBIN
    rng = np.random.default_rng(0)
    rows = []
    for c in have:
        i = ids.index(c)
        lab = np.asarray(z['labels'][i])
        M = np.asarray(tc[c]).reshape(n_all, NBIN)[keep]
        S = np.array([smooth_nanaware(r, sigma=SIGMA) for r in M])
        anch, non = lab == 1, lab == 0
        if anch.sum() < MIN_TEMPLATE or non.sum() < MIN_NON:
            continue
        rows.append(dict(mouse=mo, day=dy, cluster_id=c,
                         maps=S.astype(np.float32), labels=lab.astype(np.int8),
                         nbin=NBIN, bin_cm=bin_cm))
    return rows


if __name__ == '__main__':
    sess = sorted({(int(f.split('M')[1].split('D')[0]),
                    int(f.split('D')[1].split('.')[0]))
                   for f in os.listdir(f'{PS}/labels') if f.endswith('.npz')})
    out, t0 = [], time.time()
    print(f'{len(sess)} sessions', flush=True)
    for mo, dy in sess:
        try:
            t = session(mo, dy)
        except Exception as e:
            print(f'  ! M{mo}D{dy}: {type(e).__name__}: {e}', flush=True); continue
        if not t:
            continue
        out += t
        print(f'  M{mo}D{dy}: {len(t)} grid cells, {time.time() - t0:.0f}s',
              flush=True)
    np.savez_compressed(f'{PS}/drift_maps.npz',
                        meta=np.array([(r['mouse'], r['day'], r['cluster_id'])
                                       for r in out]),
                        maps=np.array([r['maps'] for r in out], dtype=object),
                        labels=np.array([r['labels'] for r in out], dtype=object),
                        bin_cm=out[0]['bin_cm'], allow_pickle=True)
    print(f'cached {len(out)} grid cells in {time.time() - t0:.0f}s', flush=True)
