"""Per-cell, per-state summaries of what the rate map actually does.

Usage:  python3 build_trial_maps.py [worker n_workers]

Two supplements need quantities that cannot be read off the anchoring labels,
and both need the same expensive thing -- trial x position rate maps -- so they
share one pass.

WHAT "NON-ANCHORED" MEANS FOR A CELL (supplement S1). The label says a trial did
not match the others. Three very different things produce that and the label
cannot distinguish them:

    DRIFT / REMAP   the field is still coherent but sits somewhere else
    TUNING LOSS     there is no coherent field at any offset

So each trial's smoothed map is cross-correlated against the cell's ANCHORED
TEMPLATE (the mean map over its anchored trials) at every circular offset:

    r0        correlation at zero offset -- how the label itself is judged
    r_best    the best correlation over offsets
    lag_best  where that best sits, in cm

A non-anchored trial with high r_best at a large lag is a displaced field; one
with low r_best at any lag has no field to displace. The template is built from
anchored trials only, so it is not contaminated by the trials being scored
against it, and a cell with fewer than MIN_TEMPLATE anchored trials is skipped
rather than scored against a template made of noise.

DOES FIRING RATE CHANGE WITH STATE (supplement S2). Mean rate per trial, split
by state. Figure 2 is entirely about spatial stability and says nothing about
rate, which matters most for the putative interneurons -- they lock into the
anchored state, and whether that comes with a rate change decides whether
anchoring looks like an excitability state or a purely spatial one.

Rates are computed over RUNNING samples only, matching how the maps are built,
so a state difference in how much the animal stopped cannot masquerade as a
rate difference.

Writes data/population_state/trial_map_stats.csv, one row per cell.
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
OUT = f'{PS}/trial_map_stats.csv'
SIGMA = 2.0
MIN_TEMPLATE = 8        # anchored trials needed before a template is trusted
MIN_STATE = 5           # trials in a state before it is summarised

_g = {}
for _c in json.load(open(f'{FIG}/lick_raster_by_trial_type.ipynb'))['cells']:
    if _c['cell_type'] != 'code':
        continue
    _s = ''.join(_c['source'])
    if _s.startswith('MOUSE, DAY') or 'INV = pd.DataFrame' in _s:
        continue
    exec(compile(_s, '<nb>', 'exec'), _g)
globals().update({k: v for k, v in _g.items() if not k.startswith('__')})


def xcorr_profile(a, b):
    """Best circular correlation of a against b, and where it sits."""
    a = a - np.nanmean(a); b = b - np.nanmean(b)
    if np.nanstd(a) == 0 or np.nanstd(b) == 0:
        return np.nan, np.nan, np.nan
    n = len(a)
    denom = np.sqrt(np.nansum(a ** 2) * np.nansum(b ** 2))
    if denom == 0:
        return np.nan, np.nan, np.nan
    cc = np.array([np.nansum(np.roll(a, k) * b) for k in range(n)]) / denom
    k = int(np.argmax(cc))
    lag = k if k <= n // 2 else k - n           # signed, in bins
    return float(cc[0]), float(cc[k]), float(lag)


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
    have = [c for c in ids if c in set(int(x) for x in clusters.index)]
    if not have:
        return None
    tc = nap.compute_1d_tuning_curves(clusters[have], dt, nb_bins=n_all * NBIN,
                                      minmax=[0, n_all * TL], ep=moving)
    frac = np.asarray(z['frac_anch'], float)
    pop = frac > .5
    bin_cm = TL / NBIN
    rows = []
    for c in have:
        i = ids.index(c)
        lab = z['labels'][i]
        M = np.asarray(tc[c]).reshape(n_all, NBIN)[keep]
        S = np.array([smooth_nanaware(r, sigma=SIGMA) for r in M])
        rate = np.nanmean(M, axis=1)                 # running-only mean rate
        anch = lab == 1
        non = lab == 0
        rec = dict(mouse=mo, day=dy, cluster_id=c,
                   rate_anch=float(np.nanmean(rate[anch])) if anch.sum() >= MIN_STATE
                   else np.nan,
                   rate_non=float(np.nanmean(rate[non])) if non.sum() >= MIN_STATE
                   else np.nan,
                   rate_pop_anch=float(np.nanmean(rate[pop])) if pop.sum() >= MIN_STATE
                   else np.nan,
                   rate_pop_non=float(np.nanmean(rate[~pop])) if (~pop).sum() >= MIN_STATE
                   else np.nan,
                   n_anch=int(anch.sum()), n_non=int(non.sum()))
        if anch.sum() >= MIN_TEMPLATE:
            tmpl = np.nanmean(S[anch], axis=0)
            for tag, m in (('anch', anch), ('non', non)):
                if m.sum() < MIN_STATE:
                    continue
                r0, rb, lg = [], [], []
                for t in np.where(m)[0]:
                    a, b, l = xcorr_profile(S[t], tmpl)
                    if np.isfinite(a):
                        r0.append(a); rb.append(b); lg.append(abs(l) * bin_cm)
                if r0:
                    rec[f'r0_{tag}'] = float(np.mean(r0))
                    rec[f'rbest_{tag}'] = float(np.mean(rb))
                    rec[f'lag_{tag}'] = float(np.mean(lg))
                    rec[f'gain_{tag}'] = float(np.mean(np.array(rb) - np.array(r0)))
        rows.append(rec)
    return pd.DataFrame(rows)


if __name__ == '__main__':
    w, nw = (int(sys.argv[1]), int(sys.argv[2])) if len(sys.argv) > 2 else (0, 1)
    sess = sorted({(int(f.split('M')[1].split('D')[0]),
                    int(f.split('D')[1].split('.')[0]))
                   for f in os.listdir(f'{PS}/labels') if f.endswith('.npz')})
    mine = [s for i, s in enumerate(sess) if i % nw == w]
    path = OUT if nw == 1 else OUT.replace('.csv', f'_w{w}.csv')
    out, t0 = [], time.time()
    if os.path.exists(path):
        out.append(pd.read_csv(path))
    done = (set(map(tuple, out[0][['mouse', 'day']].drop_duplicates().values))
            if out else set())
    print(f'worker {w}/{nw}: {len(mine)} sessions, {len(done)} done', flush=True)
    for mo, dy in mine:
        if (mo, dy) in done:
            continue
        try:
            t = session(mo, dy)
        except Exception as e:
            print(f'  ! M{mo}D{dy}: {type(e).__name__}: {e}', flush=True); continue
        if t is None or not len(t):
            continue
        out.append(t)
        pd.concat(out, ignore_index=True).to_csv(path, index=False)
        print(f'  M{mo}D{dy}: {len(t)} cells, {time.time() - t0:.0f}s', flush=True)
    print(f'worker {w} done in {time.time() - t0:.0f}s', flush=True)
