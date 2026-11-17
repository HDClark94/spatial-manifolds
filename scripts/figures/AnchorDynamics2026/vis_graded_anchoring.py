"""Are visual cells on their non-anchored trials not anchored, or just less so?

The labels split each cell's trials into two groups by 2-means on how well each
trial matches the others, so 'non-anchored' says only that a trial falls in the
lower group. Here every trial of every cell gets a continuous score:

  g = r0 - r_null
  r0      correlation of the trial's smoothed rate map with the cell's anchored
          template (mean map over its anchored trials, the trial itself left out)
  r_null  the same trial circularly shifted to N_SHIFT random offsets of at least
          MIN_SHIFT_CM, mean correlation -- what a map with this trial's
          smoothness reaches by chance against this template

g = 0: no anchored structure at all. g > 0: some remains.

Summarised per cell over (i) its own non-anchored trials and (ii) the trials on
which the MEC POPULATION is non-anchored (pc1 < 0), for visual cortex and MEC.
Cells locked into one mode have no own non-anchored trials and enter (ii) only.

Writes data/population_state/vis_graded_anchoring.csv (per cell) and
       data/population_state/vis_graded_anchoring_trials.npz (per trial)
"""
import json
import os
import sys
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
OUT = f'{PS}/vis_graded_anchoring.csv'
OUT_TR = f'{PS}/vis_graded_anchoring_trials.npz'
MIN_TEMPLATE = 8
N_SHIFT = 20
MIN_SHIFT_CM = 30.0
MIN_VIS = 5

_g = {}
for _c in json.load(open(f'{FIG}/lick_raster_by_trial_type.ipynb'))['cells']:
    if _c['cell_type'] != 'code':
        continue
    _s = ''.join(_c['source'])
    if _s.startswith('MOUSE, DAY') or 'INV = pd.DataFrame' in _s:
        continue
    exec(compile(_s, '<nb>', 'exec'), _g)
vr_paths, clip_trials, TL, NBIN = _g['vr_paths'], _g['clip_trials'], _g['TL'], _g['NBIN']
REG = pd.read_csv(f'{PS}/pc1_by_region.csv')


def corr_rows(A, b):
    A = A - A.mean(1, keepdims=True); b = b - b.mean()
    d = np.sqrt((A ** 2).sum(1) * (b ** 2).sum())
    return np.where(d > 0, (A @ b) / np.where(d > 0, d, 1), np.nan)


def region(acr):
    a = str(acr)
    return 'MEC' if a.startswith('ENTm') else 'VIS' if a.startswith('VIS') else None


def session(mo, dy, rng):
    z = load_session_labels(mo, dy)
    if z is None:
        return None, None
    rr = REG[(REG.mouse == mo) & (REG.day == dy)].set_index('cluster_id')
    ids = [int(c) for c in z['cluster_id']]
    reg = np.array([region(rr.brain_region.get(c, '')) for c in ids])
    if (reg == 'VIS').sum() < MIN_VIS:
        return None, None
    bp, cp = vr_paths(mo, dy)
    beh = nap.load_file(bp); clusters = nap.load_file(cp)
    trials_all = beh['trials'].as_dataframe()
    trials, orig = clip_trials(trials_all, clusters)
    keep = np.isin(trials_all.number.values.astype(int), orig)
    tn, trav = beh['trial_number'], beh['travel']
    dtrav = trav - (float(np.asarray(tn.values)[0]) - 1) * TL
    moving = beh['S'].threshold(3.0, method='above').time_support
    n_all = len(trials_all)
    use = [i for i, c in enumerate(ids) if reg[i] in ('MEC', 'VIS') and c in clusters.index]
    tc = nap.compute_1d_tuning_curves(clusters[[ids[i] for i in use]], dtrav,
                                      nb_bins=n_all * NBIN, minmax=[0, n_all * TL], ep=moving)
    pop_non = np.asarray(z['pc1'], float) < 0
    smin = int(MIN_SHIFT_CM / (TL / NBIN))
    rows, per_trial = [], []
    for i in use:
        c = ids[i]
        M = np.asarray(tc[c]).reshape(n_all, NBIN)[keep]
        S = np.array([smooth_nanaware(r) for r in M])
        S = np.nan_to_num(S)
        lab = np.asarray(z['labels'][i], float)
        anch = lab == 1
        if anch.sum() < MIN_TEMPLATE or (S.sum(1) > 0).sum() < MIN_TEMPLATE:
            continue
        tot = S[anch].sum(0)
        g = np.full(len(S), np.nan)
        for t in range(len(S)):
            if S[t].sum() == 0:
                continue
            tmpl = (tot - S[t]) / (anch.sum() - 1) if anch[t] else tot / anch.sum()
            r0 = corr_rows(S[t][None], tmpl)[0]
            sh = rng.integers(smin, NBIN - smin, N_SHIFT)
            rn = np.nanmean(corr_rows(np.array([np.roll(S[t], k) for k in sh]), tmpl))
            g[t] = r0 - rn
        own_non = lab == 0
        rec = dict(mouse=mo, day=dy, cluster_id=c, region=reg[i],
                   locked=bool(np.nanstd(lab) == 0), n_anch=int(anch.sum()), n_non=int(own_non.sum()),
                   g_anch=np.nanmean(g[anch]),
                   g_non=np.nanmean(g[own_non]) if own_non.sum() >= 3 else np.nan,
                   g_pop_anch=np.nanmean(g[~pop_non]), g_pop_non=np.nanmean(g[pop_non]))
        # is this cell's own non-anchored residue above zero? sign-flip across trials
        v = g[own_non][np.isfinite(g[own_non])]
        if len(v) >= 5:
            from scipy.stats import wilcoxon
            rec['p_non'] = float(wilcoxon(v, alternative='greater').pvalue)
        rows.append(rec)
        per_trial.append((mo, dy, c, reg[i], g, lab, pop_non))
    return pd.DataFrame(rows), per_trial


if __name__ == '__main__':
    sess = sorted({(int(a), int(b)) for a, b in
                   REG[REG.brain_region.astype(str).str.startswith('VIS')][['mouse', 'day']].values})
    rng = np.random.default_rng(0)
    out, trials = [], []
    for mo, dy in sess:
        try:
            d, pt = session(mo, dy, rng)
        except Exception as e:
            print(f'  ! M{mo}D{dy}: {type(e).__name__}: {e}', flush=True); continue
        if d is None or not len(d):
            continue
        out.append(d); trials += pt
        print(f'  M{mo}D{dy}: {int((d.region == "VIS").sum())} VIS, {int((d.region == "MEC").sum())} MEC',
              flush=True)
    D = pd.concat(out, ignore_index=True)
    D.to_csv(OUT, index=False)
    np.savez_compressed(OUT_TR, key=np.array([f'{a}_{b}_{c}' for a, b, c, *_ in trials]),
                        region=np.array([r for *_, r, g, l, p in [(x[0], x[1], x[2], x[3], x[4], x[5], x[6]) for x in trials]]),
                        g=np.array([x[4] for x in trials], dtype=object),
                        lab=np.array([x[5] for x in trials], dtype=object),
                        pop_non=np.array([x[6] for x in trials], dtype=object))
    print('wrote', OUT)
