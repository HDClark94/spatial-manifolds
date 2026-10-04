"""Per-trial firing rate and running speed, with a speed-matched split.

Usage:  python3 trial_rate_speed.py [worker n_workers]

The rate supplement finds that cells fire less when the population is anchored.
Running speed also differs between the states, and rate depends on speed, so
the whole effect could be a behavioural difference wearing a neural label.

This recomputes the same comparison on a SPEED-MATCHED subset of trials:
within each session the anchored and non-anchored trials are matched in deciles
of mean running speed and thinned to equal counts per decile, so the two states
are compared over the same distribution of speeds. Both the matched and the
unmatched numbers are kept, along with the mean speed in each, so the size of
the correction is visible rather than assumed.

No rate maps are needed here -- only spike counts and speed per trial -- so this
is a much cheaper pass than build_trial_maps.py.

Rates use RUNNING samples only (>= 3 cm/s), matching every other rate in the
paper, and a trial's duration is its running duration, so a trial where the
animal stopped a lot is not scored as low rate for that reason.

Writes data/population_state/trial_rate_speed.csv, one row per cell.
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
from spatial_manifolds.anchoring import load_session_labels

ROOT = '/Users/harryclark/Documents/spatial-manifolds'
FIG = f'{ROOT}/scripts/figures/AnchorDynamics2026'
PS = f'{ROOT}/data/population_state'
OUT = f'{PS}/trial_rate_speed.csv'
RUN, NQ, MIN_STATE = 3.0, 10, 5

_g = {}
for _c in json.load(open(f'{FIG}/lick_raster_by_trial_type.ipynb'))['cells']:
    if _c['cell_type'] != 'code':
        continue
    _s = ''.join(_c['source'])
    if _s.startswith('MOUSE, DAY') or 'INV = pd.DataFrame' in _s:
        continue
    exec(compile(_s, '<nb>', 'exec'), _g)
globals().update({k: v for k, v in _g.items() if not k.startswith('__')})


def matched(speed, anch, rng, nq=NQ):
    """Equal trial counts per state within deciles of running speed."""
    ok = np.isfinite(speed)
    q = np.quantile(speed[ok], np.linspace(0, 1, nq + 1))
    q[0] -= 1; q[-1] += 1
    ka, kn = [], []
    for i in range(nq):
        m = ok & (speed >= q[i]) & (speed < q[i + 1])
        a, n = np.where(m & anch)[0], np.where(m & ~anch)[0]
        k = min(len(a), len(n))
        if k:
            ka += list(rng.choice(a, k, replace=False))
            kn += list(rng.choice(n, k, replace=False))
    return np.array(ka, int), np.array(kn, int)


def session(mo, dy, rng):
    z = load_session_labels(mo, dy)
    if z is None:
        return None
    bp, cp = vr_paths(mo, dy)
    if not (os.path.exists(bp) and os.path.exists(cp)):
        return None
    beh = nap.load_file(bp); clusters = nap.load_file(cp)
    trials, _ = clip_trials(beh['trials'].as_dataframe(), clusters)
    trials['number'] = trials.number.astype(int)
    ids_lab = [int(c) for c in z['cluster_id']]
    T = trials.set_index('number').loc[[t for t in z['trial'].astype(int)]]
    S = beh['S']
    st, sv = np.asarray(S.index), np.asarray(S.values, float)
    dt_s = float(np.median(np.diff(st)))
    spd, dur, wins = [], [], []
    for s0, e0 in zip(T.start.values.astype(float), T.end.values.astype(float)):
        m = (st >= s0) & (st <= e0) & (sv >= RUN)
        spd.append(float(sv[m].mean()) if m.sum() > 5 else np.nan)
        dur.append(float(m.sum() * dt_s))
        wins.append(st[m])
    spd = np.array(spd); dur = np.array(dur)
    pop = np.asarray(z['frac_anch'], float) > .5
    ka, kn = matched(spd, pop, rng)
    have = [c for c in ids_lab if int(c) in set(int(x) for x in clusters.index)]
    rows = []
    for c in have:
        t = np.asarray(clusters[c].index, float)
        cnt = np.array([np.sum((t >= w[0]) & (t <= w[-1])) if len(w) else 0
                        for w in wins], float)
        with np.errstate(invalid='ignore', divide='ignore'):
            rate = np.where(dur > 0, cnt / np.maximum(dur, 1e-9), np.nan)

        def mean_of(idx):
            v = rate[idx]
            v = v[np.isfinite(v)]
            return float(v.mean()) if len(v) >= MIN_STATE else np.nan

        rows.append(dict(
            mouse=mo, day=dy, cluster_id=int(c),
            rate_a=mean_of(np.where(pop)[0]), rate_n=mean_of(np.where(~pop)[0]),
            rate_a_sm=mean_of(ka), rate_n_sm=mean_of(kn),
            spd_a=float(np.nanmean(spd[pop])), spd_n=float(np.nanmean(spd[~pop])),
            spd_a_sm=float(np.nanmean(spd[ka])) if len(ka) else np.nan,
            spd_n_sm=float(np.nanmean(spd[kn])) if len(kn) else np.nan,
            n_match=len(ka)))
    return pd.DataFrame(rows)


if __name__ == '__main__':
    w, nw = (int(sys.argv[1]), int(sys.argv[2])) if len(sys.argv) > 2 else (0, 1)
    sess = sorted({(int(f.split('M')[1].split('D')[0]),
                    int(f.split('D')[1].split('.')[0]))
                   for f in os.listdir(f'{PS}/labels') if f.endswith('.npz')})
    mine = [s for i, s in enumerate(sess) if i % nw == w]
    path = OUT if nw == 1 else OUT.replace('.csv', f'_w{w}.csv')
    out, t0 = [], time.time()
    print(f'worker {w}/{nw}: {len(mine)} sessions', flush=True)
    for mo, dy in mine:
        try:
            t = session(mo, dy, np.random.default_rng(abs(hash((mo, dy))) % 2**32))
        except Exception as e:
            print(f'  ! M{mo}D{dy}: {type(e).__name__}: {e}', flush=True); continue
        if t is None or not len(t):
            continue
        out.append(t)
        pd.concat(out, ignore_index=True).to_csv(path, index=False)
        print(f'  M{mo}D{dy}: {len(t)} cells, {time.time()-t0:.0f}s', flush=True)
    print(f'worker {w} done in {time.time()-t0:.0f}s', flush=True)
