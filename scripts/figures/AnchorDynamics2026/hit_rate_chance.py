"""A chance level for the hit rate in Figure 1G.

A hit is a stop in the reward zone, so the question a chance level has to
answer is: given how this animal stopped on this trial, how often would it
have hit the zone had the zone been somewhere else? The null therefore takes
each trial's stops and shifts the whole set rigidly around the track. That
keeps the number of stops and the spacing between them -- both of which matter
-- and destroys only their alignment to the reward zone.

Two simpler nulls are wrong here and worth naming. The reward zone is a tenth
of the track, but 20/200 is the chance for a SINGLE stop. Treating a trial's n
stops as independent, 1 - (1 - 0.1)^n, overstates it badly, because stops
within a trial come in clusters a few centimetres apart and a rigid shift
moves them together: among trials with two or more stops the independent
formula gives 0.475 where the shuffle gives 0.228.

Trials with no stop at all are kept, as automatic misses under both the
observation and the null, because the figure's axis is a hit rate over trials.

Writes data/population_state/hit_rate_chance.csv
"""
import json
import os
import sys
import warnings

import numpy as np
import pandas as pd

warnings.filterwarnings('ignore')
FIG = os.path.dirname(os.path.abspath(__file__))
ROOT = '/Users/harryclark/Documents/spatial-manifolds'
sys.path.insert(0, f'{ROOT}/src')

_g = {}
for _c in json.load(open(f'{FIG}/lick_raster_by_trial_type.ipynb'))['cells']:
    if _c['cell_type'] != 'code':
        continue
    _s = ''.join(_c['source'])
    if _s.startswith('MOUSE, DAY') or 'INV = pd.DataFrame' in _s:
        continue
    exec(compile(_s, '<nb>', 'exec'), _g)
globals().update({k: v for k, v in _g.items() if not k.startswith('__')})

NDRAW = 500
OUT = f'{ROOT}/data/population_state/hit_rate_chance.csv'


def session(mo, dy, rng):
    bp, cp = vr_paths(mo, dy)                                   # noqa: F821
    if not (os.path.exists(bp) and os.path.exists(cp)):
        return None
    beh = nap.load_file(bp)                                     # noqa: F821
    clusters = nap.load_file(cp)                                # noqa: F821
    trials, _ = clip_trials(beh['trials'].as_dataframe(), clusters)  # noqa
    S = beh['S']
    st, sv = np.asarray(S.index), np.asarray(S.values, float)
    tv = beh['travel']
    tt, pv = np.asarray(tv.index), np.asarray(tv.values, float)
    # a stop is the onset of a contiguous period below the reward criterion
    onset = st[np.flatnonzero(np.diff((sv < STOP_SPEED).astype(int)) == 1) + 1]
    opos = np.interp(onset, tt, pv) % TL                        # noqa: F821
    rows = []
    for _, r in trials.iterrows():
        p = opos[(onset >= r.start) & (onset <= r.end)]
        if len(p):
            obs = int(((p >= RZ[0]) & (p <= RZ[1])).any())      # noqa: F821
            q = (p[None, :] + rng.uniform(0, TL, NDRAW)[:, None]) % TL  # noqa
            ch = float(((q >= RZ[0]) & (q <= RZ[1])).any(1).mean())  # noqa
        else:
            obs, ch = 0, 0.0
        rows.append(dict(mouse=mo, day=dy, trial=int(r.number),
                         ttype=r.type, n_stop=len(p), obs=obs, chance=ch))
    return pd.DataFrame(rows)


def main():
    A = pd.read_csv(f'{ROOT}/data/population_state/trial_table.csv')
    rng = np.random.default_rng(0)
    out = []
    for mo, dy in A[['mouse', 'day']].drop_duplicates().values:
        try:
            d = session(int(mo), int(dy), rng)
        except Exception as e:
            print(f'  ! M{mo}D{dy}: {type(e).__name__}: {e}', flush=True)
            continue
        if d is None:
            continue
        out.append(d)
        print(f'  M{mo}D{dy}: {len(d):4d} trials, {d.n_stop.mean():.1f} '
              f'stops/trial, hit {d.obs.mean():.3f}, chance '
              f'{d.chance.mean():.3f}', flush=True)
    D = pd.concat(out, ignore_index=True)
    D = D.merge(A[['mouse', 'day', 'trial', 'hit', 'pc1']],
                on=['mouse', 'day', 'trial'], how='inner')
    D['anch'] = D.pc1 > 0
    D.to_csv(OUT, index=False)
    print(f'\n{D.groupby(["mouse", "day"]).ngroups} sessions, {len(D)} trials')
    print(f'  reconstructed hit {D.obs.mean():.3f} against the task\'s own '
          f'label {D.hit.mean():.3f}  (agreement '
          f'{(D.obs == D.hit).mean():.3f})')
    for tt, nm in (('b', 'cued'), ('nb', 'uncued')):
        q = D[D.ttype == tt]
        s = q.groupby(['mouse', 'day']).agg(h=('hit', 'mean'),
                                            c=('chance', 'mean'))
        print(f'  {nm:7s}: hit {s.h.mean():.3f}   chance {s.c.mean():.3f} '
              f'(SD {s.c.std():.3f}, range {s.c.min():.3f}-{s.c.max():.3f}), '
              f'{len(s)} sessions')
    print(f'wrote {OUT}')


if __name__ == '__main__':
    main()
