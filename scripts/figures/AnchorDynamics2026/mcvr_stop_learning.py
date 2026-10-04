"""Where do the mice stop? The behavioural test of whether rz2 is ever learned.

Hit rate is too coarse to answer this. On CUED trials the reward zone is marked,
so an animal succeeds in a brand-new context without knowing anything -- 0.74 on
first exposure -- and pooling trial types hides that. The memory-dependent
question is where an animal chooses to stop on UNCUED trials, when nothing marks
the zone.

The two zones are 30 cm apart on a 230 cm track: rz1 at 92 cm, rz2 at 122 cm.
That separation makes the diagnostic sharp. An animal that has not learned rz2
should carry its rz1 habit into the novel context and stop near 92 cm on rz2
trials; one that has learned should stop near 122 cm. So:

  first-stop position   per trial, the first stop past the start box
  error                 signed distance from that trial's OWN reward zone
  perseveration         is the first stop ON the other context's zone?

Perseveration is the measure that matters. Hit rate conflates "did not know
where to stop" with "did not stop at all", whereas a first stop sitting on the
wrong zone is positive evidence of a specific wrong belief -- and its decline
over blocks and days is what learning looks like.

BUT ONLY IF IT MEANS THE WRONG ZONE. Scoring perseveration as "closer to the
other zone than to its own" -- the obvious definition -- is wrong, and visibly
so in the third block of training day 1, where the median first stop falls to
42-72 cm in five of six sessions. Those stops are nowhere near either zone
(92 and 122 cm); the animals are disengaging late in a long first session and
stopping early. The permissive rule scores every one of them as perseveration,
so the measure rises exactly where engagement collapses and reads as the animal
reverting to its old belief.

So a stop counts as being AT a zone only within +- ZONE_WIN cm of it, leaving a
gap around the 107 cm midpoint that belongs to neither:

  at_own / at_other     first stop within ZONE_WIN of that zone
  off_zone              within ZONE_WIN of neither -- the disengagement channel,
                        which has to be reported alongside, since a fall in
                        perseveration is only learning if off_zone is not what
                        absorbed it

Stops are contiguous runs below STOP_SPEED, reduced to one position each, so a
long pause counts once rather than in proportion to its duration.
"""
import os, sys, time, warnings
import numpy as np, pandas as pd
warnings.filterwarnings('ignore')
sys.path.insert(0, '/Users/harryclark/Documents/spatial-manifolds/src')
import pynapple as nap

SOURCE = '/Users/harryclark/Downloads/harry_multi_context_vr/dataset/'
PS = '/Users/harryclark/Documents/spatial-manifolds/data/population_state'
OUT = f'{PS}/mcvr_stops.csv'
RZ = {'rz1': 92.0, 'rz2': 122.0}
STOP_SPEED, MIN_POS, TL = 3.0, 30.0, 230.0
ZONE_WIN = 12.0     # zones are 30 cm apart, so +-12 leaves a 6 cm neutral gap


def _derive(A):
    """Zone-window columns, from first_stop alone -- no reload needed."""
    own = A.context.map(RZ).astype(float)
    other = A.context.map({'rz1': RZ['rz2'], 'rz2': RZ['rz1']}).astype(float)
    A['at_own'] = ((A.first_stop - own).abs() <= ZONE_WIN).astype(float)
    A['at_other'] = ((A.first_stop - other).abs() <= ZONE_WIN).astype(float)
    A['off_zone'] = 1.0 - A.at_own - A.at_other
    A.loc[A.first_stop.isna(), ['at_own', 'at_other', 'off_zone']] = np.nan
    # `persev` keeps the permissive definition for the record; `at_other` is the
    # one the figure uses.
    return A


def session_stops(mo, dy):
    p = f'{SOURCE}sub-M{mo}/sub-M{mo}_ses-D{dy}MCVR_ecephys+behavior.nwb'
    if not os.path.exists(p):
        return None
    d = nap.load_file(p)
    tr = d['trials'].as_dataframe()
    S, P = d['S'], d['P']
    t = np.asarray(S.index)
    sv = np.asarray(S.values, float)
    pos = np.asarray(P.values, float)[np.searchsorted(np.asarray(P.index), t)
                                      .clip(0, len(P) - 1)]
    below = sv < STOP_SPEED
    edges = np.diff(below.astype(int))
    starts = np.where(edges == 1)[0] + 1
    if below[0]:
        starts = np.r_[0, starts]
    stop_t, stop_p = t[starts], pos[starts]

    rows = []
    for _, r in tr.iterrows():
        m = (stop_t >= r.start) & (stop_t <= r.end) & (stop_p >= MIN_POS)
        sp = stop_p[m]
        first = float(sp[0]) if len(sp) else np.nan
        own = RZ.get(str(r.context), np.nan)
        other = RZ['rz2'] if str(r.context) == 'rz1' else RZ['rz1']
        # ALL stops, not just the first: how many fall in each zone window
        n_own = int((np.abs(sp - own) <= ZONE_WIN).sum())
        n_other = int((np.abs(sp - other) <= ZONE_WIN).sum())
        rows.append(dict(
            mouse=mo, day=dy, trial=int(r.number), ttype=str(r.type),
            context=str(r.context), hit=int(r.performance == 'hit'),
            first_stop=first, n_stops=int(m.sum()),
            n_own=n_own, n_other=n_other,
            err=first - own if np.isfinite(first) else np.nan,
            persev=(int(abs(first - other) < abs(first - own))
                    if np.isfinite(first) else np.nan)))
    return pd.DataFrame(rows)


def ensure_stops(rebuild=False, verbose=True):
    """First-stop table for every MCVR session, built on demand."""
    if os.path.exists(OUT) and not rebuild:
        A = _derive(pd.read_csv(OUT))
        if verbose:
            print(f'mcvr_stops.csv: {len(A)} trials, '
                  f'{A.groupby(["mouse","day"]).ngroups} sessions (cached)')
        return A
    return build_all(verbose=verbose)


def build_all(verbose=True):
    import glob, re
    sess = sorted({(int(m.group(1)), int(m.group(2))) for m in
                   (re.search(r'sub-M(\d+)_ses-D(\d+)MCVR', p) for p in
                    glob.glob(f'{SOURCE}sub-M*/sub-M*MCVR*.nwb')) if m})
    out = []
    for mo, dy in sess:
        try:
            s_ = session_stops(mo, dy)
        except Exception as e:
            print(f'  ! M{mo}D{dy}: {type(e).__name__}: {e}'); continue
        if s_ is not None:
            out.append(s_)
    A = pd.concat(out, ignore_index=True)
    A['idx'] = A.groupby('mouse').day.rank(method='dense').astype(int)
    A = _derive(A)
    A.to_csv(OUT, index=False)
    if verbose:
        print(f'built mcvr_stops.csv: {len(A)} trials, '
              f'{A.groupby(["mouse","day"]).ngroups} sessions')
    return A


if __name__ == '__main__':
    import glob, re
    sess = sorted({(int(m.group(1)), int(m.group(2))) for m in
                   (re.search(r'sub-M(\d+)_ses-D(\d+)MCVR', p) for p in
                    glob.glob(f'{SOURCE}sub-M*/sub-M*MCVR*.nwb')) if m})
    t0 = time.time()
    A = build_all()
    print(f'  ({time.time()-t0:.0f}s)\n')

    u = A[(A.ttype == 'nb') & A.first_stop.notna()]
    print('UNCUED trials — first stop position (cm), zones at rz1=92 rz2=122')
    print(u.groupby(['idx', 'context']).first_stop.median().unstack().round(1).to_string())
    print(f'\nUNCUED — perseveration: first stop WITHIN {ZONE_WIN:.0f} cm of the '
          'OTHER zone')
    print(u.groupby(['idx', 'context']).at_other.mean().unstack().round(3).to_string())
    print('\nUNCUED — first stop at its OWN zone')
    print(u.groupby(['idx', 'context']).at_own.mean().unstack().round(3).to_string())
    print('\nUNCUED — first stop at NEITHER zone (disengagement channel)')
    print(u.groupby(['idx', 'context']).off_zone.mean().unstack().round(3).to_string())
    print('\nUNCUED — |error| from own zone (cm)')
    print(u.groupby(['idx', 'context']).err.apply(lambda v: v.abs().median())
          .unstack().round(1).to_string())
    print('\nDAY 1 by block half, uncued rz2 only:')
    d1 = u[(u.idx == 1) & (u.context == 'rz2')].copy()
    d1['half'] = np.where(d1.groupby(['mouse', 'day']).trial.rank(pct=True) <= .5,
                          'first half', 'second half')
    print(d1.groupby('half').agg(first_stop=('first_stop', 'median'),
                                 persev=('persev', 'mean'),
                                 hit=('hit', 'mean'),
                                 n=('trial', 'size')).round(3).to_string())
    c = A[(A.ttype == 'b') & A.first_stop.notna()]
    print('\nCUED trials, for comparison — first stop by context and day')
    print(c.groupby(['idx', 'context']).first_stop.median().unstack().round(1).to_string())
