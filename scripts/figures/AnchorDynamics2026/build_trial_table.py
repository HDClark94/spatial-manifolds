"""One trial table for the whole project: behaviour joined to the anchoring state.

WHY THIS EXISTS. Figure 1's behavioural panels read `eye_anchoring_trials.csv`,
which restricts them to the 58 sessions that have eye tracking -- a constraint
with no scientific content here, since hit rates, trial types and the anchoring
state need no eye data. The eye table was simply where `frac_anch` happened to
live. The same accident had `theta_freq.py` and `lfp_batch.py` taking their
state from it too.

Figure 1 should depend on exactly two things: the anchoring labels (and the
shuffle-derived thresholds behind them) and, if it is ever restricted by cell
type, the classification datasheet. This table supplies both halves and nothing
else, over all 62 sessions with labels.

Columns are named to match the eye table (`ttype`, `perf`, `frac_anch`) so it is
a drop-in replacement.

    mouse, day, trial     trial numbering as in the label files (clip_trials,
                          renumbered from 1 -- NOT the raw trial numbers)
    ttype                 'b' cued / 'nb' uncued / 'p' probe
    perf                  hit / try / run / slow
    hit                   perf == 'hit'
    speed                 mean speed over RUNNING samples (>= 3 cm/s)
    frac_anch, pc1        population state, from the label build
    n_cells               cells contributing to that state

TRIAL NUMBERING IS THE TRAP. `clip_trials` renumbers surviving trials from 1 and
keeps the original in `orig_number`. The label files store the renumbered ids, so
behaviour must be taken through the same call; indexing the raw trials table by
those ids silently pairs each trial with another trial's type and outcome, which
is exactly the bug that made a supplement's trial-type agreement 0.52.
"""
import json, os, sys, time, warnings
import numpy as np, pandas as pd
warnings.filterwarnings('ignore')
sys.path.insert(0, '/Users/harryclark/Documents/spatial-manifolds/src')
import pynapple as nap
from spatial_manifolds.anchoring import load_session_labels

NB = ('/Users/harryclark/Documents/spatial-manifolds/scripts/figures/'
      'AnchorDynamics2026/lick_raster_by_trial_type.ipynb')
_g = {}
for _c in json.load(open(NB))['cells']:
    if _c['cell_type'] != 'code':
        continue
    _s = ''.join(_c['source'])
    if _s.startswith('MOUSE, DAY') or 'INV = pd.DataFrame' in _s:
        continue
    exec(compile(_s, '<nb>', 'exec'), _g)
globals().update({k: v for k, v in _g.items() if not k.startswith('__')})

PS = '/Users/harryclark/Documents/spatial-manifolds/data/population_state'
RUN_SPEED = 3.0


def session(mo, dy):
    z = load_session_labels(mo, dy)
    if z is None:
        return None
    bp, cp = vr_paths(mo, dy)
    if not (os.path.exists(bp) and os.path.exists(cp)):
        return None
    beh = nap.load_file(bp); clusters = nap.load_file(cp)
    trials, _ = clip_trials(beh['trials'].as_dataframe(), clusters)
    trials['number'] = trials.number.astype(int)
    ids = [int(t) for t in z['trial']]
    assert set(trials.number) == set(ids), f'M{mo}D{dy}: trial id mismatch'
    T = trials.set_index('number').loc[ids]

    S = beh['S']
    st, sv = np.asarray(S.index), np.asarray(S.values, float)
    spd = []
    for s0, e0 in zip(T.start.values, T.end.values):
        m = (st >= s0) & (st <= e0) & (sv >= RUN_SPEED)
        spd.append(float(sv[m].mean()) if m.sum() > 10 else np.nan)

    return pd.DataFrame(dict(
        mouse=mo, day=dy, trial=ids, ttype=T.type.values, perf=T.performance.values,
        hit=(T.performance.values == 'hit').astype(int), speed=spd,
        frac_anch=z['frac_anch'], pc1=z['pc1'],
        n_cells=int(np.isfinite(z['pc1_load']).sum())))


TABLE = f'{PS}/trial_table.csv'


def build_all(verbose=True):
    """Build the table for every session that has cached labels (~60 s)."""
    if not os.path.isdir(f'{PS}/labels'):
        raise FileNotFoundError(
            f'No anchoring labels in {PS}/labels. Run build_anchoring_labels.py '
            'first -- it classifies every cell and derives the per-cell shuffle '
            'thresholds, which takes about 35 minutes across three workers.')
    sess = sorted({(int(f.split('M')[1].split('D')[0]),
                    int(f.split('D')[1].split('.')[0]))
                   for f in os.listdir(f'{PS}/labels') if f.endswith('.npz')})
    if not sess:
        raise FileNotFoundError(f'{PS}/labels is empty; run build_anchoring_labels.py')
    out, t0 = [], time.time()
    for mo, dy in sess:
        try:
            t = session(mo, dy)
        except Exception as e:
            print(f'  ! M{mo}D{dy}: {type(e).__name__}: {e}', flush=True); continue
        if t is not None:
            out.append(t)
    A = pd.concat(out, ignore_index=True)
    A.to_csv(TABLE, index=False)
    if verbose:
        print(f'built trial_table.csv: {len(A)} trials, '
              f'{A.groupby(["mouse","day"]).ngroups} sessions ({time.time()-t0:.0f}s)')
    return A


def ensure_trial_table(rebuild=False, verbose=True):
    """Return the trial table, building it first if it is not already there.

    Lets a figure notebook run from nothing but the cached labels: the table is
    derived, not curated, so rebuilding it is always safe. Pass rebuild=True
    after changing the population definition or the classifier, since neither
    invalidates this file automatically.
    """
    if os.path.exists(TABLE) and not rebuild:
        A = pd.read_csv(TABLE)
        if verbose:
            print(f'trial_table.csv: {len(A)} trials, '
                  f'{A.groupby(["mouse","day"]).ngroups} sessions (cached)')
        return A
    return build_all(verbose=verbose)


if __name__ == '__main__':
    A = build_all()
    print(f'  trial types: {A.ttype.value_counts().to_dict()}')
    print(f'  hit rate {A.hit.mean():.3f} | frac_anch {A.frac_anch.mean():.3f} | '
          f'speed {A.speed.mean():.1f} cm/s')
