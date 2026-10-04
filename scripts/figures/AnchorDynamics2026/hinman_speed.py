"""Hinman et al. 2016's prediction, tested on the anchored population state.

Usage:  python3 hinman_speed.py [worker n_workers]

THE PREDICTION. Hinman et al. (2016, Neuron 91:666) inactivated the medial
septum and found a dissociation between two speed signals in MEC:

  the THETA FREQUENCY speed signal is abolished
  the FIRING RATE speed signal is *enhanced* -- pseudo-R2 rises
      (Wilcoxon p = 1.6e-5 over all speed-modulated cells; grid p < 0.05,
      uncharacterised p < 0.005, interneurons p < 0.05)
  the SLOPE of the rate-speed relationship does NOT change

That last line is what makes the prediction sharp. A pseudo-R2 that rises while
the slope holds is not a gain change: it is the same rate-speed line with less
scatter around it, i.e. firing becomes more RELIABLY predicted by speed.

We see theta frequency decouple from running speed during the anchored
population state. If that decoupling is reduced septal drive -- the Robinson
et al. 2024 link already noted in the paper -- then Hinman's dissociation
predicts, for the anchored state specifically:

    pR2(speed) HIGHER when anchored,  with slope UNCHANGED

This is a directional prediction that can fail in three informative ways: pR2
could be unchanged (theta decoupling is not septal), lower (anchoring degrades
rate coding too, which would be the opposite of Hinman), or higher WITH a slope
change (a gain effect, not Hinman's).

WHY BOTH MEASURES ARE COMPUTED PER CELL AND PER STATE. Reporting pR2 alone
cannot distinguish "less scatter" from "steeper line", and it is the
combination that identifies the mechanism.

THE SPEED DISTRIBUTION IS THE CONFOUND. pR2 depends on the range of speeds
sampled: a state that visits a wider speed range gets a higher pR2 for free, and
the two states do differ in running speed. Every measure is therefore computed
on DECILE-MATCHED bins with equal counts per state, so the two are compared over
the same speeds with the same n. The unmatched values are kept alongside so the
size of that correction is visible rather than assumed.

Cells are called speed-modulated POOLED ACROSS STATES, against a circular-shift
null that preserves the autocorrelation of firing -- if the definition were
applied per state it would itself be state-biased, which is the error that would
manufacture this result.

Loaders come from speed_tuning.py so the binning, the running threshold and the
matching are identical to the analysis already in the paper.
"""
import os
import sys
import time
import warnings

import numpy as np
import pandas as pd

warnings.filterwarnings('ignore')
sys.path.insert(0, '/Users/harryclark/Documents/spatial-manifolds/src')
sys.path.insert(0, '/Users/harryclark/Documents/spatial-manifolds/scripts/figures/'
                   'AnchorDynamics2026')

_st = open('/Users/harryclark/Documents/spatial-manifolds/scripts/figures/'
           'AnchorDynamics2026/speed_tuning.py').read()
exec(_st[:_st.index('\ndef run_session')])        # binned, fits, matched_bins, ...

from sklearn.linear_model import PoissonRegressor
from spatial_manifolds.anchoring import load_session_labels
from spatial_manifolds.mlencoding import poisson_pseudoR2

OUT2 = '/Users/harryclark/Documents/spatial-manifolds/data/lfp'
MIN_RATE = 0.5           # Hz, pooled; below this a Poisson fit is noise
NFOLD = 5
NSHIFT_SP = 200


def cv_pr2(speed, counts, folds=NFOLD):
    """Cross-validated Poisson pseudo-R2 of counts on speed.

    Held out rather than in-sample: an in-sample pR2 carries an optimism bias
    that grows as n falls, and although the matched design equalises n between
    states, the held-out version removes the question entirely. ynull is the
    TEST fold's own mean, so a difference in mean rate between folds is not
    scored as a failure to predict.
    """
    n = len(counts)
    if n < 4 * MINBIN or np.std(speed) == 0 or counts.sum() < 20:
        return np.nan
    idx = np.arange(n)
    out = []
    for f in range(folds):
        te = idx[f::folds]                    # interleaved: the matched bins are
        tr = np.setdiff1d(idx, te)            # already a scattered subset
        if len(te) < 20 or counts[tr].sum() < 10:
            continue
        try:
            m = PoissonRegressor(alpha=1e-6, max_iter=300)
            m.fit(speed[tr].reshape(-1, 1), counts[tr])
            pred = m.predict(speed[te].reshape(-1, 1))
        except Exception:
            continue
        out.append(float(poisson_pseudoR2(counts[te], pred,
                                          np.mean(counts[te]))))
    return float(np.nanmean(out)) if out else np.nan


def speed_modulated(speed, counts, rng):
    """Pooled circular-shift test: is this cell speed-modulated at all?"""
    if np.std(speed) == 0 or counts.sum() < 20:
        return np.nan, np.nan
    obs = abs(np.corrcoef(speed, counts)[0, 1])
    null = np.array([abs(np.corrcoef(speed, np.roll(counts, int(s)))[0, 1])
                     for s in rng.integers(MINBIN, len(counts) - MINBIN, NSHIFT_SP)])
    return obs, float((null >= obs).mean())


def run_session(mo, dy, rng):
    z = load_session_labels(mo, dy)
    if z is None:
        return None
    state = {int(t): bool(f > .5) for t, f in zip(z['trial'], z['frac_anch'])}
    got = binned(mo, dy, state)
    if got is None:
        return None
    R, speed, anch, ids = got
    if anch.sum() < MINBIN or (~anch).sum() < MINBIN:
        return None
    ka, kn = matched_bins(speed, anch, rng)
    if len(ka) < MINBIN or len(kn) < MINBIN:
        return None
    ia, iN = np.where(anch)[0], np.where(~anch)[0]
    rows = []
    for i, c in enumerate(ids):
        cnt = R[:, i] * BIN                      # back to counts for Poisson
        if cnt.sum() / (len(cnt) * BIN) < MIN_RATE:
            continue
        r_obs, p_sp = speed_modulated(speed, cnt, rng)
        rec = dict(mouse=mo, day=dy, cluster_id=int(c), r_speed=r_obs,
                   p_speed=p_sp, n_match=len(ka),
                   rate=float(cnt.sum() / (len(cnt) * BIN)))
        for tag, idx_m, idx_u in (('a', ka, ia), ('n', kn, iN)):
            rec[f'pr2_{tag}'] = cv_pr2(speed[idx_m], cnt[idx_m])
            rec[f'pr2_{tag}_unmatched'] = cv_pr2(speed[idx_u], cnt[idx_u])
            b, g, rr = fits(cnt[idx_m] / BIN, speed[idx_m])
            rec[f'slope_{tag}'], rec[f'gain_{tag}'], rec[f'r_{tag}'] = b, g, rr
            rec[f'mean_rate_{tag}'] = float(cnt[idx_m].mean() / BIN)
            rec[f'mean_speed_{tag}'] = float(speed[idx_m].mean())
        rows.append(rec)
    return pd.DataFrame(rows)


if __name__ == '__main__':
    w, nw = (int(sys.argv[1]), int(sys.argv[2])) if len(sys.argv) > 2 else (0, 1)
    d = '/Users/harryclark/Documents/spatial-manifolds/data/population_state/labels'
    sess = sorted({(int(f.split('M')[1].split('D')[0]),
                    int(f.split('D')[1].split('.')[0]))
                   for f in os.listdir(d) if f.endswith('.npz')})
    mine = [s for i, s in enumerate(sess) if i % nw == w]
    path = f'{OUT2}/hinman_speed_w{w}.csv'
    out, t0 = [], time.time()
    if os.path.exists(path):
        out.append(pd.read_csv(path))
    done = set(map(tuple, out[0][['mouse', 'day']].drop_duplicates().values)) \
        if out else set()
    print(f'worker {w}/{nw}: {len(mine)} sessions, {len(done)} done', flush=True)
    for mo, dy in mine:
        if (mo, dy) in done:
            continue
        try:
            t = run_session(mo, dy, np.random.default_rng(abs(hash((mo, dy))) % 2**32))
        except Exception as e:
            print(f'  ! M{mo}D{dy}: {type(e).__name__}: {e}', flush=True); continue
        if t is None or not len(t):
            continue
        out.append(t)
        pd.concat(out, ignore_index=True).to_csv(path, index=False)
        print(f'  M{mo}D{dy}: {len(t)} cells, {time.time()-t0:.0f}s', flush=True)
    print(f'worker {w} done in {time.time()-t0:.0f}s', flush=True)
