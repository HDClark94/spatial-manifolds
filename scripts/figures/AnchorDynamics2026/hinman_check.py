"""Is the anchored pR2(speed) advantage a firing-RATE effect?

Usage:  python3 hinman_check.py [n_sessions]

THE PROBLEM. A Poisson pseudo-R2 is not rate-invariant. The same underlying
rate-speed relationship yields a larger pR2 when the cell fires more, because
the Poisson variance that the null model must absorb grows only as the mean
while the explained signal grows as the mean squared. So if MEC cells fire at
different rates in the anchored and non-anchored states -- and the population
rate is exactly the kind of thing an anchoring state might change -- then a
higher pR2 when anchored could be nothing but a higher rate.

Decile-matching the SPEED, which hinman_speed.py does, is no defence against
this: it equalises the predictor, not the response.

THE CONTROL. Poisson thinning. A Poisson process of rate L, each of whose
events is kept independently with probability p, is a Poisson process of rate
pL: thinning a spike train produces exactly the train a lower-firing cell would
have produced with the same speed dependence. So the state with the higher mean
count is thinned, bin by bin, to the other state's mean, and both pR2 values are
recomputed on the thinned data.

  counts_thinned[b] ~ Binomial(counts[b], p),  p = target_mean / observed_mean

If the pR2 advantage survives thinning, it is about how reliably speed predicts
firing and Hinman's dissociation holds. If it vanishes, the anchored state
simply fires faster and the result is an artefact of the measure.

Thinning is repeated N_THIN times and averaged, since a single draw adds noise
of its own.
"""
import os
import sys
import time
import warnings

import numpy as np
import pandas as pd

warnings.filterwarnings('ignore')
sys.path.insert(0, '/Users/harryclark/Documents/spatial-manifolds/scripts/figures/'
                   'AnchorDynamics2026')
import hinman_speed as HS
from scipy.stats import wilcoxon
from spatial_manifolds.anchoring import load_session_labels

N_THIN = 5
OUT = '/Users/harryclark/Documents/spatial-manifolds/data/lfp/hinman_thinned.csv'


def thinned_pr2(speed_a, cnt_a, speed_n, cnt_n, rng):
    """pR2 for both states after equalising their mean counts by thinning."""
    ma, mn = cnt_a.mean(), cnt_n.mean()
    if ma <= 0 or mn <= 0:
        return np.nan, np.nan
    pa = min(1.0, mn / ma)          # thin whichever state fires faster
    pn = min(1.0, ma / mn)
    va, vn = [], []
    for _ in range(N_THIN):
        ta = rng.binomial(cnt_a.astype(int), pa).astype(float)
        tn = rng.binomial(cnt_n.astype(int), pn).astype(float)
        va.append(HS.cv_pr2(speed_a, ta))
        vn.append(HS.cv_pr2(speed_n, tn))
    return float(np.nanmean(va)), float(np.nanmean(vn))


def run_session(mo, dy, rng):
    z = load_session_labels(mo, dy)
    if z is None:
        return None
    state = {int(t): bool(f > .5) for t, f in zip(z['trial'], z['frac_anch'])}
    got = HS.binned(mo, dy, state)
    if got is None:
        return None
    R, speed, anch, ids = got
    ka, kn = HS.matched_bins(speed, anch, rng)
    if len(ka) < HS.MINBIN or len(kn) < HS.MINBIN:
        return None
    rows = []
    for i, c in enumerate(ids):
        cnt = R[:, i] * HS.BIN
        if cnt.sum() / (len(cnt) * HS.BIN) < HS.MIN_RATE:
            continue
        _, p_sp = HS.speed_modulated(speed, cnt, rng)
        ca, cn = cnt[ka], cnt[kn]
        raw_a, raw_n = HS.cv_pr2(speed[ka], ca), HS.cv_pr2(speed[kn], cn)
        th_a, th_n = thinned_pr2(speed[ka], ca, speed[kn], cn, rng)
        rows.append(dict(mouse=mo, day=dy, cluster_id=int(c), p_speed=p_sp,
                         pr2_a=raw_a, pr2_n=raw_n,
                         pr2_a_thin=th_a, pr2_n_thin=th_n,
                         rate_a=float(ca.mean() / HS.BIN),
                         rate_n=float(cn.mean() / HS.BIN),
                         sd_speed_a=float(speed[ka].std()),
                         sd_speed_n=float(speed[kn].std())))
    return pd.DataFrame(rows)


if __name__ == '__main__':
    n_max = int(sys.argv[1]) if len(sys.argv) > 1 else 20
    d = ('/Users/harryclark/Documents/spatial-manifolds/data/population_state/labels')
    sess = sorted({(int(f.split('M')[1].split('D')[0]),
                    int(f.split('D')[1].split('.')[0]))
                   for f in os.listdir(d) if f.endswith('.npz')})
    rng0 = np.random.default_rng(0)
    sess = [sess[i] for i in rng0.choice(len(sess), min(n_max, len(sess)),
                                         replace=False)]
    out, t0 = [], time.time()
    for mo, dy in sess:
        try:
            t = run_session(mo, dy, np.random.default_rng(abs(hash((mo, dy))) % 2**32))
        except Exception as e:
            print(f'  ! M{mo}D{dy}: {type(e).__name__}: {e}', flush=True); continue
        if t is not None and len(t):
            out.append(t)
            print(f'  M{mo}D{dy}: {len(t)} cells, {time.time()-t0:.0f}s', flush=True)
    A = pd.concat(out, ignore_index=True)
    A.to_csv(OUT, index=False)
    s = A[(A.p_speed < .05)].dropna(subset=['pr2_a', 'pr2_n',
                                            'pr2_a_thin', 'pr2_n_thin'])
    print(f'\n{len(s)} speed-modulated cells in '
          f'{s.groupby(["mouse","day"]).ngroups} sessions')
    print(f'  firing rate   anchored {s.rate_a.mean():.3f} Hz  '
          f'non {s.rate_n.mean():.3f} Hz  '
          f'p = {wilcoxon(s.rate_a, s.rate_n).pvalue:.2g}')
    print(f'  speed SD      anchored {s.sd_speed_a.mean():.2f}  '
          f'non {s.sd_speed_n.mean():.2f}  '
          f'p = {wilcoxon(s.sd_speed_a, s.sd_speed_n).pvalue:.2g}')
    print(f'  pR2 RAW       anchored {s.pr2_a.mean():+.4f}  '
          f'non {s.pr2_n.mean():+.4f}  '
          f'p = {wilcoxon(s.pr2_a, s.pr2_n).pvalue:.2g}')
    print(f'  pR2 THINNED   anchored {s.pr2_a_thin.mean():+.4f}  '
          f'non {s.pr2_n_thin.mean():+.4f}  '
          f'p = {wilcoxon(s.pr2_a_thin, s.pr2_n_thin).pvalue:.2g}')
    d_raw = (s.pr2_a - s.pr2_n).mean()
    d_thin = (s.pr2_a_thin - s.pr2_n_thin).mean()
    print(f'\n  effect retained after thinning: '
          f'{100 * d_thin / d_raw:.0f}%  ({d_raw:+.4f} -> {d_thin:+.4f})')
