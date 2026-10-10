"""Does MEC track UNEXPECTED speed differently when anchored?

dpR2(speed | position) fell when anchored, but it cannot be read as reduced
speed coding: anchored running is stereotyped, so speed is 4.5x more
predictable from position (R2 0.22 vs 0.05), and a covariate that carries less
independent information must contribute less. The redundancy is behavioural.

This asks the question redundancy cannot answer. Within each state, position is
partialled out of BOTH sides:

    speed residual  = speed - E[speed | position]
    rate residual   = rate  - E[rate  | position]

and the two residuals are related to each other. The speed residual is
orthogonal to position by construction, so a difference between states cannot
come from position standing in for speed. It is the quantity a velocity
integration account cares about: whether the network registers departures from
the expected speed profile.

E[. | position] is fit within state, cross-validated, from cyclic position
(sin, cos), with the SMALL model -- 50 rounds at depth 3 -- because the 200
round depth 5 model drove every single-covariate pR2 negative.

TWO SCALES, because they answer different questions:

  slope per cm/s      physical gain: Hz per cm/s of unexpected speed
  slope per z         gain relative to how much unexpected speed there IS in
                      that state

The second matters because anchored epochs have less residual speed to begin
with. If the network registered unexpected speed identically, the per-cm/s
slope would match across states while the per-z slope would be smaller when
anchored purely because a z unit is smaller. Reporting one without the other
would manufacture a result either way.
"""
import sys, time, warnings
import numpy as np, pandas as pd
warnings.filterwarnings('ignore')
import xgboost as xgb
from sklearn.model_selection import KFold
from scipy.stats import wilcoxon

# the helper lives beside this file; it used to be exec'd out of a session
# scratchpad, which stops existing when that session ends
import os as _os
_HERE = _os.path.dirname(_os.path.abspath(__file__))
s = open(_os.path.join(_HERE, 'speed_deltapr2.py')).read()
exec(s[:s.index(chr(10) + 'if __name__')])

SMALL = {'max_depth': 3, 'learning_rate': 0.05, 'subsample': 0.6, 'seed': 2925}
NR, FOLD, CAP_C = 50, 3, 20


def cv_predict(X, y, poisson):
    p = dict(SMALL)
    p['objective'] = 'count:poisson' if poisson else 'reg:squarederror'
    pred = np.empty(len(y), float)
    for tr, te in KFold(FOLD, shuffle=True, random_state=0).split(X):
        m = xgb.train(p, xgb.DMatrix(X[tr], label=y[tr]), NR)
        pred[te] = m.predict(xgb.DMatrix(X[te]))
    return pred


def run_session(mo, dy, state, cells, rng):
    got = binned_full(mo, dy, state)
    if got is None:
        return None
    Y, speed, pos, anch, ids = got
    ia, iN = matched_bins(speed, anch, rng)
    k = min(len(ia), len(iN))
    if k < 400:
        return None
    use = [c for c in cells if c in list(ids)]
    if len(use) > CAP_C:
        use = list(rng.choice(use, CAP_C, replace=False))
    if not use:
        return None

    rows = []
    for idx, lab in ((ia, 'a'), (iN, 'n')):
        ang = 2 * np.pi * pos[idx] / TL
        P_ = np.c_[np.sin(ang), np.cos(ang)]
        sp = speed[idx]
        sres = sp - cv_predict(P_, sp, poisson=False)       # unexpected speed
        sd = sres.std()
        for c in use:
            y = Y[idx, list(ids).index(c)].astype(float)
            if y.sum() < 50:
                continue
            yres = y - cv_predict(P_, y, poisson=True)
            b = np.polyfit(sres, yres, 1)[0] / BIN          # Hz per cm/s
            rows.append(dict(mouse=mo, day=dy, cluster=int(c), state=lab,
                             slope_cms=b, slope_z=b * sd,
                             r_partial=np.corrcoef(sres, yres)[0, 1],
                             rate=float(y.mean() / BIN),
                             gain_cms=b / (y.mean() / BIN) if y.mean() > 0 else np.nan,
                             sres_sd=float(sd), n_bins=k))
    return pd.DataFrame(rows)


if __name__ == '__main__':
    T = pd.read_csv(f'{OUT}/theta_frequency.csv')
    C = pd.read_csv(f'{OUT}/speed_tuning_cells.csv')
    C = C[(C.p_pool < .05) & (C.r_pool > 0)]
    sess = sorted(set(zip(T.mouse, T.day)))
    out, t0 = [], time.time()
    for i, (mo, dy) in enumerate(sess, 1):
        st = T[(T.mouse == mo) & (T.day == dy)]
        cells = C[(C.mouse == mo) & (C.day == dy)].cluster.astype(int).tolist()
        if not cells:
            continue
        rng = np.random.default_rng(abs(hash((mo, dy))) % 2**32)
        try:
            R = run_session(int(mo), int(dy),
                            dict(zip(st.trial.astype(int), st.anch.astype(bool))),
                            cells, rng)
        except Exception as e:
            print(f'  ! M{mo}D{dy}: {type(e).__name__}: {e}', flush=True)
            continue
        if R is None or not len(R):
            continue
        out.append(R)
        print(f'[{i}/{len(sess)}] M{mo}D{dy}: {R.cluster.nunique()} cells, '
              f'{time.time()-t0:.0f}s', flush=True)
    A = pd.concat(out, ignore_index=True)
    A.to_csv(f'{OUT}/residual_speed.csv', index=False)

    W = A.pivot_table(index=['mouse', 'day', 'cluster'], columns='state',
                      values=['slope_cms', 'slope_z', 'r_partial', 'gain_cms',
                              'sres_sd', 'rate']).dropna()
    print(f'\n{len(W)} cells, {A.groupby(["mouse","day"]).ngroups} sessions')
    print(f'  unexpected-speed SD: anchored {W[("sres_sd","a")].median():.2f} vs '
          f'non {W[("sres_sd","n")].median():.2f} cm/s')
    for m, lab in (('slope_cms', 'slope (Hz per cm/s unexpected)'),
                   ('slope_z', 'slope (Hz per SD unexpected)'),
                   ('gain_cms', 'gain (slope / rate, per cm/s)'),
                   ('r_partial', 'partial r (rate, speed | position)')):
        g = W[m].groupby(level=['mouse', 'day']).median().dropna()
        print(f'  {lab:36s} anchored {g["a"].median():+.5f}  non {g["n"].median():+.5f}  '
              f'p = {wilcoxon(g["a"], g["n"]).pvalue:.3g}  '
              f'({(g["a"] < g["n"]).sum()}/{len(g)} lower when anchored)')
