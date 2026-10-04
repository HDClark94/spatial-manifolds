"""Two checks on the dpR2(speed | position) result.

1. BEHAVIOURAL REDUNDANCY. dpR2(speed | position) falls if the NETWORK codes
   speed less independently -- but also, trivially, if SPEED ITSELF becomes more
   predictable from position. Anchored running is more stereotyped (speed SD 6.4
   vs 8.4 cm/s), so speed may simply be more redundant with position when
   anchored, which would shrink its unique contribution with no change in the
   neurons at all. Fit speed from cyclic position per state and compare R2.

2. OVERFITTING. Every single-covariate CV pR2 came out NEGATIVE, meaning 200
   rounds at depth 5 generalise worse than the mean on one or two features. The
   deltas are then differences between two badly fit models. Refit a subset with
   a much smaller model and check the direction is stable.
"""
import numpy as np, pandas as pd, warnings, time
warnings.filterwarnings('ignore')
import xgboost as xgb
from sklearn.model_selection import KFold
from scipy.stats import wilcoxon
s = open('speed_deltapr2.py').read(); exec(s[:s.index(chr(10) + 'if __name__')])

T = pd.read_csv(f'{OUT}/theta_frequency.csv')
C = pd.read_csv(f'{OUT}/speed_tuning_cells.csv'); C = C[(C.p_pool < .05) & (C.r_pool > 0)]
sess = sorted(set(zip(T.mouse, T.day)))

print('--- 1. is SPEED more predictable from POSITION when anchored?')
rows = []
for mo, dy in sess:
    st = T[(T.mouse == mo) & (T.day == dy)]
    rng = np.random.default_rng(0)
    try:
        got = binned_full(int(mo), int(dy), dict(zip(st.trial.astype(int), st.anch.astype(bool))))
    except Exception:
        continue
    if got is None: continue
    Y, speed, pos, anch, ids = got
    ia, iN = matched_bins(speed, anch, rng)
    if min(len(ia), len(iN)) < 400: continue
    d = dict(mouse=mo, day=dy)
    for idx, lab in ((ia, 'a'), (iN, 'n')):
        ang = 2 * np.pi * pos[idx] / TL
        X, y = np.c_[np.sin(ang), np.cos(ang)], speed[idx]
        pred = np.empty(len(y))
        for tr, te in KFold(3, shuffle=True, random_state=0).split(X):
            m = xgb.train({'objective': 'reg:squarederror', 'max_depth': 5,
                           'learning_rate': 0.05, 'subsample': 0.6, 'seed': 2925},
                          xgb.DMatrix(X[tr], label=y[tr]), 200)
            pred[te] = m.predict(xgb.DMatrix(X[te]))
        d[f'r2_{lab}'] = 1 - np.sum((y - pred) ** 2) / np.sum((y - y.mean()) ** 2)
    rows.append(d)
R = pd.DataFrame(rows).dropna()
print(f'  speed predicted from position: anchored R2 {R.r2_a.median():+.4f} vs '
      f'non {R.r2_n.median():+.4f}, p = {wilcoxon(R.r2_a, R.r2_n).pvalue:.3g} '
      f'({(R.r2_a > R.r2_n).sum()}/{len(R)} MORE predictable when anchored)')
R.to_csv(f'{OUT}/speed_from_position.csv', index=False)

print('\n--- 2. smaller model (50 rounds, depth 3), 12 sessions x 10 cells')
PARAMS_S = dict(PARAMS); PARAMS_S['max_depth'] = 3
out = []
for mo, dy in sess[::5][:12]:
    st = T[(T.mouse == mo) & (T.day == dy)]
    cells = C[(C.mouse == mo) & (C.day == dy)].cluster.astype(int).tolist()
    if not cells: continue
    rng = np.random.default_rng(abs(hash((mo, dy))) % 2**32)
    got = binned_full(int(mo), int(dy), dict(zip(st.trial.astype(int), st.anch.astype(bool))))
    if got is None: continue
    Y, speed, pos, anch, ids = got
    ia, iN = matched_bins(speed, anch, rng)
    k = min(len(ia), len(iN))
    if k < 400: continue
    use = [c for c in cells if c in list(ids)][:10]
    for c in use:
        col = list(ids).index(c); d = dict(mouse=mo, day=dy, cluster=c)
        for idx, lab in ((ia, 'a'), (iN, 'n')):
            y = Y[idx, col]
            if y.sum() < 50: continue
            ang = 2 * np.pi * pos[idx] / TL
            P_, S_ = np.c_[np.sin(ang), np.cos(ang)], speed[idx][:, None]
            def pr2(X):
                pred = np.empty(len(y))
                for tr, te in KFold(3, shuffle=True, random_state=0).split(X):
                    m = xgb.train(PARAMS_S, xgb.DMatrix(X[tr], label=y[tr]), 50)
                    pred[te] = m.predict(xgb.DMatrix(X[te]))
                return float(poisson_pseudoR2(y, pred, np.mean(y)))
            d[f'pos_{lab}'], d[f'both_{lab}'] = pr2(P_), pr2(np.c_[P_, S_])
            d[f'dspd_{lab}'] = d[f'both_{lab}'] - d[f'pos_{lab}']
        out.append(d)
S = pd.DataFrame(out).dropna(subset=['dspd_a', 'dspd_n'])
g = S.groupby(['mouse', 'day'])[['dspd_a', 'dspd_n', 'pos_a', 'pos_n']].median()
print(f'  pR2(position alone)     anchored {g.pos_a.median():+.4f} vs non {g.pos_n.median():+.4f} '
      f'(negative = still overfitting)')
print(f'  dpR2(speed | position)  anchored {g.dspd_a.median():+.4f} vs non {g.dspd_n.median():+.4f}, '
      f'p = {wilcoxon(g.dspd_a, g.dspd_n).pvalue:.3g}, {len(g)} sessions, {len(S)} cells')
