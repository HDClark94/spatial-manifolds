"""Does MEC speed tuning flatten when the population is anchored?

The LFP says the speed-coupled components of the signal (theta amplitude, and
80-200 Hz power, which tracks population spiking) lose their dependence on
running speed during anchored epochs, while the speed-independent component is
preserved. The velocity-gain account predicts that this is visible in the
spikes: individual speed-tuned cells should show flatter tuning, and speed
should be less decodable from the population, when anchored.

If instead speed tuning is unchanged while only the LFP decouples, the velocity
input account is in trouble and the effect is about the field potential rather
than about what the network encodes.

DESIGN

Firing rate and running speed in 250 ms bins, running bins only (>= 3 cm/s),
each bin assigned the anchoring state of its trial (from the same per-trial
labels used for the LFP analysis).

Per cell and per state:
  slope     OLS rate on speed, Hz per cm/s -- the primary measure, because it is
            insensitive to the narrower speed range of anchored epochs
  gain      slope / mean rate, %/cm-s, which separates a change in speed
            SENSITIVITY from a change in overall firing rate: a cell that simply
            fires less keeps its gain, a cell that stops tracking speed loses it
  r         Pearson correlation, reported alongside but not relied on

Every measure is computed twice: on all running bins, and after trimming both
states to their common 5-95% speed support, so the comparison is made over the
same speeds.

Speed cells are defined pooled across states (so the definition cannot itself be
state-biased) by a circular-shift null on the binned rate, which preserves the
autocorrelation of firing that makes a parametric p meaningless here.

Population decoding asks the same question without per-cell selection: ridge
regression from population counts to speed, 5-fold CV, with the two states
matched for bin count and speed distribution before fitting.
"""
import os, sys, json, time, warnings
import numpy as np, pandas as pd
warnings.filterwarnings('ignore')
import matplotlib; matplotlib.use('Agg')
import pynapple as nap
from sklearn.linear_model import Ridge
from sklearn.model_selection import KFold
sys.path.insert(0, '/Users/harryclark/Documents/spatial-manifolds/src')

_g = {}
for _c in json.load(open('/Users/harryclark/Documents/spatial-manifolds/scripts/'
                         'figures/AnchorDynamics2026/lick_raster_by_trial_type.ipynb'))['cells']:
    if _c['cell_type'] != 'code':
        continue
    _s = ''.join(_c['source'])
    if _s.startswith('MOUSE, DAY') or 'INV = pd.DataFrame' in _s:
        continue
    exec(compile(_s, '<nb>', 'exec'), _g)
globals().update({k: v for k, v in _g.items() if not k.startswith('__')})

OUT = '/Users/harryclark/Documents/spatial-manifolds/data/lfp'
BIN = 0.25           # s
RUN = 3.0            # cm/s, the stop threshold used throughout
MINBIN = 100         # bins per state before a cell contributes
NSHIFT = 200


def binned(mo, dy, state):
    """Per-bin rate matrix, speed and state. None if the session is unusable."""
    bp, cp = vr_paths(mo, dy)
    beh = nap.load_file(bp); clusters = nap.load_file(cp)
    trials, orig = clip_trials(beh['trials'].as_dataframe(), clusters)
    ids = spatial_cell_ids(mo, dy, clusters, classes=('GC', 'NGS', 'NS'))
    if len(ids) < 5:
        return None

    S = beh['S']
    ts, vs = np.asarray(S.index), np.asarray(S.values, dtype=float)
    edges = np.arange(ts[0], ts[-1], BIN)
    if len(edges) < 200:
        return None
    ctr = edges[:-1] + BIN / 2

    j = np.searchsorted(edges, ts) - 1
    ok = (j >= 0) & (j < len(ctr)) & np.isfinite(vs)
    cnt = np.bincount(j[ok], minlength=len(ctr))
    spd = np.bincount(j[ok], weights=vs[ok], minlength=len(ctr))
    spd = np.where(cnt > 0, spd / np.maximum(cnt, 1), np.nan)

    starts = trials.start.values.astype(float)
    ends = trials.end.values.astype(float)
    nums = trials.number.values.astype(int)
    k = np.searchsorted(starts, ctr) - 1
    inside = (k >= 0) & (k < len(starts))
    k = k.clip(0, len(starts) - 1)
    inside &= ctr <= ends[k]
    tno = np.where(inside, nums[k], -1)

    lab = np.array([state.get(int(u), np.nan) for u in tno], dtype=float)
    keep = np.isfinite(spd) & (spd >= RUN) & np.isfinite(lab)
    if keep.sum() < 2 * MINBIN:
        return None

    R = np.empty((keep.sum(), len(ids)))
    for i, c in enumerate(ids):
        sp = np.asarray(clusters[c].index, dtype=float)
        R[:, i] = np.histogram(sp, bins=edges)[0][keep] / BIN
    return R, spd[keep], lab[keep] > .5, ids


def fits(rate, speed):
    """slope (Hz per cm/s), gain (slope / mean rate) and r."""
    if len(speed) < MINBIN or np.std(speed) == 0 or np.std(rate) == 0:
        return np.nan, np.nan, np.nan
    b = np.polyfit(speed, rate, 1)[0]
    m = rate.mean()
    return b, (b / m if m > 0 else np.nan), np.corrcoef(speed, rate)[0, 1]


def common_support(speed, anch):
    a, n = speed[anch], speed[~anch]
    if len(a) < MINBIN or len(n) < MINBIN:
        return np.zeros(len(speed), bool)
    lo = max(np.quantile(a, .05), np.quantile(n, .05))
    hi = min(np.quantile(a, .95), np.quantile(n, .95))
    return (speed >= lo) & (speed <= hi)


def decode(R, speed, idx, rng, folds=5):
    """Ridge speed decoding within one state. Returns r(pred, true) and MAE."""
    X, y = R[idx], speed[idx]
    if len(y) < 200:
        return np.nan, np.nan
    pred = np.empty(len(y))
    for tr, te in KFold(folds, shuffle=True, random_state=0).split(X):
        mu, sd = X[tr].mean(0), X[tr].std(0) + 1e-9
        m = Ridge(alpha=1.0).fit((X[tr] - mu) / sd, y[tr])
        pred[te] = m.predict((X[te] - mu) / sd)
    return np.corrcoef(pred, y)[0, 1], float(np.mean(np.abs(pred - y)))


def matched_bins(speed, anch, rng, nq=10):
    """Equal bin counts per state within speed deciles."""
    q = np.quantile(speed, np.linspace(0, 1, nq + 1))
    q[-1] += 1e-6
    ka, kn = [], []
    for i in range(nq):
        m = (speed >= q[i]) & (speed < q[i + 1])
        ia, iN = np.where(m & anch)[0], np.where(m & ~anch)[0]
        k = min(len(ia), len(iN))
        if k == 0:
            continue
        ka.append(rng.choice(ia, k, replace=False))
        kn.append(rng.choice(iN, k, replace=False))
    if not ka:
        return np.array([], int), np.array([], int)
    return np.concatenate(ka), np.concatenate(kn)


def run_session(mo, dy, state, rng):
    got = binned(mo, dy, state)
    if got is None:
        return None, None
    R, speed, anch, ids = got
    if anch.sum() < MINBIN or (~anch).sum() < MINBIN:
        return None, None
    cs = common_support(speed, anch)

    # pooled speed cells, by circular shift of the rate series
    rows = []
    for i, c in enumerate(ids):
        rate = R[:, i]
        if rate.mean() <= 0:
            continue
        r_pool = np.corrcoef(speed, rate)[0, 1]
        null = np.array([np.corrcoef(speed, np.roll(rate, s))[0, 1]
                         for s in rng.integers(MINBIN, len(rate) - MINBIN, NSHIFT)])
        p_pool = (np.sum(np.abs(null) >= abs(r_pool)) + 1) / (NSHIFT + 1)
        d = dict(mouse=mo, day=dy, cluster=int(c), rate=float(rate.mean()),
                 r_pool=float(r_pool), p_pool=float(p_pool),
                 n_a=int(anch.sum()), n_n=int((~anch).sum()))
        for tag, m in (('', np.ones(len(speed), bool)), ('cs_', cs)):
            for st, lab in ((anch & m, 'a'), (~anch & m, 'n')):
                b, g, r = fits(rate[st], speed[st])
                d[f'{tag}slope_{lab}'] = b
                d[f'{tag}gain_{lab}'] = g
                d[f'{tag}r_{lab}'] = r
                d[f'{tag}rate_{lab}'] = float(rate[st].mean()) if st.sum() else np.nan
        rows.append(d)

    ia, iN = matched_bins(speed, anch, rng)
    k = min(len(ia), len(iN))
    dec = dict(mouse=mo, day=dy, n_cells=len(ids), n_matched=int(k))
    if k >= 200:
        ra, ma = decode(R, speed, ia, rng)
        rn, mn = decode(R, speed, iN, rng)
        dec.update(dec_r_a=ra, dec_r_n=rn, dec_mae_a=ma, dec_mae_n=mn,
                   speed_sd_a=float(speed[ia].std()), speed_sd_n=float(speed[iN].std()))
    return pd.DataFrame(rows), pd.DataFrame([dec])


if __name__ == '__main__':
    T = pd.read_csv(f'{OUT}/theta_frequency.csv')
    sess = sorted(set(zip(T.mouse, T.day)))
    cells, decs, t0 = [], [], time.time()
    for k, (mo, dy) in enumerate(sess, 1):
        st = T[(T.mouse == mo) & (T.day == dy)]
        state = dict(zip(st.trial.astype(int), st.anch.astype(bool)))
        rng = np.random.default_rng(abs(hash((mo, dy))) % 2**32)
        try:
            C, D = run_session(int(mo), int(dy), state, rng)
        except Exception as e:
            print(f'  ! M{mo}D{dy}: {type(e).__name__}: {e}', flush=True)
            continue
        if C is None or not len(C):
            print(f'  - M{mo}D{dy}: skipped', flush=True)
            continue
        cells.append(C); decs.append(D)
        print(f'[{k}/{len(sess)}] M{mo}D{dy}: {len(C)} cells, '
              f'{time.time()-t0:.0f}s', flush=True)
    A = pd.concat(cells, ignore_index=True)
    B = pd.concat(decs, ignore_index=True)
    A.to_csv(f'{OUT}/speed_tuning_cells.csv', index=False)
    B.to_csv(f'{OUT}/speed_tuning_decode.csv', index=False)
    print(f'\n{len(A)} cells from {A.groupby(["mouse","day"]).ngroups} sessions')
