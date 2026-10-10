"""Does theta track UNEXPECTED speed rather than speed?

The speed->theta slope halves when the population is anchored, while the cells'
speed coding is untouched. One premise reproduces that whole dissociation
without any change in gain: if theta is driven by the UNPREDICTED component of
self-motion, then when running becomes stereotyped -- and it does, speed is 4.5x
more predictable from position when anchored -- a regression of theta on TOTAL
speed must flatten in proportion, while cells that track total speed show
nothing.

The test, per state, is the spiking analysis pointed at the LFP. Cyclic position
is partialled out of both theta amplitude and speed, and the two residuals are
regressed on one another:

    slope_total      theta on speed                 expected to FALL when anchored
    slope_unexpected theta residual on speed residual
    slope_expected   theta on position-predicted speed

PREDICTION. If theta tracks unexpected speed, slope_unexpected is the SAME in
both states and the fall in slope_total is carried entirely by there being less
unexpected speed. If slope_unexpected halves too, the hypothesis is dead and the
entrainment gain really does change.

Theta amplitude is the Hilbert envelope of the 6-10 Hz signal, computed per MEC
channel group and z-scored within group before averaging, because amplitude
varies by an order of magnitude across layers. The LFP is decimated 1000 -> 250
Hz first: theta is 6-10 Hz, so 250 Hz is far above Nyquist, and the envelope is
only needed at 250 ms resolution.
"""
import os, sys, time, warnings
import h5py, numpy as np, pandas as pd
warnings.filterwarnings('ignore')
from scipy.signal import butter, filtfilt, hilbert, decimate
from sklearn.model_selection import KFold
from scipy.stats import wilcoxon
import xgboost as xgb
import pynapple as nap
sys.path.insert(0, '/Users/harryclark/Documents/spatial-manifolds/src')
sys.path.insert(0, '/Users/harryclark/Documents/spatial-manifolds/scripts/figures/'
                   'AnchorDynamics2026')
from lfp_batch import LFP_ROOT, FS, mec_groups                      # noqa

# the helper lives beside this file; it used to be exec'd out of a session
# scratchpad, which stops existing when that session ends
import os as _os
_HERE = _os.path.dirname(_os.path.abspath(__file__))
s = open(_os.path.join(_HERE, 'speed_deltapr2.py')).read()
exec(s[:s.index(chr(10) + 'if __name__')])

DEC = 4                      # 1000 -> 250 Hz
FS_D = FS // DEC
BP = butter(3, [6 / (FS_D / 2), 10 / (FS_D / 2)], btype='band')
SMALL = {'max_depth': 3, 'learning_rate': 0.05, 'subsample': 0.6, 'seed': 2925}
NR, FOLD = 50, 3


def cv_predict(X, y):
    p = dict(SMALL); p['objective'] = 'reg:squarederror'
    pred = np.empty(len(y), float)
    for tr, te in KFold(FOLD, shuffle=True, random_state=0).split(X):
        m = xgb.train(p, xgb.DMatrix(X[tr], label=y[tr]), NR)
        pred[te] = m.predict(xgb.DMatrix(X[te]))
    return pred


def theta_per_bin(mo, dy, edges):
    """Mean z-scored theta envelope in each bin, averaged over MEC groups."""
    lf = f'{LFP_ROOT}/M{mo}/D{dy}/VR/sub-M{mo}_ses-D{dy}_typ-VR_beh.nwb'
    if not os.path.exists(lf):
        return None
    f = h5py.File(lf, 'r')
    if 'processing/ecephys/LFP/lfp/data' not in f:
        f.close(); return None
    names = [n.decode() for n in f['general/extracellular_ephys/electrodes/channel_name'][:]]
    M = mec_groups(mo, dy, names)
    if len(M) < 4:
        f.close(); return None
    cols = np.sort(M.g.values)                     # h5py needs increasing order
    # one read of all MEC columns, not one per group: the volume is remote and
    # column-by-column reads dominated the runtime (113 s -> see below)
    L = f['processing/ecephys/LFP/lfp/data'][:, cols].astype(np.float32)
    f.close()
    acc = np.zeros(len(edges) - 1)
    n_ok = 0
    for gi in range(L.shape[1]):
        x = decimate(L[:, gi], DEC, ftype='fir', zero_phase=True)
        env = np.abs(hilbert(filtfilt(*BP, x)))
        if not np.isfinite(env).all() or env.std() == 0:
            continue
        env = (env - env.mean()) / env.std()       # within group, across the session
        t = np.arange(len(env)) / FS_D
        j = np.searchsorted(edges, t) - 1
        ok = (j >= 0) & (j < len(edges) - 1)
        cnt = np.bincount(j[ok], minlength=len(edges) - 1)
        sm = np.bincount(j[ok], weights=env[ok], minlength=len(edges) - 1)
        acc += np.where(cnt > 0, sm / np.maximum(cnt, 1), np.nan)
        n_ok += 1
    return acc / n_ok if n_ok else None


def run_session(mo, dy, state, rng):
    bp, cp = vr_paths(mo, dy)
    beh = nap.load_file(bp); clusters = nap.load_file(cp)
    trials, orig = clip_trials(beh['trials'].as_dataframe(), clusters)
    S, P = beh['S'], beh['P']
    ts, vs = np.asarray(S.index), np.asarray(S.values, dtype=float)
    ps, pt = np.asarray(P.values, dtype=float), np.asarray(P.index)
    edges = np.arange(ts[0], ts[-1], BIN)
    if len(edges) < 200:
        return None
    ctr = edges[:-1] + BIN / 2

    th = theta_per_bin(mo, dy, edges)
    if th is None:
        return None

    j = np.searchsorted(edges, ts) - 1
    ok = (j >= 0) & (j < len(ctr)) & np.isfinite(vs)
    cnt = np.bincount(j[ok], minlength=len(ctr))
    spd = np.bincount(j[ok], weights=vs[ok], minlength=len(ctr))
    spd = np.where(cnt > 0, spd / np.maximum(cnt, 1), np.nan)
    ang = 2 * np.pi * ps / TL
    jp = np.searchsorted(edges, pt) - 1
    okp = (jp >= 0) & (jp < len(ctr)) & np.isfinite(ang)
    cs = np.bincount(jp[okp], weights=np.cos(ang[okp]), minlength=len(ctr))
    sn = np.bincount(jp[okp], weights=np.sin(ang[okp]), minlength=len(ctr))
    pos = np.mod(np.arctan2(sn, cs), 2 * np.pi) * TL / (2 * np.pi)

    starts, ends = trials.start.values.astype(float), trials.end.values.astype(float)
    nums = trials.number.values.astype(int)
    k = np.searchsorted(starts, ctr) - 1
    inside = (k >= 0) & (k < len(starts))
    k = k.clip(0, len(starts) - 1)
    inside &= ctr <= ends[k]
    lab = np.array([state.get(int(u), np.nan) if ins else np.nan
                    for u, ins in zip(nums[k], inside)], dtype=float)

    keep = (np.isfinite(spd) & (spd >= RUN) & np.isfinite(lab) & np.isfinite(pos)
            & np.isfinite(th) & (np.bincount(jp[okp], minlength=len(ctr)) > 0))
    if keep.sum() < 2 * MINBIN:
        return None
    spd, pos, th = spd[keep], pos[keep], th[keep]
    anch = lab[keep] > .5
    ia, iN = matched_bins(spd, anch, rng)
    if min(len(ia), len(iN)) < 400:
        return None

    d = dict(mouse=mo, day=dy, n_bins=int(min(len(ia), len(iN))))
    for idx, tag in ((ia, 'a'), (iN, 'n')):
        sp, th_, po = spd[idx], th[idx], pos[idx]
        a = 2 * np.pi * po / TL
        X = np.c_[np.sin(a), np.cos(a)]
        sp_hat = cv_predict(X, sp)                 # expected speed
        th_hat = cv_predict(X, th_)                # position-locked theta
        sres, tres = sp - sp_hat, th_ - th_hat
        d[f'slope_total_{tag}'] = np.polyfit(sp, th_, 1)[0]
        d[f'slope_unexp_{tag}'] = np.polyfit(sres, tres, 1)[0]
        d[f'slope_exp_{tag}'] = np.polyfit(sp_hat, th_, 1)[0]
        d[f'sres_sd_{tag}'] = float(sres.std())
        d[f'r_unexp_{tag}'] = float(np.corrcoef(sres, tres)[0, 1])
    return d


if __name__ == '__main__':
    T = pd.read_csv(f'{OUT}/theta_frequency.csv')
    sess = sorted(set(zip(T.mouse, T.day)))
    rows, t0 = [], time.time()
    for i, (mo, dy) in enumerate(sess, 1):
        st = T[(T.mouse == mo) & (T.day == dy)]
        rng = np.random.default_rng(abs(hash((mo, dy))) % 2**32)
        try:
            d = run_session(int(mo), int(dy),
                            dict(zip(st.trial.astype(int), st.anch.astype(bool))), rng)
        except Exception as e:
            print(f'  ! M{mo}D{dy}: {type(e).__name__}: {e}', flush=True); continue
        if d is None:
            continue
        rows.append(d)
        print(f'[{i}/{len(sess)}] M{mo}D{dy}: total {d["slope_total_a"]:+.4f}/'
              f'{d["slope_total_n"]:+.4f}  unexp {d["slope_unexp_a"]:+.4f}/'
              f'{d["slope_unexp_n"]:+.4f}  {time.time()-t0:.0f}s', flush=True)
    R = pd.DataFrame(rows)
    R.to_csv(f'{OUT}/theta_residual_speed.csv', index=False)
    print(f'\n{len(R)} sessions')
    for m, lab in (('slope_total', 'theta on TOTAL speed'),
                   ('slope_unexp', 'theta on UNEXPECTED speed'),
                   ('slope_exp', 'theta on EXPECTED speed'),
                   ('sres_sd', 'SD of unexpected speed (cm/s)'),
                   ('r_unexp', 'r (theta, unexpected speed)')):
        a, n = R[f'{m}_a'].dropna(), R[f'{m}_n'].dropna()
        idx = a.index.intersection(n.index)
        a, n = a[idx], n[idx]
        print(f'  {lab:32s} anchored {a.median():+.5f}  non {n.median():+.5f}  '
              f'({100*(a.median()-n.median())/abs(n.median()):+6.1f}%)  '
              f'p = {wilcoxon(a, n).pvalue:.3g}  ({(a < n).sum()}/{len(a)} lower)')
