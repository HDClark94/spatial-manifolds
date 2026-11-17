"""Does MEC speed coding change between anchoring states? One model per cell.

Replaces the per-state OLS slopes of speed_tuning.py as the test behind Figure
S17 and the Results (2026-11-01). Those slopes had three weaknesses: speed on
the virtual track is stereotyped along it, so a slope on speed partly measures
the cell's position tuning; slope and gain were taken separately per state and
then compared, so the state's effect on overall rate and on speed sensitivity
were not separated in one fit; and panels rested on different session sets.

MODEL, per cell, on running bins (>= RUN cm/s) inside the two states' common
5-95% speed support:

    log E[count] = state-specific position map + b_speed * speed
                   + b_int * speed * anchored

The position map is a cyclic Fourier basis (N_HARM harmonics) fitted SEPARATELY
in each state, with its own intercept, because non-anchored trials re-draw the
map: a shared map would leave spatial firing for speed to explain. Speed is in
units of 10 cm/s, centred on the common support. In a log-link model the speed
coefficient is a GAIN -- the proportional change in rate per 10 cm/s -- so a
cell that only fires less in one state changes its state intercept, not b_speed.
b_int is the change in that gain when anchored, with position held. The same
model without the position terms is also fitted, to show what position does.

With the position maps in the model, b_speed is the gain on UNEXPECTED speed:
the model is unchanged if speed is replaced by its residual on the position
basis (fitted within each state), so the coefficient is estimated from
departures from the expected speed profile, which neither the position maps nor
the state's stereotyped running can stand in for (fitted both ways, 2026-11-01:
identical). Speed and position are collinear on the track, more so when
anchored, which widens this estimate's interval but does not bias it.

POPULATION DECODING of UNEXPECTED speed: within each state, speed and every
cell's rate have their position-predicted part removed (least squares on the
same basis, cross-validated), and ridge regression decodes the speed residual
from the rate residuals, on bins matched between states for count and speed
decile. Position cannot help either state here.

Writes data/lfp/speed_state_glm_cells.csv, speed_state_glm_sessions.csv
Usage:  python speed_state_glm.py
"""
import os, sys, time, warnings
import numpy as np, pandas as pd
warnings.filterwarnings('ignore')
import statsmodels.api as sm
from sklearn.linear_model import Ridge
from sklearn.model_selection import KFold

_HERE = os.path.dirname(os.path.abspath(__file__))
_s = open(os.path.join(_HERE, 'speed_deltapr2.py')).read()
exec(_s[:_s.index(chr(10) + 'if __name__')])          # binned_full and its helpers

N_HARM = int(os.environ.get('N_HARM', 4))   # 8 for the robustness check
MIN_SPK = 100          # spikes per state before a cell is fitted


def fourier(pos):
    a = 2 * np.pi * pos / TL
    return np.column_stack([f(k * a) for k in range(1, N_HARM + 1) for f in (np.sin, np.cos)])


def design(pos, spd, anch, with_pos=True):
    a = anch.astype(float)
    cols = [1 - a, a]                                  # one intercept per state
    if with_pos:
        F = fourier(pos)
        cols += list((F * (1 - a)[:, None]).T) + list((F * a[:, None]).T)
    cols += [spd, spd * a]
    return np.column_stack(cols)


def fit_cell(y, X):
    try:
        r = sm.GLM(y, X, family=sm.families.Poisson()).fit(maxiter=100)
    except Exception:
        return None
    b, se = r.params, r.bse
    return b[-2], b[-1], se[-2], se[-1]


def residual(v, B):
    out = np.empty(len(v))
    for tr, te in KFold(5, shuffle=True, random_state=0).split(B):
        coef = np.linalg.lstsq(B[tr], v[tr], rcond=None)[0]
        out[te] = v[te] - B[te] @ coef
    return out


def decode_resid(Y, spd, pos, idx):
    if len(idx) < 300:
        return np.nan, np.nan
    B = np.column_stack([np.ones(len(idx)), fourier(pos[idx])])
    ys = residual(spd[idx], B)
    Xr = np.column_stack([residual(Y[idx, i].astype(float), B) for i in range(Y.shape[1])])
    pred = np.empty(len(ys))
    for tr, te in KFold(5, shuffle=True, random_state=0).split(Xr):
        mu, sd = Xr[tr].mean(0), Xr[tr].std(0) + 1e-9
        m = Ridge(alpha=10.0).fit((Xr[tr] - mu) / sd, ys[tr])
        pred[te] = m.predict((Xr[te] - mu) / sd)
    return float(np.corrcoef(pred, ys)[0, 1]), float(ys.std())


def run_session(mo, dy, state, rng):
    got = binned_full(mo, dy, state)
    if got is None:
        return None, None
    Y, spd, pos, anch, ids = got
    if anch.sum() < MINBIN or (~anch).sum() < MINBIN:
        return None, None
    cs = common_support(spd, anch)
    if cs.sum() < 2 * MINBIN:
        return None, None
    Yc, sc, pc, ac = Y[cs], spd[cs], pos[cs], anch[cs]
    s10 = (sc - sc.mean()) / 10.
    Xp, X0 = design(pc, s10, ac), design(pc, s10, ac, with_pos=False)
    rows = []
    for i, c in enumerate(ids):
        y = Yc[:, i]
        if y[ac].sum() < MIN_SPK or y[~ac].sum() < MIN_SPK:
            continue
        fp, f0 = fit_cell(y, Xp), fit_cell(y, X0)
        if fp is None or f0 is None:
            continue
        rows.append(dict(mouse=mo, day=dy, cluster=int(c),
                         b_n=fp[0], b_int=fp[1], se_n=fp[2], se_int=fp[3],
                         b_n_nopos=f0[0], b_int_nopos=f0[1], se_int_nopos=f0[3],
                         rate_a=float(y[ac].mean() / BIN), rate_n=float(y[~ac].mean() / BIN),
                         n_a=int(ac.sum()), n_n=int((~ac).sum())))
    ia, iN = matched_bins(spd, anch, rng)
    k = min(len(ia), len(iN))
    ra, sda = decode_resid(Y, spd, pos, ia)
    rn, sdn = decode_resid(Y, spd, pos, iN)
    sess = dict(mouse=mo, day=dy, n_cells=len(ids), n_matched=int(k),
                dec_r_a=ra, dec_r_n=rn, sres_sd_a=sda, sres_sd_n=sdn,
                speed_sd_a=float(spd[anch].std()), speed_sd_n=float(spd[~anch].std()),
                lo=float(spd[cs].min()), hi=float(spd[cs].max()))
    return pd.DataFrame(rows), pd.DataFrame([sess])


if __name__ == '__main__':
    T = pd.read_csv(f'{OUT}/theta_frequency.csv')
    sess = sorted(set(zip(T.mouse, T.day)))
    cells, sessions, t0 = [], [], time.time()
    for k, (mo, dy) in enumerate(sess, 1):
        st = T[(T.mouse == mo) & (T.day == dy)]
        state = dict(zip(st.trial.astype(int), st.anch.astype(bool)))
        rng = np.random.default_rng(abs(hash((int(mo), int(dy)))) % 2 ** 32)
        try:
            C, S = run_session(int(mo), int(dy), state, rng)
        except Exception as e:
            print(f'  ! M{mo}D{dy}: {type(e).__name__}: {e}', flush=True)
            continue
        if C is None:
            print(f'  - M{mo}D{dy}: skipped', flush=True)
            continue
        cells.append(C); sessions.append(S)
        print(f'[{k}/{len(sess)}] M{mo}D{dy}: {len(C)} cells, {time.time() - t0:.0f}s', flush=True)
    sfx = '' if N_HARM == 4 else f'_h{N_HARM}'
    pd.concat(cells, ignore_index=True).to_csv(f'{OUT}/speed_state_glm_cells{sfx}.csv', index=False)
    pd.concat(sessions, ignore_index=True).to_csv(f'{OUT}/speed_state_glm_sessions{sfx}.csv', index=False)
    print('wrote speed_state_glm_cells.csv and speed_state_glm_sessions.csv')
