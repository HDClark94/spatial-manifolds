"""Does the kind of theta cycle predict what the MEC population encodes on it?

The tSC analysis (tsc_analyse.py) found that anchored and non-anchored trials
are built from nearly the same kinds of theta cycle (state decodable from cycle
type at AUC 0.53). This asks the other direction Esparza et al. (2023) propose:
colour the NEURAL activity by cycle type. Within a state, are cycles of one type
associated with a population code closer to the task frame than cycles of
another?

THE CODE'S ALIGNMENT, PER CYCLE. A Poisson decoder (the paper's own, from
position_decoding_by_anchoring.ipynb: smoothed rate maps on the 200 cm ring,
spatial cells only) is trained on ANCHORED trials, so its maps define the task
frame, and applied to each theta cycle's spike counts with the cycle's own
duration. Anchored cycles are decoded with maps from the other folds' anchored
trials (5-fold by trial), so no trial decodes itself; non-anchored cycles with
maps from every anchored trial. The decoding error is how far the population is
from the task frame on that cycle.

CONFOUNDS. Speed, theta amplitude, the cycle's total spike count and position on
the track all change both gamma and firing. Every within-session correlation is
partial, on residuals of all four (position entering as sine, cosine, second
harmonics and a 10-bin indicator, as in gamma_vs_decode_error.py). Sessions are
the unit; Holm across the components within each state.

Writes data/lfp/tsc_neural.csv (per session x state x component) and
       data/lfp/tsc_neural_cycles/M{mo}D{dy}.npz (per cycle, for the manifold)
"""
import glob
import os
import sys
import warnings

import numpy as np
import pandas as pd

warnings.filterwarnings('ignore')
FIG = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, FIG)
from scipy.stats import spearmanr, wilcoxon

import gamma_vs_decode_error as G          # notebook decoder + session binning
import tsc_analyse as A

ROOT = '/Users/harryclark/Documents/spatial-manifolds'
TSC = f'{ROOT}/data/lfp/tsc'
OUTC = f'{ROOT}/data/lfp/tsc_neural_cycles'
MIN_CYC = 200
MIN_ANCH_TR = 10


def cycle_counts(clusters, ids, t0, dur):
    edges = np.c_[t0, t0 + dur]
    C = np.zeros((len(t0), len(ids)), np.float32)
    for j, c in enumerate(ids):
        sp = np.asarray(clusters[c].index, float)
        C[:, j] = np.searchsorted(sp, edges[:, 1]) - np.searchsorted(sp, edges[:, 0])
    return C


def decode_cycles(f, counts, dur):
    logf = np.log(f + G.EPS)
    ll = counts @ logf.T - dur[:, None] * f.sum(axis=1)[None, :]
    return np.argmax(ll, axis=1)


def resid(v, Z):
    return v - Z @ np.linalg.lstsq(Z, v, rcond=None)[0]


def session(fn, project, k):
    key = os.path.basename(fn)[:-4]
    mo, dy = int(key[1:key.index('D')]), int(key[key.index('D') + 1:])
    z = np.load(fn, allow_pickle=True)
    x, level, _ = A.load_session(fn, 'ind')
    W = project(x)
    t0, dur, trial = z['t0'].astype(float), z['dur'].astype(float), z['trial'].astype(int)
    anch, spd, tamp = z['anch'].astype(bool), z['speed'].astype(float), z['theta_amp'].astype(float)

    bp, cp = G.vr_paths(mo, dy)
    beh = G.nap.load_file(bp); clusters = G.nap.load_file(cp)
    cls = G.CLS[(G.CLS.mouse == mo) & (G.CLS.day == dy)]
    ids = [int(c) for c in cls[cls.cell_class_of1.isin(['GC', 'NGS'])].cluster_id
           if int(c) in clusters.index]
    if len(ids) < G.MIN_CELL:
        return None
    # 200 ms training bins, the decoder's own resolution
    B = G.session(mo, dy)
    if B is None:
        return None
    T = pd.read_csv(f'{ROOT}/data/population_state/anchoring_trials.csv')
    st = T[(T.mouse == mo) & (T.day == dy)]
    tr_anch = dict(zip(st.trial.astype(int), st.frac_anch > 0.5))
    b_anch = np.array([tr_anch.get(int(u), False) for u in B['trial']])
    a_trials = np.unique(B['trial'][b_anch])
    if len(a_trials) < MIN_ANCH_TR:
        return None
    # the cycles' own counts and true position at the cycle's midpoint
    C = cycle_counts(clusters, ids, t0, dur)
    P = beh['P']
    tm = t0 + dur / 2
    pos = np.asarray(P.values)[np.searchsorted(np.asarray(P.index), tm).clip(0, len(P) - 1)]
    pb = (np.floor(pos / G.PBIN).astype(int) % G.NP)
    # cluster ids in B are in the order of ids? rebuild B's counts on our ids
    idmap = {c: j for j, c in enumerate(B.get('ids', ids))}
    Bc = B['counts']
    if Bc.shape[1] != len(ids):
        return None
    dec = np.full(len(t0), -1)
    folds = np.array_split(np.random.default_rng(mo * 100 + dy).permutation(a_trials), 5)
    for fold in folds:
        trn = b_anch & ~np.isin(B['trial'], fold)
        f = G.tuning(Bc[trn], B['pbin'][trn])
        m = anch & np.isin(trial, fold)
        if m.any():
            dec[m] = decode_cycles(f, C[m], dur[m])
    f_all = G.tuning(Bc[b_anch], B['pbin'][b_anch])
    m = ~anch
    if m.any():
        dec[m] = decode_cycles(f_all, C[m], dur[m])
    err = G.circ_err_cm(dec, pb)
    ok = dec >= 0
    np.savez_compressed(f'{OUTC}/{key}.npz', W=W[ok].astype(np.float32), level=level[ok],
                        err=err[ok], anch=anch[ok], spd=spd[ok], tamp=tamp[ok],
                        nspk=C[ok].sum(1), pos=pos[ok], counts=C[ok], trial=trial[ok])
    rows = []
    ph = 2 * np.pi * pb / G.NP
    for state in (True, False):
        mm = ok & (anch == state)
        if mm.sum() < MIN_CYC:
            continue
        Z = np.c_[np.ones(mm.sum()), spd[mm], np.log(tamp[mm]), C[mm].sum(1),
                  np.sin(ph[mm]), np.cos(ph[mm]), np.sin(2 * ph[mm]), np.cos(2 * ph[mm]),
                  np.eye(10)[(pb[mm] / G.NP * 10).astype(int).clip(0, 9)]]
        e_r = resid(err[mm].astype(float), Z)
        for c in range(k):
            r = spearmanr(resid(W[mm, c], Z), e_r)[0]
            rows.append(dict(session=key, state='anch' if state else 'non', component=c,
                             rho=r, n=int(mm.sum()), median_err=float(np.median(err[mm]))))
        r = spearmanr(resid(level[mm], Z), e_r)[0]
        rows.append(dict(session=key, state='anch' if state else 'non', component='level',
                         rho=r, n=int(mm.sum()), median_err=float(np.median(err[mm]))))
    return pd.DataFrame(rows) if rows else None


if __name__ == '__main__':
    os.makedirs(OUTC, exist_ok=True)
    files = sorted(f for f in glob.glob(f'{TSC}/M*D*.npz') if 'partial' not in f)
    rng = np.random.default_rng(0)
    S = [A.load_session(f, 'ind')[0] for f in files]
    X = np.concatenate([x[rng.choice(len(x), min(A.N_PER_SESSION, len(x)), replace=False)]
                        for x in S])
    T, project = A.fit(X, A.K, A.SEED)
    out = []
    for fn in files:
        try:
            d = session(fn, project, A.K)
        except Exception as e:
            print(f'  ! {os.path.basename(fn)}: {type(e).__name__}: {e}', flush=True); continue
        if d is not None:
            out.append(d)
            print(f'  {os.path.basename(fn)[:-4]}: '
                  + ', '.join(f'{s} median err {g.median_err.iloc[0]:.0f} cm (n={g.n.iloc[0]})'
                              for s, g in d.groupby('state')), flush=True)
    D = pd.concat(out, ignore_index=True)
    D.to_csv(f'{ROOT}/data/lfp/tsc_neural.csv', index=False)
    print()
    for state in ('anch', 'non'):
        res = []
        for c in [0, 1, 2, 3, 'level']:
            v = D[(D.state == state) & (D.component.astype(str) == str(c))].rho.dropna()
            if len(v) >= 6:
                res.append((c, v.median(), int((v > 0).sum()), len(v), wilcoxon(v).pvalue))
        h = A.holm([r[4] for r in res])
        print(f'{state}: partial rho (component weight vs decoding error; negative = '
              f'that component goes with a code closer to the task frame)')
        for (c, med, pos_, n, p), hh in zip(res, h):
            print(f'   tSC{c}: median rho {med:+.4f}, positive in {pos_}/{n}, p = {p:.3g}, Holm {hh:.3g}')
