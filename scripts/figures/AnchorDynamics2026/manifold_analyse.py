"""Does the anchored manifold come apart into a grid ring and a track ring?

Reads data/manifold/M{mo}D{dy}.npz (manifold_build.py).

THE MANIFOLD. For one session and one group of cells, the anchored manifold is
the ring of population vectors traced over the track on ANCHORED trials: A[y] =
mean over anchored trials of the group's population vector at track bin y (sqrt
rates, each cell z-scored over the session). A trial scored against it never
contributes to it (leave-one-trial-out for anchored trials).

ONE MATRIX PER TRIAL AND GROUP. C[x, y] = correlation across the group's cells
between the trial's population vector at track position x and the manifold at
y. Read from it:

  r0      mean_x C[x, x]              on the ring, at the right place
  shift   max_s mean_x C[x, x+s]      on the ring, displaced by one offset s
  warp    max_{s,g} mean_x C[x, s+gx] displaced and traversed at gain g
  free    mean_x max_y C[x, y]        on the ring anywhere, in any order

Each grows with freedom on noise alone, so each is also computed with the
group's cell identities permuted (same statistics, no ring: the floor).

GROUPS. Grid cells (open-field GC) and non-grid spatial cells (NGS), each on its
own ring. Because each group's vectors are standardised separately, the joint
correlation is the cell-weighted mean of the group correlations, which makes
the comparison between models exact:

  one ring    max_s [w_gc C_gc(s) + w_ng C_ng(s)]      one shared offset
  two rings   w_gc max C_gc + w_ng max C_ng             independent offsets

In the anchored state the two should coincide (both offsets ~0). If anchoring is
the locking of the grid ring to the track, non-anchored trials need two.

WHO FOLLOWS WHOM. On non-anchored trials, each cell's own map is read against
its anchored template at zero offset, at the grid population's offset for that
lap (from the OTHER grid cells when the cell is a grid cell), and against the
grid offset of a different non-anchored lap (the null: the same distribution of
offsets, nothing shared). follow_grid = r(at grid offset) - r(at another lap's
grid offset); follow_track = r(0) - r(at another lap's grid offset).

WHERE IN THE LAP. The grid ring's local offset in a 40 cm window slid along the
track: flat for a rigid displacement, a slope for a gain error.

BEHAVIOUR. Offsets estimated from the track BEFORE the stop only (start to 20
cm short of it), for each ring; if the animal stops where a ring says reward is,
stop error = -offset.

Writes data/manifold/results/M{mo}D{dy}_trials.csv, _cells.csv, _session.json,
_profiles.npz
"""
import glob
import json
import os
import sys
import warnings

import numpy as np
import pandas as pd

warnings.filterwarnings('ignore')
ROOT = '/Users/harryclark/Documents/spatial-manifolds'
IND = f'{ROOT}/data/manifold'
OUTD = f'{IND}/results'
NB = 50                     # 4 cm bins
BIN_CM = 4.0
GAINS = np.round(np.arange(.30, 1.55, .05), 2)
MIN_VALID = .8              # fraction of the lap with running data
MIN_TR = 10                 # trials per state
MIN_GC, MIN_NG = 5, 8
N_PERM = 4
WIN = 10                    # 40 cm local window
PRE_GAP = 5                 # 20 cm short of the stop
MIN_PRE = 10                # >= 40 cm of track before that


def zc(V):
    V = V - V.mean(-1, keepdims=True)
    return V / (np.sqrt((V ** 2).mean(-1, keepdims=True)) + 1e-9)


def cmat(T, A):
    return zc(T) @ zc(A).T / T.shape[-1]


X_ = np.arange(NB)
IDX_S = (X_[None, :] + X_[:, None]) % NB                         # [s, x]
IDX_W = np.array([[(np.round(s + g * X_).astype(int)) % NB for s in range(NB)]
                  for g in GAINS])                                # [g, s, x]


def shift_profile(C, m=None):
    xm = X_ if m is None else X_[m]
    return C[xm[None, :], IDX_S[:, xm]].mean(1)


def scores(C, m=None):
    xm = X_ if m is None else X_[m]
    sh = shift_profile(C, m)
    wp = C[xm[None, None, :], IDX_W[:, :, xm]].mean(2)
    gi, si = np.unravel_index(np.argmax(wp), wp.shape)
    return dict(r0=float(C[xm, xm].mean()), shift=float(sh.max()), s=int(sh.argmax()),
                warp=float(wp[gi, si]), g=float(GAINS[gi]), sw=int(si),
                free=float(C[xm].max(1).mean()), prof=sh)


def wrap_cm(s):
    return ((np.asarray(s) + NB // 2) % NB - NB // 2) * BIN_CM


def cell_xcorr(v, a):
    """Correlation of one cell's trial map v with its template a at every shift s."""
    v = v - v.mean(); a = a - a.mean()
    d = np.sqrt((v ** 2).sum() * (a ** 2).sum())
    if d == 0:
        return np.full(NB, np.nan)
    return np.array([(v * a[(X_ + s) % NB]).sum() for s in range(NB)]) / d


def session(fn, rng):
    key = os.path.basename(fn)[:-4]
    z = np.load(fn)
    X = np.sqrt(np.clip(z['maps'].astype(np.float32), 0, None))
    ok_tr = (z['valid'] >= MIN_VALID) & np.isfinite(z['pc1']) & (z['nspk'] > 0)
    ok_tr &= np.isfinite(X).all(axis=(1, 2))
    anch = z['pc1'] > 0
    if (ok_tr & anch).sum() < MIN_TR or (ok_tr & ~anch).sum() < MIN_TR:
        return None
    sd0 = X[ok_tr].std(axis=(0, 1))
    keepc = (np.nanmean(X[ok_tr], axis=(0, 1)) > np.sqrt(.2)) & (sd0 > 0)
    X, cls, ids = X[:, :, keepc], z['cls'][keepc], z['ids'][keepc]
    gi_, ni_ = np.flatnonzero(cls == 'GC'), np.flatnonzero(cls == 'NGS')
    if len(gi_) < MIN_GC or len(ni_) < MIN_NG:
        return None
    X = (X - X[ok_tr].mean(axis=(0, 1))) / X[ok_tr].std(axis=(0, 1))
    ia, inn = np.flatnonzero(ok_tr & anch), np.flatnonzero(ok_tr & ~anch)
    Asum = X[ia].sum(0)
    wg, wn = len(gi_) / (len(gi_) + len(ni_)), len(ni_) / (len(gi_) + len(ni_))
    G = {'gc': gi_, 'ng': ni_}
    rows, local, Cex, s_gc_tr, cellr = [], {}, {}, {}, {}
    for t in np.flatnonzero(ok_tr):
        is_a = bool(anch[t])
        A = (Asum - X[t]) / (len(ia) - 1) if is_a else Asum / len(ia)
        r = dict(trial=int(z['trial'][t]), anch=is_a, ttype=str(z['ttype'][t]),
                 perf=str(z['perf'][t]), stop=float(z['stop'][t]))
        S = {}
        for g, ix in G.items():
            C = cmat(X[t][:, ix], A[:, ix])
            S[g] = scores(C)
            fl = [scores(cmat(X[t][:, ix][:, p], A[:, ix]))
                  for p in (rng.permutation(len(ix)) for _ in range(N_PERM))]
            for k in ('r0', 'shift', 'warp', 'free', 'g'):
                r[f'{g}_{k}'] = S[g][k]
            for k in ('r0', 'shift', 'warp', 'free'):
                r[f'{g}_{k}_floor'] = float(np.mean([f[k] for f in fl]))
            r[f'{g}_s_cm'] = float(wrap_cm(S[g]['s'])); r[f'{g}_sw_cm'] = float(wrap_cm(S[g]['sw']))
            if g == 'gc':
                loc = []
                for c0 in range(0, NB - WIN + 1, 2):
                    mk = np.zeros(NB, bool); mk[c0:c0 + WIN] = True
                    loc.append(scores(C, mk)['s'])
                local[t] = np.array(loc)
            if t in ia[:4].tolist() + inn[:8].tolist():
                Cex[(t, g)] = C.astype(np.float32)
            # behaviour: offset from the track before the stop only
            if np.isfinite(z['stop'][t]):
                xs = int(z['stop'][t] // BIN_CM)
                mk = np.zeros(NB, bool); mk[:max(xs - PRE_GAP, 0)] = True
                if mk.sum() >= MIN_PRE:
                    Sp = scores(C, mk)
                    r[f'{g}_s_pre_cm'] = float(wrap_cm(Sp['s']))
        # one ring against two
        joint = wg * S['gc']['prof'] + wn * S['ng']['prof']
        r['one_ring'] = float(joint.max()); r['one_s_cm'] = float(wrap_cm(joint.argmax()))
        r['two_ring'] = wg * S['gc']['shift'] + wn * S['ng']['shift']
        r['joint_r0'] = wg * S['gc']['r0'] + wn * S['ng']['r0']
        r['ng_at_gc'] = float(S['ng']['prof'][S['gc']['s']])      # ng ring at the grid offset
        rows.append(r)
        s_gc_tr[t] = S['gc']['s']
        # per-cell maps against templates, for non-anchored trials
        if not is_a:
            cellr[t] = np.array([cell_xcorr(X[t][:, j], A[:, j]) for j in range(X.shape[2])])
    D = pd.DataFrame(rows)
    D.insert(0, 'session', key)
    # ---- who follows whom (non-anchored trials)
    tl = list(cellr)
    crow = []
    if len(tl) >= MIN_TR:
        R = np.stack([cellr[t] for t in tl])               # [trial, cell, shift]
        sgc = np.array([s_gc_tr[t] for t in tl])
        # grid offset for each grid cell from the OTHER grid cells
        sgc_loo = np.tile(sgc[:, None], (1, X.shape[2]))
        for j in gi_:
            others = gi_[gi_ != j]
            for k, t in enumerate(tl):
                A = Asum / len(ia)
                sgc_loo[k, j] = scores(cmat(X[t][:, others], A[:, others]))['s']
        for j in range(X.shape[2]):
            rj = R[:, j, :]
            at0 = np.nanmean(rj[:, 0])
            atg = np.nanmean(rj[np.arange(len(tl)), sgc_loo[:, j]])
            # null: the grid offset of another non-anchored lap
            nul = np.nanmean([np.nanmean(rj[k, np.delete(sgc_loo[:, j], k)]) for k in range(len(tl))])
            best = np.nanmean(np.nanmax(rj, 1))
            crow.append(dict(session=key, cluster_id=int(ids[j]), cls=str(cls[j]),
                             r_zero=at0, r_grid=atg, r_null=nul, r_best=best))
    Cc = pd.DataFrame(crow)
    # ---- session summary
    out = dict(session=key, n_gc=len(gi_), n_ngs=len(ni_), n_anch=int(D.anch.sum()),
               n_non=int((~D.anch).sum()))
    for g in ('gc', 'ng'):
        for k in ('r0', 'shift', 'warp', 'free'):
            top = D[D.anch][f'{g}_{k}'].mean() - D[D.anch][f'{g}_{k}_floor'].mean()
            for st, m in (('anch', D.anch), ('non', ~D.anch)):
                out[f'{g}_{k}_{st}'] = float(D[m][f'{g}_{k}'].mean())
                out[f'{g}_{k}_{st}_norm'] = float((D[m][f'{g}_{k}'] - D[m][f'{g}_{k}_floor']).mean() / top)
        for st, m in (('anch', D.anch), ('non', ~D.anch)):
            out[f'{g}_disp20_{st}'] = float((np.abs(D[m][f'{g}_s_cm']) > 20).mean())
            out[f'{g}_gain_{st}'] = float(D[m][f'{g}_g'].median())
    for st, m in (('anch', D.anch), ('non', ~D.anch)):
        out[f'one_{st}'] = float(D[m].one_ring.mean()); out[f'two_{st}'] = float(D[m].two_ring.mean())
        out[f'joint_r0_{st}'] = float(D[m].joint_r0.mean())
        out[f'agree20_{st}'] = float((np.abs(wrap_cm((D[m].gc_s_cm - D[m].ng_s_cm) / BIN_CM)) <= 20).mean())
    if len(Cc):
        for c in ('GC', 'NGS'):
            cc = Cc[Cc.cls == c]
            out[f'{c}_follow_grid'] = float((cc.r_grid - cc.r_null).mean())
            out[f'{c}_follow_track'] = float((cc.r_zero - cc.r_null).mean())
    os.makedirs(OUTD, exist_ok=True)
    D.to_csv(f'{OUTD}/{key}_trials.csv', index=False)
    Cc.to_csv(f'{OUTD}/{key}_cells.csv', index=False)
    json.dump(out, open(f'{OUTD}/{key}_session.json', 'w'), indent=1)
    np.savez_compressed(f'{OUTD}/{key}_profiles.npz', trials=np.array(list(local)),
                        local=np.array(list(local.values())), anch=anch[list(local)],
                        Cex_keys=np.array([f'{t}_{g}' for t, g in Cex]),
                        Cex=np.array(list(Cex.values())))
    return out


if __name__ == '__main__':
    files = sorted(glob.glob(f'{IND}/M*D*.npz'))
    if len(sys.argv) > 1:
        files = [f for f in files if any(a in f for a in sys.argv[1:])]
    for fn in files:
        try:
            o = session(fn, np.random.default_rng(0))
        except Exception as e:
            print(f'  ! {os.path.basename(fn)}: {type(e).__name__}: {e}', flush=True); continue
        if o:
            print(f"  {o['session']}: GC {o['n_gc']} NGS {o['n_ngs']}, {o['n_anch']}/{o['n_non']} | "
                  f"non: gc r0 {o['gc_r0_non_norm']:.2f} shift {o['gc_shift_non_norm']:.2f} disp "
                  f"{o['gc_disp20_non']:.2f} | ng r0 {o['ng_r0_non_norm']:.2f} shift "
                  f"{o['ng_shift_non_norm']:.2f} disp {o['ng_disp20_non']:.2f} | one {o['one_non']:.3f} "
                  f"two {o['two_non']:.3f} | NGS follow grid {o.get('NGS_follow_grid', np.nan):+.3f} "
                  f"track {o.get('NGS_follow_track', np.nan):+.3f}", flush=True)
