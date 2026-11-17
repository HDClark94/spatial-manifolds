"""Which simultaneously recorded MEC cells belong to the grid bump? A search.

Figure S18 reads the attractor bump from the grid module alone. If the
attractor is a population larger than its grid cells, other cells' activity
should carry the bump's torus coordinate too, and adding them should make the
bump more coherent on data the selection never saw. This script searches for
that population, per session, with cross-validated feature selection.

DATA. Every MEC cell of the session (attractor_cache.py): 100 ms rates
smoothed at 250 ms and z-scored, running bins only. Trials are split into five
folds of interleaved 8-trial blocks, so every fold holds both states. All
selection happens inside the four training folds; everything reported is
measured on the held-out fold.

1. THE REFERENCE BUMP. psi_j(t) = angle(sum_c r_c(t) exp(i phi_cj)) over the
   grid module, on the two torus axes of Figure S18 (spectral phases).

2. A SCORE PER CELL: torus tuning BEYOND position. On a track the bump's phase
   is a function of position whenever the map is anchored, so any place-tuned
   cell correlates with psi. The score is therefore the cross-validated gain
   in R^2 from adding the torus basis (cos, sin of psi_1, psi_2 and psi_1 -
   psi_2) to a model that already has state-specific position maps (4 Fourier
   harmonics per state) and running speed, against the same gain with psi
   circularly shifted in time (20 shifts of at least 60 s), as a z score. A
   grid cell is scored against a bump read from the OTHER grid cells, so it
   cannot predict itself.

3. FORWARD SELECTION (the wrapper). Starting from the grid module, the
   non-module cells are added one at a time in order of score. Each cell's
   phases come from its training data, phi_cj = angle(sum_t r_c(t) exp(i
   psi_j(t))). The objective is bump coherence above a random-phase null --
   the Figure S18 measure -- averaged over the two states with equal weight, on
   inner validation folds (each training fold in turn). The chosen size k* is
   the maximum of the mean inner-validation curve.

4. TESTED ON THE HELD-OUT FOLD, for (a) the grid module, (b) the module plus
   the selected cells, (c) the selected cells ALONE: bump coherence above the
   phase-permutation null in each state; and, for (c), agreement between the
   bump read from the selected non-grid cells alone and the one read from the
   grid module, mean cos(psi_c - psi_a), against grid split halves of matched
   size (attractor_reliability.py). (c) is the strong test: cells that merely
   echo position cannot reproduce the grid module's bump on non-anchored
   trials, where the bump and position part.

NULL RUN (NULL=1): every non-module cell's rate is circularly shifted in time
by its own random offset of at least 60 s before anything else, which keeps
each cell's own statistics and destroys any relation to the bump. Whatever the
search 'finds' then is what it finds by chance, and that is the floor every
result above is compared with.

Reads data/population_state/attractor_cache/*.npz
Writes data/population_state/attractor_search_{cells,sessions,curves}{_null}.csv
"""
import glob
import os
import time

import numpy as np
import pandas as pd

ROOT = '/Users/harryclark/Documents/spatial-manifolds'
PS = f'{ROOT}/data/population_state'
NULL = os.environ.get('NULL') == '1'
NFOLD, BLOCK = 5, 8
NSHIFT, MINSHIFT = 20, 600         # time-shift null, >= 60 s
NRAND = 10                         # random-phase draws in the selection objective
NPERM = 20                         # phase permutations in the held-out test
NHARM, TL = 4, 200.0


def pv_sum(Z, Q):
    """Bump coherence from running sums: Z = sum_c r_c exp(i phi_c) (per axis),
    Q = sum_c r_c^2, n cells -> |Z| / (sqrt(n/2) sqrt(Q))."""
    return np.abs(Z)


def coherence(R, ph):
    """Figure S18's population vector for one axis."""
    z = np.abs(np.exp(1j * ph) @ R)
    return z / (np.sqrt(len(ph) / 2.0) * np.linalg.norm(R, axis=0) + 1e-9)


def bump(R, phi):
    return np.stack([np.angle(np.exp(1j * phi[j]) @ R) for j in range(2)])


def pos_basis(pos, anch, spd):
    a = 2 * np.pi * pos / TL
    F = np.column_stack([f(k * a) for k in range(1, NHARM + 1) for f in (np.sin, np.cos)])
    s = anch.astype(float)
    sp = (spd - spd.mean()) / (spd.std() + 1e-9)
    return np.column_stack([1 - s, s, F * (1 - s)[:, None], F * s[:, None], sp, sp ** 2])


def tor_basis(psi):
    d = psi[0] - psi[1]
    return np.column_stack([np.cos(psi[0]), np.sin(psi[0]), np.cos(psi[1]), np.sin(psi[1]),
                            np.cos(d), np.sin(d)])


def cv_r2(X, Y, folds):
    """Cross-validated R^2 per column of Y."""
    ss_res = np.zeros(Y.shape[1]); ss_tot = np.zeros(Y.shape[1])
    for f in np.unique(folds):
        tr, te = folds != f, folds == f
        B = np.linalg.lstsq(X[tr], Y[tr], rcond=None)[0]
        e = Y[te] - X[te] @ B
        ss_res += (e ** 2).sum(0)
        ss_tot += ((Y[te] - Y[tr].mean(0)) ** 2).sum(0)
    return 1 - ss_res / ss_tot


def scores(R, mod_i, phi, Xp, folds, rng):
    """Torus gain beyond position per cell, z against time-shifted psi."""
    Y = R.T
    base = cv_r2(Xp, Y, folds)
    psi = bump(R[mod_i], phi)
    T = Y.shape[0]
    shifts = rng.integers(MINSHIFT, T - MINSHIFT, NSHIFT)

    def gain(ps, cols=None):
        Yc = Y if cols is None else Y[:, cols]
        b = base if cols is None else base[cols]
        return cv_r2(np.column_stack([Xp, tor_basis(ps)]), Yc, folds) - b
    g = gain(psi)
    gn = np.stack([gain(np.roll(psi, s, axis=1)) for s in shifts])
    # grid cells against the bump of the other module cells
    for k, c in enumerate(mod_i):
        rest = np.delete(mod_i, k)
        ps = bump(R[rest], np.delete(phi, k, axis=1))
        g[c] = gain(ps, [c])[0]
        gn[:, c] = [gain(np.roll(ps, s, axis=1), [c])[0] for s in shifts]
    z = (g - gn.mean(0)) / (gn.std(0) + 1e-9)
    return g, z


def time_phases(R, psi):
    """Each cell's torus phase on these data: angle of sum_t r(t) exp(i psi(t))."""
    return np.stack([np.angle(R @ np.exp(1j * psi[j])) for j in range(2)])


def forward_curve(R_tr, R_va, st_va, mod_i, order, psi_tr, phi_mod, rng):
    """Inner-validation coherence above random phases as cells are added."""
    ph = time_phases(R_tr, psi_tr)                 # every cell, from inner train
    ph[:, mod_i] = phi_mod                          # the module keeps its own
    rnd = rng.uniform(-np.pi, np.pi, (NRAND, 2, R_tr.shape[0]))
    a, n = st_va, ~st_va
    seq = list(mod_i) + list(order)
    Z = np.zeros((2, R_va.shape[1]), complex); Zr = np.zeros((NRAND, 2, R_va.shape[1]), complex)
    Q = np.zeros(R_va.shape[1])
    out = []
    for i, c in enumerate(seq):
        r = R_va[c]
        Z += np.exp(1j * ph[:, c])[:, None] * r[None]
        Zr += np.exp(1j * rnd[:, :, c])[:, :, None] * r[None, None]
        Q += r ** 2
        if i + 1 < len(mod_i):
            continue
        den = np.sqrt((i + 1) / 2.0) * np.sqrt(Q) + 1e-9
        e = (np.abs(Z) / den).mean(0) - (np.abs(Zr) / den).mean((0, 1))
        out.append(.5 * (e[a].mean() + e[n].mean()))
    return np.array(out)                           # index 0 = module alone


def held_out(R_te, a, n, phi, rng):
    """Coherence above the phase-permutation null, per state."""
    o = np.mean([coherence(R_te, phi[j]) for j in range(2)], 0)
    s = np.mean([np.mean([coherence(R_te, rng.permutation(phi[j])) for j in range(2)], 0)
                 for _ in range(NPERM)], 0)
    d = o - s
    return d[a].mean(), d[n].mean()


def agree(psi_x, psi_y, a, n):
    c = np.cos(psi_x - psi_y).mean(0)
    return c[a].mean(), c[n].mean()


def run(f, rng):
    z = np.load(f)
    mo, dy = int(z['mouse']), int(z['day'])
    ok, idx = z['ok'].astype(bool), z['idx']
    R = z['R'].astype(float)
    cells = list(z['cells']); ident = z['ident']
    mod_i = np.array([cells.index(c) for c in z['module']])
    phi = z['phi']
    if NULL:
        for c in range(len(cells)):
            if c not in set(mod_i):
                R[c] = np.roll(R[c], rng.integers(MINSHIFT, R.shape[1] - MINSHIFT))
    R, idx, anch = R[:, ok], idx[ok], z['anch'][ok].astype(bool)
    pos, spd = z['trackpos'][ok].astype(float), z['spd'][ok].astype(float)
    fold = (idx // BLOCK) % NFOLD
    Xp = pos_basis(pos, anch, spd)
    nonmod = np.setdiff1d(np.arange(len(cells)), mod_i)
    crow, srow, curv = [], [], []
    for f_ in range(NFOLD):
        tr, te = fold != f_, fold == f_
        Rtr, Rte = R[:, tr], R[:, te]
        inner = fold[tr]
        g, zs = scores(Rtr, mod_i, phi, Xp[tr], inner, rng)
        order = nonmod[np.argsort(-zs[nonmod])]
        # the wrapper: inner-validation curve, averaged over inner folds
        cv = []
        for v in np.unique(inner):
            itr, iva = inner != v, inner == v
            psi_i = bump(Rtr[mod_i][:, itr], phi)
            cv.append(forward_curve(Rtr[:, itr], Rtr[:, iva], anch[tr][iva], mod_i, order,
                                    psi_i, phi, rng))
        cv = np.mean(cv, 0)
        kstar = int(np.argmax(cv))
        sel = order[:kstar]
        # phases for the test, from all of the training data
        psi_tr = bump(Rtr[mod_i], phi)
        ph = time_phases(Rtr, psi_tr); ph[:, mod_i] = phi
        a_, n_ = anch[te], ~anch[te]
        res = dict(mouse=mo, day=dy, fold=f_, n_cells=len(cells), n_mod=len(mod_i), k_star=kstar,
                   gain_curve_max=float(cv.max() - cv[0]))
        res['mod_a'], res['mod_n'] = held_out(Rte[mod_i], a_, n_, phi, rng)
        both = np.r_[mod_i, sel]
        res['both_a'], res['both_n'] = held_out(Rte[both], a_, n_, ph[:, both], rng)
        psi_g = bump(Rte[mod_i], phi)
        if kstar >= 2:
            res['sel_a'], res['sel_n'] = held_out(Rte[sel], a_, n_, ph[:, sel], rng)
            res['agree_a'], res['agree_n'] = agree(bump(Rte[sel], ph[:, sel]), psi_g, a_, n_)
            # chance: the selected cells with uniform random phases (permuting
            # the phases of a handful of cells leaves most of their structure)
            pa = [agree(bump(Rte[sel], rng.uniform(-np.pi, np.pi, (2, len(sel)))),
                        psi_g, a_, n_) for _ in range(NPERM)]
            res['agree_null_a'], res['agree_null_n'] = np.mean(pa, 0)
        srow.append(res)
        for k_, v in enumerate(cv):
            curv.append(dict(mouse=mo, day=dy, fold=f_, k=k_, obj=v))
        for c in range(len(cells)):
            crow.append(dict(mouse=mo, day=dy, fold=f_, cluster_id=int(cells[c]), identity=ident[c],
                             module=c in set(mod_i), gain=g[c], z=zs[c],
                             rank=int(np.where(order == c)[0][0]) if c in set(nonmod) else -1,
                             selected=c in set(sel)))
    return pd.DataFrame(crow), pd.DataFrame(srow), pd.DataFrame(curv)


if __name__ == '__main__':
    rng = np.random.default_rng(7 if NULL else 0)
    only, t0 = os.environ.get('ONLY'), time.time()
    C, S, K = [], [], []
    for f in sorted(glob.glob(f'{PS}/attractor_cache/M*D*.npz')):
        if only and only not in f:
            continue
        c, s, k = run(f, rng)
        C.append(c); S.append(s); K.append(k)
        m = s.mean(numeric_only=True)
        print(f'  {os.path.basename(f)[:-4]}: {int(m.n_cells)} cells, module {int(m.n_mod)}; '
              f'k* {s.k_star.median():.0f}; held-out coherence a/n module {m.mod_a:.4f}/{m.mod_n:.4f} '
              f'-> with selected {m.both_a:.4f}/{m.both_n:.4f}; selected alone agree '
              f'{m.get("agree_a", np.nan):.3f}/{m.get("agree_n", np.nan):.3f} '
              f'(chance {m.get("agree_null_a", np.nan):.3f}); {time.time() - t0:.0f}s', flush=True)
    sfx = '_null' if NULL else ''
    if only:
        sfx += f'_{only}'
    pd.concat(C).to_csv(f'{PS}/attractor_search_cells{sfx}.csv', index=False)
    pd.concat(S).to_csv(f'{PS}/attractor_search_sessions{sfx}.csv', index=False)
    pd.concat(K).to_csv(f'{PS}/attractor_search_curves{sfx}.csv', index=False)
    print('done')
