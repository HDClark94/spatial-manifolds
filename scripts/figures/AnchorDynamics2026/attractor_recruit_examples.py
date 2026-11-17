"""The recruited bump on held-out data: maps, frames and speed, every session.

attractor_population_search.py chooses, per session and fold, the non-module
MEC cells that make the held-out bump more coherent. This script rebuilds the
bump those choices give, on held-out trials only, in the forms Figure S14
draws the grid module's bump, so the two can be put side by side:

  MAPS    the bump's phase on all three lattice axes, trial by trial and
          position by position (S14 B, C), from the position-binned rates of
          each held-out trial: the module alone, the module with the cells
          chosen for that trial's fold, and the chosen cells alone.
  TIME    the same three phases in 100 ms bins, for the frame strips (S14 F)
          and the bump's speed.
  SPEED   the bump's speed against the animal's (S14 J), module alone and
          module plus chosen cells, with the same phase shuffle as S14 (each
          set's phases permuted among its own cells), per state.

Every chosen cell's phases on the three axes come from the training folds:
angle(sum_t r(t) exp(i psi_j(t))), psi the module's bump on those folds. The
module keeps its own position-domain phases, as in S14. Nothing about a
held-out trial is used to choose or place the cells that read it.

Reads the cache and attractor_search_cells.csv
Writes data/population_state/attractor_recruit_speed.csv and
       data/population_state/attractor_recruit/M{mo}D{dy}.npz
"""
import glob
import os

import numpy as np
import pandas as pd

ROOT = '/Users/harryclark/Documents/spatial-manifolds'
PS = f'{ROOT}/data/population_state'
OUT = f'{PS}/attractor_recruit'
NFOLD, BLOCK, NPERM = 5, 8, 20
TL, DT = 200.0, 0.1


def proj(ph, X):
    """Phase on each axis: angle(sum_c x_c exp(i ph_jc))."""
    return np.stack([np.angle(np.exp(1j * p) @ X) for p in ph])


def run(f, CS, rng):
    z = np.load(f)
    mo, dy = int(z['mouse']), int(z['day'])
    cells = list(z['cells'])
    mod_i = np.array([cells.index(c) for c in z['module']])
    pa = z['phi_all']
    R = z['R'].astype(float)
    nb = int(z['nbin'])
    P = z['P'].astype(float).reshape(len(cells), -1, nb)
    ntr = P.shape[1]
    ok, idx = z['ok'].astype(bool), z['idx']
    anch, non = z['anch'].astype(bool), z['non'].astype(bool)
    fold_t = np.where(idx >= 0, (idx // BLOCK) % NFOLD, -1)
    fold_p = (np.arange(ntr) // BLOCK) % NFOLD
    cs = CS[(CS.mouse == mo) & (CS.day == dy)]
    T, na = R.shape[1], len(pa)
    out = {k: np.full((na, T), np.nan) for k in ('t_mod', 't_both', 't_sel')}
    out.update({k: np.full((na, ntr, nb), np.nan) for k in ('p_mod', 'p_both', 'p_sel')})
    phases, chosen = {}, {}
    for f_ in range(NFOLD):
        sel = np.array([cells.index(c) for c in
                        cs[(cs.fold == f_) & cs.selected].cluster_id if c in cells], int)
        tr, te = ok & (fold_t != f_), fold_t == f_
        ph = proj(pa, R[mod_i][:, tr])                     # the module's bump, training
        ph = np.stack([np.angle(R[:, tr] @ np.exp(1j * q)) for q in ph])
        ph[:, mod_i] = pa
        phases[f_], chosen[f_] = ph, sel
        both = np.r_[mod_i, sel]
        pt = fold_p == f_
        for k, ii in (('mod', mod_i), ('both', both), ('sel', sel)):
            if len(ii) < 2:
                continue
            out[f't_{k}'][:, te] = proj(ph[:, ii], R[ii][:, te])
            out[f'p_{k}'][:, pt] = np.stack([np.angle(np.einsum('c,ctb->tb', np.exp(1j * q),
                                                                  P[ii][:, pt]))
                                             for q in ph[:, ii]])
    # the bump's speed, as attractor_coherence.bump_speed: the two axes that
    # advance along the track, converted to cm/s by the module's phase advance
    # per cm over anchored trials
    pp = out['p_mod']
    adv = [abs(np.nanmedian(np.unwrap(w, axis=1)[:, -1] - np.unwrap(w, axis=1)[:, 0])) for w in pp]
    kk = list(np.argsort(adv)[::-1][:2])
    st = np.asarray(z['state'])[:ntr]
    dpdx = np.array([np.median(np.diff(np.unwrap(pp[k][st], axis=1), axis=1)) / (TL / nb)
                     for k in kk])
    v = z['spd'][1:].astype(float)
    same = fold_t[1:] == fold_t[:-1]
    ma, mn = anch[1:] & anch[:-1] & same, non[1:] & non[:-1] & same

    def speed(psi):
        d = (np.diff(psi[kk], axis=1) + np.pi) % (2 * np.pi) - np.pi
        return (d / dpdx[:, None]).mean(0) / DT

    def r(m, b):
        g = m & np.isfinite(b)
        return float(np.corrcoef(b[g], v[g])[0, 1])

    row = dict(mouse=mo, day=dy, n_mod=len(mod_i),
               k_star=float(np.median([len(chosen[f_]) for f_ in chosen])))
    for k in ('mod', 'both'):
        bs = speed(out[f't_{k}'])
        row[f'r_{k}_a'], row[f'r_{k}_n'] = r(ma, bs), r(mn, bs)
        nul = []
        for _ in range(NPERM):
            b0 = np.full(T - 1, np.nan)
            for f_ in range(NFOLD):
                ii = mod_i if k == 'mod' else np.r_[mod_i, chosen[f_]]
                te = fold_t == f_
                q = np.stack([rng.permutation(p) for p in phases[f_][:, ii]])
                ps = np.full((na, T), np.nan)
                ps[:, te] = proj(q, R[ii][:, te])
                b = speed(ps)
                b0[np.isfinite(b)] = b[np.isfinite(b)]
            nul.append((r(ma, b0), r(mn, b0)))
        row[f'rnull_{k}_a'], row[f'rnull_{k}_n'] = np.mean(nul, 0)
        out[f'bs_{k}'] = np.r_[np.nan, bs]
    sv = dict(out, kk=np.array(kk), dpdx=dpdx, mod_i=mod_i, cells=np.array(cells),
              ident=z['ident'], state=st, fold_p=fold_p, fold_t=fold_t,
              **{f'sel{f_}': chosen[f_] for f_ in chosen},
              **{f'ph{f_}': phases[f_] for f_ in phases})
    np.savez_compressed(f'{OUT}/M{mo}D{dy}.npz', **sv)
    return row


if __name__ == '__main__':
    os.makedirs(OUT, exist_ok=True)
    CS = pd.read_csv(f'{PS}/attractor_search_cells.csv')
    rng, rows = np.random.default_rng(5), []
    only = os.environ.get('ONLY')
    for f in sorted(glob.glob(f'{PS}/attractor_cache/M*D*.npz')):
        if only and only not in f:
            continue
        row = run(f, CS, rng)
        rows.append(row)
        print(f"  M{row['mouse']}D{row['day']}: k* {row['k_star']:.0f}; speed r a/n module "
              f"{row['r_mod_a']:.3f}/{row['r_mod_n']:.3f} (null {row['rnull_mod_a']:.3f}/"
              f"{row['rnull_mod_n']:.3f}) -> with chosen {row['r_both_a']:.3f}/{row['r_both_n']:.3f} "
              f"(null {row['rnull_both_a']:.3f}/{row['rnull_both_n']:.3f})", flush=True)
    if not only:
        pd.DataFrame(rows).to_csv(f'{PS}/attractor_recruit_speed.csv', index=False)
        print('wrote attractor_recruit_speed.csv')
