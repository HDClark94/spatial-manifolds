"""Do the cells the search selects carry the bump beyond what position gives?

attractor_population_search.py selects, per session and fold, the non-module
cells whose addition makes the held-out bump more coherent. On a track any
place-tuned cell reports the bump's phase whenever the bump is tied to
position, so agreement between the bump read from those cells and the grid
module's bump is not by itself evidence that they carry the attractor. Three
comparisons separate the two, on every held-out fold:

  RAW       agreement (mean cos of the phase difference, both axes) between the
            bump read from the selected cells alone and the module's bump.
  POSITION  the same for the bump predicted from position alone: the module's
            mean phase at each 5 cm of track in each state, from the training
            folds. What position can give, per state.
  RESIDUAL  every cell's position map (per state, 5 cm bins, training folds)
            subtracted from its rate before either bump is read. Agreement that
            survives is a shared coordinate that position does not carry.

Each against the same quantity for two disjoint random halves of the grid
module (20 splits), the reference for how well grid cells agree with each
other, and against the selected cells with random phases (chance). Because the
selected cells are compared with the whole module but a grid half only with
the other half, the matched comparison is against half B in both cases:
agreement(selected, B) beside agreement(A, B). And because the selected cells'
phases are fitted to the bump on the training folds while the module's come
from the position domain, the control for fitting is the same number of
UNSELECTED non-module cells given phases the same way, against B.

Reads the cache and attractor_search_cells{,_null}.csv
Writes data/population_state/attractor_search_eval{,_null}.csv
"""
import glob
import os

import numpy as np
import pandas as pd

ROOT = '/Users/harryclark/Documents/spatial-manifolds'
PS = f'{ROOT}/data/population_state'
NULL = os.environ.get('NULL') == '1'
NFOLD, BLOCK, MINSHIFT = 5, 8, 600
NPB, TL = 40, 200.0
NSPLIT, NRND = 20, 20


def bump(R, phi):
    return np.stack([np.angle(np.exp(1j * phi[j]) @ R) for j in range(2)])


def time_phases(R, psi):
    return np.stack([np.angle(R @ np.exp(1j * psi[j])) for j in range(2)])


def agree(x, y, a, n):
    c = np.cos(x - y).mean(0)
    return c[a].mean() if a.any() else np.nan, c[n].mean() if n.any() else np.nan


def pos_pred(R_tr, pb_tr, st_tr, pb_te, st_te):
    """Each cell's mean rate per 5 cm bin and state (training), at test bins."""
    out = np.zeros((R_tr.shape[0], len(pb_te)))
    for s in (True, False):
        M = np.zeros((R_tr.shape[0], NPB))
        for b in range(NPB):
            m = (pb_tr == b) & (st_tr == s)
            M[:, b] = R_tr[:, m].mean(1) if m.any() else 0.
        k = st_te == s
        out[:, k] = M[:, pb_te[k]]
    return out


def psi_from_pos(psi_tr, pb_tr, st_tr, pb_te, st_te):
    """The module's circular-mean phase per 5 cm bin and state (training)."""
    out = np.zeros((2, len(pb_te)))
    for s in (True, False):
        for b in range(NPB):
            m = (pb_tr == b) & (st_tr == s)
            k = (pb_te == b) & (st_te == s)
            if k.any():
                out[:, k] = (np.angle(np.exp(1j * psi_tr[:, m]).mean(1))[:, None]
                             if m.any() else 0.)
    return out


def run(f, CS, rng):
    z = np.load(f)
    mo, dy = int(z['mouse']), int(z['day'])
    cells = list(z['cells'])
    mod_i = np.array([cells.index(c) for c in z['module']])
    phi = z['phi']
    R = z['R'].astype(float)
    if NULL:                     # the same shifts the null search used
        r_ = np.random.default_rng(7)
        # the null search drew its shifts from one stream across sessions, so
        # they cannot be replayed here; any independent shift is equivalent
        for c in range(len(cells)):
            if c not in set(mod_i):
                R[c] = np.roll(R[c], r_.integers(MINSHIFT, R.shape[1] - MINSHIFT))
    ok = z['ok'].astype(bool)
    R, idx, anch = R[:, ok], z['idx'][ok], z['anch'][ok].astype(bool)
    pb = np.clip((z['trackpos'][ok] / TL * NPB).astype(int), 0, NPB - 1)
    fold = (idx // BLOCK) % NFOLD
    cs = CS[(CS.mouse == mo) & (CS.day == dy)]
    rows = []
    for f_ in range(NFOLD):
        sel_ids = cs[(cs.fold == f_) & cs.selected].cluster_id.values
        sel = np.array([cells.index(c) for c in sel_ids if c in cells])
        if len(sel) < 2:
            continue
        tr, te = fold != f_, fold == f_
        a_, n_ = anch[te], ~anch[te]
        psi_tr = bump(R[mod_i][:, tr], phi)
        ph = time_phases(R[:, tr], psi_tr)
        P = pos_pred(R[:, tr], pb[tr], anch[tr], pb[te], anch[te])
        Rte, Rres = R[:, te], R[:, te] - P
        g_raw, g_res = bump(Rte[mod_i], phi), bump(Rres[mod_i], phi)
        s_raw, s_res = bump(Rte[sel], ph[:, sel]), bump(Rres[sel], ph[:, sel])
        res = dict(mouse=mo, day=dy, fold=f_, n_sel=len(sel), n_mod=len(mod_i))
        res['sel_raw_a'], res['sel_raw_n'] = agree(s_raw, g_raw, a_, n_)
        res['sel_res_a'], res['sel_res_n'] = agree(s_res, g_res, a_, n_)
        res['pos_a'], res['pos_n'] = agree(psi_from_pos(psi_tr, pb[tr], anch[tr], pb[te], anch[te]),
                                           g_raw, a_, n_)
        ch = [agree(bump(Rres[sel], rng.uniform(-np.pi, np.pi, (2, len(sel)))), g_res, a_, n_)
              for _ in range(NRND)]
        res['chance_res_a'], res['chance_res_n'] = np.mean(ch, 0)
        unsel = np.setdiff1d(np.setdiff1d(np.arange(len(cells)), mod_i), sel)
        hr, hs, sb, sbr, ub = [], [], [], [], []
        for _ in range(NSPLIT):
            p_ = rng.permutation(len(mod_i)); h = len(mod_i) // 2
            A, B = mod_i[p_[:h]], mod_i[p_[h:2 * h]]
            ia, ib = p_[:h], p_[h:2 * h]
            gB, gBr = bump(Rte[B], phi[:, ib]), bump(Rres[B], phi[:, ib])
            hr.append(agree(bump(Rte[A], phi[:, ia]), gB, a_, n_))
            hs.append(agree(bump(Rres[A], phi[:, ia]), gBr, a_, n_))
            # the selected cells' phases fitted without half B's cells
            ph_A = time_phases(R[:, tr], bump(R[A][:, tr], phi[:, ia]))
            sb.append(agree(bump(Rte[sel], ph_A[:, sel]), gB, a_, n_))
            sbr.append(agree(bump(Rres[sel], ph_A[:, sel]), gBr, a_, n_))
            if len(unsel) >= 2:
                u = rng.choice(unsel, min(len(sel), len(unsel)), replace=False)
                ub.append(agree(bump(Rres[u], ph_A[:, u]), gBr, a_, n_))
        res['half_raw_a'], res['half_raw_n'] = np.nanmean(hr, 0)
        res['half_res_a'], res['half_res_n'] = np.nanmean(hs, 0)
        res['selB_raw_a'], res['selB_raw_n'] = np.nanmean(sb, 0)
        res['selB_res_a'], res['selB_res_n'] = np.nanmean(sbr, 0)
        res['unselB_res_a'], res['unselB_res_n'] = np.nanmean(ub, 0) if ub else (np.nan, np.nan)
        res['n_unsel'] = len(unsel)
        res['half_k'] = len(mod_i) // 2
        rows.append(res)
    return rows


if __name__ == '__main__':
    sfx = '_null' if NULL else ''
    CS = pd.read_csv(f'{PS}/attractor_search_cells{sfx}.csv')
    rng = np.random.default_rng(3)
    out = []
    for f in sorted(glob.glob(f'{PS}/attractor_cache/M*D*.npz')):
        r = run(f, CS, rng)
        out += r
        if r:
            m = pd.DataFrame(r).mean(numeric_only=True)
            print(f'  {os.path.basename(f)[:-4]}: {m.n_sel:.0f} selected; agreement with the module, '
                  f'anchored/non: raw {m.sel_raw_a:.3f}/{m.sel_raw_n:.3f}, position {m.pos_a:.3f}/'
                  f'{m.pos_n:.3f}, position removed {m.sel_res_a:.3f}/{m.sel_res_n:.3f} (chance '
                  f'{m.chance_res_a:.3f}); grid halves raw {m.half_raw_a:.3f}/{m.half_raw_n:.3f}, '
                  f'removed {m.half_res_a:.3f}/{m.half_res_n:.3f}; vs half B, removed: selected '
                  f'{m.selB_res_a:.3f}/{m.selB_res_n:.3f}, unselected {m.unselB_res_a:.3f}/'
                  f'{m.unselB_res_n:.3f}', flush=True)
    pd.DataFrame(out).to_csv(f'{PS}/attractor_search_eval{sfx}.csv', index=False)
    print('wrote', f'attractor_search_eval{sfx}.csv')
