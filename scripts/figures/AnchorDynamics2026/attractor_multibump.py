"""Does the non-anchored bump split into more than one bump?

Figure S14F draws the module's activity on the unit cell of the sheet, frame by
frame, and the non-anchored frames often show two or more blobs where the
anchored ones show one. Three readings predict different things:

  NOISE      a weaker single bump. Each frame is scaled to its own range for
             display, so when the bump is weak the kernel-smoothed noise of a
             few dozen cells is stretched to the same contrast and draws extra
             blobs. Predicts: the second peak is no higher than in the same
             frames with the cells' phases shuffled (which keeps every rate),
             and the two states do not differ at matched bump strength.
  SPLIT      the module carries two bumps at once (for instance part of it
             still held at the anchored phase). Predicts: a second peak above
             the shuffle, higher in the non-anchored state at matched
             strength, and, if one bump is the anchored one, the second peak
             nearer the phase position predicts than chance.
  FAST       one bump moving fast enough to be in two places within a 100 ms
             bin, which smears it rather than splitting it.

Per session (the Figure S14 sessions), per running 100 ms bin of each state:
the module's activity on the sheet as Figure S14F draws it (kernel sigma 0.62
rad, the pair of lattice axes whose phases are least locked in time), scaled
0-1; the peaks are the local maxima (3 x 3 neighbourhood, wrapped), and
P2 is the height of the highest peak at least pi/2 (sheet distance) from the
top one. The same on the same frames with the module's phases permuted among
its cells (NPERM times). The bump strength is the population vector above
shuffle of Figure S14 (pv - sh), averaged over the two axes.

Reads data/population_state/attractor_cache/
Writes data/population_state/attractor_multibump.csv (per session and state)
       data/population_state/attractor_multibump_frames.npz (per frame, pooled)
"""
import glob
import os

import numpy as np
import pandas as pd
from scipy.ndimage import maximum_filter

ROOT = '/Users/harryclark/Documents/spatial-manifolds'
PS = f'{ROOT}/data/population_state'
NG, SIG, NPERM, NPB, TL = 30, .62, 5, 40, 200.0
MINSEP = np.pi / 2
LAGS = [-10, -6, -4, -3, -2, -1, 1, 2, 3, 4, 6, 10]   # 100 ms bins
rng = np.random.default_rng(0)

_g = np.linspace(-np.pi, np.pi, NG, endpoint=False)
GU, GV = np.meshgrid(_g, _g, indexing='ij')


def wrap(x):
    return (x + np.pi) % (2 * np.pi) - np.pi


def sheet_dist(d1, d2):
    """Distance on the rhombic unit cell between phase differences (d1, d2),
    the shortest over the neighbouring copies of the cell."""
    best = np.full(np.shape(d1), np.inf)
    for a in (-2 * np.pi, 0, 2 * np.pi):
        for b in (-2 * np.pi, 0, 2 * np.pi):
            x, y = wrap(d1) + a, wrap(d2) + b
            best = np.minimum(best, np.hypot(x - y / 2, np.sqrt(3) / 2 * y))
    return best


def kernel(ph):
    d1 = wrap(ph[0][:, None, None] - GU[None])
    d2 = wrap(ph[1][:, None, None] - GV[None])
    K = np.exp(-(d1 ** 2 + d2 ** 2) / (2 * SIG ** 2))
    return (K / K.sum(0)).reshape(len(ph[0]), -1)


def peaks(I):
    """I: frames x NG*NG -> (p2, location of the top peak, of the second)."""
    F = I.reshape(-1, NG, NG)
    F = (F - F.min((1, 2), keepdims=True)) / (np.ptp(F, axis=(1, 2), keepdims=True) + 1e-12)
    loc = (F == maximum_filter(F, size=(1, 3, 3), mode='wrap'))
    p2 = np.zeros(len(F)); l1 = np.zeros((len(F), 2)); l2 = np.full((len(F), 2), np.nan)
    for t in range(len(F)):
        iu, iv = np.where(loc[t])
        h = F[t, iu, iv]
        o = np.argsort(-h)
        iu, iv, h = iu[o], iv[o], h[o]
        l1[t] = _g[iu[0]], _g[iv[0]]
        far = sheet_dist(_g[iu] - _g[iu[0]], _g[iv] - _g[iv[0]]) >= MINSEP
        if far.any():
            k = np.argmax(far)
            p2[t] = h[k]
            l2[t] = _g[iu[k]], _g[iv[k]]
    return p2, l1, l2


def coherence(R, ph):
    n = np.linalg.norm(R, axis=0) + 1e-9
    return np.mean([np.abs(np.exp(1j * q) @ R) for q in ph], 0) / (np.sqrt(R.shape[0] / 2) * n)


def run(f):
    z = np.load(f)
    mo, dy = int(z['mouse']), int(z['day'])
    cells = list(z['cells'])
    mi = [cells.index(c) for c in z['module']]
    R = z['R'][mi].astype(float)
    pa = z['phi_all']
    ok = z['ok'].astype(bool)
    anch, non = z['anch'].astype(bool), z['non'].astype(bool)
    psi = np.stack([np.angle(np.exp(1j * q) @ R) for q in pa])
    pair = min(((0, 1), (0, 2), (1, 2)),
               key=lambda p: abs(np.mean(np.exp(1j * (psi[p[0]][ok] - psi[p[1]][ok])))))
    ph = pa[list(pair)]
    # where position puts the bump: the module's mean phase per 5 cm on anchored bins
    pb = np.clip((z['trackpos'] / TL * NPB).astype(int), 0, NPB - 1)
    pred = np.zeros((2, NPB))
    for b in range(NPB):
        m = anch & (pb == b)
        pred[:, b] = np.angle(np.exp(1j * psi[list(pair)][:, m]).mean(1)) if m.any() else np.nan
    coh = coherence(R, ph)
    sh = np.mean([coherence(R, np.stack([rng.permutation(q) for q in ph])) for _ in range(NPERM)], 0)
    # every running bin at once, so that lags and steps can cross state boundaries
    # only where both bins are in the same state
    idx = np.where(ok)[0]
    p2, l1, l2 = peaks(R[:, idx].T @ kernel(ph))
    nul = [peaks(R[:, idx].T @ kernel(np.stack([rng.permutation(q) for q in ph])))
           for _ in range(NPERM)]
    p2n = np.mean([n_[0] for n_ in nul], 0)
    pos_of = {t: i for i, t in enumerate(idx)}
    spd = z['spd'][idx].astype(float)
    u = rng.uniform(-np.pi, np.pi, (2, 20000))
    dch = np.nanmean(sheet_dist(u[0], u[1]))
    rows, fr = [], []
    for nm, msk in (('anchored', anch), ('non-anchored', non)):
        sel = msk[idx]
        pp = pred[:, pb[idx[sel]]]
        d1 = sheet_dist(l1[sel, 0] - pp[0], l1[sel, 1] - pp[1])
        d2 = sheet_dist(l2[sel, 0] - pp[0], l2[sel, 1] - pp[1])
        st = (coh - sh)[idx[sel]]
        row = dict(mouse=mo, day=dy, n_grid=len(mi), state=nm, n_frames=int(sel.sum()),
                   p2=p2[sel].mean(), p2_null=p2n[sel].mean(), two=(p2[sel] > .8).mean(),
                   two_null=(p2n[sel] > .8).mean(), bump=st.mean(),
                   d_top=np.nanmean(d1), d_second=np.nanmean(d2), d_chance=dch,
                   r_two_speed=np.corrcoef(p2[sel] > .8, spd[sel])[0, 1],
                   pair=f'{pair[0]}{pair[1]}')
        # where is the second peak? against where the top peak is LAG bins away
        for k in LAGS:
            t0 = idx[sel]
            t1 = t0 + k
            good = np.array([(t in pos_of) and msk[t] for t in t1])
            if not good.any():
                row[f'lag{k}'] = np.nan
                continue
            i0 = np.where(sel)[0][good]
            i1 = np.array([pos_of[t] for t in t1[good]])
            v = sheet_dist(l2[i0, 0] - l1[i1, 0], l2[i0, 1] - l1[i1, 1])
            row[f'lag{k}'] = np.nanmean(v)
            row[f'lagself{k}'] = np.nanmean(sheet_dist(l1[i0, 0] - l1[i1, 0], l1[i0, 1] - l1[i1, 1]))
        # how often the top peak jumps between consecutive frames, against the shuffle
        i0 = np.where(sel)[0]
        cont = np.array([(t + 1 in pos_of) and msk[t + 1] for t in idx[sel]])
        i0 = i0[cont]
        i1 = np.array([pos_of[t + 1] for t in idx[i0]])
        stp = sheet_dist(l1[i0, 0] - l1[i1, 0], l1[i0, 1] - l1[i1, 1])
        row['jump'] = (stp > MINSEP).mean()
        row['jump_null'] = np.mean([(sheet_dist(n_[1][i0, 0] - n_[1][i1, 0],
                                                n_[1][i0, 1] - n_[1][i1, 1]) > MINSEP).mean()
                                    for n_ in nul])
        rows.append(row)
        fr.append(pd.DataFrame(dict(mouse=mo, day=dy, state=nm, p2=p2[sel], p2_null=p2n[sel],
                                    bump=st, spd=spd[sel], d_top=d1, d_second=d2)))
    return rows, pd.concat(fr)


if __name__ == '__main__':
    out, frames = [], []
    for f in sorted(glob.glob(f'{PS}/attractor_cache/M*D*.npz')):
        r, fr = run(f)
        out += r
        frames.append(fr)
        a, n = r
        print(f"  M{a['mouse']}D{a['day']} ({a['n_grid']} grid): second peak anchored "
              f"{a['p2']:.3f} (null {a['p2_null']:.3f}), non-anchored {n['p2']:.3f} (null "
              f"{n['p2_null']:.3f}); bump {a['bump']:.3f}/{n['bump']:.3f}; distance to the "
              f"position-predicted phase, non-anchored: top {n['d_top']:.2f}, second "
              f"{n['d_second']:.2f}, chance {n['d_chance']:.2f}", flush=True)
    pd.DataFrame(out).to_csv(f'{PS}/attractor_multibump.csv', index=False)
    F = pd.concat(frames)
    np.savez_compressed(f'{PS}/attractor_multibump_frames.npz',
                        **{k: F[k].values for k in F.columns if k != 'state'},
                        state=(F.state == 'anchored').values)
    print('wrote attractor_multibump.csv')
