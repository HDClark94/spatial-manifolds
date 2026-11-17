"""The bump's speed against the animal's, averaged over 2.5 s.

attractor_coherence.bump_speed correlates the two speeds over consecutive
100 ms bins. At that scale the bump's speed is distorted by the readout
itself: the bump's phase is the rate-weighted circular mean of the cells'
offsets, and a few dozen cells cover the torus unevenly, so the estimate lags
while the bump passes over a cluster of cells and jumps across the gaps
between them. A clean bump moved at a constant 30 cm/s past the 36 offsets of
the simulated module (attractor_sim_session.py) reads out at -5 to 125 cm/s;
with all 1,296 units of the sheet its step varies by 0.4%. The distortion
repeats once per lattice cycle and cancels over one, since the estimate must
advance exactly one cycle in that time.

So here both speeds are averaged over a 2.5 s window before they are
compared: a centred boxcar of 25 100 ms bins, within running stretches (at
least 80% of the window running) and within one state (the whole window in
trials of that state). 2.5 s is one lattice cycle of the simulated module,
over which its distortion cancels (r 0.38 -> 0.82 anchored, 0.41 -> 0.89 non-anchored, in
the simulation without stopping). In the recordings one
cycle along the track is far longer -- 5 to 90 s at the median running speed
(cycle_s), often longer than a trial, since the modules advance at most about
1.4 cycles per 200 cm -- so a full cycle cannot be used there; 2.5 s reduces
the distortion without removing it, and is used for every session alike. Otherwise
as bump_speed: the two axes that advance along the track, converted to cm/s
by their phase advance per cm over anchored trials and averaged; Pearson r
with the animal's speed per state; the same for 20 phase shuffles of the
module's offsets; r with the two states' speed distributions matched (equal
draws per speed decile, 20 draws); and the slope, bump cm per cm run. The
100 ms values are kept beside them for comparison.

Reads data/population_state/attractor_cache/
Writes data/population_state/attractor_speed_cycle.csv (r_* at 2.5 s, r100_* at 100 ms)
"""
import glob
import os

import numpy as np
import pandas as pd

ROOT = '/Users/harryclark/Documents/spatial-manifolds'
PS = f'{ROOT}/data/population_state'
TL, DT, NPERM, NDRAW, MINFRAC = 200.0, 0.1, 20, 20, 0.8


def axes_and_gain(P_mod, phi_all, st, nb):
    """The two advancing axes and their phase advance per cm (anchored trials)."""
    psi = np.stack([np.angle(np.exp(1j * q) @ P_mod) for q in phi_all]).reshape(len(phi_all), -1, nb)
    adv = [abs(np.median([np.unwrap(r)[-1] - np.unwrap(r)[0] for r in w])) for w in psi]
    kk = list(np.argsort(adv)[::-1][:2])
    sa = np.zeros(psi.shape[1], bool)
    sa[:min(len(st), len(sa))] = st[:len(sa)]
    dpdx = np.array([np.median(np.diff(np.unwrap(psi[k][sa], axis=1), axis=1)) / (TL / nb)
                     for k in kk])
    return kk, dpdx


def bump_speed(R_mod, ph, dpdx):
    """cm/s per 100 ms bin (NaN in the first), from the offsets ph (2 x cells):
    each axis's phase step converted by its advance per cm, the two averaged,
    as attractor_coherence.bump_speed has it. (Combining the axes by least
    squares instead was tried: it weights the faster axis, whose distortion in
    the simulated module falls at the same track positions on every anchored
    trial and so tracks the animal's own speed profile; anchored r fell from
    0.82 to 0.33 there.)"""
    psi = np.stack([np.angle(np.exp(1j * q) @ R_mod) for q in ph])
    d = (np.diff(psi, axis=1) + np.pi) % (2 * np.pi) - np.pi
    return np.r_[np.nan, (d / dpdx[:, None]).mean(0) / DT]


WIN = 25                                 # 100 ms bins: 2.5 s


def cycle_bins(dpdx, spd, ok):
    """One cycle of the faster advancing axis, in 100 ms bins at median speed."""
    return max(1, int(round(2 * np.pi / np.abs(dpdx).max() / np.median(spd[ok]) / DT)))


def boxcar(x, valid, W):
    """Centred mean over W bins of the valid samples; NaN where fewer than MINFRAC."""
    xv = np.where(valid, x, 0.0)
    k = np.ones(W)
    s = np.convolve(xv, k, 'same')
    c = np.convolve(valid.astype(float), k, 'same')
    out = np.where(c >= MINFRAC * W, s / np.maximum(c, 1), np.nan)
    return out


def within(mask, W):
    """Bins whose whole window lies in mask."""
    return np.convolve(mask.astype(float), np.ones(W), 'same') >= W - 1e-9


def session(z, rng):
    cells = list(z['cells'])
    mi = [cells.index(c) for c in z['module']]
    nb = int(z['nbin'])
    R = z['R'][mi].astype(float)
    P = z['P'][mi].astype(float)
    st = np.asarray(z['state']).astype(bool)
    ok, anch, non = (z[k].astype(bool) for k in ('ok', 'anch', 'non'))
    spd = z['spd'].astype(float)
    kk, dpdx = axes_and_gain(P, z['phi_all'], st, nb)
    if np.any(np.abs(dpdx) < 1e-6):
        return None
    W = WIN
    bs = bump_speed(R, z['phi_all'][kk], dpdx)
    val = ok & np.r_[False, ok[:-1]] & np.isfinite(bs)
    out = dict(W=W, cycle_s=cycle_bins(dpdx, spd, ok) * DT)
    sA, sV = boxcar(bs, val, W), boxcar(spd, val, W)
    # the state of each bin's trial, running or not: the window must lie in one
    # state; running within it is the boxcar's own 80% rule
    idx = z['idx']
    sa_t = (idx >= 0) & st[np.clip(idx, 0, len(st) - 1)]
    sn_t = (idx >= 0) & ~st[np.clip(idx, 0, len(st) - 1)]
    ma = within(sa_t, W) & ok & np.isfinite(sA)
    mn = within(sn_t, W) & ok & np.isfinite(sA)
    r = lambda m, b, v: float(np.corrcoef(b[m], v[m])[0, 1])
    out['r_a'], out['r_n'] = r(ma, sA, sV), r(mn, sA, sV)
    out['slope_a'] = float(np.polyfit(sV[ma], sA[ma], 1)[0])
    out['slope_n'] = float(np.polyfit(sV[mn], sA[mn], 1)[0])
    out['n_a'], out['n_n'] = int(ma.sum()), int(mn.sum())
    # the 100 ms values, as bump_speed has them
    m1a, m1n = val & anch & np.r_[False, anch[:-1]], val & non & np.r_[False, non[:-1]]
    out['r100_a'], out['r100_n'] = r(m1a, bs, spd), r(m1n, bs, spd)
    out['slope100_a'] = float(np.polyfit(spd[m1a], bs[m1a], 1)[0])
    out['slope100_n'] = float(np.polyfit(spd[m1n], bs[m1n], 1)[0])
    nul = []
    for _ in range(NPERM):
        b0 = boxcar(bump_speed(R, np.stack([rng.permutation(p) for p in z['phi_all'][kk]]), dpdx),
                    val, W)
        g0a, g0n = ma & np.isfinite(b0), mn & np.isfinite(b0)
        nul.append((r(g0a, b0, sV), r(g0n, b0, sV)))
    out['rnull_a'], out['rnull_n'] = (float(x) for x in np.mean(nul, 0))
    # speed distributions matched between the states
    edge = np.quantile(sV[ma | mn], np.linspace(0, 1, 11))
    edge[0], edge[-1] = -np.inf, np.inf
    kb = np.digitize(sV, edge[1:-1])
    rr = []
    for _ in range(NDRAW):
        ia, iN = [], []
        for k in range(10):
            xa, xn = np.where(ma & (kb == k))[0], np.where(mn & (kb == k))[0]
            take = min(len(xa), len(xn))
            if take:
                ia.append(rng.choice(xa, take, replace=False))
                iN.append(rng.choice(xn, take, replace=False))
        ia, iN = np.concatenate(ia), np.concatenate(iN)
        rr.append((np.corrcoef(sA[ia], sV[ia])[0, 1], np.corrcoef(sA[iN], sV[iN])[0, 1]))
    out['rm_a'], out['rm_n'] = (float(x) for x in np.mean(rr, 0))
    out['spd_sd_a'], out['spd_sd_n'] = float(sV[ma].std()), float(sV[mn].std())
    return out


if __name__ == '__main__':
    rng = np.random.default_rng(0)
    rows = []
    for f in sorted(glob.glob(f'{PS}/attractor_cache/M*D*.npz')):
        z = np.load(f)
        o = session(z, rng)
        if o is None:
            continue
        o.update(mouse=int(z['mouse']), day=int(z['day']), n_grid=len(z['module']))
        rows.append(o)
        print(f"  M{o['mouse']}D{o['day']}: cycle {o['cycle_s']:.1f} s; r anchored {o['r_a']:+.3f} "
              f"(100 ms {o['r100_a']:+.3f}, null {o['rnull_a']:+.3f}), non-anchored {o['r_n']:+.3f} "
              f"(100 ms {o['r100_n']:+.3f}, null {o['rnull_n']:+.3f})", flush=True)
    sim = f'{PS}/attractor_sim/sim.npz'
    if os.path.exists(sim):
        o = session(np.load(sim), rng)
        print(f"  simulated module: cycle {o['cycle_s']:.1f} s; r anchored {o['r_a']:+.3f} "
              f"(100 ms {o['r100_a']:+.3f}), non-anchored {o['r_n']:+.3f} (100 ms {o['r100_n']:+.3f})")
    pd.DataFrame(rows).to_csv(f'{PS}/attractor_speed_cycle.csv', index=False)
    print('wrote attractor_speed_cycle.csv')
