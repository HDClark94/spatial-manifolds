"""Does the grid attractor collapse in the non-anchored state, or only unpin?

THE DISTINCTION THAT MATTERS. Figure 9 and the simulation in Figure S13 say
the non-anchored state is an attractor that is still running and simply not
being corrected to the task frame. The alternative is that the attractor comes
apart -- the population stops holding a single coherent bump, and there is no
state left to be uncorrected. The two make opposite predictions about one
quantity, and nothing else in the paper measures it. The simulation puts a
number on its own side of the question: the bump's population vector is 0.700
with the gate open and 0.683 with it shut, essentially unchanged.

THE MEASURE. The module and each cell's phase offsets on the torus come from
spectral_analysis(), the same routine the toroidal figures use. It returns
three offsets per cell, but a bump on a 2D sheet has only two degrees of
freedom, so the two axes whose phases are most consistent across windows are
kept and the third, which is their beat, is dropped. The
attractor's state at one moment is then read off the population exactly as
FIGURE1_PLOT_TORUS_ANIMATION does it, by projecting the instantaneous rates
onto those offsets: the complex sum of each cell's rate times exp(i phi)
gives the bump's position on that axis, and its length relative to the
population's own norm gives how concentrated the activity is -- 1 if every
cell is firing in register with one bump, 0 if the activity bears no relation
to the lattice. Rates are the z-scored 100 ms spike counts, so this is a
moment-to-moment quantity, not a trial average.

WHY A PHASE SHUFFLE IS THE RIGHT NULL. The population vector grows with the
number of cells and with how uneven the firing rates happen to be, and firing
rates differ between the two states. Shuffling which phase offset belongs to
which cell leaves every rate in the session untouched and destroys only the
lattice organisation, so the difference between the real projection and the
shuffled one isolates the part of the bump that is the attractor's doing.

BLIND TO THE THING THAT IS NOT IN QUESTION. Where the bump sits is not asked
about. A bump that has come loose from the track still has a population
vector; one that has dissolved does not. So the measure separates "running
but uncorrected" from "not running", which is the distinction the model needs.

Writes data/population_state/attractor_coherence.csv
"""
import json
import os
import sys
import warnings

import numpy as np
import pandas as pd

warnings.filterwarnings('ignore')
sys.path.insert(0, '/Users/harryclark/Documents/spatial-manifolds/src')
from scipy.ndimage import gaussian_filter                         # noqa: E402
from scipy.stats import wilcoxon                                  # noqa: E402

from spatial_manifolds.toroidal import spectral_analysis          # noqa: E402

FIG = os.path.dirname(os.path.abspath(__file__))
ROOT = '/Users/harryclark/Documents/spatial-manifolds'
PS = f'{ROOT}/data/population_state'
OUT_CSV = f'{PS}/attractor_coherence.csv'
OUT_TRIAL = f'{PS}/attractor_bump_by_trial.csv'
OUT_PHASE = f'{PS}/attractor_phase_by_trial.csv'

NB = f'{FIG}/lick_raster_by_trial_type.ipynb'
_g = {}
for _c in json.load(open(NB))['cells']:
    if _c['cell_type'] != 'code':
        continue
    _s = ''.join(_c['source'])
    if _s.startswith('MOUSE, DAY') or 'INV = pd.DataFrame' in _s:
        continue
    exec(compile(_s, '<nb>', 'exec'), _g)
globals().update({k: v for k, v in _g.items() if not k.startswith('__')})

TIME_BS = 100          # ms, the bin the attractor state is read in
SIGMA = float(os.environ.get('SIGMA', 2.5))   # bins; the toroidal
#                        figures use 2.5, which at 100 ms bins is 250 ms
MIN_GRID = 8           # cells needed in the module
MIN_TRIAL = 20         # trials needed in each state
NPERM = 100            # phase shuffles
MOVE = 3.0             # cm/s, the task's own stop criterion


def session_maps(mo, dy, ids):
    """Unrolled position maps and 100 ms rate vectors, built as the toroidal
    figures build them: smoothed, z-scored, clipped to the ephys window."""
    bp, cp = vr_paths(mo, dy)                                      # noqa: F821
    beh = nap.load_file(bp)                                        # noqa: F821
    clusters = nap.load_file(cp)                                   # noqa: F821
    ta = beh['trials'].as_dataframe()
    _, orig = clip_trials(ta, clusters)                            # noqa: F821
    keep = np.isin(ta.number.values.astype(int), orig)
    have = [c for c in ids if c in set(int(x) for x in clusters.index)]
    if len(have) < MIN_GRID:
        return None
    tns = beh['trial_number']
    t0 = float(np.asarray(tns.values)[0])
    dt = beh['travel'] - (t0 - 1) * TL                             # noqa: F821
    nb = len(ta) * NBIN                                            # noqa: F821
    moving = beh['S'].threshold(MOVE, method='above').time_support
    # the busiest cell, not the first: the first may have no spikes at all
    busiest = clusters.index[int(np.nanargmax(clusters.firing_rate))]
    last = float(clusters[busiest].count(
        bin_size=TIME_BS, time_units='ms').index[-1])
    ep = nap.IntervalSet(start=0, end=last, time_units='s')        # noqa: F821
    spd = beh['S'].bin_average(bin_size=TIME_BS, time_units='ms', ep=ep)
    trav = beh['travel'].bin_average(
        bin_size=TIME_BS, time_units='ms', ep=ep) - (t0 - 1) * TL  # noqa: F821
    tnum = np.asarray(trav.values) // TL + t0                      # noqa: F821
    pos, tim = {}, {}
    def z(v):
        """z-score, or None if the cell has no variance to score."""
        v = np.nan_to_num(np.asarray(v)).astype(float)
        sd = v.std()
        return None if not np.isfinite(sd) or sd < 1e-9 else (v - v.mean()) / sd
    for c in have:
        tc = nap.compute_1d_tuning_curves(                         # noqa: F821
            nap.TsGroup([clusters[c]]), dt, nb_bins=nb,            # noqa: F821
            minmax=[0, len(ta) * TL], ep=moving)[0]                # noqa: F821
        ct = clusters[c].count(bin_size=TIME_BS, time_units='ms', ep=ep)
        # only the trials inside the ephys window: the rest are empty, and
        # a block of zeros puts spectral content where there is no recording
        tc = np.asarray(tc).reshape(len(ta), NBIN)[keep].ravel()    # noqa
        a = z(gaussian_filter(np.nan_to_num(tc).astype(float),
                              sigma=SIGMA))
        b = z(gaussian_filter(np.nan_to_num(np.asarray(ct)).astype(float),
                              sigma=SIGMA))
        if a is None or b is None:        # silent cell: nothing to project
            continue
        pos[c], tim[c] = a, b
    have = [c for c in have if c in pos]
    if len(have) < MIN_GRID:
        return None
    return dict(pos=pos, tim=tim, have=have, orig=orig, ntrial=len(ta),
                tnum=tnum, spd=np.asarray(spd.values),
                trackpos=np.asarray(trav.values) % TL)          # noqa: F821


def torus_offsets(pos, cells):
    """Each cell's phase offset on the module's two torus axes.

    spectral_analysis() averages each axis over the windows whose phase
    pattern agrees with some other window, and returns nothing for an axis
    where no window passes that test. Three axes are returned but only two are
    independent -- the third is the beat of the other two -- so the two whose
    phases are most consistent across windows are kept, as Gardner et al. do.
    Both tests run on the whole session, so neither can favour either state.
    """
    tcs = {i: pos[c].reshape(-1, NBIN) for i, c in enumerate(cells)}  # noqa
    out = spectral_analysis(tcs, TL, TL / NBIN)                    # noqa: F821
    phi_mod, idx_mod, cons = out[1], out[2], out[7]
    if not idx_mod:
        return None, None
    j = int(np.argmax([len(i) for i in idx_mod]))
    if len(idx_mod[j]) < MIN_GRID:
        return None, None
    phi = np.stack(phi_mod[j])
    ok = np.isfinite(phi).all(1)
    phi, c = phi[ok], np.asarray(cons[j])[ok]
    if not len(phi):
        return None, None
    # all the resolved axes are returned as well: the trajectory display in
    # the figure needs the two that advance along the track, which need not
    # be the two best resolved -- one lattice axis can lie nearly
    # perpendicular to the track and then barely moves.
    return (phi[np.argsort(-c)[:2]], [cells[i] for i in idx_mod[j]], phi)


def bump_pv(R, phi):
    """How concentrated the population is on the torus, moment by moment.

    The projection of the population onto a single bump at phase theta, at its
    best theta, divided by the population's own norm. One if every cell fires
    in register with one bump, zero if the pattern bears no relation to the
    lattice.
    """
    z = np.abs(np.exp(1j * phi) @ R)
    return z / (np.sqrt(len(phi) / 2.0) * np.linalg.norm(R, axis=0) + 1e-9)


def speed_matched(x, spd, a, n, rng, nbin=10, ndraw=40):
    """The same contrast with the two states' running speeds equated.

    Firing rates climb with speed and the mouse runs differently in the two
    states, so the samples are sorted into speed bins and the same number
    drawn from each state in every bin. The population vector is already
    normalised by the population's own norm, so a uniform change in rate
    cancels, but this removes any residual.
    """
    edge = np.quantile(spd[a | n], np.linspace(0, 1, nbin + 1))
    edge[0], edge[-1] = -np.inf, np.inf
    b = np.digitize(spd, edge[1:-1])
    out = []
    for _ in range(ndraw):
        ia, iN = [], []
        for k in range(nbin):
            ka = np.where(a & (b == k))[0]
            kn = np.where(n & (b == k))[0]
            take = min(len(ka), len(kn))
            if take:
                ia.append(rng.choice(ka, take, replace=False))
                iN.append(rng.choice(kn, take, replace=False))
        if not ia:
            return np.nan, np.nan
        out.append([x[np.concatenate(ia)].mean(),
                    x[np.concatenate(iN)].mean()])
    return tuple(float(v) for v in np.mean(out, 0))


def sheet(phi):
    """Cell positions on the 2D neural sheet, from two torus phases.

    The transformation of Gardner et al.: s1 = phi1 - phi2 / 2 and
    s2 = sqrt(3) phi2 / 2, which carries the two phases onto the rhombic unit
    cell of a hexagonal lattice. Applied to the population's firing-rate
    weighted mean phases it gives the bump's centre of mass on the sheet.
    """
    return phi[0] - phi[1] / 2.0, np.sqrt(3) / 2.0 * phi[1]


def bump_map(R, phi, mask, rng=None, nb=24, nperm=40):
    """The population on the sheet, centred on the bump, averaged over time.

    At each moment the bump's centre of mass is subtracted from every cell's
    phase, so the average is taken in the bump's own frame rather than the
    track's. A bump that is intact but merely moving still gives a peak here;
    one that has dissolved gives a flat sheet.

    Centring on an estimate the population itself provides will concentrate
    activity at the centre even when there is no bump, so the same phase
    shuffle used everywhere else is run through the identical steps and
    returned alongside. The map is accumulated on the two phases, which tile a
    square; the caller shears it onto the rhombic unit cell for display.
    """
    e = np.linspace(-np.pi, np.pi, nb + 1)

    def one(ph):
        psi = np.stack([np.angle(np.exp(1j * p) @ R) for p in ph])
        d = (ph[:, :, None] - psi[:, None, :] + np.pi) % (2 * np.pi) - np.pi
        u, v = d[0][:, mask].ravel(), d[1][:, mask].ravel()
        w = np.broadcast_to(R[:, mask], d[0][:, mask].shape).ravel()
        H = np.histogram2d(u, v, bins=[e, e], weights=w)[0]
        C = np.histogram2d(u, v, bins=[e, e])[0]
        return H / np.maximum(C, 1)

    obs = one(phi)
    if rng is None:
        return obs, None, e
    nul = np.mean([one(np.stack([rng.permutation(p) for p in phi]))
                   for _ in range(nperm)], 0)
    return obs, nul, e


def sheet_pixels(phi, r, nb=34, sig=0.62):
    """Population activity interpolated onto the unit cell of the sheet.

    Each cell sits at its pair of torus phases; a pixel takes the average of
    the cells near it, weighted by a Gaussian in toroidal distance. With a few
    dozen cells the quantile binning of Gardner et al. leaves most bins empty,
    so the activity is kernel-smoothed onto the cell instead. The caller
    shears the square of phases onto the rhombus for display.
    """
    g = np.linspace(-np.pi, np.pi, nb, endpoint=False)
    U, V = np.meshgrid(g, g, indexing='ij')
    d1 = (phi[0][:, None, None] - U[None] + np.pi) % (2 * np.pi) - np.pi
    d2 = (phi[1][:, None, None] - V[None] + np.pi) % (2 * np.pi) - np.pi
    W = np.exp(-(d1 ** 2 + d2 ** 2) / (2 * sig ** 2))
    return (W * r[:, None, None]).sum(0) / W.sum(0), g


def transition_strip(P, phi, state, lo=5, hi=15, span=9):
    """The bump at one place on the track, trial by trial, across a switch.

    Averaging a whole trial would smear the bump over the whole sheet, because
    the bump traverses the sheet as the animal runs. Fixing the track position
    instead asks the question anchoring is about: at the same place on the
    track, does the population return to the same place on the sheet? It
    should while the state holds, and stop doing so when it turns over.

    The switch chosen is the longest pair of adjacent opposite epochs.
    """
    b = np.flatnonzero(np.diff(state.astype(int)) != 0) + 1
    if not len(b):
        return None
    runs = np.diff(np.r_[0, b, len(state)])
    k = int(np.argmax(np.minimum(runs[:-1], runs[1:])))     # the cleanest one
    t0 = int(b[k])
    lo_t, hi_t = max(0, t0 - span), min(len(state), t0 + span)
    tr = np.arange(lo_t, hi_t)
    frames = []
    for t in tr:
        r = np.nan_to_num(P[:, t, lo:hi]).mean(1)
        frames.append(sheet_pixels(phi, r)[0])
    return (np.stack(frames), tr, state[tr], t0,
            sheet_pixels(phi, np.zeros(P.shape[0]))[1])


def main():
    U = pd.read_csv(f'{PS}/unit_table.csv')
    T = pd.read_csv(f'{PS}/anchoring_trials.csv')
    rng = np.random.default_rng(0)
    rows, tro, pho, only = [], [], [], os.environ.get('ONLY')
    for (mo, dy), g in U[U.in_population].groupby(['mouse', 'day']):
        if only and f'M{mo}D{dy}' != only:
            continue
        ids = [int(c) for c in g[g.identity == 'grid'].cluster_id]
        if len(ids) < MIN_GRID:
            continue
        t = T[(T.mouse == mo) & (T.day == dy)].sort_values('trial')
        if len(t) < 60:
            continue
        st = (t.pc1.values > 0)
        if st.mean() < .15 or st.mean() > .85:
            continue
        try:
            S = session_maps(mo, dy, ids)
        except Exception as e:
            print(f'  ! M{mo}D{dy}: {type(e).__name__}: {e}', flush=True)
            continue
        if S is None:
            continue
        phi, mod, phi_all = torus_offsets(S['pos'], S['have'])
        if mod is None:
            print(f'  - M{mo}D{dy}: no coherent torus axis', flush=True)
            continue
        # state of each 100 ms bin, through the kept-trial index
        pos_in = {o: i for i, o in enumerate(S['orig'])}
        idx = np.array([pos_in.get(int(x), -1) for x in S['tnum']])
        ok = (idx >= 0) & (idx < len(st)) & (S['spd'] > MOVE)
        lab = np.where(ok, np.where(idx.clip(0) < len(st),
                                    st[idx.clip(0, len(st) - 1)], False), False)
        a, n = ok & lab, ok & ~lab
        if a.sum() < 100 or n.sum() < 100:
            print(f'  - M{mo}D{dy}: too few samples', flush=True)
            continue
        R = np.stack([S['tim'][c] for c in mod])
        pv, sh = [], []
        for j in range(phi.shape[0]):
            pv.append(bump_pv(R, phi[j]))
            sh.append(np.stack([bump_pv(R, rng.permutation(phi[j]))
                                for _ in range(NPERM)]).mean(0))
        pv, sh = np.mean(pv, 0), np.mean(sh, 0)      # per 100 ms sample
        obs = np.array([pv[a].mean(), pv[n].mean()])
        nul = np.array([sh[a].mean(), sh[n].mean()])
        ea, en = obs[0] - nul[0], obs[1] - nul[1]
        wa, wn = speed_matched(pv - sh, S['spd'], a, n, rng)
        # the module's phase at one place on the track, trial by trial: the
        # quantity a drifting attractor moves and an anchored one does not
        Ppos = np.stack([S['pos'][c] for c in mod]).reshape(
            len(mod), -1, NBIN)                                # noqa: F821
        for j in range(phi.shape[0]):
            ps = np.angle(np.exp(1j * phi[j]) @ np.nan_to_num(
                Ppos.reshape(len(mod), -1))).reshape(-1, NBIN)
            z = np.angle(np.exp(1j * ps[:, 5:15]).mean(1))
            for t_ in range(len(z)):
                if t_ < len(st):
                    pho.append(dict(mouse=mo, day=dy, axis=j, trial=t_,
                                    phase=float(z[t_]), anch=bool(st[t_])))
        # the bump trial by trial, for its time course across a transition
        for t in np.unique(idx[ok]):
            m = ok & (idx == t)
            if m.sum() >= 10:
                tro.append(dict(mouse=mo, day=dy, trial=int(t),
                                bump=float((pv[m] - sh[m]).mean()),
                                anch=bool(st[int(t)]), n=int(m.sum())))
        if os.environ.get('EXAMPLE') == f'M{mo}D{dy}':
            Ba, Na, be = bump_map(R, phi, a, rng)
            Bn, Nn, _ = bump_map(R, phi, n, rng)
            P = np.stack([S['pos'][c] for c in mod])
            # The sheet displays use the two lattice axes that ADVANCE
            # along the track, which need not be the two best resolved: one
            # axis of a hexagonal lattice can lie nearly perpendicular to a
            # linear track, and then it carries no positional signal and only
            # adds noise to the sheet coordinates. The measure above keeps
            # the two best-resolved axes; the choice barely moves it.
            psi_all = np.stack([np.angle(np.exp(1j * q) @ P)
                                for q in phi_all])
            adv = [abs(np.median([np.unwrap(r)[-1] - np.unwrap(r)[0]
                                  for r in w.reshape(-1, NBIN)]))  # noqa
                   for w in psi_all]
            kk = list(np.argsort(adv)[::-1][:2])
            phi_disp, psi = phi_all[kk], psi_all[kk]
            ST = transition_strip(P.reshape(len(mod), -1, NBIN),  # noqa: F821
                                  phi_disp, st)
            ang = np.stack(sheet(psi))             # theta1, theta2
            np.savez(f'{PS}/attractor_example.npz',
                     mouse=mo, day=dy, nbin=NBIN, n_grid=len(mod),  # noqa
                     state=st, pv=pv, sh=sh, anch=a, non=n,
                     spd=S['spd'], phi=phi,
                     bump_a=Ba, bump_n=Bn, null_a=Na, null_n=Nn, bedge=be,
                     angles=ang, psi=psi, psi_all=psi_all,
                     strip=ST[0], strip_trial=ST[1], strip_state=ST[2],
                     strip_t0=ST[3], strip_grid=ST[4],
                     # the bump's phase in TIME, so its speed can be measured
                     psi_t=np.stack([np.angle(np.exp(1j * q) @ R)
                                     for q in phi_disp]),
                     spd_t=S['spd'], idx_t=idx, ok_t=ok,
                     pos_t=S['trackpos'])
            print(f'    wrote attractor_example.npz', flush=True)
        rows.append(dict(mouse=mo, day=dy, n_grid=len(mod),
                         n_gridall=len(S['have']),
                         n_axis=int(phi.shape[0]),
                         pv_anch=obs[0], pv_non=obs[1],
                         null_anch=nul[0], null_non=nul[1],
                         bump_anch=ea, bump_non=en,
                         sm_anch=wa, sm_non=wn,
                         n_samp_a=int(a.sum()), n_samp_n=int(n.sum())))
        print(f'  M{mo}D{dy}: {len(mod):2d}/{len(S["have"]):2d} grid in '
              f'module, {a.sum()/10:5.0f}/{n.sum()/10:5.0f} s | PV '
              f'{obs[0]:.3f}/{obs[1]:.3f}  shuffled {nul[0]:.3f}/{nul[1]:.3f}'
              f' | bump {ea:+.4f}/{en:+.4f} | speed-matched '
              f'{wa:+.4f}/{wn:+.4f}', flush=True)
    D = pd.DataFrame(rows)
    if only:
        print(D.to_string()); return
    sfx = '' if SIGMA == 2.5 else f'_s{SIGMA:g}'
    D.to_csv(OUT_CSV.replace('.csv', f'{sfx}.csv'), index=False)
    pd.DataFrame(tro).to_csv(OUT_TRIAL.replace('.csv', f'{sfx}.csv'),
                             index=False)
    pd.DataFrame(pho).to_csv(OUT_PHASE.replace('.csv', f'{sfx}.csv'),
                             index=False)
    print(f'\n{len(D)} sessions, {D.n_grid.sum()} grid cells in module')
    if len(D) > 5:
        def show(lab, v):
            v = np.asarray(v, float); v = v[np.isfinite(v)]
            print(f'  {lab:40s} {v.mean():+.4f}  p = '
                  f'{wilcoxon(v).pvalue:.3g}   n = {len(v)}')
        print(f'  population vector  anchored {D.pv_anch.mean():.3f}  '
              f'non-anchored {D.pv_non.mean():.3f}')
        print(f'  phase-shuffled     anchored {D.null_anch.mean():.3f}  '
              f'non-anchored {D.null_non.mean():.3f}')
        show('bump above shuffle, anchored', D.bump_anch)
        show('bump above shuffle, non-anchored', D.bump_non)
        show('anchored - non-anchored', D.bump_anch - D.bump_non)
        K = D.dropna(subset=['sm_anch'])
        print(f'  with running speed equated, {len(K)} sessions')
        show('speed-matched, anchored', K.sm_anch)
        show('speed-matched, non-anchored', K.sm_non)
        show('speed-matched, anchored - non-anchored', K.sm_anch - K.sm_non)
        d = (D.bump_anch - D.bump_non) / D.bump_anch
        print(f'  the bump keeps {100 * (1 - d.median()):.0f}% of its '
              f'strength in the non-anchored state (median)')
    print(f'wrote {OUT_CSV}')


if __name__ == '__main__':
    main()
