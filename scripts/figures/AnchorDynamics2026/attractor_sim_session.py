"""A simulated session for Figure S15: 36 grid cells of one gated attractor module.

The two-dimensional gated attractor (can_gate_model_2d.py) is run on the
virtual track for about 150 trials, with a simulated mouse that slows and
stops where its own bump, decoded against the landmarks' phase along the
track, says the reward zone (90-110 cm) is (behave=True), with the attentional gate opened and shut
in blocks of trials, so the gate alone decides when the population is
anchored. 36 units are sampled at random from the 36 x 36 sheet, as a
recording samples a module, and turned into spikes with a recorded grid cell's
mean rate and over-Poisson variability; path integration carries noise too.
Everything after that is the recordings' own pipeline, so the simulated
example is drawn and measured exactly as the recorded ones are:

  rates     100 ms counts, Gaussian-smoothed (sigma 2.5 bins) and z-scored,
            as attractor_coherence.session_maps builds them
  maps      each trial's rate in 2 cm bins of the track, unrolled, smoothed
            and z-scored the same way
  phases    attractor_coherence.torus_offsets (the spectral method) on those
            maps -- the model's own sheet coordinates are not used
  state     the anchoring classifier on each unit's trial-by-position map and
            PC1 of the unit-by-trial labels (can_gate_model.population_state),
            positive PC1 taken as anchored, as for the recordings
  open field  the same 36 units while the model forages in a 150 cm arena
            with the gate open, as rate maps

Writes data/population_state/attractor_sim/sim.npz in the attractor cache's
format, plus labels, PC1, the gate schedule and the open-field maps.
"""
import os
import sys
import time

import numpy as np
import pandas as pd
from scipy.ndimage import gaussian_filter1d

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
import attractor_coherence as AC                                  # noqa: E402
import can_gate_model as CGM                                      # noqa: E402
import can_gate_model_2d as M2                                    # noqa: E402

OUT = f'{AC.PS}/attractor_sim'
NCELL, PEAK, SEED = 36, 20.0, 0
# NOISE, made as biological as the recordings say (attractor_noise_calibration.py):
#  - path integration is noisy: a random walk on the sheet alongside the velocity
#    shift (can_gate_model_2d.PI_DIFF, rad per 0.5 ms step; 0.005 is about 0.6 rad
#    per trial when the gate is shut, and about 0.08 rad held by the gate when open)
#  - each sampled unit takes the mean rate and the over-Poisson variability (Fano
#    factor about the place-and-state expectation) of a recorded grid cell drawn
#    at random, and its 100 ms counts are gamma-Poisson with that Fano factor
#  - shared noise is left out: recorded modules have almost none (median r 0.017)
PI_NOISE = 0.005
BIN = 0.1                                   # s, the 100 ms bins
# the gate, trial by trial: anchored, lost, anchored again, lost again
BLOCKS = [(1, 40), (0, 30), (1, 40), (0, 45)]
SCHED = np.concatenate([np.full(n, M2.G_OPEN * g) for g, n in BLOCKS])
NLAP = 175                                # nominal; stops make each trial longer


def z(v):
    v = np.asarray(v, float)
    sd = v.std()
    return (v - v.mean()) / sd if sd > 1e-9 else None


if __name__ == '__main__':
    os.makedirs(OUT, exist_ok=True)
    rng = np.random.default_rng(SEED)
    keep = np.sort(rng.choice(M2.NS * M2.NS, NCELL, replace=False))
    t0 = time.time()
    print(f'track, {NLAP} trials, gate blocks {BLOCKS} ...', flush=True)
    M2.PI_DIFF = PI_NOISE
    R = M2.run(M2.G_OPEN, seed=SEED, schedule=SCHED, nlap=NLAP, keep=keep, behave=True)
    print(f'  {int(R["lap"].max()) + 1} trials in {time.time() - t0:.0f} s', flush=True)
    print('open field (gate open, foraging) ...', flush=True)
    W = M2.run(M2.G_OPEN, seed=SEED + 1, walk=True, dur=240.0, keep=keep)
    OF = M2.ratemaps_2d(W, nbin=40)

    # spikes in 100 ms bins: each unit's expected count follows its model rate,
    # scaled to a recorded grid cell's mean rate; counts are gamma-Poisson with
    # that cell's Fano factor
    dt_s = M2.DT * 4                                       # recording interval
    per = int(round(BIN / dt_s))
    nb_t = len(R['rE']) // per
    mu_m = np.clip(R['rE'][:nb_t * per], 0, None).reshape(nb_t, per, NCELL).sum(1).T
    CAL = pd.read_csv(f'{OUT}/recorded_noise.csv')
    pick = CAL.iloc[rng.choice(len(CAL), NCELL, replace=False)]
    rate_c, fano_c = pick.rate.values, pick.fano.values
    mu = mu_m / mu_m.mean(1, keepdims=True) * (rate_c * BIN)[:, None]
    k = np.where(fano_c > 1.02, rate_c * BIN / np.maximum(fano_c - 1, 1e-3), np.inf)
    g = np.ones_like(mu)
    for i in range(NCELL):
        if np.isfinite(k[i]):
            g[i] = rng.gamma(k[i], 1 / k[i], nb_t)
    cnt = rng.poisson(mu * g)
    sl = slice(per // 2, nb_t * per, per)
    lap_t = R['lap'][sl][:nb_t]
    pos_t = (R['x'][sl] % M2.TRACK)[:nb_t]
    spd_t = R['spd'][:nb_t * per].reshape(nb_t, per).mean(1)
    gate_t = R['gate'][sl][:nb_t]
    ntr = int(lap_t.max()) + 1
    # each trial's map: spikes / occupancy in 2 cm bins, unrolled and smoothed
    pb = np.clip((pos_t / M2.TRACK * AC.NBIN).astype(int), 0, AC.NBIN - 1)
    occ = np.zeros(ntr * AC.NBIN)
    np.add.at(occ, lap_t * AC.NBIN + pb, BIN)
    tim, pos = {}, {}
    cells = list(range(1, NCELL + 1))                     # cluster ids
    for i, c in enumerate(cells):
        acc = np.zeros(ntr * AC.NBIN)
        np.add.at(acc, lap_t * AC.NBIN + pb, cnt[i])
        tc = np.where(occ > 0, acc / np.maximum(occ, 1e-9), 0.0)
        a = z(AC.gaussian_filter(tc, sigma=AC.SIGMA))
        b = z(AC.gaussian_filter(cnt[i].astype(float), sigma=AC.SIGMA))
        if a is None or b is None:
            continue
        pos[c], tim[c] = a, b
    have = [c for c in cells if c in pos]
    phi, mod, phi_all = AC.torus_offsets(pos, have)
    print(f'  module found by the spectral method: {0 if mod is None else len(mod)} of '
          f'{len(have)} units', flush=True)

    # the population state, by the recordings' classifier
    PS = CGM.population_state(dict(lap=lap_t, x=pos_t,
                                   rE=np.stack([tim[c] for c in have], 1)))
    st = PS['pc1'] > 0
    gate_tr = np.array([gate_t[lap_t == L].mean() > 0 for L in range(ntr)])
    print(f'  PC1 positive on {st[gate_tr].mean():.0%} of gate-open trials and '
          f'{st[~gate_tr].mean():.0%} of gate-shut trials', flush=True)
    ok = spd_t > AC.MOVE
    # the simulated mouse's stops: where on the track each trial's stop began,
    # and whether it was in the reward zone
    stp = R['stop'].astype(bool)
    onset = np.where(stp & ~np.r_[False, stp[:-1]])[0]
    stop_pos = np.full(ntr, np.nan)
    for o in onset:
        L = int(R['lap'][o])
        if L < ntr and np.isnan(stop_pos[L]):
            stop_pos[L] = R['x'][o] % M2.TRACK
    hit = (stop_pos >= M2.RZ_LO) & (stop_pos <= M2.RZ_HI)
    print(f'  stops in the reward zone: {hit[gate_tr].mean():.0%} of gate-open trials, '
          f'{hit[~gate_tr].mean():.0%} of gate-shut ones', flush=True)
    np.savez_compressed(
        f'{OUT}/sim.npz', mouse=0, day=0, cells=np.array(have),
        ident=np.array(['grid'] * len(have)), is_grid=np.ones(len(have), bool),
        module=np.array(mod), phi=phi, phi_all=phi_all,
        R=np.stack([tim[c] for c in have]).astype(np.float32),
        P=np.stack([pos[c] for c in have]).astype(np.float32),
        state=st, idx=lap_t, ok=ok, anch=ok & st[lap_t], non=ok & ~st[lap_t],
        spd=spd_t.astype(np.float32), trackpos=pos_t.astype(np.float32), nbin=AC.NBIN,
        labels=PS['labels'], pc1=PS['pc1'], pc1_load=PS['pc1_load'], gate=gate_tr,
        of_maps=OF, sheet_index=keep, stop_pos=stop_pos, hit=hit, rate=rate_c, fano=fano_c,
        pi_noise=PI_NOISE,
        rz=np.array([M2.RZ_LO, M2.RZ_HI]), sheet_th1=M2.TH1.ravel()[keep],
        sheet_th2=M2.TH2.ravel()[keep])
    print(f'wrote {OUT}/sim.npz ({time.time() - t0:.0f} s)', flush=True)
