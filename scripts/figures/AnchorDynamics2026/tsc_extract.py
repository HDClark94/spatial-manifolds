"""Per-theta-cycle time-frequency maps, for theta-nested spectral components.

The method is Lopes-dos-Santos et al. 2018 (Neuron), as reviewed by Esparza et
al. 2023: every theta cycle is described by its own frequency x theta-phase
amplitude map, and an unsupervised decomposition across cycles (tsc_analyse.py)
finds the recurring kinds of cycle -- theta-nested spectral components, tSCs.
Here the question is whether anchored and non-anchored trials are made of
different kinds of theta cycle, which band power averaged over a trial cannot
ask.

ONE MEC GROUP PER SESSION, the superficial (L2/3) group with the most theta
power, falling back to the MEC group with the most theta if no superficial
group exists. Maps change through the layers, so mixing layers across sessions
would make the decomposition find layer rather than cycle type; the chosen
layer is stored so it can be checked.

THETA is the 6-10 Hz Hilbert phase of the same group, as in the comodulogram.
Cycles run trough to trough (phase +-pi). A cycle is kept if it lasts 1/12 to
1/5 s (5-12 Hz) and every sample in it is running (>= 3 cm/s) inside a trial
with a state label.

TWO MAPS PER CYCLE, and this is the control the decomposition cannot do
without. Theta here is not sinusoidal, so its harmonics (16, 24, 32, 40, 48 Hz)
are phase-locked to theta by construction and land in exactly the slow-gamma
range; an unsupervised method will happily return them as a 'slow gamma'
component. RAW maps are computed on the LFP. INDUCED maps are computed after
subtracting each cycle's share of the session's theta-locked average waveform
(the mean waveform as a function of theta phase, scaled by the cycle's theta
amplitude). Harmonics vanish from the induced maps; genuine gamma, whose phase
is not locked to theta's, survives. A component present only in the raw maps is
waveform shape, not gamma.

Per cycle, amplitude is AVERAGED within NPH theta-phase bins for each of NF
log-spaced frequencies (complex Morlet, N_CYC cycles). Sums and counts are
stored, not normalised maps, so the normalisation is a choice made -- and
reportable -- at analysis time.

Usage:  python tsc_extract.py [worker n_workers]
Writes  data/lfp/tsc/M{mo}D{dy}.npz
"""
import os
import sys
import time
import warnings

import numpy as np
import pandas as pd

warnings.filterwarnings('ignore')
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, '/Users/harryclark/Documents/spatial-manifolds/src')

ROOT = '/Users/harryclark/Documents/spatial-manifolds'
OUT = f'{ROOT}/data/lfp/tsc'
FREQS = np.geomspace(20., 120., 30)
N_CYC = 6.0
NPH = 18
NPH_W = 64                 # phase resolution of the evoked waveform template
THETA = (6., 10.)
RUN_SPEED = 3.0
DUR = (1 / 12, 1 / 5)


# ---------------------------------------------------------------------------
# signal helpers (pure numpy, so the core can be tested on synthetic data)
# ---------------------------------------------------------------------------
def band_analytic(x, fs, lo, hi):
    n = len(x)
    X = np.fft.rfft(x - x.mean())
    f = np.fft.rfftfreq(n, 1 / fs)
    X[(f < lo) | (f > hi)] = 0
    Y = np.zeros(n, complex)
    Y[:len(X)] = X
    Y[1:len(X) - (1 if n % 2 == 0 else 0)] *= 2
    return np.fft.ifft(Y)


def morlet_amp(x, fs, f0, n_cyc=N_CYC):
    """|x * complex Morlet| at f0, via one multiplication in the frequency domain."""
    n = len(x)
    X = np.fft.fft(x)
    f = np.fft.fftfreq(n, 1 / fs)
    sf = f0 / n_cyc                                   # spectral SD
    H = np.exp(-0.5 * ((f - f0) / sf) ** 2)
    H[f < 0] = 0
    return np.abs(np.fft.ifft(X * 2 * H)).astype(np.float32)


def cycles(theta_phase, keep, fs, dur=DUR):
    """Trough-to-trough cycles whose every sample is in `keep`.

    Returns cycle id per sample (-1 outside kept cycles), phase in [0, 2pi) from
    the cycle's trough, and per-cycle (start, stop) sample indices.
    """
    u = np.unwrap(theta_phase)
    k = np.floor((u + np.pi) / (2 * np.pi)).astype(np.int64)   # trough at phase +-pi
    ph = (u + np.pi) - 2 * np.pi * k                            # 0..2pi from the trough
    edges = np.flatnonzero(np.diff(k)) + 1
    starts = np.r_[0, edges]; stops = np.r_[edges, len(k)]
    cid = np.full(len(k), -1, np.int64)
    se = []
    nkeep = np.cumsum(np.r_[0, ~keep])                          # bad samples before i
    for a, b in zip(starts[1:-1], stops[1:-1]):                 # drop the partial ends
        d = (b - a) / fs
        if not (dur[0] <= d <= dur[1]):
            continue
        if nkeep[b] - nkeep[a] > 0:                             # any non-kept sample
            continue
        cid[a:b] = len(se)
        se.append((a, b))
    return cid, ph, np.array(se, int).reshape(-1, 2)


def cycle_maps(x, fs, theta_phase, keep, freqs=FREQS, nph=NPH):
    """Raw and induced (theta-locked waveform removed) cycle maps.

    Returns dict with raw_sum, ind_sum (ncyc, F, nph), cnt (ncyc, nph), and the
    per-cycle theta amplitude, the evoked waveform and the cycle index arrays.
    """
    th = band_analytic(x, fs, *THETA)
    cid, ph, se = cycles(theta_phase, keep, fs)
    ncyc = len(se)
    out = dict(n_cycles=ncyc, cycle_se=se)
    if ncyc == 0:
        return out
    tamp = np.abs(th)
    camp = np.array([tamp[a:b].mean() for a, b in se])
    # theta-locked average waveform as a function of phase, per unit theta amplitude
    inc = cid >= 0
    wb = np.minimum((ph[inc] / (2 * np.pi) * NPH_W).astype(int), NPH_W - 1)
    scale = camp[cid[inc]] / camp.mean()
    num = np.bincount(wb, weights=(x[inc] - x[inc].mean()) / scale, minlength=NPH_W)
    den = np.bincount(wb, minlength=NPH_W)
    evoked = num / np.maximum(den, 1)
    # subtract each cycle's scaled share of it
    x_ind = x - x[inc].mean()
    pbin_w = np.minimum((ph / (2 * np.pi) * NPH_W).astype(int), NPH_W - 1)
    sc_all = np.zeros(len(x)); sc_all[inc] = scale
    x_ind = np.where(inc, x_ind - evoked[pbin_w] * sc_all, x_ind)
    # bin amplitude by (cycle, phase bin)
    pb = np.minimum((ph[inc] / (2 * np.pi) * nph).astype(int), nph - 1)
    flat = cid[inc] * nph + pb
    cnt = np.bincount(flat, minlength=ncyc * nph).reshape(ncyc, nph)
    raw = np.empty((ncyc, len(freqs), nph), np.float32)
    ind = np.empty_like(raw)
    for j, f0 in enumerate(freqs):
        a_r = morlet_amp(x, fs, f0)[inc]
        a_i = morlet_amp(x_ind, fs, f0)[inc]
        raw[:, j, :] = np.bincount(flat, weights=a_r, minlength=ncyc * nph).reshape(ncyc, nph)
        ind[:, j, :] = np.bincount(flat, weights=a_i, minlength=ncyc * nph).reshape(ncyc, nph)
    out.update(raw_sum=raw, ind_sum=ind, cnt=cnt, theta_amp=camp.astype(np.float32),
               evoked=evoked.astype(np.float32),
               dur=np.array([(b - a) / fs for a, b in se], np.float32))
    return out


# ---------------------------------------------------------------------------
# session driver (needs the LFP volume)
# ---------------------------------------------------------------------------
def run_session(mo, dy):
    import h5py
    from lfp_batch import FS, LFP_ROOT, clip_trials, mec_groups, nap, vr_paths
    lf = f'{LFP_ROOT}/M{mo}/D{dy}/VR/sub-M{mo}_ses-D{dy}_typ-VR_beh.nwb'
    ana = f'{LFP_ROOT}/M{mo}/D{dy}/anatomy_M{mo}_D{dy}.csv'
    if not (os.path.exists(lf) and os.path.exists(ana)):
        print(f'  ? M{mo}D{dy}: LFP not readable, SKIPPED', flush=True)
        return None
    f = h5py.File(lf, 'r')
    if 'processing/ecephys/LFP/lfp/data' not in f:
        f.close(); return None
    names = [n.decode() for n in f['general/extracellular_ephys/electrodes/channel_name'][:]]
    M = mec_groups(mo, dy, names)
    if not len(M):
        f.close(); return None
    d = f['processing/ecephys/LFP/lfp/data']
    nT = d.shape[0]
    # theta power of every MEC group, on a 5-minute stretch, to choose one
    seg = slice(0, min(nT, int(300 * FS)))
    tp = {}
    for g in M.g.values:
        y = d[seg, int(g)].astype(float)
        tp[int(g)] = float(np.mean(np.abs(band_analytic(y, FS, *THETA)) ** 2))
    sup = M[M.layer.astype(str).str.match(r'ENTm[23]')]
    pool = sup if len(sup) else M
    g_sel = int(max(pool.g.values, key=lambda g: tp[int(g)]))
    layer = str(M.set_index('g').layer[g_sel])
    x = d[:, g_sel].astype(np.float64)
    f.close()

    bp, cp = vr_paths(mo, dy)
    beh = nap.load_file(bp); clusters = nap.load_file(cp)
    trials, _ = clip_trials(beh['trials'].as_dataframe(), clusters)
    T = pd.read_csv(f'{ROOT}/data/population_state/anchoring_trials.csv')
    st = T[(T.mouse == mo) & (T.day == dy)]
    if not len(st):
        return None
    state = dict(zip(st.trial.astype(int), st.frac_anch > 0.5))
    t = np.arange(nT) / FS
    S = beh['S']
    spd = np.asarray(S.values)[np.searchsorted(np.asarray(S.index), t).clip(0, len(S) - 1)]
    trial_of = np.full(nT, -1, np.int64)
    for _, tr in trials.iterrows():
        u = int(tr.number)
        if u in state:
            a, b = np.searchsorted(t, [tr.start, tr.end])
            trial_of[a:b] = u
    keep = (spd >= RUN_SPEED) & (trial_of >= 0)
    th_phase = np.angle(band_analytic(x, FS, *THETA))
    out = cycle_maps(x, FS, th_phase, keep)
    if out['n_cycles'] == 0:
        return None
    se = out['cycle_se']
    ctr = trial_of[se[:, 0]]
    out.update(trial=ctr.astype(np.int32),
               anch=np.array([state[int(u)] for u in ctr], bool),
               speed=np.array([spd[a:b].mean() for a, b in se], np.float32),
               t0=(se[:, 0] / FS).astype(np.float32),
               freqs=FREQS, nph=NPH, group=g_sel, layer=layer,
               theta_power_by_group=np.array(list(tp.items())))
    return out


if __name__ == '__main__':
    from lfp_batch import LFP_ROOT
    if not os.path.isdir(LFP_ROOT):
        sys.exit(f'LFP volume not mounted ({LFP_ROOT}); nothing to do.')
    os.makedirs(OUT, exist_ok=True)
    w, nw = (int(sys.argv[1]), int(sys.argv[2])) if len(sys.argv) > 2 else (0, 1)
    T = pd.read_csv(f'{ROOT}/data/population_state/anchoring_trials.csv')
    sess = sorted(set(zip(T.mouse.astype(int), T.day.astype(int))))
    sess = [s for i, s in enumerate(sess) if i % nw == w]
    t0 = time.time()
    for mo, dy in sess:
        dest = f'{OUT}/M{mo}D{dy}.npz'
        if os.path.exists(dest):
            continue
        try:
            r = run_session(mo, dy)
        except Exception as e:
            print(f'  ! M{mo}D{dy}: {type(e).__name__}: {e}', flush=True); continue
        if r is None:
            continue
        tmp = dest + '.partial.npz'
        np.savez_compressed(tmp, **{k: v for k, v in r.items() if k != 'cycle_se'},
                            cycle_se=r['cycle_se'])
        os.replace(tmp, dest)
        print(f'  M{mo}D{dy}: {r["n_cycles"]} cycles, group {r["group"]} ({r["layer"]}), '
              f'{int(r["anch"].sum())} anchored  ({time.time() - t0:.0f}s)', flush=True)
