"""Where in the theta cycle does each frequency peak, and does it move with state?

gamma_theta_phase.py showed that slow and fast gamma reach their amplitude peaks
at different points in the theta cycle: +94 and +48 degrees, a within-session
difference of +45 degrees (R = 0.87, Rayleigh p = 5e-15 over 44 sessions). That
test used two bands fixed in advance. This computes the same thing as a
CONTINUOUS function of frequency, which is the form that can show whether the
separation is a property of two bands or of a gradient running through them --
and whether the picture differs between anchoring states.

For each MEC channel group the theta phase comes from a 6-10 Hz Hilbert, and the
amplitude envelope of every frequency from 20 to 120 Hz from its own narrow
band. Mean envelope is binned by theta phase and each frequency row is
normalised to sum to one, so the map shows WHERE in the cycle a frequency peaks
and not how much power it carries -- otherwise the 1/f background would dominate
every row and the panel would show the spectrum rather than the coupling.

FILTERING IS DONE IN THE FREQUENCY DOMAIN, once. A filtfilt-plus-Hilbert per
band costs about forty bands' worth of full-length transforms per channel; here
the signal is Fourier transformed once and each band is an inverse transform of
the positive-frequency components inside it, which gives the analytic signal
directly. Lengths are padded to a fast FFT size -- a session is ~1.8M samples
and that length factors badly enough to dominate the runtime otherwise.

Running samples only, trials speed-matched between states as elsewhere.

Writes data/lfp/theta_phase_comodulogram.npz
"""
import os
import sys
import time
import warnings

import h5py
import numpy as np
import pandas as pd

warnings.filterwarnings('ignore')
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, '/Users/harryclark/Documents/spatial-manifolds/src')
from scipy.fft import fft, ifft, next_fast_len
from scipy.signal import butter, filtfilt, hilbert

from lfp_batch import (FS, LFP_ROOT, clip_trials, mec_groups, nap,
                       speed_matched, vr_paths)

ROOT = '/Users/harryclark/Documents/spatial-manifolds'
OUT = f'{ROOT}/data/lfp/theta_phase_comodulogram.npz'

THETA = (6., 10.)
FREQS = np.arange(20., 121., 2.5)      # band centres
BW = 6.0                               # half-bandwidth, Hz
N_PHASE = 24
MAX_GROUPS = 3                         # spread over depth; see module docstring
MIN_SAMP = int(30 * FS)


def analytic_bands(x, centres, bw):
    """Analytic signal for every band, from ONE forward transform."""
    n = len(x)
    nf = next_fast_len(n)
    X = fft(x, nf)
    freqs = np.fft.fftfreq(nf, 1 / FS)
    out = np.empty((len(centres), n))
    for i, f0 in enumerate(centres):
        Y = np.zeros(nf, complex)
        m = (freqs >= f0 - bw) & (freqs <= f0 + bw)
        Y[m] = 2.0 * X[m]              # keep positive frequencies only, doubled
        out[i] = np.abs(ifft(Y)[:n])
    return out


def run_session(mo, dy, rng):
    lf = f'{LFP_ROOT}/M{mo}/D{dy}/VR/sub-M{mo}_ses-D{dy}_typ-VR_beh.nwb'
    ana = f'{LFP_ROOT}/M{mo}/D{dy}/anatomy_M{mo}_D{dy}.csv'
    if not (os.path.exists(lf) and os.path.exists(ana)):
        return None
    fh = h5py.File(lf, 'r')
    if 'processing/ecephys/LFP/lfp/data' not in fh:
        fh.close(); return None
    names = [n.decode() for n in
             fh['general/extracellular_ephys/electrodes/channel_name'][:]]
    M = mec_groups(mo, dy, names)
    if len(M) < 2:
        fh.close(); return None
    take = M.iloc[np.linspace(0, len(M) - 1, min(MAX_GROUPS, len(M))).astype(int)]
    cols = np.sort(take.g.values)

    bpth, cp = vr_paths(mo, dy)
    beh = nap.load_file(bpth); clusters = nap.load_file(cp)
    trials, _ = clip_trials(beh['trials'].as_dataframe(), clusters)
    T = pd.read_csv(f'{ROOT}/data/population_state/anchoring_trials.csv')
    st = T[(T.mouse == mo) & (T.day == dy)]
    if not len(st):
        fh.close(); return None
    state = dict(zip(st.trial.astype(int), st.frac_anch > 0.5))

    d = fh['processing/ecephys/LFP/lfp/data']
    nT = d.shape[0]
    blk = (d.chunks[0] if d.chunks else 26041) * 8
    X = np.empty((nT, len(cols)), np.float32)
    for i in range(0, nT, blk):
        X[i:i + blk] = d[i:i + blk, :][:, cols]
    fh.close()

    t = np.arange(nT) / FS
    S = beh['S']
    spd = np.asarray(S.values)[np.searchsorted(np.asarray(S.index), t)
                               .clip(0, len(S) - 1)]
    run = spd >= 3.0

    info = []
    for _, tr in trials.iterrows():
        u = int(tr.number)
        if u not in state:
            continue
        m = (t >= tr.start) & (t <= tr.end) & run
        if m.sum() < FS:
            continue
        info.append((bool(state[u]), float(spd[m].mean()), m))
    if len(info) < 10:
        return None
    km = speed_matched(np.array([i[1] for i in info]),
                       np.array([i[0] for i in info]), rng)
    masks = {True: np.zeros(nT, bool), False: np.zeros(nT, bool)}
    for i in km:
        masks[info[i][0]] |= info[i][2]
    if min(masks[True].sum(), masks[False].sum()) < MIN_SAMP:
        return None

    edges = np.linspace(-np.pi, np.pi, N_PHASE + 1)
    acc = {True: np.zeros((len(FREQS), N_PHASE)),
           False: np.zeros((len(FREQS), N_PHASE))}
    for j in range(X.shape[1]):
        x = X[:, j].astype(float)
        b, a = butter(3, [THETA[0] / (FS / 2), THETA[1] / (FS / 2)], btype='band')
        n = len(x)
        ph = np.angle(hilbert(filtfilt(b, a, x), N=next_fast_len(n))[:n])
        env = analytic_bands(x, FREQS, BW)
        for s_ in (True, False):
            mm = masks[s_]
            idx = np.digitize(ph[mm], edges) - 1
            idx = np.clip(idx, 0, N_PHASE - 1)
            for k in range(N_PHASE):
                sel = idx == k
                if sel.any():
                    acc[s_][:, k] += env[:, mm][:, sel].mean(1)
    # normalise each frequency row so the map shows WHERE, not how much
    out = {}
    for s_, tag in ((True, 'anch'), (False, 'non')):
        A = acc[s_] / X.shape[1]
        out[tag] = A / A.sum(1, keepdims=True)
    return out


if __name__ == '__main__':
    T = pd.read_csv(f'{ROOT}/data/population_state/anchoring_trials.csv')
    sess = sorted(set(zip(T.mouse.astype(int), T.day.astype(int))))
    A, N, keys, t0 = [], [], [], time.time()
    for mo, dy in sess:
        rng = np.random.default_rng(abs(hash((mo, dy))) % 2 ** 32)
        try:
            r = run_session(mo, dy, rng)
        except Exception as e:
            print(f'  ! M{mo}D{dy}: {type(e).__name__}: {e}', flush=True); continue
        if r is None:
            continue
        A.append(r['anch']); N.append(r['non']); keys.append((mo, dy))
        print(f'  M{mo}D{dy}  ({time.time()-t0:.0f}s)', flush=True)
    if A:
        np.savez_compressed(OUT, anch=np.array(A), non=np.array(N),
                            freqs=FREQS, n_phase=N_PHASE,
                            sessions=np.array(keys))
        print(f'\nwrote {OUT}: {len(A)} sessions')
