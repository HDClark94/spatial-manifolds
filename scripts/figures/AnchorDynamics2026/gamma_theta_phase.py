"""At which theta phase does each gamma band reach its amplitude peak?

Figure 8B draws slow and fast gamma nested at DIFFERENT points in the theta
cycle -- slow on one flank, fast on the other. That is a specific empirical
claim and nothing in this collection had tested it: lfp_spectrum_pac.py stores
the modulation index, which says HOW STRONGLY a band is nested, and discards the
phase, which says WHERE. The schematic was drawn from the two-loop account
rather than from the data, which is the wrong way round.

This measures it. For each MEC channel group, theta phase comes from a 6-10 Hz
Hilbert transform and each gamma band's amplitude envelope from its own
band-pass; the preferred phase is the circular mean of theta phase weighted by
that envelope. Running samples only, as everywhere else.

Two things are reported. The preferred phase of each band, pooled across
sessions; and the within-session DIFFERENCE between them, which is the quantity
the figure asserts is non-zero. A paired circular test on that difference is the
test -- comparing two pooled means would not be, because preferred phase varies
between sessions with electrode position and reference.

Writes data/lfp/gamma_theta_phase.csv
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
from scipy.fft import next_fast_len
from scipy.signal import butter, filtfilt, hilbert

from lfp_batch import FS, LFP_ROOT, clip_trials, mec_groups, nap, vr_paths

ROOT = '/Users/harryclark/Documents/spatial-manifolds'
OUT = f'{ROOT}/data/lfp/gamma_theta_phase.csv'
THETA = (6., 10.)
BANDS = [('slow_gamma', 30., 48.), ('fast_gamma', 60., 100.)]
MIN_SAMP = int(30 * FS)


def bp(x, lo, hi):
    b, a = butter(3, [lo / (FS / 2), hi / (FS / 2)], btype='band')
    return filtfilt(b, a, x)


def analytic(x):
    """Hilbert transform at a fast FFT length.

    scipy.signal.hilbert FFTs the signal at its own length. A session is ~1.8M
    samples and that length factors badly, which makes the transform pathologic-
    ally slow -- slow enough that an earlier version of this script processed no
    sessions at all in 18 minutes. Padding to the next fast length and trimming
    back costs nothing and is what makes this runnable.
    """
    n = len(x)
    return hilbert(x, N=next_fast_len(n))[:n]


def circ_mean(ang, w):
    z = np.sum(w * np.exp(1j * ang)) / np.sum(w)
    return np.angle(z), np.abs(z)


def run_session(mo, dy):
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
    bpth, cp = vr_paths(mo, dy)
    beh = nap.load_file(bpth); clusters = nap.load_file(cp)
    clip_trials(beh['trials'].as_dataframe(), clusters)

    d = fh['processing/ecephys/LFP/lfp/data']
    nT = d.shape[0]
    cols = np.sort(M.g.values)
    # full-width chunk reads: column fancy-indexing on this gzip layout
    # decompresses every chunk once per column (see cross_region_gamma.py)
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
    if run.sum() < MIN_SAMP:
        return None

    rows = []
    for j in range(X.shape[1]):
        x = X[:, j].astype(float)
        ph = np.angle(analytic(bp(x, *THETA)))[run]
        for nm, lo, hi in BANDS:
            amp = np.abs(analytic(bp(x, lo, hi)))[run]
            mu, rlen = circ_mean(ph, amp)
            rows.append(dict(mouse=mo, day=dy, g=int(cols[j]), band=nm,
                             pref_phase=float(mu), concentration=float(rlen),
                             n_samp=int(run.sum())))
    return pd.DataFrame(rows)


if __name__ == '__main__':
    T = pd.read_csv(f'{ROOT}/data/population_state/anchoring_trials.csv')
    sess = sorted(set(zip(T.mouse.astype(int), T.day.astype(int))))
    out, t0 = [], time.time()
    for mo, dy in sess:
        try:
            r = run_session(mo, dy)
        except Exception as e:
            print(f'  ! M{mo}D{dy}: {type(e).__name__}: {e}', flush=True); continue
        if r is None:
            continue
        out.append(r)
        print(f'  M{mo}D{dy}: {r.g.nunique()} groups ({time.time()-t0:.0f}s)',
              flush=True)
    if out:
        pd.concat(out, ignore_index=True).to_csv(OUT, index=False)
        print(f'\nwrote {OUT}')
