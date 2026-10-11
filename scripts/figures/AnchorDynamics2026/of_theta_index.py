"""Open-field theta index per cell, computed rather than read off.

Usage:  python3 of_theta_index.py

WHY IT IS COMPUTED HERE. detect_grids.py reads a theta_index.parquet out of
each session's tuning_scores folder, but no such file exists in this dataset --
the reference is to something that was never generated. So the index is built
directly, in the standard form, and from the OPEN FIELD rather than the virtual
track, for the same reason grid score and speed score are: a measure taken on
the session being explained cannot be used to explain it.

THE DEFINITION (Yartsev et al. 2011; Langston et al. 2010).

    1. spike-train autocorrelogram, 2 ms bins, lags out to 500 ms, zero lag
       excluded, computed on one side and mirrored
    2. mean subtracted, Hamming tapered, zero-padded to 2^16
    3. power spectrum of that
    4. theta index = peak power in 5-11 Hz / mean power from 0 to 50 Hz

A ratio of a peak to a broadband mean, so it is dimensionless and insensitive
to firing rate, which matters here because the comparison runs across cell
classes whose rates differ several-fold.

NO SHUFFLE NULL, AND THE SPLIT IS NOT A THRESHOLD. A jitter null costs two
orders of magnitude more compute than the index itself, and the question this
is for -- do theta-modulated cells follow the population axis more -- is better
asked of the index as a CONTINUOUS variable than of a binary split at an
imported cut-off. The binary version is reported too, split at the antimode of
the observed distribution the way the waveform classification is, and only as a
second reading.

Both OF1 and OF2 are scored where present and reported separately, since
agreement between them is the honest reliability estimate for a per-cell score.

Writes data/population_state/of_theta_index.csv
"""
import os
import sys
import time
import warnings

import numpy as np
import pandas as pd

warnings.filterwarnings('ignore')
sys.path.insert(0, '/Users/harryclark/Documents/spatial-manifolds/src')
import pynapple as nap
from spatial_manifolds.data.curation import curate_clusters

ROOT = '/Users/harryclark/Documents/spatial-manifolds'
SRC = '/Users/harryclark/Downloads/clark2025'
OUT = f'{ROOT}/data/population_state/of_theta_index.csv'
ACG_BIN = 0.002         # s
ACG_MAX = 0.500         # s
NFFT = 1 << 16
THETA_LO, THETA_HI = 5.0, 11.0
BROAD_HI = 50.0
MIN_SPIKES = 200


def acg(t, binw=ACG_BIN, tmax=ACG_MAX):
    """One-sided autocorrelogram by pairwise differences, zero lag excluded.

    Binning the whole session and correlating would be a few FFTs of two
    million points per cell; the pairs within half a second are far fewer than
    that for every cell here.
    """
    t = np.sort(np.asarray(t, float))
    hi = np.searchsorted(t, t + tmax, side='right')
    out = np.zeros(int(round(tmax / binw)))
    for i in range(len(t)):
        d = t[i + 1:hi[i]] - t[i]
        if d.size:
            k = (d / binw).astype(int)
            np.add.at(out, k[k < len(out)], 1)
    return out


def theta_index(t):
    a = acg(t)
    if a.sum() < 50:
        return np.nan, np.nan
    a = a - a.mean()
    w = a * np.hamming(len(a))
    P = np.abs(np.fft.rfft(w, NFFT)) ** 2
    f = np.fft.rfftfreq(NFFT, ACG_BIN)
    th = (f >= THETA_LO) & (f <= THETA_HI)
    broad = (f > 0) & (f <= BROAD_HI)
    peak = float(P[th].max())
    base = float(P[broad].mean())
    return (peak / base if base > 0 else np.nan,
            float(f[th][np.argmax(P[th])]))


def session(mo, dy, of):
    cp = (f'{SRC}/M{mo}/D{dy}/{of}/sub-M{mo}_ses-D{dy}_typ-{of}'
          '_srt-kilosort4_clusters.npz')
    if not os.path.exists(cp):
        return []
    cur = curate_clusters(nap.load_file(cp))
    rows = []
    for c in [int(x) for x in cur.index]:
        t = np.asarray(cur[c].index, dtype=float)
        if len(t) < MIN_SPIKES:
            continue
        ti, fpk = theta_index(t)
        rows.append(dict(mouse=mo, day=dy, cluster_id=c, of=of,
                         n_spikes=len(t), theta_index=ti, theta_peak_hz=fpk))
    return rows


if __name__ == '__main__':
    import glob
    # parse from the path components, not by splitting on '/D' -- the source
    # root is .../Downloads/..., which contains one
    sess = sorted({(int(_p.split(os.sep)[-3][1:]),
                    int(_p.split(os.sep)[-2][1:]))
                   for _p in glob.glob(f'{SRC}/M*/D*/OF*')})
    out, t0 = [], time.time()
    for k, (mo, dy) in enumerate(sess, 1):
        for of in ('OF1', 'OF2'):
            r = session(mo, dy, of)
            out += r
            if r:
                print(f'[{k}/{len(sess)}] M{mo}D{dy} {of}: {len(r)} cells '
                      f'({time.time() - t0:.0f}s)', flush=True)
    D = pd.DataFrame(out)
    D.to_csv(OUT, index=False)
    print(f'\nwrote {OUT}: {len(D)} rows')
    if len(D):
        print(D.groupby('of').theta_index.describe()[['count', '50%', 'max']])
