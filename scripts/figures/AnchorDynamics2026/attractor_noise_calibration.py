"""How noisy recorded grid cells are, so the simulated module can be made as noisy.

For every grid cell in the module of each Figure 9 session, from its raw spike
counts in 100 ms running bins:
  rate   its mean firing rate (Hz)
  fano   the variance of its count about the count expected at that place and
         in that state (its mean in each 5 cm bin per state), over that expected
         count -- 1 for a Poisson cell whose rate is set by place and state alone
and, per session, the mean pairwise correlation of those residuals between the
module's cells (shared noise).

Reads the attractor cache and the raw VR spike files (attractor_coherence.vr_paths)
Writes data/population_state/attractor_sim/recorded_noise.csv
"""
import glob
import os
import sys

import numpy as np
import pandas as pd
import pynapple as nap

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import attractor_coherence as AC                                  # noqa: E402

rows = []
for f in sorted(glob.glob(f'{AC.PS}/attractor_cache/M*D*.npz')):
    z = np.load(f)
    mo, dy = int(z['mouse']), int(z['day'])
    mod = [int(c) for c in z['module']]
    _, cp = AC.vr_paths(mo, dy)
    clu = nap.load_file(cp)
    T = len(z['ok'])
    ep = nap.IntervalSet(start=0, end=T * 0.1)
    C = np.stack([np.asarray(clu[c].count(bin_size=100, time_units='ms', ep=ep))[:T]
                  for c in mod]).astype(float)
    ok, idx = z['ok'].astype(bool), z['idx']
    st = np.asarray(z['state']).astype(bool)
    lab = np.where(idx >= 0, st[np.clip(idx, 0, len(st) - 1)], False)
    pb = np.clip((z['trackpos'] / 200 * 40).astype(int), 0, 39)
    E = np.zeros_like(C)
    for s_ in (True, False):
        for b in range(40):
            m = ok & (lab == s_) & (pb == b)
            if m.any():
                E[:, m] = C[:, m].mean(1, keepdims=True)
    res, mu = (C - E)[:, ok], E[:, ok]
    fano = (res ** 2).sum(1) / np.maximum(mu.sum(1), 1e-9)
    rr = np.corrcoef(res)
    np.fill_diagonal(rr, np.nan)
    for c, r_, f_ in zip(mod, C[:, ok].mean(1) / 0.1, fano):
        rows.append(dict(mouse=mo, day=dy, cluster_id=c, rate=r_, fano=f_,
                         shared_r=float(np.nanmean(rr))))
    print(f'  M{mo}D{dy}: {len(mod)} cells, rate median {np.median(C[:, ok].mean(1) / 0.1):.1f} Hz, '
          f'Fano median {np.median(fano):.2f}, shared r {np.nanmean(rr):+.3f}', flush=True)
os.makedirs(f'{AC.PS}/attractor_sim', exist_ok=True)
D = pd.DataFrame(rows)
D.to_csv(f'{AC.PS}/attractor_sim/recorded_noise.csv', index=False)
print(f'wrote recorded_noise.csv: {len(D)} grid cells; rate median {D.rate.median():.1f} Hz, '
      f'Fano median {D.fano.median():.2f}, shared r median {D.groupby(["mouse", "day"]).shared_r.first().median():+.3f}')
