"""Cell label sequences aligned to population anchoring-state transitions.

Extracted from the former transitions supplement, whose figure is now panel F
of Figure 2. Kept as a module rather than inlined so the computation has one
definition.

Each cell is timed against a population state recomputed WITHOUT IT, the same
leave-one-out correction per_cell_pc1.py applies to the agreement scores.

Note what this can and cannot answer. "Do cells lead the population?" has no
content: the population state is the fraction of cells anchored thresholded at
a half, so it crosses when half of them have switched and roughly half must
lead whatever the biology (51% measured, and the number moves with smoothing --
51% unfiltered, 55% at a 5-trial filter, 56% at 7). The answerable question is
whether IDENTITIES differ in timing, and they do not: likelihood ratio between
mixed models with and without an identity term gives p = 0.15, with the four
means spanning 0.36 trials against a one-trial resolution.
"""
import os
import sys
import warnings

import numpy as np
import pandas as pd

warnings.filterwarnings('ignore')
import matplotlib

matplotlib.use('Agg')
import matplotlib.pyplot as plt
import statsmodels.formula.api as smf
from scipy.ndimage import median_filter
from scipy.stats import kruskal, wilcoxon

sys.path.insert(0, '/Users/harryclark/Documents/spatial-manifolds/src')
from spatial_manifolds.anchoring import (ANCH_COLOR, NONANCH_COLOR,
                                         load_session_labels)

plt.rcParams['font.family'] = 'Arial'
ROOT = '/Users/harryclark/Documents/spatial-manifolds'
FIG = f'{ROOT}/scripts/figures/AnchorDynamics2026'
PS = f'{ROOT}/data/population_state'
ORDER = ['grid', 'non-grid spatial', 'non-spatial', 'putative interneuron']
SHORT = ['grid', 'NGS', 'NS', 'int']
ICOL = {'grid': '#c04744', 'non-grid spatial': '#3171ae',
        'non-spatial': '#888888', 'putative interneuron': '#d95f02'}
WIN, MIN_TRANS = 6, 2
# No extra smoothing of the population state. The classifier already median
# filters each cell's labels, so filtering the average of them a second time
# delays the population's apparent switch and inflates the lead: 51% of cells
# lead at FILT=1, 55% at 5, 56% at 7.
FILT = 1

U = pd.read_csv(f'{PS}/unit_table.csv')
IDENT = {(m, d, c): i for m, d, c, i in
         zip(U.mouse, U.day, U.cluster_id, U.identity)}


def collect(loo=True):
    rows = []
    for (mo, dy), _ in U[U.sess_switches & U.in_population].groupby(['mouse', 'day']):
        z = load_session_labels(int(mo), int(dy))
        L, ids = z['labels'], z['cluster_id'].astype(int)
        in_pop = np.isfinite(z['pc1_load'])
        P = L[in_pop]
        for i, c in enumerate(ids):
            lab = L[i]
            if not np.isfinite(lab).any() or np.nanstd(lab) == 0:
                continue
            if loo and in_pop[i]:
                # the population state WITHOUT this cell
                others = np.ones(P.shape[0], bool)
                j = int(np.where(np.where(in_pop)[0] == i)[0][0]) \
                    if in_pop[i] else -1
                if j >= 0:
                    others[j] = False
                frac = np.nanmean(P[others], axis=0)
            else:
                frac = np.asarray(z['frac_anch'], float)
            st = median_filter((frac > .5).astype(float), size=FILT,
                               mode='nearest') > .5
            tr = [t for t in np.where(np.diff(st.astype(int)) != 0)[0] + 1
                  if t - WIN >= 0 and t + WIN <= len(st)]
            segs = []
            for t in tr:
                seg = lab[t - WIN:t + WIN]
                if np.isfinite(seg).all():
                    segs.append(seg if st[t] else 1 - seg)
            if len(segs) < MIN_TRANS:
                continue
            m = np.nanmean(segs, axis=0)
            lag = float(np.argmax(np.diff(m)) - (WIN - 1))
            rows.append(dict(mouse=mo, day=dy, cluster_id=int(c),
                             identity=IDENT.get((mo, dy, int(c))), lag=lag,
                             n_trans=len(segs), profile=m,
                             step=float(m[WIN:].mean() - m[:WIN].mean())))
    T = pd.DataFrame(rows)
    return T[T.identity.isin(ORDER)]


