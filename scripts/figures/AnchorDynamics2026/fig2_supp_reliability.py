"""Figure 2 supplement — how reliably is a cell's agreement score measured?

Usage:  python3 fig2_supp_reliability.py

ONE QUESTION: if the session were run twice, would a cell get the same
agreement score?

Figure 2 compares agreement with the population axis across cell identities and
finds differences. Those comparisons are only interpretable if r is a property
of the CELL rather than of the particular trials it was measured on. An
unreliable r attenuates every difference toward zero, so the identity effects in
Figure 2 are lower bounds by however much r is unreliable -- and the amount has
to be measured, not assumed.

HOW. Each session's trials are split into odd and even, which keeps the two
halves matched for slow drift across the session in a way that a first/second
split does not. The population state and each cell's labels are taken from the
cached classification, so this measures the reliability of the AGREEMENT, not
of the classifier. r is computed within each half and the two correlated across
cells.

Reliability is reported by identity as well as overall: if one identity is
measured less reliably than another -- because it fires less, or is anchored on
fewer trials -- then a difference between them in Figure 2 could be a difference
in measurement rather than in biology.

SPEARMAN-BROWN. A split-half correlation estimates the reliability of HALF a
session. The full-session reliability is higher, and the standard correction
2r/(1+r) converts one into the other. Both are reported, because the corrected
number is the one that applies to Figure 2 and the raw one is what was measured.
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
from scipy.stats import pearsonr

sys.path.insert(0, '/Users/harryclark/Documents/spatial-manifolds/src')
from spatial_manifolds.anchoring import load_session_labels

plt.rcParams['font.family'] = 'Arial'


def _lp(ax, s, dx=-.16, dy=1.0):
    """Bold panel letter, drawn separately from the title and always 10 pt,
    matching every other figure."""
    ax.text(dx, dy, s, transform=ax.transAxes, fontsize=10, weight='bold',
            va='bottom', ha='right')
ROOT = '/Users/harryclark/Documents/spatial-manifolds'
FIG = f'{ROOT}/scripts/figures/AnchorDynamics2026'
PS = f'{ROOT}/data/population_state'
ORDER = ['grid', 'non-grid spatial', 'non-spatial', 'putative interneuron']
SHORT = ['grid', 'NGS', 'NS', 'int']
ICOL = {'grid': '#c04744', 'non-grid spatial': '#3171ae',
        'non-spatial': '#888888', 'putative interneuron': '#d95f02'}
MIN_HALF = 15

U = pd.read_csv(f'{PS}/unit_table.csv')
IDENT = {(m, d, c): i for m, d, c, i in
         zip(U.mouse, U.day, U.cluster_id, U.identity)}

rows = []
for (mo, dy), _ in U[U.in_population].groupby(['mouse', 'day']):
    z = load_session_labels(int(mo), int(dy))
    L, ids = z['labels'], z['cluster_id'].astype(int)
    in_pop = np.isfinite(z['pc1_load'])
    n_tr = L.shape[1]
    if n_tr < 2 * MIN_HALF:
        continue
    halves = (np.arange(n_tr) % 2 == 0, np.arange(n_tr) % 2 == 1)
    for i, c in enumerate(ids):
        lab = L[i]
        if np.nanstd(lab) == 0:
            continue
        rr = []
        for h in halves:
            # population state WITHOUT this cell, within this half
            others = in_pop.copy()
            others[i] = False
            frac = np.nanmean(L[others][:, h], axis=0)
            v = lab[h]
            ok = np.isfinite(v) & np.isfinite(frac)
            rr.append(np.corrcoef(v[ok], frac[ok])[0, 1]
                      if ok.sum() >= MIN_HALF and np.std(v[ok]) > 0
                      and np.std(frac[ok]) > 0 else np.nan)
        if np.isfinite(rr).all():
            rows.append(dict(mouse=mo, day=dy, cluster_id=int(c),
                             identity=IDENT.get((mo, dy, int(c))),
                             r_odd=rr[0], r_even=rr[1]))
R = pd.DataFrame(rows)
R = R[R.identity.isin(ORDER)]
sb = lambda r: 2 * r / (1 + r)
rho = pearsonr(R.r_odd, R.r_even)[0]
print(f'{len(R)} cells, {R.groupby(["mouse","day"]).ngroups} sessions')
print(f'  split-half r = {rho:.3f}; Spearman-Brown corrected = {sb(rho):.3f}')
for k in ORDER:
    d = R[R.identity == k]
    rk = pearsonr(d.r_odd, d.r_even)[0]
    print(f'  {k:22s} n={len(d):5d}  split-half {rk:.3f}  corrected {sb(rk):.3f}')

fig = plt.figure(figsize=(8.6, 3.0))
gs = fig.add_gridspec(1, 3, wspace=.42, width_ratios=[1, 1, .9])

ax = fig.add_subplot(gs[0])
ax.scatter(R.r_odd, R.r_even, s=3, color='0.5', alpha=.3, lw=0)
lim = [-.6, 1.0]
ax.plot(lim, lim, color='#c04744', lw=1, ls=':')
ax.set_xlim(lim); ax.set_ylim(lim)
ax.set_xlabel('agreement r, odd trials', fontsize=8.5)
ax.set_ylabel('agreement r, even trials', fontsize=8.5)
_lp(ax, 'A')
ax.set_title(f'split-half r = {rho:.2f}\n(Spearman-Brown {sb(rho):.2f})',
             fontsize=8, loc='left')
ax.tick_params(labelsize=7); ax.spines[['top', 'right']].set_visible(False)

ax = fig.add_subplot(gs[1])
for i, k in enumerate(ORDER):
    d = R[R.identity == k]
    rk = pearsonr(d.r_odd, d.r_even)[0]
    ax.bar(i, sb(rk), width=.66, color=ICOL[k], lw=.7, edgecolor='k')
    ax.annotate(f'{sb(rk):.2f}', (i, sb(rk)), ha='center', va='bottom',
                fontsize=6.8, color='0.2')
ax.set_xticks(range(4)); ax.set_xticklabels(SHORT, fontsize=7.5)
ax.set_ylim(0, 1)
ax.set_ylabel('Reliability (corrected)', fontsize=8.5)
_lp(ax, 'B')
ax.set_title('by identity', fontsize=8, loc='left')
ax.tick_params(labelsize=7); ax.spines[['top', 'right']].set_visible(False)

# what the attenuation means for Figure 2's differences
ax = fig.add_subplot(gs[2])
M = U[U.in_population & U.r.notna() & U.identity.isin(ORDER)]
obs = [M[M.identity == k].r.mean() for k in ORDER]
rel = [sb(pearsonr(R[R.identity == k].r_odd, R[R.identity == k].r_even)[0])
       for k in ORDER]
for i, k in enumerate(ORDER):
    ax.plot([i, i], [obs[i], obs[i] / rel[i]], color='0.6', lw=.9, zorder=1)
    ax.scatter(i, obs[i], s=26, color=ICOL[k], lw=0, zorder=3)
    ax.scatter(i, obs[i] / rel[i], s=26, facecolor='none', edgecolor=ICOL[k],
               lw=1.2, zorder=3)
ax.set_xticks(range(4)); ax.set_xticklabels(SHORT, fontsize=7.5)
ax.set_ylabel('Mean agreement r', fontsize=8.5)
_lp(ax, 'C')
ax.set_title('filled = measured\nopen = attenuation-corrected', fontsize=7.5,
             loc='left')
ax.tick_params(labelsize=7); ax.spines[['top', 'right']].set_visible(False)

out = f'{FIG}/fig2_supp_reliability.pdf'
plt.savefig(out, dpi=200, bbox_inches='tight')
print(f'saved {out}')
