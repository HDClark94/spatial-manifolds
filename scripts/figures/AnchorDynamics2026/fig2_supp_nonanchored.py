"""Figure 2 supplement — what does a NON-ANCHORED trial look like for a cell?

Usage:  python3 fig2_supp_nonanchored.py

ONE QUESTION: when a cell is called non-anchored, has its field moved somewhere
else, or has it stopped having a field?

The label only says the trial did not match the others. Two very different
things produce that, and they mean opposite things about the mechanism: a
DISPLACED field means the map is intact but registered to the wrong reference,
while LOST TUNING means the spatial code itself has gone. The first is a
pointer problem, the second is a coding problem.

THE TEST, AND WHY IT NEEDS A NULL. Each trial's smoothed map is correlated
against the cell's anchored template at every circular offset. If the field has
merely moved, the correlation at zero offset collapses but the BEST correlation
over offsets is preserved. That is exactly what the raw numbers show -- zero
offset falls from +0.33 to +0.07 while the best falls only from +0.56 to +0.53
-- and it is exactly the wrong conclusion to draw from them.

A maximum over 100 offsets is large by construction. Two UNRELATED real cells,
matched template to trial within the same session, reach +0.500. The observed
non-anchored value of +0.529 sits 0.03 above that floor. The statistic has
almost no headroom, so a preserved best-offset correlation is not evidence of a
displaced field; it is evidence that the search finds something in anything.
A bin-shuffled trial map, which has no spatial structure at all, still reaches
+0.189.

THE ANSWER, then, is that non-anchored trials show a collapse of spatial
correspondence with NO credible evidence of a coherent field elsewhere. The
mean best offset on those trials is 47 cm against the 50 cm expected from a
uniform pick on a 200 cm track -- the offsets are where noise put them, not
where a field went.

This supplement exists as much for the null as for the result: without panels B
and C the raw comparison in panel A reads as displacement, which is what it
looked like before the floor was measured.
"""
import glob
import os
import sys
import warnings

import numpy as np
import pandas as pd

warnings.filterwarnings('ignore')
import matplotlib

matplotlib.use('Agg')
import matplotlib.pyplot as plt
import seaborn as sns
from scipy.stats import wilcoxon

sys.path.insert(0, '/Users/harryclark/Documents/spatial-manifolds/src')
from spatial_manifolds.anchoring import ANCH_COLOR, NONANCH_COLOR

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
ICOL = {'grid': '#c04744', 'non-grid spatial': '#3171ae',
        'non-spatial': '#888888', 'putative interneuron': '#d95f02'}
TL = 200.0

D = pd.concat([pd.read_csv(f) for f in glob.glob(f'{PS}/trial_map_stats_w*.csv')],
              ignore_index=True)
U = pd.read_csv(f'{PS}/unit_table.csv')
N = pd.read_csv(f'{PS}/trial_map_null.csv')
M = (D.merge(U[['mouse', 'day', 'cluster_id', 'identity', 'in_population']],
             on=['mouse', 'day', 'cluster_id']).query('in_population'))
M = M[M.identity.isin(ORDER)].dropna(subset=['r0_anch', 'r0_non'])
NULL_SHUF, NULL_OTHER = N.null_shuffle.mean(), N.null_othercell.mean()
print(f'{len(M)} cells with a template and both states, '
      f'{M.groupby(["mouse","day"]).ngroups} sessions')
print(f'  zero offset : anchored {M.r0_anch.mean():+.3f}  '
      f'non-anchored {M.r0_non.mean():+.3f}  p={wilcoxon(M.r0_anch,M.r0_non).pvalue:.2g}')
print(f'  best offset : anchored {M.rbest_anch.mean():+.3f}  '
      f'non-anchored {M.rbest_non.mean():+.3f}')
print(f'  NULL best offset: other cell {NULL_OTHER:.3f}, bin-shuffled {NULL_SHUF:.3f}'
      f'  ({len(N)} cells, {N.groupby(["mouse","day"]).ngroups} sessions)')
print(f'  headroom above the other-cell floor: anchored '
      f'{M.rbest_anch.mean()-NULL_OTHER:+.3f}, non-anchored '
      f'{M.rbest_non.mean()-NULL_OTHER:+.3f}')

fig = plt.figure(figsize=(9.8, 3.0))
gs = fig.add_gridspec(1, 4, wspace=.46)

ax = fig.add_subplot(gs[0])
L = pd.melt(M[['r0_anch', 'r0_non']].rename(
    columns={'r0_anch': 'anchored', 'r0_non': 'non-anchored'}))
sns.boxplot(data=L, x='variable', y='value', ax=ax, showfliers=False,
            palette=[ANCH_COLOR, NONANCH_COLOR], width=.6, linewidth=.9,
            medianprops=dict(color='k', lw=1.4))
ax.axhline(0, color='0.6', lw=.7, ls=':')
ax.set_xlabel(''); ax.set_ylabel('r at zero offset', fontsize=8.5)
_lp(ax, 'A')
ax.set_title(f'spatial correspondence collapses\np = '
             f'{wilcoxon(M.r0_anch, M.r0_non).pvalue:.0e}', fontsize=7.5, loc='left')
ax.tick_params(labelsize=7); ax.spines[['top', 'right']].set_visible(False)

ax = fig.add_subplot(gs[1])
L = pd.melt(M[['rbest_anch', 'rbest_non']].rename(
    columns={'rbest_anch': 'anchored', 'rbest_non': 'non-anchored'}))
sns.boxplot(data=L, x='variable', y='value', ax=ax, showfliers=False,
            palette=[ANCH_COLOR, NONANCH_COLOR], width=.6, linewidth=.9,
            medianprops=dict(color='k', lw=1.4))
ax.axhline(NULL_OTHER, color='k', lw=1.3, ls='--')
ax.annotate(f'unrelated cells {NULL_OTHER:.2f}', (-.45, NULL_OTHER + .012),
            fontsize=6.2, color='k')
ax.axhline(NULL_SHUF, color='0.45', lw=1.1, ls=':')
ax.annotate(f'no structure {NULL_SHUF:.2f}', (-.45, NULL_SHUF + .012),
            fontsize=6.2, color='0.45')
ax.set_xlabel(''); ax.set_ylabel('best r over offsets', fontsize=8.5)
_lp(ax, 'B')
ax.set_title('but the best-offset search\nhas almost no headroom',
             fontsize=7.5, loc='left')
ax.tick_params(labelsize=7); ax.spines[['top', 'right']].set_visible(False)

ax = fig.add_subplot(gs[2])
for v, c, nm in ((M.lag_anch, ANCH_COLOR, 'anchored'),
                 (M.lag_non, NONANCH_COLOR, 'non-anchored')):
    ax.hist(v, bins=40, histtype='step', lw=1.5, color=c, density=True, label=nm)
ax.axvline(TL / 4, color='k', lw=1.3, ls='--')
ax.annotate('chance\n50 cm', (TL / 4 + 2, ax.get_ylim()[1] * .62), fontsize=6.2)
ax.set_xlabel('|best offset| (cm)', fontsize=8)
ax.set_ylabel('Density', fontsize=8.5)
_lp(ax, 'C')
ax.set_title('offsets are where noise put them', fontsize=7.5, loc='left')
ax.legend(fontsize=6, frameon=False, loc='upper left')
ax.tick_params(labelsize=7); ax.spines[['top', 'right']].set_visible(False)

ax = fig.add_subplot(gs[3])
rng = np.random.default_rng(0)
for i, k in enumerate(ORDER):
    d = M[M.identity == k]
    v = d.rbest_non - NULL_OTHER
    ax.scatter(i + rng.uniform(-.16, .16, len(v)), v, s=2, color=ICOL[k],
               alpha=.2, lw=0)
    ax.errorbar(i, v.mean(), yerr=v.std(ddof=1) / np.sqrt(len(v)), color='k',
                marker='_', ms=14, lw=1.5, capsize=3, zorder=4)
ax.axhline(0, color='k', lw=1.1, ls='--')
ax.set_xticks(range(4)); ax.set_xticklabels(['grid', 'NGS', 'NS', 'int'],
                                            fontsize=7.5)
ax.set_ylabel('best r above the\nunrelated-cell floor', fontsize=8)
_lp(ax, 'D')
ax.set_title('no identity recovers a field', fontsize=7.5, loc='left')
ax.tick_params(labelsize=7); ax.spines[['top', 'right']].set_visible(False)

out = f'{FIG}/fig2_supp_nonanchored.pdf'
plt.savefig(out, dpi=200, bbox_inches='tight')
print(f'saved {out}')
