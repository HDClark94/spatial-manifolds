"""Exploratory figure for vis_graded_anchoring.py (not a paper figure).

A  every trial's anchoring score g (r with the cell's anchored template minus
   the trial's shifted null), by the cell's own label, MEC and VIS
B  per cell: mean g on its non-anchored trials against its anchored trials
C  per session: share of the anchored-trial score kept on the trials on which
   the MEC POPULATION is non-anchored (all cells, locked ones included)
"""
import os
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy.stats import wilcoxon
from spatial_manifolds.anchoring import ANCH_COLOR, NONANCH_COLOR

PS = '/Users/harryclark/Documents/spatial-manifolds/data/population_state'
OUT = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'explore_vis_graded.pdf')
RC = {'MEC': '0.15', 'VIS': '#3f9b4f'}
D = pd.read_csv(f'{PS}/vis_graded_anchoring.csv')
T = np.load(f'{PS}/vis_graded_anchoring_trials.npz', allow_pickle=True)
fig, ax = plt.subplots(1, 4, figsize=(10.4, 2.9), gridspec_kw=dict(width_ratios=[1, 1, 1.1, .8], wspace=.45))
bins = np.linspace(-.4, 1, 57)
for k, reg in enumerate(('MEC', 'VIS')):
    a_ = ax[k]
    m = T['region'] == reg
    for lab, col, nm in ((1, ANCH_COLOR, 'anchored'), (0, NONANCH_COLOR, 'non-anchored')):
        g = np.concatenate([g[l == lab] for g, l in zip(T['g'][m], T['lab'][m])])
        g = g[np.isfinite(g)]
        a_.hist(g, bins, color=col, alpha=.6, density=True, label=f'{nm} (n={len(g)})', lw=0)
    a_.axvline(0, color='k', lw=.8, ls='--')
    a_.set_title(f'{reg}: every trial, by its own label', fontsize=7.5, color=RC[reg])
    a_.set_xlabel('anchoring score g\n(r with template - shifted null)', fontsize=7)
    a_.legend(fontsize=5.8, frameon=False); a_.tick_params(labelsize=6.5)
    a_.spines[['top', 'right']].set_visible(False)
ax[0].set_ylabel('density', fontsize=7.5)
a_ = ax[2]
V = D[~D.locked]
for reg in ('MEC', 'VIS'):
    v = V[V.region == reg]
    a_.scatter(v.g_anch, v.g_non, s=2, color=RC[reg], alpha=.35, lw=0, label=reg, rasterized=True)
a_.plot([0, 1], [0, 1], color='0.5', lw=.7); a_.axhline(0, color='0.5', lw=.6, ls=':')
r = V.assign(q=V.g_non / V.g_anch).groupby('region').q.median()
a_.set_title(f'per cell: kept on own non-anchored trials\nmedian MEC {r.MEC:.2f}, VIS {r.VIS:.2f} of anchored',
             fontsize=7, loc='left')
a_.set_xlabel('g, anchored trials', fontsize=7); a_.set_ylabel('g, non-anchored trials', fontsize=7)
a_.legend(fontsize=6, frameon=False, markerscale=3); a_.tick_params(labelsize=6.5)
a_.spines[['top', 'right']].set_visible(False)
a_ = ax[3]
s = D.groupby(['mouse', 'day', 'region']).apply(lambda g: (g.g_pop_non / g.g_pop_anch).median()).unstack().dropna()
for i in range(len(s)):
    a_.plot([0, 1], [s.MEC.iloc[i], s.VIS.iloc[i]], color='0.75', lw=.6)
for k, reg in enumerate(('MEC', 'VIS')):
    a_.scatter(np.full(len(s), k), s[reg], s=10, color=RC[reg], zorder=3, lw=0)
    a_.plot([k - .2, k + .2], [s[reg].median()] * 2, color='k', lw=1.5)
a_.set_xticks([0, 1]); a_.set_xticklabels(['MEC', 'VIS'], fontsize=7); a_.set_xlim(-.5, 1.5)
a_.axhline(1, color='0.5', lw=.6, ls=':')
a_.set_ylabel('score kept when MEC population\nis non-anchored (fraction)', fontsize=7)
a_.set_title(f'p = {wilcoxon(s.MEC, s.VIS).pvalue:.1g}, {len(s)} sessions', fontsize=7)
a_.tick_params(labelsize=6.5); a_.spines[['top', 'right']].set_visible(False)
fig.savefig(OUT, bbox_inches='tight'); print('wrote', OUT)
