"""Graphical abstract, in the Cell Press stacked-panel style.

Four stacked statements, read top to bottom: what the phenomenon is and that
cells do it together, which cells express it, why it matters for behaviour, and
what the state is.

NOTHING HERE IS TYPED IN EXCEPT THE FIGURE 2 AND FIGURE 7 BAR VALUES. The
population raster, the behaviour and the pupil profile are computed from the
same cached tables the figures use, so they cannot drift from the paper the way
the retired Figure 2D values once did:

  raster      data/population_state/labels/M26D18.npz -- the cell x trial label
              matrix, its PC1, and the per-cell PC1 loading. Cells are ordered
              as in Figure 1: varying cells by loading, then a gap, then the
              cells locked into one mode, grouped by which.
  behaviour   data/eye_anchoring/eye_anchoring_trials.csv -- hit rate by state
              and cue condition, the state being the sign of PC1.
  the pupil   data/eye_anchoring/eye_position_profiles_gated.npz -- Figure 5I,
              drawn as that panel draws it: both states' absolute profiles with
              their SEM, the reward zone and the two black-box segments shaded.
              The per-bin significance markers are left off; they are
              uncorrected in Figure 5 and mark extent rather than testing it,
              which is not a distinction a graphical abstract can carry.

A panel titled "One global state, read out by cell type -- not built by the
local circuit" was dropped to make room; it survives in the highlights.

Set in Arial like every other figure -- the Avenir Next used for the document's
display type is deliberately NOT used inside figure PDFs.

Writes graphical_abstract.pdf
"""
import os

import matplotlib
import numpy as np
import pandas as pd
from scipy.stats import wilcoxon

matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.colors import BoundaryNorm, ListedColormap
from matplotlib.patches import Patch, Rectangle

plt.rcParams['font.family'] = 'Arial'
plt.rcParams['pdf.fonttype'] = 42

FIG = os.path.dirname(os.path.abspath(__file__))
ROOT = '/Users/harryclark/Documents/spatial-manifolds'
OUT = f'{FIG}/graphical_abstract.pdf'

ANCH = '#b53f8f'          # magenta, as in the figures
NON = '#4f9d9d'           # teal
GRID_C = '#c04744'
INT_C = '#d95f02'
GREY = '#9a9a9a'
SLOW_G, FAST_G, THETA_C = '#2f7d32', '#e8701a', '#7b4173'   # bands, as Fig 7
TA = ListedColormap([NON, ANCH])
NORM = BoundaryNorm([-0.5, 0.5, 1.5], TA.N)

fig = plt.figure(figsize=(3.55, 6.45))
G = fig.add_gridspec(4, 1, height_ratios=[1.22, 0.70, 0.76, 0.80],
                     hspace=.88, left=.04, right=.96, top=.952, bottom=.018)


def banner(ax, text):
    ax.text(.5, 1.05, text, transform=ax.transAxes, ha='center', va='bottom',
            fontsize=7.0, weight='bold', linespacing=1.35)


# ── 1. the phenomenon, cell and population ───────────────────────────────────
ax = fig.add_subplot(G[0]); ax.axis('off')
ax.set_xlim(0, 1); ax.set_ylim(0, 1)
banner(ax, 'Entorhinal firing anchors to the task on some\n'
           'trials and not others — and cells do it together')

# the track, over the two example rate maps
ax.add_patch(Rectangle((.02, .80), .46, .075, facecolor='0.15', lw=0))
ax.add_patch(Rectangle((.225, .80), .052, .075, facecolor='#9ecae1', lw=0))
ax.text(.25, .895, 'reward zone', ha='center', fontsize=4.8, color='0.3')
ax.text(.02, .775, 'start', fontsize=4.6, color='0.45', va='top')
ax.text(.48, .775, '200 cm', fontsize=4.6, color='0.45', va='top', ha='right')

rng = np.random.default_rng(3)
for k, (x0, col, lab) in enumerate(((.02, ANCH, 'anchored'),
                                    (.265, NON, 'non-anchored'))):
    axm = ax.inset_axes([x0, .22, .215, .44])
    n_tr, n_pos = 26, 50
    if k == 0:
        M = np.exp(-((np.arange(n_pos)[None, :] - 24) ** 2) / 18)
        M = M * (0.55 + 0.45 * rng.random((n_tr, 1)))
    else:
        M = np.zeros((n_tr, n_pos))
        for t in range(n_tr):
            c = rng.integers(4, n_pos - 4)
            M[t] = np.exp(-((np.arange(n_pos) - c) ** 2) / 18)
        M = M * (0.55 + 0.45 * rng.random((n_tr, 1)))
    axm.imshow(M + 0.06 * rng.random((n_tr, n_pos)), aspect='auto',
               cmap='viridis', interpolation='nearest')
    axm.set_xticks([]); axm.set_yticks([])
    for sp in axm.spines.values():
        sp.set_color(col); sp.set_linewidth(1.4)
    ax.text(x0 + .1075, .685, lab, ha='center', fontsize=5.4, color=col)
    if k == 0:
        axm.set_ylabel('trials', fontsize=4.8, labelpad=1.5)
ax.text(.25, .125, 'one cell, two states', ha='center', va='top',
        fontsize=5.0, color='0.4')

# the population: the Figure 1 raster with its PC1 beside it
z = np.load(f'{ROOT}/data/population_state/labels/M26D18.npz', allow_pickle=True)
L, load, pc1 = z['labels'], z['pc1_load'], z['pc1']
C = pd.read_csv(f'{ROOT}/data/population_state/anchoring_cells.csv')
C = C[(C.mouse == 26) & (C.day == 18)]
keep = np.isin(z['cluster_id'], C.cluster_id[C.in_population].values)
L, load = L[keep], load[keep]
# Figure 1's block structure: a constant row contributes nothing to the
# decomposition, so cells locked into one mode are moved past a gap and grouped
# by which mode, instead of being scattered through the ordered block
with np.errstate(invalid='ignore'):
    varies = np.array([np.isfinite(r).sum() > 1 and np.nanstd(r) > 0
                       for r in L])
ord_v = np.where(varies)[0][np.argsort(load[varies])[::-1]]
lk = np.where(~varies)[0]
_fr = np.array([np.nanmean(L[i]) if np.isfinite(L[i]).any() else 0.
                for i in lk]) if len(lk) else np.array([])
ord_l = lk[np.argsort(-_fr)] if len(lk) else lk
gap = max(2, int(round(.025 * L.shape[0])))
blocks = [L[ord_v]] + ([np.full((gap, L.shape[1]), np.nan), L[ord_l]]
                       if len(ord_l) else [])
Lb = np.vstack(blocks)
ntr = Lb.shape[1]

axr = ax.inset_axes([.58, .22, .27, .62])
axr.imshow(Lb.T, aspect='auto', cmap=TA, norm=NORM, interpolation='nearest',
           extent=[0, Lb.shape[0], ntr, 0])
axr.set_xticks([]); axr.tick_params(labelsize=4.8, length=2, pad=1)
axr.set_yticks([0, ntr])
axr.set_ylabel('trial', fontsize=4.8, labelpad=1)
for sp in axr.spines.values():
    sp.set_linewidth(.5)

axq = ax.inset_axes([.875, .22, .105, .62])
tr = np.arange(ntr)
axq.fill_betweenx(tr, 0, pc1, where=pc1 >= 0, color=ANCH, linewidth=0,
                  edgecolor='none', interpolate=True)
axq.fill_betweenx(tr, 0, pc1, where=pc1 < 0, color=NON, linewidth=0,
                  edgecolor='none', interpolate=True)
axq.axvline(0, color='0.4', lw=.5)
axq.set_ylim(ntr, 0); axq.set_xticks([]); axq.set_yticks([])
axq.set_xlabel('PC1', fontsize=4.8, labelpad=1)
for sp in axq.spines.values():
    sp.set_visible(False)
ax.text(.785, .125, f'{Lb.shape[0]} MEC cells, one session',
        ha='center', va='top', fontsize=5.0, color='0.4')

# ── 2. which cells ───────────────────────────────────────────────────────────
ax = fig.add_subplot(G[1])
banner(ax, 'Every class follows it — grid cells and\n'
           'interneurons about twice as strongly')
names = ['grid', 'inter-\nneuron', 'non-grid\nspat.', 'non-\nspat.']
vals = [0.185, 0.131, 0.081, 0.056]
cols = [GRID_C, INT_C, GREY, GREY]
ax.bar(range(4), vals, .64, color=cols, linewidth=0)
ax.axhline(0, color='0.35', lw=.9)
ax.set_xticks(range(4)); ax.set_xticklabels(names, fontsize=5.6)
ax.set_ylabel('agreement with the\nstate, over chance', fontsize=5.8)
ax.tick_params(axis='y', labelsize=5.4)
ax.set_ylim(0, .235)
for i, v in enumerate(vals):
    ax.text(i, v + .010, f'+{v:.3f}', ha='center', fontsize=5.2, color=cols[i])
ax.spines[['top', 'right']].set_visible(False)

# ── 3. why it matters ────────────────────────────────────────────────────────
ax = fig.add_subplot(G[2]); ax.axis('off')
ax.set_xlim(0, 1); ax.set_ylim(0, 1)
banner(ax, 'Anchored trials are better performed — most\n'
           'when the reward zone carries no cue')
axh = ax.inset_axes([.17, .38, .66, .56])
E = pd.read_csv(f'{ROOT}/data/eye_anchoring/eye_anchoring_trials.csv')
E = E.dropna(subset=['pc1', 'perf', 'ttype'])
E['hit'] = (E.perf == 'hit').astype(float)
E['anch'] = E.pc1 > 0
S_ = E.groupby(['mouse', 'day', 'ttype', 'anch']).hit.mean().unstack('anch').dropna()
S_.columns = ['non', 'anch']
for j, (tt, lab) in enumerate((('b', 'cued'), ('nb', 'uncued'))):
    g = E[E.ttype == tt]
    for k, (st, col) in enumerate(((True, ANCH), (False, NON))):
        axh.bar(j + (k - .5) * .34, g[g.anch == st].hit.mean(), .32,
                color=col, linewidth=0)
    d = S_.xs(tt, level='ttype').anch - S_.xs(tt, level='ttype').non
    hi = max(g[g.anch].hit.mean(), g[~g.anch].hit.mean())
    axh.text(j, hi + .045, f'+{d.mean():.2f}', ha='center', fontsize=5.4,
             color='0.2')
axh.set_xticks([0, 1]); axh.set_xticklabels(['cued', 'uncued'], fontsize=5.6)
axh.set_ylim(0, 1.14); axh.set_yticks([0, .4, .8])
axh.set_ylabel('hit rate', fontsize=5.4, labelpad=1)
axh.tick_params(labelsize=5.0, length=2, pad=1)
axh.spines[['top', 'right']].set_visible(False)
axh.legend(handles=[Patch(facecolor=ANCH, label='anchored'),
                    Patch(facecolor=NON, label='non-anchored')],
           fontsize=4.8, frameon=False, loc='upper right', handlelength=.9,
           handleheight=.9, borderpad=.1, labelspacing=.25)
_W = (S_.anch - S_.non).unstack('ttype').dropna()
_p = wilcoxon(_W.nb, _W.b).pvalue
ax.text(.5, .20, f'the benefit is larger when the zone must be found by path\n'
                 f'integration ({(_W.nb - _W.b).mean():+.3f}, p = {_p:.0e}'
                 .replace('e-0', 'e-') + f', {len(_W)} sessions)',
        ha='center', va='top', fontsize=5.2, color='0.35', linespacing=1.4)

# ── 4. what the state is ─────────────────────────────────────────────────────
ax = fig.add_subplot(G[3]); ax.axis('off')
ax.set_xlim(0, 1); ax.set_ylim(0, 1)
banner(ax, 'The state is attentional engagement, with a\n'
           'frequency-specific change in the field potential')

# Figure 5I as that panel draws it: both states' absolute profiles, SEM across
# sessions, reward zone and the two black-box segments shaded
axp = ax.inset_axes([.085, .32, .44, .60])
_z = np.load(f'{ROOT}/data/eye_anchoring/eye_position_profiles_gated.npz')
_A, _N = _z['anch'], _z['non']
_x = (np.arange(_A.shape[1]) + .5) * (200.0 / _A.shape[1])
axp.axvspan(90, 110, color='#d8e4d0', alpha=.55, lw=0, zorder=0)
for _x0 in (0, 170):
    axp.axvspan(_x0, _x0 + 30, color='0.90', lw=0, zorder=0)
for _M, _c, _l in ((_A, ANCH, 'anchored'), (_N, NON, 'non-anchored')):
    _mu = np.nanmean(_M, axis=0)
    _se = np.nanstd(_M, axis=0) / np.sqrt(np.sum(np.isfinite(_M), axis=0))
    axp.fill_between(_x, _mu - _se, _mu + _se, color=_c, alpha=.30,
                     linewidth=0, edgecolor='none', zorder=2)
    axp.plot(_x, _mu, color=_c, lw=1.1, zorder=3, label=_l)
axp.axhline(0, color='0.75', lw=.7, zorder=1)
axp.set_xlim(0, 200); axp.set_xticks([0, 100, 200])
axp.set_xlabel('position (cm)', fontsize=5.4, labelpad=1)
axp.set_ylabel('z(pupil radius)', fontsize=5.4, labelpad=1)
axp.tick_params(labelsize=5.0, length=2, pad=1)
axp.legend(fontsize=4.6, frameon=False, loc='lower left', handlelength=.8,
           borderpad=.1, labelspacing=.2)
axp.spines[['top', 'right']].set_visible(False)

axg = ax.inset_axes([.70, .32, .26, .60])
gv = [-.094, .105, -.108]
axg.bar(range(3), gv, .62, color=[THETA_C, SLOW_G, FAST_G], linewidth=0)
axg.axhline(0, color='0.35', lw=.9)
axg.set_xticks(range(3))
axg.set_xticklabels(['theta', 'slow γ', 'fast γ'], fontsize=5.0)
axg.set_ylim(-.16, .16); axg.set_yticks([-.1, 0, .1])
axg.tick_params(labelsize=5.0, length=2, pad=1)
axg.set_ylabel('anchored $-$ non (z)', fontsize=5.4, labelpad=1)
axg.spines[['top', 'right']].set_visible(False)

_same = int((np.nanmean(_A - _N, axis=0) < 0).sum())
ax.text(.5, .10, f'the pupil is narrower at {_same} of {_A.shape[1]} positions on '
                 f'the track — a tonic shift, not\nattention paid at the reward '
                 f'zone; the band difference reverses sign at 67 Hz',
        ha='center', va='top', fontsize=5.2, color='0.35', linespacing=1.4)

fig.savefig(OUT, bbox_inches='tight', dpi=300)
print(f'wrote {OUT}')
