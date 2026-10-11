"""Graphical abstract, in the Cell Press stacked-panel style.

Four stacked statements, read top to bottom: what the phenomenon is, which cells
express it, why it matters for behaviour, and what the state is.

The numbers are the paper's own, and the last two panels are computed here from
the same cached tables the figures use rather than typed in:

  behaviour   data/eye_anchoring/eye_anchoring_trials.csv -- hit rate by state
              and cue condition, the state being the sign of PC1
  the pupil   data/eye_anchoring/eye_position_profiles_gated.npz -- the Figure
              5I profiles, drawn here as the anchored minus non-anchored
              DIFFERENCE rather than as the two absolute curves. Pupil radius
              swings 2.2 z along the track against a 0.27 z difference between
              states, so on an absolute axis at this size the two curves are one
              curve; the difference form is what makes "lower at every position"
              visible, and it is the form PAPER.md says is needed for exactly
              that reason.

A panel titled "One global state, read out by cell type -- not built by the
local circuit" was dropped to make room.

Set in Arial like every other figure -- the Avenir Next used for the document's
display type is deliberately NOT used inside figure PDFs.

Writes graphical_abstract.pdf
"""
import os

import pandas as pd
from scipy.stats import wilcoxon

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.patches import Patch, Rectangle

plt.rcParams['font.family'] = 'Arial'
plt.rcParams['pdf.fonttype'] = 42

FIG = os.path.dirname(os.path.abspath(__file__))
ROOT = '/Users/harryclark/Documents/spatial-manifolds'
OUT = f'{FIG}/graphical_abstract.pdf'

ANCH = '#b53f8f'          # magenta, as in the figures
SLOW_G, FAST_G, THETA_C = '#2f7d32', '#e8701a', '#7b4173'   # bands, as Fig 7
NON = '#4f9d9d'           # teal
GRID_C = '#c04744'
INT_C = '#d95f02'
GREY = '#9a9a9a'

# generous hspace: each panel carries a two-line bold banner ABOVE its axes, so
# the usual spacing puts the next banner through the previous panel's content
fig = plt.figure(figsize=(3.55, 6.15))
G = fig.add_gridspec(4, 1, height_ratios=[1.00, 0.76, 0.84, 0.86],
                     hspace=.95, left=.04, right=.96, top=.950, bottom=.020)


def banner(ax, text):
    ax.text(.5, 1.05, text, transform=ax.transAxes, ha='center', va='bottom',
            fontsize=7.0, weight='bold', linespacing=1.35)


# ── 1. the phenomenon ────────────────────────────────────────────────────────
ax = fig.add_subplot(G[0]); ax.axis('off')
ax.set_xlim(0, 1); ax.set_ylim(0, 1)
banner(ax, 'Entorhinal firing anchors to the task\non some trials and not others')
# track strip
ax.add_patch(Rectangle((.06, .70), .88, .13, facecolor='0.15', lw=0))
ax.add_patch(Rectangle((.45, .70), .10, .13, facecolor='#9ecae1', lw=0))
ax.text(.50, .865, 'reward zone', ha='center', fontsize=5.4, color='0.3')
ax.text(.06, .655, 'start', fontsize=5.2, color='0.45', va='top')
ax.text(.94, .655, '200 cm', fontsize=5.2, color='0.45', va='top', ha='right')

rng = np.random.default_rng(3)
for k, (x0, col, lab) in enumerate(((.07, ANCH, 'anchored'),
                                    (.55, NON, 'non-anchored'))):
    sub = fig.add_axes([0, 0, 1, 1])        # placeholder, repositioned below
    sub.remove()
    axm = ax.inset_axes([x0, .06, .38, .46])
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
        sp.set_color(col); sp.set_linewidth(1.6)
    axm.set_title(lab, fontsize=6.0, color=col, pad=2.0)
    if k == 0:
        axm.set_ylabel('trials', fontsize=5.4, labelpad=1.5)

# ── 2. which cells ───────────────────────────────────────────────────────────
ax = fig.add_subplot(G[1])
# The superseded version read 'spatial coding in general does not' over the
# values +0.119, +0.080, -0.000 and -0.024. Those came from a signed
# correlation scored against an absolute null; corrected, every class
# exceeds its own chance level and the claim is graded, not exclusive
# (Figure 2D).
banner(ax, 'Every class follows it \u2014 grid cells and\ninterneurons about twice as strongly')
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
    if v > .02:
        ax.text(i, v + .010, f'+{v:.3f}', ha='center', fontsize=5.2, color=cols[i])
ax.spines[['top', 'right']].set_visible(False)

# ── 3. why it matters ────────────────────────────────────────────────────────
ax = fig.add_subplot(G[2]); ax.axis('off')
ax.set_xlim(0, 1); ax.set_ylim(0, 1)
banner(ax, 'Anchored trials are better performed \u2014 most\n'
           'when the reward zone carries no cue')
axh = ax.inset_axes([.17, .38, .66, .56])
E = pd.read_csv(f'{ROOT}/data/eye_anchoring/eye_anchoring_trials.csv')
E = E.dropna(subset=['pc1', 'perf', 'ttype'])
E['hit'] = (E.perf == 'hit').astype(float)
E['anch'] = E.pc1 > 0
# per-session paired benefit, which is what the statistic is computed on; the
# bars show the pooled rate so the panel reads as performance, not as a delta
S_ = E.groupby(['mouse', 'day', 'ttype', 'anch']).hit.mean().unstack('anch').dropna()
S_.columns = ['non', 'anch']
for j, (tt, lab) in enumerate((('b', 'cued'), ('nb', 'uncued'))):
    g = E[E.ttype == tt]
    for k, (st, col) in enumerate(((True, ANCH), (False, NON))):
        v = g[g.anch == st].hit.mean()
        axh.bar(j + (k - .5) * .34, v, .32, color=col, linewidth=0)
    d = (S_.xs(tt, level='ttype').anch - S_.xs(tt, level='ttype').non)
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

# the pupil, as Figure 5I: lower at every position on the track, which is what
# separates a change in engagement from attention paid at particular places
axp = ax.inset_axes([.10, .33, .34, .60])
_z = np.load(f'{ROOT}/data/eye_anchoring/eye_position_profiles_gated.npz')
_A, _N = _z['anch'], _z['non']
_x = (np.arange(_A.shape[1]) + .5) * (200.0 / _A.shape[1])
_d = _A - _N
_mu = np.nanmean(_d, axis=0)
_se = np.nanstd(_d, axis=0) / np.sqrt(np.sum(np.isfinite(_d), axis=0))
axp.axvspan(90, 110, color='#d8e4d0', alpha=.55, lw=0, zorder=0)
for _x0 in (0, 170):
    axp.axvspan(_x0, _x0 + 30, color='0.90', lw=0, zorder=0)
axp.axhline(0, color='0.55', lw=.8, zorder=1)
axp.fill_between(_x, _mu - _se, _mu + _se, color=ANCH, alpha=.30,
                 linewidth=0, edgecolor='none', zorder=2)
axp.plot(_x, _mu, color=ANCH, lw=1.2, zorder=3)
axp.set_xlim(0, 200); axp.set_xticks([0, 100, 200])
axp.set_ylim(-.62, .06); axp.set_yticks([-.5, -.25, 0])
axp.set_xlabel('position (cm)', fontsize=5.4, labelpad=1)
axp.set_ylabel('z(pupil), anchored\n$-$ non-anchored', fontsize=5.0, labelpad=1)
axp.tick_params(labelsize=5.0, length=2, pad=1)
axp.spines[['top', 'right']].set_visible(False)

# the field potential: slow gamma up, fast gamma and theta down
axg = ax.inset_axes([.63, .33, .32, .60])
gv = [-.094, .105, -.108]
gc = [THETA_C, SLOW_G, FAST_G]
axg.bar(range(3), gv, .62, color=gc, linewidth=0)
axg.axhline(0, color='0.35', lw=.9)
axg.set_xticks(range(3))
axg.set_xticklabels(['theta', 'slow \u03b3', 'fast \u03b3'], fontsize=5.0)
axg.set_ylim(-.16, .16); axg.set_yticks([-.1, 0, .1])
axg.tick_params(labelsize=5.0, length=2, pad=1)
axg.set_ylabel('anchored $-$ non (z)', fontsize=5.4, labelpad=1)
axg.spines[['top', 'right']].set_visible(False)

_same = int((_mu < 0).sum())
ax.text(.5, .09, f'the pupil is narrower at {_same} of {len(_mu)} positions on the '
                 f'track \u2014 a tonic shift,\nnot attention paid at the reward '
                 f'zone; the band difference reverses sign at 67 Hz',
        ha='center', va='top', fontsize=5.2, color='0.35', linespacing=1.4)

fig.savefig(OUT, bbox_inches='tight', dpi=300)
print(f'wrote {OUT}')
