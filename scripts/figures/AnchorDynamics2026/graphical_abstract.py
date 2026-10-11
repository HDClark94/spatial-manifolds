"""Graphical abstract, in the Cell Press stacked-panel style.

Four stacked statements, read top to bottom: what the phenomenon is, which cells
express it, what the state is, and where it comes from. Numbers are the paper's
own.

The pupil used to appear only as a one-line strap at the foot and the field
potential not at all, which left the abstract claiming a state without showing
either of the two measurements that characterise it.

Set in Arial like every other figure -- the Avenir Next used for the document's
display type is deliberately NOT used inside figure PDFs.

Writes graphical_abstract.pdf
"""
import os

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.patches import FancyBboxPatch, Rectangle

plt.rcParams['font.family'] = 'Arial'
plt.rcParams['pdf.fonttype'] = 42

FIG = os.path.dirname(os.path.abspath(__file__))
OUT = f'{FIG}/graphical_abstract.pdf'

ANCH = '#b53f8f'          # magenta, as in the figures
SLOW_G, FAST_G, THETA_C = '#2f7d32', '#e8701a', '#7b4173'   # bands, as Fig 7
NON = '#4f9d9d'           # teal
GRID_C = '#c04744'
INT_C = '#d95f02'
GREY = '#9a9a9a'

# generous hspace: each panel carries a two-line bold banner ABOVE its axes, so
# the usual spacing puts the next banner through the previous panel's content
fig = plt.figure(figsize=(3.55, 6.45))
G = fig.add_gridspec(4, 1, height_ratios=[1.00, 0.78, 0.86, 0.96],
                     hspace=.92, left=.04, right=.96, top=.950, bottom=.020)


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

# ── 3. what the state is ─────────────────────────────────────────────────────
ax = fig.add_subplot(G[2]); ax.axis('off')
ax.set_xlim(0, 1); ax.set_ylim(0, 1)
banner(ax, 'A focused, constricted-pupil state, with a\nfrequency-specific change in the field potential')

# the pupil, as performance against pupil size -- the measurement that says the
# anchored state is focused rather than drowsy
axp = ax.inset_axes([.10, .33, .34, .60])
hit = [.826, .712, .659, .582, .552, .534, .472, .481, .440, .391]
axp.plot(range(1, 11), hit, color=ANCH, lw=1.3, marker='o', ms=2.2,
         mew=0, clip_on=False)
axp.set_xlim(.5, 10.5); axp.set_ylim(.33, .89)
axp.set_xticks([1, 10]); axp.set_xticklabels(['narrow\npupil', 'wide'], fontsize=5.0,
                    linespacing=1.2)
axp.set_yticks([.4, .6, .8]); axp.tick_params(labelsize=5.0, length=2, pad=1)
axp.set_ylabel('hit rate', fontsize=5.4, labelpad=1)
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
ax.text(.5, .09, 'the pupil narrows on entry to the anchored state (\u22120.41 z);\n'
                 'the band difference reverses sign at 67 Hz in 25 of 26 sessions',
        ha='center', va='top', fontsize=5.2, color='0.35', linespacing=1.4)

# ── 4. where it comes from ───────────────────────────────────────────────────
ax = fig.add_subplot(G[3]); ax.axis('off')
ax.set_xlim(0, 1); ax.set_ylim(0, 1)
banner(ax, 'One global state, read out by cell type —\nnot built by the local circuit')
regions = [('MEC', .09), ('subicular', .32), ('visual', .565), ('cerebellum', .80)]
for nm, x in regions:
    ax.add_patch(FancyBboxPatch((x, .50), .17, .17,
                                boxstyle='round,pad=0.012,rounding_size=0.02',
                                facecolor='#eef2f6', edgecolor='#9fb3c8', lw=.7))
    ax.text(x + .085, .585, nm, ha='center', va='center', fontsize=5.3)
ax.annotate('', xy=(.95, .80), xytext=(.05, .80),
            arrowprops=dict(arrowstyle='-', color=ANCH, lw=2.4, alpha=.5))
ax.text(.5, .845, 'the same anchoring state in every structure recorded',
        ha='center', fontsize=5.6, color=ANCH)
for _, x in regions:
    ax.plot([x + .085, x + .085], [.68, .78], color=ANCH, lw=1.0, alpha=.55)

ax.text(.5, .455, 'connected pairs share it no more than\nunconnected pairs  —  it is not propagated locally',
        ha='center', va='top', fontsize=5.6, color='0.25')
ax.add_patch(FancyBboxPatch((.14, .05), .72, .19,
                            boxstyle='round,pad=0.015,rounding_size=0.03',
                            facecolor='#fdf2f8', edgecolor=ANCH, lw=.9))
ax.text(.5, .145, 'anchored trials  →  better path integration',
        ha='center', va='center', fontsize=5.9, color=ANCH, weight='bold')

fig.savefig(OUT, bbox_inches='tight', dpi=300)
print(f'wrote {OUT}')
