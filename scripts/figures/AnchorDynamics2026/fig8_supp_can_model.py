"""Supplement to Figure 8: the gated attractor, simulated.

Figure 8 proposes that the anchoring state is the gain on a landmark
correction and not a change in the entorhinal circuit. This builds that
circuit and asks what it produces. The simulation is can_gate_model.py and the
test of its prediction in the recordings is gamma_vs_decode_error.py; this
script only draws them.

It is a supplement rather than a main figure because three of the eight
measured quantities do not come out, and two of those -- the spike-field
invariances -- cannot be scored at all. Panel F states that rather than
leaving it to the reader.

Writes fig8_supp_can_model.pdf
"""
import os

import matplotlib
import numpy as np
import pandas as pd

matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.patches import FancyArrowPatch, FancyBboxPatch
from scipy.stats import wilcoxon

plt.rcParams['font.family'] = 'Arial'
plt.rcParams['mathtext.fontset'] = 'custom'
plt.rcParams['mathtext.rm'] = 'Arial'
plt.rcParams['pdf.fonttype'] = 42

FIG = os.path.dirname(os.path.abspath(__file__))
ROOT = '/Users/harryclark/Documents/spatial-manifolds'
OUT = f'{FIG}/fig8_supp_can_model.pdf'

ANCH, NON = '#a8559e', '#3f9b8f'
SLOW_C, FAST_C, THETA_C = '#2f7d32', '#e8701a', '#7b4173'
LAND_C, SELF_C, ACH = '#c04744', '#2b6cb0', '#b8860b'

Z = np.load(f'{FIG}/can_gate_model.npz')
R = pd.read_csv(f'{ROOT}/data/lfp/gamma_vs_decode_error.csv')

fig = plt.figure(figsize=(9.4, 5.0))
gs = fig.add_gridspec(2, 4, hspace=.70, wspace=.46, left=.055, right=.985,
                      top=.90, bottom=.10)


def lp(ax, s, x=-.26, y=1.06):
    ax.text(x, y, s, transform=ax.transAxes, fontsize=10, weight='bold',
            va='bottom', ha='right')


# ---- A: the circuit ---------------------------------------------------------
ax = fig.add_subplot(gs[0, 0]); ax.axis('off')
ax.set_xlim(0, 1); ax.set_ylim(0, 1)
lp(ax, 'A', x=-.10)
ax.set_title('what was simulated', fontsize=7.6, loc='left', color='0.25')
for x_, y_, w_, h_, nm, fc, tc in (
        (.02, .70, .44, .17, 'landmark\nphase', '#fbf0ef', LAND_C),
        (.02, .13, .44, .17, 'error ring\nfast inhibition', '#fdf0e6', FAST_C),
        (.54, .40, .44, .20, 'attractor ring\nslow inhibition', '#eef5ee', SLOW_C)):
    ax.add_patch(FancyBboxPatch((x_, y_), w_, h_, boxstyle='round,pad=.018',
                                fc=fc, ec='0.72', lw=.8))
    ax.text(x_ + w_ / 2, y_ + h_ / 2, nm, fontsize=5.6, ha='center',
            va='center', color=tc, linespacing=1.25)
ax.add_patch(FancyArrowPatch((.24, .68), (.24, .32), arrowstyle='-|>',
                             mutation_scale=7, lw=1.2, color=LAND_C,
                             shrinkA=1, shrinkB=1))
ax.add_patch(FancyArrowPatch((.47, .22), (.74, .38), arrowstyle='-|>',
                             mutation_scale=7, lw=1.4, color=ACH,
                             shrinkA=1, shrinkB=1, connectionstyle='arc3,rad=-.2'))
ax.add_patch(FancyArrowPatch((.74, .62), (.47, .30), arrowstyle='-[',
                             mutation_scale=5, lw=1.0, color='0.5',
                             shrinkA=1, shrinkB=1, connectionstyle='arc3,rad=-.2'))
ax.text(.64, .18, 'gate', fontsize=5.8, color=ACH, ha='center')
ax.text(.995, .905, 'prediction\n(subtractive)', fontsize=5.0,
        color='0.5', ha='right', va='top', linespacing=1.2)
ax.text(.0, -.09, 'the gate scales only the error’s drive onto the\n'
                  'attractor; the landmark input arrives in both states',
        fontsize=5.4, color='0.45', va='top', linespacing=1.35)

# ---- B, C: the code under each setting --------------------------------------
for j, (key, nm, col) in enumerate((('map_open', 'gate OPEN', ANCH),
                                    ('map_shut', 'gate SHUT', NON))):
    ax = fig.add_subplot(gs[0, 1 + j])
    M = Z[key]
    ax.imshow(M / np.nanmax(M), aspect='auto', cmap='viridis',
              interpolation='nearest', origin='upper')
    ax.set_xticks([0, M.shape[1] - 1])
    ax.set_xticklabels(['0', 'one grid period'], fontsize=6)
    ax.set_ylabel('lap', fontsize=7.5); ax.tick_params(labelsize=6.5)
    ax.set_title(nm, fontsize=7.6, loc='left', color=col)
    lp(ax, 'BC'[j], x=-.20)
ax.text(1.0, -.21, 'one simulated grid cell: lap-to-lap r = +0.997 open, '
        '−0.059 shut', transform=ax.transAxes, fontsize=5.6,
        color='0.45', ha='right', va='top')

# ---- D: the spectra ---------------------------------------------------------
ax = fig.add_subplot(gs[0, 3])
ax.semilogy(Z['f'], Z['P_open'], color=ANCH, lw=1.0, label='open')
ax.semilogy(Z['f'], Z['P_shut'], color=NON, lw=1.0, label='shut')
ax.set_xlim(2, 120); ax.set_ylim(1e-6, 1e2)
ax.set_xlabel('frequency (Hz)', fontsize=7.5)
ax.set_ylabel('power', fontsize=7.5)
ax.legend(fontsize=5.8, frameon=False, loc='upper right')
ax.set_title('the simulated field potential', fontsize=7.6, loc='left',
             color='0.25')
ax.tick_params(labelsize=6.5); ax.spines[['top', 'right']].set_visible(False)
lp(ax, 'D', x=-.28)

# ---- E: the difference across frequency -------------------------------------
ax = fig.add_subplot(gs[1, 0])
ax.axhline(0, color='0.6', lw=.8, ls=':')
ax.axvspan(30, 48, color=SLOW_C, alpha=.12, lw=0)
ax.axvspan(60, 100, color=FAST_C, alpha=.12, lw=0)
ax.plot(Z['f'], Z['d'], color='0.25', lw=.9)
ax.set_xlim(2, 120); ax.set_ylim(-.65, .65)
ax.set_xlabel('frequency (Hz)', fontsize=7.5)
ax.set_ylabel('open $-$ shut (log10)', fontsize=7.5)
ax.set_title('the spectrum rotates', fontsize=7.6, loc='left', color='0.25')
ax.tick_params(labelsize=6.5); ax.spines[['top', 'right']].set_visible(False)
lp(ax, 'E', x=-.30)

# ---- F: the scorecard -------------------------------------------------------
ax = fig.add_subplot(gs[1, 1]); ax.axis('off')
ax.set_xlim(0, 1); ax.set_ylim(0, 1)
lp(ax, 'F', x=-.10)
ax.set_title('what it does and does not get', fontsize=7.6, loc='left',
             color='0.25')
ROWS = [('fields recur, or not', True), ('slow gamma up', True),
        ('fast gamma down', True), ('theta–fast coupling down', True),
        ('fast gamma tracks mismatch', True),
        ('theta–slow coupling flat', False),
        ('theta–speed decoupling', False),
        ('spike timing unchanged', None)]
# markers drawn rather than typed: Arial has no check or ballot-x glyph and
# matplotlib substitutes a box for both
for i, (nm, ok) in enumerate(ROWS):
    y = .95 - i * .108
    if ok:
        ax.plot(.04, y, marker='o', ms=4.0, color='#1b7a3d', mew=0)
    elif ok is False:
        ax.plot(.04, y, marker='x', ms=4.4, mew=1.5, color='#8a1c1c')
    else:
        ax.plot(.04, y, marker='_', ms=5.0, mew=1.4, color='0.55')
    ax.text(.11, y, nm, fontsize=5.9, color='0.3' if ok else '0.45',
            va='center')
ax.text(.0, .055, 'the dash cannot be scored: the surrogate spikes carry\n'
                  'no gamma structure, so PPC is zero in both states',
        fontsize=5.2, color='0.5', va='top', linespacing=1.35)

# ---- G: the prediction, in the model ----------------------------------------
ax = fig.add_subplot(gs[1, 2])
ax.plot(range(1, 11), Z['dec'], color=FAST_C, marker='o', ms=3, lw=1.2)
ax.set_xlabel('decile of |mismatch|, gate shut', fontsize=7.5)
ax.set_ylabel('fast-$\\gamma$ amplitude', fontsize=7.5)
ax.set_title('the prediction, in the model', fontsize=7.6, loc='left',
             color='0.25')
ax.tick_params(labelsize=6.5); ax.spines[['top', 'right']].set_visible(False)
lp(ax, 'G', x=-.30)

# ---- H: the prediction, in the recordings -----------------------------------
ax = fig.add_subplot(gs[1, 3])
rng = np.random.default_rng(0)
COLS = [('r_fast_non', 'fast\nnon-anch', FAST_C), ('r_fast_anch', 'fast\nanch', FAST_C),
        ('r_slow_non', 'slow\nnon-anch', SLOW_C), ('r_slow_anch', 'slow\nanch', SLOW_C)]
for i, (c, lab, col) in enumerate(COLS):
    v = R[c].dropna()
    ax.scatter(i + rng.uniform(-.16, .16, len(v)), v, s=4, color=col, alpha=.4,
               lw=0)
    ax.errorbar(i, v.mean(), yerr=v.std(ddof=1) / np.sqrt(len(v)), color='k',
                marker='_', ms=10, lw=1.2, capsize=2, zorder=4)
    p_ = wilcoxon(v).pvalue
    ax.annotate(f'{p_:.0e}'.replace('e-0', 'e-') if p_ < .01 else f'{p_:.2f}',
                (i, .99), xycoords=('data', 'axes fraction'), ha='center',
                va='top', fontsize=5.2,
                color='#8a1c1c' if p_ < .05 else '0.5')
ax.axhline(0, color='k', lw=.9, ls='--')
ax.set_xticks(range(4)); ax.set_xticklabels([c[1] for c in COLS], fontsize=5.8)
ax.set_ylabel('partial r with |decode error|', fontsize=7.0)
ax.set_title(f'and in the recordings ({len(R)} sessions)', fontsize=7.6,
             loc='left', color='0.25')
ax.tick_params(labelsize=6.5); ax.spines[['top', 'right']].set_visible(False)
lp(ax, 'H', x=-.30)

fig.savefig(OUT, dpi=300, bbox_inches='tight')
print(f'wrote {OUT}')
