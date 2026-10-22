"""Supplement to Figure 9: the same gated attractor, as a 2-D grid network.

The ring in Figure S13 carries one module's phase along the track, which is
enough for the field potential but assumes the thing a grid model ought to
produce. This is the same circuit built on a two-dimensional sheet whose units
fire in hexagonal grids in an open arena, driven along a straight line through
that arena and teleported back to the start at the end of each lap, which is
what a virtual linear track is. The teleport is what makes the gate matter:
the animal's position jumps back, its path integrator does not, so the bump
arrives at the next lap carrying a whole lap of translation and something has
to put it back.

The simulation is can_gate_model_2d.py; this script only draws it.

Writes fig8_supp_can_model_2d.pdf
"""
import os

import matplotlib
import numpy as np
import pandas as pd

matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.colors import BoundaryNorm, ListedColormap
from scipy.stats import wilcoxon

plt.rcParams['font.family'] = 'Arial'
plt.rcParams['mathtext.fontset'] = 'custom'
plt.rcParams['mathtext.rm'] = 'Arial'
plt.rcParams['pdf.fonttype'] = 42

FIG = os.path.dirname(os.path.abspath(__file__))
ROOT = '/Users/harryclark/Documents/spatial-manifolds'
OUT = f'{FIG}/fig8_supp_can_model_2d.pdf'

ANCH, NON = '#a8559e', '#3f9b8f'
ACH = '#b8860b'
TRACK = 200.0

Z = np.load(f'{FIG}/can_gate_model_2d.npz')

BANDS = [('theta', 6, 10), ('slow\n$\\gamma$', 30, 48),
         ('fast\n$\\gamma$', 60, 100)]
LINE = [(48, 52), (98, 102), (148, 152)]
_S = pd.read_csv(f'{ROOT}/data/lfp/spectrum_by_anchoring.csv')
_S = _S[~_S.freq.apply(lambda f: any(lo <= f <= hi for lo, hi in LINE))]
_S = _S.pivot_table(index=['mouse', 'day', 'freq'], columns='state',
                    values='log_p').reset_index()


def measured(lo, hi):
    b = _S[(_S.freq >= lo) & (_S.freq <= hi)].groupby(['mouse', 'day'])[
        ['anch', 'non']].mean()
    return (b.anch - b.non).dropna()


fig = plt.figure(figsize=(9.6, 5.0))
gs = fig.add_gridspec(2, 4, hspace=.72, wspace=.48, left=.055, right=.985,
                      top=.90, bottom=.10, width_ratios=[1.35, .95, .75, 1.25])


def lp(ax, s, x=-.24, y=1.06):
    ax.text(x, y, s, transform=ax.transAxes, fontsize=10, weight='bold',
            va='bottom', ha='right')


def bare(ax):
    ax.tick_params(labelsize=6.5)
    ax.spines[['top', 'right']].set_visible(False)


# ---- A: grid cells in the arena, and the track through it ------------------
GA = gs[0, 0].subgridspec(2, 3, hspace=.30, wspace=.14)
M_, GSC = Z['maps'], Z['grid_score']
pick = np.argsort(-GSC)[np.linspace(0, len(GSC) // 3, 6).astype(int)]
for k, idx in enumerate(pick):
    ax = fig.add_subplot(GA[k // 3, k % 3])
    ax.imshow(M_[idx].T, origin='lower', cmap='jet', interpolation='bilinear')
    ax.set_xticks([]); ax.set_yticks([])
    ax.text(.5, -.02, f'{GSC[idx]:.2f}', transform=ax.transAxes, fontsize=5.4,
            color='0.35', ha='center', va='top')
    if k == 0:
        lp(ax, 'A', x=-.14, y=1.10)
        ax.set_title('six units in a 150 cm arena, with their grid scores',
                     fontsize=7.6, loc='left', color='0.25')

# ---- B: one unit on the track, with the gate closing halfway ---------------
ax = fig.add_subplot(gs[0, 1])
lp(ax, 'B', x=-.26)
HALF = int(Z['sw_half'])
SM = Z['sw_map']
ax.imshow(SM / np.nanmax(SM), aspect='auto', cmap='viridis',
          interpolation='nearest', extent=[0, TRACK, len(SM), 0])
ax.axhline(HALF, color='w', lw=1.4)
ax.text(TRACK * .97, HALF - .5, 'gate closes', fontsize=5.8, color='w',
        ha='right', va='bottom')
ax.set_xlim(0, TRACK); ax.set_xticks([0, 100, 200])
ax.set_xlabel('position (cm)', fontsize=7.5)
ax.set_ylabel('lap', fontsize=7.5)
ax.tick_params(labelsize=6.5)
ax.set_title('the same unit on the track', fontsize=7.6, loc='left',
             color='0.25')

# ---- C: the population state, as Figure 1 measures it ----------------------
ax = fig.add_subplot(gs[0, 2])
lp(ax, 'C', x=-.22)
L_ = Z['sw_labels']
L_ = L_[~np.isnan(L_).all(1)][np.argsort(-Z['sw_load'])]
ax.imshow(L_.T, aspect='auto', cmap=ListedColormap([NON, ANCH]),
          norm=BoundaryNorm([-.5, .5, 1.5], 2), interpolation='nearest',
          extent=[0, L_.shape[0], L_.shape[1], 0])
ax.axhline(HALF, color='k', lw=1.0)
ax.set_xticks([]); ax.tick_params(labelsize=6.5, labelleft=False)
ax.set_xlabel(f'{L_.shape[0]} units', fontsize=7.5)
ax.set_title('the population state', fontsize=7.6, loc='left', color='0.25')
axq = ax.inset_axes([1.06, 0, .30, 1], sharey=ax)
pc1_ = Z['sw_pc1']; tr_ = np.arange(len(pc1_))
axq.fill_betweenx(tr_, 0, pc1_, where=pc1_ >= 0, color=ANCH, lw=0,
                  edgecolor='none', interpolate=True)
axq.fill_betweenx(tr_, 0, pc1_, where=pc1_ < 0, color=NON, lw=0,
                  edgecolor='none', interpolate=True)
axq.axvline(0, color='0.4', lw=.5); axq.axhline(HALF, color='k', lw=1.0)
axq.set_xticks([]); axq.tick_params(labelleft=False, length=0)
for sp in axq.spines.values():
    sp.set_visible(False)
axq.set_xlabel('PC1', fontsize=7.0)

# ---- D: the two states, as Figure 8E draws them ----------------------------
GD = gs[0, 3].subgridspec(2, 1, hspace=.14, height_ratios=[1.0, .95])
axs = fig.add_subplot(GD[0])
for k, col, lab in (('open', ANCH, 'gate open'), ('shut', NON, 'gate shut')):
    axs.plot(Z['sp_f'], Z[f'rel_{k}'], color=col, lw=1.1, label=lab)
axs.set_xscale('log'); axs.set_xlim(2, 120)
axs.tick_params(labelbottom=False, labelsize=6.5)
axs.set_ylabel('spectrum', fontsize=6.8, labelpad=2)
axs.legend(fontsize=5.6, frameon=False, loc='upper right', handlelength=1.1,
           handletextpad=.4, borderaxespad=.2, labelspacing=.25)
axs.spines[['top', 'right']].set_visible(False)
axs.set_title('the two states, as spectra\nand as deviations', fontsize=7.6,
              loc='left', color='0.25')
lp(axs, 'D', x=-.15)

ax = fig.add_subplot(GD[1], sharex=axs)
for lo, hi in ((6, 10), (30, 48), (60, 100)):
    ax.axvspan(lo, hi, color='0.88', alpha=.6, linewidth=0, zorder=0)
ax.axhline(0, color='0.6', lw=.7, ls=':', zorder=1)
for k, col in (('open', ANCH), ('shut', NON)):
    ax.plot(Z['sp_f'], Z[f'z_{k}'], color=col, lw=1.2)
ax.set_xlim(2, 120); ax.set_xticks([2, 8, 20, 50, 100])
ax.set_xticklabels(['2', '8', '20', '50', '100'], fontsize=6.5)
ax.xaxis.set_minor_formatter(plt.NullFormatter())
ax.set_xlabel('frequency (Hz)', fontsize=7.5)
ax.set_ylabel('deviation (z)', fontsize=7.0, labelpad=2)
ax.tick_params(labelsize=6.5); ax.spines[['top', 'right']].set_visible(False)

# ---- E: the bands, measured against the sheet and the ring -----------------
ax = fig.add_subplot(gs[1, 0])
lp(ax, 'E', x=-.20)
R1 = np.load(f'{FIG}/can_gate_model.npz')
w = .26
for j, (nm, lo, hi) in enumerate(BANDS):
    g = measured(lo, hi)
    se = g.std() / np.sqrt(len(g))
    p = wilcoxon(g).pvalue
    ax.bar(j - w, g.mean(), w, color='#444', lw=0,
           label='recordings' if not j else None)
    ax.plot([j - w] * 2, [g.mean() - se, g.mean() + se], color='0.25', lw=1)
    m1 = (R1['f'] >= lo) & (R1['f'] <= hi)
    ax.bar(j, float(R1['d'][m1].mean()), w, color='0.72', lw=0,
           label='ring (S13)' if not j else None)
    ax.bar(j + w, float(Z['band_d'][j]), w, color=ACH, lw=0,
           label='sheet' if not j else None)
    ax.annotate(f'{p:.0e}'.replace('e-0', 'e-') if p < .01 else f'{p:.2f}',
                (j - w, .015), xycoords=('data', 'axes fraction'),
                ha='center', va='bottom', fontsize=5.0, color='0.35')
ax.axhline(0, color='k', lw=.9)
ax.set_yscale('symlog', linthresh=.02, linscale=.55)
ax.set_yticks([-.2, -.05, -.02, 0, .02, .05])
ax.set_yticklabels(['-0.2', '-0.05', '-0.02', '0', '0.02', '0.05'],
                   fontsize=5.8)
ax.set_xticks(range(len(BANDS)))
ax.set_xticklabels([b[0] for b in BANDS], fontsize=6.2)
ax.set_ylabel('anchored $-$ non-anchored\n(log10 power)', fontsize=6.8)
ax.legend(fontsize=5.6, frameon=False, loc='lower left', handlelength=.9,
          handletextpad=.4, borderaxespad=.2)
ax.set_title('the sheet gets the sizes the ring\noverstated', fontsize=7.6,
             loc='left', color='0.25')
bare(ax)

# ---- F: theta-gamma coupling, both models against the recordings -----------
ax = fig.add_subplot(gs[1, 1])
lp(ax, 'F', x=-.26)
mi = Z['mi']
sheet = [100 * (mi[0] - mi[1]) / mi[1], 100 * (mi[2] - mi[3]) / mi[3]]
ring = [16.4, -24.8]
meas = [0.0, -15.9]
for j, (nm, mm, rr, ss) in enumerate(zip(
        ['theta–slow', 'theta–fast'], meas, ring, sheet)):
    ax.bar(j - w, mm, w, color='#444', lw=0)
    ax.bar(j, rr, w, color='0.72', lw=0)
    ax.bar(j + w, ss, w, color=ACH, lw=0)
ax.axhline(0, color='k', lw=.9)
ax.set_xticks(range(2))
ax.set_xticklabels(['theta–slow', 'theta–fast'], fontsize=6.2)
ax.set_ylabel('coupling change (%)', fontsize=7.0)
ax.text(0, .04, 'measured theta-slow is unchanged', transform=ax.transAxes,
        fontsize=5.4, color='0.45')
ax.set_title('coupling: neither model gets\nboth', fontsize=7.6, loc='left',
             color='0.25')
bare(ax)

# ---- G: the prediction, in the sheet ---------------------------------------
ax = fig.add_subplot(gs[1, 2])
lp(ax, 'G', x=-.26)
ax.plot(range(1, 11), Z['dec'], color='#e8701a', marker='o', ms=3, lw=1.2)
ax.set_xlabel('decile of |mismatch|,\ngate shut', fontsize=7.2)
ax.set_ylabel('fast-$\\gamma$ amplitude', fontsize=7.2)
ax.set_title(f'the prediction, r = {float(Z["pred_r"]):+.3f}', fontsize=7.6,
             loc='left', color='0.25')
bare(ax)

# ---- H: what the sheet settles --------------------------------------------
ax = fig.add_subplot(gs[1, 3]); ax.axis('off')
ax.set_xlim(0, 1); ax.set_ylim(0, 1)
lp(ax, 'H', x=-.06)
ax.set_title('what the sheet adds, and what it\ncosts', fontsize=7.6,
             loc='left', color='0.25')
ROWS = [('units are grid cells, median score '
         f'{np.nanmedian(Z["gs_all"]):.2f}', True),
        ('anchoring switches with the gate', True),
        ('theta falls with the state', True),
        ('slow gamma rises', None),
        ('fast gamma falls, right size', True),
        ('theta–fast coupling falls', False),
        ('fast gamma tracks the mismatch', False)]
for i, (nm, ok) in enumerate(ROWS):
    y = .95 - i * .098
    if ok:
        ax.plot(.05, y, marker='o', ms=4.0, color='#1b7a3d', mew=0)
    elif ok is False:
        ax.plot(.05, y, marker='x', ms=4.4, mew=1.5, color='#8a1c1c')
    else:
        ax.plot(.05, y, marker='_', ms=5.0, mew=1.4, color='0.55')
    ax.text(.13, y, nm, fontsize=5.9, color='0.3', va='center')
ax.text(0, .22, "the error population's pool averages over 1296 units,\n"
                "not 128, and never reaches the strongly oscillating\n"
                "regime the ring has; that regime is what carried the\n"
                "coupling and the prediction in Figure S13",
        fontsize=5.1, color='0.5', va='top', linespacing=1.4)

fig.savefig(OUT, dpi=300, bbox_inches='tight')
print(f'wrote {OUT}')
print(f'  bands (sheet): ' + ' '.join(f'{b[0][:5]} {v:+.3f}'
      for b, v in zip(BANDS, Z['band_d'])))
print(f'  coupling (sheet): theta-slow {sheet[0]:+.1f}%  '
      f'theta-fast {sheet[1]:+.1f}%   prediction r {float(Z["pred_r"]):+.3f}')
