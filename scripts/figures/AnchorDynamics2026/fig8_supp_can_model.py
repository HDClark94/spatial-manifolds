"""Supplement to Figure 9: the gated attractor as a ring and as a 2-D sheet.

The circuit of Figure 9, built twice. First as a ring carrying one module's
phase along the track, which is enough for the field potential but assumes
what a grid model ought to produce. Then as a two-dimensional sheet whose
units fire in hexagonal grids in an open arena, driven along a line through
that arena and teleported back to the start each lap, which is what a virtual
linear track is. The teleport is what gives the gate work to do: the animal's
position jumps back and its path integrator does not.

In both, the gate is shut, then open, then shut again, as the recordings'
population state comes in blocks rather than one switch. The last column
compares the two models against the recordings.

The simulations are can_gate_model.py and can_gate_model_2d.py; this script
only draws them.

Writes fig8_supp_can_model.pdf
"""
import importlib.util
import os

import matplotlib
import numpy as np
import pandas as pd

matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.colors import BoundaryNorm, ListedColormap
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
LAND_C, SELF_C, ACH = '#7a6a53', '#6b7f8c', '#b8860b'
ATTR_C, ERR_C = '#2f7d32', '#8a6bbf'   # attractor, error
RING_C, SHEET_C, DATA_C = '0.72', ACH, '#444'
TRACK = 200.0

def _model(name, fn):
    """The simulation itself, for its geometry. Both are __main__-guarded,
    so importing one defines its constants and runs nothing."""
    sp = importlib.util.spec_from_file_location(name, f'{FIG}/{fn}')
    m = importlib.util.module_from_spec(sp); sp.loader.exec_module(m)
    return m


MOD1 = _model('cgm_ring', 'can_gate_model.py')
MOD2 = _model('cgm_sheet', 'can_gate_model_2d.py')

Z = np.load(f'{FIG}/can_gate_model.npz')
Z2 = np.load(f'{FIG}/can_gate_model_2d.npz')
# the same sheet run again with three times the noise, to ask whether the
# model's near-perfect state classification is a property of the mechanism or
# of a noiseless simulation
_NZ = f'{FIG}/can_gate_model_2d_noisy.npz'
Z3 = np.load(_NZ) if os.path.exists(_NZ) else None
R = pd.read_csv(f'{ROOT}/data/lfp/gamma_vs_decode_error.csv')

# the measured band changes, in the model's units: a difference of log10 power
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


# what each model does to theta-gamma coupling, and to the prediction
RING_MI = [16.4, -24.8]
_mi = Z2['mi']
SHEET_MI = [100 * (_mi[0] - _mi[1]) / _mi[1], 100 * (_mi[2] - _mi[3]) / _mi[3]]
MEAS_MI = [0.0, -15.9]
RING_PRED = float(np.corrcoef(Z['mism'], Z['ampfast'])[0, 1])
# where in the theta cycle each band peaks (Figure S10 measures this)
RING_PH = [197.0, 20.0]
SHEET_PH = [float(Z2['ph'][0]), float(Z2['ph'][2])]
MEAS_PH = [110.0, 41.0]
SHEET_PRED = float(Z2['pred_r'])

fig = plt.figure(figsize=(9.7, 9.4))
gs = fig.add_gridspec(4, 4, hspace=.66, wspace=.58, left=.055, right=.985,
                      top=.962, bottom=.047,
                      width_ratios=[1.15, 1.0, .95, 1.05],
                      height_ratios=[.86, 1.0, 1.0, 1.06])


def lp(ax, s, x=-.26, y=1.06):
    ax.text(x, y, s, transform=ax.transAxes, fontsize=10, weight='bold',
            va='bottom', ha='right')


def bare(ax):
    ax.tick_params(labelsize=6.5)
    ax.spines[['top', 'right']].set_visible(False)


def blocks(ax, edges, n, horizontal=True, c='k'):
    for e in edges:
        (ax.axhline if horizontal else ax.axvline)(e, color=c, lw=1.0)


def example_unit(src):
    """Which unit B and F draw: the one the model's own PC1 loads on most."""
    L = src['sw_labels']
    use = ~np.isnan(L).all(1)
    return int(np.where(use)[0][int(np.argmax(src['sw_load']))])


def ratemap_panel(ax, M, edges, title, labels=None):
    ax.imshow(M / np.nanmax(M), aspect='auto', cmap='viridis',
              interpolation='nearest', extent=[0, TRACK, len(M), 0])
    for e in edges:
        ax.axhline(e, color='w', lw=1.3)
    ax.text(TRACK * .97, edges[0] - .5, 'gate opens', fontsize=5.6, color='w',
            ha='right', va='bottom')
    ax.text(TRACK * .97, edges[1] - .5, 'and shuts', fontsize=5.6, color='w',
            ha='right', va='bottom')
    ax.set_xlim(0, TRACK); ax.set_xticks([0, 100, 200])
    ax.set_xlabel('position (cm)', fontsize=7.5)
    ax.set_ylabel('lap', fontsize=7.5)
    ax.tick_params(labelsize=6.5)
    ax.set_title(title, fontsize=7.6, loc='left', color='0.25')
    if labels is not None:
        # what the classifier makes of this one cell, lap by lap
        q = ax.inset_axes([1.03, 0, .10, 1], sharey=ax)
        v = np.where(np.isnan(labels), np.nan, labels)[:, None]
        q.imshow(v, aspect='auto', cmap=ListedColormap([NON, ANCH]),
                 norm=BoundaryNorm([-.5, .5, 1.5], 2),
                 interpolation='nearest', extent=[0, 1, len(labels), 0])
        for e in edges:
            q.axhline(e, color='w', lw=1.0)
        q.set_xticks([]); q.tick_params(labelleft=False, length=0)
        q.text(.5, 1.01, 'call', transform=q.transAxes, fontsize=6.0,
               ha='center', va='bottom', color='0.35')
        for sp in q.spines.values():
            sp.set_linewidth(.4)


def state_panel(ax, L, load, pc1, edges, title):
    L = L[~np.isnan(L).all(1)][np.argsort(-load)]
    ax.imshow(L.T, aspect='auto', cmap=ListedColormap([NON, ANCH]),
              norm=BoundaryNorm([-.5, .5, 1.5], 2), interpolation='nearest',
              extent=[0, L.shape[0], L.shape[1], 0])
    for e in edges:
        ax.axhline(e, color='k', lw=1.0)
    ax.set_xticks([]); ax.tick_params(labelsize=6.5, labelleft=False)
    ax.set_xlabel(f'{L.shape[0]:,} units', fontsize=7.5)
    ax.set_title(title, fontsize=7.6, loc='left', color='0.25')
    q = ax.inset_axes([1.07, 0, .32, 1], sharey=ax)
    tr = np.arange(len(pc1))
    q.fill_betweenx(tr, 0, pc1, where=pc1 >= 0, color=ANCH, lw=0,
                    edgecolor='none', interpolate=True)
    q.fill_betweenx(tr, 0, pc1, where=pc1 < 0, color=NON, lw=0,
                    edgecolor='none', interpolate=True)
    q.axvline(0, color='0.4', lw=.5)
    for e in edges:
        q.axhline(e, color='k', lw=1.0)
    q.set_xticks([]); q.tick_params(labelleft=False, length=0)
    for sp in q.spines.values():
        sp.set_visible(False)
    q.set_title('PC1', fontsize=6.6, color='0.35', pad=2.5)


def ring_geometry(ax):
    """What the ring is made of: 128 units carrying one module's phase."""
    n = MOD1.N
    t = np.arange(n) * 2 * np.pi / n
    act = np.exp(2.2 * (np.cos(t - np.pi / 2) - 1.0))   # the model's kernel
    ax.add_patch(plt.Circle((0, 0), 1.0, fill=False, ec='0.82', lw=.7,
                            zorder=1))
    ax.scatter(np.cos(t), np.sin(t), c=act, cmap='viridis', s=7.0, lw=0,
               zorder=3)
    _r = 1.20
    ax.add_patch(FancyArrowPatch((_r * np.cos(2.2), _r * np.sin(2.2)),
                                 (_r * np.cos(1.05), _r * np.sin(1.05)),
                                 connectionstyle='arc3,rad=-.26',
                                 arrowstyle='-|>', mutation_scale=8,
                                 color=ACH, lw=1.1, zorder=4))
    ax.text(-1.32, 1.20, 'running moves\nthe bump', fontsize=5.5, color=ACH,
            ha='center', va='center', linespacing=1.3)
    ax.text(0, .08, f'{n} units', fontsize=6.4, color='0.3', ha='center')
    ax.text(0, -.22, f'one cycle = {MOD1.LAM:.0f} cm\nof track',
            fontsize=5.6, color='0.45', ha='center', va='center',
            linespacing=1.35)
    ax.set_xlim(-1.75, 1.6); ax.set_ylim(-1.45, 1.75)
    ax.set_aspect('equal'); ax.axis('off')
    ax.set_title('the ring: one module\u2019s phase', fontsize=7.6, loc='left',
                 color='0.25')


def sheet_geometry(ax):
    """What the sheet is made of: 1,296 units on the lattice's unit cell."""
    ns = MOD2.NS
    i = np.arange(ns) * 2 * np.pi / ns
    p1, p2 = np.meshgrid(i, i, indexing='ij')
    # hexkern falls off as a^2 + ab + b^2, so the coordinates in which the
    # bump is round are s1 = a + b/2, s2 = (sqrt3/2) b. The minus sign of the
    # Gardner transform belongs to axes 60 degrees apart, not 120.
    s1, s2 = p1 + p2 / 2.0, (np.sqrt(3) / 2.0) * p2
    ax.scatter(s1, s2, c=MOD2.hexkern(p1 - 2.9, p2 - 2.4), cmap='viridis',
               s=2.8, lw=0, zorder=3)
    u = np.array([2 * np.pi, 0.0])
    v = np.array([np.pi, np.sqrt(3) * np.pi])
    cor = np.array([[0, 0], u, u + v, v, [0, 0]], float)
    ax.plot(cor[:, 0], cor[:, 1], color='0.78', lw=.7, zorder=1)
    for vec, nm, off in ((u, r'$\theta_1$', (0.0, -1.15)),
                         (v, r'$\theta_2$', (-1.05, 0.25))):
        ax.add_patch(FancyArrowPatch((0, 0), tuple(vec * .40),
                                     arrowstyle='-|>', mutation_scale=6,
                                     color='0.3', lw=.95, zorder=4))
        ax.text(*(vec * .47 + np.array(off)), nm, fontsize=6.2,
                color='0.3', ha='center', va='center')
    ax.text(np.pi, -1.5, f'{ns * ns:,} units on one cell\nof the lattice; '
            f'{MOD2.LAM:.0f} cm across', fontsize=5.6, color='0.45',
            ha='center', va='top', linespacing=1.35)
    ax.set_xlim(-1.2, 3 * np.pi + 1.0)
    ax.set_ylim(-3.6, np.sqrt(3) * np.pi + 1.0)
    ax.set_aspect('equal'); ax.axis('off')
    ax.set_title('the sheet: a hexagonal lattice', fontsize=7.6, loc='left',
                 color='0.25')


def path_panel(cell):
    """The repeated run that builds the fields, and what one unit fires."""
    g = cell.subgridspec(2, 1, hspace=.46, height_ratios=[1.25, 1.0])
    a0, a1 = fig.add_subplot(g[0]), fig.add_subplot(g[1])
    hd = np.array([np.cos(MOD2.TRACK_ANGLE), np.sin(MOD2.TRACK_ANGLE)])
    x0 = np.array([12.0, 9.0])
    phi = np.array([2.1, 0.7])                 # one unit's offsets
    xs = np.linspace(-6, 216, 440); ys = np.linspace(-10, 64, 150)
    P = np.stack(np.meshgrid(xs, ys, indexing='xy'), -1) @ MOD2.KMAT.T
    a0.imshow(MOD2.hexkern(phi[0] - P[..., 0], phi[1] - P[..., 1]),
              origin='lower', extent=[xs[0], xs[-1], ys[0], ys[-1]],
              cmap='viridis', aspect='equal', interpolation='bilinear')
    d = np.linspace(0, TRACK, 700)
    pts = x0 + d[:, None] * hd
    a0.plot(pts[:, 0], pts[:, 1], color='w', lw=1.3)
    a0.add_patch(FancyArrowPatch(tuple(pts[-1]), tuple(x0),
                                 connectionstyle='arc3,rad=-.22',
                                 arrowstyle='-|>', mutation_scale=7,
                                 color='w', lw=.9, ls=':'))
    a0.text(104, 61, 'teleported back, every lap', fontsize=5.4, color='w',
            ha='center', va='top')
    a0.set_xticks([]); a0.set_yticks([])
    for sp in a0.spines.values():
        sp.set_visible(False)
    lp(a0, 'D', x=-.13, y=1.06)
    a0.set_title('the run that builds the fields', fontsize=7.6, loc='left',
                 color='0.25')
    # the same path on three successive laps, with the gate shut: the
    # integrator is not reset by the teleport, so the phase has moved on
    step = MOD2.KMAT @ (TRACK * hd)
    Pp = pts @ MOD2.KMAT.T
    for k in range(3):
        r = MOD2.hexkern(phi[0] - Pp[:, 0] - k * step[0],
                         phi[1] - Pp[:, 1] - k * step[1])
        a1.fill_between(d, 2 - k, 2 - k + .92 * r / r.max(), lw=0,
                        edgecolor='none', color=ATTR_C, alpha=.85)
        a1.text(-6, 2.4 - k, f'lap {k + 1}', fontsize=5.4, color='0.4',
                ha='right', va='center')
    a1.set_xlim(0, TRACK); a1.set_xticks([0, 100, 200])
    a1.set_ylim(-.15, 3.1); a1.set_yticks([])
    a1.set_xlabel('position (cm)', fontsize=7.5)
    a1.tick_params(labelsize=6.5)
    a1.spines[['top', 'right', 'left']].set_visible(False)
    a1.set_title('one unit along it, gate shut', fontsize=6.8, loc='left',
                 color='0.45')


def spectra_panel(cell, src, letter, title):
    g = cell.subgridspec(2, 1, hspace=.14, height_ratios=[1.0, .95])
    a0 = fig.add_subplot(g[0])
    for k, col, lab in (('open', ANCH, 'gate open'), ('shut', NON, 'gate shut')):
        a0.plot(src['sp_f'], src[f'rel_{k}'], color=col, lw=1.1, label=lab)
    a0.set_xscale('log'); a0.set_xlim(2, 120)
    a0.tick_params(labelbottom=False, labelsize=6.5)
    a0.set_ylabel('spectrum', fontsize=6.8, labelpad=2)
    a0.legend(fontsize=5.5, frameon=False, loc='upper right', handlelength=1.0,
              handletextpad=.4, borderaxespad=.2, labelspacing=.22)
    a0.spines[['top', 'right']].set_visible(False)
    a0.set_title(title, fontsize=7.6, loc='left', color='0.25')
    lp(a0, letter, x=-.24)
    a1 = fig.add_subplot(g[1], sharex=a0)
    for lo, hi in ((6, 10), (30, 48), (60, 100)):
        a1.axvspan(lo, hi, color='0.88', alpha=.6, linewidth=0, zorder=0)
    a1.axhline(0, color='0.6', lw=.7, ls=':', zorder=1)
    for k, col in (('open', ANCH), ('shut', NON)):
        a1.plot(src['sp_f'], src[f'z_{k}'], color=col, lw=1.2)
    a1.set_xlim(2, 120); a1.set_xticks([2, 8, 20, 50, 100])
    a1.set_xticklabels(['2', '8', '20', '50', '100'], fontsize=6.5)
    a1.xaxis.set_minor_formatter(plt.NullFormatter())
    a1.set_xlabel('frequency (Hz)', fontsize=7.5)
    a1.set_ylabel('deviation (z)', fontsize=7.0, labelpad=2)
    a1.tick_params(labelsize=6.5)
    a1.spines[['top', 'right']].set_visible(False)


# ======================= row 1: the ring ====================================
# ---- A: the circuit ---------------------------------------------------------
ax = fig.add_subplot(gs[0, 0]); ax.axis('off')
ax.set_xlim(0, 1); ax.set_ylim(0, 1)
lp(ax, 'A', x=-.10)
ax.set_title('what was simulated', fontsize=7.6, loc='left', color='0.25')
for x_, y_, w_, h_, nm, fc, tc in (
        (.02, .70, .44, .17, 'landmark\nphase', '#f4f1ec', LAND_C),
        (.02, .13, .44, .17, 'error\nfast inhibition', '#f3effa', ERR_C),
        (.54, .40, .44, .20, 'attractor\nslow inhibition', '#eef5ee', ATTR_C)):
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
ax.text(.0, -.09, 'the gate scales only how much the error is allowed to\n'
                  'correct the attractor; the landmark input arrives in both\n'
                  'states. On the sheet the error moves the bump, not its input',
        fontsize=5.4, color='0.45', va='top', linespacing=1.35)


ax = fig.add_subplot(gs[0, 1]); lp(ax, 'B', x=-.14, y=1.02)
ring_geometry(ax)

ax = fig.add_subplot(gs[0, 2]); lp(ax, 'C', x=-.12, y=1.02)
sheet_geometry(ax)

path_panel(gs[0, 3])

# ======================= row 2: one unit, each way ==========================
E1 = Z['sw_edges']
B1 = example_unit(Z)
ax = fig.add_subplot(gs[1, 0]); lp(ax, 'E', x=-.24)
ratemap_panel(ax, Z['sw_map'], E1, 'a ring unit, and its call',
              labels=Z['sw_labels'][B1])

ax = fig.add_subplot(gs[1, 1]); lp(ax, 'F', x=-.22)
state_panel(ax, Z['sw_labels'], Z['sw_load'], Z['sw_pc1'], E1,
            'the ring population state')

GA = gs[1, 2].subgridspec(2, 3, hspace=.26, wspace=.12)
M2, GSC = Z2['maps'], Z2['grid_score']
pick = np.argsort(-GSC)[np.linspace(0, len(GSC) // 3, 6).astype(int)]
for k, idx in enumerate(pick):
    a = fig.add_subplot(GA[k // 3, k % 3])
    a.imshow(M2[idx].T, origin='lower', cmap='viridis',
             interpolation='bilinear')
    a.set_xticks([]); a.set_yticks([])
    a.text(.5, -.03, f'{GSC[idx]:.2f}', transform=a.transAxes, fontsize=5.2,
           color='0.35', ha='center', va='top')
    if k == 0:
        lp(a, 'G', x=-.16, y=1.12)
        a.set_title('sheet units in a 150 cm arena,\nwith grid scores',
                    fontsize=7.6, loc='left', color='0.25')

E2 = Z2['sw_edges']
B2 = example_unit(Z2)
ax = fig.add_subplot(gs[1, 3]); lp(ax, 'H', x=-.24)
ratemap_panel(ax, Z2['sw_map'], E2, 'a sheet unit, and its call',
              labels=Z2['sw_labels'][B2])

# ================ row 3: the sheet population, and the summaries ============
if Z3 is None:
    ax = fig.add_subplot(gs[2, 0]); lp(ax, 'I', x=-.22)
    state_panel(ax, Z2['sw_labels'], Z2['sw_load'], Z2['sw_pc1'], E2,
                'the sheet population state')
else:
    GG = gs[2, 0].subgridspec(1, 2, wspace=.95)
    ax = fig.add_subplot(GG[0]); lp(ax, 'I', x=-.30)
    state_panel(ax, Z2['sw_labels'], Z2['sw_load'], Z2['sw_pc1'], E2,
                'as simulated')
    ax2 = fig.add_subplot(GG[1])
    state_panel(ax2, Z3['sw_labels'], Z3['sw_load'], Z3['sw_pc1'],
                Z3['sw_edges'], '3x the noise')

# ---- J: the bands, all three -----------------------------------------------
ax = fig.add_subplot(gs[2, 1]); lp(ax, 'J', x=-.24)
w = .26
for j, (nm, lo, hi) in enumerate(BANDS):
    g = measured(lo, hi)
    se = g.std() / np.sqrt(len(g))
    p = wilcoxon(g).pvalue
    m1 = (Z['f'] >= lo) & (Z['f'] <= hi)
    ax.bar(j - w, g.mean(), w, color=DATA_C, lw=0,
           label='recordings' if not j else None)
    ax.plot([j - w] * 2, [g.mean() - se, g.mean() + se], color='0.25', lw=1)
    ax.bar(j, float(Z['d'][m1].mean()), w, color=RING_C, lw=0,
           label='ring' if not j else None)
    ax.bar(j + w, float(Z2['band_d'][j]), w, color=SHEET_C, lw=0,
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
ax.set_ylabel('anchored $-$ non (log10)', fontsize=6.8, labelpad=1)
ax.legend(fontsize=5.6, frameon=False, loc='upper right', handlelength=.9,
          handletextpad=.4, borderaxespad=.2)
ax.set_title('band power', fontsize=7.6, loc='left', color='0.25')
bare(ax)

# ---- K: theta-gamma coupling, all three ------------------------------------
ax = fig.add_subplot(gs[2, 2]); lp(ax, 'K', x=-.24)
for j, (mm, rr, ss) in enumerate(zip(MEAS_MI, RING_MI, SHEET_MI)):
    ax.bar(j - w, mm, w, color=DATA_C, lw=0)
    ax.bar(j, rr, w, color=RING_C, lw=0)
    ax.bar(j + w, ss, w, color=SHEET_C, lw=0)
ax.axhline(0, color='k', lw=.9)
ax.set_xticks(range(2))
ax.set_xticklabels(['theta–slow', 'theta–fast'], fontsize=6.2)
ax.set_ylabel('coupling change (%)', fontsize=7.0)
ax.text(.02, .04, 'measured theta–slow is unchanged', transform=ax.transAxes,
        fontsize=5.3, color='0.45')
ax.set_title('theta–gamma coupling', fontsize=7.6, loc='left', color='0.25')
bare(ax)

# ======================= row 3: spectra, and the prediction =================
SG = gs[3, :].subgridspec(1, 3, wspace=.34)
spectra_panel(SG[0], Z, 'M', 'the ring: two states, as spectra\nand as deviations')
spectra_panel(SG[1], Z2, 'N', 'the sheet: the same')

# ---- K: the noisy sheet, drawn as I and J are -----------------------------
if Z3 is not None:
    spectra_panel(SG[2], Z3, 'O', 'the sheet with 3x the noise:\nthe same')
else:
    ax = fig.add_subplot(SG[2]); lp(ax, 'O', x=-.22)
    ax.axis('off')

# ---- L: where in the theta cycle each band peaks (Figure S10) --------------
ax = fig.add_subplot(gs[2, 3], projection='polar')
lp(ax, 'L', x=-.18, y=1.02)
for nm, vals, col, mk in (('recordings', MEAS_PH, DATA_C, 'o'),
                          ('ring', RING_PH, RING_C, 's'),
                          ('sheet', SHEET_PH, SHEET_C, '^')):
    for k, (b, v) in enumerate(zip(('slow', 'fast'), vals)):
        ax.plot(np.radians(v), 1.0 - .22 * k, marker=mk, ms=5.5, color=col,
                lw=0, label=nm if k == 0 else None,
                mfc=col if b == 'slow' else 'w', mec=col, mew=1.2)
ax.set_rlim(0, 1.15); ax.set_yticks([])
ax.set_xticks(np.radians([0, 90, 180, 270]))
ax.set_xticklabels(['0', '90', '180', '270'], fontsize=6.0)
ax.tick_params(pad=-2)
ax.grid(alpha=.35, lw=.5)
ax.legend(fontsize=5.4, frameon=False, loc='lower center',
          bbox_to_anchor=(.5, -.30), ncol=3, handlelength=.8,
          columnspacing=.8, handletextpad=.3)
ax.set_title('theta phase of each band\n(filled slow, open fast)',
             fontsize=7.6, color='0.25', pad=8)
ax.text(.5, -.34, 'measured, the two sit 40 deg apart;\nboth models put them '
        'antiphase', transform=ax.transAxes, fontsize=5.4, color='0.45',
        ha='center', va='top')

fig.savefig(OUT, dpi=300, bbox_inches='tight')
print(f'wrote {OUT}')
print(f'  ring  bands ' + ' '.join(
    f'{b[0][:5]} {float(Z["d"][(Z["f"] >= b[1]) & (Z["f"] <= b[2])].mean()):+.3f}'
    for b in BANDS) + f' | coupling {RING_MI} | pred {RING_PRED:+.3f}')
print(f'  sheet bands ' + ' '.join(f'{b[0][:5]} {v:+.3f}'
      for b, v in zip(BANDS, Z2['band_d'])) +
      f' | coupling {[round(x, 1) for x in SHEET_MI]} | pred {SHEET_PRED:+.3f}')
print(f'  theta phase of band peak (slow, fast): recordings {MEAS_PH}, '
      f'ring {RING_PH}, sheet {[round(x) for x in SHEET_PH]}')
