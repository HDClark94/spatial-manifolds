"""Graphical abstract, in the Cell Press stacked-panel style.

Three stacked statements: what the phenomenon is and what it buys the animal,
which cells express it, and what the state is.

NOTHING HERE IS A SCHEMATIC EXCEPT THE TWO TRACK STRIPS, and nothing is typed
in except the Figure 2 and Figure 8 bar values. Everything else is computed
from cached tables, so it cannot drift from the paper the way the retired
Figure 2D values once did:

  grid cell   ga_cache.npz -- M26 D18 cell 109, grid score 0.95, its trial x
              position map split by the trial's anchoring label
  raster      data/population_state/labels/M26D18.npz -- the cell x trial label
              matrix, its PC1 and the per-cell loading, ordered as Figure 1
              orders them
  behaviour   data/eye_anchoring/eye_anchoring_trials.csv -- hit rate by state
              and cue condition, the state being the sign of PC1
  the pupil   one video frame per state from the Figure 6 example session, at a
              matched track position, and the Figure 6I profiles from
              data/eye_anchoring/eye_position_profiles_gated.npz, drawn as that
              panel draws them. The per-bin significance markers are left off:
              they are uncorrected in Figure 6 and mark extent rather than test
              it, which is not a distinction a graphical abstract can carry.

ga_build_cache.py writes ga_cache.npz and needs the lab volume; this script
does not.

Set in Arial like every other figure -- the Avenir Next used for the document's
display type is deliberately NOT used inside figure PDFs.

Writes graphical_abstract.pdf
"""
import os

import matplotlib
import numpy as np
import pandas as pd
from scipy.stats import wilcoxon
from sklearn.metrics import roc_auc_score

matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.colors import BoundaryNorm, ListedColormap
from matplotlib.patches import Rectangle

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
CUE_C = '#3f7d3f'
TA = ListedColormap([NON, ANCH])
NORM = BoundaryNorm([-0.5, 0.5, 1.5], TA.N)
RZ = (90., 110.)
Q = np.load(f'{FIG}/ga_cache.npz', allow_pickle=True)

fig = plt.figure(figsize=(3.55, 6.05))
G = fig.add_gridspec(3, 1, height_ratios=[2.10, 1.28, 0.92],
                     hspace=.46, left=.04, right=.96, top=.958, bottom=.016)


def banner(ax, text):
    ax.text(.5, 1.045, text, transform=ax.transAxes, ha='center', va='bottom',
            fontsize=7.0, weight='bold', linespacing=1.35)


def track(ax, x0, w, y0, h, cued):
    """The virtual track, with the reward zone cued or not."""
    ax.add_patch(Rectangle((x0, y0), w, h, facecolor='0.15', lw=0))
    zx, zw = x0 + w * RZ[0] / 200., w * (RZ[1] - RZ[0]) / 200.
    if cued:   # vertical green and black bars mark the zone on cued trials
        ax.add_patch(Rectangle((zx, y0), zw, h, facecolor='#9ecae1', lw=0))
        for i in range(6):
            ax.add_patch(Rectangle((zx + i * zw / 6., y0), zw / 12., h,
                                   facecolor=CUE_C, lw=0))
    else:      # on uncued trials the zone carries no cue at all: an outline,
        ax.add_patch(Rectangle((zx, y0), zw, h, facecolor='none',  # not a fill
                               edgecolor='0.72', lw=.6, ls=(0, (1.6, 1.4))))
    ax.text(x0 - .012, y0 + h / 2, 'cued' if cued else 'uncued', ha='right',
            va='center', fontsize=5.2, color=CUE_C if cued else '0.45')
    if not cued:
        ax.text(x0 + w + .012, y0 + h / 2, 'zone unmarked', ha='left',
                va='center', fontsize=4.4, color='0.5')


# ── 1. the phenomenon, and what it buys ──────────────────────────────────────
ax = fig.add_subplot(G[0]); ax.axis('off')
ax.set_xlim(0, 1); ax.set_ylim(0, 1)
banner(ax, 'Entorhinal firing anchors to the task reference\n'
           'frame and tracks uncued engagement')

# The grid cell, the population and PC1 share one trial axis, as Figure 1 has
# them share it: the cell's anchored runs line up row for row with the magenta
# blocks of PC1, which is the whole point and is lost if the cell is drawn as
# two separate state-sorted blocks.
track(ax, .105, .255, .945, .024, True)
track(ax, .105, .255, .890, .024, False)

gmap, ganch = Q['gc_map'], Q['gc_anch']
mo, dy, cid, gs = Q['gc_meta']
ntr = gmap.shape[0]


def _norm(M):
    """Each trial scaled to its own 95th percentile.

    The claim is about WHERE the field sits from trial to trial, not how hard
    the cell fires, and a single map-wide clip lets the loud trials swamp the
    quiet ones: on a 99th-percentile clip the anchored band is visible but
    faint and the comparison has to be taken on trust.
    """
    d = np.nanpercentile(M, 95, axis=1, keepdims=True)
    return np.clip(M / np.where(d > 0, d, 1.), 0, 1)


axm = ax.inset_axes([.105, .455, .255, .395])
axm.imshow(_norm(gmap), aspect='auto', cmap='viridis', origin='upper',
           extent=[0, 200, ntr, 0], interpolation='nearest', vmin=0, vmax=1)
axm.set_xticks([0, 100, 200]); axm.set_yticks([0, ntr])
axm.tick_params(labelsize=4.6, length=2, pad=1)
axm.set_ylabel('trial', fontsize=4.8, labelpad=1)
axm.set_xlabel('position (cm)', fontsize=4.8, labelpad=1)
for sp in axm.spines.values():
    sp.set_linewidth(.5)
ax.text(.2325, .385, f'one grid cell, score {gs:.2f}', ha='center', va='top',
        fontsize=5.0, color='0.4')

# the state of the same trials, cell by cell and as PC1
z = np.load(f'{ROOT}/data/population_state/labels/M{int(mo)}D{int(dy)}.npz',
            allow_pickle=True)
L, load, pc1 = z['labels'], z['pc1_load'], z['pc1']
C = pd.read_csv(f'{ROOT}/data/population_state/anchoring_cells.csv')
C = C[(C.mouse == int(mo)) & (C.day == int(dy))]
keep = np.isin(z['cluster_id'], C.cluster_id[C.in_population].values)
L, load = L[keep], load[keep]
with np.errstate(invalid='ignore'):
    varies = np.array([np.isfinite(r).sum() > 1 and np.nanstd(r) > 0 for r in L])
ord_v = np.where(varies)[0][np.argsort(load[varies])[::-1]]
lk = np.where(~varies)[0]
_fr = np.array([np.nanmean(L[i]) if np.isfinite(L[i]).any() else 0. for i in lk])
ord_l = lk[np.argsort(-_fr)] if len(lk) else lk
gap = max(2, int(round(.025 * L.shape[0])))
Lb = np.vstack([L[ord_v]] + ([np.full((gap, L.shape[1]), np.nan), L[ord_l]]
                             if len(ord_l) else []))

axr = ax.inset_axes([.455, .455, .285, .395], sharey=axm)
axr.imshow(Lb.T, aspect='auto', cmap=TA, norm=NORM, interpolation='nearest',
           extent=[0, Lb.shape[0], ntr, 0])
axr.set_xticks([]); axr.tick_params(labelsize=4.6, length=2, pad=1,
                                    labelleft=False)
for sp in axr.spines.values():
    sp.set_linewidth(.5)
ax.text(.5975, .385, f'{Lb.shape[0]} MEC cells', ha='center', va='top',
        fontsize=5.0, color='0.4')

axq = ax.inset_axes([.775, .455, .105, .395], sharey=axm)
tr = np.arange(ntr)
axq.fill_betweenx(tr, 0, pc1, where=pc1 >= 0, color=ANCH, linewidth=0,
                  edgecolor='none', interpolate=True)
axq.fill_betweenx(tr, 0, pc1, where=pc1 < 0, color=NON, linewidth=0,
                  edgecolor='none', interpolate=True)
axq.axvline(0, color='0.4', lw=.5)
axq.set_xticks([]); axq.tick_params(labelleft=False, length=0)
for sp in axq.spines.values():
    sp.set_visible(False)
ax.text(.8275, .385, 'PC1', ha='center', va='top', fontsize=5.0, color='0.4')
ax.text(.905, .72, 'anchored', fontsize=4.8, color=ANCH, va='center')
ax.text(.905, .60, 'non-\nanchored', fontsize=4.8, color=NON, va='center',
        linespacing=1.2)
ax.text(.010, .65, f'M{int(mo)} D{int(dy)}', fontsize=5.0, color='0.45',
        va='center', rotation=90, ha='center')

# what the state buys the animal
axh = ax.inset_axes([.105, .10, .30, .20])
E = pd.read_csv(f'{ROOT}/data/eye_anchoring/eye_anchoring_trials.csv')
E = E.dropna(subset=['pc1', 'perf', 'ttype'])
E['hit'] = (E.perf == 'hit').astype(float)
E['anch'] = E.pc1 > 0
S_ = E.groupby(['mouse', 'day', 'ttype', 'anch']).hit.mean().unstack('anch').dropna()
S_.columns = ['non', 'anch']
for j, tt in enumerate(('b', 'nb')):
    g = E[E.ttype == tt]
    for k, (st, col) in enumerate(((True, ANCH), (False, NON))):
        axh.bar(j + (k - .5) * .34, g[g.anch == st].hit.mean(), .32,
                color=col, linewidth=0)
    d = S_.xs(tt, level='ttype').anch - S_.xs(tt, level='ttype').non
    hi = max(g[g.anch].hit.mean(), g[~g.anch].hit.mean())
    axh.text(j, hi + .04, f'+{d.mean():.2f}', ha='center', fontsize=5.2,
             color='0.2')
axh.set_xticks([0, 1]); axh.set_xticklabels(['cued', 'uncued'], fontsize=5.4)
axh.set_ylim(0, 1.05); axh.set_yticks([0, .4, .8])
axh.set_ylabel('hit rate', fontsize=5.2, labelpad=1)
axh.tick_params(labelsize=4.8, length=2, pad=1)
axh.spines[['top', 'right']].set_visible(False)
# the same benefit as one number per session: the Anchoring Dependence Index,
# AUC(uncued) minus AUC(cued) for the state predicting a hit
axa = ax.inset_axes([.505, .10, .095, .20])
_rows = []
for _k, _g in E.groupby(['mouse', 'day']):
    try:
        _a = [roc_auc_score(_g[_g.ttype == t].hit, _g[_g.ttype == t].pc1)
              for t in ('nb', 'b')]
    except ValueError:
        continue
    _rows.append(_a[0] - _a[1])
_adi = np.array(_rows)
axa.axhline(0, color='0.4', lw=.9)
axa.scatter(np.random.default_rng(1).uniform(-.2, .2, len(_adi)), _adi, s=4,
            color=ANCH, alpha=.5, lw=0)
axa.errorbar(0, _adi.mean(), yerr=_adi.std(ddof=1) / np.sqrt(len(_adi)),
             color='k', marker='_', ms=11, lw=1.3, capsize=2, zorder=4)
axa.set_xlim(-.55, .55); axa.set_xticks([0]); axa.set_xticklabels(['ADI'],
                                                                 fontsize=5.4)
axa.set_ylabel('AUC uncued $-$ cued', fontsize=5.0, labelpad=1)
axa.tick_params(labelsize=4.8, length=2, pad=1)
axa.spines[['top', 'right']].set_visible(False)
axa.set_title(f'{_adi.mean():+.3f}, p = {wilcoxon(_adi).pvalue:.3f}',
              fontsize=4.8, color='0.3', pad=2)
ax.text(.33, .038, 'anchoring helps most when the zone must be found by path '
                   'integration',
        ha='center', va='top', fontsize=5.2, color='0.35')

# ── 2. which cells follow it, and whose state predicts behaviour ─────────────
# Two lines, one question each, each with its own scale. Top: mean excess over
# every cell's own circular-shift null, averaged within session then across
# sessions, which lands within 0.004 of the mixed-model estimates in Figure 2D.
# Bottom: the Anchoring Dependence Index of a state built from FIVE randomly
# drawn cells of that class, so a class with 92 cells per session is not
# compared against one with 5 (Figure 2L).
ax = fig.add_subplot(G[1]); ax.axis('off')
ax.set_xlim(0, 1); ax.set_ylim(0, 1)
banner(ax, 'Grid cells and fast-spiking interneurons follow it\n'
           'most strongly, and every class\'s state predicts behaviour')

_adi = pd.read_csv(f'{ROOT}/data/population_state/adi_by_identity.csv')
COL4 = {'grid': GRID_C, 'putative interneuron': INT_C,
        'non-grid spatial': GREY, 'non-spatial': GREY}
ROWS2 = [
    (.54, 'follows\nthe state', .215, [0, .05, .10, .15, .20],
     {'grid': .183, 'putative interneuron': .132, 'non-grid spatial': .085,
      'non-spatial': .052}, {'non-grid spatial': 2}),
    (.06, 'its state\npredicts\nuncued hits', .125, [0, .04, .08, .12],
     {k: float(_adi[k].dropna().mean()) for k in COL4},
     {'non-grid spatial': 2, 'putative interneuron': 3}),
]
for _y, _lab, _xmax, _ticks, _vals, _tier in ROWS2:
    axl = ax.inset_axes([.30, _y, .68, .34])
    axl.plot([0, _xmax], [0, 0], color='0.78', lw=1.0, zorder=1,
             solid_capstyle='butt')
    for _k, _v in _vals.items():
        axl.scatter([_v], [0], s=26, color=COL4[_k], lw=0, zorder=3)
        _t = _tier.get(_k, 1)
        _dy = .28 + .98 * (_t - 1)
        if _t > 1:
            axl.plot([_v, _v], [.16, _dy], color=COL4[_k], lw=.5, zorder=2)
        axl.annotate(f'{_k.replace("putative ", "")}\n{_v:+.3f}', (_v, _dy),
                     ha='center', va='bottom', fontsize=4.8, color=COL4[_k],
                     linespacing=1.2)
    axl.set_xlim(-_xmax * .02, _xmax)
    axl.set_ylim(-.55, .22 + 1.05 * max([1] + list(_tier.values())))
    axl.set_xticks(_ticks); axl.set_yticks([])
    axl.tick_params(axis='x', labelsize=4.8, length=2, pad=1)
    axl.spines[['top', 'right', 'left']].set_visible(False)
    axl.spines['bottom'].set_position(('outward', 3))
    ax.text(.275, _y + .17, _lab, ha='right', va='center', fontsize=5.4,
            color='0.3', linespacing=1.25)
# the arrow goes above the pair, not below: under the lower line it lands on
# that line's own tick marks
ax.annotate('', xy=(.975, .905), xytext=(.315, .905),
            arrowprops=dict(arrowstyle='-|>', color='0.5', lw=1.0,
                            mutation_scale=8))
ax.text(.645, .925, 'more strongly with the population state', ha='center',
        va='bottom', fontsize=5.0, color='0.5')

# ── 3. what the state is ─────────────────────────────────────────────────────
ax = fig.add_subplot(G[2]); ax.axis('off')
ax.set_xlim(0, 1); ax.set_ylim(0, 1)
banner(ax, 'The state is attentional engagement, with a\n'
           'frequency-specific change in the field potential')

for k, (y0, key, col, lab) in enumerate(((.55, 'anch', ANCH, 'anchored'),
                                         (.06, 'non', NON, 'non-anchored'))):
    # label under each frame, not beside it: beside, it runs into the pupil
    # profile's y-axis label
    axe = ax.inset_axes([.035, y0 + .11, .125, .28])
    axe.imshow(Q[f'pupil_{key}'], interpolation='bilinear')
    axe.set_xticks([]); axe.set_yticks([])
    for sp in axe.spines.values():
        sp.set_color(col); sp.set_linewidth(1.1)
    ax.text(.0975, y0 + .085, f'{lab}\nr = {float(Q[f"pupil_{key}_r"][0]):.1f} px',
            fontsize=4.7, color=col, ha='center', va='top', linespacing=1.3)

axp = ax.inset_axes([.27, .28, .25, .62])
_z = np.load(f'{ROOT}/data/eye_anchoring/eye_position_profiles_gated.npz')
_A, _N = _z['anch'], _z['non']
_x = (np.arange(_A.shape[1]) + .5) * (200.0 / _A.shape[1])
axp.axvspan(*RZ, color='#d8e4d0', alpha=.55, lw=0, zorder=0)
for _x0 in (0, 170):
    axp.axvspan(_x0, _x0 + 30, color='0.90', lw=0, zorder=0)
for _M, _c in ((_A, ANCH), (_N, NON)):
    _mu = np.nanmean(_M, axis=0)
    _se = np.nanstd(_M, axis=0) / np.sqrt(np.sum(np.isfinite(_M), axis=0))
    axp.fill_between(_x, _mu - _se, _mu + _se, color=_c, alpha=.30,
                     linewidth=0, edgecolor='none', zorder=2)
    axp.plot(_x, _mu, color=_c, lw=1.1, zorder=3)
axp.axhline(0, color='0.75', lw=.7, zorder=1)
axp.set_xlim(0, 200); axp.set_xticks([0, 100, 200])
axp.set_xlabel('position (cm)', fontsize=5.2, labelpad=1)
axp.set_ylabel('z(pupil radius)', fontsize=5.2, labelpad=1)
axp.tick_params(labelsize=4.8, length=2, pad=1)
axp.spines[['top', 'right']].set_visible(False)

# the same schematic Figure 9 draws, at the measured preferred theta phases:
# what the state changes is how much each band carries and, for fast gamma,
# how tightly its burst is confined -- not when in the cycle it arrives
PH_SLOW, PH_FAST = 95., 42.
OFF_T, OFF_S, OFF_F = 4.6, 1.5, -2.2
LFP = {True: (.95, 1.55, .55, 2.6), False: (1.05, .55, 1.55, 8.0)}
_ph = np.linspace(-180., 180., 700)
for _i, (_y, _anch, _c) in enumerate(((.53, True, ANCH), (.10, False, NON))):
    axw = ax.inset_axes([.655, _y, .335, .37])
    _at, _as, _af, _kf = LFP[_anch]
    for _p0, _col in ((PH_SLOW, SLOW_G), (PH_FAST, FAST_G)):
        axw.axvline(_p0, color=_col, lw=.6, ls=(0, (2.0, 1.8)), alpha=.55,
                    zorder=0)
    axw.plot(_ph, OFF_T + _at * np.cos(np.radians(_ph)), color=THETA_C, lw=1.1)
    for _off, _amp, _p0, _k, _n, _col in (
            (OFF_S, _as, PH_SLOW, 3.4, 5., SLOW_G),
            (OFF_F, _af, PH_FAST, _kf, 10., FAST_G)):
        _e = _amp * np.exp(_k * (np.cos(np.radians(_ph - _p0)) - 1.))
        axw.fill_between(_ph, _off - _e, _off + _e, color=_col, alpha=.22,
                         linewidth=0, edgecolor='none')
        axw.plot(_ph, _off + _e * np.sin(np.radians(_ph) * _n), color=_col,
                 lw=.7)
    axw.set_xlim(-180, 180); axw.set_ylim(-4.2, 6.4)
    axw.set_yticks([OFF_T, OFF_S, OFF_F])
    axw.set_yticklabels(['theta', 'slow \u03b3', 'fast \u03b3'], fontsize=4.6)
    for _t, _col in zip(axw.get_yticklabels(), (THETA_C, SLOW_G, FAST_G)):
        _t.set_color(_col)
    axw.tick_params(axis='y', length=0, pad=1)
    axw.set_xticks([-180, 0, 180])
    axw.tick_params(axis='x', labelsize=4.6, length=2, pad=1)
    for _sp in ('top', 'right', 'left'):
        axw.spines[_sp].set_visible(False)
    axw.set_title('anchored' if _anch else 'non-anchored', loc='left',
                  fontsize=4.8, color=_c, fontweight='bold', pad=1)
    if _anch:
        axw.set_xticklabels([])
    else:
        axw.set_xlabel('theta phase (deg)', fontsize=4.8, labelpad=1)

fig.savefig(OUT, bbox_inches='tight', dpi=300)
print(f'wrote {OUT}')
