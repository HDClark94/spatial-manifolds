"""Graphical abstract, in the Cell Press stacked-panel style.

Three stacked statements: what the phenomenon is and what it buys the animal,
which cells express it, and what the state is.

NOTHING HERE IS A SCHEMATIC EXCEPT THE TWO TRACK STRIPS, and nothing is typed
in except the Figure 2 and Figure 7 bar values. Everything else is computed
from cached tables, so it cannot drift from the paper the way the retired
Figure 2D values once did:

  grid cell   ga_cache.npz -- M26 D18 cell 109, grid score 0.95, its trial x
              position map split by the trial's anchoring label
  raster      data/population_state/labels/M26D18.npz -- the cell x trial label
              matrix, its PC1 and the per-cell loading, ordered as Figure 1
              orders them
  behaviour   data/eye_anchoring/eye_anchoring_trials.csv -- hit rate by state
              and cue condition, the state being the sign of PC1
  the pupil   one video frame per state from the Figure 5 example session, at a
              matched track position, and the Figure 5I profiles from
              data/eye_anchoring/eye_position_profiles_gated.npz, drawn as that
              panel draws them. The per-bin significance markers are left off:
              they are uncorrected in Figure 5 and mark extent rather than test
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
CUE_C = '#3f7d3f'
TA = ListedColormap([NON, ANCH])
NORM = BoundaryNorm([-0.5, 0.5, 1.5], TA.N)
RZ = (90., 110.)
Q = np.load(f'{FIG}/ga_cache.npz', allow_pickle=True)

fig = plt.figure(figsize=(3.55, 6.05))
G = fig.add_gridspec(3, 1, height_ratios=[2.10, 0.62, 0.92],
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
    ax.text(x0 + w / 2, y0 + h + .006, 'cued' if cued else 'uncued',
            ha='center', va='bottom', fontsize=5.4,
            color=CUE_C if cued else '0.35')
    if not cued:
        ax.text(zx + zw / 2, y0 - .004, 'zone, unmarked', ha='center',
                va='top', fontsize=4.2, color='0.5')


# ── 1. the phenomenon, and what it buys ──────────────────────────────────────
ax = fig.add_subplot(G[0]); ax.axis('off')
ax.set_xlim(0, 1); ax.set_ylim(0, 1)
banner(ax, 'Entorhinal firing anchors to the task reference\n'
           'frame and tracks uncued engagement')

track(ax, .035, .42, .935, .030, True)
track(ax, .545, .42, .935, .030, False)

# one real grid cell, its trials split by the state they were in
gmap, ganch = Q['gc_map'], Q['gc_anch']
mo, dy, cid, gs = Q['gc_meta']
def _norm(M):
    """Each trial scaled to its own 95th percentile.

    The claim this panel makes is about WHERE the field sits from trial to
    trial, not how hard the cell fires, and a single map-wide clip lets the
    loud trials swamp the quiet ones: on the 99th-percentile clip the anchored
    band is visible but faint and the comparison has to be taken on trust.
    """
    d = np.nanpercentile(M, 95, axis=1, keepdims=True)
    return np.clip(M / np.where(d > 0, d, 1.), 0, 1)
for k, (x0, st, col, lab) in enumerate(((.035, 1., ANCH, 'anchored'),
                                        (.265, 0., NON, 'non-anchored'))):
    M = gmap[ganch == st]
    axm = ax.inset_axes([x0, .50, .20, .33])
    axm.imshow(_norm(M), aspect='auto', cmap='viridis',
               origin='upper', extent=[0, 200, M.shape[0], 0],
               interpolation='nearest', vmin=0, vmax=1)
    axm.set_xticks([]); axm.set_yticks([])
    for sp in axm.spines.values():
        sp.set_color(col); sp.set_linewidth(1.3)
    ax.text(x0 + .10, .845, lab, ha='center', fontsize=5.4, color=col)
ax.text(.165, .475, f'one grid cell, {int(gmap.shape[0])} trials',
        ha='center', va='top', fontsize=5.0, color='0.4')

# the population, as Figure 1 draws it
z = np.load(f'{ROOT}/data/population_state/labels/M26D18.npz', allow_pickle=True)
L, load, pc1 = z['labels'], z['pc1_load'], z['pc1']
C = pd.read_csv(f'{ROOT}/data/population_state/anchoring_cells.csv')
C = C[(C.mouse == 26) & (C.day == 18)]
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
ntr = Lb.shape[1]

axr = ax.inset_axes([.555, .50, .30, .33])
axr.imshow(Lb.T, aspect='auto', cmap=TA, norm=NORM, interpolation='nearest',
           extent=[0, Lb.shape[0], ntr, 0])
axr.set_xticks([]); axr.set_yticks([0, ntr])
axr.tick_params(labelsize=4.8, length=2, pad=1)
axr.set_ylabel('trial', fontsize=4.8, labelpad=1)
for sp in axr.spines.values():
    sp.set_linewidth(.5)
axq = ax.inset_axes([.875, .50, .085, .33])
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
ax.text(.755, .475, f'{Lb.shape[0]} MEC cells, one session',
        ha='center', va='top', fontsize=5.0, color='0.4')

# what the state buys the animal
axh = ax.inset_axes([.30, .14, .42, .23])
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
axh.legend(handles=[Patch(facecolor=ANCH, label='anchored'),
                    Patch(facecolor=NON, label='non-anchored')],
           fontsize=4.6, frameon=False, loc='upper left', handlelength=.8,
           handleheight=.8, borderpad=.1, labelspacing=.2,
           bbox_to_anchor=(-.02, 1.28), ncol=2, columnspacing=.8)
_W = (S_.anch - S_.non).unstack('ttype').dropna()
_p = wilcoxon(_W.nb, _W.b).pvalue
ax.text(.5, .105, f'the benefit is larger when the zone must be found by path\n'
                  f'integration ({(_W.nb - _W.b).mean():+.3f}, p = {_p:.0e}'
                  .replace('e-0', 'e-') + f', {len(_W)} sessions)',
        ha='center', va='top', fontsize=5.2, color='0.35', linespacing=1.4)

# ── 2. which cells ───────────────────────────────────────────────────────────
ax = fig.add_subplot(G[1])
banner(ax, 'Every class follows it — grid cells and\n'
           'interneurons about twice as strongly')
vals = [0.185, 0.131, 0.081, 0.056]
cols = [GRID_C, INT_C, GREY, GREY]
ax.bar(range(4), vals, .64, color=cols, linewidth=0)
ax.axhline(0, color='0.35', lw=.9)
ax.set_xticks(range(4))
ax.set_xticklabels(['grid', 'inter-\nneuron', 'non-grid\nspat.', 'non-\nspat.'],
                   fontsize=5.6)
ax.set_ylabel('agreement with the\nstate, over chance', fontsize=5.8)
ax.tick_params(axis='y', labelsize=5.4)
ax.set_ylim(0, .235)
for i, v in enumerate(vals):
    ax.text(i, v + .010, f'+{v:.3f}', ha='center', fontsize=5.2, color=cols[i])
ax.spines[['top', 'right']].set_visible(False)

# ── 3. what the state is ─────────────────────────────────────────────────────
ax = fig.add_subplot(G[2]); ax.axis('off')
ax.set_xlim(0, 1); ax.set_ylim(0, 1)
banner(ax, 'The state is attentional engagement, with a\n'
           'frequency-specific change in the field potential')

for k, (y0, key, col, lab) in enumerate(((.52, 'anch', ANCH, 'anchored'),
                                         (.05, 'non', NON, 'non-anchored'))):
    axe = ax.inset_axes([.02, y0, .145, .40])
    axe.imshow(Q[f'pupil_{key}'], interpolation='bilinear')
    axe.set_xticks([]); axe.set_yticks([])
    for sp in axe.spines.values():
        sp.set_color(col); sp.set_linewidth(1.1)
    ax.text(.175, y0 + .26, lab, fontsize=4.8, color=col, va='center')
    ax.text(.175, y0 + .14, f'r = {float(Q[f"pupil_{key}_r"][0]):.1f} px',
            fontsize=4.6, color='0.45', va='center')

axp = ax.inset_axes([.33, .28, .34, .62])
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

axg = ax.inset_axes([.76, .28, .21, .62])
axg.bar(range(3), [-.094, .105, -.108], .62,
        color=[THETA_C, SLOW_G, FAST_G], linewidth=0)
axg.axhline(0, color='0.35', lw=.9)
axg.set_xticks(range(3))
axg.set_xticklabels(['theta', 'slow γ', 'fast γ'], fontsize=4.8)
axg.set_ylim(-.16, .16); axg.set_yticks([-.1, 0, .1])
axg.tick_params(labelsize=4.8, length=2, pad=1)
axg.set_ylabel('anchored $-$ non (z)', fontsize=5.2, labelpad=1)
axg.spines[['top', 'right']].set_visible(False)

_same = int((np.nanmean(_A - _N, axis=0) < 0).sum())
ax.text(.5, .04, f'the pupil is narrower at {_same} of {_A.shape[1]} positions on '
                 f'the track — a tonic shift, not\nattention paid at the reward '
                 f'zone; the band difference reverses sign at 67 Hz',
        ha='center', va='top', fontsize=5.0, color='0.35', linespacing=1.4)

fig.savefig(OUT, bbox_inches='tight', dpi=300)
print(f'wrote {OUT}')
