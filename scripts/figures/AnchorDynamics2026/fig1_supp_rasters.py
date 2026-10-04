"""Supplementary to Figure 1: population anchoring across six example sessions.

The main figure carries one session, which cannot show whether its structure is
typical or exceptional. Six are shown here, ordered by how much of the cell x
trial label variance PC1 accounts for, so the reader sees the range rather than
a single chosen case: from sessions where nearly all cells move together to
sessions where the shared component is weak.

Each entry is five panels sharing one vertical trial axis:

  cued stops | uncued stops | anchoring raster | PC1 | hit rate by state

The first four share the trial axis so that a block of anchored trials can be
read straight across into the animal's stopping behaviour. The anchoring raster
is transposed relative to the main figure (cells on x, trials on y) for the same
reason. The fifth panel leaves the trial axis and summarises the session as four bars in
the same geometry as the main figure's behaviour panel -- cued and uncued, each
split by anchoring state. The error bars are binomial standard errors because
the unit here is the trial within one session, not the session as in the main
figure, where the bars carry a SEM across sessions and the individual session
points scattered over them.

M25 D24 is included deliberately as the first entry -- it is the main figure's
session, so the comparison is anchored to something the reader has already seen.
"""
import os, sys, json, time, warnings
import numpy as np, pandas as pd
warnings.filterwarnings('ignore')
import matplotlib; matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap, BoundaryNorm
sys.path.insert(0, '/Users/harryclark/Documents/spatial-manifolds/src')

NB = ('/Users/harryclark/Documents/spatial-manifolds/scripts/figures/'
      'AnchorDynamics2026/lick_raster_by_trial_type.ipynb')
_g = {}
for _c in json.load(open(NB))['cells']:
    if _c['cell_type'] != 'code':
        continue
    _s = ''.join(_c['source'])
    if _s.startswith('MOUSE, DAY') or 'INV = pd.DataFrame' in _s:
        continue
    exec(compile(_s, '<nb>', 'exec'), _g)
globals().update({k: v for k, v in _g.items() if not k.startswith('__')})

plt.rcParams['font.family'] = 'Arial'
FIG = '/Users/harryclark/Documents/spatial-manifolds/scripts/figures/AnchorDynamics2026'
SESSIONS = [(25, 24), (26, 18), (25, 23), (27, 23), (20, 14), (21, 22)]
TA = ListedColormap([NONANCH_COLOR, ANCH_COLOR])
NORM = BoundaryNorm([-0.5, 0.5, 1.5], TA.N)
MAJOR_FILT = 9        # trials, odd; as Figure 1


def session(mo, dy):
    beh, trials, stops = load_vr(mo, dy)
    trials_a, L, frac = population_anchoring(mo, dy)
    Lc = np.nan_to_num(L - np.nanmean(L, axis=1, keepdims=True))
    U, sv, Vt = np.linalg.svd(Lc, full_matrices=False)
    load, pc1 = U[:, 0], Vt[0]
    # orient by correlation with the anchored fraction, not by the top cells'
    # baseline rate -- see the note in the main notebook
    if np.corrcoef(pc1, np.nan_to_num(frac))[0, 1] < 0:
        load, pc1 = -load, -pc1
    # SAME BLOCK STRUCTURE AS FIGURE 1. Cells whose label never changes have no
    # PC1 loading worth ordering on -- a constant row contributes nothing to the
    # decomposition -- so sorting them in with the rest scatters them through
    # the raster and makes the varying block look noisier than it is. They are
    # moved to the right of a blank gap and grouped by which mode they are stuck
    # in, so the reader can see how much of each session is locked.
    sd = np.nanstd(L, axis=1)
    varies = sd > 0
    ord_v = np.where(varies)[0][np.argsort(load[varies])[::-1]]
    lk = np.where(~varies)[0]
    frac_lk = np.array([np.nanmean(L[i]) for i in lk]) if len(lk) else np.array([])
    ord_l = lk[np.argsort(-frac_lk)] if len(lk) else lk
    gap = max(2, int(round(.025 * L.shape[0])))
    blocks = [L[ord_v]]
    if len(ord_l):
        blocks += [np.full((gap, L.shape[1]), np.nan), L[ord_l]]
    Lb = np.vstack(blocks)

    # MAJOR population transitions, as Figure 1 defines them: the thresholded
    # population state after a 9-trial median filter, which removes the short
    # flickers that would otherwise bury the raster under lines
    from scipy.ndimage import median_filter
    st = median_filter((np.asarray(frac, float) > .5).astype(float), size=MAJOR_FILT,
                       mode='nearest') > .5
    trans = list(np.where(np.diff(st.astype(int)) != 0)[0] + 1)

    return (beh, trials, stops, trials_a, Lb, frac, len(ord_v), len(ord_l),
            pc1, sv[0] ** 2 / np.sum(sv ** 2), trans)


def hit_rates(trials, trials_a, frac):
    """Hit rate per trial type x anchoring state, with binomial SE."""
    anch = dict(zip(trials_a.number.astype(int), np.asarray(frac) > .5))
    t = trials.copy()
    t['anch'] = t.number.astype(int).map(anch)
    out = {}
    for tt in ('b', 'nb'):
        for st in (True, False):
            sel = t[(t.type == tt) & (t.anch == st)]
            if len(sel) < 5:
                out[(tt, st)] = (np.nan, np.nan, len(sel)); continue
            p = float((sel.performance == 'hit').mean())
            out[(tt, st)] = (p, np.sqrt(p * (1 - p) / len(sel)), len(sel))
    return out


print('loading sessions ...')
D = {}
for mo, dy in SESSIONS:
    t0 = time.time()
    D[(mo, dy)] = session(mo, dy)
    print(f'  M{mo}D{dy}: {D[(mo,dy)][4].shape[0]} cells, '
          f'PC1 {100*D[(mo,dy)][9]:.0f}%, {D[(mo,dy)][7]} locked, '
          f'{len(D[(mo,dy)][10])} transitions, {time.time()-t0:.0f}s', flush=True)

order = sorted(SESSIONS, key=lambda k: -D[k][9])          # strongest structure first
print('order by PC1 share:', ', '.join(f'M{m}D{d}' for m, d in order))

# an explicit empty column sits between PC1 and the bar panel: the bar panel's
# y label would otherwise land on top of the PC1 fill, and wspace cannot widen
# that one gap without widening every other gap too
W = [1, 1, 1.32, .5, .30, 1.25]
fig = plt.figure(figsize=(9.8, 5.7))
gs = fig.add_gridspec(3, 13, width_ratios=W + [.5] + W,
                      wspace=.40, hspace=.56)

for k, (mo, dy) in enumerate(order):
    (beh, trials, stops, trials_a, L, frac, n_var, n_lk, pc1, var,
     trans) = D[(mo, dy)]
    r, c0 = k // 2, (k % 2) * 7
    t_first, t_last = 1, int(trials.number.max())
    tr_ax = np.arange(1, len(frac) + 1)

    for j, (tt, col, name) in enumerate((('b', CUED_C, 'cued'),
                                         ('nb', UNCUED_C, 'uncued'))):
        ax = fig.add_subplot(gs[r, c0 + j])
        stop_raster(ax, stops, trials, tt, col, '', t_first, t_last)
        ax.collections[0].set_sizes([1.4])       # helper's s=5 is too big here
        ax.set_xlabel('')
        if r == 0:
            ax.set_title(name, fontsize=7, color=col, pad=3)
        ax.set_xticks([0, 200])
        ax.tick_params(labelsize=5.5)
        if j == 0:
            ax.set_ylabel('Trial', fontsize=6.5)
            ax.yaxis.set_major_locator(plt.MaxNLocator(4))
            pos = gs[r, c0:c0 + 6].get_position(fig)
            fig.text(pos.x0, pos.y1 + .052, f'M{mo} D{dy} · {L.shape[0]} cells',
                     fontsize=7.5, ha='left', va='bottom')
        else:
            ax.set_yticklabels([])
        if r == 2 and j == 0:
            ax.set_xlabel('Position (cm)', fontsize=6.5)

    ax = fig.add_subplot(gs[r, c0 + 2])
    ax.imshow(L.T, aspect='auto', cmap=TA, norm=NORM,
              interpolation='nearest', extent=[0, L.shape[0], t_last, t_first])
    # these sessions differ widely in how often the population switches -- 4
    # transitions in the most structured, 21 in the least -- so the lines are
    # kept light enough that a flickering session reads as flickering rather
    # than as a ruled grid
    for tb in trans:
        ax.axhline(tb, color='0.1', lw=.55, ls=(0, (3, 2)), alpha=.85, zorder=4)
    if n_lk:
        # inside the panel at the foot: at the top it lands on the 'anchoring'
        # and 'PC1' titles of the two panels either side of it
        ax.annotate(f'{n_lk} locked', (L.shape[0] - n_lk / 2, .012),
                    xycoords=('data', 'axes fraction'), ha='center', va='bottom',
                    fontsize=4.8, color='0.15')
    if r == 0:
        ax.set_title('anchoring', fontsize=7, pad=3)
    ax.set_yticklabels([]); ax.tick_params(labelsize=6)
    ax.set_xticks([0, L.shape[0]])
    if r == 2:
        ax.set_xlabel('Cell', fontsize=6.5)
    for sp in ax.spines.values():
        sp.set_visible(False)

    ax = fig.add_subplot(gs[r, c0 + 3])
    ax.fill_betweenx(tr_ax, 0, pc1, where=pc1 >= 0, color=ANCH_COLOR, lw=0,
                     edgecolor='none', interpolate=True)
    ax.fill_betweenx(tr_ax, 0, pc1, where=pc1 < 0, color=NONANCH_COLOR, lw=0,
                     edgecolor='none', interpolate=True)
    ax.axvline(0, color='0.5', lw=.6)
    for tb in trans:
        ax.axhline(tb, color='0.1', lw=.55, ls=(0, (3, 2)), alpha=.85, zorder=4)
    ax.set_ylim(t_last, t_first)
    ax.set_title(f'PC1 {100*var:.0f}%\n{len(trans)} transitions', fontsize=5.8, pad=3)
    ax.set_xticks([]); ax.set_yticks([])
    ax.spines[['top', 'right', 'bottom', 'left']].set_visible(False)

    ax = fig.add_subplot(gs[r, c0 + 5])
    H = hit_rates(trials, trials_a, frac)
    XS = [0, .80, 2.1, 2.90]                       # same geometry as Figure 1
    for x, (tt, st), col in zip(XS,
                                (('b', True), ('b', False), ('nb', True), ('nb', False)),
                                (ANCH_COLOR, NONANCH_COLOR, ANCH_COLOR, NONANCH_COLOR)):
        p, e, n = H[(tt, st)]
        if not np.isfinite(p):
            continue
        ax.bar(x, 100 * p, width=.80, color=col, lw=.6, edgecolor='k', zorder=1)
        ax.errorbar(x, 100 * p, yerr=100 * e, color='k', lw=.9, capsize=2,
                    capthick=.9, zorder=5)
    ax.set_xlim(-.62, 3.52); ax.set_ylim(0, 100)
    ax.set_xticks([(XS[0] + XS[1]) / 2, (XS[2] + XS[3]) / 2])
    ax.set_xticklabels(['cued', 'uncued'], fontsize=6.5)
    ax.set_ylabel('Hit rate (%)', fontsize=6.5, labelpad=1)
    ax.set_yticks([0, 100])
    ax.tick_params(labelsize=6)
    # no y label: the top-row title already names the column, and the label
    # crowds the PC1 panel next to it. Ticks stay on every entry -- each hit-rate
    # panel sits far from the others, so borrowing a neighbour's axis is worse.
    if r == 0:
        ax.set_title('task performance', fontsize=7, pad=3)
    ax.spines[['top', 'right']].set_visible(False)
from matplotlib.patches import Patch
fig.legend(handles=[Patch(facecolor=CUED_C, edgecolor='none', label='cued'),
                    Patch(facecolor=UNCUED_C, edgecolor='none', label='uncued'),
                    Patch(facecolor=ANCH_COLOR, edgecolor='k', lw=.5, label='anchored'),
                    Patch(facecolor=NONANCH_COLOR, edgecolor='k', lw=.5,
                          label='non-anchored')],
           loc='upper right', bbox_to_anchor=(1.0, 1.075), ncol=4, fontsize=7,
           frameon=False, handlelength=1.1, columnspacing=1.2, handletextpad=.5)

for a in fig.axes:
    a.tick_params(axis='x', length=0)         # x labels kept, x tick marks removed

out = f'{FIG}/fig1_supp_population_anchoring_examples.pdf'
plt.savefig(out, dpi=200, bbox_inches='tight')
print('saved', out)
