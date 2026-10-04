"""Which regions predict which? The cross-region coupling matrix.

Every target cell is modelled twice with xgboost: a BASELINE using position
(sin/cos, the track is a ring), speed and the cell's own spike history, and a
FULL model that adds the binned spiking of a 10-cell partner pool. Coupling is
the gain, dpR2 = pR2(full) - pR2(base), so it is what those ten other cells add
ON TOP OF everything the animal's own behaviour already explains.

    A   the 4x4 matrix: rows = the region being predicted, columns = the region
        the partner pool was drawn from
    B   the controls that make the matrix readable: within > cross > time-shifted
    C   the within-session paired tests, which are the actual statistics
    D   is coupling symmetric? MEC<-VIS against VIS<-MEC in the same session
    E   does coupling depend on the anchoring state? (it does not)

Coupling also does not depend on how strongly the target cell follows the
population anchoring axis: dpR2 against the target's PC1-following rank gives
rho = +0.02, p = 0.34. That null has no panel because there is nothing to see;
it is printed by the script.

WHAT EACH CELL OF THE MATRIX IS. Pool size is fixed at 10 cells everywhere, so
no cell of the matrix is advantaged by having more partners available. The
DIAGONAL is a same-region pool drawn from cells at least 100 um away on the same
shank, or on a different shank entirely, which keeps spike-sorting bleed between
neighbouring contacts out of the ceiling. Two random pool draws are averaged.
Targets are one cell per decile of PC1-following, so each region's rank axis is
covered by construction rather than by luck.

EVERY VALUE IS FLOOR-SUBTRACTED. The `shift` condition re-runs each cross-region
pool with every partner's spike train circularly rolled by a random offset: same
cells, same rates, same history structure, timing destroyed. That floor is
+-0.003 everywhere (panel B), so subtraction barely moves the matrix -- but it
is what licenses reading the matrix as coupling rather than as a covariate-count
effect, and it is not optional off-diagonal in the state analysis, where a
baseline that generalises badly across states makes ANY added covariate look
helpful. The diagonal has no shift condition of its own and is shown raw; its
floor is assumed to match the cross-region floors measured in the same session.

THE MATRIX CANNOT BE READ ACROSS ROWS. Each cell rests on a different set of
sessions -- 27 have enough MEC cells, 16 VIS, 7 SUB, 1 LEC -- and sessions
differ enormously in overall coupling. MEC<-MEC is +0.066 among the 12 sessions
that also carry subiculum and +0.021 among the 21 that carry visual cortex: the
same quantity, three times apart, because of which sessions contribute. So the
matrix is the descriptive summary and panel C carries the claims, every
comparison paired within session.

WHAT IT SHOWS. Subicular pools predict MEC cells as well as other MEC cells do
(+0.067 vs +0.066, n = 12 sessions, p = 1.0), while visual pools predict them
clearly worse (+0.013 vs +0.021, n = 21, p = 0.001). Coupling is not detectably
asymmetric in either direction tested. And it does not depend on the anchoring
state: dpR2 in anchored and non-anchored trials is the same to three decimals in
every region (E), which is a real negative result for the idea that anchoring is
a change in how strongly cells are coupled to each other.

CAVEATS THE PANELS CARRY THEMSELVES. LEC is one session -- it is in the matrix
because the matrix is 4x4, and its row and column should not be read as a test.
Cerebellum appears only as a partner (no session has enough cerebellar cells to
supply targets), so it cannot be a row; it is the distant negative control and
is reported in B. dpR2 rises with a target's spike count (rho = +0.20), which is
why comparisons are paired on the same target cells wherever possible.

Reads the shards written by cross_region/xr_full.py.
"""
import glob
import os
import sys
import warnings

import numpy as np
import pandas as pd

warnings.filterwarnings('ignore')
import matplotlib

matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.colors import PowerNorm
from matplotlib.patches import Rectangle
from scipy.stats import spearmanr, wilcoxon

sys.path.insert(0, '/Users/harryclark/Documents/spatial-manifolds/src')
from spatial_manifolds.anchoring import ANCH_COLOR, NONANCH_COLOR

plt.rcParams['font.family'] = 'Arial'
ROOT = '/Users/harryclark/Documents/spatial-manifolds'
FIG = f'{ROOT}/scripts/figures/AnchorDynamics2026'
# xr_full2 supersedes xr_full: it lowers the target threshold to 10 cells and
# adds the anchoring-reconstruction columns. While it is still running, fall
# back to the first run so the figure always builds.
SHARDS = ([f'{ROOT}/data/cross_region_xgboost/xr_full2_w*.csv']
          if glob.glob(f'{ROOT}/data/cross_region_xgboost/xr_full2_w*.csv')
          else [f'{ROOT}/data/cross_region_xgboost/xr_full_w*.csv'])
# Square matrix: every region that serves as BOTH target and partner.
# CEREBELLUM IS THE DISTANT NEGATIVE CONTROL and is the reason the fourth slot
# is worth having: it is the one region with no anatomical business predicting
# entorhinal spiking. Its column comes from the main run; its row needed a
# separate pass (cross_region/xr_cb_targets.py) because the main run's target
# threshold of 20 same-region cells is never met by cerebellum after the
# PC1-following merge. Ten cells is enough to fill the deciles, and the pool is
# ten cells wherever it is drawn from.
# LEC is NOT here: it is a single session, so its row and column were one
# session's noise wearing the same colours as the tested cells.
REGIONS = ['MEC', 'SUB', 'VIS', 'CB']
RCOL = {'MEC': '#7b4173', 'VIS': '#3f9b4f', 'SUB': '#c0723a', 'CB': '#4c72b0',
        'LEC': '#8c6d31'}
MIN_PAIR = 4        # sessions needed before a paired test is run


def load():
    """All shards, concatenated. Raises if the run has not been done."""
    fs = sorted(f for pat in SHARDS for f in glob.glob(pat))
    if not fs:
        raise FileNotFoundError(
            f'No cross-region shards at {SHARDS[0]}. Run cross_region/xr_full.py '
            '(three workers, ~11 h) before building this figure.')
    if not any('xr_cb_targets' in f for f in fs):
        print('  ! no cerebellar targets yet -- the CB row will be empty until '
              'cross_region/xr_full2.py lands')
    D = pd.concat([pd.read_csv(f) for f in fs], ignore_index=True)
    print(f'{len(D)} rows from {len(fs)} shards, '
          f'{D.groupby(["mouse", "day"]).ngroups} sessions')
    return D


def per_session(D):
    """One dpR2 per session x target region x partner region, floor-subtracted.

    Draws are averaged first, then target cells, so a session contributes one
    number however many cells it happened to yield -- otherwise the sessions
    with the most cells would carry the means.
    """
    floor = (D[D.cond == 'shift']
             .groupby(['mouse', 'day', 'target_region', 'partner'])
             .d_all.mean().rename('floor'))
    G = (D[D.cond != 'shift']
         .groupby(['mouse', 'day', 'target_region', 'partner', 'cond'])
         .agg(d=('d_all', 'mean'), base=('pr2_base', 'mean'),
              n_cells=('cluster_id', 'nunique')).reset_index())
    G = G.merge(floor.reset_index(), on=['mouse', 'day', 'target_region',
                                         'partner'], how='left')
    # the diagonal has no shift condition of its own; use the session's mean
    # cross-region floor, which is the best available estimate of it
    sess_floor = (D[D.cond == 'shift'].groupby(['mouse', 'day'])
                  .d_all.mean().rename('sfloor'))
    G = G.merge(sess_floor.reset_index(), on=['mouse', 'day'], how='left')
    G['floor'] = G.floor.fillna(G.sfloor).fillna(0.0)
    G['dc'] = G.d - G.floor
    return G


def paired(G, treg, own, other):
    """Sessions holding both pools for one target region, paired."""
    a = G[(G.target_region == treg) & (G.partner == own)][['mouse', 'day', 'dc']]
    b = G[(G.target_region == treg) & (G.partner == other)][['mouse', 'day', 'dc']]
    return a.merge(b, on=['mouse', 'day'], suffixes=('_own', '_oth'))


def state_frame(D):
    """Floor-subtracted dpR2 within each state, per session and target region.

    Off-diagonal cells of the state matrix (train anchored, test non-anchored)
    have a floor of about +0.035 rather than zero, because a baseline that
    generalises badly across states makes any added covariate look helpful.
    Only the diagonal cells AA and NN are used here, and both are still
    floor-subtracted.
    """
    d = D[D.has_state == 1]
    if not len(d):
        return pd.DataFrame()
    f = (d[d.cond == 'shift'].groupby(['mouse', 'day', 'target_region'])
         [['d_AA', 'd_NN']].mean().rename(columns={'d_AA': 'fAA', 'd_NN': 'fNN'}))
    g = (d[d.cond != 'shift'].groupby(['mouse', 'day', 'target_region'])
         [['d_AA', 'd_NN']].mean())
    out = g.join(f, how='left').reset_index()
    out[['fAA', 'fNN']] = out[['fAA', 'fNN']].fillna(0.0)
    out['AA'] = out.d_AA - out.fAA
    out['NN'] = out.d_NN - out.fNN
    return out.dropna(subset=['AA', 'NN'])


# ── analysis ─────────────────────────────────────────────────────────────────
D = load()
G = per_session(D)
ST = state_frame(D)

print('\nmatrix (rows predicted <- columns predictor), floor-subtracted dpR2:')
MAT = G.pivot_table(index='target_region', columns='partner', values='dc',
                    aggfunc='mean').reindex(index=REGIONS, columns=REGIONS)
NMAT = G.pivot_table(index='target_region', columns='partner', values='dc',
                     aggfunc='size').reindex(index=REGIONS, columns=REGIONS)
print(MAT.round(4).to_string())
print('\nsessions per cell:')
print(NMAT.fillna(0).astype(int).to_string())

print('\ncontrols, pooled over sessions:')
for cond, nm in (('within', 'within-region pool'), ('cross', 'cross-region pool'),
                 ('shift', 'time-shifted (floor)')):
    v = (D[D.cond == cond].groupby(['mouse', 'day', 'target_region', 'partner'])
         .d_all.mean())
    print(f'  {nm:24s} {v.mean():+.4f}  (n = {len(v)} session x pair)')

print('\nwithin-session paired tests:')
TESTS = [('MEC', 'MEC', 'SUB'), ('MEC', 'MEC', 'VIS'), ('MEC', 'MEC', 'CB'),
         ('SUB', 'SUB', 'MEC'), ('SUB', 'SUB', 'VIS'), ('VIS', 'VIS', 'MEC'),
         ('CB', 'CB', 'MEC')]
RES = []
for treg, own, oth in TESTS:
    m = paired(G, treg, own, oth)
    if len(m) < MIN_PAIR:
        print(f'  {treg}<-{own} vs {treg}<-{oth}: only {len(m)} sessions, skipped')
        continue
    p = wilcoxon(m.dc_own, m.dc_oth).pvalue
    RES.append(dict(treg=treg, own=own, oth=oth, n=len(m), p=p,
                    m_own=m.dc_own.mean(), m_oth=m.dc_oth.mean(), data=m))
    print(f'  {treg}<-{own} {m.dc_own.mean():+.4f} vs {treg}<-{oth} '
          f'{m.dc_oth.mean():+.4f}  n={len(m)}  p={p:.3f}')

print('\nasymmetry (same sessions, both directions):')
ASYM = []
for a_, b_ in (('MEC', 'VIS'), ('MEC', 'SUB'), ('VIS', 'SUB')):
    u = G[(G.target_region == a_) & (G.partner == b_)][['mouse', 'day', 'dc']]
    v = G[(G.target_region == b_) & (G.partner == a_)][['mouse', 'day', 'dc']]
    m = u.merge(v, on=['mouse', 'day'], suffixes=('_f', '_r'))
    if len(m) < MIN_PAIR:
        continue
    p = wilcoxon(m.dc_f, m.dc_r).pvalue
    ASYM.append(dict(a=a_, b=b_, n=len(m), p=p, data=m))
    print(f'  {a_}<-{b_} {m.dc_f.mean():+.4f}  vs  {b_}<-{a_} {m.dc_r.mean():+.4f}'
          f'  n={len(m)}  p={p:.3f}')

if len(ST):
    print('\nanchored vs non-anchored dpR2 (floor-subtracted):')
    for r in REGIONS:
        s = ST[ST.target_region == r]
        if len(s) < MIN_PAIR:
            continue
        print(f'  {r}: anchored {s.AA.mean():+.4f}  non-anchored {s.NN.mean():+.4f}'
              f'  n={len(s)}  p={wilcoxon(s.AA, s.NN).pvalue:.3f}')

x = D[D.cond != 'shift']
rho_n, p_n = spearmanr(x.n_spikes, x.d_all)
rho_r, p_r = spearmanr(x.pc1_rank, x.d_all)
print(f'\ndpR2 vs target spike count: rho = {rho_n:+.3f}, p = {p_n:.2g}')
print(f'dpR2 vs target PC1-following rank: rho = {rho_r:+.3f}, p = {p_r:.2g}')

# ── figure ───────────────────────────────────────────────────────────────────
fig = plt.figure(figsize=(7.4, 9.2))
gs = fig.add_gridspec(3, 2, height_ratios=[1.25, 1, 1], width_ratios=[1.15, 1],
                      hspace=.5, wspace=.42)

# A: the matrix
ax = fig.add_subplot(gs[0, 0])
Mv = MAT.values.astype(float)
vmax = np.nanmax(Mv)
# gamma < 1 spreads the crowded low end: five of the twelve filled cells sit
# below 0.015 and are indistinguishable under a linear norm
cmap = plt.get_cmap('magma')
norm = PowerNorm(gamma=.55, vmin=0, vmax=vmax)
im = ax.imshow(Mv, cmap=cmap, norm=norm)
for i in range(len(REGIONS)):
    for j in range(len(REGIONS)):
        v, n = Mv[i, j], NMAT.values[i, j]
        if not np.isfinite(v):
            ax.add_patch(Rectangle((j - .5, i - .5), 1, 1, facecolor='0.92',
                                   edgecolor='white', lw=1.5, zorder=2))
            ax.text(j, i, 'n/a', ha='center', va='center', fontsize=6.5,
                    color='0.55', zorder=3)
            continue
        # magma runs dark -> bright, so pick the text colour from the actual
        # luminance of the mapped cell rather than from the value
        r_, g_, b_ = cmap(norm(v))[:3]
        lum = .299 * r_ + .587 * g_ + .114 * b_
        ax.text(j, i, f'{v:+.3f}\n{int(n)}', ha='center', va='center',
                fontsize=7, linespacing=1.15,
                color='0.1' if lum > .55 else 'white', zorder=3)
    ax.add_patch(Rectangle((i - .5, i - .5), 1, 1, fill=False, edgecolor='white',
                           lw=2.2, zorder=4))
ax.set_xticks(range(len(REGIONS))); ax.set_xticklabels(REGIONS, fontsize=8)
ax.set_yticks(range(len(REGIONS))); ax.set_yticklabels(REGIONS, fontsize=8)
for t, r in zip(ax.get_xticklabels(), REGIONS):
    t.set_color(RCOL[r])
for t, r in zip(ax.get_yticklabels(), REGIONS):
    t.set_color(RCOL[r])
ax.set_xlabel('Partner pool drawn from', fontsize=8.5)
ax.set_ylabel('Region predicted', fontsize=8.5)
ax.set_title('ΔpR² from a 10-cell pool\n(value above, sessions below)',
             fontsize=8.5, loc='left')
ax.tick_params(length=0)
cb = fig.colorbar(im, ax=ax, fraction=.046, pad=.04)
cb.ax.tick_params(labelsize=6.5); cb.outline.set_visible(False)
cb.set_label('ΔpR²', fontsize=7)

# B: the controls
ax = fig.add_subplot(gs[0, 1])
rng = np.random.default_rng(0)
for j, (cond, nm) in enumerate((('within', 'within'), ('cross', 'cross'),
                                ('shift', 'shifted'))):
    v = (D[D.cond == cond].groupby(['mouse', 'day', 'target_region', 'partner'])
         .d_all.mean().values)
    ax.scatter(j + rng.uniform(-.14, .14, len(v)), v, s=7, color='0.35', lw=0,
               alpha=.45, zorder=2)
    ax.errorbar(j, v.mean(), yerr=v.std(ddof=1) / np.sqrt(len(v)), color='k',
                marker='_', ms=16, lw=1.6, capsize=3, zorder=3)
    ax.annotate(f'{v.mean():+.3f}', (j, -.055), ha='center', fontsize=6.5,
                color='0.3')
ax.axhline(0, color='0.6', lw=.7, ls=':')
ax.set_xticks([0, 1, 2]); ax.set_xticklabels(['within', 'cross', 'shifted'],
                                             fontsize=8)
ax.set_xlim(-.5, 2.5); ax.set_ylim(-.07, .32)
ax.set_ylabel('ΔpR²', fontsize=8.5)
ax.set_title('controls: the shifted floor sits at zero', fontsize=8.5, loc='left')
ax.tick_params(labelsize=7.5); ax.spines[['top', 'right']].set_visible(False)

# C: within-session paired tests -- the actual statistics
ax = fig.add_subplot(gs[1, :])
for j, r in enumerate(RES):
    x0, x1 = j * 1.6, j * 1.6 + .62
    m = r['data']
    for _, row in m.iterrows():
        ax.plot([x0, x1], [row.dc_own, row.dc_oth], color='0.72', lw=.7, zorder=1)
    ax.scatter(np.full(len(m), x0), m.dc_own, s=13, color=RCOL[r['own']], lw=0,
               zorder=3)
    ax.scatter(np.full(len(m), x1), m.dc_oth, s=13, color=RCOL[r['oth']], lw=0,
               zorder=3)
    for xx, vv in ((x0, m.dc_own), (x1, m.dc_oth)):
        ax.plot([xx - .2, xx + .2], [vv.mean()] * 2, color='k', lw=1.8, zorder=4)
    star = ('n.s.' if r['p'] > .05 else '*' if r['p'] > .01
            else '**' if r['p'] > .001 else '***')
    top = max(m.dc_own.max(), m.dc_oth.max())
    ax.plot([x0, x0, x1, x1], [top + .02, top + .035, top + .035, top + .02],
            color='0.3', lw=.8)
    ax.text((x0 + x1) / 2, top + .042, f'{star}\np={r["p"]:.3f}', ha='center',
            fontsize=6.3, color='0.25', linespacing=1.1)
    ax.text((x0 + x1) / 2, -.055, f'{r["treg"]} targets\n{r["own"]} vs '
            f'{r["oth"]} pool\nn={r["n"]}', ha='center', fontsize=6.8,
            color='0.25', linespacing=1.2)
ax.axhline(0, color='0.6', lw=.7, ls=':')
ax.set_xlim(-.5, 1.6 * len(RES) - .4)
ax.set_xticks([])
ax.set_ylabel('ΔpR² (floor-subtracted)', fontsize=8.5)
ax.set_title('paired within session: subicular pools predict MEC cells as well '
             'as MEC pools do', fontsize=8.5, loc='left')
ax.tick_params(labelsize=7.5)
ax.spines[['top', 'right', 'bottom']].set_visible(False)

# D: symmetry
ax = fig.add_subplot(gs[2, 0])
for k, a_ in enumerate(ASYM):
    m = a_['data']
    ax.scatter(m.dc_f, m.dc_r, s=18, lw=.5, edgecolor='w',
               color=RCOL[a_['a']] if k == 0 else RCOL[a_['b']],
               label=f'{a_["a"]}↔{a_["b"]} (n={a_["n"]}, p={a_["p"]:.2f})')
lim = [0, max(.02, np.nanmax(Mv) * 1.05)]
ax.plot(lim, lim, color='0.6', lw=.8, ls=':')
ax.set_xlim(lim); ax.set_ylim(lim)
ax.set_xlabel('ΔpR², first ← second', fontsize=8.5)
ax.set_ylabel('ΔpR², second ← first', fontsize=8.5)
ax.set_title('coupling is symmetric', fontsize=8.5, loc='left')
ax.legend(fontsize=6, frameon=False, loc='upper left')
ax.tick_params(labelsize=7.5); ax.spines[['top', 'right']].set_visible(False)

# E: does the anchoring state change coupling?
ax = fig.add_subplot(gs[2, 1])
if len(ST):
    shown = [r for r in REGIONS if (ST.target_region == r).sum() >= MIN_PAIR]
    for j, r in enumerate(shown):
        s = ST[ST.target_region == r]
        for x_, v_, c_ in ((j - .17, s.AA, ANCH_COLOR), (j + .17, s.NN,
                                                         NONANCH_COLOR)):
            ax.scatter(np.full(len(s), x_), v_, s=10, color=c_, lw=0, alpha=.65,
                       zorder=2)
            ax.errorbar(x_, v_.mean(), yerr=v_.std(ddof=1) / np.sqrt(len(s)),
                        color='k', marker='_', ms=10, lw=1.3, capsize=2.5,
                        zorder=3)
        p = wilcoxon(s.AA, s.NN).pvalue
        # axes fraction, so it does not depend on limits that are still settling
        ax.annotate(f'n.s.\nn={len(s)}' if p > .05 else f'p={p:.3f}\nn={len(s)}',
                    (j, .94), xycoords=('data', 'axes fraction'), ha='center',
                    va='top', fontsize=6.3, color='0.3', linespacing=1.1)
    ax.set_xlim(-.5, len(shown) - .5 + .8)      # room for the legend
    ax.set_ylim(top=ST[['AA', 'NN']].max().max() * 1.45)
    ax.set_xticks(range(len(shown)))
    ax.set_xticklabels(shown, fontsize=8)
    for t, r in zip(ax.get_xticklabels(), shown):
        t.set_color(RCOL[r])
ax.axhline(0, color='0.6', lw=.7, ls=':')
ax.set_ylabel('ΔpR² (floor-subtracted)', fontsize=8.5)
ax.set_title('coupling does not track the state', fontsize=8.5, loc='left')
h = [plt.Line2D([], [], marker='o', ls='', ms=4, color=ANCH_COLOR,
                label='anchored'),
     plt.Line2D([], [], marker='o', ls='', ms=4, color=NONANCH_COLOR,
                label='non-anchored')]
ax.legend(handles=h, fontsize=6.5, frameon=False, loc='center right')
ax.tick_params(labelsize=7.5); ax.spines[['top', 'right']].set_visible(False)

for lab, a_ in zip('ABCDE', fig.axes[:2] + fig.axes[3:6]):
    a_.text(-.16, 1.06, lab, transform=a_.transAxes, fontsize=11,
            fontweight='bold', va='bottom')

out = f'{FIG}/fig5_cross_region_coupling.pdf'
plt.savefig(out, dpi=200, bbox_inches='tight')
MAT.to_csv(f'{ROOT}/data/cross_region_xgboost/coupling_matrix.csv')
G.to_csv(f'{ROOT}/data/cross_region_xgboost/coupling_per_session.csv', index=False)
print(f'\nsaved {out}')
print(f'saved coupling_matrix.csv and coupling_per_session.csv')
