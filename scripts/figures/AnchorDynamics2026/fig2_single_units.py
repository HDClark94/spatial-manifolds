"""Figure 2 — which single units follow the population anchoring state, and why.

Usage:  python3 fig2_single_units.py [mouse day]

Figure 1 establishes a population state. This asks what the individual cells are
doing inside it:

    A, B  two example sessions, each as the full MEC anchoring raster, PC1 of
          that raster, one example unit from each of the four identity groups,
          and two LOCKED units -- one stuck anchored, one stuck non-anchored

EVERY PANEL IN A SESSION ROW SHARES THE TRIAL AXIS, running down the page:
raster, PC1 and every rate map. The raster is therefore drawn cells-on-x,
trials-on-y, which is the transpose of the usual convention but the only
orientation in which a block of anchored trials can be followed straight across
into the rate map that produced it.

LOCKED UNITS ARE WHY 'NO AGREEMENT' IS NOT THE SAME AS 'DISAGREES'. Under the
absolute gate a cell whose two clusters both fall on the same side of its null
is called anchored on every trial, or non-anchored on every trial. Such a cell
has zero label variance, so its correlation with the population axis is
undefined -- it appears as "no agreement" while in fact it never had the
opportunity to agree or disagree. They are 10% of the population in M26 D18 and
23% in M28 D25, so leaving them out of the figure would misrepresent what the
population is made of.
    C     agreement with the population axis by cell identity
    D     the fraction of cells that significantly follow, by identity
    D     the same, expressed as excess over chance, with the test against it
    E     how many cells individually beat their own null, per mouse
    F     the same question in time: what each identity does around a
          population switch
    G     agreement BETWEEN identities, as a 4x4 matrix over its own null
    H     LOCKED CELLS: which identities never leave one anchoring mode

The monosynaptic connectivity that corroborates G -- grid cells and interneurons
are WIRED together, not merely correlated -- is a supplement
(fig2_supp_interneurons.py) rather than a panel here. It is a second, optional
line of support for a claim G already makes on its own, and it arrives with a
detector that needs five panels of validation before its matrices can be read.

CHANCE IS NOT ZERO, which is the reason panels C and D are separate. A cell
anchored on most trials correlates with a population that is also anchored on
most trials whatever the coordination, so each cell carries its own
circular-shift null and the comparison must be made against that. The nulls sit
near +0.15, which is most of the raw agreement: read against zero, all four
identities appear to follow the state; read against chance, only grid cells
(+0.119) and putative interneurons (+0.080) do, while non-grid spatial cells sit
on their null (-0.000) and non-spatial cells below it (-0.024).

The two panels are reconcilable and both are shown: 19-24% of non-grid and
non-spatial cells individually beat the 5% false-positive rate (E), so those
classes contain genuine followers -- but their group average is at chance, a
minority of followers inside a majority at the null.

STATISTICS ARE HIERARCHICAL. Cells are nested in sessions in mice, so a test
over cells treats 9,386 correlated observations as independent. Every
comparison here is a mixed model, r ~ group + (1 | mouse) + (1 | mouse:day),
with pairwise contrasts Holm-corrected. The cell-level Mann-Whitney values were
between four and ten orders of magnitude smaller, which is the size of the
pseudoreplication.

WHY ROW F EXISTS. Both sessions' locked examples turned out to be non-grid
spatial cells, which looks like a finding and is not: non-grid spatial cells are
70% of the population and are locked at almost exactly their base rate. The real
asymmetry is in the other two classes, and it runs in opposite directions --
putative interneurons lock ON while non-spatial cells lock OFF. Row F shows
that, so the examples above are not over-read.

ROW F USES SWITCHING SESSIONS ONLY, unlike the rows above it. The question it
asks -- which cells stayed locked while the population moved -- is undefined in
a session whose population never moved, where every cell is locked trivially.
Requiring at least 10 trials in each state leaves 35 of 62 sessions. The
restriction strengthens the result rather than producing it: the interneuron
always:never ratio goes from 23:1 over all sessions to 39:1 here.

The speed test that used to occupy this row -- are dissenters the open-field
speed-modulated cells? -- is a clean null (p = 0.99) and is reported in the
notebook rather than given four panels here.

The interneuron classification has its own supplementary figure
(fig2_supp_interneurons.py), which is where the waveform bimodality and its
relation to firing rate belong -- it is a method, not a result.

RATE MAPS ARE SMOOTHED AS IN FIGURE 1: NaN-aware, along POSITION only, never
across trials. The trial axis is where the anchoring banding lives, and
smoothing it would manufacture the structure the figure exists to show.

WHY F IS ASKED AT ALL. On the linear track speed and position covary strongly,
so a cell that looks spatially stable while the population is not could simply
be tracking speed, which does not stop when the map does. The open field breaks
that confound: position and speed are close to independent there, so a cell's
open-field speed score is a clean read of speed modulation, measured in a
different environment from the one the anchoring is defined in.

THE ANSWER IS NO, and the measure is good enough for that to mean something.
Top-decile dissenters have an open-field speed score of +0.046 against +0.044
for the rest (p = 0.99), and are slightly LESS likely to be speed-significant
(49% vs 55%). The pooled rank correlation between dissent and speed is
rho = -0.042 -- significant only because n = 4217, trivially small, and in the
wrong direction for the hypothesis.

THE SPEED SCORE IS COMPUTED, NOT THE SUPPLIED ONE. Each session ships a
time-shifted speed GLM whose score is negative for nearly every cell and
significant in ~1% of entorhinal cells; a null against that would be
uninterpretable. `of_speed_score.py` computes r(rate, speed) against a
circular-shift null instead: 51% of cells beat their null and the score
replicates across OF1 and OF2 at r = 0.75. Interneurons come out as the most
speed-modulated group (+0.084), which is the expected sanity check.

BEWARE THE POOLED CORRELATION IN F. Pooling cells across mice gives
rho(follow, speed) = +0.066 at p = 2e-5, which looks like a real relationship
and is not: per mouse it runs -0.213, -0.018, +0.059, +0.005, -0.119, -0.016,
+0.185 -- four negative, three positive. The per-mouse values are plotted
alongside so the pooled number cannot be read on its own.
"""
import os
import sys
import warnings

import numpy as np
import pandas as pd

warnings.filterwarnings('ignore')
import matplotlib

matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.colors import BoundaryNorm, ListedColormap
from matplotlib.patches import Patch
import seaborn as sns
import statsmodels.formula.api as smf
from scipy.stats import kruskal, mannwhitneyu, spearmanr, wilcoxon

sys.path.insert(0, '/Users/harryclark/Documents/spatial-manifolds/src')
from spatial_manifolds.anchoring import (ANCH_COLOR, NONANCH_COLOR,
                                         load_session_labels)

plt.rcParams['font.family'] = 'Arial'
# panel letters are set as mathtext \bf inside the axes titles, so mathtext
# must resolve to Arial Bold rather than the default DejaVu
plt.rcParams['mathtext.fontset'] = 'custom'
plt.rcParams['mathtext.rm'] = 'Arial'
plt.rcParams['mathtext.it'] = 'Arial:italic'
plt.rcParams['mathtext.bf'] = 'Arial:bold'


def _lp(ax, s, dx=-.16, dy=1.0):
    """Bold panel letter, drawn separately from the title.

    Every panel letter in this figure used to be the first token of the title
    string -- as plain (non-bold) text for some panels and as inline mathtext
    bold for others -- so it rendered at whatever fontsize that panel's title
    happened to use (6.8 to 8.5 here) rather than at one consistent size. This
    draws the letter on its own, always at 10 pt bold, matching every other
    figure in the paper.
    """
    ax.text(dx, dy, s, transform=ax.transAxes, fontsize=10, weight='bold',
            va='bottom', ha='right')
ROOT = '/Users/harryclark/Documents/spatial-manifolds'
FIG = f'{ROOT}/scripts/figures/AnchorDynamics2026'
PS = f'{ROOT}/data/population_state'
# Principal-cell identities first, in the order used elsewhere, with the
# putative interneuron last: it is a waveform call, not an open-field class, so
# it does not belong inside the spatial series.
ORDER = ['grid', 'non-grid spatial', 'non-spatial', 'putative interneuron']
COL_GC, COL_NGS, COL_OTHER = '#c04744', '#3171ae', '#888888'   # as figure 3
ICOL = {'grid': COL_GC, 'non-grid spatial': COL_NGS, 'non-spatial': COL_OTHER,
        'putative interneuron': '#d95f02',
        'locked anchored': ANCH_COLOR, 'locked non-anchored': NONANCH_COLOR}

SIGMA = 2.0        # position bins (4 cm), matching Figure 1
# Rate-map colormap. Figure 1 uses viridis, so that is the default and anything
# else is a deliberate departure from the rest of the paper.
CMAP = os.environ.get('FIG2_CMAP', 'viridis')
MAJOR_FILT = 9        # trials, odd; as Figure 1 -- see session_row
# Deliberately NOT M26 D18, which Figure 1 uses as its example.
#
# Chosen so the GRID panel is legible. Selection within a session is still by
# agreement alone (see example_units), but that only picks the best cell the
# session HAS: M25 D23 held just three grid cells in the population and its
# best fired at ~2 Hz, so the panel showed a correct but near-empty map. These
# two sessions have 23 and 34 grid cells, and their top-agreement grid cells
# are also the best-structured ones in the dataset -- M25 D24 cl 160
# (r = 0.92, open-field grid score 1.33, track spatial information 0.83,
# 13.5 Hz peak) and M28 D23 cl 115 (r = 0.69, 0.97, 0.95, 3.9 Hz). Both are
# switching sessions carrying all four identities and both locked classes, as
# the rows require.
EXAMPLES = [(25, 24), (28, 23)]

# r_null: each cell's own circular-shift chance level. It is computed by
# build_per_cell_pc1.py but dropped when per_cell_pc1.csv is written, so it has
# to come from pc1_by_region.csv. Without it panel C has no reference, and the
# reference is not near zero: a cell anchored on most trials correlates with a
# population that is also anchored on most trials whatever the coordination.
NUL = pd.read_csv(f'{PS}/pc1_by_region.csv')[
    ['mouse', 'day', 'cluster_id', 'r_null']]
U = pd.read_csv(f'{PS}/unit_table.csv').merge(
    NUL, on=['mouse', 'day', 'cluster_id'], how='left')
M = U[U.in_population & U.r.notna()].copy()
M = M[M.identity.isin(ORDER)]
print(f'{len(M)} entorhinal cells with an agreement score, '
      f'{M.groupby(["mouse", "day"]).ngroups} sessions, {M.mouse.nunique()} mice')


def per_mouse(df, fn):
    return df.groupby('mouse').apply(fn).dropna()


# ── example units ────────────────────────────────────────────────────────────
def example_units(mo, dy):
    """One unit per identity group: the best-agreeing cell of each type.

    Picking by agreement rather than by eye means the four panels show each
    group at ITS OWN best, so a group that follows the population weakly is
    represented by its strongest case rather than a flattering one -- the
    comparison between panels is then a fair one.
    """
    d = M[(M.mouse == mo) & (M.day == dy)]
    out = {}
    for k in ORDER:
        g = d[d.identity == k].sort_values('r', ascending=False)
        if len(g):
            out[k] = int(g.iloc[0].cluster_id)
    # Locked cells: zero label variance, so r is undefined. Chosen by firing
    # rate, since the point is to SHOW the map -- a silent cell would make a
    # locked map look like an absence of data rather than a stable one.
    lk = U[(U.mouse == mo) & (U.day == dy) & U.in_population & (U.label_sd == 0)]
    for want, tag in ((1.0, 'locked anchored'), (0.0, 'locked non-anchored')):
        g = (lk[lk.cell_frac_anch == want]
             .sort_values('firing_rate', ascending=False))
        if len(g):
            out[tag] = int(g.iloc[0].cluster_id)
    return out


def smooth_map(Mm, sigma=SIGMA):
    """NaN-aware smoothing along POSITION only, per trial -- as in Figure 1."""
    from spatial_manifolds.anchoring import smooth_nanaware
    return np.array([smooth_nanaware(r, sigma=sigma) for r in Mm])


def rate_maps(mo, dy, ids):
    """trial x position rate maps, via the same route the labels were built."""
    import json
    import pynapple as nap
    g = {}
    for c in json.load(open(f'{FIG}/lick_raster_by_trial_type.ipynb'))['cells']:
        if c['cell_type'] != 'code':
            continue
        src = ''.join(c['source'])
        if src.startswith('MOUSE, DAY') or 'INV = pd.DataFrame' in src:
            continue
        exec(compile(src, '<nb>', 'exec'), g)
    bp, cp = g['vr_paths'](mo, dy)
    beh = nap.load_file(bp); clusters = nap.load_file(cp)
    trials, orig = g['clip_trials'](beh['trials'].as_dataframe(), clusters)
    keep = np.isin(beh['trials'].as_dataframe().number.values.astype(int), orig)
    tn, trav = beh['trial_number'], beh['travel']
    TL, NBIN = g['TL'], g['NBIN']
    dt = trav - (float(np.asarray(tn.values)[0]) - 1) * TL
    moving = beh['S'].threshold(3.0, method='above').time_support
    n_tr = len(beh['trials'].as_dataframe())
    tc = nap.compute_1d_tuning_curves(clusters[list(ids)], dt,
                                      nb_bins=n_tr * NBIN,
                                      minmax=[0, n_tr * TL], ep=moving)
    return {c: np.asarray(tc[c]).reshape(n_tr, NBIN)[keep] for c in ids}, TL




# ── figure ───────────────────────────────────────────────────────────────────
TA = ListedColormap([NONANCH_COLOR, ANCH_COLOR])
NORM = BoundaryNorm([-.5, .5, 1.5], 2)

fig = plt.figure(figsize=(11.2, 10.4))
outer = fig.add_gridspec(4, 1, height_ratios=[1.18, 1.18, 1, 1], hspace=.48)


def session_row(spec, mo, dy, first):
    """Raster, PC1, and one rate map per identity, all on the trial axis."""
    z = load_session_labels(mo, dy)
    in_pop = np.isfinite(z['pc1_load'])
    L, load_ = z['labels'][in_pop], z['pc1_load'][in_pop]
    n_tr = L.shape[1]
    # Cells that never leave one mode are separated out to the right. Their PC1
    # loading is ~0 by construction -- a constant row contributes nothing to the
    # decomposition -- so sorting by loading buries them in the middle of the
    # raster, where they read as a washed-out band of cells that follow weakly
    # rather than as cells that never varied at all. A gap of NaN columns
    # renders as figure background and keeps the two groups visibly distinct.
    sd = np.nanstd(L, axis=1)
    varies = sd > 0
    ord_v = np.where(varies)[0][np.argsort(load_[varies])[::-1]]
    lk = np.where(~varies)[0]
    # locked cells grouped by which mode they are stuck in
    frac_lk = np.array([np.nanmean(L[i]) for i in lk]) if len(lk) else np.array([])
    ord_l = lk[np.argsort(-frac_lk)] if len(lk) else lk
    gap = max(3, int(round(.025 * L.shape[0])))
    blocks = [L[ord_v]]
    if len(ord_l):
        blocks += [np.full((gap, n_tr), np.nan), L[ord_l]]
    L = np.vstack(blocks)
    n_var, n_lk = len(ord_v), len(ord_l)
    order = np.arange(L.shape[0])
    picks = example_units(mo, dy)
    maps, TL = rate_maps(mo, dy, list(picks.values()))
    # An empty column between PC1 and the rate maps: wspace is uniform across
    # the row, so the only way to separate the population panels from the
    # single-unit panels is a spacer with no axes drawn in it.
    g = spec.subgridspec(1, 3 + len(picks),
                         width_ratios=[1.05, .26, .22] + [.74] * len(picks),
                         wspace=.24)

    # MAJOR population transitions, computed exactly as Figure 1 computes them:
    # the thresholded population state after a 9-trial median filter, which
    # removes the short flickers. Marking every raw state change would bury the
    # raster under lines -- these sessions carry ten or more runs, half of them
    # only a few trials long -- and filtering first keeps the block structure
    # intact rather than silently dropping a boundary next to a brief run.
    from scipy.ndimage import median_filter
    _st = median_filter((np.asarray(z['frac_anch'], float) > .5).astype(float),
                        size=MAJOR_FILT, mode='nearest') > .5
    trans = list(np.where(np.diff(_st.astype(int)) != 0)[0] + 1)

    ax = fig.add_subplot(g[0])
    ax.imshow(L[order].T, aspect='auto', cmap=TA, norm=NORM,
              interpolation='nearest', extent=[0, L.shape[0], n_tr, 1])
    for tb in trans:
        ax.axhline(tb, color='0.15', lw=.9, ls='--', zorder=4)
    ax.set_ylabel('Trial', fontsize=8)
    ax.set_xlabel('Cell', fontsize=8)
    if n_lk:
        ax.annotate('varies (PC1 order)', (n_var / 2, 1.015),
                    xycoords=('data', 'axes fraction'), ha='center', fontsize=6.2,
                    color='0.3')
        ax.annotate(f'locked ({n_lk})', (L.shape[0] - n_lk / 2, 1.015),
                    xycoords=('data', 'axes fraction'), ha='center', fontsize=6.2,
                    color='0.3')
    # the count lives in the block label, not the title: the title runs under
    # the PC1 panel beside it and collides with its heading
    ax.set_title(f'M{mo} D{dy} — {n_var + n_lk} MEC cells', fontsize=8.5,
                 loc='left', pad=12)
    ax.tick_params(labelsize=7)
    for sp in ax.spines.values():
        sp.set_visible(False)

    # PC1 exactly as Figure 1 draws it: a signed fill, no spines, zero line only
    axp = fig.add_subplot(g[1])
    pc = np.nan_to_num(np.asarray(z['pc1'], float))
    y = np.arange(1, len(pc) + 1)
    axp.fill_betweenx(y, 0, pc, where=pc >= 0, color=ANCH_COLOR, linewidth=0,
                      edgecolor='none', interpolate=True)
    axp.fill_betweenx(y, 0, pc, where=pc < 0, color=NONANCH_COLOR, linewidth=0,
                      edgecolor='none', interpolate=True)
    axp.axvline(0, color='0.4', lw=.7)
    for tb in trans:
        axp.axhline(tb, color='0.15', lw=.9, ls='--', zorder=4)
    axp.set_ylim(n_tr, 1)
    axp.set_xticks([])
    axp.tick_params(labelleft=False, left=False)
    axp.set_title(f'PC1\n({100 * float(z["pc1_var"]):.0f}% var)', fontsize=7.5,
                  pad=4)
    for sp in axp.spines.values():
        sp.set_visible(False)

    for k, (ident, cid) in enumerate(picks.items()):
        axm = fig.add_subplot(g[k + 3])
        Mm = smooth_map(maps[cid])
        vmax = np.nanpercentile(Mm, 99)
        axm.imshow(Mm, aspect='auto', cmap=CMAP, vmin=0, vmax=vmax,
                   interpolation='nearest', extent=[0, TL, n_tr, 1])
        # bounded to the track, like the bottom spine: the anchoring strip
        # beyond TL is a legend, not part of the map, and a line through it
        # reads as data
        for tb in trans:
            axm.axhline(tb, color='0.25', lw=.8, ls='--', zorder=4,
                        xmax=TL / (TL + 15))
        lab = z['labels'][list(z['cluster_id'].astype(int)).index(cid)]
        for t, v in enumerate(lab):
            if np.isfinite(v):
                axm.add_patch(plt.Rectangle((TL + 4, t + 1), 10, 1, clip_on=False,
                                            lw=0, color=ANCH_COLOR if v > .5
                                            else NONANCH_COLOR))
        # peak rate, top right, so the colour scale is quantified per panel
        # the label sits on whatever the colormap puts at its top end, so it
        # flips with a reversed map rather than staying white on a pale panel
        _rev = CMAP.endswith('_r') or CMAP == 'binary'
        axm.annotate(f'{np.nanmax(Mm):.1f} Hz', (.975, .975),
                     xycoords='axes fraction', ha='right', va='top', fontsize=6.2,
                     color='0.1' if _rev else 'white',
                     bbox=dict(facecolor='white' if _rev else '0.15',
                               alpha=.55 if _rev else .45, lw=0, pad=1.1))
        row = U[(U.mouse == mo) & (U.day == dy) & (U.cluster_id == cid)].iloc[0]
        short = {'grid': 'grid', 'putative interneuron': 'interneuron',
                 'non-grid spatial': 'non-grid spatial',
                 'non-spatial': 'non-spatial',
                 'locked anchored': 'locked: always anchored',
                 'locked non-anchored': 'locked: never anchored'}[ident]
        if ident.startswith('locked'):
            # a locked cell still HAS an identity -- naming it stops the group
            # reading as a fifth cell type rather than a state of the others
            short = f'{short}\n({row.identity})'
        sub = ('r undefined' if not np.isfinite(row.r) else f'r = {row.r:+.2f}')
        axm.set_title(f'{short}\n{sub}', fontsize=6.2, loc='left',
                      color=ICOL[ident])
        axm.set_xlim(0, TL + 15)
        axm.set_xlabel('Position (cm)', fontsize=7.5)
        axm.set_xticks([0, 100, 200])
        axm.set_yticklabels([]); axm.tick_params(labelsize=6.5)
        # left and bottom kept: the maps carry real position and trial axes, and
        # a frameless image reads as a picture rather than as plotted data
        axm.spines[['top', 'right']].set_visible(False)
        axm.spines['bottom'].set_bounds(0, TL)   # strip sits beyond the track


for i, (mo, dy) in enumerate(EXAMPLES):
    session_row(outer[i], mo, dy, first=(i == 0))
    # The two example rows carried no panel letter at all, unlike every other
    # panel in the figure. Placed from the gridspec cell rather than from an
    # axes, because each row is a 3 + n_units subgrid and lettering its first
    # axes would put the letter inside the raster.
    _bb = outer[i].get_position(fig)
    fig.text(_bb.x0 - .012, _bb.y1 + .004, 'AB'[i], fontsize=10, weight='bold',
             va='bottom', ha='left')

# ── population panels ────────────────────────────────────────────────────────
# the .16 column is an empty spacer: C is a schematic and D-G are results, and
# wspace is uniform, so the break has to be made with a column
g3 = outer[2].subgridspec(1, 6, width_ratios=[1.12, .16, .98, .98, 1.02, .82],
                          wspace=.56)
# H is a stacked-bar summary and does not need the width of a scatter, so it
# is narrowed and the gap to I widened -- at the old uniform wspace the two
# read as one block
g4 = outer[3].subgridspec(1, 3, width_ratios=[.68, 1.05, 1.30], wspace=.60)
SHORT = ['grid', 'NGS', 'NS', 'int']   # full names collide at this width

def lmm_contrast(df, dv, a, b):
    """dv ~ group + (1|mouse) + (1|mouse:day) for one pair of groups.

    Cells are nested in sessions in mice. A cell-level rank test treats them as
    independent and returns p values orders of magnitude too small; this is the
    same comparison with that structure restored.
    """
    d = df[df.identity.isin([a, b])].dropna(subset=[dv]).copy()
    d['g'] = (d.identity == a).astype(int)
    d['sess'] = d.mouse.astype(str) + 'D' + d.day.astype(str)
    try:
        m = smf.mixedlm(f'{dv} ~ g', d, groups=d['mouse'],
                        vc_formula={'sess': '0 + C(sess)'}).fit(reml=True,
                                                                method='lbfgs')
        p = float(m.pvalues['g'])
        return (p if np.isfinite(p) else np.nan), float(m.params['g'])
    except Exception:
        return np.nan, np.nan


def holm(ps):
    ps = np.asarray(ps, float)
    out = np.full(len(ps), np.nan)
    ok = np.where(np.isfinite(ps))[0]
    run = 0.0
    for rank, idx in enumerate(ok[np.argsort(ps[ok])]):
        run = max(run, ps[idx] * (len(ok) - rank))
        out[idx] = min(run, 1.0)
    return out


def lmm_vs0(df, dv):
    """Is `dv` different from zero, with mouse and session random effects?"""
    d = df.dropna(subset=[dv]).copy()
    d['sess'] = d.mouse.astype(str) + 'D' + d.day.astype(str)
    for kw in (dict(vc_formula={'sess': '0 + C(sess)'}), {}):
        try:
            m = smf.mixedlm(f'{dv} ~ 1', data=d, groups='mouse', **kw).fit(
                reml=True, method='lbfgs')
            p = float(m.pvalues['Intercept'])
            if np.isfinite(p):
                return p
        except Exception:
            pass
    return wilcoxon(d.groupby('mouse')[dv].mean().values).pvalue


Mb = M[M.identity.isin(ORDER)].copy()
Mb['excess'] = Mb.r - Mb.r_null          # agreement above this cell's own chance

# C: how the per-cell null is built. One real cell, worked through.
# Replaces the raw-r boxplot, which showed the same data as D without the
# reference that makes it interpretable -- and the reference is the thing most
# likely to be misread, since chance here sits near +0.15 rather than 0.
def null_demo(spec, mo, dy):
    z = load_session_labels(mo, dy)
    in_pop = np.isfinite(z['pc1_load'])
    P = z['labels'][in_pop]
    pop_ids = list(z['cluster_id'][in_pop])
    frac = np.asarray(z['frac_anch'], float)

    def axis_from(Mx):
        Mc = np.nan_to_num(Mx - np.nanmean(Mx, axis=1, keepdims=True))
        pc = np.linalg.svd(Mc, full_matrices=False)[2][0]
        return -pc if np.corrcoef(pc, np.nan_to_num(frac))[0, 1] < 0 else pc

    # a cell with a clear, typical relationship -- median r among followers
    cand = M[(M.mouse == mo) & (M.day == dy) & (M.cls == 'follows')]
    cid = int(cand.iloc[(cand.r - cand.r.median()).abs().argsort().iloc[0]].cluster_id)
    k = pop_ids.index(cid)
    v = np.nan_to_num(P[k])
    pc = axis_from(np.delete(P, k, axis=0))          # leave-one-out axis
    r = float(np.corrcoef(v, pc)[0, 1])
    rng_ = np.random.default_rng(0)
    shifts = rng_.integers(5, len(v) - 5, 200)
    null = np.array([abs(np.corrcoef(np.roll(v, sh), pc)[0, 1]) for sh in shifts])

    gg = spec.subgridspec(3, 1, height_ratios=[.42, .42, 1], hspace=.55)
    tr = np.arange(1, len(v) + 1)
    axa = fig.add_subplot(gg[0])
    axa.imshow(v[None, :], aspect='auto', cmap=TA, norm=NORM,
               interpolation='nearest', extent=[1, len(v), 0, 1])
    axa.set_yticks([]); axa.set_xticks([])
    axa.set_ylabel('cell', fontsize=6, rotation=0, ha='right', va='center')
    _lp(axa, 'C')
    axa.set_title(f'the null, for one cell (M{mo} D{dy} #{cid})', fontsize=7.5,
                  loc='left')
    for sp in axa.spines.values():
        sp.set_visible(False)
    axb = fig.add_subplot(gg[1])
    axb.imshow(np.roll(v, int(len(v) * .37))[None, :], aspect='auto', cmap=TA,
               norm=NORM, interpolation='nearest', extent=[1, len(v), 0, 1])
    axb.set_yticks([]); axb.set_xticks([])
    axb.set_ylabel('shifted', fontsize=6, rotation=0, ha='right', va='center')
    axb.set_xlabel('Trial  →  circularly shifted, 200×', fontsize=6)
    for sp in axb.spines.values():
        sp.set_visible(False)

    axh = fig.add_subplot(gg[2])
    axh.hist(null, bins=28, color='0.78', lw=0)
    axh.axvline(null.mean(), color='k', lw=1.4, ls='--')
    axh.axvline(abs(r), color=ANCH_COLOR, lw=1.6)
    axh.annotate(f'chance\n{null.mean():.2f}', (null.mean(), axh.get_ylim()[1] * .98),
                 ha='right', va='top', fontsize=5.8, color='0.2')
    axh.annotate(f'observed\n{r:+.2f}', (abs(r), axh.get_ylim()[1] * .98),
                 ha='left', va='top', fontsize=5.8, color=ANCH_COLOR)
    axh.set_xlabel('|r| with the population axis', fontsize=7)
    axh.set_ylabel('shifts', fontsize=7.5)
    axh.tick_params(labelsize=6.5)
    axh.spines[['top', 'right']].set_visible(False)
    # key for the two strips above. It hangs off the histogram rather than off
    # the strips themselves: placed under the strips it falls inside the
    # histogram's axes area and is painted over by it. "Trial" is explicit
    # because the row-H legend below uses the same two colours for a cell's
    # permanent state rather than one trial's label.
    axh.legend(handles=[Patch(facecolor=ANCH_COLOR, label='anchored trial'),
                        Patch(facecolor=NONANCH_COLOR, label='non-anchored trial')],
               loc='upper center', bbox_to_anchor=(.5, -.52), ncol=2,
               fontsize=5.8, frameon=False, handlelength=.9, handleheight=.9,
               handletextpad=.4, columnspacing=.9, borderpad=0)


Mb = M[M.identity.isin(ORDER)].copy()
Mb['excess'] = Mb.r - Mb.r_null          # agreement above this cell's own chance
null_demo(g3[0], *EXAMPLES[0])

# D: the same thing as excess over chance, with the test against chance
ax = fig.add_subplot(g3[2])
P0 = {k: lmm_vs0(Mb[Mb.identity == k], 'excess') for k in ORDER}
sns.boxplot(data=Mb, x='identity', y='excess', order=ORDER, ax=ax,
            showfliers=False, palette=[ICOL[k] for k in ORDER], width=.62,
            linewidth=.9, medianprops=dict(color='k', lw=1.4))
ax.axhline(0, color='k', lw=1.4, ls='--', zorder=5)
for i_, k in enumerate(ORDER):
    pv = P0[k]
    star = ('n.s.' if not np.isfinite(pv) or pv > .05 else
            '*' if pv > .01 else '**' if pv > .001 else '***')
    v = Mb[Mb.identity == k].excess.dropna()
    ax.annotate(star, (i_, .97), xycoords=('data', 'axes fraction'), ha='center',
                va='top', fontsize=7, color='0.15')
    pm = Mb[Mb.identity == k].groupby('mouse').excess.mean().dropna()
    ax.annotate(f'{v.mean():+.3f}\n{int((pm > 0).sum())}/{len(pm)} mice',
                (i_, .02), xycoords=('data', 'axes fraction'), ha='center',
                fontsize=5.6, color='0.3', linespacing=1.2)
ax.set_xticks(range(4)); ax.set_xticklabels(SHORT, fontsize=6.5)
ax.set_xlabel('')
ax.set_ylabel('Excess over chance\n(r − own null)', fontsize=8)
_lp(ax, 'D')
ax.set_title('only grid and interneurons\nexceed it', fontsize=7.5, loc='left')
ax.tick_params(labelsize=7); ax.spines[['top', 'right']].set_visible(False)
print('  excess over chance, LMM vs 0: ' +
      '  '.join(f'{SHORT[i_]} {Mb[Mb.identity==k].excess.mean():+.3f} '
                f'(p={P0[k]:.2g})' for i_, k in enumerate(ORDER)))

# E: how many cells individually beat their own null, per mouse
ax = fig.add_subplot(g3[3])
rng = np.random.default_rng(0)
for i_, k in enumerate(ORDER):
    pm = (M[M.identity == k].groupby('mouse').cls
          .apply(lambda v: (v == 'follows').mean()).dropna())
    ax.bar(i_, pm.mean(), width=.66, color=ICOL[k], lw=.7, edgecolor='k', zorder=1)
    ax.scatter(i_ + rng.uniform(-.12, .12, len(pm)), pm, s=8, color='0.15',
               alpha=.85, lw=0, zorder=3)
    ax.errorbar(i_, pm.mean(), yerr=pm.std(ddof=1) / np.sqrt(len(pm)), color='k',
                lw=1.2, capsize=3, zorder=4)
ax.axhline(.05, color='k', lw=1.3, ls='--', zorder=5)
ax.annotate('chance (5%)', (3.45, .075), ha='right', fontsize=6, color='0.2')
ax.set_xticks(range(4)); ax.set_xticklabels(SHORT, fontsize=6.5)
ax.set_ylabel('Fraction beating own null', fontsize=8)
_lp(ax, 'E')
ax.set_title(f'per mouse (n={M.mouse.nunique()})', fontsize=8, loc='left')
ax.tick_params(labelsize=7); ax.spines[['top', 'right']].set_visible(False)

# F: the same question in time -- when the population switches, what do the
# identities do? The alignment lives in transition_profiles.py so it has one
# definition. Each cell is timed against a population state recomputed WITHOUT
# it, since the population is otherwise partly made of the cell being timed.
ax = fig.add_subplot(g3[4])
import transition_profiles as TP
TP.WIN, TP.FILT = 6, 1
TR = TP.collect(loo=True)
_lags = np.arange(-TP.WIN, TP.WIN)
_keep = _lags >= -5                       # show -5 to +5 only
for k in ORDER:
    pr = np.vstack(TR[TR.identity == k].profile.values)
    mu = pr.mean(0)[_keep]
    se = (pr.std(0, ddof=1) / np.sqrt(len(pr)))[_keep]
    ax.fill_between(_lags[_keep], mu - se, mu + se, color=ICOL[k], alpha=.25,
                    linewidth=0, edgecolor='none')
    ax.plot(_lags[_keep], mu, color=ICOL[k], lw=1.5)
ax.axvline(0, color='0.35', lw=1.1, ls='--')
ax.set_xlim(-5, 5); ax.set_xticks([-4, -2, 0, 2, 4])
ax.set_xlabel('Trial relative to population switch', fontsize=7.5)
ax.set_ylabel('P(cell anchored)', fontsize=8)
_lp(ax, 'F')
ax.set_title('all identities switch together', fontsize=8, loc='left')
ax.tick_params(labelsize=7); ax.spines[['top', 'right']].set_visible(False)

# G: which identities agree with EACH OTHER. The transition profiles suggest
# grid cells and interneurons move together; this tests it directly.
# Excess over an independent circular-shift null, so a pair of cells that are
# both anchored most of the time does not score for that alone.
IDENT = {(m, d, c): i for m, d, c, i in
         zip(U.mouse, U.day, U.cluster_id, U.identity)}
AG = f'{PS}/identity_agreement.csv'
if os.path.exists(AG):
    A2 = pd.read_csv(AG)
else:
    _rng = np.random.default_rng(0)
    _rows = []
    for (mo, dy), _ in U[U.in_population & U.sess_switches].groupby(['mouse', 'day']):
        z = load_session_labels(int(mo), int(dy))
        L2, ids2 = z['labels'], z['cluster_id'].astype(int)
        keep = np.isfinite(z['pc1_load']) & (np.nanstd(L2, axis=1) > 0)
        V = np.nan_to_num(L2[keep])
        if len(V) < 8:
            continue
        lab = np.array([IDENT.get((mo, dy, int(c))) for c in ids2[keep]])
        Cm = np.corrcoef(V)
        Nm = np.mean([np.corrcoef(np.array(
            [np.roll(v, int(_rng.integers(5, V.shape[1] - 5))) for v in V]))
            for _ in range(20)], axis=0)
        for a in ORDER:
            for b in ORDER:
                ia, ib = np.where(lab == a)[0], np.where(lab == b)[0]
                if len(ia) < 2 or len(ib) < 2:
                    continue
                m_ = np.ix_(ia, ib)
                co, cn = Cm[m_].copy(), Nm[m_].copy()
                if a == b:
                    iu = ~np.eye(len(ia), dtype=bool)
                    co, cn = co[iu], cn[iu]
                _rows.append(dict(mouse=mo, day=dy, a=a, b=b,
                                  excess=float(np.nanmean(co) - np.nanmean(cn))))
    A2 = pd.DataFrame(_rows)
    A2.to_csv(AG, index=False)

ax = fig.add_subplot(g3[5])
Mx = (A2.groupby(['a', 'b']).excess.mean().unstack()
      .reindex(index=ORDER, columns=ORDER))
im = ax.imshow(Mx.values, cmap='magma', vmin=0, vmax=np.nanmax(Mx.values))
cmap_ = plt.get_cmap('magma')
for i_ in range(4):
    for j_ in range(4):
        v = Mx.values[i_, j_]
        if not np.isfinite(v):
            continue
        rgb = cmap_(v / np.nanmax(Mx.values))[:3]
        lum = .299 * rgb[0] + .587 * rgb[1] + .114 * rgb[2]
        ax.text(j_, i_, f'{v:.3f}', ha='center', va='center', fontsize=5.6,
                color='0.1' if lum > .55 else 'white')
ax.set_xticks(range(4)); ax.set_xticklabels(SHORT, fontsize=6.2)
ax.set_yticks(range(4)); ax.set_yticklabels(SHORT, fontsize=6.2)
for t, k in zip(ax.get_xticklabels(), ORDER):
    t.set_color(ICOL[k])
for t, k in zip(ax.get_yticklabels(), ORDER):
    t.set_color(ICOL[k])
_lp(ax, 'G')
ax.set_title('agreement between\nidentities (over null)', fontsize=7.5,
             loc='left')
ax.tick_params(length=0, labelsize=6.2)
_gi = A2[((A2.a == 'grid') & (A2.b == 'putative interneuron'))].excess
_off = A2[(A2.a != A2.b) & ~(((A2.a == 'grid') & (A2.b == 'putative interneuron')) |
                             ((A2.a == 'putative interneuron') & (A2.b == 'grid')))].excess
print(f'  grid-interneuron agreement {_gi.mean():+.4f} (n={len(_gi)} sessions, '
      f'p={wilcoxon(_gi).pvalue:.2g}); other off-diagonal {_off.mean():+.4f}')

# ── H: locked cells ──────────────────────────────────────────────────────────
# SWITCHING SESSIONS ONLY. The question is which cells stayed locked while the
# population moved, so a session whose population never moved cannot answer it:
# every cell there is locked by construction, and including them measures how
# often a session is single-state rather than how often a cell is. It inflates
# the locked fraction from 27% to 31% and, because single-state sessions are not
# a random sample, would let session composition leak into the identity
# comparison.
P = U[U.in_population & U.identity.isin(ORDER) & U.sess_switches].copy()
P['lockstate'] = np.where(P.label_sd != 0, 'varies',
                          np.where(P.cell_frac_anch == 1, 'always', 'never'))
ct = pd.crosstab(P.identity, P.lockstate).reindex(ORDER)
row = ct.div(ct.sum(1), axis=0)

ax = fig.add_subplot(g4[0])
left = np.zeros(4)
LOCK_LAB = {'always': 'always anchored', 'never': 'always non-anchored',
            'varies': 'varies'}
for st, c in (('always', ANCH_COLOR), ('never', NONANCH_COLOR), ('varies', '0.8')):
    ax.barh(range(4), row[st], left=left, color=c, height=.66, lw=0,
            label=LOCK_LAB[st])
    left += row[st].values
ax.legend(fontsize=6, frameon=False, loc='upper center',
          bbox_to_anchor=(.5, -.28), ncol=1, handlelength=1.2,
          handletextpad=.5, labelspacing=.35)
ax.set_yticks(range(4)); ax.set_yticklabels(SHORT, fontsize=6.5)
ax.set_xlim(0, 1); ax.set_xlabel('Fraction of cells', fontsize=7.5)
_lp(ax, 'H')
ax.set_title(f'locked state by identity\n{100*(P.lockstate!="varies").mean():.0f}% '
             f'locked, {P.groupby(["mouse","day"]).ngroups} switching sessions',
             fontsize=6.8, loc='left')
ax.tick_params(labelsize=7); ax.spines[['top', 'right']].set_visible(False)

# the asymmetry, per mouse so it is not one animal
ax = fig.add_subplot(g4[1])
for i, k in enumerate(ORDER):
    pm = []
    for mo, g in P[P.identity == k].groupby('mouse'):
        a = (g.lockstate == 'always').sum(); n = (g.lockstate == 'never').sum()
        if a + n >= 8:
            pm.append(np.log2((a + .5) / (n + .5)))
    if not pm:
        continue
    ax.scatter(np.full(len(pm), i) + rng.uniform(-.12, .12, len(pm)), pm, s=14,
               color=ICOL[k], lw=0, zorder=3)
    ax.plot([i - .22, i + .22], [np.mean(pm)] * 2, color='k', lw=2, zorder=4)
ax.axhline(0, color='0.5', lw=.9)
ax.set_xticks(range(4)); ax.set_xticklabels(SHORT, fontsize=6.5)
ax.set_ylabel('log2  always anchored :\nalways non-anchored', fontsize=7)
_lp(ax, 'I')
ax.set_title('lock direction, per mouse', fontsize=7.5, loc='left')
ax.tick_params(labelsize=7); ax.spines[['top', 'right']].set_visible(False)

# ── I: why interneurons lock ON — speed coding ──────────────────────────────
# A cell tuned to running speed has a profile that repeats on every trial,
# because the speed profile along the track is itself stereotyped. The
# classifier compares each trial's rate map against the cell's own template, so
# such a cell is called anchored on every trial and never varies. That predicts
# the locked-ON cells should be the speed-modulated ones -- and entorhinal speed
# cells are predominantly fast-spiking interneurons, which is exactly the class
# that locks on. Speed scores come from the OPEN FIELD, so they cannot have been
# shaped by the task dynamics being explained.
ax = fig.add_subplot(g4[2])
SP = P.dropna(subset=['speed_r', 'speed_p']).copy()
SP['cls'] = np.where(SP.putative_int, 'interneuron', 'principal')
STATES = ['always', 'varies', 'never']
# H's colours, so the lock states mean the same thing in both panels; cells are
# GROUPED BY CLASS rather than by state, so the within-class gradient -- which
# is the comparison the panel makes -- is read along a contiguous run of bars
SCOL = {'always': ANCH_COLOR, 'varies': '0.8', 'never': NONANCH_COLOR}
SLAB = {'always': 'always\nanchored', 'varies': 'varies', 'never': 'always\nnon-anch.'}
xs, cols, labs, stats = [], [], [], {}
pos = 0.0
for cls in ('interneuron', 'principal'):
    vals = []
    for st in STATES:
        g = SP[(SP.cls == cls) & (SP.lockstate == st)].speed_r
        if len(g) < 5:
            continue
        med = g.median(); se = g.std() / np.sqrt(len(g))
        ax.bar(pos, med, .80, color=SCOL[st], linewidth=.6, edgecolor='0.35')
        ax.errorbar(pos, med, yerr=se, fmt='none', ecolor='0.25', elinewidth=.9,
                    capsize=0)
        ax.text(pos, med + se + .004, f'{len(g)}', ha='center', fontsize=4.8,
                color='0.35')
        xs.append(pos); labs.append(SLAB[st]); vals.append((st, g, med, se))
        pos += 1.0
    stats[cls] = {st: g for st, g, _, _ in vals}
    pos += 0.9                                   # gap between the two classes
# within-class always vs varies, which is the test the panel is making
ptxt = []
for cls in ('interneuron', 'principal'):
    a, v = stats[cls].get('always'), stats[cls].get('varies')
    if a is None or v is None:
        continue
    pv = mannwhitneyu(a, v).pvalue
    ptxt.append(f'{cls[:3]} {pv:.1g}')
    # bracket the two bars being compared
    i0 = xs[0] if cls == 'interneuron' else xs[3]
    i1 = i0 + 1.0
    yb = max(a.median(), v.median()) * 1.55 + .012
    ax.plot([i0, i0, i1, i1], [yb - .004, yb, yb, yb - .004], color='0.35', lw=.7)
    ax.text((i0 + i1) / 2, yb + .002,
            f'p = {pv:.1g}' if pv >= .001 else f'p = {pv:.0e}'.replace('e-0', 'e-'),
            ha='center', fontsize=5.4, color='0.2')
ax.set_xticks(xs); ax.set_xticklabels(labs, fontsize=5.6)
ax.set_ylabel('open-field speed score', fontsize=7)
ax.set_ylim(top=ax.get_ylim()[1] * 1.30)
# class names under their own group of bars
for cls, cx in (('interneuron', np.mean(xs[:3])), ('principal', np.mean(xs[3:]))):
    ax.annotate(cls, (cx, -.34), xycoords=('data', 'axes fraction'), ha='center',
                fontsize=6.4, weight='bold',
                color=ICOL['putative interneuron'] if cls == 'interneuron' else '0.35')
_lp(ax, 'J')
ax.set_title('locked-ON cells are the speed cells', fontsize=6.8,
             loc='left')
ax.tick_params(labelsize=7); ax.spines[['top', 'right']].set_visible(False)
_a = stats['interneuron']['always']; _v = stats['interneuron']['varies']
_p = mannwhitneyu(_a, _v).pvalue
print(f'  speed by lock state (interneurons): always {_a.median():+.4f} (n={len(_a)}), '
      f'varies {_v.median():+.4f} (n={len(_v)}), p={_p:.3g}')

print(f'  locked: {100*(P.lockstate!="varies").mean():.1f}% of cells; '
      f'interneuron always:never = {ct.loc["putative interneuron","always"]}:'
      f'{ct.loc["putative interneuron","never"]}, '
      f'non-spatial = {ct.loc["non-spatial","always"]}:{ct.loc["non-spatial","never"]}')

out = (f'{FIG}/fig2_single_units.pdf' if CMAP == 'viridis'
       else f'{FIG}/fig2_single_units_{CMAP}.pdf')
plt.savefig(out, dpi=200, bbox_inches='tight')
print(f'saved {out}')
