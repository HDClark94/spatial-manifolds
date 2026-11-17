"""Is the anchoring state ONE state, and how well does PC1 capture it?

Figure 7 shows that each region has a coherent axis and that the axes agree. It
does not show that a SINGLE axis is an adequate description of a region's label
matrix. PC1 is by construction the largest component of whatever is there; that
it exists says nothing about whether the rest is noise or a second state.

WHY THE OBVIOUS MEASURE DOES NOT WORK. The natural statistic is the fraction of
variance PC1 explains, and it is uninterpretable across regions here. Variance
explained by the top component of an n x T matrix rises as n falls -- with few
cells the leading eigenvalue absorbs a larger share of the total by chance
alone -- and MEC contributes hundreds of label-varying cells where visual cortex
contributes tens and cerebellum fewer still. A plot of in-sample PC1 variance
explained by region would mostly be a plot of how many cells each region
contributed, running in the direction that makes the WORST-sampled region look
the most one-dimensional.

THE FIX IS TO SCORE OUT OF SAMPLE. Build the axis from one half of a region's
cells and measure how much variance it explains in the OTHER half:

    VE_oos(u) = || B_centred . u ||^2 / || B_centred ||_F^2

for a unit-norm trial-space direction u estimated from A. A direction fitted to
noise in A explains nothing in B, so this cannot be inflated by small n the way
in-sample variance explained is. It still IMPROVES with n, because the axis is
better estimated, but it improves toward the truth rather than away from it, and
its chance level does not move with n at all.

TWO PANELS, ANSWERING TWO DIFFERENT QUESTIONS.

A  NEURON-DROPPING CURVE. VE_oos of PC1 against the number of cells used to
   build the axis, per region, with the shuffle null. This does not hide the
   cell-count dependence behind a single matched number -- it shows it, so that
   any two regions can be compared at a common n and so that a region which has
   not yet plateaued is visible as one that has not yet plateaued.

B  IS IT ONE STATE OR SEVERAL? VE_oos by component index at matched n. PC1, PC2,
   PC3 ... are orthogonal by construction, so if the population carried two
   independent states both would generalise out of sample. If only the first
   clears the null, a single axis is an adequate description and "the population
   anchoring state" is a well-posed object. This is the panel that licenses the
   phrase, and it is a stronger test than any amount of in-sample variance.

THE NULL preserves each cell's anchored fraction AND its temporal
autocorrelation by circularly shifting its label sequence, destroying only the
alignment BETWEEN cells. That matters: anchoring comes in long blocks, so an
i.i.d. shuffle would be far too easy a null -- it would destroy the block
structure that gives a single cell's labels their low-dimensional character and
would overstate how much of the real VE_oos reflects cross-cell agreement.
Because the null is computed at the same n, from the same cells, with the same
marginals, the cell-count bias cancels in the difference.

WHAT A SECOND MAP WOULD LOOK LIKE (added 2026-11-01). A population could hold
more than one stable map, each expressed by a subset of cells and switching on
its own schedule (Low et al., 2021); with enough cells that would appear as a
second component that generalises out of sample. Panels A, B and E simulate it:
cells that follow their population's block-structured state on a fraction
SIM_FOLLOW of trials and otherwise run their own blocky labels, SIM_FOLLOW set
so a one-state simulation matches MEC's PC1 excess at MATCH_N cells per half
(0.034 against 0.035). Two populations, 50/50 or 75/25, each with its own
independent state, put a clear PC2 above its null, and it grows with the cells
used; MEC's PC2 stays at the null at every n. The simulations go through the
same out-of-sample code and the same circular-shift null as the data.

Writes data/population_state/region_state_dimensionality.csv
       data/population_state/region_state_dimensionality_sim.csv
       fig6_supp_dimensionality.pdf
"""
import os
import sys
import warnings

import numpy as np
from scipy.ndimage import gaussian_filter1d
import pandas as pd

warnings.filterwarnings('ignore')
import matplotlib

matplotlib.use('Agg')
import matplotlib.pyplot as plt

sys.path.insert(0, '/Users/harryclark/Documents/spatial-manifolds/src')
from spatial_manifolds.anchoring import load_session_labels

plt.rcParams['font.family'] = 'Arial'
# mathtext must resolve to Arial, not the DejaVu default, or rho/times and any
# bold panel letters render in a different face from the rest of the figure
plt.rcParams['mathtext.fontset'] = 'custom'
plt.rcParams['mathtext.rm'] = 'Arial'
plt.rcParams['mathtext.it'] = 'Arial:italic'
plt.rcParams['mathtext.bf'] = 'Arial:bold'
plt.rcParams['pdf.fonttype'] = 42

ROOT = '/Users/harryclark/Documents/spatial-manifolds'
FIG = f'{ROOT}/scripts/figures/AnchorDynamics2026'
PS = f'{ROOT}/data/population_state'
OUT_CSV = f'{PS}/region_state_dimensionality.csv'
OUT_PDF = f'{FIG}/fig6_supp_dimensionality.pdf'
OUT_SIM = f'{PS}/region_state_dimensionality_sim.csv'
SIM_CELLS, SIM_TRIALS, SIM_SESS, SIM_REP = 100, 220, 30, 30
SIM_FOLLOW = .30          # calibrated: one-state PC1 excess 0.034 at n = 8 (MEC 0.035)
SIM_BLOCK, SIM_OWN_BLOCK = 15, 8     # mean block length (trials) of a state, of a cell's own labels
SIM_NGRID = [4, 8, 16, 32, 48]
# greys: the regions own the colours in C and D (subicular is orange)
SCENARIOS = [('one state', (1.0,), '0.1'), ('two states, 50/50', (.5, .5), '0.45'),
             ('two states, 75/25', (.75, .25), '0.72')]

NGRID = [4, 6, 8, 12, 16, 24, 32, 48, 64]
N_REP = 60            # resamples per (session, region, n)
N_COMP = 5            # components scored out of sample
# Panel A only: an illustration, not a calibrated simulation. Each cell follows
# its population's gate on SIM_SHOW_FOLLOW of trials, so the two gates are
# visible, and the session is chosen (seed) so the two gates mostly disagree
# (state r = -0.37); B and E use the calibrated SIM_FOLLOW.
SIM_SHOW_FOLLOW, SIM_SHOW_SEED = .9, 25
MATCH_N = 8           # cells per half for the by-component panel
MIN_TRIALS = 20
MIN_SESS = 5          # sessions required before a point is drawn
MIN_SESS_FAINT = 3    # drawn, but dashed and labelled, between these two
REGIONS = ['MEC', 'SUB', 'VIS', 'CB']
RCOL = {'MEC': '#7b4173', 'VIS': '#3f9b4f', 'SUB': '#c0723a', 'CB': '#4c72b0'}
RNAME = {'MEC': 'MEC', 'SUB': 'subicular', 'VIS': 'visual', 'CB': 'cerebellum'}


def fam(r):
    r = str(r)
    if r.startswith('ENTm'):
        return 'MEC'
    if any(r.startswith(x) for x in ('PRE', 'POST', 'PAR', 'SUB')):
        return 'SUB'
    if r.startswith('VIS'):
        return 'VIS'
    if any(k in r for k in ('CENT', 'CUL', 'ARB', 'DEC', 'FOTU', 'PYR', 'SIM',
                            'AN', 'PRM', 'COPY', 'FL', 'NOD', 'UVU')):
        return 'CB'
    return 'other'


def centred(M):
    """Cells centred on their own mean across trials, NaNs to zero after."""
    return np.nan_to_num(M - np.nanmean(M, axis=1, keepdims=True))


def ve_out_of_sample(A, B, k=N_COMP):
    """Variance of B explained by each of the top-k trial-space axes of A.

    The axes come from A alone and are orthonormal, so this is an honest
    out-of-sample measure: a component fitted to noise in A projects B onto an
    arbitrary direction and captures the chance share, whatever n was used.
    """
    Ac, Bc = centred(A), centred(B)
    tot = float((Bc ** 2).sum())
    if tot <= 0:
        return [np.nan] * k
    Vt = np.linalg.svd(Ac, full_matrices=False)[2]
    out = []
    for i in range(k):
        if i >= Vt.shape[0]:
            out.append(np.nan); continue
        out.append(float(((Bc @ Vt[i]) ** 2).sum() / tot))
    return out


def shift_null(M, rng):
    """Circularly shift each cell independently.

    Keeps each cell's anchored fraction and its run structure, breaks only the
    alignment between cells -- which is exactly the quantity PC1 is meant to be
    measuring.
    """
    T = M.shape[1]
    return np.array([np.roll(row, rng.integers(1, T)) for row in M])


def run_session(mo, dy, C, rng):
    z = load_session_labels(mo, dy)
    if z is None:
        return []
    L, ids = z['labels'], [int(c) for c in z['cluster_id']]
    if L.shape[1] < MIN_TRIALS:
        return []
    g = C[(C.mouse == mo) & (C.day == dy)].set_index('cluster_id').macro.to_dict()
    byreg = {}
    for k, c in enumerate(ids):
        v = L[k]
        if np.nanstd(v) == 0 or np.all(np.isnan(v)):
            continue
        r = g.get(c)
        if r in REGIONS:
            byreg.setdefault(r, []).append(v)

    rows = []
    for r, cells in byreg.items():
        M = np.array(cells)
        navail = len(M)
        for n in NGRID:
            if 2 * n > navail:
                continue
            obs = np.full((N_REP, N_COMP), np.nan)
            nul = np.full((N_REP, N_COMP), np.nan)
            for rep in range(N_REP):
                p = rng.permutation(navail)[:2 * n]
                A, B = M[p[:n]], M[p[n:]]
                obs[rep] = ve_out_of_sample(A, B)
                Ms = shift_null(M[p], rng)
                nul[rep] = ve_out_of_sample(Ms[:n], Ms[n:])
            for comp in range(N_COMP):
                rows.append(dict(mouse=mo, day=dy, region=r, n=n,
                                 comp=comp + 1, n_avail=navail,
                                 n_trials=L.shape[1],
                                 ve=np.nanmean(obs[:, comp]),
                                 ve_null=np.nanmean(nul[:, comp]),
                                 # Upper tail of this session's OWN null, so a
                                 # component can be called present or absent per
                                 # session rather than only on a group mean.
                                 # Used by session_dimensionality(), which is
                                 # reported to stdout but deliberately not
                                 # plotted -- see its docstring.
                                 ve_null_p95=np.nanpercentile(nul[:, comp], 95),
                                 ve_sd=np.nanstd(obs[:, comp])))
    return rows


def session_dimensionality(D, at_n=None, crit='p95'):
    """Leading consecutive components that beat that session's own null.

    REPORTED TO STDOUT, NOT PLOTTED. This was briefly a third panel of pie
    charts, one per region, and it was removed: the cross-region comparison it
    invites cannot be made. Power to resolve a SECOND dimension scales with cell
    count, and the median cells per half runs from 48 in MEC to 10 in visual
    cortex, so MEC's higher rate of two-dimensional sessions is what that
    sampling predicts rather than a property of entorhinal cortex -- and the
    16 of 28 visual sessions registering no state at all measures the sampling,
    not visual cortex. The verdict is also more sensitive to the pass criterion
    than to the region (see below). Panels A and B answer the dimensionality
    question without either problem, so the numbers are kept as a diagnostic and
    the plot is not.

    AT THE LARGEST n EACH SESSION SUPPORTS, not at MATCH_N. Matching cell count
    is required to COMPARE regions fairly (panel B), but it is the wrong choice
    for asking how many dimensions a session has: at n = 8 a single session has
    almost no power, and the first version of this panel reported 68% of MEC
    sessions as having no detectable state at all -- against a pooled PC1 at
    p = 3e-16. That number was measuring the cell count, not the data.

    Consecutive from PC1 on purpose: components are ordered by variance, so a
    session whose PC1 fails but whose PC3 happens to clear is noise, not a
    three-dimensional state, and counting clearances anywhere in the list would
    score it as one. Sessions where even PC1 fails are kept as a category rather
    than dropped -- "no detectable state" is an outcome, and discarding them
    would inflate every region's apparent dimensionality.

    THE CRITERION MATTERS AND IS DELIBERATELY THE STRICT ONE. Calling a
    component present whenever its mean exceeds the null MEAN is far too
    liberal: a null component sits above its own mean half the time, so runs of
    them accumulate, and that criterion classifies 26 of 60 MEC sessions as
    three-dimensional or more. Against the 95th percentile of the session's own
    null the same data give 2 of 60. We use the strict criterion and report the
    liberal one alongside, because the difference between them is larger than
    any difference between regions.
    """
    d = D if at_n is None else D[D.n == at_n]
    if at_n is None:
        keep = d.groupby(['mouse', 'day', 'region']).n.transform('max') == d.n
        d = d[keep]
    out = []
    for (mo, dy, r), g in d.groupby(['mouse', 'day', 'region']):
        g = g.sort_values('comp')
        ref = g.ve_null_p95.values if crit == 'p95' else g.ve_null.values
        k = 0
        for ok in (g.ve.values > ref):
            if not ok:
                break
            k += 1
        out.append(dict(mouse=mo, day=dy, region=r, dim=k, n=int(g.n.iloc[0]),
                        dim_cat='none' if k == 0 else ('3+' if k >= 3 else str(k))))
    return pd.DataFrame(out)


# ── simulation: what one state and two states look like to the same analysis ──
def sim_blocks(T, mean, rng):
    s_ = np.empty(T); t = 0; v = rng.random() < .5
    while t < T:
        L = rng.geometric(1 / mean); s_[t:t + L] = v; t += L; v = not v
    return s_


def sim_session(rng, groups, follow=SIM_FOLLOW, N=SIM_CELLS, T=SIM_TRIALS):
    """Cells in len(groups) populations, each with its own block-structured
    state; a cell follows its population's state on a fraction `follow` of trials
    and otherwise runs its own blocky labels. Returns labels, population of each
    cell and the populations' states."""
    sizes = np.round(np.array(groups) * N).astype(int)
    M, pop, states = [], [], []
    for g, n in enumerate(sizes):
        st = sim_blocks(T, SIM_BLOCK, rng); states.append(st)
        for _ in range(n):
            f = rng.random(T) < follow
            M.append(np.where(f, st, sim_blocks(T, SIM_OWN_BLOCK, rng))); pop.append(g)
    return np.array(M), np.array(pop), np.array(states)


def simulate():
    rng = np.random.default_rng(1)
    rows = []
    for name, groups, _ in SCENARIOS:
        for k in range(SIM_SESS):
            M = sim_session(rng, groups)[0]
            for n in SIM_NGRID:
                obs, nul = [], []
                for _ in range(SIM_REP):
                    p = rng.permutation(len(M))[:2 * n]
                    obs.append(ve_out_of_sample(M[p[:n]], M[p[n:]]))
                    Ms = shift_null(M[p], rng)
                    nul.append(ve_out_of_sample(Ms[:n], Ms[n:]))
                for c in range(N_COMP):
                    rows.append(dict(scenario=name, sess=k, n=n, comp=c + 1,
                                     excess=np.nanmean(np.array(obs)[:, c]) - np.nanmean(np.array(nul)[:, c])))
        print(f'  simulated: {name}', flush=True)
    S = pd.DataFrame(rows)
    S.to_csv(OUT_SIM, index=False)
    return S


def main():
    # --replot rebuilds the figure from the saved CSV. The resampling loop is
    # minutes of work and is unchanged by any layout decision, so iterating on
    # the figure must not require recomputing it.
    if '--replot' in sys.argv:
        D = pd.read_csv(OUT_CSV)
        S = pd.read_csv(OUT_SIM) if os.path.exists(OUT_SIM) and '--resim' not in sys.argv else simulate()
        make_figure(D, S)
        report(D)
        return
    C = pd.read_csv(f'{ROOT}/data/cell_classifications_v2.csv')
    C['macro'] = C.brain_region.map(fam)
    rng = np.random.default_rng(0)
    sess = sorted({(int(m), int(d)) for m, d in
                   C[['mouse', 'day']].drop_duplicates().values})
    rows = []
    for k, (mo, dy) in enumerate(sess, 1):
        r = run_session(mo, dy, C, rng)
        rows += r
        print(f'[{k}/{len(sess)}] M{mo}D{dy} {len(r)} rows', flush=True)
    D = pd.DataFrame(rows)
    if D.empty:
        sys.exit('no rows -- check labels directory')
    D['excess'] = D.ve - D.ve_null
    D.to_csv(OUT_CSV, index=False)
    print(f'\nwrote {OUT_CSV} ({len(D)} rows, '
          f'{D.drop_duplicates(["mouse", "day"]).shape[0]} sessions)')
    make_figure(D, simulate())
    report(D)


def make_figure(D, S):
    # ---------------------------------------------------------------- figure
    fig = plt.figure(figsize=(9.6, 7.7))
    gs = fig.add_gridspec(3, 1, height_ratios=[.5, 1, 1], hspace=.42)
    top = gs[1].subgridspec(1, 2, width_ratios=[1.6, 1], wspace=.35)
    bot = gs[2].subgridspec(1, 3, wspace=.48)
    sim_panels(fig, gs[0], top, S)
    axes = [fig.add_subplot(bot[0]), fig.add_subplot(bot[1])]

    ax = axes[0]
    P1 = D[D.comp == 1]
    for r in REGIONS:
        d = P1[P1.region == r]
        if d.empty:
            continue
        g = d.groupby('n').ve.agg(['mean', 'sem', 'count'])
        # Points resting on very few sessions are drawn dashed and labelled
        # rather than silently included: the subicular and cerebellar curves run
        # out of sessions long before MEC does, and a solid line at n = 24 built
        # from 3 sessions would read as the same kind of evidence as a solid
        # line at n = 24 built from 53.
        solid = g[g['count'] >= MIN_SESS]
        faint = g[(g['count'] >= MIN_SESS_FAINT) & (g['count'] < MIN_SESS)]
        if solid.empty and faint.empty:
            continue
        if not solid.empty:
            ax.fill_between(solid.index, solid['mean'] - solid['sem'],
                            solid['mean'] + solid['sem'], color=RCOL[r],
                            alpha=.22, linewidth=0, edgecolor='none')
            ax.plot(solid.index, solid['mean'], color=RCOL[r], lw=1.3,
                    marker='o', ms=2.6,
                    label=f'{RNAME[r]} ({int(solid["count"].max())} sess.)')
        if not faint.empty:
            lab = None if not solid.empty else \
                f'{RNAME[r]} ({int(faint["count"].max())} sess. only)'
            ax.plot(faint.index, faint['mean'], color=RCOL[r], lw=1.0, ls=':',
                    marker='o', ms=2.0, alpha=.55, label=lab)
    gn = P1.groupby('n').ve_null.mean()
    ax.plot(gn.index, gn.values, color='0.45', lw=1.0, ls='--',
            label='circular-shift null')
    # Scale to the well-sampled points. The subicular curve has one 3-session
    # point at n = 24 sitting three times above everything else; left in the
    # limits it compresses every solid curve into the lower third of the panel
    # to show one value that is labelled as unreliable anyway. It is clipped
    # rather than deleted -- the CSV keeps it and the faint tier still draws any
    # sparse point that falls inside the range.
    _solid = P1.groupby(['region', 'n']).ve.agg(['mean', 'count'])
    _solid = _solid[_solid['count'] >= MIN_SESS]['mean']
    ax.set_ylim(0, float(_solid.max()) * 1.18)
    ax.set_xscale('log', base=2)
    ax.set_xticks([4, 8, 16, 32, 64])
    ax.set_xticklabels(['4', '8', '16', '32', '64'], fontsize=6.5)
    ax.set_xlabel('cells used to build the axis', fontsize=7.5)
    ax.set_ylabel('out-of-sample variance\nexplained by PC1', fontsize=7.5)
    ax.set_title('PC1 generalises to held-out cells,\nand improves with sampling',
                 fontsize=8)
    ax.legend(fontsize=5.6, frameon=False, loc='upper left')
    ax.tick_params(labelsize=6.5)
    ax.spines[['top', 'right']].set_visible(False)
    ax.text(-.20, 1.0, 'C', transform=ax.transAxes, fontsize=10, weight='bold',
            va='bottom', ha='right')

    ax = axes[1]
    w = .20
    M = D[D.n == MATCH_N]
    # Cerebellum reaches MATCH_N in a single session; a one-session bar beside a
    # sixty-session bar invites exactly the comparison it cannot support, so
    # regions below MIN_SESS are excluded here and said to be excluded.
    present = [r for r in REGIONS
               if M[(M.region == r) & (M.comp == 1)].shape[0] >= MIN_SESS]
    dropped = [r for r in REGIONS
               if 0 < M[(M.region == r) & (M.comp == 1)].shape[0] < MIN_SESS]
    for j, r in enumerate(present):
        d = M[M.region == r]
        g = d.groupby('comp').excess.agg(['mean', 'sem'])
        x = np.arange(len(g)) + (j - (len(present) - 1) / 2) * w
        ax.bar(x, g['mean'], w, color=RCOL[r], linewidth=0,
               label=f'{RNAME[r]} ({d[d.comp == 1].shape[0]})')
        ax.errorbar(x, g['mean'], yerr=g['sem'], fmt='none', ecolor='0.3',
                    elinewidth=.8, capsize=0)
    ax.axhline(0, color='0.4', lw=1)
    ax.set_xticks(np.arange(N_COMP))
    ax.set_xticklabels([f'PC{i+1}' for i in range(N_COMP)], fontsize=6.5)
    ax.set_xlabel(f'component (axis from {MATCH_N} cells)', fontsize=7.5)
    ax.set_ylabel('out-of-sample VE\nover null', fontsize=7.5)
    _r = (D[(D.n == MATCH_N) & (D.comp == 1)].excess.mean() /
          D[(D.n == MATCH_N) & (D.comp == 2)].excess.mean())
    # Not "only PC1 generalises": PC2 does clear its null (p = 4e-4). It is the
    # RATIO that carries the claim -- PC1 is ~50x larger -- so the title states
    # the ratio rather than a cleaner dichotomy than the data support.
    ax.set_title(f'one dominant axis:\nPC1 is {_r:.0f}$\\times$ PC2', fontsize=8)
    if dropped:
        ax.text(.5, -.42, ', '.join(RNAME[r] for r in dropped) +
                f' excluded (< {MIN_SESS} sessions at n = {MATCH_N})',
                transform=ax.transAxes, fontsize=5, color='0.5', ha='center')
    ax.legend(fontsize=5.6, frameon=False)
    ax.tick_params(labelsize=6.5)
    ax.spines[['top', 'right']].set_visible(False)
    ax.text(-.20, 1.0, 'D', transform=ax.transAxes, fontsize=10, weight='bold',
            va='bottom', ha='right')

    pc2_by_n(fig.add_subplot(bot[2]), D, S)
    fig.savefig(OUT_PDF, bbox_inches='tight', dpi=220)
    print(f'wrote {OUT_PDF}')


def sim_ratemaps(fig, spec, M, pop, n_per=4, seed=11):
    """Rate maps (trial x position) for n_per cells drawn at random from each
    population of the simulated session. Each cell is a periodic 1D grid cell
    with its own spacing and phase. On its anchored trials its fields sit at
    that phase; on non-anchored trials the phase is re-drawn every trial, by an
    offset shared by the cells of its population (one attractor per population),
    as the non-anchored data behave (re-drawn, not drifting). The labels are the
    simulation's; only the firing is added here, for display."""
    from matplotlib.colors import ListedColormap
    ANCH_C, NONANCH_C = '#a8559e', '#3f9b8f'
    rng = np.random.default_rng(seed)
    T, X = M.shape[1], np.arange(0, 200, 2.) + 1
    off = rng.random((pop.max() + 1, T))                    # per population, per trial
    pick = [c for g_ in range(pop.max() + 1)
            for c in np.sort(rng.choice(np.where(pop == g_)[0], n_per, replace=False))]
    g = spec.subgridspec(1, 2 * n_per + 1, width_ratios=[1] * n_per + [.2] + [1] * n_per,
                         wspace=.18)
    slots = list(range(n_per)) + list(range(n_per + 1, 2 * n_per + 1))
    for k, c in enumerate(pick):
        sp, ph = rng.uniform(55, 100), rng.uniform(0, 1)
        sh = np.where(M[c] == 1, ph, ph + off[pop[c]])[:, None] * sp
        d = (X[None, :] - sh) % sp
        d = np.minimum(d, sp - d)
        rate = 12 * np.exp(-d ** 2 / (2 * (.13 * sp) ** 2)) + .3
        R = gaussian_filter1d(rng.poisson(rate * .25).astype(float), 1.2, axis=1)
        gi = g[slots[k]].subgridspec(1, 2, width_ratios=[.09, 1], wspace=.06)
        a_ = fig.add_subplot(gi[0])
        a_.imshow(M[c][:, None], aspect='auto', cmap=ListedColormap([NONANCH_C, ANCH_C]),
                  vmin=0, vmax=1, interpolation='nearest')
        a_.set_axis_off()
        am = fig.add_subplot(gi[1])
        am.imshow(R, aspect='auto', cmap='viridis', extent=[0, 200, T, 0],
                  interpolation='nearest', vmin=0, vmax=np.percentile(R, 99))
        am.set_yticks([]); am.set_xticks([0, 200] if k in (0, n_per) else [])
        am.tick_params(labelsize=5.8, length=2, pad=1)
        if k == 0:
            am.set_xlabel('position (cm)', fontsize=6.3, labelpad=1)
            a_.text(-1.2, T / 2, 'trial', rotation=90, ha='right', va='center', fontsize=6.5)
        for s_ in am.spines.values():
            s_.set_visible(False)
        am.text(.5, 1.03, f'cell {c + 1}', transform=am.transAxes, fontsize=6,
                color='0.35', va='bottom', ha='center')
        if k % n_per == 0:
            am.text(0, 1.2, f'population {pop[c] + 1}', transform=am.transAxes, fontsize=6.8,
                    color='0.25', va='bottom')
            if k == 0:
                a_.text(-3.5, 1.38, 'A', transform=a_.transAxes, fontsize=10, weight='bold',
                        va='bottom', ha='right')
                am.text(0, 1.42, 'two populations, each switching between its own maps',
                        transform=am.transAxes, fontsize=8, va='bottom')


def sim_panels(fig, spec_maps, spec, S):
    """A: one simulated two-population session, example rate maps above its
    label raster; B: the by-component test on the simulations at MATCH_N, as D
    does on the data."""
    from matplotlib.colors import ListedColormap
    ANCH_C, NONANCH_C = '#a8559e', '#3f9b8f'
    M, pop, states = sim_session(np.random.default_rng(SIM_SHOW_SEED), (.5, .5), follow=SIM_SHOW_FOLLOW)
    sim_ratemaps(fig, spec_maps, M, pop)
    g = spec[0].subgridspec(1, 4, width_ratios=[1, .05, .05, .55], wspace=.08)
    ax = fig.add_subplot(g[0])
    ax.imshow(M.T, aspect='auto', cmap=ListedColormap([NONANCH_C, ANCH_C]), vmin=0, vmax=1,
              interpolation='nearest', extent=[0, len(M), M.shape[1], 0])
    ax.axvline((pop == 0).sum(), color='w', lw=1.5)
    for k_, lab in enumerate(('population 1', 'population 2')):
        ax.text(((pop < k_).sum() + (pop <= k_).sum()) / 2, -4, lab, ha='center', va='bottom',
                fontsize=6.5, color='0.25')
    ax.set_xticks([]); ax.set_xlabel(f'{len(M)} simulated cells', fontsize=7)
    ax.set_ylabel('trial', fontsize=7.5); ax.tick_params(labelsize=6.5)
    for sp_ in ax.spines.values():
        sp_.set_visible(False)
    for j in range(2):
        a_ = fig.add_subplot(g[1 + j])
        a_.imshow(states[j][:, None], aspect='auto', cmap=ListedColormap([NONANCH_C, ANCH_C]),
                  vmin=0, vmax=1, interpolation='nearest')
        a_.set_xticks([]); a_.set_yticks([])
        a_.set_xlabel(f'state {j + 1}', fontsize=5.6, rotation=90, labelpad=2)
    # PC1 and PC2 of this session, z-scored, as Figure 1 draws PC1
    Mc = centred(M)
    Vt = np.linalg.svd(Mc, full_matrices=False)[2]
    gg = g[3].subgridspec(1, 2, wspace=.15)
    for j in range(2):
        a_ = fig.add_subplot(gg[j])
        sc = Vt[j] * np.sign(np.corrcoef(Vt[j], states[j])[0, 1] or 1)
        sc = (sc - sc.mean()) / sc.std()
        y = np.arange(len(sc))
        a_.fill_betweenx(y, 0, sc, where=sc >= 0, color=ANCH_C, lw=0, edgecolor='none', interpolate=True)
        a_.fill_betweenx(y, 0, sc, where=sc < 0, color=NONANCH_C, lw=0, edgecolor='none', interpolate=True)
        a_.axvline(0, color='0.5', lw=.6)
        a_.set_ylim(len(sc), 0); a_.set_yticks([]); a_.set_xticks([])
        a_.set_title(f'PC{j + 1} (z)', fontsize=6.5, pad=2)
        r_ = [abs(np.corrcoef(Vt[j], st)[0, 1]) for st in states]
        a_.set_xlabel(f'|r| with state\n1: {r_[0]:.2f}, 2: {r_[1]:.2f}', fontsize=5.6)
        for sp_ in a_.spines.values():
            sp_.set_visible(False)

    ax = fig.add_subplot(spec[1])
    w = .26
    for j, (name, _, col) in enumerate(SCENARIOS):
        d = S[(S.scenario == name) & (S.n == MATCH_N)]
        gq = d.groupby('comp').excess.agg(['mean', 'sem'])
        x = np.arange(len(gq)) + (j - 1) * w
        ax.bar(x, gq['mean'], w, color=col, lw=0, label=name)
        ax.errorbar(x, gq['mean'], yerr=gq['sem'], fmt='none', ecolor='0.3', elinewidth=.8, capsize=0)
    ax.axhline(0, color='0.4', lw=1)
    ax.set_xticks(np.arange(N_COMP)); ax.set_xticklabels([f'PC{i + 1}' for i in range(N_COMP)], fontsize=6.5)
    ax.set_xlabel(f'component (axis from {MATCH_N} cells)', fontsize=7.5)
    ax.set_ylabel('out-of-sample VE\nover null', fontsize=7.5)
    ax.set_title(f'simulated ({SIM_SESS} sessions each)', fontsize=8)
    ax.legend(fontsize=5.6, frameon=False)
    ax.tick_params(labelsize=6.5); ax.spines[['top', 'right']].set_visible(False)
    ax.text(-.25, 1.0, 'B', transform=ax.transAxes, fontsize=10, weight='bold', va='bottom', ha='right')


def pc2_by_n(ax, D, S):
    """E: PC2's out-of-sample excess as the axis is built from more cells, MEC
    against the simulations. A second map in a subset of cells would show here
    as sampling grows."""
    for name, _, col in SCENARIOS:
        g = S[(S.scenario == name) & (S.comp == 2)].groupby('n').excess.agg(['mean', 'sem'])
        ax.fill_between(g.index, g['mean'] - g['sem'], g['mean'] + g['sem'], color=col, alpha=.25,
                        lw=0, edgecolor='none')
        ax.plot(g.index, g['mean'], color=col, lw=1.2, ls='--', marker='o', ms=2.2,
                label=f'simulated, {name}')
    d = D[(D.region == 'MEC') & (D.comp == 2)].groupby('n').excess.agg(['mean', 'sem', 'count'])
    d = d[d['count'] >= MIN_SESS]
    ax.fill_between(d.index, d['mean'] - d['sem'], d['mean'] + d['sem'], color=RCOL['MEC'], alpha=.25,
                    lw=0, edgecolor='none')
    ax.plot(d.index, d['mean'], color=RCOL['MEC'], lw=1.5, marker='o', ms=2.6, label='MEC data')
    ax.axhline(0, color='0.4', lw=.8)
    ax.set_xscale('log', base=2); ax.set_xticks([4, 8, 16, 32, 64])
    ax.set_xticklabels(['4', '8', '16', '32', '64'], fontsize=6.5)
    ax.set_xlabel('cells used to build the axis', fontsize=7.5)
    ax.set_ylabel('PC2 out-of-sample VE\nover null', fontsize=7.5)
    ax.set_title('a second map would grow with\nsampling; MEC\'s PC2 does not', fontsize=8)
    ax.legend(fontsize=5.4, frameon=False, loc='upper left')
    ax.tick_params(labelsize=6.5); ax.spines[['top', 'right']].set_visible(False)
    ax.text(-.25, 1.0, 'E', transform=ax.transAxes, fontsize=10, weight='bold', va='bottom', ha='right')
    m32 = d.loc[32] if 32 in d.index else None
    if m32 is not None:
        print(f'  E: MEC PC2 excess at n = 32: {m32["mean"]:+.4f} ({int(m32["count"])} sessions)')
    for name, _, _ in SCENARIOS:
        v = S[(S.scenario == name) & (S.comp == 2) & (S.n == 32)].excess
        print(f'  E: {name} PC2 excess at n = 32: {v.mean():+.4f}')


def report(D):
    M = D[D.n == MATCH_N]
    present = [r for r in REGIONS if not M[M.region == r].empty]
    # ------------------------------------------------------------- reporting
    print('\nPC1 out-of-sample VE at n =', MATCH_N)
    for r in present:
        d = M[(M.region == r) & (M.comp == 1)]
        print(f'  {RNAME[r]:11s} {d.ve.mean():.3f} vs null {d.ve_null.mean():.3f}'
              f'  excess {d.excess.mean():+.3f}  ({len(d)} sessions)')
    for lab, kw in [('all available cells, strict (null p95)', {}),
                    ('all available cells, LIBERAL (null mean)', dict(crit='mean')),
                    (f'matched n={MATCH_N}, strict', dict(at_n=MATCH_N))]:
        DIM = session_dimensionality(D, **kw)
        print(f'\nper-session dimensionality -- {lab}')
        for r in REGIONS:
            d = DIM[DIM.region == r]
            if d.empty:
                continue
            bits = '  '.join(f'{c}:{int((d.dim_cat == c).sum()):2d}'
                             for c in ['none', '1', '2', '3+'])
            withstate = d[d.dim >= 1]
            frac1 = (100 * (withstate.dim == 1).mean()) if len(withstate) else float('nan')
            print(f'  {RNAME[r]:11s} n={len(d):3d} cells/half={int(d.n.median()):3d}  '
                  f'{bits}   1-D among those with a state: {frac1:.0f}% '
                  f'({int((withstate.dim == 1).sum())}/{len(withstate)})')
    print('\nexcess by component (all regions pooled, n =', MATCH_N, ')')
    for c in range(1, N_COMP + 1):
        d = M[M.comp == c]
        from scipy.stats import wilcoxon
        p = wilcoxon(d.excess.dropna()).pvalue if d.excess.notna().sum() > 5 else np.nan
        print(f'  PC{c}: {d.excess.mean():+.4f}  p={p:.2g}  n={len(d)}')


if __name__ == '__main__':
    main()
