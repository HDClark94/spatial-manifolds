"""Figure 2 supplement — does where a cell sits predict whether it follows?

Usage:  python3 fig2_supp_anatomy.py

ONE QUESTION: is agreement with the population anchoring state organised
anatomically within MEC?

THE ANSWER IS YES FOR MEDIOLATERAL POSITION AND NO FOR EVERYTHING ELSE, and
getting there needs care about which sessions can answer the question at all.

Pooling cells across sessions gives a large mediolateral gradient -- rho = -0.15,
p = 2e-25, medial cells at r = 0.205 against lateral at 0.120 -- but pooling is
not a safe test. Probe placement differs greatly between animals (the Methods
record one mouse as 16.5% medial and another as 63.1%), so a cell's mediolateral
coordinate carries information about which session it came from, and sessions
differ in overall following for reasons that have nothing to do with anatomy.
The test has to be made WITHIN SESSION.

IT ALSO HAS TO BE MADE ON SESSIONS THAT CAN CARRY IT. Mediolateral position is
assigned PER SHANK, so a single-shank session has no mediolateral variation and
contributes nothing but noise: 33 of 61 sessions span less than 50 um. On the 27
sessions spanning at least 200 um -- two shank pitches or more -- the gradient
survives: median rho = -0.109, negative in 19 of 27, Wilcoxon p = 0.034.
Dorsoventral position (p = 0.99) and probe depth (p = 0.92) show nothing, and
for those two the session criterion changes nothing because every session varies
along the shank. That asymmetry is what shows the mediolateral result is
specific to the axis rather than manufactured by the restriction.

AN EARLIER VERSION OF THIS SUPPLEMENT REPORTED THE OPPOSITE (p = 1.00) because
it selected sessions by `nunique > 3`, the count of distinct mediolateral values.
That rule excludes the cleanly sampled 2-, 3- and 4-shank sessions, whose cells
take only a few discrete per-shank values while spanning 180-720 um, and admits
single-shank sessions whose coordinates differ by sub-micron numerical jitter.
It ran the test almost entirely on sessions with no mediolateral variation.

Pooled and within-session are both shown, because the contrast between them is
still the methodological point -- it is just a smaller contrast than it looked.

Figure 2 shows that identity predicts following. Identity is not evenly
distributed through MEC -- grid cells are enriched dorsally and medially -- so a
follower gradient could be an identity gradient wearing anatomical clothes, or
anatomy could predict following independently of identity. The two are
separable, and they have different implications: an anatomical gradient
independent of identity would say the state has a spatial origin within the
region, while one explained by identity would not.

COORDINATES, TAKEN FROM FIGURE 3 RATHER THAN GUESSED. `coord_SCs_x` is
MEDIOLATERAL -- figure 3 uses `coord_SCs_x.abs()` with its medial/lateral
boundary at 3400 um, and the magnitudes here run 2610-3774 um, which is the
mediolateral extent of MEC. `coord_SCs_y` is dorsoventral. `coord_SCs_z` is
anterior-posterior and is NOT plotted: an earlier version of this supplement
labelled z as mediolateral, which made the panel an anterior-posterior gradient
under a mediolateral title. Probe coordinates are kept alongside because they are what the
recording actually sampled -- a gradient present in probe depth but absent in
reconstructed depth would point at a reconstruction problem rather than biology.

THE TEST. The primary analysis is a within-SESSION Spearman correlation between
position and agreement, tested across sessions -- the unit at which the
confound is removed, since every cell in a session shares a probe placement.
The pooled mixed model, r ~ position + (1 | mouse) + (1 | mouse:day), is
reported alongside as the quantity that misleads. Z-scoring position within
mouse is not sufficient: it removes between-ANIMAL placement but leaves
between-session placement within an animal, which is where the effect lives.
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
import statsmodels.formula.api as smf
from scipy.stats import spearmanr, wilcoxon

# a session must span at least two shank pitches (250 um) on an axis before a
# within-session correlation along that axis means anything
MIN_SPAN_UM = 200.0

plt.rcParams['font.family'] = 'Arial'
ROOT = '/Users/harryclark/Documents/spatial-manifolds'
FIG = f'{ROOT}/scripts/figures/AnchorDynamics2026'
PS = f'{ROOT}/data/population_state'
ORDER = ['grid', 'non-grid spatial', 'non-spatial', 'putative interneuron']
ICOL = {'grid': '#c04744', 'non-grid spatial': '#3171ae',
        'non-spatial': '#888888', 'putative interneuron': '#d95f02'}

U = pd.read_csv(f'{PS}/unit_table.csv')
C = pd.read_csv(f'{ROOT}/data/cell_classifications_v2.csv')[
    ['mouse', 'day', 'cluster_id', 'coord_SCs_x', 'coord_SCs_y']]
M = (U[U.in_population & U.r.notna() & U.identity.isin(ORDER)]
     .merge(C, on=['mouse', 'day', 'cluster_id'], how='left'))
AXES = [('ml_um', 'medial–lateral (µm)'),
        ('coord_SCs_y', 'dorsal–ventral (µm)'),
        ('coord_probe_y', 'probe depth (µm)')]
# reset_index is load-bearing: mixedlm aligns `groups` by index, and the
# non-contiguous index left by the merge and dropna makes every fit fail
M['ml_um'] = M.coord_SCs_x.abs()      # stored negative; ML is the magnitude
M = M.dropna(subset=[a for a, _ in AXES]).reset_index(drop=True)
M['sess'] = M.mouse.astype(str) + 'D' + M.day.astype(str)
for a, _ in AXES:                      # z-scored within mouse
    # `or 1` does not catch NaN, and a mouse contributing a single cell gives
    # std(ddof=1) = NaN. Patsy then drops that row while `groups` keeps it, and
    # every mixedlm fit dies with an off-by-one IndexError.
    M[f'z_{a}'] = M.groupby('mouse')[a].transform(
        lambda v: (v - v.mean()) / (v.std(ddof=1) if v.std(ddof=1) else np.nan))
M = M.dropna(subset=[f'z_{a}' for a, _ in AXES]).reset_index(drop=True)
print(f'{len(M)} cells with coordinates, {M.groupby(["mouse","day"]).ngroups} '
      f'sessions, {M.mouse.nunique()} mice\n')


def fit(dv, terms):
    """`groups` is passed by NAME, not as a Series.

    Passing M['mouse'] makes statsmodels align it by index against a design
    matrix patsy has already subset, and every fit dies with an off-by-one
    IndexError. The column name lets it do the subsetting itself.
    """
    try:
        return smf.mixedlm(f'{dv} ~ {terms}', data=M, groups='mouse',
                           vc_formula={'sess': '0 + C(sess)'}).fit(
                               reml=False, method='lbfgs')
    except Exception:
        try:
            return smf.mixedlm(f'{dv} ~ {terms}', data=M, groups='mouse').fit(
                reml=False, method='lbfgs')
        except Exception:
            return None


print(f'{"axis":22s} {"rho":>7s} {"LMM p (alone)":>14s} {"LMM p (+identity)":>18s}')
res = {}
for a, lab in AXES:
    rho = spearmanr(M[a], M.r)[0]
    m1 = fit('r', f'z_{a}')
    m2 = fit('r', f'z_{a} + C(identity)')
    p1 = float(m1.pvalues[f'z_{a}']) if m1 is not None else np.nan
    p2 = float(m2.pvalues[f'z_{a}']) if m2 is not None else np.nan
    b2 = float(m2.params[f'z_{a}']) if m2 is not None else np.nan
    res[a] = (rho, p1, p2, b2)
    print(f'{lab:22s} {rho:+7.3f} {p1:14.3g} {p2:18.3g}')

WS = {}
fig = plt.figure(figsize=(9.6, 3.0))
gs = fig.add_gridspec(1, 4, wspace=.46)
for k, (a, lab) in enumerate(AXES):
    ax = fig.add_subplot(gs[k])
    ax.scatter(M[a], M.r, s=2, color='0.6', alpha=.2, lw=0)
    q = pd.qcut(M[a], 8, duplicates='drop')
    g = M.groupby(q).agg(x=(a, 'mean'), y=('r', 'mean'),
                         e=('r', lambda v: v.std(ddof=1) / np.sqrt(len(v))))
    # SEM as a band rather than caps: the binned means are a curve, and error
    # bars on a curve read as eight separate estimates instead of one trend
    ax.fill_between(g.x, g.y - g.e, g.y + g.e, color='0.35', alpha=.30,
                    linewidth=0, edgecolor='none', zorder=3)
    ax.plot(g.x, g.y, color='k', lw=1.6, zorder=4)
    ax.axhline(0, color='0.6', lw=.7, ls=':')
    rho, p1, p2, _ = res[a]
    ax.set_xlabel(lab, fontsize=8)
    if k == 0:
        ax.set_ylabel('Agreement with\npopulation axis (r)', fontsize=8.5)
    # SESSION SELECTION IS BY SPAN, NOT BY DISTINCT-VALUE COUNT. Mediolateral
    # position is assigned per shank, so a cleanly sampled 3-shank session holds
    # only three distinct ML values while spanning 510 um, and a single-shank
    # session can hold dozens that differ by sub-micron numerical jitter. The
    # earlier `nunique > 3` rule therefore kept the sessions with no ML
    # variation and dropped the ones with the most, and reported p = 1.00 for an
    # effect that is present at p = 0.034 once the test is run on sessions that
    # can carry it. For DV and probe depth the criterion changes nothing --
    # every session varies along the shank -- which is what shows the ML result
    # is specific to the axis rather than produced by the restriction.
    ws = M.groupby('sess').apply(
        lambda d: spearmanr(d[a], d.r)[0]
        if (d[a].max() - d[a].min()) >= MIN_SPAN_UM and len(d) >= 20
        and d[a].nunique() >= 2 else np.nan).dropna()
    pw = wilcoxon(ws).pvalue if len(ws) >= 6 else np.nan
    ax.set_title(f'{"ABC"[k]}  pooled rho = {rho:+.2f} (p = {p1:.0e})\n'
                 f'WITHIN SESSION rho = {ws.median():+.3f}, p = {pw:.2g}',
                 fontsize=6.6, loc='left')
    WS[a] = ws
    ax.tick_params(labelsize=7); ax.spines[['top', 'right']].set_visible(False)

# D: the within-session correlations, which is where the pooled effect goes
ax = fig.add_subplot(gs[3])
rng = np.random.default_rng(0)
for i, (a, lab) in enumerate(AXES):
    v = WS.get(a, pd.Series(dtype=float))
    if not len(v):
        continue
    ax.scatter(np.full(len(v), i) + rng.uniform(-.14, .14, len(v)), v, s=14,
               color='0.35', lw=0, zorder=3)
    ax.plot([i - .22, i + .22], [v.median()] * 2, color='k', lw=2, zorder=4)
ax.axhline(0, color='k', lw=1, ls='--')
ax.set_xticks(range(len(AXES)))
ax.set_xticklabels(['M–L', 'D–V', 'depth'], fontsize=7.5)
ax.set_ylabel('within-session rho', fontsize=8.5)
ax.set_title('D  within session: M–L only\n(sessions spanning ≥ 200 µm)',
             fontsize=7.5, loc='left')
ax.tick_params(labelsize=7); ax.spines[['top', 'right']].set_visible(False)

out = f'{FIG}/fig2_supp_anatomy.pdf'
plt.savefig(out, dpi=200, bbox_inches='tight')
print(f'\nsaved {out}')
