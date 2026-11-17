"""Figure S19 -- the narrow-spiking cells as a continuum of firing rate.

Usage:  python3 fig5_supp_narrow_continuum.py

Figure 5 calls a cell an interneuron if its waveform is narrow AND its rate is
in the top quartile (>= 10.3 Hz). The rate cut is a convention: log rate is not
bimodal, and the narrow cells below it inhibit their partners far more often
than broad-spiking cells do. So rather than add a 'slow interneuron' class,
this figure drops the rate cut and asks how each connectivity measure, and
state following, varies with firing rate across ALL narrow cells, against
broad-spiking cells at the same rate. Broad cells are the baseline that
absorbs detection power, which rises with spike count for any cell.

    A   the narrow cells' rates: one mode, no gap at the cut
    B   fraction of cells inhibiting at least one tested partner, by rate
    C   inhibitory out-degree: % of tested partners inhibited
    D   inhibition from narrow cells onto each postsynaptic class, by rate
    E   excitatory output, by rate -- principal cells should show here
    F   excitatory input received, by the cell's own rate
    G   latency of inhibitory connections, by presynaptic rate
    H   state following (agreement above chance with the MEC PC1), by rate
    I   connected vs matched unconnected pairs that include a narrow cell,
        by that cell's rate tercile -- Figure 5I along the continuum

STEP TESTS. 'Continuum' is tested, not assumed. A two-class account predicts a
jump at the 10.3 Hz cut, so each per-cell measure is tested for one by local
regression discontinuity: narrow cells within RD_BW log10 units of the cut
(6.5-16 Hz), a separate straight line in log rate on each side, and the jump
between them at the cut. A step fitted over the whole rate range is NOT used:
it depends on how the trend is modelled -- with a straight line in log rate
the steps in B and H came out upward, with a 4-df spline downward (both tried
2026-11-01) -- because the step absorbs whatever curvature the trend misses.
The trend is reported separately as the slope of a straight-line fit. Counts are Poisson with the log
number of tested partners as offset, standard errors clustered on session;
following is a mixed model with mouse as random effect and session nested.

UNIMODALITY (A) is Silverman's critical-bandwidth test, not a mixture BIC: the
narrow log-rate distribution is left-skewed (a tail of slow cells), and a
two-component mixture wins on BIC for skew alone.

Partner classes: narrow cells are 'narrow' whatever their open-field class;
broad cells keep it (grid, non-grid spatial, non-spatial).

Caches data/monosyn/pairs_by_cell.csv (every tested pair's label correlation,
probe distance and whether connected, with cell ids).
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
import statsmodels.api as sm
import statsmodels.formula.api as smf
from scipy.stats import gaussian_kde, mannwhitneyu

ROOT = '/Users/harryclark/Documents/spatial-manifolds'
sys.path.insert(0, f'{ROOT}/src')
from spatial_manifolds.anchoring import load_session_labels            # noqa: E402

plt.rcParams['font.family'] = 'Arial'
FIG = f'{ROOT}/scripts/figures/AnchorDynamics2026'
MS = f'{ROOT}/data/monosyn'
PS = f'{ROOT}/data/population_state'
NAR_C, BRD_C = '#d95f02', '0.45'
EXC_C, INH_C = '#1b998b', '#6d2e9e'
ICOL = {'grid': '#c04744', 'non-grid spatial': '#3171ae', 'non-spatial': '#888888',
        'narrow': NAR_C}
POST = ['grid', 'non-grid spatial', 'non-spatial', 'narrow']
EDGES = np.array([.3, 1.5, 3, 5, 7.5, 10.3, 15, 22, 35, 200])
MIN_BIN = 15
DIST_TOL, N_CTRL = 40.0, 20
IDENTITY_EFFECT = 0.11
SMOOTH = 'cr(lr, df=4)'          # rate held, for the narrow-vs-broad comparison in H
RD_BW = .2                        # log10 units either side of the cut


def _lp(ax, s, dx=-.2, dy=1.0):
    ax.text(dx, dy, s, transform=ax.transAxes, fontsize=10, weight='bold',
            va='bottom', ha='right')


def _style(ax):
    ax.tick_params(labelsize=7)
    ax.spines[['top', 'right']].set_visible(False)


def _ratex(ax, label='Firing rate (Hz)'):
    ax.set_xscale('log')
    ax.set_xticks([1, 3, 10, 30, 100]); ax.set_xticklabels(['1', '3', '10', '30', '100'])
    ax.set_xlim(.4, 150)
    ax.axvline(CUT, color='0.6', lw=.8, ls=':', zorder=0)
    ax.set_xlabel(label, fontsize=8)


# ── cells ────────────────────────────────────────────────────────────────────
U = pd.read_csv(f'{PS}/unit_table.csv')
CUT = float(U.rate_threshold_hz.dropna().iloc[0])
U['nar'] = U.narrow.astype(bool)
U['cls5'] = np.where(U.nar, 'narrow', U.identity)
C = pd.concat([pd.read_csv(f) for f in sorted(glob.glob(f'{MS}/connections2_w*.csv'))],
              ignore_index=True)
K = pd.concat([pd.read_csv(f) for f in sorted(glob.glob(f'{MS}/cells2_w*.csv'))],
              ignore_index=True)
K = K.merge(U[['mouse', 'day', 'cluster_id', 'nar', 'cls5', 'firing_rate']],
            on=['mouse', 'day', 'cluster_id'], how='inner')
K = K[K.firing_rate > 0].copy()
K['sess'] = K.mouse.astype(str) + '_' + K.day.astype(str)
K['lr'] = np.log10(K.firing_rate)
K['above'] = (K.firing_rate >= CUT).astype(int)
# tested partners of each class in the cell's session (every other tested cell)
cnt = K.groupby(['sess', 'cls5']).size().unstack(fill_value=0)
for c in POST:
    K[f'np_{c}'] = [cnt.loc[s, c] - (k == c) if c in cnt else 0 for s, k in zip(K.sess, K.cls5)]
K['n_part'] = K[[f'np_{c}' for c in POST]].sum(1)
K = K[K.n_part > 0]
key = list(zip(K.mouse, K.day, K.cluster_id))
CLS = dict(zip(key, K.cls5))
C = C[[(m, d, p) in CLS and (m, d, q) in CLS for m, d, p, q in zip(C.mouse, C.day, C.pre, C.post)]]
C['cpost'] = [CLS[(m, d, q)] for m, d, q in zip(C.mouse, C.day, C.post)]
for kind in ('inh', 'exc'):
    o = C[C.kind == kind].groupby(['mouse', 'day', 'pre']).size()
    i = C[C.kind == kind].groupby(['mouse', 'day', 'post']).size()
    K[f'{kind}_out'] = [o.get(k, 0) for k in key]
    K[f'{kind}_in'] = [i.get(k, 0) for k in key]
    for c in POST:
        oc = C[(C.kind == kind) & (C.cpost == c)].groupby(['mouse', 'day', 'pre']).size()
        K[f'{kind}_to_{c}'] = [oc.get(k, 0) for k in key]
K['bin'] = np.digitize(K.firing_rate, EDGES) - 1
N, B = K[K.nar], K[~K.nar]
print(f'{len(K)} tested cells ({len(N)} narrow, {len(B)} broad), {K.sess.nunique()} sessions; '
      f'{(C.kind == "inh").sum()} inhibitory, {(C.kind == "exc").sum()} excitatory connections')


def binned(D, f):
    """f(group) -> (value, lo, hi) per rate bin with at least MIN_BIN cells."""
    out = []
    for b, g in D.groupby('bin'):
        if len(g) >= MIN_BIN and 0 <= b < len(EDGES) - 1:
            out.append((10 ** g.lr.mean(), *f(g), len(g)))
    return np.array(out)


def frac_ci(g, col):
    k, n = (g[col] > 0).sum(), len(g)
    p = k / n; z = 1.96
    c = (p + z * z / (2 * n)) / (1 + z * z / n)
    h = z * np.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / (1 + z * z / n)
    return 100 * p, 100 * (c - h), 100 * (c + h)


def pct_ratio(num, den):
    """Pooled % of pairs, with a cell bootstrap interval."""
    def f(g):
        rng = np.random.default_rng(0)
        x, y = g[num].values, g[den].values
        bs = [x[i].sum() / max(y[i].sum(), 1) for i in
              (rng.integers(0, len(x), len(x)) for _ in range(500))]
        return 100 * x.sum() / max(y.sum(), 1), *(100 * np.percentile(bs, [2.5, 97.5]))
    return f


def step_test(D, y, offset='n_part'):
    """Poisson, offset log partners: the straight-line slope on log rate, and
    the step at the cut over a smooth function of log rate."""
    d = D.assign(off=np.log(D[offset]))
    cl = dict(cov_type='cluster', cov_kwds={'groups': d.sess.astype('category').cat.codes})
    m1 = smf.glm(f'Q("{y}") ~ lr', d, family=sm.families.Poisson(), offset=d.off).fit(**cl)
    w = near(d)
    m2 = smf.glm(f'Q("{y}") ~ dl * above', w, family=sm.families.Poisson(), offset=w.off).fit(
        cov_type='cluster', cov_kwds={'groups': w.sess.astype('category').cat.codes})
    return (m1.params['lr'], m1.pvalues['lr'], np.exp(m2.params['above']), m2.pvalues['above'],
            np.exp(m2.conf_int().loc['above']).values, len(w))


def near(D):
    """The cells within RD_BW of the cut, with log rate centred on it."""
    lc = np.log10(CUT)
    return D[(D.lr - lc).abs() < RD_BW].assign(dl=lambda d: d.lr - lc)


def silverman(x, n_boot=500, seed=0):
    """Silverman's test of unimodality: the smallest KDE bandwidth giving one
    mode, and how often smoothed bootstrap samples need a larger one."""
    grid = np.linspace(x.min() - .5, x.max() + .5, 800)
    sd = x.std(ddof=1)

    def modes(h, v):
        y = gaussian_kde(v, bw_method=h / v.std(ddof=1))(grid)
        return int(((y[1:-1] > y[:-2]) & (y[1:-1] > y[2:])).sum())
    lo, hi = 1e-3, 1.
    for _ in range(40):
        mid = (lo + hi) / 2
        lo, hi = (mid, hi) if modes(mid, x) > 1 else (lo, mid)
    rng = np.random.default_rng(seed)
    k = 0
    for _ in range(n_boot):
        v = rng.choice(x, len(x))
        v = v.mean() + (v - v.mean() + hi * rng.standard_normal(len(x))) / np.sqrt(1 + hi ** 2 / sd ** 2)
        k += modes(hi, v) > 1
    return hi, (k + 1) / (n_boot + 1)


def matched_ratio(y, offset='n_part', lo=1.5, hi=CUT):
    """Narrow over broad at matched rate below the cut: Poisson with log rate."""
    d = K[(K.firing_rate >= lo) & (K.firing_rate < hi)]
    d = d.assign(off=np.log(d[offset]), narrow=d.nar.astype(int))
    m = smf.glm(f'{y} ~ lr + narrow', d, family=sm.families.Poisson(),
                offset=d.off).fit(cov_type='cluster', cov_kwds={'groups': d.sess.astype('category').cat.codes})
    return np.exp(m.params['narrow']), np.exp(m.conf_int().loc['narrow']).values, m.pvalues['narrow']


def two_lines(ax, f, ylab, title, letter, stats=None, right=False):
    for D, col, nm in ((B, BRD_C, 'broad'), (N, NAR_C, 'narrow')):
        a = binned(D, f)
        ax.fill_between(a[:, 0], a[:, 2], a[:, 3], color=col, alpha=.22, lw=0, edgecolor='none')
        ax.plot(a[:, 0], a[:, 1], '-o', color=col, ms=3.2, lw=1.4, label=nm)
    _ratex(ax); ax.set_ylabel(ylab, fontsize=8)
    ax.set_title(title, fontsize=7.5, loc='left')
    _lp(ax, letter); _style(ax)
    if stats:
        ax.text(.98 if right else .02, .98, stats, transform=ax.transAxes, fontsize=5.8, va='top',
                ha='right' if right else 'left', color='0.25', linespacing=1.35)


fig = plt.figure(figsize=(9.2, 8.4))
G = fig.add_gridspec(3, 3, hspace=.62, wspace=.42)

# A: one mode
ax = fig.add_subplot(G[0, 0])
lrn = np.log10(U[U.nar & (U.firing_rate > 0)].firing_rate.values)
lrb = np.log10(U[~U.nar & (U.firing_rate > 0)].firing_rate.dropna().values)
bins = np.linspace(-1, 2.3, 34)
ax.hist(10 ** lrb, bins=10 ** bins, density=False, weights=np.full(len(lrb), len(lrn) / len(lrb)),
        histtype='step', color=BRD_C, lw=1.1, label='broad (scaled)')
ax.hist(10 ** lrn, bins=10 ** bins, color=NAR_C, alpha=.85, lw=0, label='narrow')
h_crit, p_uni = silverman(lrn)
_ratex(ax, 'Firing rate (Hz)'); ax.set_xlim(.1, 200)
ax.set_xticks([.1, 1, 10, 100]); ax.set_xticklabels(['0.1', '1', '10', '100'])
ax.set_ylabel('Cells', fontsize=8)
ax.set_title(f'narrow cells: one mode, no gap at\nthe {CUT:.1f} Hz cut (dotted)', fontsize=7.5, loc='left')
ax.legend(fontsize=5.8, frameon=False, loc='upper left')
ax.text(.02, .76, f'{len(lrn)} narrow cells,\nunimodal (Silverman\np = {p_uni:.2f})', transform=ax.transAxes,
        fontsize=5.8, va='top', color='0.25', linespacing=1.35)
_lp(ax, 'A'); _style(ax)
print(f'A narrow log rate: Silverman critical bandwidth {h_crit:.3f}, p = {p_uni:.2f} against '
      f'one mode ({len(lrn)} cells, {(lrn < np.log10(CUT)).mean():.0%} below the cut)')

# B: any inhibitory target
sN = step_test(N, 'inh_out')
rB = matched_ratio('inh_out')
two_lines(fig.add_subplot(G[0, 1]), lambda g: frac_ci(g, 'inh_out'),
          'Cells inhibiting ≥1 partner (%)',
          'inhibiting at least one partner', 'B')
# C: out-degree
ax = fig.add_subplot(G[0, 2])
two_lines(ax, pct_ratio('inh_out', 'n_part'), 'Partners inhibited (%)',
          'inhibitory out-degree', 'C',
          f'narrow, jump at cut x{sN[2]:.2f} (p = {sN[3]:.2g})\n'
          f'narrow / broad, 1.5-{CUT:.1f} Hz: x{rB[0]:.1f} (p = {rB[2]:.1g})')
ax.legend(fontsize=6, frameon=False, loc='center left')
print(f'B/C inh out, narrow: slope {sN[0]:+.2f} per log10 Hz (p {sN[1]:.2g}), step x{sN[2]:.2f} '
      f'(95% CI {sN[4].round(2)}, {sN[5]} cells; p {sN[3]:.2g}); narrow/broad below cut x{rB[0]:.2f} {rB[1].round(2)} p {rB[2]:.2g}')

# D: inhibition from narrow cells onto each postsynaptic class
ax = fig.add_subplot(G[1, 0])
for c in POST:
    a = binned(N, pct_ratio(f'inh_to_{c}', f'np_{c}'))
    ax.plot(a[:, 0], a[:, 1], '-o', color=ICOL[c], ms=3, lw=1.3,
            label={'non-grid spatial': 'NGS', 'non-spatial': 'NS'}.get(c, c))
    ax.fill_between(a[:, 0], a[:, 2], a[:, 3], color=ICOL[c], alpha=.13, lw=0, edgecolor='none')
a = binned(B, pct_ratio('inh_out', 'n_part'))
ax.plot(a[:, 0], a[:, 1], '--', color='k', lw=1, label='broad pre, all post')
_ratex(ax, 'Presynaptic firing rate (Hz)'); ax.set_ylabel('Inhibited (% of pairs)', fontsize=8)
ax.set_title('inhibition from narrow cells,\nby postsynaptic class', fontsize=7.5, loc='left')
ax.legend(fontsize=5.6, frameon=False, loc='upper left', title='postsynaptic', title_fontsize=5.6)
_lp(ax, 'D'); _style(ax)
for c in POST:
    s = step_test(N.assign(npc=N[f'np_{c}']).query('npc > 0'), f'inh_to_{c}', 'npc')
    print(f'D narrow -> {c}: slope {s[0]:+.2f} (p {s[1]:.2g}), step x{s[2]:.2f} (p {s[3]:.2g})')

# E: excitatory output
sE = step_test(N, 'exc_out')
rE = matched_ratio('exc_out')
two_lines(fig.add_subplot(G[1, 1]), pct_ratio('exc_out', 'n_part'), 'Partners excited (%)',
          'excitatory output', 'E',
          f'narrow, jump at cut x{sE[2]:.2f} (p = {sE[3]:.2g})\n'
          f'narrow / broad, 1.5-{CUT:.1f} Hz: x{rE[0]:.2f} (p = {rE[2]:.2g})', right=True)
print(f'E exc out, narrow: slope {sE[0]:+.2f} (p {sE[1]:.2g}), step x{sE[2]:.2f} {sE[4].round(2)} (p {sE[3]:.2g}); '
      f'narrow/broad below cut x{rE[0]:.2f} {rE[1].round(2)} p {rE[2]:.2g}')

# F: excitatory input received
sF = step_test(N, 'exc_in')
rF = matched_ratio('exc_in')
two_lines(fig.add_subplot(G[1, 2]), pct_ratio('exc_in', 'n_part'), 'Partners exciting it (%)',
          'excitatory input received', 'F',
          f'narrow, jump at cut x{sF[2]:.2f} (p = {sF[3]:.2g})\n'
          f'narrow / broad, 1.5-{CUT:.1f} Hz: x{rF[0]:.2f} (p = {rF[2]:.2g})', right=True)
print(f'F exc in, narrow: slope {sF[0]:+.2f} (p {sF[1]:.2g}), step x{sF[2]:.2f} {sF[4].round(2)} (p {sF[3]:.2g}); '
      f'narrow/broad below cut x{rF[0]:.2f} {rF[1].round(2)} p {rF[2]:.2g}')

# G: inhibitory latency by presynaptic rate
ax = fig.add_subplot(G[2, 0])
BINW = 100 / 101
L = C[C.kind == 'inh'].copy()
RT = dict(zip(key, K.firing_rate)); NR = dict(zip(key, K.nar))
L['rate'] = [RT[(m, d, p)] for m, d, p in zip(L.mouse, L.day, L.pre)]
L['nar'] = [NR[(m, d, p)] for m, d, p in zip(L.mouse, L.day, L.pre)]
L['lat'] = (L.lag_ms / BINW).round()
L['lr'] = np.log10(L.rate); L['bin'] = np.digitize(L.rate, EDGES) - 1
for D, col, nm in ((L[~L.nar], BRD_C, 'broad'), (L[L.nar], NAR_C, 'narrow')):
    a = binned(D, lambda g: (g.lat.mean(), g.lat.mean() - 1.96 * g.lat.sem(),
                             g.lat.mean() + 1.96 * g.lat.sem()))
    ax.fill_between(a[:, 0], a[:, 2], a[:, 3], color=col, alpha=.22, lw=0, edgecolor='none')
    ax.plot(a[:, 0], a[:, 1], '-o', color=col, ms=3.2, lw=1.4, label=nm)
Ln = L[L.nar].assign(sess=lambda d: d.mouse.astype(str) + '_' + d.day.astype(str),
                     above=lambda d: (d.rate >= CUT).astype(int))
_cl = dict(cov_type='cluster', cov_kwds={'groups': Ln.sess.astype('category').cat.codes})
mL1 = smf.ols('lat ~ lr', Ln).fit(**_cl)
_w = near(Ln)
mL = smf.ols('lat ~ dl * above', _w).fit(
    cov_type='cluster', cov_kwds={'groups': _w.sess.astype('category').cat.codes})
_ratex(ax, 'Presynaptic firing rate (Hz)'); ax.set_ylabel('Inhibitory latency (ms)', fontsize=8)
ax.set_ylim(1.5, 4)
ax.set_title('inhibitory latency', fontsize=7.5, loc='left')
ax.text(.02, .98, f'narrow, jump at cut {mL.params["above"]:+.2f} ms (p = {mL.pvalues["above"]:.2g})',
        transform=ax.transAxes, fontsize=5.8, va='top', color='0.25')
ax.legend(fontsize=6, frameon=False, loc='lower right')
_lp(ax, 'G'); _style(ax)
print(f'G latency, narrow: slope {mL1.params["lr"]:+.2f} ms per log10 Hz (p {mL1.pvalues["lr"]:.2g}), '
      f'step {mL.params["above"]:+.2f} ms {mL.conf_int().loc["above"].values.round(2)} (p {mL.pvalues["above"]:.2g})')

# H: state following
R = pd.read_csv(f'{PS}/pc1_by_region.csv')
R = R[R.is_mec].dropna(subset=['excess']).merge(
    U[['mouse', 'day', 'cluster_id', 'nar', 'firing_rate', 'identity']],
    on=['mouse', 'day', 'cluster_id'])
R = R[R.firing_rate > 0].copy()
R['lr'] = np.log10(R.firing_rate); R['bin'] = np.digitize(R.firing_rate, EDGES) - 1
R['above'] = (R.firing_rate >= CUT).astype(int)
R['sess'] = R.mouse.astype(str) + '_' + R.day.astype(str)
ax = fig.add_subplot(G[2, 1])
for D, col, nm in ((R[~R.nar], BRD_C, 'broad'), (R[R.nar], NAR_C, 'narrow')):
    a = binned(D, lambda g: (g.excess.mean(), g.excess.mean() - 1.96 * g.excess.sem(),
                             g.excess.mean() + 1.96 * g.excess.sem()))
    ax.fill_between(a[:, 0], a[:, 2], a[:, 3], color=col, alpha=.22, lw=0, edgecolor='none')
    ax.plot(a[:, 0], a[:, 1], '-o', color=col, ms=3.2, lw=1.4, label=nm)
Rn = R[R.nar]
mH1 = smf.mixedlm('excess ~ lr', Rn, groups=Rn['mouse'],
                  vc_formula={'sess': '0 + sess'}).fit(reml=True)
_w = near(Rn)
mH = smf.mixedlm('excess ~ dl * above', _w, groups=_w['mouse'],
                 vc_formula={'sess': '0 + sess'}).fit(reml=True)
# does rate account for the narrow cells' stronger following? all MEC cells
Ra = R.assign(narrow=R.nar.astype(int))
mH3 = smf.mixedlm(f'excess ~ {SMOOTH} + narrow', Ra, groups=Ra['mouse'],
                  vc_formula={'sess': '0 + sess'}).fit(reml=True)
mH0 = smf.mixedlm('excess ~ narrow', Ra, groups=Ra['mouse'],
                  vc_formula={'sess': '0 + sess'}).fit(reml=True)
Rm = R[(R.firing_rate >= 1.5) & (R.firing_rate < CUT)].assign(narrow=lambda d: d.nar.astype(int))
mH2 = smf.mixedlm('excess ~ lr + narrow', Rm, groups=Rm['mouse'],
                  vc_formula={'sess': '0 + sess'}).fit(reml=True)
_ratex(ax); ax.set_ylabel('Agreement above chance', fontsize=8)
ax.set_title('state following (MEC switching cells)', fontsize=7.5, loc='left')
ax.text(.02, .98, f'narrow, jump at cut {mH.params["above"]:+.3f} (p = {mH.pvalues["above"]:.2g})\n'
        f'narrow - broad, 1.5-{CUT:.1f} Hz: {mH2.params["narrow"]:+.3f} (p = {mH2.pvalues["narrow"]:.2g})',
        transform=ax.transAxes, fontsize=5.8, va='top', color='0.25', linespacing=1.35)
_lp(ax, 'H'); _style(ax)
print(f'H narrow vs broad, all rates: {mH0.params["narrow"]:+.3f} (p {mH0.pvalues["narrow"]:.2g}); '
      f'with rate held {mH3.params["narrow"]:+.3f} (p {mH3.pvalues["narrow"]:.2g}); {len(Ra)} cells')
print(f'H following, narrow ({len(Rn)} cells): slope {mH1.params["lr"]:+.3f} per log10 Hz '
      f'(p {mH1.pvalues["lr"]:.2g}), step {mH.params["above"]:+.3f} {mH.conf_int().loc["above"].values.round(3)} (p {mH.pvalues["above"]:.2g}, {len(near(Rn))} cells); '
      f'narrow - broad below cut {mH2.params["narrow"]:+.3f} (p {mH2.pvalues["narrow"]:.2g}, '
      f'{int(Rm.nar.sum())} narrow vs {int((~Rm.nar).sum())} broad)')

# I: connected vs matched unconnected pairs containing a narrow cell
PAIRS = f'{MS}/pairs_by_cell.csv'
if os.path.exists(PAIRS):
    P = pd.read_csv(PAIRS)
else:
    rows = []
    for (mo, dy), cells in K.groupby(['mouse', 'day']):
        z = load_session_labels(int(mo), int(dy))
        if z is None:
            continue
        Lb, ids = z['labels'], z['cluster_id'].astype(int)
        lab = {int(c): Lb[i] for i, c in enumerate(ids) if np.nanstd(Lb[i]) > 0}
        pos = {int(r.cluster_id): (r.px, r.py) for _, r in cells.iterrows()}
        use = [c for c in cells.cluster_id.astype(int) if c in lab]
        if len(use) < 8:
            continue
        cc = C[(C.mouse == mo) & (C.day == dy)]
        conn = {frozenset((int(a), int(b))) for a, b in zip(cc.pre, cc.post)}
        for i in range(len(use)):
            for j in range(i + 1, len(use)):
                a, b = use[i], use[j]
                x, y = lab[a], lab[b]
                m = np.isfinite(x) & np.isfinite(y)
                if m.sum() < 20 or np.std(x[m]) == 0 or np.std(y[m]) == 0:
                    continue
                pa, pb = pos[a], pos[b]
                rows.append((mo, dy, a, b, frozenset((a, b)) in conn,
                             float(np.corrcoef(x[m], y[m])[0, 1]),
                             float(np.hypot(pa[0] - pb[0], pa[1] - pb[1]))))
    P = pd.DataFrame(rows, columns=['mouse', 'day', 'a', 'b', 'conn', 'r', 'd']).dropna()
    P.to_csv(PAIRS, index=False)
P['ca'] = [CLS.get((m, d, c)) for m, d, c in zip(P.mouse, P.day, P.a)]
P['cb'] = [CLS.get((m, d, c)) for m, d, c in zip(P.mouse, P.day, P.b)]
P = P.dropna(subset=['ca', 'cb'])
P = P[(P.ca == 'narrow') | (P.cb == 'narrow')].copy()
ra = np.array([RT.get((m, d, c), np.nan) for m, d, c in zip(P.mouse, P.day, P.a)])
rb = np.array([RT.get((m, d, c), np.nan) for m, d, c in zip(P.mouse, P.day, P.b)])
# the narrow cell's rate; for narrow+narrow pairs the geometric mean of both
P['nrate'] = np.where((P.ca == 'narrow') & (P.cb == 'narrow'), np.sqrt(ra * rb),
                      np.where(P.ca == 'narrow', ra, rb))
P['partner'] = np.where(P.ca == 'narrow', P.cb, P.ca)
terc = np.nanpercentile(N.firing_rate, [100 / 3, 200 / 3])
P['t'] = np.digitize(P.nrate, terc)
ax = fig.add_subplot(G[2, 2])
TL = [f'< {terc[0]:.0f} Hz', f'{terc[0]:.0f}-{terc[1]:.0f} Hz', f'> {terc[1]:.0f} Hz']
for t in range(3):
    q = P[P.t == t]
    a, pool = q[q.conn], q[~q.conn]
    ctl = []
    for _, r in a.iterrows():
        cand = pool[(pool.mouse == r.mouse) & (pool.day == r.day) & (pool.partner == r.partner)
                    & ((pool.d - r.d).abs() <= DIST_TOL)]
        if len(cand):
            ctl.append(cand.sample(min(N_CTRL, len(cand)), random_state=0).r.values)
    ctl = np.concatenate(ctl) if ctl else np.array([])
    p_ = mannwhitneyu(a.r, ctl).pvalue
    sd = np.sqrt((a.r.var(ddof=1) * (len(a) - 1) + ctl.var(ddof=1) * (len(ctl) - 1))
                 / (len(a) + len(ctl) - 2))
    mde = 2.80 * sd * np.sqrt(1 / len(a) + 1 / len(ctl))
    for off, v, c in ((-.17, ctl, '0.72'), (.17, a.r.values, NAR_C)):
        ax.errorbar(t + off, v.mean(), yerr=v.std(ddof=1) / np.sqrt(len(v)), color=c, marker='o',
                    ms=4.5, lw=1.4, capsize=3)
    ax.annotate(f'{"n.s." if p_ > .05 else f"p = {p_:.2g}"}\nn={len(a)}\nMDE {mde:.3f}', (t, .99),
                xycoords=('data', 'axes fraction'), ha='center', va='top', fontsize=5.6,
                color='0.35', linespacing=1.35)
    print(f'I narrow-rate tercile {TL[t]}: connected {len(a)} r={a.r.mean():+.4f}, '
          f'control r={ctl.mean():+.4f}, p={p_:.2g}, MDE={mde:.3f}')
ax.set_xticks(range(3)); ax.set_xticklabels(TL, fontsize=6.5)
ax.set_xlim(-.5, 2.5); ax.set_ylim(top=ax.get_ylim()[1] * 1.32)
ax.set_xlabel("Narrow cell's firing rate", fontsize=8)
ax.set_ylabel('Anchoring agreement (r)', fontsize=8)
ax.set_title('pairs with a narrow cell: connected\n(orange) vs matched unconnected (grey)',
             fontsize=7.5, loc='left')
_lp(ax, 'I'); _style(ax)

out = f'{FIG}/fig5_supp_narrow_continuum.pdf'
plt.savefig(out, dpi=200, bbox_inches='tight')
print(f'\nsaved {out}')
