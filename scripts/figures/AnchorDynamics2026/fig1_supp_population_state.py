"""Figure S2: from cell labels to one population state per trial, related to
Figures 1 and 2.

Two summaries of a cells x trials label matrix are in use: the fraction of cells
anchored on each trial, and the trial's z-scored score on the matrix's first
principal component (signed to rise with the fraction). They weight cells
differently. The fraction counts every cell once; PC1 weights each cell by how
far its labels move with the others, so cells that follow the population count
for more and cells that do not count for less. Recorded populations contain
both, and nothing in the recordings says which cells are which. This figure
compares the two where the truth is known.

A     one simulated session (population_state_sim.py, built on the cells of
      Figure S1): the true state, and labels from the whole track for the three
      kinds of cell, followers, cells switching each on their own (incoherent)
      and a second group switching together on a state of its own (competing)
B     the two estimates on one population drawn from it
C-E   how well each estimate tracks the true state (correlation over trials),
      against the share of followers, for each of the three classifiers of
      Figure 2; the rest of the population incoherent (solid) or competing
      (dashed)
F     how much of the true state's prediction of the uncued outcome each
      estimate recovers, the measure of Figure 2G (AUC over a circular-shift
      null, as a fraction of the true state's own)
G     recorded sessions: how closely the two estimates agree, per structure and
      classifier
H     recorded sessions: how well each black-box estimate recovers the
      whole-track state of the same kind, the best estimate of the state
      the recordings offer
I     Figure 2I's black-box tests under each estimate

Reads data/classifier_sim/population_state_sim.npz, the unfiltered labels of
build_neighbour_labels.py, and neighbour_test.csv (PC1) / neighbour_test_frac.csv.
"""
import os
import sys

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.colors import ListedColormap
from scipy.stats import wilcoxon

HERE = '/Users/harryclark/Documents/spatial-manifolds/scripts/figures/AnchorDynamics2026'
sys.path.insert(0, HERE)
sys.path.insert(0, '/Users/harryclark/Documents/spatial-manifolds/src')
from neighbour_test import MIN_CELLS, NT, PS, REGIONS, auc_null, pc1_state  # noqa: E402
from spatial_manifolds.anchoring import ANCH_COLOR, NONANCH_COLOR  # noqa: E402

plt.rcParams['font.family'] = 'Arial'
plt.rcParams['mathtext.default'] = 'regular'
SIM = '/Users/harryclark/Documents/spatial-manifolds/data/classifier_sim/population_state_sim.npz'
CACHE = '/Users/harryclark/Documents/spatial-manifolds/data/classifier_sim/population_state_scores.csv'
Z = np.load(SIM)
NS = int(Z['n_sess'])
STR = ['whole', 'task', 'box']
NAME = {'whole': 'whole track', 'task': 'task region', 'box': 'black box'}
CCOL = {'whole': '0.62', 'task': '#e0b43c', 'bb': '0.2', 'box': '0.2'}
MCOL = {'frac': '0.15', 'pc1': '#d95f02'}                 # the two estimates
MNAME = {'frac': 'fraction anchored', 'pc1': 'PC1 (z)'}
RCOL = {'MEC': '#7b4173', 'VIS': '#3f9b4f', 'SUB': '#c0723a', 'CB': '#4c72b0'}
FS = [1., .75, .5, .25]                                    # share of followers
N_CELLS, N_DRAW = 30, 25


def lp(ax, s_, x=-.2, y=1.04):
    ax.text(x, y, s_, transform=ax.transAxes, fontsize=10, weight='bold', va='bottom', ha='right')


def clean(ax):
    ax.tick_params(labelsize=6.5)
    ax.spines[['top', 'right']].set_visible(False)


def fast_auc_null(x, y, rng, n_shift=200, shift_min=5):
    """auc_null of neighbour_test.py, vectorised: the AUC of x for y and the mean
    AUC under circular shifts of x, from ranks (ties averaged), which a shift
    only rolls."""
    from scipy.stats import rankdata
    ok = np.isfinite(x)
    x, y = x[ok], np.asarray(y)[ok].astype(bool)
    n1, n0 = y.sum(), (~y).sum()
    if n1 < 5 or n0 < 5 or np.std(x) == 0:
        return np.nan, np.nan
    rk = rankdata(x)
    auc = lambda r: (r[y].sum() - n1 * (n1 + 1) / 2) / (n1 * n0)
    n = len(x)
    sh = rng.integers(shift_min, n - shift_min, n_shift)
    idx = (np.arange(n)[None, :] - sh[:, None]) % n
    null = ((rk[idx][:, y].sum(1) - n1 * (n1 + 1) / 2) / (n1 * n0)).mean()
    return auc(rk), null


def estimates(L):
    """The two per-trial estimates from a cells x trials label matrix."""
    L = L[~np.isnan(L).all(axis=1)]
    return {'frac': np.nanmean(L, axis=0), 'pc1': pc1_state(L)}


def draw(s, k, f, rest, rng, n=N_CELLS):
    """A population of n cells from session s's pool, a share f of them followers."""
    nf = int(round(f * n))
    a = Z[f's{s}_followers_{k}']
    b = Z[f's{s}_{rest}_{k}']
    return np.r_[a[rng.choice(len(a), nf, replace=False)],
                 b[rng.choice(len(b), n - nf, replace=False)]]


# ── scores over simulated populations, cached ────────────────────────────────
if os.path.exists(CACHE):
    S = pd.read_csv(CACHE)
else:
    rows = []
    for s in range(NS):
        truth, unc, hit = Z[f's{s}_truth'], Z[f's{s}_uncued'], Z[f's{s}_hit'].astype(int)
        rng = np.random.default_rng(s)
        t_auc, t_null = fast_auc_null(truth[unc].astype(float), hit[unc], np.random.default_rng(s))
        for k in STR:
            for rest in ('incoherent', 'competing'):
                for f in FS:
                    for d in range(N_DRAW):
                        E = estimates(draw(s, k, f, rest, rng))
                        for m, e in E.items():
                            ok = np.isfinite(e)
                            r = np.corrcoef(e[ok], truth[ok])[0, 1]
                            a, a0 = fast_auc_null(e[unc], hit[unc], np.random.default_rng(d))
                            rows.append(dict(s=s, k=k, rest=rest, f=f, d=d, m=m, r=r,
                                             auc_x=a - a0, true_x=t_auc - t_null))
        print(f'  session {s} scored', flush=True)
    S = pd.DataFrame(rows)
    S.to_csv(CACHE, index=False)
S['share'] = S.auc_x / S.true_x

# ── recorded sessions: agreement between the two estimates ───────────────────
REAL = []
for fn in sorted(x for x in os.listdir(NT) if x.startswith('M') and x.endswith('.npz')):
    z = np.load(f'{NT}/{fn}', allow_pickle=True)
    for r in REGIONS:
        m = z['region'] == r
        if (~np.isnan(z['labels_full'][m]).all(axis=1)).sum() < MIN_CELLS[r]:
            continue
        EW = estimates(z['labels_full'][m])
        for k, v in (('whole', 'full'), ('task', 'tz'), ('box', 'bb')):
            E = estimates(z[f'labels_{v}'][m])
            ok = np.isfinite(E['frac']) & np.isfinite(E['pc1'])
            rec = dict(f=fn, region=r, k=k, r=np.corrcoef(E['frac'][ok], E['pc1'][ok])[0, 1])
            for m_ in ('frac', 'pc1'):
                ok = np.isfinite(E[m_]) & np.isfinite(EW[m_])
                rec[f'whole_{m_}'] = np.corrcoef(E[m_][ok], EW[m_][ok])[0, 1]
            REAL.append(rec)
REAL = pd.DataFrame(REAL)

# ── figure ───────────────────────────────────────────────────────────────────
fig = plt.figure(figsize=(7.2, 10.0))
gs = fig.add_gridspec(4, 1, height_ratios=[1.05, 1.0, 1.0, .95], hspace=.62)

# A: one simulated session, whole-track labels of each group
top = gs[0].subgridspec(1, 2, width_ratios=[1.35, 1], wspace=.3)
ga = top[0].subgridspec(2, 3, height_ratios=[.09, 1], hspace=.04, wspace=.08)
s0 = 0
truth = Z[f's{s0}_truth']
cm = ListedColormap([NONANCH_COLOR, ANCH_COLOR])
for j, (grp, lab) in enumerate((('followers', 'followers'), ('incoherent', 'incoherent'),
                                 ('competing', 'competing'))):
    at = fig.add_subplot(ga[0, j])
    if j == 1:
        at.text(.5, .5, 'each its own', transform=at.transAxes, ha='center', va='center',
                fontsize=5.5, color='0.35')
    else:
        at.imshow((Z[f's{s0}_other'] if grp == 'competing' else truth)[None, :], aspect='auto',
                  cmap=cm, vmin=0, vmax=1, interpolation='nearest')
    at.set_xticks([]); at.set_yticks([])
    at.set_title(lab, fontsize=7, pad=3)
    ax = fig.add_subplot(ga[1, j])
    L = Z[f's{s0}_{grp}_whole'][:30]
    ax.imshow(np.ma.masked_invalid(L), aspect='auto', cmap=cm, vmin=0, vmax=1,
              interpolation='nearest')
    ax.set_yticks([]); ax.tick_params(labelsize=6)
    ax.set_xlabel('trial', fontsize=6.5)
    if j == 0:
        ax.set_ylabel('cells', fontsize=7)
        lp(at, 'A', x=-.12, y=1.3)
        at.text(-.03, .5, 'state', transform=at.transAxes, ha='right', va='center', fontsize=6)

# B: the two estimates on one population, half followers, half incoherent
ax = fig.add_subplot(top[1])
rng = np.random.default_rng(3)
E = estimates(draw(s0, 'whole', .5, 'incoherent', rng))
tt = np.arange(len(truth))
ax.fill_between(tt, -3, 3, where=truth == 1, color=ANCH_COLOR, alpha=.12, lw=0,
                edgecolor='none', step='mid')
z_fr = (E['frac'] - np.nanmean(E['frac'])) / np.nanstd(E['frac'])
ax.plot(tt, z_fr, color=MCOL['frac'], lw=.8, label=MNAME['frac'] + ' (z)')
ax.plot(tt, E['pc1'], color=MCOL['pc1'], lw=.8, label=MNAME['pc1'])
for m, e in (('frac', E['frac']), ('pc1', E['pc1'])):
    ok = np.isfinite(e)
    print(f'  B: {m} r with truth {np.corrcoef(e[ok], truth[ok])[0, 1]:.2f}')
ax.set_ylim(-3, 3); ax.set_xlim(0, len(tt) - 1)
ax.set_xlabel('trial', fontsize=7); ax.set_ylabel('estimate (z)', fontsize=7)
ax.legend(fontsize=5.8, frameon=False, loc='lower left', ncol=2, bbox_to_anchor=(0, -.02))
ax.set_title(f'{N_CELLS} cells, half followers; shaded, truly anchored', fontsize=6.8, loc='left')
lp(ax, 'B', x=-.16, y=1.04)
clean(ax)

# C-E: correlation with the truth against the share of followers
mid = gs[1].subgridspec(1, 4, width_ratios=[1, 1, 1, 1.15], wspace=.5)
for j, k in enumerate(STR):
    ax = fig.add_subplot(mid[j])
    for rest, ls in (('incoherent', '-'), ('competing', '--')):
        for m in ('frac', 'pc1'):
            g = S[(S.k == k) & (S.rest == rest) & (S.m == m)].groupby('f').r
            mu, se = g.mean(), g.std() / np.sqrt(g.count())
            ax.errorbar(mu.index, mu, yerr=se, color=MCOL[m], ls=ls, lw=1, marker='o', ms=2.5,
                        capsize=0, label=f'{MNAME[m]}, {rest}' if j == 0 else None)
    ax.set_xticks(FS); ax.set_xticklabels(['1', '.75', '.5', '.25'])
    ax.set_xlim(1.07, .18); ax.set_ylim(-.1, 1)
    ax.axhline(0, color='0.7', lw=.5, ls=':')
    ax.set_xlabel('share of followers', fontsize=6.8)
    if j == 0:
        ax.set_ylabel('r with the true state', fontsize=7)
    ax.set_title(NAME[k], fontsize=7, color={'whole': '0.45', 'task': '#a07c10', 'box': '0.15'}[k],
                 loc='left')
    lp(ax, 'CDE'[j], x=-.25 if j else -.3)
    clean(ax)
    if j == 0:
        ax.legend(fontsize=5.3, frameon=False, loc='lower left', handlelength=2.2)

# F: share of the true state's outcome prediction, a quarter followers, per
# classifier, the rest incoherent (solid) or competing (hatched)
ax = fig.add_subplot(mid[3])
for ir, rest in enumerate(('incoherent', 'competing')):
    for i, k in enumerate(STR):
        for jm, m in enumerate(('frac', 'pc1')):
            g = S[(S.k == k) & (S.rest == rest) & (S.m == m) & (S.f == .25)]
            v = g.groupby('s').share.mean()
            x = ir * 3.6 + i + (jm - .5) * .38
            ax.bar(x, v.mean(), width=.34, color=MCOL[m], alpha=.85 if ir == 0 else .45, lw=0,
                   hatch=None if ir == 0 else '////', edgecolor=MCOL[m])
            ax.errorbar(x, v.mean(), yerr=v.std() / np.sqrt(len(v)), color='k', lw=.8, capsize=1.2)
    ax.text(ir * 3.6 + 1, -.34, rest, transform=ax.get_xaxis_transform(), ha='center', va='top',
            fontsize=6.3)
ax.axhline(1, color='0.6', lw=.6, ls=':'); ax.axhline(0, color='0.5', lw=.6)
ax.set_xticks([ir * 3.6 + i for ir in range(2) for i in range(3)])
ax.set_xticklabels(['whole', 'task', 'box'] * 2, fontsize=5.8, rotation=90)
ax.set_ylabel("share of the true\nstate's AUC over null", fontsize=6.8, labelpad=1)
ax.set_title('outcome, a quarter followers', fontsize=7, loc='left')
lp(ax, 'F', x=-.32)
clean(ax)

# G: recorded sessions, agreement between the estimates
bot = gs[2].subgridspec(1, 2, width_ratios=[1.25, 1], wspace=.3)
ax = fig.add_subplot(bot[0])
rng = np.random.default_rng(0)
for i, k in enumerate(STR):
    for j, r in enumerate(REGIONS):
        v = REAL[(REAL.k == k) & (REAL.region == r)].r
        x = i * 5 + j
        ax.scatter(x + rng.uniform(-.2, .2, len(v)), v, s=4, color=RCOL[r], lw=0, alpha=.6)
        ax.plot([x - .32, x + .32], [v.median()] * 2, color='k', lw=1.2)
    ax.text(i * 5 + 1.5, -.27, NAME[k], ha='center', va='top', fontsize=6.8, transform=ax.get_xaxis_transform(),
            color={'whole': '0.45', 'task': '#a07c10', 'box': '0.15'}[k])
ax.set_xticks([i * 5 + j for i in range(3) for j in range(4)])
ax.set_xticklabels(REGIONS * 3, fontsize=5.6, rotation=90)
for t, r in zip(ax.get_xticklabels(), REGIONS * 3):
    t.set_color(RCOL[r])
ax.set_ylim(-.1, 1.02)
ax.set_ylabel('r, fraction vs PC1 (z)', fontsize=7)
ax.set_title('recorded sessions: how far the two estimates agree', fontsize=7, loc='left')
lp(ax, 'G', x=-.15)
clean(ax)

# H: which black-box estimate recovers the whole-track state of its own kind
ax = fig.add_subplot(bot[1])
rng = np.random.default_rng(1)
for j, r in enumerate(REGIONS):
    g = REAL[(REAL.k == 'box') & (REAL.region == r)]
    for jm, m in enumerate(('frac', 'pc1')):
        x = j * 2.6 + jm
        ax.scatter(x + rng.uniform(-.15, .15, len(g)), g[f'whole_{m}'], s=4, color=MCOL[m], lw=0,
                   alpha=.5)
        ax.plot([x - .3, x + .3], [g[f'whole_{m}'].median()] * 2, color='k', lw=1.2)
    for _, row in g.iterrows():
        ax.plot([j * 2.6, j * 2.6 + 1], [row.whole_frac, row.whole_pc1], color='0.85', lw=.4, zorder=0)
    d = g.whole_pc1 - g.whole_frac
    p = wilcoxon(d).pvalue
    ax.text(j * 2.6 + .5, 1.0, f'p = {p:.2g}' if p < .05 else 'n.s.', ha='center', fontsize=5.6)
    ax.text(j * 2.6 + .5, -.2, r, transform=ax.get_xaxis_transform(), ha='center', fontsize=7,
            color=RCOL[r])
    print(f'  H {r}: black box vs whole track, frac {g.whole_frac.median():.2f}, '
          f'pc1 {g.whole_pc1.median():.2f}, paired p = {p:.2g}')
ax.set_xticks([j * 2.6 + jm for j in range(4) for jm in range(2)])
ax.set_xticklabels(['frac', 'PC1'] * 4, fontsize=5.8)
ax.axhline(0, color='0.6', lw=.5, ls=':')
ax.set_ylim(-.45, 1.08)
ax.set_ylabel('r, black-box state vs\nwhole-track state', fontsize=7)
ax.set_title('which recovers the whole-track state', fontsize=7, loc='left')
lp(ax, 'H', x=-.2)
clean(ax)

# I: Figure 2I's black-box tests under each estimate
ax = fig.add_subplot(gs[3])
A_ = {'frac': pd.read_csv(f'{PS}/neighbour_test_frac.csv'), 'pc1': pd.read_csv(f'{PS}/neighbour_test.csv')}
TESTS = [('same_bb', 'G box'), ('nbr_bb', 'H box'), ('res_bb', 'H box res.')]
for j, r in enumerate(REGIONS):
    for i, (c, lab) in enumerate(TESTS):
        for jm, m in enumerate(('frac', 'pc1')):
            v = A_[m][A_[m].region == r][f'{c}_x'].dropna()
            x = j * 4 + i + (jm - .5) * .38
            p = wilcoxon(v).pvalue
            ax.bar(x, v.mean(), width=.34, color=MCOL[m], alpha=.85 if p < .05 else .3, lw=0)
            ax.errorbar(x, v.mean(), yerr=v.sem(), color='k', lw=.8, capsize=1.2)
ax.axhline(0, color='0.5', lw=.6)
ax.set_xticks([j * 4 + i for j in range(4) for i in range(3)])
ax.set_xticklabels([l for _ in REGIONS for _, l in TESTS], fontsize=5.4, rotation=90)
for j, r in enumerate(REGIONS):
    ax.text(j * 4 + 1, -.42, r, transform=ax.get_xaxis_transform(), ha='center', fontsize=7,
            color=RCOL[r])
ax.set_ylabel('AUC over null', fontsize=7)
ax.set_title('the black-box tests of Figure 2I under each estimate: dark, fraction anchored; '
             'orange, PC1 (z); pale, p ≥ 0.05', fontsize=7, loc='left')
lp(ax, 'I', x=-.07)
clean(ax)

# numbers for the caption
for k in STR:
    for rest in ('incoherent', 'competing'):
        for f in (1., .5, .25):
            g = S[(S.k == k) & (S.rest == rest) & (S.f == f)].groupby('m')
            print(f'  {k:5s} {rest:10s} f={f:.2f}: r frac {g.r.mean()["frac"]:.2f}, pc1 '
                  f'{g.r.mean()["pc1"]:.2f}; outcome share frac {g.share.mean()["frac"]:.2f}, '
                  f'pc1 {g.share.mean()["pc1"]:.2f}')
print(REAL.groupby(['region', 'k']).r.median().round(2).unstack())
out = f'{HERE}/fig1_supp_population_state.pdf'
plt.savefig(out, dpi=200, bbox_inches='tight')
plt.savefig(out.replace('.pdf', '.png'), dpi=110, bbox_inches='tight')
print(f'saved {out}')
