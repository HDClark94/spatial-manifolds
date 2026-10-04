"""Figure 6: one state, shared across structures, expressed most strongly by a cell type.

THE QUESTION THIS FIGURE NOW ASKS. Earlier versions asked "which regions follow the
MEC axis?", which makes entorhinal cortex the privileged frame: a region can only
look like a follower to the extent it resembles MEC, and "does visual cortex have
a state of its own?" cannot be asked at all. Here every region builds its OWN
first principal component of its own cell x trial label matrix and the axes are
compared with one another, so no region defines the frame.

WHY A RAW AXIS-TO-AXIS CORRELATION IS NOT ENOUGH. An axis estimated from a few
noisy cells is itself noisy, so a low correlation between two regions' axes can
mean they express different states OR that one barely expresses a state at all --
opposite conclusions. Each region's SPLIT-HALF reliability separates them: build
its axis twice from disjoint halves of its own cells and correlate. Dividing the
cross-region correlation by the geometric mean of the two reliabilities
(Spearman's correction for attenuation) asks the question the paper needs --
given that both regions express a state, is it the SAME state?

Everything is size-matched to 8 cells per half, because reliability rises with
cell count and MEC contributes hundreds of cells where visual cortex contributes
tens.

WHAT CAME OUT, AND WHY THE FIGURE WAS REFRAMED. Every region has a coherent state
of its own, visual cortex included (split-half +0.330, p = 2.4e-5). Corrected for
reliability the states are indistinguishable from identical (MEC-SUB +0.920,
MEC-VIS +0.863, neither different from 1), and MEC-SUB does not differ from
MEC-VIS (p = 0.66). Meanwhile at matched cell count MEC's own state is only
marginally more coherent than visual cortex's (+0.417 vs +0.334, p = 0.065).

So the region is barely the relevant variable. The CELL TYPE is: grid cells
exceed visual cortex by +0.149 per cell (p = 9.9e-6) while other entorhinal cells
exceed it by only +0.050. One state, present everywhere recorded, read out most
strongly by grid cells and local inhibition -- which is what Figures 2 and 3
already said, and which explains why a visual-cortex state predicts behaviour as
well as an entorhinal one (panel F).

CAVEAT KEPT IN VIEW. Disattenuation amplifies noise: the MEC-VIS CI is
[0.49, 1.23] and 8 of 19 sessions exceed 1, so "not different from 1" excludes
differences larger than about 0.3 and no more. SUB-VIS (n=4) and CB (n=1) are not
testable.

Outputs: fig6_cross_region.pdf
Inputs:  pc1_heldout.csv (build_heldout_pc1.py), region_axis_coherence.csv
         (region_axis_coherence.py), nonmec_population_state.csv
"""
import sys
import warnings

import numpy as np
import pandas as pd

warnings.filterwarnings('ignore')
import matplotlib

matplotlib.use('Agg')
import matplotlib.pyplot as plt
from scipy.stats import pearsonr, wilcoxon
from sklearn.metrics import roc_auc_score

sys.path.insert(0, '/Users/harryclark/Documents/spatial-manifolds/src')

plt.rcParams['font.family'] = 'Arial'
# mathtext must resolve to Arial, not the DejaVu/STIX default, or rho, times
# and the panel letters render in a different face from the rest of the figure
plt.rcParams['mathtext.fontset'] = 'custom'
plt.rcParams['mathtext.rm'] = 'Arial'
plt.rcParams['mathtext.it'] = 'Arial:italic'
plt.rcParams['mathtext.bf'] = 'Arial:bold'
plt.rcParams['pdf.fonttype'] = 42

ROOT = '/Users/harryclark/Documents/spatial-manifolds'
FIG = f'{ROOT}/scripts/figures/AnchorDynamics2026'
PS = f'{ROOT}/data/population_state'
OUT = f'{FIG}/fig6_cross_region.pdf'

GC_C, INT_C = '#c04744', '#d95f02'
RCOL = {'MEC': '#7b4173', 'VIS': '#3f9b4f', 'SUB': '#c0723a', 'CB': '#4c72b0'}
MIN_CELLS = 10
REGIONS = ['MEC', 'SUB', 'VIS', 'CB']


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


# ---- per-cell expression, every cell scored out-of-sample --------------------
D = pd.read_csv(f'{PS}/pc1_heldout.csv')
U = pd.read_csv(f'{PS}/unit_table.csv')[['mouse', 'day', 'cluster_id', 'identity']]
D = D.merge(U, on=['mouse', 'day', 'cluster_id'], how='left')
D['macro'] = D.brain_region.map(fam)
D['sess'] = D.mouse.astype(str) + 'D' + D.day.astype(str)
V = D.dropna(subset=['r'])
MECg = (V.macro == 'MEC') & (V.identity == 'grid')
MECi = (V.macro == 'MEC') & (V.identity == 'putative interneuron')
MECo = (V.macro == 'MEC') & (~V.identity.isin(['grid', 'putative interneuron']))
GRP = [('grid\n(MEC)', MECg, GC_C), ('int\n(MEC)', MECi, INT_C),
       ('other\n(MEC)', MECo, '#9a9a9a'),
       ('SUB/\nPARA', V.macro == 'SUB', RCOL['SUB']),
       ('VIS', V.macro == 'VIS', RCOL['VIS']), ('CB', V.macro == 'CB', RCOL['CB'])]

# ---- each region's own axis --------------------------------------------------
A = pd.read_csv(f'{PS}/region_axis_coherence.csv')
A['pair'] = A.reg_i + '-' + A.reg_j
REL = pd.concat([A[['mouse', 'day', 'reg_i', 'r_ii']]
                 .rename(columns={'reg_i': 'reg', 'r_ii': 'rel'}),
                 A[['mouse', 'day', 'reg_j', 'r_jj']]
                 .rename(columns={'reg_j': 'reg', 'r_jj': 'rel'})])

N = pd.read_csv(f'{PS}/nonmec_population_state.csv')

print("each region's own state (split-half, size-matched):")
for r in REGIONS:
    x = REL[REL.reg == r].groupby(['mouse', 'day']).rel.mean().dropna()
    if not len(x):
        continue
    p = wilcoxon(x).pvalue if len(x) >= 6 else np.nan
    print(f'  {r:4s} n={len(x):2d}  {x.mean():+.3f}  p={p:.3g}')
print('\ncross-region, corrected for reliability:')
for pr in ['MEC-SUB', 'MEC-VIS', 'SUB-VIS']:
    x = A[A.pair == pr].disatt.dropna()
    if len(x) < 4:
        continue
    p = wilcoxon(x - 1).pvalue if len(x) >= 6 else np.nan
    print(f'  {pr:8s} n={len(x):2d}  raw {A[A.pair == pr].r_ij.mean():+.3f}  '
          f'disatt {x.mean():+.3f}  vs 1: p={p:.3g}')

# ---- behaviour ---------------------------------------------------------------
beh = []
for (mo, dy), d in N.groupby(['mouse', 'day']):
    for r in REGIONS:
        f, nc = d[f'frac_{r}'], d[f'n_{r}']
        if f.isna().all() or nc.max() < MIN_CELLS:
            continue
        rec = dict(mouse=mo, day=dy, region=r)
        for tt, lab in (('b', 'cued'), ('nb', 'uncued')):
            s = d[(d.ttype == tt) & f.notna()]
            if len(s) >= 10 and 0 < s.hit.mean() < 1:
                rec[f'auc_{lab}'] = roc_auc_score(s.hit, s[f'frac_{r}'])
        beh.append(rec)
B = pd.DataFrame(beh)
B['adi'] = B.auc_uncued - B.auc_cued

# ============================== figure =========================================
fig = plt.figure(figsize=(9.8, 6.8))
G = fig.add_gridspec(2, 3, hspace=.64, wspace=.44, width_ratios=[1.25, 1, 1])


def tidy(ax):
    ax.tick_params(labelsize=7)
    ax.spines[['top', 'right']].set_visible(False)


def lp(ax, s, x=-.20):
    ax.text(x, 1.05, s, transform=ax.transAxes, fontsize=10, weight='bold',
            va='bottom', ha='right')


def jit(n, j, rng, w=.15):
    return np.full(n, j) + rng.uniform(-w, w, n)


rng = np.random.default_rng(0)

# ---- A: one session, three regional states -----------------------------------
cand = N[(N.n_VIS >= MIN_CELLS) & (N.n_SUB >= MIN_CELLS) & (N.n_MEC >= 30)]
key = cand.groupby(['mouse', 'day']).apply(
    lambda d: d.frac_MEC.std() * min(d.n_VIS.iloc[0], d.n_SUB.iloc[0]) ** .25
    if d.frac_MEC.between(.2, .8).mean() > .2 else np.nan).dropna()
mo, dy = key.idxmax()
d = N[(N.mouse == mo) & (N.day == dy)].sort_values('trial')
ax = fig.add_subplot(G[0, 0])
for r in ('MEC', 'SUB', 'VIS'):
    ax.plot(d.trial, d[f'frac_{r}'], color=RCOL[r], lw=1.1,
            label=f'{r} ({int(d[f"n_{r}"].iloc[0])})')
ax.axhline(.5, color='0.6', ls=':', lw=.9)
ax.set_xlabel('trial', fontsize=8)
ax.set_ylabel('fraction of cells anchored', fontsize=8)
rs = pearsonr(d.frac_MEC.dropna(), d.frac_SUB.dropna())
rv = pearsonr(d.frac_MEC, d.frac_VIS)
ax.set_title(f'M{int(mo)} D{int(dy)}: one state, three structures\n'
             f'MEC–SUB r = {rs[0]:.2f},  MEC–VIS r = {rv[0]:.2f}', fontsize=8.5)
ax.legend(fontsize=6.5, frameon=False, ncol=3, loc='lower left')
tidy(ax)
lp(ax, 'A', x=-.16)

# ---- B: each region's own state ----------------------------------------------
ax = fig.add_subplot(G[0, 1])
for j, r in enumerate(REGIONS):
    x = REL[REL.reg == r].groupby(['mouse', 'day']).rel.mean().dropna()
    if not len(x):
        continue
    ax.scatter(jit(len(x), j, rng), x, s=11, color=RCOL[r], alpha=.75, lw=0,
               zorder=3)
    ax.plot([j - .28, j + .28], [x.median()] * 2, color='k', lw=2, zorder=4)
    ax.annotate(f'{len(x)}', (j, .02), xycoords=('data', 'axes fraction'),
                ha='center', va='bottom', fontsize=5.6, color='0.45')
ax.axhline(0, color='0.5', lw=.9, ls='--')
ax.set_xticks(range(len(REGIONS)))
ax.set_xticklabels(REGIONS, fontsize=7.5)
ax.set_ylabel("split-half reliability of\nthe region's OWN axis", fontsize=8)
ax.set_title('every region has a state of\nits own, visual cortex included',
             fontsize=8.5)
tidy(ax)
lp(ax, 'B', x=-.26)

# ---- C: and it is the same state ---------------------------------------------
ax = fig.add_subplot(G[0, 2])
prs = ['MEC-SUB', 'MEC-VIS', 'SUB-VIS']
for j, pr in enumerate(prs):
    x = A[A.pair == pr].disatt.dropna()
    if not len(x):
        continue
    ax.scatter(jit(len(x), j, rng), x, s=11, color='#c04744', alpha=.75, lw=0,
               zorder=3)
    ax.plot([j - .28, j + .28], [x.mean()] * 2, color='k', lw=2, zorder=4)
    lab = f'{len(x)}' if len(x) >= 6 else f'{len(x)}\nn.t.'
    ax.annotate(lab, (j, .02), xycoords=('data', 'axes fraction'), ha='center',
                va='bottom', fontsize=5.6, color='0.45')
    if len(x) >= 6:
        p = wilcoxon(x - 1).pvalue
        ax.annotate(f'p={p:.2f}', (j, .99), xycoords=('data', 'axes fraction'),
                    ha='center', va='top', fontsize=6, color='0.45')
ax.axhline(1, color='0.4', lw=1.1, ls='--')
ax.set_ylim(-1.4, 3.3)
ax.text(len(prs) - .45, 1.08, 'identical', fontsize=6, color='0.4', ha='right')
ax.set_xticks(range(len(prs)))
ax.set_xticklabels(prs, fontsize=6.8, rotation=16, ha='right')
ax.set_ylabel('agreement / sqrt(' + r'$r_{ii}r_{jj}$' + ')', fontsize=8)
ax.set_title('corrected for each region\'s own\nreliability, it is the SAME state',
             fontsize=8.5)
tidy(ax)
lp(ax, 'C', x=-.24)

# ---- D: per-cell expression, out-of-sample -----------------------------------
ax = fig.add_subplot(G[1, 0])
for j, (nm, m, col) in enumerate(GRP):
    per = V[m].groupby('sess').r.mean().dropna()
    if not len(per):
        continue
    ax.scatter(jit(len(per), j, rng), per, s=9, color=col, alpha=.7, lw=0,
               zorder=3)
    ax.plot([j - .28, j + .28], [per.median()] * 2, color='k', lw=2, zorder=4)
ax.axhline(0, color='0.4', lw=1, ls='--')
ax.set_xticks(range(len(GRP)))
ax.set_xticklabels([n for n, _, _ in GRP], fontsize=6.2)
ax.set_ylabel('per-cell expression of\nthe state, $r$', fontsize=8)
ax.set_title('what differs is the CELL TYPE: grid cells and interneurons\n'
             'express it twice as strongly as anything else', fontsize=8.5)
tidy(ax)
lp(ax, 'D', x=-.17)

# ---- E: cell type beats region ----------------------------------------------
# Three paired within-session contrasts on one axis. The first two are per-cell
# expression; the third is the region-level axis reliability from panel B. They
# are different quantities and are drawn together only because the comparison
# between them is the panel's point -- the cell-type contrast is three times the
# region contrast, which does not reach significance.
ax = fig.add_subplot(G[1, 1])
vis_r = V[V.macro == 'VIS'].groupby('sess').r.mean()
items = []
for nm, m, col in (('grid\nvs VIS', MECg, GC_C), ('other MEC\nvs VIS', MECo, '#9a9a9a')):
    a = V[m].groupby('sess').r.mean()
    jj = pd.concat([a.rename('a'), vis_r.rename('b')], axis=1).dropna()
    items.append((nm, (jj.a - jj.b), col))
mv = A[(A.reg_i == 'MEC') & (A.reg_j == 'VIS')][['r_ii', 'r_jj']].dropna()
items.append(('MEC axis\nvs VIS axis', mv.r_ii - mv.r_jj, RCOL['MEC']))
for j, (nm, x, col) in enumerate(items):
    ax.scatter(jit(len(x), j, rng), x, s=11, color=col, alpha=.75, lw=0, zorder=3)
    ax.plot([j - .28, j + .28], [x.median()] * 2, color='k', lw=2, zorder=4)
    p = wilcoxon(x).pvalue
    ax.annotate(f'p={p:.0e}'.replace('e-0', 'e-') if p < .001 else f'p={p:.3f}',
                (j, .99), xycoords=('data', 'axes fraction'), ha='center',
                va='top', fontsize=6, color='#a33' if p < .05 else '0.45')
    ax.annotate(f'{len(x)}', (j, .02), xycoords=('data', 'axes fraction'),
                ha='center', va='bottom', fontsize=5.6, color='0.45')
ax.axhline(0, color='0.4', lw=1, ls='--')
ax.set_xticks(range(len(items)))
ax.set_xticklabels([n for n, _, _ in items], fontsize=6.4)
ax.set_ylabel('advantage over visual cortex,\npaired within session', fontsize=8)
ax.set_title('cell type, not region', fontsize=8.5)
tidy(ax)
lp(ax, 'E', x=-.24)

# ---- F: behaviour, which a global state predicts ----------------------------
ax = fig.add_subplot(G[1, 2])
for j, r in enumerate(REGIONS):
    a = B[B.region == r].adi.dropna()
    if not len(a):
        continue
    ax.scatter(jit(len(a), j, rng), a, s=10, color=RCOL[r], alpha=.7, lw=0,
               zorder=3)
    ax.plot([j - .28, j + .28], [a.median()] * 2, color='k', lw=2, zorder=4)
    if len(a) >= 6:
        p = wilcoxon(a).pvalue
        ax.annotate(f'{p:.3f}', (j, .99), xycoords=('data', 'axes fraction'),
                    ha='center', va='top', fontsize=6,
                    color='#a33' if p < .05 else '0.45')
    ax.annotate(f'{len(a)}', (j, .02), xycoords=('data', 'axes fraction'),
                ha='center', va='bottom', fontsize=5.6, color='0.45')
ax.axhline(0, color='0.4', lw=1, ls='--')
ax.set_xticks(range(len(REGIONS)))
ax.set_xticklabels(REGIONS, fontsize=7.5)
ax.set_ylabel('ADI: AUC(uncued) $-$ AUC(cued)', fontsize=8)
ax.set_title('a state read from any structure\ncarries the behavioural effect',
             fontsize=8.5)
tidy(ax)
lp(ax, 'F', x=-.24)

fig.savefig(OUT, bbox_inches='tight')
print(f'\nwrote {OUT}')
