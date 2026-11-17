"""Figure S4: open-field coding of the cells of each structure, related to Figure 2.

A     the model: one cell's spike counts in 20 ms bins predicted from its own
      open-field variables alone, position (x, y), head direction (sin, cos)
      and speed, by a Poisson XGBoost model, scored on held-out folds by the
      Poisson likelihood of the observed spikes, over a constant rate
B     forward selection (Hardcastle et al. 2017) for one cell: the held-out
      gain of each model, fold by fold, and the path selection takes
C     held-out gain of the selected model, by class, every classified cell
D     four MEC cells, one per row: observed and held-out predicted rate maps,
      head-direction and speed tuning
E-H   the share of each structure's cells in each class (formerly S3 G-J)

Reads data/of_xgb/of_xgb_cells.csv (of_xgb_decomp.py) and
data/of_xgb/of_xgb_examples.npz (build_of_xgb_examples.py).
"""
import os
import sys

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy.ndimage import gaussian_filter
from scipy.stats import wilcoxon

HERE = '/Users/harryclark/Documents/spatial-manifolds/scripts/figures/AnchorDynamics2026'
sys.path.insert(0, HERE)
from of_xgb_decomp import OUT  # noqa: E402

plt.rcParams['font.family'] = 'Arial'
plt.rcParams['mathtext.default'] = 'regular'
REGIONS = ['MEC', 'VIS', 'SUB', 'CB']
RCOL = {'MEC': '#7b4173', 'VIS': '#3f9b4f', 'SUB': '#c0723a', 'CB': '#4c72b0'}
VCOL = {'P': '#00798c', 'H': '#d1a000', 'S': '#8c564b'}       # the three variables
CLASSES = ['P', 'H', 'S', 'PH', 'PS', 'HS', 'PHS', '']
G = pd.read_csv(f'{OUT}/of_xgb_cells.csv', keep_default_na=False)
for m in CLASSES[:-1]:
    G[f'llh_{m}'] = pd.to_numeric(G[f'llh_{m}'], errors='coerce')
Z = np.load(f'{OUT}/of_xgb_examples.npz')
CC = pd.read_csv(f'{os.path.dirname(OUT)}/cell_classifications_v2.csv')
KIND = {'GC': 'grid cell', 'NGS': 'non-grid spatial', 'NS': 'non-spatial'}


def lp(ax, s_, x=-.2, y=1.04):
    ax.text(x, y, s_, transform=ax.transAxes, fontsize=10, weight='bold', va='bottom', ha='right')


def clean(ax):
    ax.tick_params(labelsize=6.5)
    ax.spines[['top', 'right']].set_visible(False)


fig = plt.figure(figsize=(7.2, 9.9))
gs = fig.add_gridspec(4, 1, height_ratios=[2.35, 1.45, 3.3, 1.2], hspace=.42)

# ── A: the model ────────────────────────────────────────────────────────────
ax = fig.add_subplot(gs[0])
ax.set_xlim(0, 100); ax.set_ylim(0, 43); ax.axis('off')
lp(ax, 'A', x=.0, y=.97)
F, y, mu = Z['trace_F'], Z['trace_y'], Z['trace_mu']
n = len(y)
tx = np.linspace(7, 30, n)
rows = [('x', 0, 'P', 31.5), ('y', 1, 'P', 27.5), (r'sin $\theta$', 2, 'H', 22.5),
        (r'cos $\theta$', 3, 'H', 18.5), ('speed', 4, 'S', 13.5)]
for lab, k, v, y0 in rows:
    f = F[:, k]
    f = (f - f.min()) / (np.ptp(f) or 1)
    ax.plot(tx, y0 + 3.2 * f, color=VCOL[v], lw=.8)
    ax.text(6.4, y0 + 1.6, lab, ha='right', va='center', fontsize=6.3, color=VCOL[v])
for v, (lo, hi) in (('P', (27.5, 34.7)), ('H', (18.5, 25.7)), ('S', (13.5, 16.7))):
    ax.plot([1.6, 1.6], [lo, hi], color=VCOL[v], lw=1.2, solid_capstyle='butt')
    ax.text(.9, (lo + hi) / 2, v, ha='right', va='center', fontsize=8, color=VCOL[v], weight='bold')
for t in tx[y > 0]:
    ax.plot([t, t], [37.5, 40], color='k', lw=.5)
ax.text(6.4, 38.75, 'spikes', ha='right', va='center', fontsize=6.3)
ib = int(n * .62)
ax.axvspan(tx[ib] - .35, tx[ib] + .35, ymin=12.8 / 43, ymax=40.6 / 43, color='0.8', lw=0, zorder=0)
ax.plot([7, 7 + 23 / 10 * 2], [11.6, 11.6], color='k', lw=1)
ax.text(7, 10.4, '2 s', fontsize=6, va='top')
ax.text(18.5, 10.4, 'one 20 ms bin, shaded', fontsize=6, va='top', ha='center', color='0.4')
ax.text(15.5, 4.5, "the cell's own open-field variables only:\nno other cell's spikes enter the model",
        fontsize=6.3, ha='center', va='center', style='italic', color='0.25')
ax.annotate('', xy=(35.5, 25), xytext=(31, 25), arrowprops=dict(arrowstyle='->', lw=.8))

# the trees, summed in log space: rate = exp(sum of tree outputs)
ax.text(44, 41.5, 'XGBoost, Poisson', ha='center', fontsize=7.5)
ax.text(44, 39.4, 'depth = number of inputs', ha='center', fontsize=5.8, color='0.4')
for lab, y0 in (('tree 1', 34), ('tree 2', 26.5), ('tree N', 13.5)):
    root, kids = (44, y0 + 3), [(41.3, y0), (46.7, y0)]
    leaves = [(40, y0 - 2.8), (42.6, y0 - 2.8), (45.4, y0 - 2.8), (48, y0 - 2.8)]
    for k_ in kids:
        ax.plot([root[0], k_[0]], [root[1], k_[1]], color='0.3', lw=.6, zorder=1)
    for i, l in enumerate(leaves):
        k_ = kids[i // 2]
        ax.plot([k_[0], l[0]], [k_[1], l[1]], color='0.3', lw=.6, zorder=1)
    pts = np.array([root] + kids + leaves)
    ax.scatter(pts[:, 0], pts[:, 1], s=9, color='white', edgecolor='0.2', lw=.6, zorder=2)
    ax.text(37.3, y0 + .5, lab, ha='right', va='center', fontsize=6)
    ax.plot([48.6, 53.6], [y0, 23 + (y0 - 23) * .12], color='0.5', lw=.5)
ax.scatter([44] * 3, [18.4, 19.6, 20.8], s=1.5, color='k')
ax.scatter([55], [23], s=230, color='white', edgecolor='k', lw=.8, zorder=3)
ax.text(55, 23, 'Σ', ha='center', va='center', fontsize=8, zorder=4)
ax.annotate('', xy=(61.5, 32), xytext=(56.6, 24.6), arrowprops=dict(arrowstyle='->', lw=.8))
ax.text(57.4, 29, 'exp', fontsize=6.3, ha='right')

# right: the held-out rate scored against the observed spikes
rx = np.linspace(63, 97, n)
ax.plot(rx, 31 + 6 * mu / mu.max(), color='0.15', lw=.7)
ax.text(80, 38.5, 'predicted rate, held-out bins', ha='center', fontsize=6.3)
ax.text(64.5, 28.4, 'Poisson likelihood of', fontsize=6.3, ha='left', va='center')
ax.text(64.5, 26.6, 'the observed spikes', fontsize=6.3, ha='left', va='center')
for t in rx[y > 0]:
    ax.plot([t, t], [22, 24.4], color='k', lw=.5)
ax.annotate('', xy=(80, 17.8), xytext=(80, 21.2), arrowprops=dict(arrowstyle='->', lw=.8))
ax.text(80, 16.4, 'bits per spike over a constant rate', ha='center', va='center', fontsize=6.3)

# cross-validation: 50 sections, fold k tests every tenth; the next fold
# along stops the boosting early; the rest train
ax.text(66.5, 11.6, '10-fold cross-validation, 50 sections of the session', ha='center', fontsize=6.3)
sec = np.arange(50)
for j, (k, y0) in enumerate(((0, 8.2), (1, 6.0), (9, 2.6))):
    col = np.where(sec % 10 == k, '0.1', np.where(sec % 10 == (k + 1) % 10, '0.55', '0.86'))
    for i in sec:
        ax.add_patch(plt.Rectangle((40 + i * .6, y0), .52, 1.4, color=col[i], lw=0))
    ax.text(39.4, y0 + .7, f'fold {k + 1}', ha='right', va='center', fontsize=5.6)
ax.scatter([37.6] * 3, [4.4, 5.0, 5.6], s=1, color='k')
for i, (lab, c) in enumerate((('train', '0.86'), ('early stopping', '0.55'), ('test', '0.1'))):
    ax.add_patch(plt.Rectangle((72.5, 8.2 - 2.4 * i), 1.4, 1.4, color=c, lw=0))
    ax.text(74.6, 8.9 - 2.4 * i, lab, fontsize=5.8, va='center')

# ── B: forward selection for the walkthrough cell ───────────────────────────
row = gs[1].subgridspec(1, 2, width_ratios=[1.7, 1], wspace=.32)
ax = fig.add_subplot(row[0])
models, L, visited = list(Z['models']), Z['walk_llh'], set(Z['walk_visited'])
XS = {'P': 0, 'H': 1, 'S': 2, 'PH': 3.5, 'PS': 4.5, 'HS': 5.5, 'PHS': 7}
path = ['P', 'PH', 'PHS']
for m, v in zip(models, L):
    x = XS[m]
    seen, on = m in visited, m in path
    c_ = '0.1' if on else '0.6'
    ax.scatter(np.full(len(v), x) + np.linspace(-.18, .18, len(v)), v, s=6,
               color=c_ if seen else 'none', edgecolor='none' if seen else '0.6', lw=.5, zorder=3)
    ax.plot([x - .3, x + .3], [np.nanmean(v)] * 2, color=c_, lw=1.6, zorder=4)
for a, b in zip(path[:-1], path[1:]):
    la, lb = L[models.index(a)], L[models.index(b)]
    p = wilcoxon(lb, la, alternative='greater').pvalue
    ya = max(np.nanmax(la), np.nanmax(lb)) + .3
    ax.annotate('', xy=(XS[b] - .3, ya), xytext=(XS[a] + .3, ya),
                arrowprops=dict(arrowstyle='->', lw=.8, connectionstyle='arc3,rad=-.15'))
    ax.text((XS[a] + XS[b]) / 2, ya + .25, f'p = {p:.3f}', ha='center', fontsize=5.8)
ax.set_xticks(list(XS.values()))
ax.set_xticklabels(list(XS), fontsize=6.5)
for t, m in zip(ax.get_xticklabels(), XS):
    t.set_color('k' if m in path else '0.55')
    t.set_weight('bold' if m in path else 'normal')
for x, s_ in ((1, 'one variable'), (4.5, 'add one'), (7, 'add one')):
    ax.text(x, -.3, s_, transform=ax.get_xaxis_transform(), ha='center', fontsize=6, color='0.35')
ax.axhline(0, color='0.6', lw=.6, ls=':')
ax.set_ylim(0, np.nanmax(L) + 1.1)
ax.set_ylabel('held-out gain\n(bits/spike)', fontsize=7)
mo, dy, c = Z['c3_id']
ax.set_title(f'forward selection, MEC M{mo} D{dy} cell {c}: one point per fold\n'
             f'dark, the best at each step; hollow, never fitted by the selection',
             fontsize=6.5, loc='left', pad=4)
lp(ax, 'B', x=-.13, y=1.0)
clean(ax)

# C: held-out gain of the selected model, each class, every classified cell
ax = fig.add_subplot(row[1])
g = G[G.model != '']
vals = [g.loc[g.model == m, f'llh_{m}'].clip(lower=1e-3).values for m in CLASSES[:-1]]
bp = ax.boxplot(vals, positions=range(7), widths=.6, showfliers=False, patch_artist=True,
                medianprops=dict(color='k', lw=1), whiskerprops=dict(lw=.7), capprops=dict(lw=.7),
                boxprops=dict(lw=.7))
for b in bp['boxes']:
    b.set_facecolor('0.88')
ax.set_yscale('log')
ax.set_xticks(range(7)); ax.set_xticklabels(CLASSES[:-1], fontsize=6.5)
for i, v in enumerate(vals):
    ax.text(i, 1.02, len(v), transform=ax.get_xaxis_transform(), ha='center', fontsize=5.3,
            color='0.4', rotation=90, va='bottom')
ax.set_ylabel('held-out gain (bits/spike)', fontsize=7)
ax.set_title('selected model, all structures', fontsize=6.5, loc='left', pad=14)
lp(ax, 'C', x=-.3, y=1.0)
clean(ax)

# ── D: example cells, observed against held-out prediction ─────────────────
def smooth2(M, s=1.6):
    """Gaussian smoothing over visited bins, spreading into the gaps between
    them; bins with little visited neighbourhood stay blank."""
    w = np.isfinite(M).astype(float)
    ws = gaussian_filter(w, s)
    with np.errstate(invalid='ignore'):
        out = gaussian_filter(np.nan_to_num(M), s) / ws
    out[ws < .3] = np.nan
    return out


D = gs[2].subgridspec(4, 5, width_ratios=[.9, 1, 1, 1.45, 1.45], wspace=.3, hspace=.5)
hd = np.linspace(-180, 180, len(Z['c0_H_obs']) + 1)
hd = (hd[:-1] + hd[1:]) / 2
se = Z['S_edges']
sp = (se[:-1] + se[1:]) / 2
for i in range(4):
    mo, dy, c = Z[f'c{i}_id']
    m = str(Z[f'c{i}_model'])
    kind = CC[(CC.mouse == mo) & (CC.day == dy) & (CC.cluster_id == c)].cell_class_of1.iloc[0]
    at = fig.add_subplot(D[i, 0]); at.axis('off')
    at.text(1, .72, m, ha='right', va='center', fontsize=10, weight='bold',
            color=VCOL[m] if len(m) == 1 else 'k')
    at.text(1, .3, f'{KIND[kind]}\nM{mo} D{dy} cell {c}\n{Z[f"c{i}_rate"]:.1f} Hz, '
            f'{Z[f"c{i}_llh"]:.2f} bits/spike', ha='right', va='center', fontsize=5.6, color='0.3')
    if i == 0:
        lp(at, 'D', x=.2, y=1.0)
    po, pp = smooth2(Z[f'c{i}_P_obs']), smooth2(Z[f'c{i}_P_pred'])
    vmax = np.nanmax(po)
    for j, (M, lab) in enumerate(((po, 'observed'), (pp, 'held-out model'))):
        a = fig.add_subplot(D[i, 1 + j])
        a.imshow(M, origin='lower', cmap='viridis', vmin=0, vmax=vmax, extent=[0, 100, 0, 100],
                 interpolation='nearest')
        a.set_xticks([]); a.set_yticks([])
        if i == 0:
            a.set_title(lab, fontsize=6.5)
        if j == 0:
            a.text(-4, 50, f'{vmax:.0f} Hz', rotation=90, ha='right', va='center', fontsize=5.5,
                   color='0.3')
    for j, (x_, key, xl, xt) in enumerate(((hd, 'H', 'head direction (°)', [-180, 0, 180]),
                                           (sp, 'S', 'speed (cm/s)', [0, 20, 40]))):
        a = fig.add_subplot(D[i, 3 + j])
        a.plot(x_, Z[f'c{i}_{key}_pred'], color='0.7', lw=2.2, label='held-out model')
        a.plot(x_, Z[f'c{i}_{key}_obs'], color='k', lw=.8, label='observed')
        a.set_ylim(0, None); a.set_xticks(xt)
        a.set_xlim(*((-180, 180) if key == 'H' else (0, 41)))
        if i == 3:
            a.set_xlabel(xl, fontsize=6.5)
        else:
            a.set_xticklabels([])
        if j == 0:
            a.set_ylabel('Hz', fontsize=6.5)
        if i == 0 and j == 1:
            a.legend(fontsize=5.6, frameon=False, loc='lower right', bbox_to_anchor=(1.05, .95),
                     ncol=2, handlelength=1.4, columnspacing=.8)
        clean(a)

# ── E-H: the share of each structure's cells in each class ─────────────────
E = gs[3].subgridspec(1, 4, wspace=.5)
for j, r in enumerate(REGIONS):
    ax = fig.add_subplot(E[j])
    g = G[G.region == r]
    fr = np.array([(g.model == c).mean() for c in CLASSES]) * 100
    ax.bar(range(len(CLASSES)), fr, width=.75,
           color=[RCOL[r]] * (len(CLASSES) - 1) + ['0.75'], lw=0)
    ax.set_xticks(range(len(CLASSES)))
    ax.set_xticklabels(CLASSES[:-1] + ['none'], fontsize=6.3, rotation=90)
    ax.set_ylim(0, max(60, fr.max() * 1.1))
    ax.set_title(f'{r}: {len(g)} cells,\n{g.groupby(["mouse", "day"]).ngroups} sessions',
                 fontsize=6.5, loc='left', color=RCOL[r])
    if j == 0:
        ax.set_ylabel('cells (%)', fontsize=7)
    lp(ax, 'EFGH'[j], x=-.2, y=1.06)
    clean(ax)
    print(f'  {r} XGBoost classes ({len(g)} cells): ' +
          ', '.join(f'{c or "none"} {v:.1f}%' for c, v in zip(CLASSES, fr)))

for m, v in zip(CLASSES[:-1], vals):
    print(f'  {m}: {len(v)} cells, median {np.median(v):.3f} bits/spike')
for a, b in zip(path[:-1], path[1:]):
    print(f'  walkthrough {a} -> {b}: p = '
          f'{wilcoxon(L[models.index(b)], L[models.index(a)], alternative="greater").pvalue:.4f}')
out = f'{HERE}/fig1_supp_of_decomp.pdf'
plt.savefig(out, dpi=200, bbox_inches='tight')
plt.savefig(out.replace('.pdf', '.png'), dpi=110, bbox_inches='tight')
print(f'saved {out}')
