"""Figure S3: the anchoring state in each recorded structure, related to Figure 1.

A     hit rate by state and cue condition, with the state computed from the
      cells of each recorded structure in turn, on all trials
B     the Anchoring Dependence Index, AUC(uncued) - AUC(cued), on the same sessions
C, D  the same on the speed-matched subset: within session and cue type, equal
      numbers of anchored and non-anchored trials per decile of trial mean
      running speed. On the track speed and position are tightly coupled, so a
      state difference in speed could otherwise carry the behavioural effect
E     the anchored - non-anchored speed difference per session, before and after
      matching, per structure
F     the share of trials matching keeps
G     agreement between structures, as Figure 3G does for identities: the mean
      correlation of anchoring label sequences between cells of two structures,
      over an independent circular-shift null, switching sessions only
(The open-field decomposition, G-J here until 2026-10-31, is now Figure S4,
fig1_supp_of_decomp.py.)
H     the same per session, for each structure recorded alongside MEC: within
      MEC, between MEC and the structure, and within the structure
"""

import os, sys, warnings
import numpy as np, pandas as pd
warnings.filterwarnings('ignore')
import matplotlib; matplotlib.use('Agg')
import matplotlib.pyplot as plt
from scipy.stats import wilcoxon
from sklearn.metrics import roc_auc_score
sys.path.insert(0, '/Users/harryclark/Documents/spatial-manifolds/src')
from spatial_manifolds.anchoring import ANCH_COLOR, NONANCH_COLOR, load_session_labels
from of_xgb_decomp import fam

plt.rcParams['font.family'] = 'Arial'
FIG = '/Users/harryclark/Documents/spatial-manifolds/scripts/figures/AnchorDynamics2026'
PS = '/Users/harryclark/Documents/spatial-manifolds/data/population_state'
REGIONS = ['MEC', 'VIS', 'SUB', 'CB']
RCOL = {'MEC': '#7b4173', 'VIS': '#3f9b4f', 'SUB': '#c0723a', 'CB': '#4c72b0'}
# a session enters for a structure with >= MIN_PER_TYPE trials of each cue type and
# >= MIN_PER_CELL trials in each state within each type; its speed-matched subset
# is used (C-F) if each cue type keeps >= MIN_MATCHED trials, >= 5 per state and
# both outcomes
MIN_PER_CELL, MIN_PER_TYPE = 10, 20
N_Q, MIN_MATCHED = 10, 20

D = pd.read_csv(f'{PS}/nonmec_population_state.csv')


def match(sub, state, rng, n_q=N_Q):
    """Indices of a speed-matched subset of `sub`, equal counts per decile."""
    q = np.quantile(sub.speed, np.linspace(0, 1, n_q + 1))
    q[0] -= 1; q[-1] += 1
    keep = []
    for lo, hi in zip(q[:-1], q[1:]):
        m = (sub.speed >= lo) & (sub.speed < hi)
        ia, iN = np.where(m & state)[0], np.where(m & ~state)[0]
        k = min(len(ia), len(iN))
        if k:
            keep += list(rng.choice(ia, k, replace=False))
            keep += list(rng.choice(iN, k, replace=False))
    return np.array(sorted(keep), dtype=int)


rows = []
for (mo, dy), d in D.groupby(['mouse', 'day']):
    rng = np.random.default_rng(abs(hash((mo, dy))) % 2**32)
    for r in REGIONS:
        if d[f'frac_{r}'].isna().all():
            continue
        rec, ok, matched = dict(mouse=mo, day=dy, region=r), True, True
        for tt, nm in (('b', 'cued'), ('nb', 'uncued')):
            sub = d[(d.ttype == tt) & d[f'frac_{r}'].notna() & d.speed.notna()].reset_index(drop=True)
            st = (sub[f'frac_{r}'] > .5).values
            if len(sub) < MIN_PER_TYPE or st.sum() < MIN_PER_CELL or (~st).sum() < MIN_PER_CELL:
                ok = False; break
            rec[f'auc_{nm}'] = roc_auc_score(sub.hit, sub[f'frac_{r}'])
            rec[f'hit_{nm}_a'] = sub.hit[st].mean()
            rec[f'hit_{nm}_n'] = sub.hit[~st].mean()
            rec[f'dspd_{nm}'] = sub.speed[st].mean() - sub.speed[~st].mean()
            rec[f'n_{nm}'] = len(sub)
            k = match(sub, st, rng)
            ms, mst = sub.iloc[k], st[k]
            if (len(k) < MIN_MATCHED or mst.sum() < 5 or (~mst).sum() < 5
                    or ms.hit.nunique() < 2):
                matched = False; continue
            rec[f'm_auc_{nm}'] = roc_auc_score(ms.hit, ms[f'frac_{r}'])
            rec[f'm_hit_{nm}_a'] = ms.hit[mst].mean()
            rec[f'm_hit_{nm}_n'] = ms.hit[~mst].mean()
            rec[f'm_dspd_{nm}'] = ms.speed[mst].mean() - ms.speed[~mst].mean()
            rec[f'm_n_{nm}'] = len(k)
        if not ok:
            continue
        rec['adi'] = rec['auc_uncued'] - rec['auc_cued']
        rec['matched'] = matched
        if matched:
            rec['m_adi'] = rec['m_auc_uncued'] - rec['m_auc_cued']
        else:
            rec = {k: v for k, v in rec.items() if not k.startswith('m_')}
        rows.append(rec)
B = pd.DataFrame(rows)
BM = B[B.matched]
B.to_csv(f'{PS}/region_behaviour.csv', index=False)

# K: agreement between structures, computed as Figure 2G (fig2_single_units.py)
# with region family in place of identity. Units excluded since the labels were
# built (ISI exclusion) are dropped here.
U = pd.read_csv(f'{PS}/unit_table.csv')
C = pd.read_csv(f'{os.path.dirname(PS)}/cell_classifications_v2.csv')
U = U.merge(C[['mouse', 'day', 'cluster_id', 'cell_class_of1']].rename(
    columns={'cell_class_of1': 'cls_now'}), on=['mouse', 'day', 'cluster_id'], how='left')
U = U[U.cls_now != 'EXCL']
FAM = {(m, d, c): fam(b) for m, d, c, b in zip(U.mouse, U.day, U.cluster_id, U.brain_region)}
rng = np.random.default_rng(0)
rows = []
for (mo, dy), _ in U[U.sess_switches].groupby(['mouse', 'day']):
    z = load_session_labels(int(mo), int(dy))
    L, ids = z['labels'], z['cluster_id'].astype(int)
    lab = np.array([FAM.get((mo, dy, int(c))) for c in ids])
    keep = (lab != None) & (np.nanstd(L, axis=1) > 0)
    V, lab = np.nan_to_num(L[keep]), lab[keep]
    if len(V) < 8:
        continue
    Cm = np.corrcoef(V)
    Nm = np.mean([np.corrcoef(np.array(
        [np.roll(v, int(rng.integers(5, V.shape[1] - 5))) for v in V]))
        for _ in range(20)], axis=0)
    for a in REGIONS:
        for b in REGIONS:
            ia, ib = np.where(lab == a)[0], np.where(lab == b)[0]
            if len(ia) < 2 or len(ib) < 2:
                continue
            co, cn = Cm[np.ix_(ia, ib)], Nm[np.ix_(ia, ib)]
            if a == b:
                iu = ~np.eye(len(ia), dtype=bool)
                co, cn = co[iu], cn[iu]
            rows.append(dict(mouse=mo, day=dy, a=a, b=b, n_a=len(ia), n_b=len(ib),
                             excess=float(np.nanmean(co) - np.nanmean(cn))))
AG = pd.DataFrame(rows)
AG.to_csv(f'{PS}/region_agreement.csv', index=False)

print(f'{B.groupby(["mouse","day"]).ngroups} sessions, {BM.groupby(["mouse","day"]).ngroups} speed-matched')
for r in REGIONS:
    for tag, b, pre in (('all trials', B[B.region == r], ''), ('matched', BM[BM.region == r], 'm_')):
        for nm in ('cued', 'uncued'):
            ha, hn = b[f'{pre}hit_{nm}_a'], b[f'{pre}hit_{nm}_n']
            print(f'  {r} {nm}, {tag} (n={len(b)}): anchored {100*ha.mean():.1f}%'
                  f' vs non {100*hn.mean():.1f}%, p={wilcoxon(ha, hn).pvalue:.3g}')

# ── figure ───────────────────────────────────────────────────────────────────
fig = plt.figure(figsize=(7.6, 9.6))

def lp(ax, s_, x=-.26, y=1.02):
    """Bold panel letter, 10 pt, as in every other figure."""
    ax.text(x, y, s_, transform=ax.transAxes, fontsize=10,
            weight='bold', va='bottom', ha='right')


gs = fig.add_gridspec(4, 4, height_ratios=[1.15, 1.15, 1, 1.1], hspace=.62, wspace=.52)

def stars(p):
    return 'n.s.' if p > .05 else ('*' if p > .01 else ('**' if p > .001 else '***'))


def hit_panel(ax, B, pre, title, letter):
    """Hit rate by state and cue, all structures on one axis."""
    rng = np.random.default_rng(0)
    DX = (-.66, -.30, .30, .66)                  # cued anchored/non, uncued anchored/non
    CELLS = (('cued', 'a'), ('cued', 'n'), ('uncued', 'a'), ('uncued', 'n'))
    for j, r in enumerate(REGIONS):
        b = B[B.region == r]
        x0 = 2.2 * j
        for (nm, st), dx in zip(CELLS, DX):
            v = 100 * b[f'{pre}hit_{nm}_{st}']
            c = ANCH_COLOR if st == 'a' else NONANCH_COLOR
            ax.bar(x0 + dx, v.mean(), width=.34, color=c, lw=.7, edgecolor='k', zorder=1,
                   label=('anchored' if st == 'a' else 'non-anch') if j == 0 and nm == 'cued' else None)
            ax.errorbar(x0 + dx, v.mean(), yerr=v.std(ddof=1) / np.sqrt(len(v)),
                        color='k', lw=1, capsize=2, zorder=4)
            ax.scatter(x0 + dx + rng.uniform(-.08, .08, len(v)), v, s=4, color='0.25',
                       alpha=.5, lw=0, zorder=3)
        for nm, (xa, xn) in (('cued', DX[:2]), ('uncued', DX[2:])):
            p = wilcoxon(b[f'{pre}hit_{nm}_a'], b[f'{pre}hit_{nm}_n']).pvalue
            ax.plot([x0 + xa, x0 + xa, x0 + xn, x0 + xn], [104, 107, 107, 104], color='0.3', lw=.8)
            ax.text(x0 + (xa + xn) / 2, 108, stars(p), ha='center', fontsize=7, color='0.2')
        ax.text(x0, -.2, f'{r} (n={len(b)})', transform=ax.get_xaxis_transform(), ha='center',
                va='top', fontsize=8.5, color=RCOL[r])
    ax.set_xticks([2.2 * j + d for j in range(len(REGIONS)) for d in (-.48, .48)])
    ax.set_xticklabels(['cued', 'uncued'] * len(REGIONS), fontsize=7.5)
    ax.set_ylim(0, 120); ax.set_yticks([0, 50, 100]); ax.set_xlim(-1.05, 2.2 * 3 + 1.05)
    ax.set_ylabel('Hit rate (%)', fontsize=8.5)
    ax.set_title(title, fontsize=8.5, loc='left')
    if letter == 'A':
        ax.legend(fontsize=7, frameon=False, loc='upper left', bbox_to_anchor=(.62, 1.22), ncol=2)
    lp(ax, letter, x=-.065, y=1.10)
    ax.tick_params(labelsize=7.5); ax.spines[['top', 'right']].set_visible(False)


def adi_panel(ax, B, col, title, letter):
    """The Anchoring Dependence Index per session, by structure."""
    rng = np.random.default_rng(0)
    for j, r in enumerate(REGIONS):
        v = B[B.region == r][col]
        ax.scatter(j + rng.uniform(-.13, .13, len(v)), v, s=8, color=RCOL[r], lw=0, alpha=.7, zorder=2)
        ax.errorbar(j + .3, v.mean(), yerr=v.std(ddof=1) / np.sqrt(len(v)), color='k', marker='o',
                    ms=3.5, lw=1, capsize=2, zorder=3)
        p = wilcoxon(v).pvalue
        ax.text(j + .1, .74, stars(p), ha='center', fontsize=7, color='0.2')
        print(f'  {r} {col} (n={len(v)}): {v.mean():+.3f}, p={p:.3g}')
    ax.axhline(0, color='0.6', lw=.7, ls=':')
    ax.set_xticks(range(len(REGIONS))); ax.set_xticklabels(REGIONS, fontsize=8)
    for t, r in zip(ax.get_xticklabels(), REGIONS):
        t.set_color(RCOL[r])
    ax.set_ylim(-.4, .82)
    ax.set_ylabel('ADI (AUC uncued − cued)', fontsize=8.5)
    ax.set_title(title, fontsize=8.5, loc='left')
    lp(ax, letter, x=-.30, y=1.10)
    ax.tick_params(labelsize=7.5); ax.spines[['top', 'right']].set_visible(False)


hit_panel(fig.add_subplot(gs[0, :3]), B, '', 'state from each structure, all trials', 'A')
adi_panel(fig.add_subplot(gs[0, 3]), B, 'adi', 'cue dependence', 'B')
hit_panel(fig.add_subplot(gs[1, :3]), BM, 'm_', 'speed-matched trials', 'C')
adi_panel(fig.add_subplot(gs[1, 3]), BM, 'm_adi', 'speed-matched', 'D')

# E: what matching does to the speed difference between states, per structure;
# each session contributes its cued (circle) and uncued (square) trials
ax = fig.add_subplot(gs[2, :3])
for j, r in enumerate(REGIONS):
    b = BM[BM.region == r]
    for nm, mk in (('cued', 'o'), ('uncued', 's')):
        v0, v1 = b[f'dspd_{nm}'].values, b[f'm_dspd_{nm}'].values
        x0, x1 = 2.2 * j - .4, 2.2 * j + .4
        for a_, b_ in zip(v0, v1):
            ax.plot([x0, x1], [a_, b_], color='0.82', lw=.5, zorder=1)
        ax.scatter(np.full(len(v0), x0), v0, s=9, marker=mk, color='0.45', lw=0, zorder=2)
        ax.scatter(np.full(len(v1), x1), v1, s=9, marker=mk, color=RCOL[r], lw=0, zorder=2)
    d0 = np.r_[b.dspd_cued, b.dspd_uncued]; d1 = np.r_[b.m_dspd_cued, b.m_dspd_uncued]
    ax.text(2.2 * j, -.2, f'{r} (n={len(b)})', transform=ax.get_xaxis_transform(), ha='center',
            va='top', fontsize=8.5, color=RCOL[r])
    print(f'  {r} speed difference anchored - non: {d0.mean():+.2f} -> {d1.mean():+.2f} cm/s '
          f'(|d| {np.abs(d0).mean():.2f} -> {np.abs(d1).mean():.2f})')
ax.axhline(0, color='0.6', lw=.7, ls=':')
ax.set_xticks([2.2 * j + d for j in range(len(REGIONS)) for d in (-.4, .4)])
ax.set_xticklabels(['before', 'after'] * len(REGIONS), fontsize=7.5)
ax.set_xlim(-1.05, 2.2 * 3 + 1.05)
ax.set_ylabel('speed difference\n(anchored − non, cm/s)', fontsize=8.5)
ax.set_title('trial mean running speed, before and after matching (circle cued, square uncued)',
             fontsize=8.5, loc='left')
lp(ax, 'E', x=-.065, y=1.04)
ax.tick_params(labelsize=7.5); ax.spines[['top', 'right']].set_visible(False)

# F: the cost of matching
ax = fig.add_subplot(gs[2, 3])
for j, r in enumerate(REGIONS):
    b = BM[BM.region == r]
    for k, nm in enumerate(('cued', 'uncued')):
        v = 100 * b[f'm_n_{nm}'] / b[f'n_{nm}']
        x = j + (k - .5) * .38
        ax.bar(x, v.mean(), width=.34, color=RCOL[r], alpha=1 if k else .55, lw=0)
        ax.errorbar(x, v.mean(), yerr=v.std(ddof=1) / np.sqrt(len(v)), color='k', lw=1, capsize=2)
ax.set_xticks(range(len(REGIONS))); ax.set_xticklabels(REGIONS, fontsize=8)
for t, r in zip(ax.get_xticklabels(), REGIONS):
    t.set_color(RCOL[r])
ax.set_ylim(0, 100)
ax.set_ylabel('trials retained (%)', fontsize=8.5)
ax.set_title('cost of matching\n(pale cued, dark uncued)', fontsize=7.5, loc='left')
lp(ax, 'F', x=-.30, y=1.04)
ax.tick_params(labelsize=7.5); ax.spines[['top', 'right']].set_visible(False)

# G: agreement between structures over the null, as Figure 2G; cells with fewer
# than five sessions are greyed
ax = fig.add_subplot(gs[3, :2])
Mx = AG.groupby(['a', 'b']).excess.mean().unstack().reindex(index=REGIONS, columns=REGIONS)
Nx = AG.groupby(['a', 'b']).size().unstack().reindex(index=REGIONS, columns=REGIONS).fillna(0)
vmax = np.nanmax(Mx.values[Nx.values >= 5])
ax.imshow(np.where(Nx.values >= 5, Mx.values, np.nan), cmap='magma', vmin=0, vmax=vmax)
cm = plt.get_cmap('magma')
for i in range(4):
    for j in range(4):
        v, n = Mx.values[i, j], int(Nx.values[i, j])
        if not np.isfinite(v):
            continue
        if n >= 5:
            rgb = cm(max(v, 0) / vmax)[:3]
            col = '0.1' if .299 * rgb[0] + .587 * rgb[1] + .114 * rgb[2] > .55 else 'white'
        else:
            col = '0.55'
        ax.text(j, i - .12, f'{v:.3f}', ha='center', va='center', fontsize=7, color=col)
        ax.text(j, i + .22, f'n={n}', ha='center', va='center', fontsize=5.5, color=col)
ax.set_xticks(range(4)); ax.set_xticklabels(REGIONS, fontsize=8)
ax.set_yticks(range(4)); ax.set_yticklabels(REGIONS, fontsize=8)
for t, r in zip(ax.get_xticklabels() + ax.get_yticklabels(), REGIONS * 2):
    t.set_color(RCOL[r])
ax.tick_params(length=0)
ax.set_title('agreement over null', fontsize=7.5, loc='left')
lp(ax, 'G', x=-.14, y=1.02)

# H: the same within sessions that record both MEC and the structure
ax = fig.add_subplot(gs[3, 2:])
W = AG.pivot_table(index=['mouse', 'day'], columns=['a', 'b'], values='excess')
for k, x in enumerate(['VIS', 'SUB', 'CB']):
    m = W[[('MEC', 'MEC'), ('MEC', x), (x, x)]].dropna()
    xs = 4 * k + np.array([0, 1, 2])
    for _, row in m.iterrows():
        ax.plot(xs, row.values, color='0.82', lw=.5, zorder=1)
    for xx, col, c in zip(xs, m.columns, ['MEC', None, x]):
        v = m[col]
        ax.errorbar(xx, v.mean(), yerr=v.std(ddof=1) / np.sqrt(len(v)), color='k', marker='o',
                    ms=4, mfc=RCOL[c] if c else 'white', mec='k', lw=1, capsize=2, zorder=3)
    ax.text(xs[1], -.115, f'n={len(m)}', ha='center', fontsize=6.5, color='0.35')
    print(f'  MEC-{x} ({len(m)} sessions): within MEC {m.iloc[:, 0].mean():.3f}, '
          f'between {m.iloc[:, 1].mean():.3f}, within {x} {m.iloc[:, 2].mean():.3f}')
ax.axhline(0, color='0.6', lw=.7, ls=':')
ax.set_xticks([4 * k + d for k in range(3) for d in range(3)])
ax.set_xticklabels([l for x in ['VIS', 'SUB', 'CB'] for l in ('MEC', f'MEC-{x}', x)],
                   fontsize=6.5, rotation=40, ha='right')
for t, r in zip(ax.get_xticklabels(), [r for x in ['VIS', 'SUB', 'CB'] for r in ('MEC', None, x)]):
    t.set_color(RCOL[r] if r else 'k')
ax.set_ylim(-.13, None)
ax.set_ylabel('agreement over null', fontsize=8.5)
ax.set_title('within and between, same sessions', fontsize=7.5, loc='left')
lp(ax, 'H', x=-.24, y=1.02)
ax.tick_params(labelsize=7); ax.spines[['top', 'right']].set_visible(False)

out = f'{FIG}/fig1_supp_regions.pdf'
plt.savefig(out, dpi=200, bbox_inches='tight')
print(f'\nsaved {out}')
