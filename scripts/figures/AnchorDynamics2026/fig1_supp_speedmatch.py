"""NOT PART OF THE FIGURE SET.

Speed matching is applied where it is needed -- the LFP analyses, where
theta amplitude and frequency both scale steeply with running speed. The
behavioural comparison in Figure 1 does not use it. This script is kept
only as a record that the check was run: the anchoring-behaviour link
survives matching (MEC uncued hit rate 68.2% anchored vs 42.4%,
p = 2.5e-4, 21 sessions), so nothing in Figure 1 rests on the speed
difference between states.
"""

import os, sys, warnings
import numpy as np, pandas as pd
warnings.filterwarnings('ignore')
import matplotlib; matplotlib.use('Agg')
import matplotlib.pyplot as plt
from scipy.stats import wilcoxon, mannwhitneyu
from sklearn.metrics import roc_auc_score
sys.path.insert(0, '/Users/harryclark/Documents/spatial-manifolds/src')
from spatial_manifolds.anchoring import ANCH_COLOR, NONANCH_COLOR

plt.rcParams['font.family'] = 'Arial'
FIG = '/Users/harryclark/Documents/spatial-manifolds/scripts/figures/AnchorDynamics2026'
PS = '/Users/harryclark/Documents/spatial-manifolds/data/population_state'
REGIONS = ['MEC', 'VIS', 'SUB', 'CB']
RCOL = {'MEC': '#7b4173', 'VIS': '#3f9b4f', 'SUB': '#c0723a', 'CB': '#4c72b0'}
N_Q, MIN_MATCHED = 10, 20
# session gating copied from Figure 1: >= MIN_N trials in each of the four
# trial-type x state cells, applied to the UNMATCHED data so that the same
# sessions enter before and after and the comparison is like-for-like
MIN_N = 10
# same session gating as Figure 1 and the companion supplement
MIN_PER_CELL, MIN_PER_TYPE = 10, 20

D = pd.read_csv(f'{PS}/nonmec_population_state.csv')
D = D[D.speed.notna()]


def match(sub, state, rng, n_q=N_Q):
    """Indices of a speed-matched subset of `sub`, equal counts per decile."""
    q = np.quantile(sub.speed, np.linspace(0, 1, n_q + 1))
    q[0] -= 1; q[-1] += 1
    keep = []
    for lo, hi in zip(q[:-1], q[1:]):
        m = (sub.speed >= lo) & (sub.speed < hi)
        ia = np.where(m & state)[0]
        iN = np.where(m & ~state)[0]
        k = min(len(ia), len(iN))
        if k:
            keep += list(rng.choice(ia, k, replace=False))
            keep += list(rng.choice(iN, k, replace=False))
    return np.array(sorted(keep), dtype=int)


rows, pre, post = [], [], []
for (mo, dy), d in D.groupby(['mouse', 'day']):
    rng = np.random.default_rng(abs(hash((mo, dy))) % 2**32)
    for r in REGIONS:
        if d[f'frac_{r}'].isna().all():
            continue
        rec = dict(mouse=mo, day=dy, region=r)
        ok = True
        for tt, nm in (('b', 'cued'), ('nb', 'uncued')):
            sub = d[(d.ttype == tt) & d[f'frac_{r}'].notna()].reset_index(drop=True)
            if len(sub) < MIN_PER_TYPE:
                ok = False; break
            st = (sub[f'frac_{r}'] > .5).values
            if st.sum() < MIN_PER_CELL or (~st).sum() < MIN_PER_CELL:
                ok = False; break
            # before
            rec[f'spd_a_pre_{nm}'] = sub.speed[st].mean()
            rec[f'spd_n_pre_{nm}'] = sub.speed[~st].mean()
            rec[f'auc_pre_{nm}'] = roc_auc_score(sub.hit, sub[f'frac_{r}'])
            # after
            k = match(sub, st, rng)
            if len(k) < MIN_MATCHED:
                ok = False; break
            ms = sub.iloc[k]; mst = st[k]
            if mst.sum() < 5 or (~mst).sum() < 5 or ms.hit.nunique() < 2:
                ok = False; break
            rec[f'spd_a_post_{nm}'] = ms.speed[mst].mean()
            rec[f'spd_n_post_{nm}'] = ms.speed[~mst].mean()
            rec[f'auc_post_{nm}'] = roc_auc_score(ms.hit, ms[f'frac_{r}'])
            rec[f'hit_{nm}_a'] = ms.hit[mst].mean()
            rec[f'hit_{nm}_n'] = ms.hit[~mst].mean()
            rec[f'n_pre_{nm}'], rec[f'n_post_{nm}'] = len(sub), len(k)
            if r == 'MEC':
                pre.append(sub.assign(state=st)[['speed', 'state']])
                post.append(ms.assign(state=mst)[['speed', 'state']])
        if ok:
            rec['adi_pre'] = rec['auc_pre_uncued'] - rec['auc_pre_cued']
            rec['adi_post'] = rec['auc_post_uncued'] - rec['auc_post_cued']
            rows.append(rec)
B = pd.DataFrame(rows)
B.to_csv(f'{PS}/speedmatched_behaviour.csv', index=False)
PRE, POST = pd.concat(pre, ignore_index=True), pd.concat(post, ignore_index=True)

print(f'{B.groupby(["mouse","day"]).ngroups} sessions')
for r in REGIONS:
    b = B[B.region == r]
    if len(b) < 3:
        print(f'  {r}: only {len(b)} sessions'); continue
    print(f'  {r}: n={len(b)}  AUC cued {b.auc_pre_cued.mean():.3f}->'
          f'{b.auc_post_cued.mean():.3f}  uncued {b.auc_pre_uncued.mean():.3f}->'
          f'{b.auc_post_uncued.mean():.3f}  ADI {b.adi_pre.mean():+.3f}->'
          f'{b.adi_post.mean():+.3f}')
    for nm in ('cued', 'uncued'):
        ha, hn = b[f'hit_{nm}_a'], b[f'hit_{nm}_n']
        m = ha.notna() & hn.notna()
        if m.sum() >= 3:
            print(f'      {nm} hit rate after matching: anchored {100*ha[m].mean():.1f}%'
                  f' vs non {100*hn[m].mean():.1f}%, p={wilcoxon(ha[m], hn[m]).pvalue:.3g}')

# ── figure ───────────────────────────────────────────────────────────────────
fig = plt.figure(figsize=(7.6, 8.8))
gs = fig.add_gridspec(3, 4, height_ratios=[1, 1.15, 1.15], hspace=.58, wspace=.52)

bins = np.linspace(0, 80, 33)
for j, (P, ttl) in enumerate(((PRE, 'before matching'), (POST, 'after matching'))):
    ax = fig.add_subplot(gs[0, j])
    for st, c, lb in ((True, ANCH_COLOR, 'anchored'), (False, NONANCH_COLOR, 'non-anch')):
        v = P.speed[P.state == st]
        h, _ = np.histogram(v, bins=bins, density=True)
        ax.plot(bins[:-1] + np.diff(bins) / 2, h, color=c, lw=1.6, label=lb)
        ax.fill_between(bins[:-1] + np.diff(bins) / 2, 0, h, color=c, alpha=.18,
                        linewidth=0, edgecolor='none')
    d_ = (P.speed[P.state].mean() - P.speed[~P.state].mean())
    ax.set_title(f'{ttl}\nΔ = {d_:+.2f} cm/s', fontsize=8.5, loc='left')
    ax.set_xlabel('Trial mean running speed (cm/s)', fontsize=8)
    if j == 0:
        ax.set_ylabel('Density', fontsize=8.5); ax.legend(fontsize=7, frameon=False)
    ax.tick_params(labelsize=7.5); ax.spines[['top', 'right']].set_visible(False)

ax = fig.add_subplot(gs[0, 3])
b = B[B.region == 'MEC']
for nm, mk in (('cued', 'o'), ('uncued', 's')):
    dpre = (b[f'spd_a_pre_{nm}'] - b[f'spd_n_pre_{nm}']).dropna()
    dpost = (b[f'spd_a_post_{nm}'] - b[f'spd_n_post_{nm}']).dropna()
    for i in range(len(dpre)):
        ax.plot([0, 1], [dpre.iloc[i], dpost.iloc[i]], color='0.8', lw=.5, zorder=1)
    ax.scatter(np.zeros(len(dpre)), dpre, s=10, marker=mk, color='0.35', lw=0, zorder=2)
    ax.scatter(np.ones(len(dpost)), dpost, s=10, marker=mk, color=ANCH_COLOR, lw=0,
               zorder=2)
ax.axhline(0, color='0.6', lw=.7, ls=':')
ax.set_xticks([0, 1]); ax.set_xticklabels(['before', 'after'], fontsize=8)
ax.set_xlim(-.35, 1.35)
ax.set_ylabel('speed difference\n(anchored − non, cm/s)', fontsize=8)
ax.set_title('per session', fontsize=8.5, loc='left')
ax.tick_params(labelsize=7.5); ax.spines[['top', 'right']].set_visible(False)

for j, r in enumerate(REGIONS):
    ax = fig.add_subplot(gs[1, j])
    b = B[B.region == r].dropna(subset=['hit_cued_a', 'hit_cued_n',
                                        'hit_uncued_a', 'hit_uncued_n'])
    if len(b) < 3:
        ax.text(.5, .5, f'{r}\ntoo few sessions', ha='center', va='center',
                fontsize=8, color='0.4', transform=ax.transAxes); ax.axis('off'); continue
    XS = [0, .8, 2.1, 2.9]
    vals = [b.hit_cued_a, b.hit_cued_n, b.hit_uncued_a, b.hit_uncued_n]
    cols = [ANCH_COLOR, NONANCH_COLOR, ANCH_COLOR, NONANCH_COLOR]
    rng = np.random.default_rng(0)
    for x, v, c in zip(XS, vals, cols):
        ax.bar(x, 100 * v.mean(), width=.78, color=c, lw=.7, edgecolor='k', zorder=1)
        ax.errorbar(x, 100 * v.mean(), yerr=100 * v.std(ddof=1) / np.sqrt(len(v)),
                    color='k', lw=1, capsize=2.5, zorder=4)
        ax.scatter(x + rng.uniform(-.15, .15, len(v)), 100 * v, s=5, color='0.25',
                   alpha=.5, lw=0, zorder=3)
    for x0, x1, va, vn in ((0, .8, b.hit_cued_a, b.hit_cued_n),
                           (2.1, 2.9, b.hit_uncued_a, b.hit_uncued_n)):
        p = wilcoxon(va, vn).pvalue
        star = 'n.s.' if p > .05 else ('*' if p > .01 else ('**' if p > .001 else '***'))
        ax.plot([x0, x0, x1, x1], [104, 107, 107, 104], color='0.3', lw=.8)
        ax.text((x0 + x1) / 2, 108, star, ha='center', fontsize=7, color='0.2')
    ax.set_xticks([.4, 2.5]); ax.set_xticklabels(['cued', 'uncued'], fontsize=7.5)
    ax.set_ylim(0, 120); ax.set_yticks([0, 50, 100])
    ax.set_title(f'{r}, speed-matched (n={len(b)})', fontsize=8.5, loc='left',
                 color=RCOL[r])
    if j == 0:
        ax.set_ylabel('Hit rate (%)', fontsize=8.5)
    ax.tick_params(labelsize=7.5); ax.spines[['top', 'right']].set_visible(False)

ax = fig.add_subplot(gs[2, 0])
for j, r in enumerate(REGIONS):
    b = B[B.region == r]
    if len(b) < 3:
        continue
    for k, nm in enumerate(('cued', 'uncued')):
        x = j + (k - .5) * .34
        for tag, mf in (('pre', 'white'), ('post', RCOL[r])):
            xx = x + (.07 if tag == 'post' else -.07)
            v = b[f'auc_{tag}_{nm}']
            ax.errorbar(xx, v.mean(), yerr=v.std(ddof=1) / np.sqrt(len(v)),
                        color=RCOL[r], marker='o' if k == 0 else 's', ms=4, lw=1,
                        capsize=2, mfc=mf)
ax.axhline(.5, color='0.6', lw=.7, ls=':')
ax.set_xticks(range(len(REGIONS))); ax.set_xticklabels(REGIONS, fontsize=8)
ax.set_ylabel('AUC: state predicts hit', fontsize=8.5)
ax.set_title('open = before, filled = after\ncircle = cued, square = uncued',
             fontsize=7.5, loc='left')
ax.tick_params(labelsize=7.5); ax.spines[['top', 'right']].set_visible(False)

ax = fig.add_subplot(gs[2, 2])
for j, r in enumerate(REGIONS):
    b = B[B.region == r]
    if len(b) < 3:
        continue
    for i in range(len(b)):
        ax.plot([j - .16, j + .16], [b.adi_pre.iloc[i], b.adi_post.iloc[i]],
                color='0.85', lw=.5, zorder=1)
    ax.scatter(np.full(len(b), j - .16), b.adi_pre, s=10, color='0.45', lw=0, zorder=2)
    ax.scatter(np.full(len(b), j + .16), b.adi_post, s=10, color=RCOL[r], lw=0, zorder=2)
    p = wilcoxon(b.adi_post).pvalue
    ax.text(j, .42, 'n.s.' if p > .05 else '*', ha='center', fontsize=7.5)
ax.axhline(0, color='0.6', lw=.7, ls=':')
ax.set_xticks(range(len(REGIONS))); ax.set_xticklabels(REGIONS, fontsize=8)
ax.set_ylabel('ADI (AUC uncued − cued)', fontsize=8.5)
ax.set_title('grey = before, colour = after', fontsize=7.5, loc='left')
ax.tick_params(labelsize=7.5); ax.spines[['top', 'right']].set_visible(False)

ax = fig.add_subplot(gs[2, 3])
b = B[B.region == 'MEC']
for k, nm in enumerate(('cued', 'uncued')):
    v = 100 * b[f'n_post_{nm}'] / b[f'n_pre_{nm}']
    ax.bar(k, v.mean(), width=.6, color='0.6', lw=.7, edgecolor='k')
    ax.errorbar(k, v.mean(), yerr=v.std(ddof=1) / np.sqrt(len(v)), color='k', lw=1,
                capsize=3)
ax.set_xticks([0, 1]); ax.set_xticklabels(['cued', 'uncued'], fontsize=8)
ax.set_ylabel('trials retained (%)', fontsize=8.5)
ax.set_title('cost of matching', fontsize=8.5, loc='left')
ax.set_ylim(0, 100)
ax.tick_params(labelsize=7.5); ax.spines[['top', 'right']].set_visible(False)

out = f'{FIG}/fig1_supp_speed_matching.pdf'
plt.savefig(out, dpi=200, bbox_inches='tight')
print(f'\nsaved {out}')
