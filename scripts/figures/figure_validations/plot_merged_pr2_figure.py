"""Within- vs cross-group pairwise coupling (raw pR2 and delta pR2) for the
100 sampled units (50 narrow-origin + 50 wide-origin) in the merged net7
dominance-test network. Coupling matrix ordered by group, plus a
within-vs-cross summary distribution with a Mann-Whitney test.
"""
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from scipy import stats

BASE = '/Users/harryclark/Documents/spatial-manifolds/data/rnn_xgboost'
c = np.load(f'{BASE}/pr2_merged_net7_cache.npz')
group_id = c['group_id']; pr2_base = c['pr2_base']; pr2_cell = c['pr2_cell']; pr2_full = c['pr2_full']
K0, K1 = int(c['K0']), int(c['K1'])
N = len(group_id)
eye = np.eye(N, dtype=bool)
within = group_id[:, None] == group_id[None, :]
delta = pr2_full - pr2_base[None, :]

wv = delta[within & ~eye]; xv = delta[~within]
u_stat, p_val = stats.mannwhitneyu(wv, xv)

fig, axes = plt.subplots(1, 3, figsize=(11, 3.8), gridspec_kw={'width_ratios': [1, 1, 0.8]})

for ax, mat, title in zip(axes[:2], [pr2_cell, delta], ['raw pR2', 'delta pR2']):
    m = mat.copy()
    np.fill_diagonal(m, np.nan)
    im = ax.imshow(m, cmap='viridis', vmin=0, vmax=np.nanpercentile(m, 99))
    ax.axhline(K0 - 0.5, color='white', lw=1.2); ax.axvline(K0 - 0.5, color='white', lw=1.2)
    ax.set_title(title, fontsize=10)
    ax.set_xlabel('target unit'); ax.set_ylabel('predictor unit')
    ax.set_xticks([K0/2, K0 + K1/2]); ax.set_xticklabels(['narrow', 'wide'], fontsize=8)
    ax.set_yticks([K0/2, K0 + K1/2]); ax.set_yticklabels(['narrow', 'wide'], fontsize=8)
    plt.colorbar(im, ax=ax, fraction=0.046, pad=0.04)

ax = axes[2]
parts = ax.violinplot([wv, xv], showmedians=True)
for pc_, color in zip(parts['bodies'], ['#3171ae', '#c0392b']):
    pc_.set_facecolor(color); pc_.set_alpha(0.7)
ax.set_xticks([1, 2]); ax.set_xticklabels([f'within\n(n={len(wv)})', f'cross\n(n={len(xv)})'], fontsize=8)
ax.set_ylabel('delta pR2')
ax.set_title(f'within={np.mean(wv):.4f}\ncross={np.mean(xv):.4f}\np={p_val:.1e}', fontsize=9)
ax.spines[['top', 'right']].set_visible(False)

fig.suptitle('merged net7 (matched-yield pair): pairwise unit coupling, narrow-origin vs wide-origin',
             fontsize=10, fontweight='bold')
plt.tight_layout()
out = f'{BASE}/merged_net7_dominance_Ng2048/pr2_within_cross_coupling'
plt.savefig(out + '.png', dpi=150, bbox_inches='tight')
plt.savefig(out + '.pdf', bbox_inches='tight')
print(f'saved {out}.png', flush=True)
