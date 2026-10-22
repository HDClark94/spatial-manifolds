import pandas as pd, numpy as np
import matplotlib; matplotlib.use('Agg')
import matplotlib.pyplot as plt

df = pd.read_csv('/Users/harryclark/Documents/spatial-manifolds/data/module_independence_cca_results.csv')
of1 = df[df.session_type == 'OF1'].set_index(['mouse', 'day'])
vr = df[df.session_type == 'VR'].set_index(['mouse', 'day'])
common = of1.index.intersection(vr.index)

fig, axes = plt.subplots(1, 2, figsize=(10, 4.2))

ax = axes[0]
r_of1, r_vr = of1.loc[common, 'rho_obs'], vr.loc[common, 'rho_obs']
sig_both = (of1.loc[common, 'p_value'] < 0.05) & (vr.loc[common, 'p_value'] < 0.05)
ax.scatter(r_of1[sig_both], r_vr[sig_both], color='#3171ae', s=35, label='significant in both', zorder=3)
ax.scatter(r_of1[~sig_both], r_vr[~sig_both], color='#c0392b', s=35, label='disagreement', zorder=3)
lims = [0, max(r_of1.max(), r_vr.max()) * 1.1]
ax.plot(lims, lims, 'k--', lw=1, alpha=0.5)
corr = np.corrcoef(r_of1, r_vr)[0, 1]
ax.set_xlabel('OF1 rho_obs'); ax.set_ylabel('VR rho_obs')
ax.set_title(f'OF1 vs VR cross-module correlation\n(r={corr:.2f}, n={len(common)} sessions)', fontsize=9)
ax.legend(fontsize=7, frameon=False)
ax.spines[['top', 'right']].set_visible(False)

ax = axes[1]
order = df.groupby(['mouse', 'day'])['rho_obs'].mean().sort_values().index
df['ceiling_max'] = df[['ceiling_mod0', 'ceiling_mod1']].max(axis=1)
x = np.arange(len(order))
for i, (m, d) in enumerate(order):
    for stype, marker, dx in [('OF1', 'o', -0.15), ('VR', '^', 0.15)]:
        row = df[(df.mouse == m) & (df.day == d) & (df.session_type == stype)]
        if len(row) == 0:
            continue
        row = row.iloc[0]
        color = '#3171ae' if row.p_value < 0.05 else '#999999'
        ax.scatter(i + dx, row.rho_obs, color=color, marker=marker, s=18, zorder=3)
        ax.plot([i + dx, i + dx], [row.null_mean, row.rho_obs], color=color, lw=0.7, alpha=0.5, zorder=2)
        if not np.isnan(row.ceiling_max):
            ax.scatter(i + dx, row.ceiling_max, color='#2ecc71', marker='_', s=40, zorder=3)
ax.set_xticks(x); ax.set_xticklabels([f'M{m}D{d}' for m, d in order], rotation=90, fontsize=6)
ax.set_ylabel('canonical correlation')
ax.set_title('rho_obs (dot) vs null (line base) vs ceiling (green tick)\nOF1=circle, VR=triangle, grey=n.s.', fontsize=9)
ax.spines[['top', 'right']].set_visible(False)

plt.tight_layout()
out = '/Users/harryclark/Documents/spatial-manifolds/data/module_independence_cca_summary'
plt.savefig(out + '.png', dpi=150, bbox_inches='tight')
plt.savefig(out + '.pdf', bbox_inches='tight')
print(f'saved {out}.png')
