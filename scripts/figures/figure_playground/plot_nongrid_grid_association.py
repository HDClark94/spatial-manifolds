import pandas as pd, numpy as np
import matplotlib; matplotlib.use('Agg')
import matplotlib.pyplot as plt

df = pd.read_csv('/Users/harryclark/Documents/spatial-manifolds/data/nongrid_grid_association_results.csv')

fig, axes = plt.subplots(1, 2, figsize=(10, 4))

ax = axes[0]
ax.hist(df.D, bins=60, color='#3171ae', alpha=0.85)
ax.axvline(0, color='k', ls='--', lw=1)
ax.axvline(df.D.mean(), color='#c0392b', lw=1.5, label=f'mean={df.D.mean():.4f}')
ax.set_xlabel('D = mean ΔpR²(grid) − mean ΔpR²(non-grid)')
ax.set_ylabel('non-grid cells')
ax.set_title(f'Grid-association score, n={len(df)} cells\n{(df.q_value<0.05).sum()}/{len(df)} sig. at q<0.05', fontsize=9)
ax.legend(fontsize=8, frameon=False)
ax.spines[['top', 'right']].set_visible(False)

ax = axes[1]
obs = np.sort(df.p_value.values)
n = len(obs)
expected = np.arange(1, n + 1) / n
ax.plot(expected, obs, '.', color='#3171ae', ms=2, alpha=0.5)
ax.plot([0, 1], [0, 1], 'k--', lw=1)
ax.set_xlabel('expected p (uniform null)')
ax.set_ylabel('observed p (sorted)')
ax.set_title('P-value QQ plot vs uniform null\n(on the diagonal = no true effect)', fontsize=9)
ax.spines[['top', 'right']].set_visible(False)

fig.suptitle('Non-grid cells show no elevated coupling to grid cells vs other non-grid cells',
             fontsize=10, fontweight='bold')
plt.tight_layout()
out = '/Users/harryclark/Documents/spatial-manifolds/data/nongrid_grid_association_summary'
plt.savefig(out + '.png', dpi=150, bbox_inches='tight')
plt.savefig(out + '.pdf', bbox_inches='tight')
print(f'saved {out}.png')
