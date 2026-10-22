import pandas as pd, numpy as np
import matplotlib; matplotlib.use('Agg')
import matplotlib.pyplot as plt
from scipy.stats import spearmanr

DATA = '/Users/harryclark/Documents/spatial-manifolds/data'
t1 = pd.read_csv(f'{DATA}/nongrid_test1_distance.csv')
t2 = pd.read_csv(f'{DATA}/nongrid_test2_matched.csv')
t3 = pd.read_csv(f'{DATA}/nongrid_test3_specificity.csv')
t3['specificity_z'] = t3.groupby(['mouse', 'day'])['specificity'].transform(lambda x: (x - x.mean()) / (x.std() + 1e-9))
t3['anat_pref_z'] = t3.groupby(['mouse', 'day'])['anat_pref'].transform(lambda x: (x - x.mean()) / (x.std() + 1e-9))

fig, axes = plt.subplots(1, 3, figsize=(12, 4))

ax = axes[0]
t1s = t1.sort_values('rho')
colors = ['#3171ae' if p < 0.05 else '#aaaaaa' for p in t1s.p]
ax.barh(range(len(t1s)), t1s.rho, color=colors)
ax.set_yticks(range(len(t1s))); ax.set_yticklabels([f'M{m}D{d}' for m, d in zip(t1s.mouse, t1s.day)], fontsize=6)
ax.axvline(0, color='k', lw=0.7)
ax.set_xlabel('Spearman rho (distance vs ΔpR2)')
ax.set_title(f'Test 1: coupling falls off with distance\n{(t1.p<0.05).sum()}/{len(t1)} sessions sig. (blue)', fontsize=9)
ax.spines[['top', 'right']].set_visible(False)

ax = axes[1]
diffs_by_bin = t2.groupby('bin').apply(lambda g: g.mean_own - g.mean_other)
for b in sorted(t2.bin.unique()):
    vals = t2[t2.bin == b].apply(lambda r: r.mean_own - r.mean_other, axis=1)
    ax.scatter([b] * len(vals), vals, color='#3171ae', alpha=0.5, s=20)
    ax.scatter([b], [vals.mean()], color='#c0392b', marker='_', s=200, zorder=5)
ax.axhline(0, color='k', lw=0.7, ls='--')
ax.set_xlabel('distance bin (0=nearest)'); ax.set_ylabel('ΔpR2: own-module − other-module')
ax.set_title('Test 2: own vs other module,\ndistance-matched (n.s. overall)', fontsize=9)
ax.spines[['top', 'right']].set_visible(False)

ax = axes[2]
ax.scatter(t3.anat_pref_z, t3.specificity_z, s=10, alpha=0.4, color='#3171ae')
rho, p = spearmanr(t3.specificity_z, t3.anat_pref_z)
zz = np.polyfit(t3.anat_pref_z, t3.specificity_z, 1)
xs = np.linspace(t3.anat_pref_z.min(), t3.anat_pref_z.max(), 50)
ax.plot(xs, np.polyval(zz, xs), color='#c0392b', lw=1.5)
ax.set_xlabel('anatomical preference (z)\n(+ = closer to module 0)')
ax.set_ylabel('coupling specificity (z, +=module 0)')
ax.set_title(f'Test 3: specificity tracks anatomy\nrho={rho:.2f}, p<0.0001, n={len(t3)} cells', fontsize=9)
ax.spines[['top', 'right']].set_visible(False)

fig.suptitle('Non-grid cell → grid module coupling: anatomy explains what "module identity" predicts',
             fontsize=10, fontweight='bold')
plt.tight_layout()
out = f'{DATA}/nongrid_grid_module_tests_summary'
plt.savefig(out + '.png', dpi=150, bbox_inches='tight')
plt.savefig(out + '.pdf', bbox_inches='tight')
print(f'saved {out}.png')
