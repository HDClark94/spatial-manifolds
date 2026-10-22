"""Full 4-panel figure: are non-grid spatial cells 'grid-associated'?
Reuses data/nongrid_grid_association_results.csv (no recomputation)."""
import pandas as pd, numpy as np
import matplotlib; matplotlib.use('Agg')
import matplotlib.pyplot as plt

BLUE, RED, GREY = '#3171ae', '#c0392b', '#999999'
df = pd.read_csv('/Users/harryclark/Documents/spatial-manifolds/data/nongrid_grid_association_results.csv')
n_cells = len(df)
n_sessions = df[['mouse', 'day']].drop_duplicates().shape[0]
n_sig_fdr = (df.q_value < 0.05).sum()
n_sig_nom = (df.p_value < 0.05).sum()

fig, axes = plt.subplots(2, 2, figsize=(10, 8.5))

# ── Panel A: per-cell scatter, grid dpR2 vs non-grid dpR2 ────────────────────
ax = axes[0, 0]
lo = min(df.mean_grid_dpr2.min(), df.mean_nongrid_dpr2.min())
hi = max(df.mean_grid_dpr2.max(), df.mean_nongrid_dpr2.max())
pad = (hi - lo) * 0.05
lims = [lo - pad, hi + pad]
sig = df.p_value < 0.05
ax.scatter(df.mean_nongrid_dpr2[~sig], df.mean_grid_dpr2[~sig], s=6, alpha=0.25, color=BLUE, label='n.s.')
ax.scatter(df.mean_nongrid_dpr2[sig], df.mean_grid_dpr2[sig], s=10, alpha=0.7, color=RED,
           label=f'nominal p<0.05 (n={sig.sum()})')
ax.plot(lims, lims, 'k--', lw=1, alpha=0.6)
ax.set_xlim(lims); ax.set_ylim(lims)
ax.set_xlabel('mean ΔpR² to other non-grid cells')
ax.set_ylabel('mean ΔpR² to grid cells')
ax.set_title('A. Per-cell coupling: grid vs non-grid covariates\n(on the diagonal = equally coupled to both)', fontsize=9)
ax.legend(fontsize=7, frameon=False, loc='upper left')
ax.spines[['top', 'right']].set_visible(False)

# ── Panel B: distribution of D ───────────────────────────────────────────────
ax = axes[0, 1]
ax.hist(df.D, bins=60, color=BLUE, alpha=0.85)
ax.axvline(0, color='k', ls='--', lw=1)
ax.axvline(df.D.mean(), color=RED, lw=1.5, label=f'mean D={df.D.mean():.4f}')
ax.set_xlabel('D = mean ΔpR²(grid) − mean ΔpR²(non-grid)')
ax.set_ylabel('non-grid cells')
ax.set_title(f'B. Grid-association score\n{n_sig_fdr}/{n_cells} sig. at q<0.05 (BH-FDR)', fontsize=9)
ax.legend(fontsize=8, frameon=False)
ax.spines[['top', 'right']].set_visible(False)

# ── Panel C: per-session mean D ──────────────────────────────────────────────
ax = axes[1, 0]
sess = df.groupby(['mouse', 'day']).agg(meanD=('D', 'mean'), n=('D', 'size'),
                                         sem=('D', lambda x: x.std() / np.sqrt(len(x)))).reset_index()
sess = sess.sort_values('meanD')
y = np.arange(len(sess))
ax.errorbar(sess.meanD, y, xerr=sess['sem'], fmt='o', ms=3, color=BLUE, ecolor=GREY, elinewidth=0.8, capsize=0)
ax.axvline(0, color='k', ls='--', lw=1)
ax.set_yticks(y[::2])
ax.set_yticklabels([f'M{m}D{d}' for m, d in zip(sess.mouse, sess.day)][::2], fontsize=5)
ax.set_xlabel('session mean D (± SEM)')
ax.set_title(f'C. Per-session consistency (n={n_sessions} sessions)', fontsize=9)
ax.spines[['top', 'right']].set_visible(False)

# ── Panel D: p-value QQ plot ──────────────────────────────────────────────────
ax = axes[1, 1]
obs = np.sort(df.p_value.values)
n = len(obs)
expected = np.arange(1, n + 1) / n
ax.plot(expected, obs, '.', color=BLUE, ms=2, alpha=0.5)
ax.plot([0, 1], [0, 1], 'k--', lw=1)
ax.set_xlabel('expected p (uniform null)')
ax.set_ylabel('observed p (sorted)')
ax.set_title('D. P-value distribution vs uniform null\n(on diagonal = no true effect; above = deficit)', fontsize=9)
ax.spines[['top', 'right']].set_visible(False)

fig.suptitle(f'Non-grid spatial cells are not preferentially grid-associated\n'
             f'(n={n_cells} non-grid cells, {n_sessions} OF1 sessions; '
             f'{n_sig_fdr}/{n_cells} sig. at q<0.05, {n_sig_nom}/{n_cells} at nominal p<0.05)',
             fontsize=11, fontweight='bold')
plt.tight_layout(rect=[0, 0, 1, 0.94])
out = '/Users/harryclark/Documents/spatial-manifolds/data/nongrid_grid_association_full_figure'
plt.savefig(out + '.png', dpi=150, bbox_inches='tight')
plt.savefig(out + '.pdf', bbox_inches='tight')
print(f'saved {out}.png')
