"""Figure: spike-theta phase locking by anchoring state.

A  phase histogram for one cell, both states, spike counts equalised
B  session medians of MRL, paired (count- and speed-matched)
C  every cell, anchored MRL against non-anchored
D  speed dependence of locking (top minus bottom speed tercile), paired

The direct test of the entrainment reading: rate coding of speed is unchanged
between states, so if locking also turns out unchanged then "entrainment" is
the wrong word for what anchoring does to the LFP.
"""
import os, sys
import numpy as np, pandas as pd
import matplotlib; matplotlib.use('Agg')
import matplotlib.pyplot as plt
from scipy.stats import wilcoxon
sys.path.insert(0, '/Users/harryclark/Documents/spatial-manifolds/src')
from spatial_manifolds.anchoring import ANCH_COLOR, NONANCH_COLOR

s = open('/private/tmp/claude-501/-Users-harryclark-Documents-spatial-manifolds/'
         '35f3d8d1-3ead-4d48-8089-491c15b000b7/scratchpad/phase_locking.py').read()
exec(s[:s.index(chr(10) + 'if __name__')])

plt.rcParams['font.family'] = 'Arial'
FIG = '/Users/harryclark/Documents/spatial-manifolds/scripts/figures/AnchorDynamics2026'
A = pd.read_csv(f'{OUT}/phase_locking.csv').dropna(subset=['mrl_a', 'mrl_n'])

# example: a well-locked cell at the median change, selection not conditioned on
# the direction of that change
cand = A[(A.n_matched > 3000) & (A[['mrl_a', 'mrl_n']].max(1) > .08)].copy()
cand['d'] = cand.mrl_a - cand.mrl_n
ex = cand.iloc[(cand.d - (A.mrl_a - A.mrl_n).median()).abs().argsort()].iloc[0]
MO, DY, CL = int(ex.mouse), int(ex.day), int(ex.cluster)
print(f'example M{MO}D{DY} cluster {CL}: MRL {ex.mrl_a:.3f} anchored vs {ex.mrl_n:.3f}')

T = pd.read_csv(f'{OUT}/theta_frequency.csv')
extra = f'{OUT}/theta_frequency_extra.csv'
if os.path.exists(extra):
    T = pd.concat([T, pd.read_csv(extra)], ignore_index=True)
st = T[(T.mouse == MO) & (T.day == DY)]
pa, pn = one_cell(MO, DY, dict(zip(st.trial.astype(int), st.anch.astype(bool))), CL,
                  np.random.default_rng(0))

fig, axes = plt.subplots(2, 2, figsize=(7.4, 6.0))

ax = axes[0, 0]
# phase in this dataset runs 0..2pi, not -pi..pi; two cycles are drawn because a
# single cycle cut at an arbitrary point hides whether the peak is real
bins = np.linspace(0, 2 * np.pi, 25)
ctr = bins[:-1] + np.diff(bins) / 2
for v, c, lb in ((pa, ANCH_COLOR, 'anchored'), (pn, NONANCH_COLOR, 'non-anchored')):
    h, _ = np.histogram(np.mod(v, 2 * np.pi), bins=bins)
    h = h / h.sum()
    ax.plot(np.r_[ctr, ctr + 2 * np.pi], np.r_[h, h], color=c, lw=2, label=lb)
ax.legend(fontsize=7.5, frameon=False)
ax.set_title(f'M{MO} D{DY} cluster {CL} — MRL {ex.mrl_a:.3f} vs {ex.mrl_n:.3f}',
             fontsize=9.5, loc='left')
ax.set_xlabel('Theta phase (two cycles)', fontsize=9)
ax.set_ylabel('Spike density', fontsize=9)
ax.set_xlim(0, 4 * np.pi)
ax.set_xticks([0, np.pi, 2 * np.pi, 3 * np.pi, 4 * np.pi])
ax.set_xticklabels(['0', 'π', '2π', '3π', '4π'])

ax = axes[0, 1]
g = A.groupby(['mouse', 'day'])[['mrl_a', 'mrl_n']].median().dropna()
for i in range(len(g)):
    ax.plot([0, 1], [g.mrl_n.iloc[i], g.mrl_a.iloc[i]], color='0.75', lw=.6, zorder=1)
ax.scatter(np.zeros(len(g)), g.mrl_n, s=14, color=NONANCH_COLOR, lw=0, zorder=2)
ax.scatter(np.ones(len(g)), g.mrl_a, s=14, color=ANCH_COLOR, lw=0, zorder=2)
for x, v, c in ((0, g.mrl_n, NONANCH_COLOR), (1, g.mrl_a, ANCH_COLOR)):
    ax.plot([x - .18, x + .18], [v.median()] * 2, color=c, lw=2.4, zorder=3)
ax.set_xticks([0, 1]); ax.set_xticklabels(['non-anch', 'anchored'], fontsize=8)
ax.set_xlim(-.4, 1.4)
ax.set_ylabel('MRL (count- and speed-matched)', fontsize=9)
ax.set_title(f'{len(g)} sessions — p = {wilcoxon(g.mrl_a, g.mrl_n).pvalue:.2g}',
             fontsize=9.5, loc='left')

ax = axes[1, 0]
lim = np.nanpercentile(np.r_[A.mrl_a, A.mrl_n], [0, 99])
ax.scatter(A.mrl_n, A.mrl_a, s=6, color='0.45', alpha=.35, lw=0)
ax.plot(lim, lim, color='0.2', lw=.9, ls='--')
ax.set_xlim(*lim); ax.set_ylim(*lim)
ax.set_xlabel('MRL, non-anchored', fontsize=9)
ax.set_ylabel('MRL, anchored', fontsize=9)
ax.set_title(f'{len(A)} cells — {(A.mrl_a < A.mrl_n).mean()*100:.0f}% below identity',
             fontsize=9.5, loc='left')

ax = axes[1, 1]
D = A.dropna(subset=['mrl_lo_a', 'mrl_hi_a', 'mrl_lo_n', 'mrl_hi_n']).copy()
D['dep_a'], D['dep_n'] = D.mrl_hi_a - D.mrl_lo_a, D.mrl_hi_n - D.mrl_lo_n
gd = D.groupby(['mouse', 'day'])[['dep_a', 'dep_n']].median().dropna()
for i in range(len(gd)):
    ax.plot([0, 1], [gd.dep_n.iloc[i], gd.dep_a.iloc[i]], color='0.75', lw=.6, zorder=1)
ax.scatter(np.zeros(len(gd)), gd.dep_n, s=14, color=NONANCH_COLOR, lw=0, zorder=2)
ax.scatter(np.ones(len(gd)), gd.dep_a, s=14, color=ANCH_COLOR, lw=0, zorder=2)
for x, v, c in ((0, gd.dep_n, NONANCH_COLOR), (1, gd.dep_a, ANCH_COLOR)):
    ax.plot([x - .18, x + .18], [v.median()] * 2, color=c, lw=2.4, zorder=3)
ax.axhline(0, color='0.5', lw=.7, ls=':')
ax.set_xticks([0, 1]); ax.set_xticklabels(['non-anch', 'anchored'], fontsize=8)
ax.set_xlim(-.4, 1.4)
ax.set_ylabel('MRL, fast minus slow tercile', fontsize=9)
ax.set_title(f'{len(gd)} sessions — p = {wilcoxon(gd.dep_a, gd.dep_n).pvalue:.2g}',
             fontsize=9.5, loc='left')

for a in axes.ravel():
    a.spines[['top', 'right']].set_visible(False)
    a.tick_params(labelsize=7.5)
plt.tight_layout()
out = f'{FIG}/fig6_phase_locking_by_state.pdf'
plt.savefig(out, dpi=200, bbox_inches='tight')
print('saved', out)
