"""Figure: MEC speed tuning and speed decoding by anchoring state.

A  one speed cell's tuning curve in each state (mean +/- SEM per speed bin)
B  session medians of gain (slope / mean rate) for speed cells, paired
C  every speed cell, anchored gain against non-anchored gain
D  population speed decoding, bins matched for count and speed distribution
E  how much UNEXPECTED speed there is in each state (SD of speed after
   position is partialled out) -- the thing that actually changes
F  sensitivity to unexpected speed, Hz per cm/s -- the thing that does not

Gain rather than raw slope carries panels B-C: a cell that simply fires less
when anchored has a smaller slope with its tuning shape untouched, and the
question here is whether speed sensitivity changes, not whether rate does.
"""
import sys, json
import numpy as np, pandas as pd
import matplotlib; matplotlib.use('Agg')
import matplotlib.pyplot as plt
from scipy.stats import wilcoxon
sys.path.insert(0, '/Users/harryclark/Documents/spatial-manifolds/src')
from spatial_manifolds.anchoring import ANCH_COLOR, NONANCH_COLOR

exec(open('/private/tmp/claude-501/-Users-harryclark-Documents-spatial-manifolds/'
          '35f3d8d1-3ead-4d48-8089-491c15b000b7/scratchpad/speed_tuning.py')
     .read().split('if __name__')[0])

plt.rcParams['font.family'] = 'Arial'
FIG = '/Users/harryclark/Documents/spatial-manifolds/scripts/figures/AnchorDynamics2026'
C = pd.read_csv(f'{OUT}/speed_tuning_cells.csv')
D = pd.read_csv(f'{OUT}/speed_tuning_decode.csv')
C['speed_cell'] = (C.p_pool < .05) & (C.r_pool > 0)
S = C[C.speed_cell].dropna(subset=['gain_a', 'gain_n'])

# Example cell: the one closest to the MEDIAN change in gain across all speed
# cells. Selecting a cell that flattens would misrepresent the result -- the
# population answer is that tuning does not flatten -- so the selection is not
# allowed to condition on the direction of the change.
cand = S[(S.rate > 2) & (S.n_a > 400) & (S.n_n > 400) & (S.r_pool > .25)].copy()
cand['d'] = cand.gain_a - cand.gain_n
ex = cand.iloc[(cand.d - S.eval('gain_a - gain_n').median()).abs().argsort()].iloc[0]
MO, DY, CL = int(ex.mouse), int(ex.day), int(ex.cluster)
print(f'example cell M{MO}D{DY} cluster {CL}: '
      f'gain {ex.gain_a:.4f} anchored vs {ex.gain_n:.4f}, r_pool {ex.r_pool:.2f}')

T = pd.read_csv(f'{OUT}/theta_frequency.csv')
st = T[(T.mouse == MO) & (T.day == DY)]
R, speed, anch, ids = binned(MO, DY, dict(zip(st.trial.astype(int), st.anch.astype(bool))))
rate = R[:, list(ids).index(CL)]

fig, axes = plt.subplots(2, 3, figsize=(9.8, 6.0))
ax = axes[0, 0]
edges = np.arange(np.floor(np.quantile(speed, .02)), np.quantile(speed, .98), 5.)
ctr = edges[:-1] + 2.5
for m, c, lb in ((anch, ANCH_COLOR, 'anchored'), (~anch, NONANCH_COLOR, 'non-anchored')):
    mu = np.full(len(ctr), np.nan); se = np.full(len(ctr), np.nan)
    for i in range(len(ctr)):
        v = rate[m & (speed >= edges[i]) & (speed < edges[i + 1])]
        if len(v) >= 10:
            mu[i], se[i] = v.mean(), v.std() / np.sqrt(len(v))
    ax.fill_between(ctr, mu - se, mu + se, color=c, alpha=.25, lw=0, edgecolor='none')
    ax.plot(ctr, mu, color=c, lw=2, label=lb)
ax.set_xlabel('Running speed (cm/s)', fontsize=9)
ax.set_ylabel('Firing rate (Hz)', fontsize=9)
ax.set_title(f'M{MO} D{DY} cluster {CL}', fontsize=9.5, loc='left')
ax.legend(fontsize=7.5, frameon=False)

ax = axes[0, 1]
G = S.groupby(['mouse', 'day'])[['gain_a', 'gain_n']].median().dropna()
for i in range(len(G)):
    ax.plot([0, 1], [G.gain_n.iloc[i], G.gain_a.iloc[i]], color='0.75', lw=.6, zorder=1)
ax.scatter(np.zeros(len(G)), G.gain_n, s=14, color=NONANCH_COLOR, lw=0, zorder=2)
ax.scatter(np.ones(len(G)), G.gain_a, s=14, color=ANCH_COLOR, lw=0, zorder=2)
for x, v, c in ((0, G.gain_n, NONANCH_COLOR), (1, G.gain_a, ANCH_COLOR)):
    ax.plot([x - .18, x + .18], [v.median()] * 2, color=c, lw=2.4, zorder=3)
ax.axhline(0, color='0.5', lw=.7, ls=':')
ax.set_xticks([0, 1]); ax.set_xticklabels(['non-anch', 'anchored'], fontsize=8)
ax.set_xlim(-.4, 1.4)
ax.set_ylabel('gain (slope / mean rate)', fontsize=9)
ax.set_title(f'{len(G)} sessions — p = {wilcoxon(G.gain_a, G.gain_n).pvalue:.2g}',
             fontsize=9.5, loc='left')

ax = axes[0, 2]
lim = np.nanpercentile(np.r_[S.gain_a, S.gain_n], [1, 99])
ax.scatter(S.gain_n, S.gain_a, s=6, color='0.45', alpha=.35, lw=0)
ax.plot(lim, lim, color='0.2', lw=.9, ls='--')
ax.set_xlim(*lim); ax.set_ylim(*lim)
ax.set_xlabel('gain, non-anchored', fontsize=9)
ax.set_ylabel('gain, anchored', fontsize=9)
below = (S.gain_a < S.gain_n).mean() * 100
ax.set_title(f'{len(S)} speed cells — {below:.0f}% below identity', fontsize=9.5, loc='left')

ax = axes[1, 0]
d = D.dropna(subset=['dec_r_a', 'dec_r_n'])
for i in range(len(d)):
    ax.plot([0, 1], [d.dec_r_n.iloc[i], d.dec_r_a.iloc[i]], color='0.75', lw=.6, zorder=1)
ax.scatter(np.zeros(len(d)), d.dec_r_n, s=14, color=NONANCH_COLOR, lw=0, zorder=2)
ax.scatter(np.ones(len(d)), d.dec_r_a, s=14, color=ANCH_COLOR, lw=0, zorder=2)
for x, v, c in ((0, d.dec_r_n, NONANCH_COLOR), (1, d.dec_r_a, ANCH_COLOR)):
    ax.plot([x - .18, x + .18], [v.median()] * 2, color=c, lw=2.4, zorder=3)
ax.set_xticks([0, 1]); ax.set_xticklabels(['non-anch', 'anchored'], fontsize=8)
ax.set_xlim(-.4, 1.4)
ax.set_ylabel('speed decoding r (pred, true)', fontsize=9)
ax.set_title(f'{len(d)} sessions — p = {wilcoxon(d.dec_r_a, d.dec_r_n).pvalue:.2g}',
             fontsize=9.5, loc='left')

# E, F: the residual-speed analysis -- position partialled out of both sides
V = pd.read_csv(f'{OUT}/residual_speed.csv')
W = V.pivot_table(index=['mouse', 'day', 'cluster'], columns='state',
                  values=['slope_cms', 'sres_sd']).dropna()


def pairpanel(ax, va, vn, ylab, title):
    for i in range(len(va)):
        ax.plot([0, 1], [vn[i], va[i]], color='0.75', lw=.6, zorder=1)
    ax.scatter(np.zeros(len(va)), vn, s=14, color=NONANCH_COLOR, lw=0, zorder=2)
    ax.scatter(np.ones(len(va)), va, s=14, color=ANCH_COLOR, lw=0, zorder=2)
    for x, v, c in ((0, vn, NONANCH_COLOR), (1, va, ANCH_COLOR)):
        ax.plot([x - .18, x + .18], [np.median(v)] * 2, color=c, lw=2.4, zorder=3)
    ax.set_xticks([0, 1]); ax.set_xticklabels(['non-anch', 'anchored'], fontsize=8)
    ax.set_xlim(-.4, 1.4)
    ax.set_ylabel(ylab, fontsize=9)
    ax.set_title(f'{title} — p = {wilcoxon(va, vn).pvalue:.2g}', fontsize=9.5, loc='left')


sd = W['sres_sd'].groupby(level=['mouse', 'day']).median().dropna()
pairpanel(axes[1, 1], sd['a'].values, sd['n'].values,
          'SD of unexpected speed (cm/s)', f'{len(sd)} sessions')
sl = W['slope_cms'].groupby(level=['mouse', 'day']).median().dropna()
pairpanel(axes[1, 2], sl['a'].values, sl['n'].values,
          'Hz per cm/s unexpected speed', f'{len(sl)} sessions')

for a in axes.ravel():
    a.spines[['top', 'right']].set_visible(False)
    a.tick_params(labelsize=7.5)
plt.tight_layout()
out = f'{FIG}/fig6_speed_tuning_by_state.pdf'
plt.savefig(out, dpi=200, bbox_inches='tight')
print('saved', out)
