"""Figure 2 supplement — does firing rate change with the anchoring state?

Usage:  python3 fig2_supp_rate.py

ONE QUESTION: is anchoring accompanied by a change in how much cells fire, or
only in where they fire?

THE ANSWER IS NO, and the way that became clear is the point of the supplement.
The effect is small and ITS SIGN DEPENDS ON THE MEASUREMENT. On the same 6,517
cells:

    spatially averaged, unmatched      log2 ratio  -0.090   p = 5e-64
    time averaged,      unmatched      log2 ratio  -0.028   p = 0.004
    time averaged,      speed-matched  log2 ratio  +0.036   p = 1e-14

Every one of those is "significant". They disagree about direction. Nothing that
flips sign under a change of denominator and a behavioural control should be
reported as a rate effect, and the honest conclusion is that anchoring is about
WHERE cells fire, not how much.

WHY THE MEASURES DISAGREE. A rate read off a rate map is averaged over POSITION
BINS, so it divides by occupancy -- and occupancy is set by running speed. A
rate computed as spikes divided by running seconds does not. The two correlate
at r = 0.96 across cells and differ systematically in mean (6.84 vs 9.03 Hz),
which is enough for a 3% state difference to change sign between them. The
time-averaged version is the firing rate; the spatially averaged one is a
property of the map.

AND WHY SPEED MATCHING STILL MATTERS. Pooled mean speed barely differs between
the states (31.16 vs 31.14 cm/s), which looks like it makes matching
unnecessary. It does not: matching within session, in deciles of trial mean
speed with equal counts per decile, still moves the estimate by 0.064 in log2 --
larger than the effect itself. The pooled means hide within-session structure.

ALSO CONTROLLED: the circularity. A cell must not be split by its OWN label,
because a trial with more spikes gives a better-estimated map, which is more
likely to match the others, which earns the anchored label. Every split here
uses the POPULATION state, defined by the other cells.

Rates use RUNNING samples only, and a trial's duration is its running duration,
so a trial where the animal stopped a lot is not scored as low rate for that.
"""
import glob
import os
import sys
import warnings

import numpy as np
import pandas as pd

warnings.filterwarnings('ignore')
import matplotlib

matplotlib.use('Agg')
import matplotlib.pyplot as plt
from scipy.stats import wilcoxon

sys.path.insert(0, '/Users/harryclark/Documents/spatial-manifolds/src')
from spatial_manifolds.anchoring import ANCH_COLOR, NONANCH_COLOR

plt.rcParams['font.family'] = 'Arial'
# mathtext must resolve to Arial, not the DejaVu default, or rho/times and any
# bold panel letters render in a different face from the rest of the figure
plt.rcParams['mathtext.fontset'] = 'custom'
plt.rcParams['mathtext.rm'] = 'Arial'
plt.rcParams['mathtext.it'] = 'Arial:italic'
plt.rcParams['mathtext.bf'] = 'Arial:bold'
# the aligned numeric block is set in monospace; without naming one that exists
# on this machine matplotlib falls back to DejaVuSansMono and embeds it
plt.rcParams['font.monospace'] = ['Menlo', 'Courier New', 'Courier']


def _lp(ax, s, dx=-.16, dy=1.0):
    """Bold panel letter, drawn separately from the title and always 10 pt,
    matching every other figure (the titles here previously carried a plain,
    non-bold leading letter sized to that panel's own title fontsize)."""
    ax.text(dx, dy, s, transform=ax.transAxes, fontsize=10, weight='bold',
            va='bottom', ha='right')
ROOT = '/Users/harryclark/Documents/spatial-manifolds'
FIG = f'{ROOT}/scripts/figures/AnchorDynamics2026'
PS = f'{ROOT}/data/population_state'
ORDER = ['grid', 'non-grid spatial', 'non-spatial', 'putative interneuron']
SHORT = ['grid', 'NGS', 'NS', 'int']
ICOL = {'grid': '#c04744', 'non-grid spatial': '#3171ae',
        'non-spatial': '#888888', 'putative interneuron': '#d95f02'}

D = pd.concat([pd.read_csv(f) for f in glob.glob(f'{PS}/trial_map_stats_w*.csv')],
              ignore_index=True)
R = pd.concat([pd.read_csv(f) for f in glob.glob(f'{PS}/trial_rate_speed_w*.csv')],
              ignore_index=True)
U = pd.read_csv(f'{PS}/unit_table.csv')
M = (D.merge(R, on=['mouse', 'day', 'cluster_id'])
     .merge(U[['mouse', 'day', 'cluster_id', 'identity', 'in_population']],
            on=['mouse', 'day', 'cluster_id']).query('in_population'))
M = M[M.identity.isin(ORDER)].reset_index(drop=True)
lg = lambda a, b: np.log2((a + .1) / (b + .1))
MEAS = [('rate_pop_anch', 'rate_pop_non', 'spatially av.\n(unmatched)'),
        ('rate_a', 'rate_n', 'time av.\n(unmatched)'),
        ('rate_a_sm', 'rate_n_sm', 'time av.\nSPEED-MATCHED')]

print(f'{len(M)} cells, {M.groupby(["mouse","day"]).ngroups} sessions, '
      f'{M.mouse.nunique()} mice')
print(f'  speed unmatched {M.spd_a.mean():.2f} vs {M.spd_n.mean():.2f} cm/s; '
      f'matched {M.spd_a_sm.mean():.2f} vs {M.spd_n_sm.mean():.2f}')
STAT = {}
for a, b, nm in MEAS:
    d = M.dropna(subset=[a, b])
    STAT[nm] = (lg(d[a], d[b]).mean(), wilcoxon(d[a], d[b]).pvalue, len(d))
    print(f'  {nm.replace(chr(10), " "):34s} log2 {STAT[nm][0]:+.4f}  '
          f'p={STAT[nm][1]:.1g}  n={STAT[nm][2]}')

fig = plt.figure(figsize=(9.8, 5.8))
gs = fig.add_gridspec(2, 4, hspace=.62, wspace=.5)
rng = np.random.default_rng(0)

ax = fig.add_subplot(gs[0, 0])
for i, (a, b, nm) in enumerate(MEAS):
    d = M.dropna(subset=[a, b]); v = lg(d[a], d[b])
    c = ANCH_COLOR if 'MATCH' in nm else '0.55'
    ax.scatter(i + rng.uniform(-.15, .15, len(v)), v, s=1.5, color=c, alpha=.2, lw=0)
    ax.errorbar(i, v.mean(), yerr=v.std(ddof=1) / np.sqrt(len(v)), color='k',
                marker='_', ms=15, lw=1.6, capsize=3, zorder=4)
    ax.annotate(f'{v.mean():+.3f}', (i, .03), xycoords=('data', 'axes fraction'),
                ha='center', fontsize=6.2, color='0.25')
ax.axhline(0, color='k', lw=1, ls='--')
ax.set_ylim(-.8, .8)
ax.set_xticks(range(3)); ax.set_xticklabels([m[2] for m in MEAS], fontsize=5.6)
ax.set_ylabel('log2 (anchored / non-anchored)', fontsize=8)
_lp(ax, 'A')
ax.set_title('the sign depends on the measure', fontsize=7.5, loc='left')
ax.tick_params(labelsize=7); ax.spines[['top', 'right']].set_visible(False)

ax = fig.add_subplot(gs[0, 1])
c = M.dropna(subset=['rate_pop_anch', 'rate_a'])
ax.scatter(c.rate_pop_anch, c.rate_a, s=2, color='0.5', alpha=.25, lw=0)
lim = [0, 60]
ax.plot(lim, lim, color=ANCH_COLOR, lw=1, ls=':')
ax.set_xlim(lim); ax.set_ylim(lim)
ax.set_xlabel('spatially averaged (Hz)', fontsize=7.5)
ax.set_ylabel('time averaged (Hz)', fontsize=8)
_lp(ax, 'B')
ax.set_title(f'r = {np.corrcoef(c.rate_pop_anch, c.rate_a)[0,1]:.2f} but offset'
             f'\n(occupancy vs seconds)', fontsize=7.5, loc='left')
ax.tick_params(labelsize=7); ax.spines[['top', 'right']].set_visible(False)

ax = fig.add_subplot(gs[0, 2])
for i, (a, b) in enumerate((('spd_a', 'spd_n'), ('spd_a_sm', 'spd_n_sm'))):
    d = M.dropna(subset=[a, b]); v = d[a] - d[b]
    ax.scatter(i + rng.uniform(-.15, .15, len(v)), v, s=1.5, color='0.55',
               alpha=.2, lw=0)
    ax.errorbar(i, v.mean(), yerr=v.std(ddof=1) / np.sqrt(len(v)), color='k',
                marker='_', ms=15, lw=1.6, capsize=3, zorder=4)
ax.axhline(0, color='k', lw=1, ls='--')
ax.set_xticks([0, 1]); ax.set_xticklabels(['unmatched', 'matched'], fontsize=7)
ax.set_xlim(-.5, 1.5)
ax.set_ylabel('speed difference (cm/s)', fontsize=8)
_lp(ax, 'C')
ax.set_title('speed barely differs either way', fontsize=7.5, loc='left')
ax.tick_params(labelsize=7); ax.spines[['top', 'right']].set_visible(False)

ax = fig.add_subplot(gs[0, 3])
own = M.dropna(subset=['rate_anch', 'rate_non'])
for i, (v, c_) in enumerate(((lg(own.rate_anch, own.rate_non), '0.6'),
                             (lg(M.rate_pop_anch, M.rate_pop_non), ANCH_COLOR))):
    ax.scatter(i + rng.uniform(-.15, .15, len(v)), v, s=1.5, color=c_, alpha=.2, lw=0)
    ax.errorbar(i, v.mean(), yerr=v.std(ddof=1) / np.sqrt(len(v)), color='k',
                marker='_', ms=15, lw=1.6, capsize=3, zorder=4)
    ax.annotate(f'{v.mean():+.3f}', (i, .03), xycoords=('data', 'axes fraction'),
                ha='center', fontsize=6.2, color='0.25')
ax.axhline(0, color='k', lw=1, ls='--')
ax.set_ylim(-.8, .8)
ax.set_xticks([0, 1])
ax.set_xticklabels(['own label\n(circular)', 'population\nstate'], fontsize=6.5)
ax.set_xlim(-.5, 1.5)
ax.set_ylabel('log2 ratio', fontsize=8)
_lp(ax, 'D')
ax.set_title('and on the circularity control', fontsize=7.5, loc='left')
ax.tick_params(labelsize=7); ax.spines[['top', 'right']].set_visible(False)

for j, (a, b, nm) in enumerate(((('rate_a', 'rate_n', 'unmatched')),
                                (('rate_a_sm', 'rate_n_sm', 'speed-matched')))):
    ax = fig.add_subplot(gs[1, j])
    d = M.dropna(subset=[a, b])
    for i, k in enumerate(ORDER):
        q = d[d.identity == k]; v = lg(q[a], q[b])
        ax.scatter(i + rng.uniform(-.16, .16, len(v)), v, s=1.5, color=ICOL[k],
                   alpha=.2, lw=0)
        ax.errorbar(i, v.mean(), yerr=v.std(ddof=1) / np.sqrt(len(v)), color='k',
                    marker='_', ms=13, lw=1.5, capsize=3, zorder=4)
    ax.axhline(0, color='k', lw=1, ls='--')
    ax.set_ylim(-.6, .6)
    ax.set_xticks(range(4)); ax.set_xticklabels(SHORT, fontsize=7)
    ax.set_ylabel('log2 ratio', fontsize=8)
    _lp(ax, "EF"[j])
    ax.set_title(f'{nm}, by identity', fontsize=7.5, loc='left')
    ax.tick_params(labelsize=7); ax.spines[['top', 'right']].set_visible(False)

ax = fig.add_subplot(gs[1, 2])
for j, (a, b) in enumerate((('rate_a', 'rate_n'), ('rate_a_sm', 'rate_n_sm'))):
    d = M.dropna(subset=[a, b])
    pm = d.groupby('mouse').apply(lambda q: lg(q[a], q[b]).mean())
    ax.scatter(np.full(len(pm), j) + rng.uniform(-.12, .12, len(pm)), pm, s=18,
               color=ANCH_COLOR if j else '0.55', lw=0, zorder=3)
    ax.plot([j - .2, j + .2], [pm.mean()] * 2, color='k', lw=2, zorder=4)
ax.axhline(0, color='k', lw=1, ls='--')
ax.set_xticks([0, 1]); ax.set_xticklabels(['unmatched', 'matched'], fontsize=7)
ax.set_xlim(-.5, 1.5)
ax.set_ylabel('log2 ratio, per mouse', fontsize=8)
_lp(ax, 'G')
ax.set_title(f'per mouse (n={M.mouse.nunique()})', fontsize=7.5, loc='left')
ax.tick_params(labelsize=7); ax.spines[['top', 'right']].set_visible(False)

ax = fig.add_subplot(gs[1, 3])
ax.axis('off')
ax.text(0, 1.0, 'The effect is not robust', fontsize=8.5, fontweight='bold', va='top')
txt = '\n'.join(f'{nm.replace(chr(10), " "):26s}{v:+.3f}'
                for nm, (v, _, _) in STAT.items())
ax.text(0, .84, txt, fontsize=6.2, va='top', family='monospace', color='0.2',
        linespacing=1.6)
ax.text(0, .46,
        'Three measures of the same quantity,\n'
        'same cells, all "significant", two signs.\n'
        'A ~3% difference that reverses under a\n'
        'change of denominator and a behavioural\n'
        'control is not a rate effect.\n\n'
        'Anchoring is about WHERE cells fire,\n'
        'not how much.', fontsize=6.8, va='top', color='0.2', linespacing=1.5)

out = f'{FIG}/fig2_supp_rate.pdf'
plt.savefig(out, dpi=200, bbox_inches='tight')
print(f'saved {out}')
