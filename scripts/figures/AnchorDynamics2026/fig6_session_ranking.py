"""Which session best represents the Figure 5 population trend?

Usage:  python3 fig6_session_ranking.py

The population makes four directional claims about the anchored state -- theta
amplitude lower, theta frequency lower, and both speed-theta correlations lower
-- plus one null claim, that running speed does not differ. An example panel
should carry all of them.

TWO DIFFERENT QUESTIONS, ANSWERED SEPARATELY. "Most representative" is
ambiguous and the two readings pick different sessions:

  TYPICAL   the session whose effects sit closest to the population median on
            every measure. This is what an honest example is: a reader who
            takes it as the average session is not misled.
  CLEAREST  the session with the largest effects in the correct direction. This
            makes the point visible, at the cost of being an outlier -- the
            reader sees a stronger effect than the data support.

Both are reported. They are not interchangeable and the choice should be
deliberate: a figure whose example is the clearest session, presented without
saying so, overstates the result.

HOW THE SCORES ARE BUILT. Each of the four effects is z-scored across sessions
and sign-oriented so that POSITIVE means "in the population's direction". Then

    clarity   = mean of the four oriented z-scores
    typicality = -mean |oriented z - median oriented z|, so 0 is the median
                 session and more negative is further from it

Speed is held out of both scores and reported alongside: since the population
claim about speed is a null, a good example is one where |speed difference| is
small, and folding a null into a directional score would be meaningless.

LEGIBILITY IS A SEPARATE GATE, not part of the score. A session that alternates
state every two or three trials can be perfectly representative in effect size
and unreadable as a raster. `blocky` is the median run length of the
median-filtered state sequence; below about 6 the raster reads as noise.
"""
import os
import sys
import warnings

import numpy as np
import pandas as pd

warnings.filterwarnings('ignore')
import matplotlib

matplotlib.use('Agg')
import matplotlib.pyplot as plt
from scipy.ndimage import median_filter
from scipy.stats import pearsonr

sys.path.insert(0, '/Users/harryclark/Documents/spatial-manifolds/src')
from spatial_manifolds.anchoring import (ANCH_COLOR, NONANCH_COLOR,
                                         load_session_labels)

plt.rcParams['font.family'] = 'Arial'
FIG = '/Users/harryclark/Documents/spatial-manifolds/scripts/figures/AnchorDynamics2026'
LFP = '/Users/harryclark/Documents/spatial-manifolds/data/lfp'
MINTR = 15
# the four directional effects; all have a NEGATIVE population median, so the
# oriented score is the negated value
EFFECTS = [('d_amp', 'theta\namplitude'), ('d_hz', 'theta\nfrequency'),
           ('d_r_amp', 'r(speed,\namplitude)'), ('d_r_hz', 'r(speed,\nfrequency)')]


def build():
    T = pd.read_csv(f'{LFP}/theta_frequency.csv')
    T['anch'] = T.anch.astype(bool)
    rows = []
    for (mo, dy), d in T.groupby(['mouse', 'day']):
        a, n = d[d.anch], d[~d.anch]
        if len(a) < MINTR or len(n) < MINTR:
            continue
        z = load_session_labels(int(mo), int(dy))
        if z is None:
            continue
        st = median_filter((np.asarray(z['frac_anch']) > .5).astype(float),
                           size=9, mode='nearest') > .5
        runs = np.diff(np.r_[0, np.where(np.diff(st.astype(int)) != 0)[0] + 1,
                             len(st)])
        rows.append(dict(
            mouse=int(mo), day=int(dy), ncell=int(np.isfinite(z['pc1_load']).sum()),
            ntr=len(d), blocky=float(np.median(runs)), nblock=len(runs),
            d_amp=100 * (a.amp.mean() - n.amp.mean()) / d.amp.mean(),
            d_hz=a.inst_hz.mean() - n.inst_hz.mean(),
            d_spd=a.speed.mean() - n.speed.mean(),
            d_r_amp=pearsonr(a.speed, a.amp)[0] - pearsonr(n.speed, n.amp)[0],
            d_r_hz=pearsonr(a.speed, a.inst_hz)[0] - pearsonr(n.speed, n.inst_hz)[0]))
    return pd.DataFrame(rows)


S = build()
for c, _ in EFFECTS:
    z = (S[c] - S[c].mean()) / S[c].std(ddof=1)
    S[f'z_{c}'] = -z                       # positive = population's direction
Z = S[[f'z_{c}' for c, _ in EFFECTS]]
S['n_correct'] = sum((S[c] < 0).astype(int) for c, _ in EFFECTS)
S['clarity'] = Z.mean(axis=1)
S['typicality'] = -(Z - Z.median()).abs().mean(axis=1)
S['legible'] = (S.blocky >= 6) & (S.ncell >= 60) & (S.ntr >= 100)
S['abs_spd'] = S.d_spd.abs()
S = S.sort_values('typicality', ascending=False).reset_index(drop=True)
S.to_csv(f'{LFP}/fig6_session_ranking.csv', index=False)

print(f'{len(S)} sessions with >= {MINTR} trials in both states\n')
print('population medians: ' + '  '.join(
    f'{n.replace(chr(10), " ")} {S[c].median():+.3f}' for c, n in EFFECTS)
    + f'   speed {S.d_spd.median():+.2f}')
cols = ['mouse', 'day', 'ncell', 'ntr', 'blocky', 'd_amp', 'd_hz', 'd_spd',
        'd_r_amp', 'd_r_hz', 'n_correct', 'clarity', 'typicality', 'legible']
print('\nRANKED BY TYPICALITY (closest to the median session on all four):')
print(S[cols].to_string(index=False, float_format=lambda v: f'{v:+.2f}'))
print('\nRANKED BY CLARITY (largest effects in the right direction):')
print(S.sort_values('clarity', ascending=False)[cols].head(8)
      .to_string(index=False, float_format=lambda v: f'{v:+.2f}'))

ok = S[(S.n_correct == 4) & S.legible]
print(f'\nall four effects AND legible: {len(ok)} session(s)')
if len(ok):
    for _, r in ok.iterrows():
        print(f'  M{r.mouse}D{r.day}: clarity {r.clarity:+.2f}, '
              f'typicality {r.typicality:+.2f}, |speed diff| {r.abs_spd:.2f} cm/s')

# ── figure ───────────────────────────────────────────────────────────────────
fig = plt.figure(figsize=(9.6, 7.2))
gs = fig.add_gridspec(1, 3, width_ratios=[1.5, .62, .62], wspace=.42)
lab = [f'M{int(r.mouse)} D{int(r.day)}' for _, r in S.iterrows()]

ax = fig.add_subplot(gs[0])
M = Z.loc[S.index].values
v = np.nanmax(np.abs(M))
im = ax.imshow(M, aspect='auto', cmap='RdBu_r', vmin=-v, vmax=v)
for i in range(M.shape[0]):
    for j in range(M.shape[1]):
        ax.text(j, i, f'{S[EFFECTS[j][0]].values[i]:+.2f}', ha='center',
                va='center', fontsize=6,
                color='white' if abs(M[i, j]) > .62 * v else '0.15')
ax.set_xticks(range(len(EFFECTS)))
ax.set_xticklabels([n for _, n in EFFECTS], fontsize=7.5)
ax.set_yticks(range(len(S))); ax.set_yticklabels(lab, fontsize=7)
for t, leg in zip(ax.get_yticklabels(), S.legible):
    t.set_color('0.15' if leg else '0.6')
# oriented z is the NEGATED effect, and RdBu_r sends high values to red, so
# red is the population's direction and blue is against it
ax.set_title('effect by session — RED matches the population direction, '
             'BLUE opposes it\n(cells show the raw effect; rows ordered by '
             'typicality)', fontsize=8, loc='left')
ax.tick_params(length=0)
cb = fig.colorbar(im, ax=ax, fraction=.04, pad=.02)
cb.set_label('z across sessions', fontsize=7); cb.ax.tick_params(labelsize=6.5)
cb.outline.set_visible(False)

for k, (col, nm, good) in enumerate((('clarity', 'clarity', 'higher = stronger'),
                                     ('typicality', 'typicality',
                                      'higher = more median'))):
    ax = fig.add_subplot(gs[k + 1])
    y = np.arange(len(S))
    c = [ANCH_COLOR if r.n_correct == 4 else '0.72' for _, r in S.iterrows()]
    ax.barh(y, S[col].values, color=c, height=.72,
            edgecolor=['0.2' if l else 'none' for l in S.legible], lw=.8)
    ax.axvline(0, color='0.4', lw=.7)
    ax.set_ylim(len(S) - .5, -.5); ax.set_yticks([])
    ax.set_title(f'{nm}\n{good}', fontsize=8, loc='left')
    ax.tick_params(labelsize=7)
    ax.spines[['top', 'right', 'left']].set_visible(False)

h = [plt.Line2D([], [], marker='s', ls='', color=ANCH_COLOR,
                label='all four effects correct'),
     plt.Line2D([], [], marker='s', ls='', color='0.72', label='one or more wrong'),
     plt.Line2D([], [], marker='s', ls='', mfc='none', mec='0.2',
                label='legible raster (outlined)')]
fig.legend(handles=h, loc='lower center', ncol=3, fontsize=7.5, frameon=False,
           bbox_to_anchor=(.5, -.015))
out = f'{FIG}/fig6_session_ranking.pdf'
plt.savefig(out, dpi=200, bbox_inches='tight')
print(f'\nsaved {out}')
print(f'saved {LFP}/fig6_session_ranking.csv')
