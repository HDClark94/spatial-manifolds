"""Speed-theta coupling across a session, not pooled over it.

The paired panels compare one correlation per state per session, which cannot
show whether the coupling actually sags *while* the population is anchored or
merely differs on average between two pools of trials. Here the correlation is
computed in a sliding window of trials and plotted on the trial axis alongside
the anchored fraction, so the two can be read against each other directly.

Window r is autocorrelated by construction (neighbouring windows share trials),
so the window-r vs window-anchoring correlation cannot be tested with a
parametric p. Significance comes from a circular shift of the state vector
within session: the shift preserves the run structure of the states and the
entire theta/speed time series, and only breaks their alignment.

Both example sessions are shown. M21 D22 carries the scatter panel of
fig6_speed_theta_coupling; M25 D23 is the trial-axis example and runs the other
way, so plotting it is the honest test of whether this readout tracks a session's
direction rather than manufacturing the group effect in every session.

The bottom row is the across-session version of the same readout, and it is what
the claim rests on. A single session does not resolve the effect: a 25-trial
window correlation has a standard error near 0.2, and the states come in long
blocks (median 19 per session), so only 2 of 55 sessions beat their own shift
null. Pooling windows by state across sessions does resolve it, and does so for
window widths of 15, 25 and 40 trials (p = 1e-3, 5e-4, 4e-3).
"""
import sys
import numpy as np, pandas as pd
import matplotlib; matplotlib.use('Agg')
import matplotlib.pyplot as plt
from scipy.stats import spearmanr
sys.path.insert(0, '/Users/harryclark/Documents/spatial-manifolds/src')
from spatial_manifolds.anchoring import ANCH_COLOR, NONANCH_COLOR

plt.rcParams['font.family'] = 'Arial'
FIG = '/Users/harryclark/Documents/spatial-manifolds/scripts/figures/AnchorDynamics2026'
WIN = 25            # trials per window; ~1/8 of a session, enough for a stable r
NSHIFT = 2000
SESSIONS = [(21, 22), (25, 23)]
SUMMARY = ('/Users/harryclark/Documents/spatial-manifolds/data/lfp/'
           'speed_theta_timecourse.csv')      # written by stt_all.py

T = pd.read_csv('/Users/harryclark/Documents/spatial-manifolds/data/lfp/theta_frequency.csv')
T['anch'] = T.anch.astype(bool)
rng = np.random.default_rng(0)


def sliding(sp, am, an, win=WIN):
    """Window-centre trial, r, slope and anchored fraction for each window."""
    n = len(sp)
    c, r, b, f = [], [], [], []
    for i in range(n - win + 1):
        s, a, k = sp[i:i + win], am[i:i + win], an[i:i + win]
        m = np.isfinite(s) & np.isfinite(a)
        if m.sum() < win * .6 or np.std(s[m]) == 0:
            continue
        c.append(i + win // 2)
        r.append(np.corrcoef(s[m], a[m])[0, 1])
        b.append(np.polyfit(s[m], a[m], 1)[0])
        f.append(np.nanmean(k))
    return map(np.asarray, (c, r, b, f))


fig = plt.figure(figsize=(7.4, 7.2))
gs = fig.add_gridspec(3, 2, height_ratios=[1, 1, 1.15], hspace=.55, wspace=.3)
axes = np.array([[fig.add_subplot(gs[0, :])], [fig.add_subplot(gs[1, :])]])
for row, (MO, DY) in enumerate(SESSIONS):
    d = T[(T.mouse == MO) & (T.day == DY)].sort_values('trial')
    tr = d.trial.values
    full = np.arange(tr.min(), tr.max() + 1)
    sp = np.full(len(full), np.nan); am = np.full(len(full), np.nan)
    an = np.full(len(full), np.nan)
    idx = tr - tr.min()
    sp[idx], am[idx], an[idx] = d.speed.values, d.amp.values, d.anch.values.astype(float)

    c, r, b, f = sliding(sp, am, an)
    rho, _ = spearmanr(r, f)

    # circular shift of the state vector only: theta and speed stay put
    null = np.empty(NSHIFT)
    for j, k in enumerate(rng.integers(1, len(an), NSHIFT)):
        _, _, _, fs = sliding(sp, am, np.roll(an, k))
        null[j] = spearmanr(r, fs)[0]
    p = (np.sum(np.abs(null) >= abs(rho)) + 1) / (NSHIFT + 1)

    ax = axes[row, 0]
    st = np.nan_to_num(an) > .5
    ax.fill_between(full, -1, 1, where=st, color=ANCH_COLOR, alpha=.16,
                    linewidth=0, edgecolor='none', step='mid')
    ax.axhline(0, color='0.5', lw=.7, ls=':')
    ax.plot(full[c], r, color='0.15', lw=1.6, zorder=3)
    ax.set_ylim(-1, 1)
    ax.set_ylabel(f'r (speed, theta amp)\n{WIN}-trial window', fontsize=8.5)
    ax.set_xlim(full[0], full[-1])
    ax.set_title(f'M{MO} D{DY} — window r vs anchored fraction: '
                 f'$\\rho$ = {rho:+.2f}, shift-null p = {p:.3g}',
                 fontsize=9.5, loc='left')
    ax.spines[['top', 'right']].set_visible(False)
    ax.tick_params(labelsize=7.5)
    print(f'M{MO}D{DY}: {len(full)} trials, {len(r)} windows, '
          f'rho = {rho:+.3f}, p = {p:.4g}, '
          f'r anchored {np.mean(np.array(r)[f > .5]):+.3f} '
          f'vs non {np.mean(np.array(r)[f < .5]):+.3f}')

axes[-1, 0].set_xlabel('Trial (window centre)', fontsize=9)
h = plt.Rectangle((0, 0), 1, 1, color=ANCH_COLOR, alpha=.16, lw=0)
axes[0, 0].legend([h], ['anchored'], fontsize=7.5, frameon=False, loc='lower right')

# ---- across sessions: the level at which this readout does resolve
S = pd.read_csv(SUMMARY)
from scipy.stats import wilcoxon

ax = fig.add_subplot(gs[2, 0])
ax.hist(S.rho, bins=np.arange(-1, 1.01, .1), color='0.75', edgecolor='white', lw=.6)
ax.axvline(0, color='0.4', lw=.8, ls=':')
ax.axvline(S.rho.median(), color='#b5651d', lw=2)
ax.set_xlabel(r'per-session $\rho$ (window r, anchored fraction)', fontsize=8.5)
ax.set_ylabel('Sessions', fontsize=9)
ax.set_title(f'median {S.rho.median():+.2f}, {(S.rho < 0).sum()}/{len(S)} negative\n'
             f'{(S.p < .05).sum()}/{len(S)} beat their own shift null',
             fontsize=8.5, loc='left')

P = S.dropna(subset=['r_anch', 'r_non'])
ax = fig.add_subplot(gs[2, 1])
for i in range(len(P)):
    ax.plot([0, 1], [P.r_non.iloc[i], P.r_anch.iloc[i]], color='0.75', lw=.6, zorder=1)
ax.scatter(np.zeros(len(P)), P.r_non, s=14, color=NONANCH_COLOR, lw=0, zorder=2)
ax.scatter(np.ones(len(P)), P.r_anch, s=14, color=ANCH_COLOR, lw=0, zorder=2)
for x, v, col in ((0, P.r_non, NONANCH_COLOR), (1, P.r_anch, ANCH_COLOR)):
    ax.plot([x - .18, x + .18], [v.median()] * 2, color=col, lw=2.4, zorder=3)
ax.axhline(0, color='0.5', lw=.7, ls=':')
ax.set_xticks([0, 1]); ax.set_xticklabels(['non-anch\nwindows', 'anchored\nwindows'],
                                          fontsize=8)
ax.set_xlim(-.4, 1.4)
ax.set_ylabel('window r (speed, theta amp)', fontsize=9)
ax.set_title(f'{len(P)} sessions — p = {wilcoxon(P.r_anch, P.r_non).pvalue:.2g}',
             fontsize=9, loc='left')
for a in fig.axes:
    a.spines[['top', 'right']].set_visible(False)
    a.tick_params(labelsize=7.5)
out = f'{FIG}/fig6_speed_theta_timecourse.pdf'
plt.savefig(out, dpi=200, bbox_inches='tight')
print('saved', out)
