"""Figure S17: MEC speed coding by anchoring state, one model per cell.

Rebuilt 2026-11-01 on speed_state_glm.py (the previous version, on per-state
OLS slopes of VR-defined speed cells with panels on different session sets, is
superseded). Cells are grouped entirely outside the task, as in the Results:
open-field speed modulation crossed with the waveform interneuron call.

A  one open-field speed interneuron, rate against speed in each state, on the
   two states' common speed support; the cell nearest its group's median change
B  speed gain (log-rate per 10 cm/s, as % rate change) in each state, position
   held: per session, inverse-variance weighted over the group's cells
C  the change when anchored, % of the non-anchored gain, 95% interval from a
   hierarchical bootstrap (mice, then sessions); filled, position held (the
   model); open, position left out; dashed, the fall in theta's speed slope
D  decoding of UNEXPECTED speed (position removed from speed and rates) from
   the population, bins matched between states
E  how much unexpected speed there is in each state

Writes fig7_supp_speed_tuning.pdf
"""
import os, sys, warnings
import numpy as np, pandas as pd
warnings.filterwarnings('ignore')
import matplotlib; matplotlib.use('Agg')
import matplotlib.pyplot as plt
import statsmodels.formula.api as smf
sys.path.insert(0, '/Users/harryclark/Documents/spatial-manifolds/src')
from spatial_manifolds.anchoring import ANCH_COLOR, NONANCH_COLOR

_HERE = os.path.dirname(os.path.abspath(__file__))
# the GLM helpers first: they exec a notebook whose names must not overwrite
# the summary's
_s = open(os.path.join(_HERE, 'speed_state_glm.py')).read()
exec(_s[:_s.rindex('if __name__')])                    # binned_full, common_support, OUT
_s = open(os.path.join(_HERE, 'speed_state_summary.py')).read()
exec(_s[:_s.rindex('if __name__')])                    # load, per_session, hboot, GROUP_ORDER

plt.rcParams['font.family'] = 'Arial'
plt.rcParams['pdf.fonttype'] = 42
FIG = _HERE
THETA_FALL = -42.3      # theta amplitude's speed slope, 0.343 -> 0.198 (Figure 8 text)
GCOL = {'speed interneurons': '#d95f02', 'speed principal': '#555555',
        'non-speed interneurons': '#f2b48a', 'non-speed principal': '#b0b0b0'}
GLAB = {'speed interneurons': 'speed\ninterneurons', 'speed principal': 'speed\nprincipal',
        'non-speed interneurons': 'non-speed\ninterneurons', 'non-speed principal': 'non-speed\nprincipal'}

D = load()
SS = pd.read_csv(f'{OUT}/speed_state_glm_sessions.csv').dropna(subset=['dec_r_a', 'dec_r_n'])
rng = np.random.default_rng(0)
RES, RES0, PS_ = {}, {}, {}
for g in GROUP_ORDER:
    for cols, store in (({}, RES), (dict(col_n='b_n_nopos', col_i='b_int_nopos',
                                         col_se='se_int_nopos'), RES0)):
        S = per_session(D[D.group == g], **cols)
        S = S[S.n >= 2]
        lo, hi = hboot(S, rng)
        p = smf.mixedlm('b_i ~ 1', S, groups=S['mouse']).fit(reml=True).pvalues['Intercept']
        store[g] = dict(pct=100 * S.b_i.mean() / S.b_n.mean(), lo=lo, hi=hi, p=p, S=S)
    PS_[g] = RES[g]['S']

fig = plt.figure(figsize=(10.2, 6.6))
gs = fig.add_gridspec(2, 3, hspace=.55, wspace=.42, width_ratios=[1, 1.25, 1])


def lp(ax, s):
    ax.text(-.2, 1.04, s, transform=ax.transAxes, fontsize=10, weight='bold', va='bottom')


def tidy(ax):
    ax.spines[['top', 'right']].set_visible(False); ax.tick_params(labelsize=7)


def star(p):
    return f'p = {p:.1e}'.replace('e-0', 'e-') if p < 1e-3 else f'p = {p:.2g}'


# A: an example speed interneuron, the one nearest its group's median change
ax = fig.add_subplot(gs[0, 0]); lp(ax, 'A')
Q = D[(D.group == 'speed interneurons') & (D.rate_a > 5) & (D.n_a > 400) & (D.n_n > 400)].copy()
Q['chg'] = Q.b_int / Q.b_n
med = (RES['speed interneurons']['pct']) / 100
ex = Q.iloc[(Q.chg - med).abs().argsort()].iloc[0]
MO, DY, CL = int(ex.mouse), int(ex.day), int(ex.cluster)
T = pd.read_csv(f'{OUT}/theta_frequency.csv')
st = T[(T.mouse == MO) & (T.day == DY)]
Y, spd, pos, anch, ids = binned_full(MO, DY, dict(zip(st.trial.astype(int), st.anch.astype(bool))))
cs = common_support(spd, anch)
rate = Y[:, list(ids).index(CL)] / BIN
edges = np.arange(np.floor(spd[cs].min()), spd[cs].max() + 5, 5.)
ctr = edges[:-1] + 2.5
for m, c, lb in ((anch & cs, ANCH_COLOR, 'anchored'), (~anch & cs, NONANCH_COLOR, 'non-anchored')):
    mu = np.full(len(ctr), np.nan); se = np.full(len(ctr), np.nan)
    for i in range(len(ctr)):
        v = rate[m & (spd >= edges[i]) & (spd < edges[i + 1])]
        if len(v) >= 10:
            mu[i], se[i] = v.mean(), v.std() / np.sqrt(len(v))
    ax.fill_between(ctr, mu - se, mu + se, color=c, alpha=.25, lw=0, edgecolor='none')
    ax.plot(ctr, mu, color=c, lw=1.8, label=lb)
ax.set_xlabel('running speed (cm/s)', fontsize=8); ax.set_ylabel('firing rate (Hz)', fontsize=8)
ax.set_title(f'an open-field speed interneuron\nM{MO} D{DY} cluster {CL}', fontsize=7.6, loc='left')
ax.legend(fontsize=6.5, frameon=False, loc='upper left')
tidy(ax)
print(f'  A: M{MO}D{DY} c{CL}, gain change {100 * ex.chg:+.1f}% (group {100 * med:+.1f}%)')

# B: gain per group and state, sessions paired
ax = fig.add_subplot(gs[0, 1:]); lp(ax, 'B')
for k, g in enumerate(GROUP_ORDER):
    S = PS_[g]
    gn, ga = 100 * S.b_n, 100 * (S.b_n + S.b_i)
    x0 = 2.6 * k
    for a_, b_ in zip(gn, ga):
        ax.plot([x0, x0 + 1], [a_, b_], color='0.8', lw=.5, zorder=1)
    for x_, v, c in ((x0, gn, NONANCH_COLOR), (x0 + 1, ga, ANCH_COLOR)):
        ax.scatter(np.full(len(v), x_), v, s=7, color=c, lw=0, zorder=2)
        ax.plot([x_ - .25, x_ + .25], [v.mean()] * 2, color=c, lw=2, zorder=3)
    r = RES[g]
    ax.text(x0 + .5, 1.0, f'{r["pct"]:+.0f}%\n{star(r["p"])}', transform=ax.get_xaxis_transform(),
            ha='center', va='top', fontsize=6.2, color='#a33' if r['p'] < .05 else '0.4')
    ax.text(x0 + .5, -.13, GLAB[g] + f'\n({len(S)} sess.)', transform=ax.get_xaxis_transform(),
            ha='center', va='top', fontsize=6.6)
ax.axhline(0, color='0.6', lw=.6, ls=':')
ax.set_xticks([]); ax.set_xlim(-.6, 2.6 * 3 + 1.6)
ax.set_ylabel('speed gain, % rate per 10 cm/s\n(position held)', fontsize=8)
_t, _b = ax.get_ylim(); ax.set_ylim(_t, _b + .25 * (_b - _t))
ax.set_title('gain in each state (teal non-anchored, purple anchored); bars, mean over sessions; '
             'p, mixed model (mouse)', fontsize=7, loc='left')
tidy(ax)

# C: change with 95% CI, position held and not
ax = fig.add_subplot(gs[1, 0]); lp(ax, 'C')
for k, g in enumerate(GROUP_ORDER[:2]):
    for dy_, store, mk in ((-.12, RES, 'o'), (.12, RES0, 'o')):
        r = store[g]
        filled = store is RES
        ax.errorbar(r['pct'], k + dy_, xerr=[[r['pct'] - r['lo']], [r['hi'] - r['pct']]],
                    fmt=mk, ms=5, color=GCOL[g], mfc=GCOL[g] if filled else 'white', lw=1.2,
                    capsize=0)
ax.axvline(0, color='0.5', lw=.7)
ax.axvline(THETA_FALL, color='#7b4173', lw=1, ls=(0, (3, 2)))
ax.text(THETA_FALL - 2, -.47, "theta's speed\nslope, Fig. 8", ha='right', va='top', fontsize=5.8,
        color='#7b4173')
ax.set_yticks([0, 1]); ax.set_yticklabels([GLAB[g] for g in GROUP_ORDER[:2]], fontsize=7)
ax.set_ylim(1.7, -.5)
ax.set_xlabel('change in speed gain when anchored\n(% of non-anchored, 95% CI)', fontsize=8)
ax.plot([], [], 'o', color='0.3', ms=5, label='position held')
ax.plot([], [], 'o', color='0.3', mfc='white', ms=5, label='position left out')
ax.legend(fontsize=6.2, frameon=False, loc='lower left', handletextpad=.2)
ax.set_title('the speed cells', fontsize=7.6, loc='left')
tidy(ax)


def pairs(ax, a, n, ylab, title, letter):
    lp(ax, letter)
    for x, y in zip(n, a):
        ax.plot([0, 1], [x, y], color='0.8', lw=.5, zorder=1)
    for x_, v, c in ((0, n, NONANCH_COLOR), (1, a, ANCH_COLOR)):
        ax.scatter(np.full(len(v), x_), v, s=10, color=c, lw=0, zorder=2)
        ax.plot([x_ - .2, x_ + .2], [np.median(v)] * 2, color=c, lw=2, zorder=3)
    p = smf.mixedlm('d ~ 1', SS.assign(d=a - n), groups=SS['mouse']).fit(reml=True).pvalues['Intercept']
    ax.set_xticks([0, 1]); ax.set_xticklabels(['non-\nanchored', 'anchored'], fontsize=7)
    ax.set_xlim(-.5, 1.5); ax.set_ylabel(ylab, fontsize=8)
    ax.set_title(f'{title}\n{len(a)} sessions, {(a > n).sum()} higher anchored; {star(p)}',
                 fontsize=7.2, loc='left')
    tidy(ax)
    return p


pD = pairs(fig.add_subplot(gs[1, 1]), SS.dec_r_a.values, SS.dec_r_n.values,
           'decoding r, unexpected speed', 'population decoding of unexpected speed', 'D')
pE = pairs(fig.add_subplot(gs[1, 2]), SS.sres_sd_a.values, SS.sres_sd_n.values,
           'SD of unexpected speed (cm/s)', 'how much unexpected speed there is', 'E')
fig.suptitle('Speed interneurons keep their speed gain in both states; speed principal cells '
             'lose some of theirs once position is held', fontsize=9.5, x=.06, ha='left', y=.99)
out = f'{FIG}/fig7_supp_speed_tuning.pdf'
plt.savefig(out, dpi=200, bbox_inches='tight')
print('saved', out)
for g in GROUP_ORDER:
    r, r0 = RES[g], RES0[g]
    print(f'  {g:24s} held {r["pct"]:+.1f}% [{r["lo"]:+.1f}, {r["hi"]:+.1f}] p={r["p"]:.3g} | '
          f'left out {r0["pct"]:+.1f}% [{r0["lo"]:+.1f}, {r0["hi"]:+.1f}] p={r0["p"]:.3g} | '
          f'{(PS_[g].b_i < 0).sum()}/{len(PS_[g])} sessions lower anchored')
print(f'  D decoding {SS.dec_r_a.median():.3f} vs {SS.dec_r_n.median():.3f}, mixed p={pD:.3g}; '
      f'E unexpected SD {SS.sres_sd_a.median():.1f} vs {SS.sres_sd_n.median():.1f}, mixed p={pE:.3g}')
