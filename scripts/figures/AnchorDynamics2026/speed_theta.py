"""Does the speed-theta relationship break down between anchoring states?

Theta amplitude and frequency both track running speed, so a state difference in
theta could be nothing more than a state difference in speed. Two separable
questions:

  1. OFFSET  -- at matched speed, is theta smaller when the population is
     anchored? (the intercept / main effect of state)
  2. GAIN    -- is theta's *sensitivity* to speed different between states, i.e.
     does the coupling itself break down? (the speed x state interaction)

Per session: Pearson r of speed against theta within each state, and an OLS
amp ~ speed * anch whose interaction term is the gain test. Sessions are then
combined by Fisher-z on the correlations and by a sign test / Wilcoxon on the
per-session interaction, so no session dominates.

Speed is genuinely less variable inside anchored epochs (SD 6.4 vs 8.4 cm/s,
p = 1e-8), which attenuates a correlation on its own. The slope is therefore the
primary measure -- it is unaffected by range restriction -- and the r panels are
shown alongside it. Controls in st_check.py: the slope difference survives
trimming both states to their common 5-95% speed support (0.12 vs 0.34,
p = 0.008), the per-session dr does not track the per-session dSD
(rho = -0.01, p = 0.92), and a circular-shift null on the state vector, which
preserves run structure and any within-session drift, gives p = 0.005.
"""
import warnings
import numpy as np, pandas as pd
warnings.filterwarnings('ignore')
import matplotlib; matplotlib.use('Agg')
import matplotlib.pyplot as plt
from scipy.stats import pearsonr, wilcoxon, ttest_1samp
import sys
sys.path.insert(0, '/Users/harryclark/Documents/spatial-manifolds/src')
from spatial_manifolds.anchoring import ANCH_COLOR, NONANCH_COLOR

plt.rcParams['font.family'] = 'Arial'
FIG = '/Users/harryclark/Documents/spatial-manifolds/scripts/figures/AnchorDynamics2026'
MINTR = 15                      # trials per state before a within-state r means anything
# The scatter panel's session is chosen from the data, not hardcoded. A fixed
# choice silently breaks when the labels change: M21 D22 was the example until
# the gated classifier made it single-state, at which point the panel crashed on
# an empty vector. The session picked is the one closest to the MEDIAN change in
# coupling among those with enough trials in both states, so the panel cannot be
# a flattering outlier either.
MO, DY = None, None

T = pd.read_csv('/Users/harryclark/Documents/spatial-manifolds/data/lfp/theta_frequency.csv')
T['anch'] = T.anch.astype(bool)


def fit(d, y):
    """OLS y ~ speed * anch. Returns per-state slopes and the interaction test."""
    d = d.dropna(subset=['speed', y])
    s, a, v = d.speed.values, d.anch.values.astype(float), d[y].values
    X = np.column_stack([np.ones_like(s), s, a, s * a])
    beta, *_ = np.linalg.lstsq(X, v, rcond=None)
    resid = v - X @ beta
    dof = len(v) - X.shape[1]
    cov = np.linalg.pinv(X.T @ X) * (resid @ resid) / dof
    se = np.sqrt(np.diag(cov))
    from scipy.stats import t as tdist
    p_int = 2 * tdist.sf(abs(beta[3] / se[3]), dof)          # gain
    p_state = 2 * tdist.sf(abs(beta[2] / se[2]), dof)        # offset at speed 0
    return beta, se, p_int, p_state


rows = []
for (mo, dy), d in T.groupby(['mouse', 'day']):
    a, n = d[d.anch], d[~d.anch]
    if len(a) < MINTR or len(n) < MINTR:
        continue
    r = dict(mouse=mo, day=dy, n_anch=len(a), n_non=len(n))
    for y in ('amp', 'inst_hz'):
        r[f'r_anch_{y}'] = pearsonr(a.speed, a[y])[0]
        r[f'r_non_{y}'] = pearsonr(n.speed, n[y])[0]
        b, se, pi, ps = fit(d, y)
        r[f'slope_non_{y}'], r[f'slope_anch_{y}'] = b[1], b[1] + b[3]
        r[f'inter_{y}'], r[f'p_inter_{y}'] = b[3], pi
        r[f'state_{y}'], r[f'p_state_{y}'] = b[2], ps
    rows.append(r)
R = pd.DataFrame(rows)
print(f'{len(R)} sessions with >= {MINTR} trials in both states '
      f'(of {T.groupby(["mouse","day"]).ngroups})\n')

fz = lambda x: np.arctanh(np.clip(x, -.999, .999))
for y, lab in (('amp', 'theta amplitude'), ('inst_hz', 'theta frequency')):
    ra, rn = R[f'r_anch_{y}'], R[f'r_non_{y}']
    w = wilcoxon(ra, rn)
    print(f'--- speed vs {lab} ---')
    print(f'  within-state r   anchored {np.tanh(fz(ra).mean()):+.3f}   '
          f'non-anchored {np.tanh(fz(rn).mean()):+.3f}   '
          f'Wilcoxon p = {w.pvalue:.3g}  ({(ra > rn).sum()}/{len(R)} higher when anchored)')
    sl_a, sl_n = R[f'slope_anch_{y}'], R[f'slope_non_{y}']
    ti = ttest_1samp(R[f'inter_{y}'], 0)
    print(f'  slope            anchored {sl_a.mean():+.4f}   '
          f'non-anchored {sl_n.mean():+.4f}   per unit speed')
    print(f'  GAIN  (speed x state)  mean {R[f"inter_{y}"].mean():+.4f}, '
          f't p = {ti.pvalue:.3g}, '
          f'{(R[f"p_inter_{y}"] < .05).sum()}/{len(R)} sessions individually p<.05')
    print(f'  OFFSET (state)         mean {R[f"state_{y}"].mean():+.4f}, '
          f't p = {ttest_1samp(R[f"state_{y}"], 0).pvalue:.3g}, '
          f'{(R[f"p_state_{y}"] < .05).sum()}/{len(R)} sessions individually p<.05\n')

R.to_csv('/Users/harryclark/Documents/spatial-manifolds/data/lfp/speed_theta_coupling.csv',
         index=False)

# ---------------------------------------------------------------- figure
# common speed support: trim both states to their overlapping 5-95% range, so
# the slopes are compared over the same speeds rather than over whatever range
# each state happened to occupy.
trim = []
for (mo, dy), d in T.groupby(['mouse', 'day']):
    a, n = d[d.anch], d[~d.anch]
    if len(a) < MINTR or len(n) < MINTR:
        continue
    lo = max(a.speed.quantile(.05), n.speed.quantile(.05))
    hi = min(a.speed.quantile(.95), n.speed.quantile(.95))
    a, n = a[a.speed.between(lo, hi)], n[n.speed.between(lo, hi)]
    if len(a) < MINTR or len(n) < MINTR:
        continue
    trim.append((np.polyfit(a.speed, a.amp, 1)[0], np.polyfit(n.speed, n.amp, 1)[0]))
trim = np.array(trim)

fig, axes = plt.subplots(2, 2, figsize=(7.4, 6.0))
axes = axes.ravel()

if MO is None:
    cand = R[(R.n_anch >= 40) & (R.n_non >= 40)].copy()
    if not len(cand):
        cand = R.copy()
    cand['d'] = cand.r_anch_amp - cand.r_non_amp
    pick = cand.iloc[(cand.d - R.eval('r_anch_amp - r_non_amp').median()).abs().argsort()].iloc[0]
    MO, DY = int(pick.mouse), int(pick.day)
    print(f'scatter panel session chosen from the data: M{MO}D{DY} '
          f'(dr = {pick.d:+.3f}, median {R.eval("r_anch_amp - r_non_amp").median():+.3f})')
d = T[(T.mouse == MO) & (T.day == DY)]
ax = axes[0]
for m, c, lb in ((d.anch, ANCH_COLOR, 'anchored'), (~d.anch, NONANCH_COLOR, 'non-anchored')):
    s_ = d[m]
    ax.scatter(s_.speed, s_.amp, s=9, color=c, alpha=.55, lw=0, label=lb)
    xs = np.linspace(s_.speed.min(), s_.speed.max(), 50)
    ax.plot(xs, np.polyval(np.polyfit(s_.speed, s_.amp, 1), xs), color=c, lw=2)
    r, _ = pearsonr(s_.speed, s_.amp)
    ax.text(.03, .95 if lb == 'anchored' else .87, f'{lb}: r = {r:.2f}',
            transform=ax.transAxes, fontsize=7.5, color=c, va='top')
_, _, pi, _ = fit(d, 'amp')
ax.text(.03, .79, f'speed x state p = {pi:.2g}', transform=ax.transAxes,
        fontsize=7.5, color='0.3', va='top')
ax.set_xlabel('Trial mean speed (cm/s)', fontsize=9)
ax.set_ylabel('Theta amplitude', fontsize=9)
ax.set_title(f'M{MO} D{DY}', fontsize=9.5, loc='left')
ax.legend(fontsize=7.5, frameon=False, loc='lower right')


def paired(ax, va, vn, ylab, title):
    for i in range(len(va)):
        ax.plot([0, 1], [vn[i], va[i]], color='0.75', lw=.6, zorder=1)
    ax.scatter(np.zeros(len(va)), vn, s=14, color=NONANCH_COLOR, lw=0, zorder=2)
    ax.scatter(np.ones(len(va)), va, s=14, color=ANCH_COLOR, lw=0, zorder=2)
    for x, v, c in ((0, vn, NONANCH_COLOR), (1, va, ANCH_COLOR)):
        ax.plot([x - .18, x + .18], [np.median(v)] * 2, color=c, lw=2.4, zorder=3)
    ax.axhline(0, color='0.5', lw=.7, ls=':')
    ax.set_xticks([0, 1]); ax.set_xticklabels(['non-anch', 'anchored'], fontsize=8)
    ax.set_xlim(-.4, 1.4)
    ax.set_ylabel(ylab, fontsize=9)
    ax.set_title(f'{title} — p = {wilcoxon(va, vn).pvalue:.2g}', fontsize=9.5, loc='left')


paired(axes[1], trim[:, 0], trim[:, 1], 'Slope (amp per cm/s)',
       f'{len(trim)} sessions, matched speed range')
paired(axes[2], R.r_anch_amp.values, R.r_non_amp.values, 'r (speed, theta amplitude)',
       f'{len(R)} sessions')
paired(axes[3], R.r_anch_inst_hz.values, R.r_non_inst_hz.values,
       'r (speed, theta frequency)', f'{len(R)} sessions')

for ax in axes:
    ax.spines[['top', 'right']].set_visible(False)
    ax.tick_params(labelsize=7.5)
plt.tight_layout()
out = f'{FIG}/fig6_speed_theta_coupling.pdf'
plt.savefig(out, dpi=200, bbox_inches='tight')
print('saved', out)
