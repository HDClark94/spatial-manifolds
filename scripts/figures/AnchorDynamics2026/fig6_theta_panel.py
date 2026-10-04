"""Figure 5 — theta and the population anchoring state, in one panel.

    A   one session on a shared trial axis: the population anchoring raster,
        theta amplitude, theta frequency and running speed
    B   theta amplitude, theta frequency and speed by state, paired across
        sessions, one bar pair each
    C   the gain: within-state correlation of speed with theta amplitude and
        with theta frequency, and the scatter behind it for one session

The example session is chosen FROM THE DATA, not hardcoded: among sessions with
at least 15 trials in each state, enough cells to read a raster, a significant
amplitude difference, and a state sequence that comes in BLOCKS rather than
flickering, the one whose effect sits closest to the dataset median. The
blockiness requirement is not cosmetic -- a session that alternates state every
few trials is representative in effect size but unreadable as a panel, and the
first version of this figure picked exactly such a session. A hardcoded example silently breaks when labels change -- the
previous figure's session became single-state under the gated classifier and the
panel crashed on an empty vector -- and it risks showing a flattering outlier.

AMPLITUDE IS NORMALISED WITHIN SESSION for the bar panel. Absolute LFP amplitude
depends on electrode impedance and reference and spans 50-180 across sessions,
so raw values would give a between-session spread that swamps the within-session
effect. Frequency (Hz) and speed (cm/s) are comparable across sessions and are
plotted raw.

Trials are NOT speed-matched in the bar panels, so that the speed panel can
answer "do the states differ in speed at all?". The speed-matched statistics are
printed alongside, and it is those that carry the claim.
"""
import os, sys, warnings
import numpy as np, pandas as pd
warnings.filterwarnings('ignore')
import matplotlib; matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap, BoundaryNorm
from scipy.stats import wilcoxon, mannwhitneyu, pearsonr
sys.path.insert(0, '/Users/harryclark/Documents/spatial-manifolds/src')
from spatial_manifolds.anchoring import ANCH_COLOR, NONANCH_COLOR, load_session_labels

plt.rcParams['font.family'] = 'Arial'
FIG = '/Users/harryclark/Documents/spatial-manifolds/scripts/figures/AnchorDynamics2026'
LFP = '/Users/harryclark/Documents/spatial-manifolds/data/lfp'
MINTR, SMOOTH = 15, 5

T = pd.read_csv(f'{LFP}/theta_frequency.csv')
T['anch'] = T.anch.astype(bool)

# ── per-session summaries ────────────────────────────────────────────────────
rows = []
for (mo, dy), d in T.groupby(['mouse', 'day']):
    a, n = d[d.anch], d[~d.anch]
    if len(a) < MINTR or len(n) < MINTR:
        continue
    m = d[d.speed_matched] if 'speed_matched' in d else d
    ma, mn = m[m.anch], m[~m.anch]
    r = dict(mouse=mo, day=dy, n_a=len(a), n_n=len(n),
             amp_a=a.amp.mean() / d.amp.mean(), amp_n=n.amp.mean() / d.amp.mean(),
             hz_a=a.inst_hz.mean(), hz_n=n.inst_hz.mean(),
             spd_a=a.speed.mean(), spd_n=n.speed.mean())
    if len(ma) >= 5 and len(mn) >= 5:                 # speed-matched check
        r['amp_a_m'] = ma.amp.mean() / m.amp.mean()
        r['amp_n_m'] = mn.amp.mean() / m.amp.mean()
        r['hz_a_m'], r['hz_n_m'] = ma.inst_hz.mean(), mn.inst_hz.mean()
    for tag, col in (('r_amp', 'amp'), ('r_hz', 'inst_hz')):
        r[f'{tag}_a'] = pearsonr(a.speed, a[col])[0]
        r[f'{tag}_n'] = pearsonr(n.speed, n[col])[0]
    rows.append(r)
S = pd.DataFrame(rows)
print(f'{len(S)} sessions with >= {MINTR} trials in both states (of '
      f'{T.groupby(["mouse","day"]).ngroups})')

# ── choose the example session from the data ─────────────────────────────────
cand = []
for _, r in S.iterrows():
    z = load_session_labels(int(r.mouse), int(r.day))
    if z is None:
        continue
    ncell = int(np.isfinite(z['pc1_load']).sum())
    d = T[(T.mouse == r.mouse) & (T.day == r.day)]
    p = mannwhitneyu(d[d.anch].amp, d[~d.anch].amp)[1]
    eff = 100 * (r.amp_a - r.amp_n)
    # state run structure: a legible raster needs blocks, not flicker
    st = np.asarray(z['frac_anch']) > .5
    runs = np.diff(np.r_[0, np.where(np.diff(st.astype(int)) != 0)[0] + 1, len(st)])
    blocky = np.median(runs) >= 8
    if ncell >= 80 and r.n_a + r.n_n >= 100 and p < .05 and eff < 0 and blocky:
        cand.append((abs(eff - 100 * (S.amp_a - S.amp_n).median()), r.mouse, r.day,
                     ncell, eff))
cand.sort()
_, MO, DY, NC, EFF = cand[0]
MO, DY = int(MO), int(DY)
print(f'example session (data-driven): M{MO}D{DY} — {NC} cells, '
      f'amplitude {EFF:+.1f}% vs dataset median '
      f'{100*(S.amp_a - S.amp_n).median():+.1f}%')

z = load_session_labels(MO, DY)
in_pop = np.isfinite(z['pc1_load'])
L, load = z['labels'][in_pop], z['pc1_load'][in_pop]
order = np.argsort(load)[::-1]
te = T[(T.mouse == MO) & (T.day == DY)].sort_values('trial')
n_tr = L.shape[1]
tr_ax = np.arange(1, n_tr + 1)
sm = lambda v: pd.Series(v).rolling(SMOOTH, center=True, min_periods=2).mean().values

# ── layout ───────────────────────────────────────────────────────────────────
fig = plt.figure(figsize=(7.5, 9.4))
gs = fig.add_gridspec(9, 3, height_ratios=[1.5, .62, .62, .62, .30, 1.15, .22, 1.15, .06],
                      hspace=.38, wspace=.46)

TA = ListedColormap([NONANCH_COLOR, ANCH_COLOR])
NORM = BoundaryNorm([-0.5, 0.5, 1.5], TA.N)
ax = fig.add_subplot(gs[0, :])
ax.imshow(L[order], aspect='auto', cmap=TA, norm=NORM, interpolation='nearest',
          extent=[1, n_tr, L.shape[0], 1])
ax.set_ylabel('Cell (PC1 order)', fontsize=8.5)
ax.set_title(f'M{MO} D{DY}', fontsize=9.5, loc='left')
ax.set_xticklabels([])
for sp in ax.spines.values():
    sp.set_visible(False)

for k, (col, lab, c) in enumerate((('amp', 'Theta amp.', '#b5651d'),
                                   ('inst_hz', 'Theta (Hz)', '#3b6ea5'),
                                   ('speed', 'Speed (cm/s)', '#4f8f4f'))):
    ax = fig.add_subplot(gs[k + 1, :])
    v = np.full(n_tr, np.nan)
    ok = (te.trial.values >= 1) & (te.trial.values <= n_tr)
    v[te.trial.values[ok] - 1] = te[col].values[ok]
    ax.plot(tr_ax, v, color='0.78', lw=.5)
    ax.plot(tr_ax, sm(v), color=c, lw=1.6)
    lo, hi = np.nanpercentile(v, [1, 99]); pad = (hi - lo) * .35
    ax.set_ylim(lo - pad, hi + pad)
    ax.set_ylabel(lab, fontsize=8)
    ax.tick_params(labelsize=7)
    ax.spines[['top', 'right']].set_visible(False)
    if k < 2:
        ax.set_xticklabels([])
    else:
        ax.set_xlabel('Trial', fontsize=8.5)
    ax.set_xlim(1, n_tr)


def barpair(ax, va, vn, ylab, title, fmt='{:.2f}'):
    """Anchored vs non-anchored, bars with per-session points, paired test."""
    rng = np.random.default_rng(0)
    for x, v, c in ((0, vn, NONANCH_COLOR), (1, va, ANCH_COLOR)):
        ax.bar(x, np.mean(v), width=.72, color=c, lw=.8, edgecolor='k', zorder=1)
        ax.errorbar(x, np.mean(v), yerr=np.std(v, ddof=1) / np.sqrt(len(v)),
                    color='k', lw=1.1, capsize=3, capthick=1.1, zorder=4)
        ax.scatter(x + rng.uniform(-.16, .16, len(v)), v, s=6, color='0.25',
                   alpha=.55, lw=0, zorder=3)
    p = wilcoxon(va, vn).pvalue
    lo = min(np.min(va), np.min(vn)); hi = max(np.max(va), np.max(vn))
    span = hi - lo
    y = hi + .10 * span
    ax.plot([0, 0, 1, 1], [y, y + .04 * span, y + .04 * span, y], color='0.3', lw=.8)
    star = 'n.s.' if p > .05 else ('*' if p > .01 else ('**' if p > .001 else '***'))
    ax.text(.5, y + .07 * span, star, ha='center', fontsize=7.5, color='0.2')
    ax.set_ylim(max(0, lo - .25 * span), y + .22 * span)
    ax.set_xticks([0, 1]); ax.set_xticklabels(['non-anch', 'anchored'], fontsize=7.5)
    ax.set_ylabel(ylab, fontsize=8.5)
    ax.set_title(title, fontsize=8.5, loc='left')
    ax.tick_params(labelsize=7)
    ax.spines[['top', 'right']].set_visible(False)
    return p


p_amp = barpair(fig.add_subplot(gs[5, 0]), S.amp_a.values, S.amp_n.values,
                'Theta amplitude\n(session-normalised)', 'amplitude')
p_hz = barpair(fig.add_subplot(gs[5, 1]), S.hz_a.values, S.hz_n.values,
               'Theta frequency (Hz)', 'frequency')
p_sp = barpair(fig.add_subplot(gs[5, 2]), S.spd_a.values, S.spd_n.values,
               'Speed (cm/s)', 'speed')


def pairdots(ax, va, vn, ylab, title):
    for i in range(len(va)):
        ax.plot([0, 1], [vn[i], va[i]], color='0.78', lw=.6, zorder=1)
    ax.scatter(np.zeros(len(vn)), vn, s=13, color=NONANCH_COLOR, lw=0, zorder=2)
    ax.scatter(np.ones(len(va)), va, s=13, color=ANCH_COLOR, lw=0, zorder=2)
    for x, v, c in ((0, vn, NONANCH_COLOR), (1, va, ANCH_COLOR)):
        ax.plot([x - .17, x + .17], [np.median(v)] * 2, color=c, lw=2.2, zorder=3)
    ax.axhline(0, color='0.6', lw=.7, ls=':')
    p = wilcoxon(va, vn).pvalue
    ax.set_xticks([0, 1]); ax.set_xticklabels(['non-anch', 'anchored'], fontsize=7.5)
    ax.set_xlim(-.4, 1.4)
    ax.set_ylabel(ylab, fontsize=8.5)
    ax.set_title(f'{title} — p = {p:.2g}', fontsize=8.5, loc='left')
    ax.tick_params(labelsize=7)
    ax.spines[['top', 'right']].set_visible(False)
    return p


p_ra = pairdots(fig.add_subplot(gs[7, 0]), S.r_amp_a.values, S.r_amp_n.values,
                'r (speed, amplitude)', 'gain: amplitude')
p_rh = pairdots(fig.add_subplot(gs[7, 1]), S.r_hz_a.values, S.r_hz_n.values,
                'r (speed, frequency)', 'gain: frequency')

ax = fig.add_subplot(gs[7, 2])
d = T[(T.mouse == MO) & (T.day == DY)]
for m, c, lb in ((d.anch, ANCH_COLOR, 'anchored'), (~d.anch, NONANCH_COLOR, 'non-anch')):
    s_ = d[m]
    ax.scatter(s_.speed, s_.amp, s=5, color=c, alpha=.5, lw=0)
    xs = np.linspace(s_.speed.min(), s_.speed.max(), 40)
    ax.plot(xs, np.polyval(np.polyfit(s_.speed, s_.amp, 1), xs), color=c, lw=1.6)
ax.set_xlabel('Speed (cm/s)', fontsize=8.5)
ax.set_ylabel('Theta amplitude', fontsize=8.5)
ax.set_title(f'M{MO} D{DY}', fontsize=8.5, loc='left')
ax.tick_params(labelsize=7)
ax.spines[['top', 'right']].set_visible(False)

h = [plt.Line2D([], [], marker='s', ls='', color=ANCH_COLOR, label='anchored'),
     plt.Line2D([], [], marker='s', ls='', color=NONANCH_COLOR, label='non-anchored')]
fig.legend(handles=h, loc='lower center', ncol=2, fontsize=8, frameon=False,
           bbox_to_anchor=(.5, .002))

out = f'{FIG}/fig6_theta_anchoring.pdf'
plt.savefig(out, dpi=200, bbox_inches='tight')
print(f'saved {out}')
print(f'  amplitude p={p_amp:.3g}  frequency p={p_hz:.3g}  speed p={p_sp:.3g}')
print(f'  gain amplitude p={p_ra:.3g}  gain frequency p={p_rh:.3g}')
if 'amp_a_m' in S:
    M = S.dropna(subset=['amp_a_m', 'amp_n_m'])
    print(f'  speed-matched: amplitude p={wilcoxon(M.amp_a_m, M.amp_n_m).pvalue:.3g}, '
          f'frequency p={wilcoxon(M.hz_a_m, M.hz_n_m).pvalue:.3g} (n={len(M)})')
