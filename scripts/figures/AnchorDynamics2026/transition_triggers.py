"""What triggers a population anchoring transition?

The paper establishes that the state exists, that grid cells and interneurons
follow it, that the local wiring does not carry it, and that pupil tracks it.
It does not say what makes it switch. Two families of account make opposite
predictions and the task separates them:

  LANDMARK-DRIVEN -- the state is re-established by visual contact with the cue,
    so entries into the anchored state should follow CUED (beaconed) trials, and
    long runs of uncued trials should precede exits.

  ENDOGENOUS -- the state is set by something the task does not control (an
    arousal-linked input, per Figure 5), so trial type should carry no
    information about when it switches.

Behaviour gives a third possibility worth separating from both: the state may
follow performance rather than cue availability, i.e. errors precede exits.
That is distinct from the landmark account because misses occur on cued trials
too.

THE NULL IS THE WHOLE ANALYSIS. Transitions are rare and states come in long
blocks, so any statistic computed in a window around a transition is compared
against the same statistic at CIRCULARLY SHIFTED transition times within the
same session. The shift preserves the number of transitions, the run-length
structure of the state, and every session's own trial-type schedule and hit
rate; it destroys only the alignment between them. A parametric test on these
windows would be badly anticonservative, because neighbouring trials share a
state and the windows overlap.

Outputs
  data/population_state/transition_triggers.csv   per-session statistics
  fig_transition_triggers.pdf                     the figure
"""
import sys
import warnings

import numpy as np
import pandas as pd

warnings.filterwarnings('ignore')
import matplotlib

matplotlib.use('Agg')
import matplotlib.pyplot as plt
from scipy.ndimage import median_filter
from scipy.stats import wilcoxon

sys.path.insert(0, '/Users/harryclark/Documents/spatial-manifolds/src')
from spatial_manifolds.anchoring import ANCH_COLOR, NONANCH_COLOR

plt.rcParams['font.family'] = 'Arial'
plt.rcParams['pdf.fonttype'] = 42

ROOT = '/Users/harryclark/Documents/spatial-manifolds'
FIG = f'{ROOT}/scripts/figures/AnchorDynamics2026'
OUT_CSV = f'{ROOT}/data/population_state/transition_triggers.csv'
OUT_PDF = f'{FIG}/fig_transition_triggers.pdf'

MAJOR_FILT = 9        # trials; the Figure 1/2 convention for a major transition
W = 5                 # window half-width, trials
N_SHIFT = 500         # circular shifts of the transition times, per session
MIN_TRIALS = 40
GRID_C = '#c04744'

T = pd.read_csv(f'{ROOT}/data/population_state/anchoring_trials.csv')
N = pd.read_csv(f'{ROOT}/data/population_state/nonmec_population_state.csv')
M = T.merge(N[['mouse', 'day', 'trial', 'speed', 'ttype', 'hit']],
            on=['mouse', 'day', 'trial'], how='left')
M['cued'] = (M.ttype == 'b').astype(float)
M = M.sort_values(['mouse', 'day', 'trial']).reset_index(drop=True)


def state_and_transitions(fa):
    """Thresholded state after the 9-trial median filter, and its sign changes."""
    st = median_filter((np.asarray(fa, float) > .5).astype(float),
                       size=MAJOR_FILT, mode='nearest') > .5
    tr = np.where(np.diff(st.astype(int)) != 0)[0] + 1
    return st, tr


def window_mean(x, idx, lo, hi, n):
    """Mean of x over [i+lo, i+hi) for every transition i, clipped to the session."""
    out = []
    for i in idx:
        a, b = max(0, i + lo), min(n, i + hi)
        if b > a:
            out.append(np.nanmean(x[a:b]))
    return np.nan if not out else float(np.nanmean(out))


rows = []
rng = np.random.default_rng(0)
for (mo, dy), d in M.groupby(['mouse', 'day']):
    n = len(d)
    if n < MIN_TRIALS:
        continue
    st, tr = state_and_transitions(d.frac_anch.values)
    if len(tr) == 0:
        continue
    entries = [i for i in tr if st[i]]          # into the anchored state
    exits = [i for i in tr if not st[i]]        # out of it
    cued, hit, spd = d.cued.values, d.hit.values.astype(float), d.speed.values

    for nm, idx in (('entry', entries), ('exit', exits)):
        if not idx:
            continue
        # observed: the three trials immediately BEFORE the transition
        obs = dict(cued=window_mean(cued, idx, -3, 0, n),
                   hit=window_mean(hit, idx, -3, 0, n),
                   speed=window_mean(spd, idx, -3, 0, n))
        # null: the same windows around circularly shifted transition times,
        # which keeps each session's schedule, hit rate and run structure and
        # breaks only the alignment to the state
        nul = {k: [] for k in obs}
        for _ in range(N_SHIFT):
            sh = (np.array(idx) + rng.integers(1, n)) % n
            for k, x in (('cued', cued), ('hit', hit), ('speed', spd)):
                nul[k].append(window_mean(x, sh, -3, 0, n))
        r = dict(mouse=mo, day=dy, kind=nm, n_trans=len(idx), n_trials=n)
        for k in obs:
            nd = np.asarray(nul[k], float)
            r[k] = obs[k]
            r[f'{k}_null'] = float(np.nanmean(nd))
            r[f'{k}_z'] = ((obs[k] - np.nanmean(nd)) / np.nanstd(nd)
                           if np.nanstd(nd) > 0 else np.nan)
        rows.append(r)

R = pd.DataFrame(rows)
R.to_csv(OUT_CSV, index=False)
print(f'{len(R)} session x kind rows, '
      f'{R.groupby(["mouse", "day"]).ngroups} sessions\n')

# ---- does the state depend on trial type or performance at all? ---------------
# Separate question from the trigger one: even if nothing predicts the SWITCH,
# the state might still be associated with trial type or with being correct.
base = []
for (mo, dy), d in M.groupby(['mouse', 'day']):
    if len(d) < MIN_TRIALS:
        continue
    st, _ = state_and_transitions(d.frac_anch.values)
    if st.all() or not st.any():
        continue
    base.append(dict(mouse=mo, day=dy,
                     cued_A=d.cued.values[st].mean(),
                     cued_N=d.cued.values[~st].mean(),
                     hit_A=d.hit.values[st].mean(),
                     hit_N=d.hit.values[~st].mean(),
                     spd_A=d.speed.values[st].mean(),
                     spd_N=d.speed.values[~st].mean()))
B = pd.DataFrame(base)
print(f'state-association: {len(B)} sessions that switch')
for lab, a, b in (('cued fraction', 'cued_A', 'cued_N'),
                  ('hit rate', 'hit_A', 'hit_N'),
                  ('speed', 'spd_A', 'spd_N')):
    dd = (B[a] - B[b]).dropna()
    p = wilcoxon(dd).pvalue
    print(f'  {lab:14s} anchored {B[a].mean():.3f}  non {B[b].mean():.3f}  '
          f'diff {dd.mean():+.3f}  p={p:.3g}')

print('\ntrigger tests (observed vs circular-shift null, z per session):')
summ = {}
for kind in ('entry', 'exit'):
    sub = R[R.kind == kind]
    print(f'  --- {kind} (n={len(sub)} sessions) ---')
    for k in ('cued', 'hit', 'speed'):
        z = sub[f'{k}_z'].dropna()
        if len(z) < 5:
            continue
        p = wilcoxon(z).pvalue
        summ[(kind, k)] = (z.mean(), p, len(z))
        print(f'    {k:6s} obs {sub[k].mean():.3f}  null {sub[k + "_null"].mean():.3f}'
              f'  median z {z.median():+.3f}  p={p:.3g}  ({int((z > 0).sum())}/{len(z)} > 0)')

# ============================== figure =========================================
fig = plt.figure(figsize=(9.4, 6.4))
G = fig.add_gridspec(2, 3, hspace=.55, wspace=.42)


def tidy(ax):
    ax.tick_params(labelsize=7)
    ax.spines[['top', 'right']].set_visible(False)


def lp(ax, s, x=-.22):
    ax.text(x, 1.04, s, transform=ax.transAxes, fontsize=10, weight='bold',
            va='bottom', ha='right')


# A-C: the state's association with trial type, performance and speed
for k, (ax_i, lab, a, b, ylab) in enumerate((
        (0, 'cued (beaconed) fraction', 'cued_A', 'cued_N', 'fraction cued'),
        (1, 'hit rate', 'hit_A', 'hit_N', 'hit rate'),
        (2, 'running speed', 'spd_A', 'spd_N', 'speed (cm/s)'))):
    ax = fig.add_subplot(G[0, ax_i])
    for _, r in B.iterrows():
        ax.plot([0, 1], [r[b], r[a]], color='0.75', lw=.6, zorder=1)
    ax.plot([0, 1], [B[b].mean(), B[a].mean()], 'o-', color=GRID_C, ms=6, lw=2,
            zorder=3)
    dd = (B[a] - B[b]).dropna()
    ax.set_xticks([0, 1])
    ax.set_xticklabels(['non-anch', 'anchored'], fontsize=7.5)
    ax.set_xlim(-.3, 1.3)
    ax.set_ylabel(ylab, fontsize=8)
    ax.set_title(f'{lab}\n$\\Delta$ {dd.mean():+.3f}, p = {wilcoxon(dd).pvalue:.2g}',
                 fontsize=8.5)
    tidy(ax)
    lp(ax, 'ABC'[ax_i])

# D-F: the trigger test
for k, (ax_i, key, lab) in enumerate(((0, 'cued', 'cued fraction'),
                                      (1, 'hit', 'hit rate'),
                                      (2, 'speed', 'speed'))):
    ax = fig.add_subplot(G[1, ax_i])
    xs, labels = [], []
    for j, kind in enumerate(('entry', 'exit')):
        z = R[R.kind == kind][f'{key}_z'].dropna()
        if not len(z):
            continue
        ax.scatter(np.full(len(z), j) + rng.normal(0, .06, len(z)), z, s=7,
                   color='0.6', alpha=.65, linewidth=0, zorder=2)
        ax.plot([j - .22, j + .22], [z.median()] * 2, color=GRID_C, lw=2.2,
                zorder=3)
        p = wilcoxon(z).pvalue
        labels.append(f'{kind}\np={p:.2g}')
        xs.append(j)
    ax.axhline(0, color='k', ls=':', lw=1)
    ax.set_xticks(xs); ax.set_xticklabels(labels, fontsize=7.5)
    ax.set_xlim(-.5, 1.5)
    ax.set_ylabel(f'{lab}, z vs shift null', fontsize=8)
    ax.set_title(f'{lab} in the 3 trials\nbefore a transition', fontsize=8.5)
    tidy(ax)
    lp(ax, 'DEF'[ax_i])

fig.savefig(OUT_PDF, bbox_inches='tight')
print(f'\nwrote {OUT_PDF}\nwrote {OUT_CSV}')
