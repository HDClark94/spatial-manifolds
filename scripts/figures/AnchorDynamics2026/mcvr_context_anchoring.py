"""Is the population more anchored in the familiar context than the novel one,
and does that change with experience?

MCVR alternates two reward-zone contexts within every session: rz1, which the
animals already know, and rz2, which is novel at first exposure. Because the two
alternate WITHIN a session, the comparison is immune to everything that differs
between sessions -- probe drift, cell yield, arousal, day-to-day behaviour. The
learning prediction is specific: rz2 should start less anchored and converge on
rz1 as it stops being novel, while rz1 stays flat.

    A       one session: population state across trials, with the context blocks
    B, C    WHERE THE ANIMAL FIRST STOPS, by block on day 1 and across days
    D, E    STOP DISCRIMINATION: are the stops at the right zone or the wrong one?
    F, G    PROPORTION ANCHORED, on exactly the same axes

Left column is the first session block by block, right column is training days
1-6, and all three rows share those x-axes, so the behavioural signature sits
directly above the neural one at both timescales. Everything is restricted to
UNCUED trials, where nothing marks the zone, so stopping in the right place
requires knowing where it is.

HIT RATE IS THE WRONG READOUT and is not shown. On cued trials the zone is
marked, so an animal scores 0.74 on its first ever exposure to rz2 without
knowing anything about it; and within day 1 hit rate FALLS (0.68 -> 0.51) as the
animal disengages, moving opposite to any learning.

D/E IS THE MEASURE THAT CARRIES THE CLAIM. The two zones are 30 cm apart (rz1 at
92, rz2 at 122), so the diagnostic is whether stops land on the animal's own
zone or on the other one: (own - other) / (own + other) over all stops within
+- 12 cm of either, per session. On first exposure that index is far lower in
the novel context (+0.34 vs +0.76, 5/6 sessions, p = 0.09 paired), and from day
2 onward the gap is gone (+0.46 vs +0.51, p = 0.62). The deficit is confined to
the first session.

TWO MEASURES THAT LOOK LIKE LEARNING AND ARE NOT, both reported here so the
reader can see why D/E is the one used:

  the median first stop in the novel context moves 86 -> 113 cm across six days
      (B/C), which looks like the animal homing in on rz2 -- but stops never
      concentrate there: the fraction landing within 12 cm of rz2 stays flat
      near 0.17, and about 60% land near neither zone throughout. The median
      moves because the whole distribution shifts later, not because the animal
      found the zone.
  scoring perseveration as "first stop closer to the other zone" makes day 1
      look like habit -- but in block 3 the median first stop is 42-72 cm in
      five of six sessions, nowhere near either zone. Those are disengaged
      early stops, and the permissive rule counts every one as perseveration.

WITHIN DAY 1 THERE IS NO IMPROVEMENT (D). The novel context sits below the
familiar one in every block, and both decline across blocks as the animal
disengages. The novel context catches up BETWEEN session 1 and session 2, not
inside session 1. The earlier reading of a within-day-1 gain came from the
first-stop median, which the disengagement drives on its own.

Blocks are indexed WITHIN context (novel block 1 = the animal's first ever
exposure to rz2) because two of the six day-1 sessions open with a short
familiar lead-in that would otherwise misalign them.

Day 1 is the animal's FIRST exposure to the multi-context task: MCVR day
numbering continues from the standard VR task, so the first day present in this
dataset is the first encounter, and each animal contributes only 5-6 days.

CONTROLS. Contexts are compared within session and paired, so session-level
differences cannot produce the effect. Running speed DOES differ between
contexts (36.1 vs 33.3 cm/s, p = 8e-5), and anchoring covaries with speed, so
the comparison is run twice: on all trials, and on a speed-matched subset drawn
within each session across deciles of trial mean running speed. The matched
version is the one that carries the claim. Sessions contribute only
when the population state exists (>= MIN_CELLS entorhinal cells) and both
contexts carry >= MIN_TRIALS trials.

The statistical unit is the session. Session number is the rank of the recording
day within each mouse, so "session 1" is that animal's first MCVR day rather
than a calendar date.
"""
import os, sys, warnings
import numpy as np, pandas as pd
warnings.filterwarnings('ignore')
import matplotlib; matplotlib.use('Agg')
import matplotlib.pyplot as plt
from scipy.stats import wilcoxon, spearmanr, linregress, mannwhitneyu
sys.path.insert(0, '/Users/harryclark/Documents/spatial-manifolds/src')
sys.path.insert(0, '/Users/harryclark/Documents/spatial-manifolds/scripts/figures/'
                   'AnchorDynamics2026')
from build_mcvr_labels import ensure_mcvr_tables
from spatial_manifolds.anchoring import ANCH_COLOR, NONANCH_COLOR

plt.rcParams['font.family'] = 'Arial'
FIG = '/Users/harryclark/Documents/spatial-manifolds/scripts/figures/AnchorDynamics2026'
MIN_CELLS, MIN_TRIALS = 10, 20
CTX = {'rz1': ('familiar', '#2ca02c'), 'rz2': ('novel', '#d62728')}

ALL = ensure_mcvr_tables()
# Training day must be counted over EVERY MCVR session the animal ran, not only
# the ones that yield a population state. M22's probe missed entorhinal cortex
# entirely, so ranking after the cell filter promoted that animal's second and
# third exposures to "day 1" -- the one index the novelty claim rests on.
ALL['sess_idx'] = ALL.groupby('mouse').day.rank(method='dense').astype(int)
M = ALL[ALL.n_cells >= MIN_CELLS].copy()

def speed_match(d, rng, n_q=8):
    """Indices of a subset with the same speed distribution in both contexts."""
    v = d.speed.values
    ok = np.isfinite(v)
    q = np.quantile(v[ok], np.linspace(0, 1, n_q + 1))
    q[0] -= 1; q[-1] += 1
    keep = []
    c1 = (d.context == 'rz1').values
    for lo, hi in zip(q[:-1], q[1:]):
        m = ok & (v >= lo) & (v < hi)
        a, b = np.where(m & c1)[0], np.where(m & ~c1)[0]
        k = min(len(a), len(b))
        if k:
            keep += list(rng.choice(a, k, replace=False))
            keep += list(rng.choice(b, k, replace=False))
    return np.array(sorted(keep), dtype=int)


rows = []
for (mo, dy), d in M.groupby(['mouse', 'day']):
    d = d.reset_index(drop=True)
    _rng = np.random.default_rng(abs(hash((mo, dy))) % 2**32)
    _k = speed_match(d, _rng)
    dm = d.iloc[_k] if len(_k) else d.iloc[[]]
    r = dict(mouse=mo, day=dy, sess_idx=int(d.sess_idx.iloc[0]),
             n_cells=int(d.n_cells.iloc[0]))
    ok = True
    for c in CTX:
        s = d[d.context == c]
        if len(s) < MIN_TRIALS or s.frac_anch.isna().all():
            ok = False; break
        r[f'frac_{c}'] = s.frac_anch.mean()
        r[f'panch_{c}'] = (s.frac_anch > .5).mean()
        r[f'hit_{c}'] = s.hit.mean()
        r[f'speed_{c}'] = s.speed.mean()
        r[f'n_{c}'] = len(s)
        sm_ = dm[dm.context == c]
        r[f'frac_{c}_sm'] = sm_.frac_anch.mean() if len(sm_) >= MIN_TRIALS else np.nan
        r[f'speed_{c}_sm'] = sm_.speed.mean() if len(sm_) >= MIN_TRIALS else np.nan
        r[f'n_{c}_sm'] = len(sm_)
    if ok:
        r['d_frac'] = r['frac_rz2'] - r['frac_rz1']
        rows.append(r)
S = pd.DataFrame(rows)
S.to_csv('/Users/harryclark/Documents/spatial-manifolds/data/population_state/'
         'mcvr_context_anchoring.csv', index=False)

print(f'{len(S)} sessions from {S.mouse.nunique()} mice '
      f'(of {M.groupby(["mouse","day"]).ngroups} with a population state)')
p_frac = wilcoxon(S.frac_rz1, S.frac_rz2).pvalue
print(f'  anchoring   familiar {S.frac_rz1.mean():.3f}  novel {S.frac_rz2.mean():.3f}'
      f'  p = {p_frac:.3g}  ({(S.frac_rz2 > S.frac_rz1).sum()}/{len(S)} higher in novel)')
p_hit = wilcoxon(S.hit_rz1, S.hit_rz2).pvalue
print(f'  hit rate    familiar {S.hit_rz1.mean():.3f}  novel {S.hit_rz2.mean():.3f}'
      f'  p = {p_hit:.3g}')
p_spd = wilcoxon(S.speed_rz1, S.speed_rz2).pvalue
print(f'  speed       familiar {S.speed_rz1.mean():.1f}  novel {S.speed_rz2.mean():.1f}'
      f' cm/s  p = {p_spd:.3g}'
      + ('   <-- differs; anchoring comparison needs speed matching' if p_spd < .05 else ''))
print()
SM = S.dropna(subset=['frac_rz1_sm', 'frac_rz2_sm'])
if len(SM) >= 5:
    p_sm = wilcoxon(SM.frac_rz1_sm, SM.frac_rz2_sm).pvalue
    print(f'  SPEED-MATCHED ({len(SM)} sessions, '
          f'{SM[["n_rz1_sm", "n_rz2_sm"]].sum(axis=1).median():.0f} trials median):')
    print(f'    anchoring familiar {SM.frac_rz1_sm.mean():.3f}  '
          f'novel {SM.frac_rz2_sm.mean():.3f}  p = {p_sm:.3g}  '
          f'({(SM.frac_rz2_sm > SM.frac_rz1_sm).sum()}/{len(SM)} higher in novel)')
    print(f'    speed     familiar {SM.speed_rz1_sm.mean():.1f}  '
          f'novel {SM.speed_rz2_sm.mean():.1f} cm/s  '
          f'p = {wilcoxon(SM.speed_rz1_sm, SM.speed_rz2_sm).pvalue:.3g}')
print()
for c, (nm, _) in CTX.items():
    rho, pv = spearmanr(S.sess_idx, S[f'frac_{c}'])
    print(f'  {nm:8s} vs session number: rho = {rho:+.3f}, p = {pv:.3g}')
rho_d, p_d = spearmanr(S.sess_idx, S.d_frac)
print(f'  novel - familiar  vs session number: rho = {rho_d:+.3f}, p = {p_d:.3g}')

# ── behaviour: where the animal first stops on uncued trials ─────────────────
# Behaviour is taken over EVERY MCVR session; the population state only over
# sessions with enough entorhinal cells. The two panels share an x-axis but not
# an n, and each panel says which it is -- forcing behaviour through the cell
# filter would throw away a third of the sessions for a measure that needs no
# spikes at all.
from mcvr_stop_learning import ensure_stops, RZ, ZONE_WIN
ST = ensure_stops()
ST = ST[ST.ttype == 'nb']
B = ALL[ALL.ttype == 'nb'].merge(
    ST[['mouse', 'day', 'trial', 'first_stop', 'n_own', 'n_other']],
    on=['mouse', 'day', 'trial'], how='inner')
U = M[M.ttype == 'nb'].copy()
print(f'\nuncued trials: {len(B)} with stops '
      f'({B.groupby(["mouse","day"]).ngroups} sessions), '
      f'{len(U)} with a population state '
      f'({U.groupby(["mouse","day"]).ngroups} sessions)')

MIN_BLOCK = 5       # uncued trials per context block
MIN_SESS = 3        # sessions a block index needs before it is plotted


def summarise(g):
    """One row per group: the two behavioural measures plus anchoring.

    `stop_ratio` is the discrimination index over ALL stops in the group,
    (own - other) / (own + other) within +- ZONE_WIN cm of each zone: +1 if
    every zone-adjacent stop is at the correct zone, -1 if every one is at the
    wrong one, 0 if they are split. It is a ratio of summed counts, not a mean
    of per-trial values, so trials with no stop near either zone drop out
    instead of being scored as ignorance -- which is what makes it survive the
    late-session disengagement that breaks every per-trial measure.
    """
    o = g.n_own.sum() if 'n_own' in g else np.nan
    t = g.n_other.sum() if 'n_other' in g else np.nan
    return pd.Series(dict(
        first_stop=g.first_stop.median() if 'first_stop' in g else np.nan,
        stop_ratio=(o - t) / (o + t) if np.isfinite(o) and (o + t) > 0 else np.nan,
        frac_anch=g.frac_anch.mean() if 'frac_anch' in g else np.nan,
        n_trials=len(g)))


def by_day(df):
    """One row per session per context."""
    return (df.groupby(['sess_idx', 'mouse', 'day', 'context'])
            .apply(summarise).reset_index())


def by_block(df):
    """One row per day-1 block per context, indexed within context.

    `k` counts blocks separately per context, so novel block 1 is the animal's
    first ever exposure to rz2. Two of the six day-1 sessions open with a short
    familiar lead-in, which a raw block index would misalign.
    """
    out = []
    for (mo, dy), g in df[df.sess_idx == 1].groupby(['mouse', 'day']):
        g = g.sort_values('trial')
        cv = g.context.values
        e = np.r_[0, np.where(cv[1:] != cv[:-1])[0] + 1, len(g)]
        seen = {c: 0 for c in CTX}
        for i in range(len(e) - 1):
            sl, c = g.iloc[e[i]:e[i + 1]], cv[e[i]]
            if c not in CTX:
                continue
            seen[c] += 1
            if len(sl) < MIN_BLOCK:
                continue
            r = dict(mouse=mo, day=dy, context=c, k=seen[c])
            r.update(summarise(sl).to_dict())
            out.append(r)
    return pd.DataFrame(out)


def kmax(blk):
    """Largest block index still carried by MIN_SESS sessions in both contexts."""
    n = blk.groupby(['context', 'k']).size().unstack(fill_value=0)
    ok = [k for k in n.columns if (n[k] >= MIN_SESS).all()]
    return max(ok) if ok else 0


BLK_B, BLK_U = by_block(B), by_block(U)
DAY_B, DAY_U = by_day(B), by_day(U)
KS = list(range(1, min(kmax(BLK_B), kmax(BLK_U)) + 1))
DAYS = sorted(set(DAY_B.sess_idx) & set(DAY_U.sess_idx))
BLK_B, BLK_U = BLK_B[BLK_B.k <= KS[-1]], BLK_U[BLK_U.k <= KS[-1]]
DAY_B = DAY_B[DAY_B.sess_idx.isin(DAYS)]
DAY_U = DAY_U[DAY_U.sess_idx.isin(DAYS)]

print(f'  day 1: {BLK_B.groupby(["mouse","day"]).ngroups} sessions, '
      f'blocks 1-{KS[-1]}; training days {DAYS[0]}-{DAYS[-1]}')
W = DAY_B.pivot_table(index=['mouse', 'day', 'sess_idx'], columns='context',
                      values='stop_ratio').reset_index().dropna()
print('\nSTOP DISCRIMINATION INDEX, uncued trials, by training day:')
print(W.groupby('sess_idx')[['rz1', 'rz2']].agg(['mean', 'size'])
      .round(3).to_string())
d1, rest = W[W.sess_idx == 1], W[W.sess_idx > 1]
p1 = wilcoxon(d1.rz1, d1.rz2).pvalue
pr = wilcoxon(rest.rz1, rest.rz2).pvalue
print(f'  day 1   familiar {d1.rz1.mean():+.3f}  novel {d1.rz2.mean():+.3f}  '
      f'p = {p1:.3f}  ({(d1.rz1 > d1.rz2).sum()}/{len(d1)} worse in novel)')
print(f'  days 2+ familiar {rest.rz1.mean():+.3f}  novel {rest.rz2.mean():+.3f}  '
      f'p = {pr:.3f}  ({(rest.rz1 > rest.rz2).sum()}/{len(rest)})')
for c, (nm, _) in CTX.items():
    b = DAY_B[DAY_B.context == c].dropna(subset=['stop_ratio'])
    r, pv = spearmanr(b.sess_idx, b.stop_ratio)
    print(f'  {nm:8s} vs day: rho = {r:+.3f}, p = {pv:.2g}   '
          f'first stop {b.groupby("sess_idx").first_stop.median().round(1).to_dict()}')


def panel(ax, df, xcol, col, ylab, xs, title=None, zones=False, zone_text=False):
    """Mean +- SEM across SESSIONS of an already-per-session quantity."""
    for c, (nm, cc) in CTX.items():
        g = df[df.context == c].dropna(subset=[col]).groupby(xcol)[col]
        mu = g.mean().reindex(xs)
        se = g.agg(lambda v: v.std(ddof=1) / np.sqrt(max(len(v), 1))).reindex(xs)
        ax.errorbar(xs, mu.values, yerr=se.values, color=cc, marker='o', ms=4,
                    lw=1.7, capsize=2.5, label=nm, zorder=3)
    if zones:
        for c, (nm, cc) in CTX.items():
            ax.axhline(RZ[c], color=cc, lw=.9, ls=':', alpha=.9, zorder=1)
            if zone_text:      # only on the right column, which has outside room
                ax.text(1.02, RZ[c], f'{nm}\nzone',
                        transform=ax.get_yaxis_transform(), fontsize=6, color=cc,
                        va='center', linespacing=.9)
    ax.set_xticks(xs)
    if ylab:
        ax.set_ylabel(ylab, fontsize=8.5)
    if title:
        ax.set_title(title, fontsize=8.5, loc='left')
    ax.tick_params(labelsize=7.5); ax.spines[['top', 'right']].set_visible(False)


fig = plt.figure(figsize=(7.3, 9.6))
gs = fig.add_gridspec(4, 2, height_ratios=[.75, 1, 1, 1], width_ratios=[1, 1.1],
                      hspace=.45, wspace=.42)

ex = S.iloc[(S.n_cells).argmax()]
d = M[(M.mouse == ex.mouse) & (M.day == ex.day)].sort_values('trial')
ax = fig.add_subplot(gs[0, :])
x = np.arange(1, len(d) + 1)
for c, (nm, col) in CTX.items():
    ax.fill_between(x, 0, 1, where=(d.context == c).values, color=col, alpha=.13,
                    linewidth=0, edgecolor='none', step='mid', label=nm)
ax.plot(x, d.frac_anch.values, color='0.15', lw=1.1)
ax.axhline(.5, color='0.6', lw=.7, ls=':')
ax.set_ylim(0, 1); ax.set_xlim(1, len(d))
ax.set_xlabel('Trial', fontsize=8.5)
ax.set_ylabel('Proportion anchored', fontsize=8.5)
ax.set_title(f'M{int(ex.mouse)} D{int(ex.day)} — {int(ex.n_cells)} cells, '
             f'training day {int(ex.sess_idx)}', fontsize=9, loc='left')
ax.legend(fontsize=7.5, frameon=False, ncol=2)
ax.tick_params(labelsize=7.5); ax.spines[['top', 'right']].set_visible(False)

ROWS = [(1, BLK_B, DAY_B, 'first_stop', 'First stop (cm)', True, (60, 128)),
        (2, BLK_B, DAY_B, 'stop_ratio',
         f'Stop discrimination\n(own − other zone, ±{ZONE_WIN:.0f} cm)',
         False, (-.35, 1.05)),
        (3, BLK_U, DAY_U, 'frac_anch', 'Proportion anchored', False, (.25, 1.0))]
LET = iter('BCDEFG')

for row, blk, day, col, ylab, zones, ylim in ROWS:
    L = fig.add_subplot(gs[row, 0])
    panel(L, blk, 'k', col, ylab, KS,
          'training day 1, by block' if row == 1 else None, zones=zones)
    R = fig.add_subplot(gs[row, 1])
    panel(R, day, 'sess_idx', col, None, DAYS,
          'across training days' if row == 1 else None, zones=zones,
          zone_text=zones)
    # Same y-limits left and right: the two columns are the same measure at two
    # timescales, so they have to be readable against each other.
    for a_ in (L, R):
        a_.set_ylim(*ylim)
        if col == 'frac_anch':
            a_.axhline(.5, color='0.6', lw=.7, ls=':', zorder=1)
        if col == 'stop_ratio':
            a_.axhline(0, color='0.6', lw=.7, ls=':', zorder=1)
    R.tick_params(labelleft=False)
    # n sits along the bottom of each panel, where nothing else competes for room
    for a_, df_, xc in ((L, blk, 'k'), (R, day, 'sess_idx')):
        for j_, xv in enumerate(KS if xc == 'k' else DAYS):
            nn = df_[(df_[xc] == xv) & df_[col].notna()].groupby(
                ['mouse', 'day']).ngroups
            a_.annotate(f'n={nn}' if j_ == 0 else f'{nn}',
                        (xv, .015), xycoords=('data', 'axes fraction'),
                        ha='center', fontsize=6, color='0.5')
    if row == 1:
        L.legend(fontsize=7, frameon=False, loc='upper left', ncol=2,
                 columnspacing=1, handletextpad=.4)
    if row == 2:
        # the day-1 novelty deficit, which is the result this figure rests on
        R.plot([1.06, 1.06], [d1.rz2.mean() + .05, d1.rz1.mean() - .05],
               color='0.3', lw=.9)
        R.annotate(f'p = {p1:.2f}', (1.13, (d1.rz1.mean() + d1.rz2.mean()) / 2),
                   fontsize=6.5, color='0.3', va='center')
    if row == 3:
        L.set_xlabel('Block number within context\n'
                     '(novel block 1 = first ever exposure)', fontsize=8)
        R.set_xlabel('Training day on the multi-context task\n'
                     '(day 1 = first ever exposure)', fontsize=8)
    for a_ in (L, R):
        a_.text(-.17 if a_ is L else -.09, 1.04, next(LET), transform=a_.transAxes,
                fontsize=11, fontweight='bold', va='bottom')
ax.text(-.055, 1.02, 'A', transform=ax.transAxes, fontsize=11,
        fontweight='bold', va='bottom')

out = f'{FIG}/mcvr_context_anchoring.pdf'
plt.savefig(out, dpi=200, bbox_inches='tight')
print(f'\nsaved {out}')
