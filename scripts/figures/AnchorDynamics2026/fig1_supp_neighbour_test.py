"""Figure S4: what the task does to behaviour across a session, and whether the
anchoring state lasts, behind the test of Figure 2. Cued trials mark the reward
zone, so the cue, not the mouse's own sense of where it is, sets where it stops;
uncued trials leave that to the mouse. Numbers from neighbour_test.py, labels
from build_neighbour_labels.py.

A     the test of Figure 2H on one session: each trial's type and outcome, and
      the unfiltered state of two structures from both classifiers; an uncued
      trial is predicted from the nearest cued trial before it
B     how performance moves over the session: hit rate on cued and uncued
      trials in the example session, the hit rate of each per session, and how
      much it drifts from block to block beyond a trial-shuffle null
C     whether each structure's state lasts from trial to trial, from each
      classifier: autocorrelation over a trial-order permutation
D     what the neighbours' own behaviour says about the uncued outcome, the
      reason it is regressed out in Figure 2H
E     per structure, across sessions: how far the black-box state follows the
      whole-track state, against the whole-track state's persistence; ringed,
      the sessions of Figure 2D-F
"""
import os, sys
import numpy as np, pandas as pd
import matplotlib; matplotlib.use('Agg')
import matplotlib.pyplot as plt
from scipy.stats import wilcoxon, spearmanr
sys.path.insert(0, '/Users/harryclark/Documents/spatial-manifolds/src')
sys.path.insert(0, '/Users/harryclark/Documents/spatial-manifolds/scripts/figures/AnchorDynamics2026')
from neighbour_test import load, neighbours, NT, PS, REGIONS, SPEED_PROF

plt.rcParams['font.family'] = 'Arial'
FIG = '/Users/harryclark/Documents/spatial-manifolds/scripts/figures/AnchorDynamics2026'
RCOL = {'MEC': '#7b4173', 'VIS': '#3f9b4f', 'SUB': '#c0723a', 'CB': '#4c72b0'}
R = pd.read_csv(f'{PS}/neighbour_test.csv')
P = pd.read_csv(f'{PS}/neighbour_persistence.csv')
EX, EX_TR, EX_TARGET = (21, 22), (40, 72), 55      # example session, trial window, target
NAME = {'full': 'whole track', 'tz': 'task region', 'bb': 'black box'}
TZ_C, BB_C = '#e6c86e', '0.25'                       # the two stretches of track
TGT_C, NBR_C = '0.62', '0.87'                        # target and neighbour trials in A
CUED_C, UNCUED_C = '#3f9b4f', '#c0398d'              # trial types, as Figure 1
BLOCK = 10                                           # trials of one type per block in B


def lp(ax, s, x=-.16, y=1.04):
    ax.text(x, y, s, transform=ax.transAxes, fontsize=10, weight='bold', va='bottom', ha='right')


def stars(p):
    return 'n.s.' if p > .05 else ('*' if p > .01 else ('**' if p > .001 else '***'))


def clean(ax):
    ax.tick_params(labelsize=7); ax.spines[['top', 'right']].set_visible(False)


fig = plt.figure(figsize=(7.4, 8.9))
gs = fig.add_gridspec(4, 3, height_ratios=[1.25, 1, 1, .9], hspace=.75, wspace=.45)

# ── A: the test on one session ──────────────────────────────────────────────
g = gs[0, :].subgridspec(2, 1, height_ratios=[.55, 1], hspace=.08)
mo, dy, beh, st = load(f'M{EX[0]}D{EX[1]}.npz')
tt, hit = beh.ttype.values, beh.hit.values
i0, i1 = EX_TR
x = np.arange(i0, i1)
nb = neighbours(tt)
tgt = EX_TARGET if tt[EX_TARGET] == 'nb' else next(i for i in range(EX_TARGET, i1) if tt[i] == 'nb')
ax = fig.add_subplot(g[0])
for i in x:
    y_, c = (1, CUED_C) if tt[i] == 'b' else (0, UNCUED_C)
    ax.scatter(i, y_, marker='o', s=22, color=c if hit[i] else 'white', edgecolor=c, lw=.8)
for j in nb[tgt]:
    ax.axvspan(j - .45, j + .45, color=NBR_C, lw=0)
ax.axvspan(tgt - .45, tgt + .45, color=TGT_C, lw=0)
for j in nb[tgt]:
    ax.annotate('', xy=(tgt, 1.55), xytext=(j, 1.55),
                arrowprops=dict(arrowstyle='->', color='0.25', lw=.8,
                                connectionstyle='arc3,rad=-.35'))
ax.set_yticks([0, 1]); ax.set_yticklabels(['uncued', 'cued'], fontsize=6.5)
for t_, c in zip(ax.get_yticklabels(), (UNCUED_C, CUED_C)):
    t_.set_color(c)
ax.text(i1 - .2, 1.55, 'filled, hit; open, miss', ha='right', fontsize=6, color='0.3')
ax.set_ylim(-.6, 2.1); ax.set_xlim(i0 - .8, i1 - .2)
ax.set_xticks([]); ax.spines[['top', 'right', 'bottom']].set_visible(False)
ax.tick_params(labelsize=6.5, length=0)
ax.set_title(f'one session (M{EX[0]} D{EX[1]}): an uncued trial (dark) is predicted from the state on the '
             'nearest cued trial before it (light)', fontsize=7.5, loc='left', pad=10)
lp(ax, 'A', x=-.07, y=1.0)
ax = fig.add_subplot(g[1])
for r in ('MEC', 'CB'):
    for v, ls in (('tz', '-'), ('bb', ':')):
        if (r, v) in st:
            ax.plot(x, st[(r, v)][i0:i1], color=RCOL[r], lw=1.2 if v == 'tz' else 1, ls=ls,
                    label=f'{r}, {NAME[v]}')
for j in nb[tgt]:
    ax.axvspan(j - .45, j + .45, color=NBR_C, lw=0)
ax.axvspan(tgt - .45, tgt + .45, color=TGT_C, lw=0)
ax.set_xlim(i0 - .8, i1 - .2); ax.set_xlabel('trial', fontsize=7.5)
ax.set_ylabel('state, PC1 (z)', fontsize=7.5)
ax.legend(fontsize=6, frameon=False, ncol=4, loc='upper left', bbox_to_anchor=(0, 1.02))
ax.set_ylim(-3.2, 2.6)
clean(ax)

# ── B: how performance moves over the session ──────────────────────────────
# block hit rate (BLOCK trials of one type per block) and its SD across blocks,
# against the SD with that type's outcomes shuffled in order: the excess is how
# much performance drifts beyond what its own hit rate gives by chance
def drift(h, rng, n=500):
    nb = len(h) // BLOCK
    if nb < 3:
        return np.nan
    sd = lambda y: y[:nb * BLOCK].reshape(nb, BLOCK).mean(1).std()
    return sd(h) - np.mean([sd(rng.permutation(h)) for _ in range(n)])


rng = np.random.default_rng(0)
rows_ = []
for f in sorted(f for f in os.listdir(NT) if f.endswith('.npz')):
    z = np.load(f'{NT}/{f}', allow_pickle=True)
    if 'ttype' not in z.files:
        continue
    h_, t_ = z['hit'].astype(float), z['ttype']
    rows_.append({f'{k}_{t}': v for t in ('b', 'nb')
                  for k, v in (('rate', h_[t_ == t].mean()), ('drift', drift(h_[t_ == t], rng)))})
DR = pd.DataFrame(rows_).dropna()
gb = gs[1, :].subgridspec(1, 3, width_ratios=[2.2, .8, .8], wspace=.55)
ax = fig.add_subplot(gb[0])
for t_, c, nm in (('b', CUED_C, 'cued'), ('nb', UNCUED_C, 'uncued')):
    i_ = np.flatnonzero(tt == t_)
    k_ = np.convolve(hit[i_].astype(float), np.ones(BLOCK) / BLOCK, mode='valid')
    ax.plot(i_[BLOCK // 2:BLOCK // 2 + len(k_)], k_, color=c, lw=1.3, label=nm)
ax.axvspan(i0, i1, color='0.92', lw=0, zorder=0)
ax.text((i0 + i1) / 2, 1.02, 'A', ha='center', va='bottom', fontsize=6, color='0.4')
ax.set_ylim(0, 1.1); ax.set_xlim(0, len(tt))
ax.set_xlabel('trial', fontsize=7.5); ax.set_ylabel(f'hit rate\n({BLOCK} trials of the type)', fontsize=7.5)
ax.legend(fontsize=6, frameon=False, loc='lower right', ncol=2)
ax.set_title(f'performance over the session (M{EX[0]} D{EX[1]})', fontsize=7.5, loc='left')
lp(ax, 'B', x=-.12); clean(ax)
for j, (k, yl, ttl) in enumerate((('rate', 'hit rate', 'hit rate'),
                                  ('drift', 'block SD over shuffle', 'drift beyond chance'))):
    ax = fig.add_subplot(gb[1 + j])
    a_, n_ = DR[f'{k}_b'], DR[f'{k}_nb']
    for xx, v, c in ((0, a_, CUED_C), (1, n_, UNCUED_C)):
        ax.bar(xx, v.mean(), width=.6, color=c, alpha=.75, lw=0)
        ax.errorbar(xx, v.mean(), yerr=v.sem(), color='k', lw=1, capsize=2)
        ax.scatter(xx + rng.uniform(-.15, .15, len(v)), v, s=3, color='k', alpha=.35, lw=0, zorder=3)
    p_ = wilcoxon(a_, n_).pvalue
    top_ = max(a_.max(), n_.max())
    ax.plot([0, 0, 1, 1], [top_ * 1.04, top_ * 1.08, top_ * 1.08, top_ * 1.04], color='0.3', lw=.8)
    ax.text(.5, top_ * 1.1, stars(p_), ha='center', fontsize=6.5)
    if k == 'drift':
        ax.axhline(0, color='0.6', lw=.7, ls=':')
        for xx, v in ((0, a_), (1, n_)):
            print(f'  drift {"cued" if xx == 0 else "uncued"}: {v.mean():+.3f} over shuffle, '
                  f'p = {wilcoxon(v).pvalue:.2g}')
    print(f'  {k}: cued {a_.mean():.3f}, uncued {n_.mean():.3f}, paired p = {p_:.2g}, {len(DR)} sessions')
    ax.set_xticks([0, 1]); ax.set_xticklabels(['cued', 'uncued'], fontsize=6.5)
    ax.set_ylabel(yl, fontsize=7); ax.set_title(ttl, fontsize=7.5, loc='left')
    clean(ax)


# ── C: persistence of the state ────────────────────────────────────────────
ax = fig.add_subplot(gs[2, 0:2])
Q = P.assign(x=P.ac - P.ac_null)
for r in REGIONS:
    for v, ls in (('tz', '-'), ('bb', ':')):
        q = Q[(Q.region == r) & (Q.labels == v)].groupby('lag').x
        ax.errorbar(q.mean().index + (.08 if v == 'bb' else 0), q.mean(), yerr=q.sem(),
                    color=RCOL[r], ls=ls, lw=1.1, capsize=0, marker='o' if v == 'tz' else None, ms=2.5)
ax.axhline(0, color='0.6', lw=.7, ls=':')
ax.set_xlabel('lag (trials)', fontsize=7.5)
ax.set_ylabel('autocorrelation\nover permuted', fontsize=7.5)
ax.set_title('does the state last?\n(solid task region, dotted black box)', fontsize=7.5, loc='left')
for k, r in enumerate(REGIONS):
    ax.text(1.0, .97 - k * .09, r, transform=ax.transAxes, color=RCOL[r], fontsize=6.5, ha='right', va='top')
lp(ax, 'C', x=-.25); clean(ax)

# ── D: the neighbours' own behaviour ───────────────────────────────────────
ax = fig.add_subplot(gs[2, 2])
S1 = R.drop_duplicates(['mouse', 'day'])
labs = [('beh_hit', 'hit'), ('beh_speed', 'speed'), ('beh_speed_pre', 'speed\npre-zone'),
        ('beh_speed_rz', 'speed\nin zone')]
rng = np.random.default_rng(0)
for k, (c, l) in enumerate(labs):
    v = S1[f'{c}_x'].dropna()
    ax.bar(k, v.mean(), width=.6, color='0.75', lw=0)
    ax.errorbar(k, v.mean(), yerr=v.sem(), color='k', lw=1, capsize=2)
    ax.scatter(k + rng.uniform(-.15, .15, len(v)), v, s=3, color='0.3', alpha=.5, lw=0)
    ax.text(k, .33, stars(wilcoxon(v).pvalue), ha='center', fontsize=6.5)
ax.axhline(0, color='0.6', lw=.7, ls=':')
ax.set_xticks(range(len(labs))); ax.set_xticklabels([l for _, l in labs], fontsize=6.3)
ax.set_ylim(-.4, .38)
ax.set_ylabel('AUC over null', fontsize=7.5)
ax.set_title(f"neighbours' behaviour predicts\nthe outcome ({len(S1)} sessions)", fontsize=7.5, loc='left')
lp(ax, 'D', x=-.25); clean(ax)


# ── E: does the black box recover the state? ─────────────────────────────
# per session and structure, the correlation of the black-box state (fraction of
# cells anchored) with the whole-track state, against how persistent the
# whole-track state is from one trial to the next
SES = pd.read_csv(f'{PS}/neighbour_sessions.csv')
RNAME = {'MEC': 'MEC', 'VIS': 'visual cortex', 'SUB': 'subicular complex', 'CB': 'cerebellum'}
EXS = {'MEC': ((20, 23), 'D'), 'SUB': ((28, 25), 'E'), 'VIS': ((26, 12), 'F')}   # Figure 2D-F
ge = gs[3, :].subgridspec(1, 4, wspace=.4)
for j, r in enumerate(('MEC', 'SUB', 'VIS', 'CB')):
    ax = fig.add_subplot(ge[j])
    S_ = SES[SES.region == r]
    if r == 'MEC':
        sc_ = ax.scatter(S_.ac_full, S_.r_bb, c=S_.n_gc, cmap='Greys', vmin=-10,
                         vmax=S_.n_gc.quantile(.95), s=10, edgecolor='0.2', lw=.4, zorder=3)
        cax = ax.inset_axes([.88, .08, .04, .4])
        cb = fig.colorbar(sc_, cax=cax)
        cb.ax.set_ylim(0, S_.n_gc.quantile(.95)); cb.set_ticks([0, 20])
        cb.ax.set_title('grid\ncells', fontsize=5.3, pad=2); cb.ax.tick_params(labelsize=5, length=2)
    else:
        ax.scatter(S_.ac_full, S_.r_bb, s=10, color='0.45', edgecolor='0.2', lw=.4, zorder=3)
    if r in EXS:
        (mo_, dy_), l_ = EXS[r]
        e = S_[(S_.mouse == mo_) & (S_.day == dy_)]
        ax.scatter(e.ac_full, e.r_bb, s=45, facecolor='none', edgecolor=TZ_C, lw=1.3, zorder=4)
        ax.annotate(f'2{l_}', (e.ac_full.iloc[0], e.r_bb.iloc[0]), xytext=(-13, 3),
                    textcoords='offset points', fontsize=6, color='#a07c10')
    rho, p = spearmanr(S_.ac_full, S_.r_bb)
    ax.axhline(0, color='0.6', lw=.6, ls=':')
    ax.set_xlim(-.15, 1.0); ax.set_ylim(-.35, .9)
    ax.set_xlabel('whole-track persistence\n(lag 1)', fontsize=6.5)
    if j == 0:
        ax.set_ylabel('black box vs\nwhole track (r)', fontsize=7)
        lp(ax, 'E', x=-.32, y=1.06)
    ax.set_title(f'{RNAME[r]} ({len(S_)})\nrho = {rho:.2f}, p = {p:.1g}', fontsize=6.5, loc='left')
    clean(ax)
    print(f'  {r}: black box vs whole track, rho with persistence {rho:.2f} (p={p:.1g}, n={len(S_)})')

out = f'{FIG}/fig1_supp_neighbour_test.pdf'
plt.savefig(out, dpi=200, bbox_inches='tight')
print(f'saved {out}')
