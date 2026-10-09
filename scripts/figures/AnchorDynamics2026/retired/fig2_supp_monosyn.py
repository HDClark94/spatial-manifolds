"""Figure 2, supplement — is the anchoring state carried by a wired local network?

Usage:  python3 fig2_supp_monosyn.py

Figure 2 shows that grid cells and putative interneurons follow the population
anchoring state, and follow each other, while other spatial cells do not. That
invites a circuit explanation: perhaps the cells that share a state are the
cells that are synaptically connected. This tests it directly.

WHY THIS IS A SUPPLEMENT AND NOT A FIGURE. Five of the eight panels (A-D, H)
establish that the detector works, which is a method. The one result that
advances the main claim -- that grid cells and interneurons are wired together,
panel F -- is promoted into Figure 2 itself, next to the trial-level agreement
matrix it corroborates. What is left here is the detector, the excitatory
matrix, and a NULL (panel G), and a null closes off an alternative rather than
advancing an argument.

    A, B  example correlograms: an excitatory peak and an inhibitory trough,
          each against its hollow-Gaussian predictor
    C     the two are physiologically distinct -- inhibition is slower
    D     and come from the right cells: inhibitory connections originate from
          waveform-classified interneurons, which were never used to find them
    E, F  connection probability between identities, excitatory and inhibitory
    G     THE TEST: do connected pairs share the anchoring state more than
          unconnected pairs of the same identities at the same distance?
    H     detection validated against a jitter null, both signs

THE DETECTOR HAD TO BE EXTENDED FIRST. The published criteria look for a
short-latency PEAK, so they find excitatory connections only. Everything an
inhibitory cell does appears as a TROUGH and was invisible in principle -- which
made the interneuron half of the question unanswerable rather than merely
underpowered. `monosyn.detect(kind='inh')` mirrors every criterion with the
Poisson test on the lower tail. It finds 7,775 connections against 1,059
excitatory, at the same false-positive rate.

WHAT IS FOUND. Connectivity is strongly organised by identity, and in the
predicted places: int -> grid is the strongest pathway in either matrix (3.88%,
enriched 5.0x over within-session identity shuffling, p = 0.0005), and
grid -> grid the strongest excitatory one (0.217%, enriched 2.6x, p = 0.004).

WHAT IS NOT. Connected pairs do NOT share the anchoring state more than
unconnected pairs matched for identity and probe distance -- grid-interneuron
pairs agree at +0.175 connected against +0.184 unconnected (p = 0.46), and
interneuron-interneuron at +0.089 against +0.093 (p = 0.83).

THE NULL IS ONLY AS GOOD AS ITS POWER, so panel G reports the minimum
detectable effect beside every comparison. It is what decides whether a null
means anything, and here it splits the result:

    grid+int   n=194   MDE 0.044   vs an identity effect of ~0.11  -> informative
    int+int    n=119   MDE 0.042                                   -> informative
    grid+grid  n=27    MDE 0.157   LARGER than the effect sought   -> UNINFORMATIVE

For grid-interneuron pairs a wiring effect a third the size of the identity
effect would have been caught, and none was; that null is real. It survives the
pessimistic assumption that the matched controls contribute no more information
than the connected pairs themselves (MDE 0.060, still well under 0.11). The
grid-grid comparison is a different matter: with 27 connected pairs it could not
have detected a wiring effect as large as the ENTIRE identity effect, so its
p = 0.31 means underpowered, not absent, and it is labelled that way rather than
being counted as part of the null.

So the local network is real and the shared state is real, but the second is not
explained by the first -- for the pairs where the question can be answered.
Anchoring travels with cell identity, not with synaptic partnership. The
matching is what makes this interpretable: connected cells are close together
and of particular types, and both predict agreement on their own.
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
from scipy.stats import mannwhitneyu

sys.path.insert(0, '/Users/harryclark/Documents/spatial-manifolds/src')
sys.path.insert(0, '/Users/harryclark/Documents/spatial-manifolds/scripts/figures/'
                   'AnchorDynamics2026')
from spatial_manifolds.anchoring import (ANCH_COLOR, NONANCH_COLOR,
                                         load_session_labels)

plt.rcParams['font.family'] = 'Arial'
ROOT = '/Users/harryclark/Documents/spatial-manifolds'
FIG = f'{ROOT}/scripts/figures/AnchorDynamics2026'
MS = f'{ROOT}/data/monosyn'
PS = f'{ROOT}/data/population_state'
ORDER = ['grid', 'non-grid spatial', 'non-spatial', 'putative interneuron']
SHORT = ['grid', 'NGS', 'NS', 'int']
ICOL = {'grid': '#c04744', 'non-grid spatial': '#3171ae',
        'non-spatial': '#888888', 'putative interneuron': '#d95f02'}
EXC_C, INH_C = '#2f6f9f', '#b5406a'
DIST_TOL, N_CTRL = 40.0, 20
# the effect any null here has to be able to resolve: the identity effect from
# Figure 2 (grid agreement over chance +0.119, grid-interneuron pair excess
# +0.109). A comparison whose MDE exceeds this cannot speak to the question.
IDENTITY_EFFECT = 0.11

C = pd.concat([pd.read_csv(f) for f in glob.glob(f'{MS}/connections2_w*.csv')],
              ignore_index=True)
K = pd.concat([pd.read_csv(f) for f in glob.glob(f'{MS}/cells2_w*.csv')],
              ignore_index=True)
U = pd.read_csv(f'{PS}/unit_table.csv')
ID = {(m, d, c): i for m, d, c, i in
      zip(U.mouse, U.day, U.cluster_id, U.identity)}
C['ip'] = [ID.get((m, d, c)) for m, d, c in zip(C.mouse, C.day, C.pre)]
C['iq'] = [ID.get((m, d, c)) for m, d, c in zip(C.mouse, C.day, C.post)]
K['identity'] = [ID.get((m, d, c)) for m, d, c in zip(K.mouse, K.day, K.cluster_id)]
Ki = K.dropna(subset=['identity'])

S = K.groupby(['mouse', 'day']).agg(
    tested=('n_tested', 'first'), exc=('n_exc', 'first'), inh=('n_inh', 'first'),
    null_e=('n_null', 'first'), null_i=('n_null_inh', 'first')).reset_index()
nn = S.dropna(subset=['null_e'])
print(f'{len(C)} connections, {len(S)} sessions: '
      f'{int(S["exc"].sum())} excitatory, {int(S["inh"].sum())} inhibitory')
print(f'  jitter null ({len(nn)} sessions): exc FDR '
      f'{nn["null_e"].sum() / max(nn["exc"].sum(), 1):.2f}, inh FDR '
      f'{nn["null_i"].sum() / max(nn["inh"].sum(), 1):.2f}')

# testable ordered pairs per identity pair
POSS = {}
for (mo, dy), g in Ki.groupby(['mouse', 'day']):
    n = g.identity.value_counts()
    for a in ORDER:
        for b in ORDER:
            na, nb = n.get(a, 0), n.get(b, 0)
            POSS[(a, b)] = POSS.get((a, b), 0) + (na * (na - 1) if a == b else na * nb)


def prob_matrix(kind):
    cc = C[C.kind == kind].dropna(subset=['ip', 'iq'])
    o = cc.groupby(['ip', 'iq']).size().to_dict()
    return np.array([[100 * o.get((a, b), 0) / POSS[(a, b)] if POSS.get((a, b))
                      else np.nan for b in ORDER] for a in ORDER]), len(cc)


def shuffle_enrichment(kind, a, b, n_shuf=2000, seed=0):
    """Observed count of a->b against within-session identity shuffling.

    Identities are permuted among the cells of each session, so the number of
    cells, the number of connections and the probe geometry are all held fixed
    and only the identity labelling moves.
    """
    rng = np.random.default_rng(seed)
    cc = C[C.kind == kind]
    obs, null = 0, np.zeros(n_shuf)
    for (mo, dy), g in Ki.groupby(['mouse', 'day']):
        ids = g.cluster_id.astype(int).values
        lab = g.identity.values
        e = cc[(cc.mouse == mo) & (cc.day == dy)]
        e = e[e.pre.isin(ids) & e.post.isin(ids)]
        if not len(e):
            continue
        pos = {c: i for i, c in enumerate(ids)}
        pi = np.array([pos[int(x)] for x in e.pre])
        qi = np.array([pos[int(x)] for x in e.post])
        obs += int(((lab[pi] == a) & (lab[qi] == b)).sum())
        for s in range(n_shuf):
            sl = rng.permutation(lab)
            null[s] += ((sl[pi] == a) & (sl[qi] == b)).sum()
    p = (1 + (null >= obs).sum()) / (n_shuf + 1)
    return obs, float(null.mean()), obs / max(null.mean(), 1e-9), p


for _k, _a, _b in (('exc', 'grid', 'grid'),
                   ('inh', 'putative interneuron', 'grid')):
    _o, _e, _r, _p = shuffle_enrichment(_k, _a, _b)
    print(f'  {_k} {_a} -> {_b}: {_o} obs vs {_e:.1f} shuffled '
          f'({_r:.2f}x, p={_p:.4f})')

# ── the pair-level anchoring test, cached ───────────────────────────────────
PAIRS = f'{MS}/pairs_by_identity.csv'
if os.path.exists(PAIRS):
    P = pd.read_csv(PAIRS)
else:
    GI = {'grid', 'putative interneuron'}
    rows = []
    for (mo, dy), cells in K.groupby(['mouse', 'day']):
        z = load_session_labels(int(mo), int(dy))
        if z is None:
            continue
        L, ids = z['labels'], z['cluster_id'].astype(int)
        lab = {int(c): L[i] for i, c in enumerate(ids) if np.nanstd(L[i]) > 0}
        pos = {int(r.cluster_id): (r.px, r.py) for _, r in cells.iterrows()}
        use = [c for c in cells.cluster_id.astype(int) if c in lab]
        if len(use) < 8:
            continue
        cc = C[(C.mouse == mo) & (C.day == dy)]
        conn = {frozenset((int(r.pre), int(r.post))) for _, r in cc.iterrows()}
        for i in range(len(use)):
            for j in range(i + 1, len(use)):
                a, b = use[i], use[j]
                ia, ib = ID.get((mo, dy, a)), ID.get((mo, dy, b))
                if ia is None or ib is None:
                    continue
                x, y = lab[a], lab[b]
                m = np.isfinite(x) & np.isfinite(y)
                if m.sum() < 20 or np.std(x[m]) == 0 or np.std(y[m]) == 0:
                    continue
                p_, q_ = pos.get(a), pos.get(b)
                rows.append(dict(
                    mouse=mo, day=dy,
                    pair=('grid+int' if {ia, ib} == GI else
                          'grid+grid' if ia == ib == 'grid' else
                          'int+int' if ia == ib == 'putative interneuron' else 'other'),
                    conn=frozenset((a, b)) in conn,
                    r=float(np.corrcoef(x[m], y[m])[0, 1]),
                    d=float(np.hypot(p_[0] - q_[0], p_[1] - q_[1]))
                    if p_ and q_ else np.nan))
    P = pd.DataFrame(rows).dropna(subset=['r', 'd'])
    P.to_csv(PAIRS, index=False)

PAIR_ORDER = ['grid+grid', 'grid+int', 'int+int', 'other']
PCOL = {'grid+grid': ICOL['grid'], 'grid+int': '#7a5aa8',
        'int+int': ICOL['putative interneuron'], 'other': '0.6'}
RES = {}
for k in PAIR_ORDER:
    q = P[P.pair == k]; a = q[q.conn]
    pool = q[~q.conn]
    ctl = []
    for _, r in a.iterrows():
        cand = pool[(pool.mouse == r.mouse) & (pool.day == r.day) &
                    ((pool.d - r.d).abs() <= DIST_TOL)]
        if len(cand):
            ctl.append(cand.sample(min(N_CTRL, len(cand)), random_state=0).r.values)
    ctl = np.concatenate(ctl) if ctl else np.array([])
    # Minimum detectable effect, two-sided alpha .05 at 80% power. A null is
    # only worth reporting if it could have caught the effect being denied, and
    # the effect to beat is the IDENTITY effect of Figure 2 (~0.11) -- if a
    # comparison cannot resolve that, its p value carries no information.
    p_ = (mannwhitneyu(a.r, ctl).pvalue
          if len(a) >= 10 and len(ctl) > 10 else np.nan)
    if len(a) > 1 and len(ctl) > 1:
        sd = np.sqrt((a.r.var(ddof=1) * (len(a) - 1)
                      + ctl.var(ddof=1) * (len(ctl) - 1))
                     / (len(a) + len(ctl) - 2))
        mde = 2.80 * sd * np.sqrt(1 / len(a) + 1 / len(ctl))
    else:
        mde = np.nan
    RES[k] = (a.r.values, ctl, p_, mde)
    print(f'  {k:10s} connected {len(a):5d} r={a.r.mean():+.4f}  '
          f'control r={np.mean(ctl):+.4f}  p={p_:.2g}  MDE={mde:.3f}'
          f'{"  <-- UNDERPOWERED" if mde > IDENTITY_EFFECT else ""}')

# ── figure ───────────────────────────────────────────────────────────────────
fig = plt.figure(figsize=(10.6, 6.2))
gs = fig.add_gridspec(2, 4, hspace=.52, wspace=.46)

# A, B: example correlograms of each sign
import monosyn as MSY

top = {}
for kind in ('exc', 'inh'):
    d = C[C.kind == kind].sort_values('z', ascending=(kind == 'inh'))
    top[kind] = d.iloc[0]
SS = None
for k_, (kind, col, nm) in enumerate((('exc', EXC_C, 'excitatory: peak'),
                                      ('inh', INH_C, 'inhibitory: trough'))):
    ax = fig.add_subplot(gs[0, k_])
    t = top[kind]
    try:
        if SS is None or (SS['mo'], SS['dy']) != (int(t.mouse), int(t.day)):
            Sx = MSY.session_ccgs(int(t.mouse), int(t.day))
            SS = dict(mo=int(t.mouse), dy=int(t.day), S=Sx)
        Sx = SS['S']
        i = int(np.where(Sx['ids'] == int(t.pre))[0][0])
        j = int(np.where(Sx['ids'] == int(t.post))[0][0])
        lg = Sx['lags'] * 1000
        w = np.abs(lg) <= 25
        ax.bar(lg[w], Sx['raw'][i, j][w], width=1.0, color='0.35', lw=0, zorder=2)
        ax.plot(lg[w], Sx['base'][i, j][w], color=col, lw=1.5, zorder=3)
        ax.axvspan(0.7, 4.7, color=col, alpha=.13, lw=0, zorder=0)
        ax.set_xlim(-25, 25)
        v = Sx['raw'][i, j][w]
        ax.set_ylim(max(0, v.min() * .88), v.max() * 1.04)
        ax.set_title(f'{"AB"[k_]}  {nm}\nM{int(t.mouse)} D{int(t.day)}: '
                     f'{int(t.pre)} → {int(t.post)}', fontsize=7.5, loc='left')
    except Exception as e:
        ax.text(.5, .5, f'unavailable\n{type(e).__name__}', ha='center', va='center',
                transform=ax.transAxes, fontsize=7, color='0.5')
        ax.set_title(f'{"AB"[k_]}  {nm}', fontsize=7.5, loc='left')
    ax.set_xlabel('Lag (ms)', fontsize=8)
    ax.set_ylabel('Spike count', fontsize=8.5)
    ax.tick_params(labelsize=7); ax.spines[['top', 'right']].set_visible(False)

# C: lag distributions
ax = fig.add_subplot(gs[0, 2])
for kind, col, nm in (('exc', EXC_C, 'excitatory'), ('inh', INH_C, 'inhibitory')):
    v = C[C.kind == kind].lag_ms
    ax.hist(v, bins=np.arange(.5, 5.5, .5), histtype='step', lw=1.6, color=col,
            density=True, label=f'{nm} (n={len(v)})')
ax.set_xlabel('Latency (ms)', fontsize=8)
ax.set_ylabel('Density', fontsize=8.5)
ax.set_title('C  inhibition is slower', fontsize=8, loc='left')
ax.legend(fontsize=6, frameon=False)
ax.tick_params(labelsize=7); ax.spines[['top', 'right']].set_visible(False)

# D: who makes inhibitory connections?
ax = fig.add_subplot(gs[0, 3])
base = Ki.identity.value_counts(normalize=True)
for k_, (kind, col) in enumerate((('exc', EXC_C), ('inh', INH_C))):
    pre = C[C.kind == kind].dropna(subset=['ip']).ip.value_counts(normalize=True)
    en = [pre.get(a, 0) / base.get(a, 1) for a in ORDER]
    ax.bar(np.arange(4) + (k_ - .5) * .38, en, width=.38, color=col, lw=0,
           label='excitatory' if kind == 'exc' else 'inhibitory')
ax.axhline(1, color='k', lw=1, ls='--')
ax.set_ylim(0, max(ax.get_ylim()[1], 1.2) * 1.28)
ax.set_xticks(range(4)); ax.set_xticklabels(SHORT, fontsize=7)
ax.set_ylabel('Presynaptic enrichment', fontsize=8)
ax.set_title('D  inhibition comes from\ninterneurons', fontsize=7.5, loc='left')
ax.legend(fontsize=6, frameon=False)
ax.tick_params(labelsize=7); ax.spines[['top', 'right']].set_visible(False)

# E, F: connection probability matrices
for k_, (kind, nm) in enumerate((('exc', 'E  excitatory'), ('inh', 'F  inhibitory'))):
    ax = fig.add_subplot(gs[1, k_])
    Mx, n = prob_matrix(kind)
    im = ax.imshow(Mx, cmap='magma', vmin=0, vmax=np.nanmax(Mx))
    cm = plt.get_cmap('magma')
    for i in range(4):
        for j in range(4):
            if not np.isfinite(Mx[i, j]):
                continue
            rgb = cm(Mx[i, j] / np.nanmax(Mx))[:3]
            lum = .299 * rgb[0] + .587 * rgb[1] + .114 * rgb[2]
            ax.text(j, i, f'{Mx[i, j]:.2f}', ha='center', va='center', fontsize=5.8,
                    color='0.1' if lum > .55 else 'white')
    ax.set_xticks(range(4)); ax.set_xticklabels(SHORT, fontsize=6.5)
    ax.set_yticks(range(4)); ax.set_yticklabels(SHORT, fontsize=6.5)
    for t_, a in zip(ax.get_xticklabels(), ORDER):
        t_.set_color(ICOL[a])
    for t_, a in zip(ax.get_yticklabels(), ORDER):
        t_.set_color(ICOL[a])
    ax.set_xlabel('postsynaptic', fontsize=7.5)
    ax.set_ylabel('presynaptic', fontsize=7.5)
    ax.set_title(f'{nm} (% of pairs, n={n})', fontsize=7.5, loc='left')
    ax.tick_params(length=0, labelsize=6.5)

# G: the test
ax = fig.add_subplot(gs[1, 2])
rng = np.random.default_rng(0)
for i, k in enumerate(PAIR_ORDER):
    a, ctl, p, mde = RES[k]
    weak = np.isfinite(mde) and mde > IDENTITY_EFFECT
    for off, v, c in ((-.19, ctl, '0.72'), (.19, a, PCOL[k])):
        if not len(v):
            continue
        ax.errorbar(i + off, v.mean(), yerr=v.std(ddof=1) / np.sqrt(len(v)),
                    color=c, marker='o', ms=4.5, lw=1.4, capsize=3, zorder=3,
                    alpha=.42 if weak else 1.0)
    # an underpowered comparison is not a null and is not labelled as one
    lab = ('underpowered' if weak else
           'n.s.' if p > .05 else f'{p:.2g}') if np.isfinite(p) else ''
    ax.annotate(lab, (i, .99), xycoords=('data', 'axes fraction'), ha='center',
                va='top', fontsize=6, color='#a33' if weak else '0.3')
    ax.annotate(f'n={len(a)}\nMDE {mde:.2f}', (i, .92),
                xycoords=('data', 'axes fraction'), ha='center', va='top',
                fontsize=5.4, color='#a33' if weak else '0.45', linespacing=1.35)
ax.set_xticks(range(4))
ax.set_xticklabels(['grid\n+grid', 'grid\n+int', 'int\n+int', 'other'], fontsize=6.5)
ax.set_xlim(-.5, 3.5)
ax.set_ylabel('Anchoring agreement (r)', fontsize=8)
ax.set_title('G  open = matched unconnected,\nfilled = connected', fontsize=7.5,
             loc='left')
ax.set_ylim(top=ax.get_ylim()[1] * 1.14)   # headroom for the power labels
ax.tick_params(labelsize=7); ax.spines[['top', 'right']].set_visible(False)

# H: jitter null
ax = fig.add_subplot(gs[1, 3])
for k_, (kind, col, ce, cn) in enumerate((('exc', EXC_C, 'exc', 'null_e'),
                                          ('inh', INH_C, 'inh', 'null_i'))):
    ax.scatter(nn[ce], nn[cn], s=20, color=col, lw=.4, edgecolor='w', zorder=3,
               label=f'{kind} (FDR {nn[cn].sum()/max(nn[ce].sum(),1):.2f})')
mx = max(nn['inh'].max(), 1) * 1.1
ax.plot([0, mx], [0, mx], color='0.6', lw=.8, ls=':')
ax.set_xscale('symlog'); ax.set_yscale('symlog')
ax.set_xlabel('connections detected', fontsize=8)
ax.set_ylabel('after ±10 ms jitter', fontsize=8)
ax.set_title(f'H  both signs validate\n({len(nn)} sessions)', fontsize=7.5, loc='left')
ax.legend(fontsize=6, frameon=False, loc='upper left')
ax.tick_params(labelsize=7); ax.spines[['top', 'right']].set_visible(False)

out = f'{FIG}/fig2_supp_monosyn.pdf'
plt.savefig(out, dpi=200, bbox_inches='tight')
print(f'\nsaved {out}')
