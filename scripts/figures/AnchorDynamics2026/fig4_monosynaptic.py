"""Figure 4 — the local circuit: wired together, but the wiring does not carry
the state.

Usage:  python3 fig4_monosynaptic.py

This dataset carries no interneuron label, so one has to be derived from the
waveform. The main figure leans on that group heavily -- interneurons are one of
the two identities that follow the population state (2D), and they pair with
grid cells in the identity-agreement
pairing (2G) -- so the group has to be shown to be real, not merely a threshold
applied to a histogram.

    A   the call itself: peak-to-trough against firing rate, with the marginal
        distributions and both thresholds
    B   what a classified interneuron does to its target: a causal TROUGH
    C   what a broad-spiking cell does: a causal PEAK, same scale
    D   THE VALIDATION: inhibitory connections originate from the cells panel A
        called interneurons, 3.1x over their share of the population
    E,F both connection matrices, as a percentage of the ordered pairs each
        session could actually test
    G   the two connection types are kinetically distinct -- inhibition is slower
    H   detection validated against a jitter null, both signs
    I   and the question the connectivity raises: does the wiring carry the
        anchoring state? It does not.

COLUMNS 2 AND 3 ARE ONE SIGN EACH, top to bottom: the inhibitory example (B)
sits above the inhibitory matrix (E), the excitatory example (C) above the
excitatory matrix (F). So each column reads as a single sign from one pair to
the whole population, and the two columns are the comparison.

THE THREAD. A is a classification and nothing more -- two thresholds on two
waveform statistics. B-E are an independent physiological test of it, and the
test is severe: the detector works on spike-train cross-correlograms and has no
access to waveform shape, firing rate, or the class labels. If the group in A
were an arbitrary cut, there would be no reason for its members to be the cells
that suppress their targets. They are, by a factor of three, and with the slower
kinetics expected of GABA_A rather than AMPA. That is what licenses calling them
interneurons.

Once the connections are established they answer a second question for free.
They show grid cells and interneurons are WIRED together (E), which is why the
trial-level pairing in Figure 2G is a subnetwork rather than two populations
tracking one signal -- and then I shows the shared anchoring state does NOT
travel through those synapses.

BOTH MATRICES ARE SHOWN because neither is interpretable alone: the excitatory matrix is an order of
magnitude sparser throughout, and without the inhibitory one beside it there is
no way to see that this is a property of the SIGN rather than of the identities.
Each is scaled to its own maximum for the same reason -- a shared scale renders
the excitatory matrix uniformly black.

WHY THE SPEED-SCORE VALIDATION WAS DROPPED. It used to sit here: putative
interneurons are the most speed-modulated group in the open field, p ~ 1e-50.
True, but weak evidence -- speed modulation is common in MEC and not specific to
inhibition, so the test could not have failed informatively. The monosynaptic
test can: if these cells were not inhibitory, panel D would sit at 1.0.

WHAT THE THRESHOLDS ARE. Peak-to-trough does the real work, cut at a FIXED
0.4 ms. That is a convention rather than a fitted quantity, and deliberately so:
an antimode refitted per dataset moves with whatever cells are supplied, so
sessions, mice and reanalyses would each get a slightly different boundary and
cells would change class for reasons having nothing to do with the cells. The
unsupervised fit is kept as corroboration and drawn beside it -- components 0.25
and 0.69 ms separated by 4.0 SD, antimode at 0.42 ms, within one histogram bin
of 0.4 -- and the empirical density minimum sits at 0.38-0.40 ms, so 0.4 is if
anything the better cut as well as the conventional one. The choice is also
nearly inconsequential here: 0.423 against 0.400 moves 12 cells of 842.

Firing rate is NOT fitted that way either, because log rate separates at only
1.8 SD and has no usable antimode; the top quartile is a second requirement laid
on top of the waveform call. It is a conservative one: it rejects 43% of
narrow-spiking cells, which fall back to their open-field class.

THE DETECTOR HAD TO BE EXTENDED TO SEE ANY OF THIS. The published criteria
detect a causal PEAK and therefore excitation only; everything an inhibitory
cell does appears as a TROUGH and was invisible in principle, which would have
made panel D unanswerable rather than merely underpowered.
`monosyn.detect(kind='inh')` mirrors every criterion onto the lower tail of the
same Poisson test, so both signs are held to identical thresholds and each
carries its own jitter null (H).

THE NULL IN I IS ONLY AS GOOD AS ITS POWER, so the minimum detectable effect is
printed beside every comparison. The effect to beat is the identity effect of
Figure 2, ~0.11. grid+int (n=193, MDE 0.045) and int+int (n=116, MDE 0.043)
resolve effects a third that size, so those nulls are informative. grid+grid
(n=27, MDE 0.157) cannot resolve an effect as large as the one being tested for,
so it is greyed and labelled underpowered rather than counted as a null.

COLOUR CARRIES TWO SEPARATE THINGS in this figure and they must not be
confusable. Cell IDENTITY is the red/blue/grey/orange palette used throughout
the paper; connection SIGN is teal and deep purple, which that palette does not
use, and which differ in lightness so the pairing survives greyscale.
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
from sklearn.mixture import GaussianMixture

sys.path.insert(0, '/Users/harryclark/Documents/spatial-manifolds/src')
sys.path.insert(0, '/Users/harryclark/Documents/spatial-manifolds/scripts/figures/'
                   'AnchorDynamics2026')
from spatial_manifolds.anchoring import load_session_labels

plt.rcParams['font.family'] = 'Arial'
# panel letters are set as mathtext \bf inside the axes titles, so mathtext
# must resolve to Arial Bold rather than the default DejaVu
plt.rcParams['mathtext.fontset'] = 'custom'
plt.rcParams['mathtext.rm'] = 'Arial'
plt.rcParams['mathtext.it'] = 'Arial:italic'
plt.rcParams['mathtext.bf'] = 'Arial:bold'


def _lp(ax, s, dx=-.16, dy=1.0):
    """Bold panel letter, drawn separately from the title (see fig2_single_units.py
    for why: inline mathtext or a plain leading letter ties the letter's size to
    that panel's own title fontsize, which varies panel to panel)."""
    ax.text(dx, dy, s, transform=ax.transAxes, fontsize=10, weight='bold',
            va='bottom', ha='right')
ROOT = '/Users/harryclark/Documents/spatial-manifolds'
FIG = f'{ROOT}/scripts/figures/AnchorDynamics2026'
MS = f'{ROOT}/data/monosyn'
PS = f'{ROOT}/data/population_state'
ORDER = ['grid', 'non-grid spatial', 'non-spatial', 'putative interneuron']
SHORT = ['grid', 'NGS', 'NS', 'int']
ICOL = {'grid': '#c04744', 'non-grid spatial': '#3171ae',
        'non-spatial': '#888888', 'putative interneuron': '#d95f02'}
INT_C, OTH_C = '#d95f02', '0.62'
# Connection SIGN, which must not be confusable with cell IDENTITY: the identity
# palette is red / blue / grey / orange, so the signs take teal and deep purple.
# They also differ in lightness, so the pairing survives greyscale printing.
EXC_C, INH_C = '#1b998b', '#6d2e9e'
DIST_TOL, N_CTRL = 40.0, 20
# the effect any null in H has to resolve: the identity effect of Figure 2
IDENTITY_EFFECT = 0.11


def mixture(v, n=2):
    """Two-component fit on the log of a positive quantity, plus its antimode."""
    x = np.log(v[np.isfinite(v) & (v > 0)]).reshape(-1, 1)
    gm = GaussianMixture(n, n_init=10, random_state=0).fit(x)
    mu = gm.means_.ravel(); sd = np.sqrt(gm.covariances_.ravel())
    o = np.argsort(mu); mu, sd = mu[o], sd[o]
    grid = np.linspace(mu[0], mu[1], 2000).reshape(-1, 1)
    thr = float(np.exp(grid[np.argmin(gm.score_samples(grid))][0]))
    sep = float((mu[1] - mu[0]) / np.sqrt(np.mean(sd ** 2)))
    return np.exp(mu), sep, thr, gm


# ── the classification ───────────────────────────────────────────────────────
U = pd.read_csv(f'{PS}/unit_table.csv')
W = U.dropna(subset=['peak_to_trough_duration', 'firing_rate'])
ptt = W.peak_to_trough_duration.values * 1000        # ms
rate = W.firing_rate.values
mu_w, sep_w, anti_w, _ = mixture(ptt)
mu_r, sep_r, _, _ = mixture(rate)
# The boundary is a FIXED 0.4 ms, not the fitted antimode. The fit is kept as
# corroboration -- it lands at 0.42 ms, within a histogram bin of 0.4, and the
# empirical density minimum is at 0.38-0.40 -- but a refitted cut moves with
# whatever subset of cells is supplied, so cells would change class for reasons
# unrelated to the cells. Taken from the unit table so the figure and the
# classification cannot disagree.
thr_w = float(U.wave_threshold_s.dropna().iloc[0]) * 1000     # ms
rate_q = float(U.rate_threshold_hz.dropna().iloc[0])
narrow, fast = ptt < thr_w, rate >= rate_q
put = narrow & fast
print(f'{len(W)} cells with a waveform')
print(f'  peak-to-trough: components {mu_w.round(3)} ms, separation {sep_w:.2f} SD; '
      f'threshold {thr_w:.2f} ms used (fitted antimode {anti_w:.3f} ms)')
print(f'  firing rate   : components {mu_r.round(2)} Hz, separation {sep_r:.2f} SD '
      f'(not bimodal; top quartile {rate_q:.1f} Hz used instead)')
print(f'  narrow {narrow.sum()}, fast {fast.sum()}, both {int(put.sum())}; '
      f'rate gate rejects {int((narrow & ~fast).sum())} narrow cells')

# ── the connections ──────────────────────────────────────────────────────────
C = pd.concat([pd.read_csv(f) for f in glob.glob(f'{MS}/connections2_w*.csv')],
              ignore_index=True)
K = pd.concat([pd.read_csv(f) for f in glob.glob(f'{MS}/cells2_w*.csv')],
              ignore_index=True)
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

    Identities are permuted among the cells of each session, holding cell
    counts, connection counts and probe geometry fixed so that only the
    labelling moves.
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
    p_ = (mannwhitneyu(a.r, ctl).pvalue
          if len(a) >= 10 and len(ctl) > 10 else np.nan)
    # Minimum detectable effect, two-sided alpha .05 at 80% power. A null is
    # only worth reporting if it could have caught the effect being denied.
    if len(a) > 1 and len(ctl) > 1:
        sd = np.sqrt((a.r.var(ddof=1) * (len(a) - 1) + ctl.var(ddof=1) * (len(ctl) - 1))
                     / (len(a) + len(ctl) - 2))
        mde = 2.80 * sd * np.sqrt(1 / len(a) + 1 / len(ctl))
    else:
        mde = np.nan
    RES[k] = (a.r.values, ctl, p_, mde)
    print(f'  {k:10s} connected {len(a):5d} r={a.r.mean():+.4f}  '
          f'control r={np.mean(ctl):+.4f}  p={p_:.2g}  MDE={mde:.3f}'
          f'{"  <-- UNDERPOWERED" if mde > IDENTITY_EFFECT else ""}')

# ── figure ───────────────────────────────────────────────────────────────────
# Three rows of three, with column 1 wider to give the joint scatter room for
# its marginals. COLUMNS 2 AND 3 ARE SIGN-COHERENT: the inhibitory example sits
# above the inhibitory matrix, the excitatory example above the excitatory one,
# so each column reads as one sign from example to population. That is why the
# matrices appear here in the order inhibitory-then-excitatory, which is the
# reverse of the order they are computed in.
fig = plt.figure(figsize=(9.2, 8.2))
G = fig.add_gridspec(3, 3, width_ratios=[1.32, 1, 1], hspace=.52, wspace=.46)
gs = {'A': G[0, 0], 'B': G[0, 1], 'C': G[0, 2],      # call, inh example, exc example
      'D': G[1, 0], 'E': G[1, 1], 'F': G[1, 2],      # enrichment, inh matrix, exc matrix
      'G': G[2, 0], 'H': G[2, 1], 'I': G[2, 2]}      # latency, jitter null, the test

# A: the call, as a joint scatter with marginals
ga = gs['A'].subgridspec(2, 2, width_ratios=[1, .26], height_ratios=[.26, 1],
                          hspace=.06, wspace=.06)
axs = fig.add_subplot(ga[1, 0])
axt = fig.add_subplot(ga[0, 0], sharex=axs)
axr = fig.add_subplot(ga[1, 1], sharey=axs)
lr = np.log10(rate)
axs.scatter(ptt[~put], lr[~put], s=2.6, color=OTH_C, alpha=.30, lw=0, zorder=2)
axs.scatter(ptt[put], lr[put], s=3.4, color=INT_C, alpha=.75, lw=0, zorder=3)
axs.axvline(thr_w, color='0.25', lw=1.0, ls='--', zorder=4)
axs.axhline(np.log10(rate_q), color='0.25', lw=1.0, ls='--', zorder=4)
axs.set_xlim(0, 1.25); axs.set_ylim(np.log10(.05), np.log10(200))
axs.set_yticks(np.log10([.1, 1, 10, 100]))
axs.set_yticklabels(['0.1', '1', '10', '100'])
axs.set_xlabel('Peak-to-trough (ms)', fontsize=8)
axs.set_ylabel('Firing rate (Hz)', fontsize=8)
axs.tick_params(labelsize=7); axs.spines[['top', 'right']].set_visible(False)
bw = np.linspace(0, 1.25, 80)
axt.hist(ptt[~put], bins=bw, color=OTH_C, lw=0, alpha=.75)
axt.hist(ptt[put], bins=bw, color=INT_C, lw=0, alpha=.9)
axt.axvline(thr_w, color='0.25', lw=1.0, ls='--')
axt.set_axis_off()
br = np.linspace(np.log10(.05), np.log10(200), 80)
axr.hist(lr[~put], bins=br, color=OTH_C, lw=0, alpha=.75, orientation='horizontal')
axr.hist(lr[put], bins=br, color=INT_C, lw=0, alpha=.9, orientation='horizontal')
axr.axhline(np.log10(rate_q), color='0.25', lw=1.0, ls='--')
axr.set_axis_off()
_lp(axt, 'A')
axt.set_title(f'narrow AND fast\nn={int(put.sum())} ({100 * put.mean():.0f}%)',
              fontsize=7.5, loc='left')
# the thresholds belong against the lines they mark, not in the title. The
# fitted antimode is drawn faintly beside the 0.4 ms cut as corroboration of it.
axs.axvline(anti_w, color='0.55', lw=.8, ls=':', zorder=4)
axs.annotate(f'{thr_w:.1f} ms used\n(fit {anti_w:.2f}, {sep_w:.1f} SD apart)',
             (anti_w + .04, np.log10(.065)), fontsize=5.8, color='0.25',
             va='bottom', linespacing=1.35)
axs.annotate(f'{rate_q:.0f} Hz (top quartile)', (1.22, np.log10(rate_q * 1.25)),
             fontsize=5.8, color='0.25', ha='right')

# B, C: what each kind of cell does to its target. The presynaptic cell in B is
# one panel A called an interneuron; in C it is a broad-spiking cell. Neither
# fact was available to the detector.
import monosyn as MSY

ex = {}
ci = C[(C.kind == 'inh') & (C.ip == 'putative interneuron')].sort_values('z')
ce = C[(C.kind == 'exc') & (C.ip != 'putative interneuron')].sort_values(
    'z', ascending=False)
ex['inh'], ex['exc'] = ci.iloc[0], ce.iloc[0]
SS = {}
for k_, (kind, col, nm) in enumerate((
        ('inh', INH_C, 'classified interneuron → trough'),
        ('exc', EXC_C, 'broad-spiking cell → peak'))):
    ax = fig.add_subplot(gs['BC'[k_]])
    _lp(ax, 'BC'[k_])
    t = ex[kind]
    try:
        key = (int(t.mouse), int(t.day))
        if key not in SS:
            SS[key] = MSY.session_ccgs(*key)
        Sx = SS[key]
        i = int(np.where(Sx['ids'] == int(t.pre))[0][0])
        j = int(np.where(Sx['ids'] == int(t.post))[0][0])
        lg = Sx['lags'] * 1000
        w = np.abs(lg) <= 25
        ax.bar(lg[w], Sx['raw'][i, j][w], width=1.0, color='0.35', lw=0, zorder=2)
        ax.plot(lg[w], Sx['base'][i, j][w], color=col, lw=1.5, zorder=3)
        ax.axvspan(0.7, 4.7, color=col, alpha=.13, lw=0, zorder=0)
        ax.set_xlim(-25, 25)
        v = Sx['raw'][i, j][w]
        # headroom at the top so the inset sits in blank space rather than on
        # top of the correlogram it is magnifying
        ylo, yhi = max(0, v.min() * .88), v.max() * 1.04
        ax.set_ylim(ylo, ylo + (yhi - ylo) * 2.45)
        ax.set_title(f'{nm}\nM{int(t.mouse)} D{int(t.day)}: '
                     f'{int(t.pre)} → {int(t.post)}', fontsize=7.5, loc='left')
        # The effect lives in a 4 ms window that is 8% of the plotted range, so
        # the wide view shows the predictor is tracking the slow structure and
        # the inset shows what the test actually fired on. Both are needed: the
        # inset alone could not show that the deviation is local.
        iw = np.abs(lg) <= 5
        # pushed to the right-hand half so it clears the zoom box and the
        # correlogram's own structure, which sits left of centre in both panels
        axi = ax.inset_axes([.66, .60, .34, .38])
        axi.bar(lg[iw], Sx['raw'][i, j][iw], width=.95, color='0.35', lw=0, zorder=2)
        axi.plot(lg[iw], Sx['base'][i, j][iw], color=col, lw=1.2, zorder=3)
        axi.axvspan(0.7, 4.7, color=col, alpha=.13, lw=0, zorder=0)
        vi = Sx['raw'][i, j][iw]
        pad = max((vi.max() - vi.min()) * .18, 1)
        axi.set_xlim(-5.5, 5.5); axi.set_ylim(vi.min() - pad, vi.max() + pad * 2.2)
        axi.set_xticks([-5, 0, 5]); axi.set_yticks([])
        axi.tick_params(labelsize=5.4, length=2, pad=1)
        axi.set_title('±5 ms', fontsize=5.8, pad=1.5, color='0.3')
        for sp in axi.spines.values():
            sp.set_linewidth(.6); sp.set_color('0.5')
        axi.spines[['top', 'right', 'left']].set_visible(False)
        # point at the bin the detector actually fired on, taken from the
        # connection record rather than re-found, so the arrow cannot drift
        # from the lag the statistics were computed at
        di = int(np.argmin(np.abs(lg - float(t.lag_ms))))
        axi.annotate('trough' if kind == 'inh' else 'peak',
                     xy=(lg[di], Sx['raw'][i, j][di]),
                     xytext=(10, 20 if kind == 'inh' else 12),
                     textcoords='offset points', fontsize=6, color=col,
                     ha='left', va='center',
                     # the arrow is dark rather than the sign colour: drawn in
                     # the sign colour it reads as a kink in the predictor line
                     # it has to cross
                     arrowprops=dict(arrowstyle='->', color='0.25', lw=1.0,
                                     shrinkA=1, shrinkB=2))
        # mark the magnified region on the main axes. indicate_inset_zoom draws
        # the box at the INSET's data limits, which run outside the main y range
        # here and spill past the axes, so the region is drawn directly instead.
        ax.add_patch(plt.Rectangle((-5.5, ylo), 11, yhi - ylo, fill=False,
                                   edgecolor='0.55', lw=.7, zorder=5))
    except Exception as e:
        ax.text(.5, .5, f'unavailable\n{type(e).__name__}', ha='center',
                va='center', transform=ax.transAxes, fontsize=7, color='0.5')
        ax.set_title(f'{nm}', fontsize=7.5, loc='left')
    ax.set_xlabel('Lag (ms)', fontsize=8)
    ax.set_ylabel('Spike count', fontsize=8.5)
    ax.tick_params(labelsize=7); ax.spines[['top', 'right']].set_visible(False)

# G: latency, as paired bars -- excitatory left, inhibitory right, at each
# latency. The CCG is binned at 100/101 ms, so every detected lag falls on an
# exact multiple of that; rounding recovers the bin index and the bars sit on
# the integer millisecond they represent. Each sign is normalised within itself,
# since there are seven times as many inhibitory connections and the comparison
# is of SHAPE, not of count.
ax = fig.add_subplot(gs['G'])
BINW = 100 / 101
kk = (C.lag_ms / BINW).round().astype(int)
ks = np.arange(1, 5)
for s_, (kind, col, nm) in enumerate((('exc', EXC_C, 'excitatory'),
                                      ('inh', INH_C, 'inhibitory'))):
    m = C.kind == kind
    frac = [float((kk[m] == k).mean()) for k in ks]
    med = float(np.median(kk[m]))
    ax.bar(ks + (s_ - .5) * .38, frac, width=.38, color=col, lw=0,
           label=f'{nm} (n={int(m.sum())}, median {med:.0f} ms)')
ax.set_xticks(ks)
ax.set_ylim(0, max(ax.get_ylim()[1], .5) * 1.26)
ax.set_xlabel('Latency (ms)', fontsize=8)
ax.set_ylabel('Fraction of connections', fontsize=8)
_lp(ax, 'G')
ax.set_title('inhibition is slower', fontsize=8, loc='left')
ax.legend(fontsize=6, frameon=False, loc='upper right')
ax.tick_params(labelsize=7); ax.spines[['top', 'right']].set_visible(False)

# D: THE VALIDATION
ax = fig.add_subplot(gs['D'])
base = Ki.identity.value_counts(normalize=True)
for k_, (kind, col) in enumerate((('exc', EXC_C), ('inh', INH_C))):
    pre = C[C.kind == kind].dropna(subset=['ip']).ip.value_counts(normalize=True)
    en = [pre.get(a, 0) / base.get(a, 1) for a in ORDER]
    ax.bar(np.arange(4) + (k_ - .5) * .38, en, width=.38, color=col, lw=0,
           label='excitatory' if kind == 'exc' else 'inhibitory')
ax.axhline(1, color='k', lw=1, ls='--')
ax.set_ylim(0, max(ax.get_ylim()[1], 1.2) * 1.30)
ax.set_xticks(range(4)); ax.set_xticklabels(SHORT, fontsize=7)
for t_, a in zip(ax.get_xticklabels(), ORDER):
    t_.set_color(ICOL[a])
ax.set_ylabel('Presynaptic enrichment', fontsize=8)
_lp(ax, 'D')
ax.set_title('the cells in A are the ones\nthat inhibit', fontsize=7.5, loc='left')
ax.legend(fontsize=6, frameon=False)
ax.tick_params(labelsize=7); ax.spines[['top', 'right']].set_visible(False)

# E, F: both connection matrices. The excitatory one is only interpretable
# against the inhibitory one, and the int row is the whole point of the panel.
# Each is scaled to its own maximum because the two differ by an order of
# magnitude; a shared scale would render the excitatory matrix uniformly black.
cm = plt.get_cmap('magma')
for slot, kind, nm in (('E', 'inh', 'inhibitory'), ('F', 'exc', 'excitatory')):
    ax = fig.add_subplot(gs[slot])
    Mx, nk = prob_matrix(kind)
    ax.imshow(Mx, cmap='magma', vmin=0, vmax=np.nanmax(Mx))
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
    if slot == 'E':
        ax.set_ylabel('presynaptic', fontsize=7.5)
    _lp(ax, slot)
    ax.set_title(f'{nm} (% of pairs)\nn={nk}', fontsize=7.5, loc='left')
    ax.set_box_aspect(1)
    ax.tick_params(length=0, labelsize=6.5)

# G: jitter null
ax = fig.add_subplot(gs['H'])
for kind, col, ce_, cn in (('exc', EXC_C, 'exc', 'null_e'),
                           ('inh', INH_C, 'inh', 'null_i')):
    ax.scatter(nn[ce_], nn[cn], s=20, color=col, lw=.4, edgecolor='w', zorder=3,
               label=f'{kind} (FDR {nn[cn].sum()/max(nn[ce_].sum(),1):.2f})')
mx = max(nn['inh'].max(), 1) * 1.1
ax.plot([0, mx], [0, mx], color='0.6', lw=.8, ls=':')
ax.set_xscale('symlog'); ax.set_yscale('symlog')
ax.set_xlabel('connections detected', fontsize=8)
ax.set_ylabel('after jitter', fontsize=8)
_lp(ax, 'H')
ax.set_title(f'both signs validate\n({len(nn)} sessions)', fontsize=7.5, loc='left')
ax.legend(fontsize=6, frameon=False, loc='upper left')
ax.tick_params(labelsize=7); ax.spines[['top', 'right']].set_visible(False)

# H: does the wiring carry the state?
ax = fig.add_subplot(gs['I'])
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
_lp(ax, 'I')
ax.set_title('but the wiring does not carry\nthe state (open = matched unconn.)',
             fontsize=7.5, loc='left')
ax.set_ylim(top=ax.get_ylim()[1] * 1.16)
ax.tick_params(labelsize=7); ax.spines[['top', 'right']].set_visible(False)

out = f'{FIG}/fig4_monosynaptic.pdf'
plt.savefig(out, dpi=200, bbox_inches='tight')
print(f'\nsaved {out}')
