"""Figure S21: the classifier contrast of Figure S20, drawn as the construction
of the test and its two results. Numbers from neighbour_test.py, labels from
build_neighbour_labels.py.

A     the cued and uncued tracks, and where hits and misses differ in running
      (speed by position), shaded for the stretch of track each classifier is
      trained on besides the whole track: the task region (50-110 cm) and the
      black box (-30 to 30 cm, the end of the previous trial and the start of
      this one)
B     miss minus hit speed per session on the black box and the task region,
      cued and uncued, and cued against uncued
C     the two hypotheses: above, in the abstract, a map tied to the world that
      either slips against it (H1, behaviour follows the map) or moves with it
      (H2, the map follows behaviour);
      below, as toy cells run through the same classifiers: a
      cell that carries the state (grid-like fields that hold their place on
      anchored trials) and one that follows behaviour (a speed cell);
      the true state, the outcome and each classifier's labels per trial, and
      the pattern each predicts for H
D-F   MEC in Figure 1's session (M26 D18), the subicular complex (M25 D25)
      and visual cortex (M21 D21): each classifier's labels, cells in that
      classifier's own PC1 order, with the outcome of each cued and uncued trial
G     per structure, across sessions: how far the black-box state follows the
      whole-track state, against the whole-track state's persistence
H     the trial's own state against its outcome, uncued trials, per classifier
I     the state on the neighbouring cued trials against the uncued outcome, per
      classifier and with the neighbours' behaviour regressed out
"""
import sys
import numpy as np, pandas as pd
import matplotlib; matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap
from scipy.stats import wilcoxon
sys.path.insert(0, '/Users/harryclark/Documents/spatial-manifolds/src')
sys.path.insert(0, '/Users/harryclark/Documents/spatial-manifolds/scripts/figures/AnchorDynamics2026')
from spatial_manifolds.anchoring import ANCH_COLOR, NONANCH_COLOR, smooth_nanaware
from neighbour_test import NT, PS, SPEED_PROF
import fig2_example_panel                                        # D-F
REGIONS = ['MEC', 'SUB', 'VIS', 'CB']                           # the order of D-F, G, H and I

plt.rcParams['font.family'] = 'Arial'
plt.rcParams['mathtext.default'] = 'regular'                  # H1/H2 subscripts in Arial
FIG = '/Users/harryclark/Documents/spatial-manifolds/scripts/figures/AnchorDynamics2026'
CCOL = {'full': '0.62', 'tz': '#e0b43c', 'bb': '0.2'}           # one colour per classifier
NAME = {'full': 'whole track', 'tz': 'task region', 'bb': 'black box'}
CUED_C, UNCUED_C = '#3f9b4f', '#c0398d'                        # trial types, as Figure 1
# D-F: one session per structure, chosen by pick_fig2_examples.py (a session that
# displays the state, nearest its structure's median black-box result) and set
# in build_fig2_example_maps.EXAMPLES; read here from that script's output so
# the maps and the rasters cannot come from different sessions.
R = pd.read_csv(f'{PS}/neighbour_test.csv')
EXMAPS = np.load(f'{PS}/fig2_example_maps.npz')                # build_fig2_example_maps.py
EX, EX_SUB, EX_VIS = (tuple(int(v) for v in EXMAPS[f'{r}_session']) for r in ('MEC', 'SUB', 'VIS'))


def lp(ax, s, x=-.16, y=1.04):
    ax.text(x, y, s, transform=ax.transAxes, fontsize=10, weight='bold', va='bottom', ha='right')


def stars(p):
    return 'n.s.' if p > .05 else ('*' if p > .01 else ('**' if p > .001 else '***'))


def clean(ax):
    ax.tick_params(labelsize=7); ax.spines[['top', 'right']].set_visible(False)


fig = plt.figure(figsize=(11.8, 11.9))
gs = fig.add_gridspec(3, 1, height_ratios=[1.35, 1.95, 1.6], hspace=.3)
top = gs[0].subgridspec(1, 3, width_ratios=[1, .78, 2.7], wspace=.3)

# ── A: speed by position, and the stretch each classifier uses ────────────
ga = top[0].subgridspec(2, 1, height_ratios=[.42, 1], hspace=.05)
axr = fig.add_subplot(ga[0])
ax = fig.add_subplot(ga[1], sharex=axr)
# the two trial types' tracks, drawn as Figure 1 draws them: black boxes at
# both ends, dotted walls between, green bars over the reward zone when cued
from matplotlib.patches import Rectangle
trng = np.random.default_rng(7)
axr.axvspan(90, 110, ymin=.04, ymax=.83, color='#dbe7d3', lw=0, zorder=0)
for y0, cued in ((1.25, True), (.15, False)):
    axr.add_patch(Rectangle((0, y0), 200, .6, facecolor='k', lw=0, zorder=1))
    axr.plot([30, 170], [y0 - .03, y0 - .03], color='#8a5a2b', lw=1.2, solid_capstyle='butt', zorder=1)
    xd = trng.uniform(31, 169, 90)
    axr.scatter(xd, y0 + trng.uniform(.08, .52, 90), s=trng.uniform(.3, 3.5, 90), color='0.72', lw=0, zorder=2)
    if cued:
        for xb in (91.5, 98.5, 105.5):
            axr.add_patch(Rectangle((xb, y0), 3, .6, facecolor='#2fc12f', lw=0, zorder=3))
    axr.text(-4, y0 + .3, 'cued' if cued else 'uncued', ha='right', va='center', fontsize=6.3,
             color=CUED_C if cued else UNCUED_C)
axr.text(100, 2.05, 'reward zone', ha='center', va='bottom', fontsize=6, color='0.3')
TXTC = {'full': '0.45', 'tz': '#a07c10', 'bb': '0.15'}
axr.set_ylim(-.15, 2.45); axr.axis('off')
SP = np.load(SPEED_PROF)
pc = np.arange(5, 200, 10)
prof = {k: [] for k in [('nb', 1), ('nb', 0), ('b', 1), ('b', 0)]}
for f in SP.files:
    z = np.load(f'{NT}/{f}.npz', allow_pickle=True)
    for (t_, h_), L in prof.items():
        m = (z['ttype'] == t_) & (z['hit'] == h_)
        if m.sum() >= 5:
            L.append(np.nanmean(SP[f][m], axis=0))
for a_, b_ in ((0, 30), (170, 200)):
    ax.axvspan(a_, b_, color=CCOL['bb'], alpha=.15, lw=0)
ax.axvspan(50, 110, color=CCOL['tz'], alpha=.3, lw=0)
for x_ in (90, 110):
    ax.axvline(x_, color='0.5', lw=.6, ls=':')
for x_, t_, c in ((15, 'black\nbox', TXTC['bb']), (185, 'black\nbox', TXTC['bb']),
                  (80, 'task\nregion', TXTC['tz'])):
    ax.text(x_, .985, t_, transform=ax.get_xaxis_transform(), ha='center', va='top', fontsize=5.3,
            color=c, linespacing=1)
for (t_, h_), L in prof.items():
    L = np.array(L); mu, se = np.nanmean(L, 0), np.nanstd(L, 0) / np.sqrt(len(L))
    c = UNCUED_C if t_ == 'nb' else CUED_C
    ax.plot(pc, mu, color=c, lw=1.2, ls='-' if h_ else '--',
            label=f'{"uncued" if t_ == "nb" else "cued"} {"hit" if h_ else "miss"}')
    ax.fill_between(pc, mu - se, mu + se, color=c, alpha=.15, lw=0, edgecolor='none')
ax.set_xlim(0, 200); ax.set_xlabel('track position (cm)', fontsize=7.5)
ax.set_ylabel('speed (cm/s)', fontsize=7.5)
ax.legend(fontsize=5.3, frameon=False, loc='lower left', bbox_to_anchor=(.0, .0), handlelength=1.6)
lp(axr, 'A', x=-.3, y=.95); clean(ax)

# ── B: how much hits and misses differ in speed on the two stretches ──────
# per session, miss minus hit mean speed on each stretch, cued and uncued; the
# black box is the previous trial's last 30 cm and this trial's first 30 cm
SEG = [('box', None), ('task', (5, 11))]
rows_ = []
for f in SP.files:
    z = np.load(f'{NT}/{f}.npz', allow_pickle=True)
    P, tt_, h_ = SP[f], z['ttype'], z['hit']
    prev = np.r_[np.full((1, P.shape[1]), np.nan), P[:-1]]
    v = {'box': np.nanmean(np.c_[prev[:, 17:20], P[:, 0:3]], 1), 'task': np.nanmean(P[:, 5:11], 1)}
    for t_ in ('b', 'nb'):
        m = tt_ == t_
        if (h_[m] == 1).sum() >= 5 and (h_[m] == 0).sum() >= 5:
            rows_.append(dict(ses=f, t=t_, **{k: np.nanmean(v[k][m & (h_ == 0)]) - np.nanmean(v[k][m & (h_ == 1)])
                                            for k, _ in SEG}))
DS = pd.DataFrame(rows_)
ax = fig.add_subplot(top[1].subgridspec(2, 1, height_ratios=[.42, 1], hspace=.05)[1])  # level with A
rng = np.random.default_rng(2)
for j, (k, _) in enumerate(SEG):
    ax.axvspan(j - .45, j + .45, color=CCOL['bb' if k == 'box' else 'tz'], alpha=.13 if k == 'box' else .25,
               lw=0, zorder=0)
    for d_, (t_, c) in zip((-.19, .19), (('b', CUED_C), ('nb', UNCUED_C))):
        v = DS[DS.t == t_][k].dropna().values
        ax.bar(j + d_, v.mean(), width=.34, color=c, alpha=.75, lw=0)
        ax.errorbar(j + d_, v.mean(), yerr=v.std(ddof=1) / np.sqrt(len(v)), color='k', lw=1, capsize=2)
        ax.scatter(j + d_ + rng.uniform(-.08, .08, len(v)), v, s=2.5, color='k', alpha=.35, lw=0, zorder=3)
        print(f'  speed miss-hit {t_} {k}: {v.mean():+.1f} cm/s (n={len(v)}, vs 0 p={wilcoxon(v).pvalue:.1g})')
    w_ = DS.pivot_table(index='ses', columns='t', values=k).dropna()
    p_ = wilcoxon(w_['b'], w_['nb']).pvalue
    ax.plot([j - .19, j - .19, j + .19, j + .19], [31, 33, 33, 31], color='0.3', lw=.8)
    ax.text(j, 34, stars(p_), ha='center', fontsize=6.5)
    print(f'  speed miss-hit {k}, cued vs uncued: {len(w_)} sessions, p={p_:.1g}')
ax.axhline(0, color='0.5', lw=.7)
ax.set_xticks(range(len(SEG))); ax.set_xticklabels(['black box', 'task region'], fontsize=6.5)
ax.set_xlim(-.55, 1.55); ax.set_ylim(-22, 40)
ax.set_ylabel('speed, miss − hit (cm/s)', fontsize=7)
ax.set_title('hits vs misses on each stretch\n(green cued, pink uncued)', fontsize=6.6, loc='left')
lp(ax, 'B', x=-.34, y=1.04); clean(ax)


# ── C: two toy cells, one for each hypothesis ─────────────────────────────
# The paper's classifier (trial_cluster_labels, no median filter) run on two
# simulated cells over one simulated session. The state runs in epochs of
# trials; cued and uncued trials are drawn 50/50, cued trials are mostly hits
# throughout and an uncued hit is likelier when anchored. The STATE cell is grid-like: its
# fields hold their place along the track on anchored trials and shift on
# non-anchored ones, everywhere on the track. The BEHAVIOUR cell is a speed
# cell, driven by the mean measured speed profile of its trial type and
# outcome (panel A);
# it carries no state. Rate maps are -30 to 200 cm, the first 30 cm the end of
# the previous trial.
TOY_X = np.arange(-30, 200, 2) + 1.0
STATE = np.r_[np.ones(10), np.zeros(9), np.ones(11), np.zeros(10)].astype(bool)


def toy_session(rng):
    cued = rng.random(len(STATE)) < .5
    hit = rng.random(len(STATE)) < np.where(cued, .9, np.where(STATE, .85, .2))
    return np.where(cued, 'b', 'nb'), hit


def toy_speed(tt, hit, prev_tt, prev_hit, rng):
    """Speed on -30..200 cm from the measured mean profile of each trial type and outcome."""
    L = {k: np.nanmean(np.array(v), 0) for k, v in prof.items()}
    cur = np.interp(TOY_X % 200, pc, L[(tt, int(hit))], period=200)
    prev = np.interp(TOY_X + 200, pc, L[(prev_tt, int(prev_hit))])
    v = np.where(TOY_X < 0, prev, cur)
    return np.clip(v + rng.normal(0, 2, v.shape), 0, None)


def toy_maps(rng):
    tt, hit = toy_session(rng)
    phase = np.where(STATE, 0., rng.uniform(0, 70, len(STATE)))
    Ms, Mb = [], []
    for t in range(len(STATE)):
        xs = TOY_X - phase[t]
        grid = .3 + 9 * sum(np.exp(-(xs - c) ** 2 / (2 * 7 ** 2)) for c in np.arange(-100, 300, 70))
        v = toy_speed(tt[t], hit[t], tt[t - 1] if t else 'b', hit[t - 1] if t else True, rng)
        Ms.append(rng.poisson(grid * .4) / .4)
        Mb.append(rng.poisson((.2 + .3 * v) * 1.) / 1.)
    return tt, hit, np.array(Ms), np.array(Mb)


from spatial_manifolds.anchoring import trial_cluster_labels
seg = lambda M, lo, hi: M[:, (TOY_X >= lo) & (TOY_X < hi)]
tt_toy, hit_toy, MS, MB = toy_maps(np.random.default_rng(4))
gb = top[2].subgridspec(2, 2, height_ratios=[.6, 1], wspace=.18, hspace=.42)
cmap = ListedColormap([NONANCH_COLOR, ANCH_COLOR])


def rings(ax, x0, world, mp, col):
    """One world ring over one map ring, with the arrows that tie each point on
    the map to its place in the world. world and mp are (shear, shift, phase):
    the shear tilts the ring, the shift moves it, the phase slides the points
    round it. Equal changes to both keep the arrows vertical."""
    t = np.linspace(0, 2 * np.pi, 200)
    tk = np.deg2rad([-160, -105, -50, 15, 75, 135])
    xy = lambda tt, cy, sh, dx, ph: (x0 + dx + 1.15 * np.cos(tt + ph),
                                     cy + .3 * np.sin(tt + ph) + sh * np.cos(tt + ph))
    for cy, (sh, dx, ph), c in ((1.35, world, '0.3'), (0, mp, col)):
        ax.plot(*xy(t, cy, sh, dx, 0), color=c, lw=1.6, zorder=2)
    pw, pm = xy(tk, 1.35, *world), xy(tk, 0, *mp)
    for k in range(len(tk)):
        ax.annotate('', xy=(pm[0][k], pm[1][k]), xytext=(pw[0][k], pw[1][k]), zorder=1,
                    arrowprops=dict(arrowstyle='<|-|>', color='0.55', lw=.7, mutation_scale=5,
                                    shrinkA=2.5, shrinkB=2.5))
    ax.scatter(*pw, s=11, facecolor='w', edgecolor='0.3', lw=.8, zorder=3)
    ax.scatter(*pm, s=11, facecolor='w', edgecolor=col, lw=.8, zorder=3)


# The two hypotheses, abstractly: anchored, the map sits on the world point for
# point. A state (H1) lets the map slip against a world that has not
# changed; a record of behaviour moves with the world, the mapping intact.
CHANGE = [('map slips,\nbehaviour same', ((0, 0, 0), (-.38, .3, .55))),
          ('behaviour changes,\nmap follows', ((.38, .3, 0), (.38, .3, 0)))]
HYP = [('H$_1$: behaviour follows the map', MS, 'grid-like cell',
        [.26, .3, .12]),
       ('H$_2$: the map follows behaviour', MB, 'speed cell',
        [.26, .3, 0])]
for row, ((name, M, desc, pred), (how, (w2, m2))) in enumerate(zip(HYP, CHANGE)):
    asch = fig.add_subplot(gb[0, row])
    asch.axhline(.68, color='0.75', lw=.6, ls=(0, (2, 2)), zorder=0)
    rings(asch, 1.3, (0, 0, 0), (0, 0, 0), ANCH_COLOR)
    rings(asch, 7.5, w2, m2, NONANCH_COLOR)
    asch.annotate('', xy=(5.4, .68), xytext=(3.6, .68),
                  arrowprops=dict(arrowstyle='-|>', color='0.2', lw=1, mutation_scale=8))
    asch.text(4.5, .88, how, ha='center', va='bottom', fontsize=5.6, color='0.2', linespacing=1.1)
    for x_, lab_, c in ((1.3, 'anchored', ANCH_COLOR), (7.8, 'non-anchored', NONANCH_COLOR)):
        asch.text(x_, -.55, lab_, ha='center', va='top', fontsize=6, color=c, weight='bold')
    asch.text(-.1, 1.35, 'world', ha='right', va='center', fontsize=5.8, color='0.3')
    asch.text(-.1, 0, 'map', ha='right', va='center', fontsize=5.8, color='0.3')
    asch.set_xlim(-1.2, 9.4); asch.set_ylim(-1.0, 2.1); asch.axis('off')
    asch.set_title(name,
                   fontsize=6.8, loc='left', weight='bold', pad=2)
    if row == 0:
        lp(asch, 'C', x=-.06, y=1.0)
    gr = gb[1, row].subgridspec(1, 8, width_ratios=[1.5, .2, .2, .2, .2, .2, .4, 1.2], wspace=.14)
    am = fig.add_subplot(gr[0])
    am.imshow(M, aspect='auto', cmap='viridis', extent=[-30, 200, len(M), 0],
              vmax=np.percentile(M, 99), interpolation='nearest')
    for (lo, hi), k in (((-30, 30), 'bb'), ((50, 110), 'tz')):
        am.add_patch(plt.Rectangle((lo, 1.01), hi - lo, .07, transform=am.get_xaxis_transform(),
                                   color=CCOL[k], clip_on=False, lw=0))
    am.axvline(0, color='w', lw=.6, ls=':')
    am.set_yticks([]); am.set_ylabel('trials', fontsize=6.3)
    am.set_xticks([0, 100, 200]); am.tick_params(labelsize=5.8)
    am.set_xlabel('position (cm)', fontsize=6.3, labelpad=1)
    am.text(0, 1.11, desc, transform=am.transAxes, fontsize=5.8, color='0.3', va='bottom')
    lab = {k: trial_cluster_labels(seg(M, *span), medfilt_size=0)[0]
           for k, span in (('bb', (-30, 30)), ('tz', (50, 110)))}
    # outcomes as D-F: filled hit, white miss, grey the other trial type
    out_ = lambda t_, c: (np.where(tt_toy == t_, hit_toy.astype(float), .5), ListedColormap(['white', '0.88', c]))
    cols = [('state', STATE.astype(float), cmap), ('cued', *out_('b', CUED_C)),
            ('uncued', *out_('nb', UNCUED_C)), ('box', lab['bb'], cmap), ('task', lab['tz'], cmap)]
    for j, (cn, v, cm) in enumerate(cols):
        ac = fig.add_subplot(gr[1 + j])
        ac.imshow(v[:, None], aspect='auto', cmap=cm, vmin=0, vmax=1, interpolation='nearest')
        ac.set_xticks([]); ac.set_yticks([])
        for sp in ac.spines.values():
            sp.set_linewidth(.5); sp.set_color('0.5')
        ac.set_title(cn, fontsize=5.4, pad=2, rotation=90,
                     color={'box': TXTC['bb'], 'task': TXTC['tz'], 'cued': CUED_C,
                            'uncued': UNCUED_C}.get(cn, '0.2'))
    ap = fig.add_subplot(gr[7])
    ap.bar(range(3), pred, width=.7, color=[CCOL[k] for k in ('full', 'tz', 'bb')], alpha=.85, lw=0)
    if pred[2] == 0:
        ap.text(2, .015, '0', ha='center', fontsize=6, color=CCOL['bb'])
    ap.axhline(0, color='0.5', lw=.7)
    ap.set_ylim(-.02, .36); ap.set_yticks([]); ap.set_xticks(range(3))
    ap.set_xticklabels(['whole', 'task', 'box'], fontsize=5.4, rotation=90)
    ap.set_title('predicted AUC\nover null, in G', fontsize=5.6, pad=2)
    ap.spines[['top', 'right', 'left']].set_visible(False)


# ── D-F: three sessions through the three classifiers ─────────────────────
row = gs[1].subgridspec(2, 1, height_ratios=[.1, 1], hspace=0)[1].subgridspec(1, 3, wspace=.22)


def example(spec, ses, region, letter, title):
    z = np.load(f'{NT}/M{ses[0]}D{ses[1]}.npz', allow_pickle=True)
    fig2_example_panel.draw(fig, spec, z, region, EXMAPS[f'{region}_maps'],
                            EXMAPS[f'{region}_cells'], title, letter)


example(row[0], EX, 'MEC', 'D', f'MEC, M{EX[0]} D{EX[1]}')
example(row[1], EX_SUB, 'SUB', 'E', f'subicular complex, M{EX_SUB[0]} D{EX_SUB[1]}')
example(row[2], EX_VIS, 'VIS', 'F', f'visual cortex, M{EX_VIS[0]} D{EX_VIS[1]}')

def bars(ax, cols, kinds, ticklabels, title, letter, ylim, reg_y=-.27):
    """Per structure, one bar per measure: mean, SEM, session points, signed-rank stars."""
    nc = len(cols)
    w = .86 / nc
    rng = np.random.default_rng(1)
    ticks, labs = [], []
    for j, r in enumerate(REGIONS):
        g = R[R.region == r]
        for k, (c, kind) in enumerate(zip(cols, kinds)):
            v = g[f'{c}_x'].dropna()
            xx = j + (k - (nc - 1) / 2) * w
            resid = c.startswith('res_')
            ax.bar(xx, v.mean(), width=w * .82, color=CCOL[kind], alpha=.35 if resid else .85,
                   hatch='////' if resid else None, edgecolor=CCOL[kind] if resid else 'none', lw=0)
            ax.errorbar(xx, v.mean(), yerr=v.sem(), color='k', lw=1.1, capsize=2.5, capthick=1.1, zorder=4)
            ax.scatter(xx + rng.uniform(-w * .25, w * .25, len(v)), v, s=4, color='k', alpha=.4,
                       lw=0, zorder=3)
            ax.text(xx, ylim[1] * .94, stars(wilcoxon(v).pvalue), ha='center', fontsize=6)
            ticks.append(xx); labs.append(ticklabels[k])
        ax.text(j, reg_y, f'{r} ({g[f"{cols[0]}_x"].notna().sum()})', transform=ax.get_xaxis_transform(),
                ha='center', va='top', fontsize=7.5, color='k')
    ax.axhline(0, color='0.5', lw=.7)
    ax.set_xticks(ticks); ax.set_xticklabels(labs, fontsize=5.7, rotation=90)
    ax.set_xlim(-.55, len(REGIONS) - .45); ax.set_ylim(*ylim)
    ax.set_ylabel('AUC over null', fontsize=7.5)
    ax.set_title(title, fontsize=7.5, loc='left')
    lp(ax, letter, x=-.09); clean(ax)


def trial_strip(ax, x0, tt, lab, faded=False):
    """One trial as a strip of track, 0.1 units per cm: black boxes at both
    ends, the task region marked, the type in its colour above."""
    a = .3 if faded else 1
    ax.add_patch(Rectangle((x0, 0), 20, 1, facecolor='0.93', edgecolor='0.6', lw=.5, alpha=a))
    for b0 in (0, 17):
        ax.add_patch(Rectangle((x0 + b0, 0), 3, 1, facecolor='k', lw=0, alpha=a))
    ax.add_patch(Rectangle((x0 + 5, 0), 6, 1, facecolor=CCOL['tz'], lw=0, alpha=.55 * a))
    c = CUED_C if tt == 'b' else UNCUED_C
    ax.text(x0 + 10, 1.35, lab, ha='center', va='bottom', fontsize=6.2, color=c, alpha=a)


def stretch(ax, a_, b_, y, kind, text=None):
    """A bracket under the strips marking the stretch a classifier reads."""
    ax.plot([a_, a_, b_, b_], [y + .25, y, y, y + .25], color=CCOL[kind], lw=1.3,
            solid_capstyle='butt', clip_on=False)
    if text:
        ax.text((a_ + b_) / 2, y - .2, text, ha='center', va='top', fontsize=5.6, color=TXTC[kind])


def schematic_G(ax):
    """G: the trial's own stretches against its own outcome."""
    trial_strip(ax, 0, 'b', 'trial n-1', faded=True)
    trial_strip(ax, 22, 'nb', 'uncued trial n')
    stretch(ax, 22, 42, -.4, 'full', 'whole track')
    stretch(ax, 27, 33, -1.75, 'tz', 'task')
    stretch(ax, 19, 25, -1.75, 'bb', 'box')
    ax.annotate('', xy=(49, .5), xytext=(43, .5), arrowprops=dict(arrowstyle='->', lw=.8))
    ax.text(50, .5, 'outcome of\ntrial n', ha='left', va='center', fontsize=6.2)
    ax.set_xlim(-1, 66); ax.set_ylim(-4.6, 3.8); ax.axis('off')


def schematic_H(ax):
    """H: the nearest preceding cued trial's stretches against the uncued outcome."""
    for k, (tt, lab, f) in enumerate((('b', 'cued trial, nearest\nwithin 3 before n', False),
                                      ('nb', 'n-1', True), ('nb', 'uncued trial n', False),
                                      ('b', 'n+1: never used', True))):
        trial_strip(ax, 22 * k, tt, lab, faded=f)
    stretch(ax, 5, 11, -.4, 'tz', 'task')
    stretch(ax, -3, 3, -.4, 'bb', 'box')
    ax.annotate('', xy=(55, 2.4), xytext=(15, 3.9),
                arrowprops=dict(arrowstyle='->', lw=.8, connectionstyle='arc3,rad=-.2'))
    ax.text(37, 5.5, 'its state → outcome of trial n', ha='center', va='bottom', fontsize=6.2)
    ax.set_xlim(-4, 87); ax.set_ylim(-4.6, 6.4); ax.axis('off')


def verdict(ax):
    """I: which hypothesis each structure's pattern in G and H fits. Top two rows,
    what each hypothesis predicts per test (filled, above the null; open, at it);
    below, each structure (filled, above its null, p < 0.05; open, not). The
    black-box tests decide: H1 if every one is above its null, H2 suffices if
    none is. The residual task-region test is shown but does not decide, since
    regressing behaviour out removes only the behaviour it measures."""
    tests = [('same_tz', 'G\ntask'), ('same_bb', 'G\nbox'), ('nbr_bb', 'H\nbox'),
             ('res_bb', 'H\nbox\nres.'), ('res_tz', 'H\ntask\nres.')]
    decide = ['same_bb', 'nbr_bb', 'res_bb']
    pred = {'H$_1$': [1, 1, 1, 1, 1], 'H$_2$': [1, 0, 0, 0, 0]}
    rows = list(pred) + REGIONS
    for i, name in enumerate(rows):
        y = -i - (.6 if i >= 2 else 0)
        if i < 2:
            hits = pred[name]
            for j, h in enumerate(hits):
                ax.scatter(j, y, s=46, marker='s', color='0.35' if h else 'white',
                           edgecolor='0.35', lw=.8)
            ax.text(-.7, y, f'{name} predicts', ha='right', va='center', fontsize=6.6)
            continue
        g = R[R.region == name]
        sig = {}
        for j, (c, _) in enumerate(tests):
            v = g[f'{c}_x'].dropna()
            sig[c] = bool(v.mean() > 0 and wilcoxon(v).pvalue < .05)
            ax.scatter(j, y, s=46, color='k' if sig[c] else 'white', edgecolor='k', lw=.8,
                       alpha=1 if c in decide or c == 'same_tz' else .45)
            ax.text(j, y - .32, f'{v.mean():+.2f}', ha='center', va='top', fontsize=4.8, color='0.4')
        k = sum(sig[c] for c in decide)
        v_ = 'H$_1$' if k == len(decide) else ('H$_1$ in part' if k else 'H$_2$ suffices')
        ax.text(-.7, y, f'{name} ({len(g[f"{decide[0]}_x"].dropna())})', ha='right', va='center',
                fontsize=6.6)
        ax.text(len(tests) - .35, y, v_, ha='left', va='center', fontsize=6.8,
                weight='bold' if k == len(decide) else 'normal')
    ax.axhline(-1.8, color='0.75', lw=.6, xmin=.0, xmax=1)
    for j, (c, lab) in enumerate(tests):
        ax.text(j, .75, lab, ha='center', va='bottom', fontsize=6,
                color=TXTC['bb'] if 'box' in lab else TXTC['tz'])
    ax.add_patch(Rectangle((.55, -6.35), 3, 7.0, fill=False, ec=CCOL['bb'], lw=.8, ls='--'))
    ax.text(2, -6.55, 'the black-box tests decide', ha='center', va='top', fontsize=5.8,
            color='0.3')
    ax.set_xlim(-.6, len(tests) + 1.6); ax.set_ylim(-7.0, 2.2); ax.axis('off')


bot = gs[2].subgridspec(1, 3, width_ratios=[3.3, 3.9, 2.3], wspace=.3)
bg = bot[0].subgridspec(2, 1, height_ratios=[.6, 1], hspace=.3)
bh = bot[1].subgridspec(2, 1, height_ratios=[.6, 1], hspace=.3)
ax = fig.add_subplot(bg[0]); schematic_G(ax); lp(ax, 'G', x=.02, y=.9)
ax = fig.add_subplot(bh[0]); schematic_H(ax); lp(ax, 'H', x=.0, y=.9)
bars(fig.add_subplot(bg[1]), ['same_full', 'same_tz', 'same_bb'], ['full', 'tz', 'bb'],
     ['whole', 'task', 'box'],
     "own state → own outcome (uncued trials)", '', (-.3, .72), reg_y=-.27)
bars(fig.add_subplot(bh[1]), ['nbr_tz', 'nbr_bb', 'res_tz', 'res_bb'], ['tz', 'bb', 'tz', 'bb'],
     ['task', 'box', 'task res.', 'box res.'],
     'preceding cued trial\'s state → uncued outcome', '', (-.3, .52), reg_y=-.37)
bi = bot[2].subgridspec(2, 1, height_ratios=[.08, 1])
ax = fig.add_subplot(bi[1]); verdict(ax); lp(ax, 'I', x=-.03, y=1.0)

out = f'{FIG}/fig1_supp_neighbour_bars.pdf'
plt.savefig(out, dpi=200, bbox_inches='tight')
print(f'saved {out}')
