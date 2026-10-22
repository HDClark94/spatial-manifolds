"""Figure 8 supplement: is either gamma band's amplitude built out of
fast-spiking interneuron firing? Computed by gamma_envelope_rate.py.

The companion supplement (fig7_supp_gamma_locking.py) asked whether
interneurons FOLLOW the gamma rhythm. This one asks whether the rhythm's
amplitude is MADE of their firing, which is the claim the Figure 9 model
actually needs and which locking cannot see: a cell can be perfectly locked to
a rhythm it contributes nothing to.

    A  the distance control, which turns out to bound the leak, not measure it
    B  partial correlation of population rate with band amplitude, by class
    C  both populations against a phase-randomised null, per session
    D  band specificity: fast minus slow, for each class
    E  what each class does around an amplitude peak
    F  the same by anchoring state

Writes fig7_supp_gamma_envelope.pdf
"""
import os

import matplotlib
import numpy as np
import pandas as pd

matplotlib.use('Agg')
import matplotlib.pyplot as plt
from scipy.stats import wilcoxon

plt.rcParams['font.family'] = 'Arial'
plt.rcParams['pdf.fonttype'] = 42

ROOT = '/Users/harryclark/Documents/spatial-manifolds'
FIG = os.path.dirname(os.path.abspath(__file__))
OUT = f'{FIG}/fig7_supp_gamma_envelope.pdf'
ANCH_COLOR, NONANCH_COLOR = '#a8559e', '#3f9b8f'
# Figure 2 reserves red and blue for cell types, so the two populations here
# take the interneuron orange of Figure S11 and a neutral grey.
CINT, CPRN = '#d95f02', '#555555'
BANDS = [('slow', 'slow gamma\n30–48 Hz'), ('fast', 'fast gamma\n60–100 Hz')]
BSHORT = {'slow': 'slow', 'fast': 'fast'}


def lp(ax, s, x=-.26, y=1.0):
    ax.text(x, y, s, transform=ax.transAxes, fontsize=10, weight='bold',
            va='bottom', ha='right')


def tidy(ax):
    ax.tick_params(labelsize=6.5)
    ax.spines[['top', 'right']].set_visible(False)


def strip(ax, j, v, c, w=.11, seed=0):
    r = np.random.default_rng(seed)
    ax.scatter(np.full(len(v), j) + r.uniform(-w, w, len(v)), v, s=7,
               color=c, lw=0, alpha=.8, zorder=3)
    ax.plot([j - .22, j + .22], [np.median(v)] * 2, color='k', lw=1.6, zorder=4)


def ptxt(p):
    return f'p = {p:.0e}'.replace('e-0', 'e-') if p < .01 else f'p = {p:.2f}'


D = pd.read_csv(f'{ROOT}/data/lfp/gamma_envelope_rate.csv')
P = np.load(f'{ROOT}/data/lfp/gamma_envelope_peri.npz')
HW, DEC = int(P['hw']), int(P['dec'])

# The session is the independent unit: a session contributes several envelope
# groups and they see largely the same spikes, so each session is reduced to
# its median across groups before anything is tested.
S = D.groupby(['mouse', 'day', 'band']).median(numeric_only=True).reset_index()
NS_ = S.groupby(['mouse', 'day']).ngroups
print(f'{len(D)} group-bands, {NS_} sessions, '
      f'median {D.n_int.median():.0f} interneurons and '
      f'{D.n_princ.median():.0f} principal cells per envelope group')


def by_band(col):
    return {b: S[S.band == b][col].dropna().values for b, _ in BANDS}


def paired(c1, c2):
    """Session-matched pairs of two columns, per band."""
    out = {}
    for b, _ in BANDS:
        g = S[S.band == b][[c1, c2]].dropna()
        out[b] = (g[c1].values, g[c2].values)
    return out


fig = plt.figure(figsize=(10, 5.1))
G = fig.add_gridspec(2, 3, hspace=.78, wspace=.46, left=.075, right=.985,
                     top=.845, bottom=.115)

# ---- A: the distance control, as three points ------------------------------
ax = fig.add_subplot(G[0, 0]); lp(ax, 'A')
STEPS = [('r_int_own', '≤ 60 µm\n(leak)'), ('r_int', '120–360 µm\n(reported)'),
         ('r_int_far', '> 360 µm')]
for b, lab in BANDS:
    g = S[S.band == b]
    v = [g[c].median() for c, _ in STEPS]
    n = [int(g[c].notna().sum()) for c, _ in STEPS]
    ax.plot(range(3), v, marker='o', ms=4, lw=1.3,
            color=CINT if b == 'fast' else '0.55',
            ls='-' if b == 'fast' else (0, (3, 1.6)), label=f'{b} gamma')
    print(f'  A {b}: ' + ', '.join(f'{l.splitlines()[0]} {x:+.4f} (n={k})'
                                   for (_, l), x, k in zip(STEPS, v, n)))
ax.axhline(0, color='0.6', lw=.7)
ax.set_xticks(range(3))
ax.set_xticklabels([l for _, l in STEPS], fontsize=6.0, linespacing=1.2)
ax.set_xlim(-.3, 2.3)
ax.set_ylabel('partial r, interneuron rate\nwith band amplitude', fontsize=7.5)
for b, _ in BANDS:
    g = S[S.band == b][['r_int_own', 'r_int', 'r_int_far']].dropna()
    print(f'  A {b}: own vs window p = '
          f'{wilcoxon(g.r_int_own, g.r_int).pvalue:.3g}, window vs far p = '
          f'{wilcoxon(g.r_int, g.r_int_far).pvalue:.3g} (n = {len(g)})')
ax.set_title('cell-to-field distance. UNDERPOWERED:\na median of one '
             'interneuron sits ≤ 60 µm', fontsize=7.2, loc='left')
ax.text(.5, -.42, f'own-group estimate from a median of '
        f'{D.n_int_own.median():.0f} interneuron; the paired own-vs-window '
        f'test is null in\nboth bands, so this panel bounds the leak rather '
        f'than measuring it. B onward use 120–360 µm.',
        transform=ax.transAxes, ha='center', va='top', fontsize=5.3,
        color='0.45', linespacing=1.35)
ax.legend(fontsize=5.8, frameon=False, loc='best')
tidy(ax)

# ---- B: partial correlation by class and band ------------------------------
ax = fig.add_subplot(G[0, 1]); lp(ax, 'B')
w = .34
for j, (b, _) in enumerate(BANDS):
    g = S[S.band == b][['r_int', 'r_princ']].dropna()
    for i, (cl, c, lab) in enumerate((('r_int', CINT, 'interneuron'),
                                      ('r_princ', CPRN, 'principal'))):
        v = g[cl].values
        ax.bar(j + (i - .5) * w, np.median(v), w * .9, color=c, lw=0,
               label=lab if j == 0 else None)
        ax.errorbar(j + (i - .5) * w, np.median(v),
                    yerr=[[np.median(v) - np.quantile(v, .25)],
                          [np.quantile(v, .75) - np.median(v)]],
                    fmt='none', ecolor='0.3', lw=.7, capsize=1.5)
    p = wilcoxon(g.r_int, g.r_princ).pvalue
    ax.text(j, ax.get_ylim()[1] * .98, ptxt(p), ha='center', va='top',
            fontsize=5.8, color='0.3')
    print(f'  B {b}: int {g.r_int.median():+.4f} vs princ '
          f'{g.r_princ.median():+.4f}, p = {p:.3g}')
ax.axhline(0, color='0.6', lw=.7)
ax.set_xticks(range(2)); ax.set_xticklabels([l for _, l in BANDS], fontsize=6.5,
                                            linespacing=1.2)
ax.set_ylabel('partial r with band amplitude', fontsize=7.5)
ax.set_title('each population\'s rate against the\namplitude, given the other',
             fontsize=7.2, loc='left')
ax.legend(fontsize=5.8, frameon=False, loc='best')
tidy(ax)

# ---- C: against the phase-randomised null, both populations ----------------
ax = fig.add_subplot(G[0, 2]); lp(ax, 'C')
for j, (b, _) in enumerate(BANDS):
    for i, (c_, col_, lab) in enumerate((('z_int', CINT, 'interneuron'),
                                         ('z_princ', CPRN, 'principal'))):
        v = S[S.band == b][c_].dropna().values
        strip(ax, j + (i - .5) * .34, v, col_, w=.08, seed=j * 2 + i)
        p = wilcoxon(v).pvalue
        print(f'  C {b} {c_}: z median {np.median(v):+.2f}, p = {p:.3g}, '
              f'positive in {int((v > 0).sum())}/{len(v)}')
        if j == 0:
            ax.scatter([], [], s=7, color=col_, label=lab)
ax.axhline(0, color='0.5', lw=.8, ls='--')
ax.set_xticks(range(2)); ax.set_xticklabels([l for _, l in BANDS], fontsize=6.5,
                                            linespacing=1.2)
ax.set_xlim(-.5, 1.5)
ax.set_ylabel('z against phase-randomised rate', fontsize=7.5)
ax.set_title('the only honest null here: same\nspectrum, redrawn phases',
             fontsize=7.2, loc='left')
ax.legend(fontsize=5.8, frameon=False, loc='best')
tidy(ax)

# ---- D: band specificity ---------------------------------------------------
ax = fig.add_subplot(G[1, 0]); lp(ax, 'D')
piv = S.pivot_table(index=['mouse', 'day'], columns='band',
                    values=['r_int', 'r_princ']).dropna()
dif = {'interneuron': (piv[('r_int', 'fast')] - piv[('r_int', 'slow')]).values,
       'principal': (piv[('r_princ', 'fast')] - piv[('r_princ', 'slow')]).values}
for j, (k, c) in enumerate((('interneuron', CINT), ('principal', CPRN))):
    strip(ax, j, dif[k], c, seed=10 + j)
    print(f'  D {k}: fast minus slow {np.median(dif[k]):+.4f}, '
          f'p = {wilcoxon(dif[k]).pvalue:.3g}')
pint = wilcoxon(dif['interneuron'], dif['principal']).pvalue
ax.text(.5, ax.get_ylim()[1], f'interaction\n{ptxt(pint)}', ha='center',
        va='top', fontsize=6, color='0.2', linespacing=1.2)
print(f'  D interaction: p = {pint:.3g}')
ax.axhline(0, color='0.5', lw=.8, ls='--')
ax.set_xticks(range(2)); ax.set_xticklabels(['interneuron', 'principal'],
                                            fontsize=6.5)
ax.set_xlim(-.5, 1.5)
ax.set_ylabel('partial r, fast − slow', fontsize=7.5)
ax.set_title('does the fast band depend on the\ninterneurons more than the '
             'slow?', fontsize=7.2, loc='left')
tidy(ax)

# ---- E: peri-event population rate -----------------------------------------
ax = fig.add_subplot(G[1, 1]); lp(ax, 'E')
tt = np.arange(-HW, HW + 1) * DEC
for b, _ in BANDS:
    for cl, c in (('int', CINT), ('princ', CPRN)):
        k = f'{b}_{cl}'
        if k not in P:
            continue
        M = P[k]
        m, e = M.mean(0), M.std(0) / np.sqrt(len(M))
        ax.fill_between(tt, m - e, m + e, color=c, alpha=.18, linewidth=0,
                        edgecolor='none')
        ax.plot(tt, m, color=c, lw=1.3, ls='-' if b == 'fast' else (0, (3, 1.6)),
                label=f'{cl} {b}')
        print(f'  E {k}: peak {m.max():+.3f} at {tt[int(np.argmax(m))]:+d} ms')
ax.axvline(0, color='0.6', lw=.7, ls=':')
ax.set_xlabel('time from amplitude peak (ms)', fontsize=7.5)
ax.set_ylabel('population rate (z)', fontsize=7.5)
ax.set_title('solid fast, dashed slow; orange\ninterneuron, grey principal',
             fontsize=7.2, loc='left')
tidy(ax)

# ---- F: by anchoring state -------------------------------------------------
ax = fig.add_subplot(G[1, 2]); lp(ax, 'F')
labs, prs = [], []
for b, _ in BANDS:
    for cl, who in (('r_int', 'int'), ('r_princ', 'princ')):
        g = S[S.band == b][[f'{cl}_a', f'{cl}_n']].dropna()
        labs.append(f'{BSHORT[b]}\n{who}'); prs.append((g.iloc[:, 0].values,
                                                        g.iloc[:, 1].values))
for i, (a_, n_) in enumerate(prs):
    ax.bar(i - .18, np.median(a_), .34, color=ANCH_COLOR, lw=0,
           label='anchored' if i == 0 else None)
    ax.bar(i + .18, np.median(n_), .34, color=NONANCH_COLOR, lw=0,
           label='non-anchored' if i == 0 else None)
raw = [wilcoxon(a_, n_).pvalue for a_, n_ in prs]
order = np.argsort(raw); run_ = 0.0; holm = np.empty(len(raw))
for r_, i_ in enumerate(order):
    run_ = max(run_, raw[i_] * (len(raw) - r_)); holm[i_] = min(run_, 1.0)
for i, h in enumerate(holm):
    ax.text(i, max(np.median(prs[i][0]), np.median(prs[i][1])), f'{h:.2f}',
            ha='center', va='bottom', fontsize=5.8, color='0.3')
print('  F Holm: ' + ', '.join(f'{l.replace(chr(10), " ")} {h:.3f}'
                               for l, h in zip(labs, holm)))
ax.axhline(0, color='0.6', lw=.7)
ax.set_xticks(range(len(labs))); ax.set_xticklabels(labs, fontsize=6,
                                                    linespacing=1.2)
ax.set_ylabel('partial r with band amplitude', fontsize=7.5)
ax.set_title('and whether any of it moves with\nthe anchoring state',
             fontsize=7.2, loc='left')
ax.legend(fontsize=5.8, frameon=False, loc='best')
tidy(ax)

fig.text(.075, .965, 'Is either gamma band built out of fast-spiking '
         'interneuron firing?', fontsize=9, weight='bold')
fig.text(.075, .925, 'Population rate against band amplitude, in 5 ms bins, '
         'running samples only, with the other population\'s rate, running '
         'speed and theta amplitude partialled out.', fontsize=6.6,
         color='0.35')
fig.savefig(OUT)
print(f'\nwrote {OUT}')
