"""Figure 7 supplement — where in the theta cycle each frequency peaks.

The band analysis in Figure 7 fixes two gamma bands in advance and asks how
strongly each is nested in theta. It cannot ask WHERE in the cycle they peak,
because the modulation index discards phase. This does, as a continuous
function of frequency, for both anchoring states.

    A, B  the phase-frequency map for each state, over 28 sessions
    C     their difference, which is where the state effect would live
    D     the same information reduced to preferred phase against frequency
    E-G   the same question against DEPTH: theta-triggered LFP and CSD down
          one shank, both states overlaid, for three example sessions
    H     the laminar phase gradient over 27 sessions, by layer

WHAT IT SHOWS. The relationship is a GRADIENT, not a step between two bands:
preferred phase runs smoothly from about 130 deg of theta at 20-30 Hz to about
41 deg at 60-100 Hz. Paired within session, that separation holds in each state
on its own -- +34 deg when anchored (p = 5e-4, 24/28 sessions) and +52 deg when
not (p = 1e-5, 26/28). A two-band test cannot show it.

THE STATE MOVES IT VERY LITTLE, AND NOT RELIABLY. An earlier version of this
docstring claimed the gradient sits 'roughly 20 deg later in the cycle in the
non-anchored state' and that the slow-minus-fast separation is 'itself
state-dependent'. Both came from comparing two grand-average curves, which is
not a test: they are two circular means over the SAME 28 sessions, and the gap
between them is not the mean of the per-session gaps once phases wrap. Paired
within session the anchored state prefers a phase 6.6 deg earlier in the slow
band (p = 0.33) and 2.8 deg earlier in the fast band (p = 0.16), and the
separation does not differ between states (-7.3 deg, p = 0.27). Panel D carries
these as n.s.

DEPTH IS THE OTHER PLACE A STATE EFFECT COULD HIDE, and it does not. Theta
advances by about 100 deg across the entorhinal layers -- the laminar signature
of the travelling wave -- and the gradient is identical in the two states:
referenced to each session's own layer 2, every layer differs by less than 3
deg with p >= 0.06 (Wilcoxon, 27 sessions). The theta-triggered averages in
E-G carry both states on the same axes and they superimpose.

WHY A AND B LOOK IDENTICAL. They nearly are: the two maps share 94% of their
pixel variance, the maps themselves deviate from uniform by only about 3%, and
the state difference is about a quarter of that again. No eye differences two
images at that ratio, which is what panel C is for -- and both states'
preferred-phase ridges are now drawn on BOTH maps so the comparison does not
depend on seeing it in the colour.

EACH FREQUENCY ROW IS NORMALISED TO SUM TO ONE, so colour is where a frequency
peaks and not how much power it carries. Without that the 1/f background would
dominate every row and the panels would redraw the spectrum.

MAINS DOES NOT DRIVE THESE PANELS, and the normalisation is why: a strong line
adds no colour once its row is normalised. Measured, the 48-52 Hz band is if
anything LESS modulated than its neighbours (0.0081 against 0.0102), which is
what a line uncorrelated with theta does -- it dilutes the coupling rather than
creating it. The band is marked on the panels so the reader can see which row
it is.

COLOUR LIMITS ARE THE 1ST AND 99TH PERCENTILES of the two maps, symmetric about
uniform. The maps deviate from uniform by only about 3%, so a scale chosen by
eye is likely to be far too wide: an earlier version ran +/-14% and the
structure filled a fifth of the colour range.

Computed by theta_phase_comodulogram.py. Writes fig7_supp_comodulogram.pdf
"""
import os

import matplotlib
import numpy as np
from scipy.stats import wilcoxon

matplotlib.use('Agg')
import matplotlib.pyplot as plt

plt.rcParams['font.family'] = 'Arial'
plt.rcParams['mathtext.fontset'] = 'custom'
plt.rcParams['mathtext.rm'] = 'Arial'
plt.rcParams['mathtext.bf'] = 'Arial:bold'
plt.rcParams['pdf.fonttype'] = 42

ROOT = '/Users/harryclark/Documents/spatial-manifolds'
FIG = os.path.dirname(os.path.abspath(__file__))
OUT = f'{FIG}/fig7_supp_comodulogram.pdf'
ANCH_COLOR, NONANCH_COLOR = '#a8559e', '#3f9b8f'


def lp(ax, s, x=-.20, y=1.0):
    ax.text(x, y, s, transform=ax.transAxes, fontsize=10, weight='bold',
            va='bottom', ha='right')


z = np.load(f'{ROOT}/data/lfp/theta_phase_comodulogram.npz')
A, N, f = z['anch'], z['non'], z['freqs']
NP = int(z['n_phase'])
cent = np.linspace(-np.pi, np.pi, NP + 1)[:-1] + np.pi / NP
deg = np.degrees(cent)
NS = A.shape[0]


def pref(M):
    """Preferred phase per frequency, on [0, 360).

    The standard [-180, 180) branch cut falls straight through these values,
    which lie between about +40 and +180 deg, and put a full-axis jump at the
    lowest frequency.
    """
    return np.degrees(np.angle((M * np.exp(1j * cent)[None, :]).sum(1))) % 360.


def ridge(M):
    """Preferred phase per frequency on the panels' own [-180, 180) axis.

    Broken with NaN wherever consecutive frequencies jump more than half a
    cycle, so a wrap at the branch cut does not draw a line straight across
    the map.
    """
    p_ = (pref(M) + 180.) % 360. - 180.
    out = p_.astype(float).copy()
    brk = np.abs(np.diff(p_)) > 180.
    out[1:][brk] = np.nan
    return out


mA, mN = A.mean(0) * NP, N.mean(0) * NP
both = np.concatenate([mA, mN])
vmin, vmax = np.percentile(both, 1), np.percentile(both, 99)
hw = max(1 - vmin, vmax - 1)
vmin, vmax = 1 - hw, 1 + hw
print(f'{NS} sessions, {len(f)} frequencies, {NP} phase bins')
print(f'colour limits {vmin:.3f} - {vmax:.3f}')

fig = plt.figure(figsize=(9.6, 6.6))
_outer = fig.add_gridspec(2, 1, height_ratios=[1, 1], hspace=.46,
                          left=.06, right=.975, top=.915, bottom=.075)
gs = _outer[0].subgridspec(1, 4, width_ratios=[1, 1, 1, 1.18], wspace=.60)
gs2 = _outer[1].subgridspec(1, 4, width_ratios=[1, 1, 1, 1.18], wspace=.60)

for k, (M, tag, col) in enumerate(((mA, 'anchored', ANCH_COLOR),
                                   (mN, 'non-anchored', NONANCH_COLOR))):
    ax = fig.add_subplot(gs[k])
    im = ax.pcolormesh(deg, f, M, cmap='magma', vmin=vmin, vmax=vmax,
                       shading='nearest', rasterized=True)
    # BOTH ridges on BOTH maps. The two states share 94% of their pixel
    # variance, so side by side they are indistinguishable by eye; drawing the
    # other state's preferred phase on top of each map puts the difference --
    # the only thing that differs -- in one place where it can be seen.
    for _M2, _c2, _ls, _lab in ((mA, '#f2c3ec', '-', 'anchored'),
                                (mN, '#9ff0e4', (0, (2.2, 1.4)),
                                 'non-anchored')):
        ax.plot(ridge(_M2), f, color=_c2, lw=1.1, ls=_ls, zorder=5,
                label=_lab if k == 0 else None)
    if k == 0:
        ax.legend(fontsize=4.8, frameon=False, loc='upper left',
                  labelcolor='w', handlelength=1.4, borderpad=.1,
                  handletextpad=.4)
    ax.axhspan(48, 52, facecolor='none', edgecolor='w', lw=.5, ls=(0, (2, 2)))
    ax.text(178, 50, 'mains', fontsize=5.2, color='w', ha='right', va='center')
    ax.set_xticks([-180, -90, 0, 90, 180])
    ax.set_xlabel('theta phase (deg)', fontsize=7.5)
    if k == 0:
        ax.set_ylabel('frequency (Hz)', fontsize=7.5)
    else:
        ax.tick_params(labelleft=False)
    ax.set_title(tag, fontsize=8, color=col)
    ax.tick_params(labelsize=6.5)
    lp(ax, 'AB'[k])
cb = fig.colorbar(im, ax=ax, fraction=.045, pad=.03)
cb.ax.set_ylabel('amplitude (1 = uniform)', fontsize=5.4, labelpad=2)
cb.ax.tick_params(labelsize=5.4); cb.outline.set_visible(False)

# ---- C: the difference, which is where the state effect is ------------------
ax = fig.add_subplot(gs[2])
D = mA - mN
lim = np.percentile(np.abs(D), 99)
imd = ax.pcolormesh(deg, f, D, cmap='RdBu_r', vmin=-lim, vmax=lim,
                    shading='nearest', rasterized=True)
ax.set_xticks([-180, -90, 0, 90, 180])
ax.set_xlabel('theta phase (deg)', fontsize=7.5)
ax.tick_params(labelsize=6.5)
ax.set_title('anchored − non-anchored', fontsize=8, color='0.25')
# the state effect here is a phase SHIFT, which shows as a dipole rather than a
# blob: anchored higher on the earlier flank, lower on the later one, strongest
# where the amplitude itself is (50-75 Hz). It is a small difference on a noisy
# map and the caption says so.
ax.text(.5, -.30, 'a shift, not a gain: red earlier, blue later',
        transform=ax.transAxes, fontsize=5.8, color='0.45', ha='center')
lp(ax, 'C')
cbd = fig.colorbar(imd, ax=ax, fraction=.045, pad=.03)
cbd.ax.set_ylabel('difference', fontsize=5.4, labelpad=2)
cbd.ax.tick_params(labelsize=5.4); cbd.outline.set_visible(False)

# ---- the paired test behind panel D -----------------------------------------
# One preferred phase per session per band per state, then the WITHIN-SESSION
# difference. The grand-average curves cannot be tested against each other --
# they are two circular means over the same 28 sessions, and the gap between
# them is not the mean of the per-session gaps once phases wrap.
SLOW, FAST = (30., 48.), (60., 100.)


def band_phase(M, lo, hi):
    """Preferred theta phase per session, over one frequency band."""
    m = (f >= lo) & (f <= hi)
    r = M[:, m, :].mean(1)
    return np.angle((r * np.exp(1j * cent)[None, :]).sum(1))


def paired(a, b):
    """Wrapped within-session difference, its circular mean, and a p value."""
    d = np.degrees(np.angle(np.exp(1j * (a - b))))
    return (float(np.degrees(np.angle(np.mean(np.exp(1j * np.radians(d)))))),
            float(np.median(d)), float(wilcoxon(d).pvalue), d)


def star(p):
    return ('n.s.' if not np.isfinite(p) or p > .05 else
            '*' if p > .01 else '**' if p > .001 else '***')


STATS = {}
for _nm, (_lo, _hi) in (('slow', SLOW), ('fast', FAST)):
    _pa, _pn = band_phase(A, _lo, _hi), band_phase(N, _lo, _hi)
    STATS[_nm] = paired(_pa, _pn)
    print(f'  {_nm} {_lo:.0f}-{_hi:.0f} Hz, anchored - non-anchored: '
          f'circular mean {STATS[_nm][0]:+.1f} deg, median {STATS[_nm][1]:+.1f}, '
          f'p = {STATS[_nm][2]:.3g} ({star(STATS[_nm][2])}, n = {NS})')
# the gradient itself, which is what the panel is for, IS present in each state
for _tag, _M in (('anchored', A), ('non-anchored', N)):
    _m, _md, _p, _ = paired(band_phase(_M, *SLOW), band_phase(_M, *FAST))
    print(f'  slow-minus-fast separation, {_tag}: {_m:+.1f} deg '
          f'(median {_md:+.1f}, p = {_p:.3g})')
_m, _md, _p, _ = paired(band_phase(A, *SLOW) - band_phase(A, *FAST),
                        band_phase(N, *SLOW) - band_phase(N, *FAST))
print(f'  is that separation state-dependent? {_m:+.1f} deg, p = {_p:.3g} '
      f'({star(_p)})')

# ---- D: preferred phase against frequency -----------------------------------
ax = fig.add_subplot(gs[3])
for M, c, lab in ((A, ANCH_COLOR, 'anchored'), (N, NONANCH_COLOR, 'non-anchored')):
    per = np.array([pref(M[i]) for i in range(M.shape[0])])
    mu = np.degrees(np.angle(np.mean(np.exp(1j * np.radians(per)), 0))) % 360.
    se = np.degrees(np.std(np.radians(per), 0) / np.sqrt(len(per)))
    ax.fill_between(f, mu - se, mu + se, color=c, alpha=.28, lw=0)
    ax.plot(f, mu, color=c, lw=1.5, label=lab)
ax.axvspan(*SLOW, color='0.88', alpha=.6, lw=0, zorder=0)
ax.axvspan(*FAST, color='0.88', alpha=.6, lw=0, zorder=0)
# the state comparison, tested within session, over the band it belongs to
for _nm, (_lo, _hi) in (('slow', SLOW), ('fast', FAST)):
    _mu, _mdn, _p, _ = STATS[_nm]
    _x = (_lo + _hi) / 2
    ax.text(_x, 199, _nm, fontsize=6, color='0.45', ha='center')
    ax.text(_x, 186, star(_p), fontsize=6.6, color='0.3', ha='center',
            va='center')
    ax.text(_x, 174, f'{_mu:+.0f}\u00b0', fontsize=5.4, color='0.45',
            ha='center', va='center')
ax.set_xlabel('frequency (Hz)', fontsize=7.5)
ax.set_ylabel('preferred theta phase (deg)', fontsize=7.5)
ax.set_title('a gradient, not two bands', fontsize=8, color='0.25')
ax.text(.5, -.30, 'anchored − non-anchored, paired by session',
        transform=ax.transAxes, fontsize=5.8, color='0.45', ha='center')
ax.set_ylim(0, 210)
ax.legend(fontsize=6, frameon=False, loc='lower left')
ax.tick_params(labelsize=6.5)
ax.spines[['top', 'right']].set_visible(False)
lp(ax, 'D', x=-.26)

# =============================================================================
# E-H: the same question against DEPTH rather than against frequency
# =============================================================================
# Theta in MEC is not in phase along the probe: it advances through the layers,
# which is the laminar signature of the entorhinal travelling wave. If the
# anchoring state reorganised the circuit, that gradient is somewhere it could
# show. Computed by theta_depth_profile.py.
DZ = 60.0                      # um between channel groups, exactly, every session
EXAMPLES = ['M21D21', 'M25D23', 'M28D19']
LAYERS = ['ENTm1', 'ENTm2', 'ENTm3', 'ENTm5', 'ENTm6']
REF_LAYER = 'ENTm2'            # present in 24 of 27 sessions; see below

zd = np.load(f'{ROOT}/data/lfp/theta_depth_profile.npz', allow_pickle=True)
SESS = sorted({k.split('__')[0] for k in zd.files})
tt = (np.arange(zd[f'{SESS[0]}__trig_anch'].shape[1])
      - zd[f'{SESS[0]}__trig_anch'].shape[1] // 2) / 1000.


def csd(M):
    """Current source density across depth, sinks positive.

    The first and last rows are returned as NaN. A second spatial derivative
    has no defined value at the ends of the array, and np.gradient's one-sided
    estimate there is large enough to set the colour limits and drain the
    interior of the map, which is the part being looked at.
    """
    sm = np.apply_along_axis(
        lambda v: np.convolve(v, np.array([.25, .5, .25]), 'same'), 0, M)
    C = -np.gradient(np.gradient(sm, DZ, axis=0), DZ, axis=0)
    C[[0, -1]] = np.nan
    return C


for _k, _s in enumerate(EXAMPLES):
    ax = fig.add_subplot(gs2[_k])
    y = zd[f'{_s}__y']; lay = zd[f'{_s}__layer']
    ta, tn = zd[f'{_s}__trig_anch'], zd[f'{_s}__trig_non']
    C = csd(ta)
    # colour limits from the INTERIOR. Theta is several times larger at the
    # ends of the entorhinal span than in the middle, so limits taken over the
    # whole map leave the laminar structure the panel is for washed out.
    lim = np.nanpercentile(np.abs(C[2:-2]), 97)
    ax.pcolormesh(tt, y, C, cmap='RdBu_r', vmin=-lim, vmax=lim,
                  shading='nearest', rasterized=True)
    sc = DZ * 1.1 / max(np.abs(ta).max(), np.abs(tn).max())
    for _M, _c, _ls in ((tn, '#1d6b62', (0, (2.2, 1.4))), (ta, '#7d2f74', '-')):
        for i in range(len(y)):
            ax.plot(tt, y[i] + sc * _M[i], color=_c, lw=.45, ls=_ls,
                    zorder=3 if _c.startswith('#7d') else 2)
    # name the layers down the right-hand edge rather than tick every group
    for L in LAYERS:
        m = lay == L
        if m.sum() >= 2:
            ax.text(.205, y[m].mean(), L.replace('ENTm', 'L'), fontsize=5.2,
                    color='0.3', va='center', ha='left')
    # probe_y runs from the tip upward, so plotting it increasing UP puts the
    # tip at the bottom, which is the way a depth profile is read
    ax.set_xlim(-.2, .2); ax.set_ylim(y.min() - DZ, y.max() + DZ)
    ax.set_xticks([-.2, 0, .2])
    ax.set_xlabel('time from theta peak (s)', fontsize=7.5)
    if _k == 0:
        ax.set_ylabel('depth along probe (\u00b5m)', fontsize=7.5)
    ax.set_title(f'{_s[:3]} {_s[3:]}', fontsize=7.5, color='0.25')
    ax.tick_params(labelsize=6.5)
    lp(ax, 'EFG'[_k])

# ---- H: the laminar phase gradient, every session, both states --------------
# Phase is referenced to each session's own layer 2 groups. The raw reference
# is the probe tip, which sits in a different lamina in different penetrations,
# so averaging about it would average about a moving origin.
rows = {L: {'a': [], 'n': []} for L in LAYERS}
for _s in SESS:
    lay = zd[f'{_s}__layer']
    if (lay == REF_LAYER).sum() == 0:
        continue
    for _tag, _key in (('a', 'phase_anch'), ('n', 'phase_non')):
        ph = zd[f'{_s}__{_key}']
        ref = np.angle(np.mean(np.exp(1j * ph[lay == REF_LAYER])))
        for L in LAYERS:
            m = lay == L
            rows[L][_tag].append(
                np.angle(np.mean(np.exp(1j * (ph[m] - ref)))) if m.sum()
                else np.nan)

ax = fig.add_subplot(gs2[3])
xs = np.arange(len(LAYERS))
PH = {t: np.array([np.array(rows[L][t], float) for L in LAYERS]) for t in 'an'}
for _t, _c, _lab in (('a', ANCH_COLOR, 'anchored'),
                     ('n', NONANCH_COLOR, 'non-anchored')):
    mu = np.array([np.degrees(np.angle(np.nanmean(np.exp(1j * v[np.isfinite(v)]))))
                   for v in PH[_t]])
    se = np.array([np.degrees(np.std(v[np.isfinite(v)]) /
                              np.sqrt(max(np.isfinite(v).sum(), 1)))
                   for v in PH[_t]])
    ax.fill_between(xs, mu - se, mu + se, color=_c, alpha=.28, lw=0)
    ax.plot(xs, mu, color=_c, lw=1.5, marker='o', ms=3, label=_lab)
print('  laminar theta phase, referenced to each session\'s own ENTm2:')
for i, L in enumerate(LAYERS):
    a, n_ = PH['a'][i], PH['n'][i]
    m = np.isfinite(a) & np.isfinite(n_)
    d = np.degrees(np.angle(np.exp(1j * (a[m] - n_[m]))))
    pv = wilcoxon(d).pvalue if m.sum() >= 5 else np.nan
    print(f'    {L}: n={int(m.sum()):>3}  anchored '
          f'{np.degrees(np.angle(np.mean(np.exp(1j * a[m])))):+7.1f} deg  '
          f'non {np.degrees(np.angle(np.mean(np.exp(1j * n_[m])))):+7.1f}  '
          f'delta {np.degrees(np.angle(np.mean(np.exp(1j * np.radians(d))))):+5.1f}  '
          f'p = {pv:.3g} ({star(pv)})')
    ax.text(i, .97, star(pv), fontsize=6.4, color='0.3', ha='center',
            va='top', transform=ax.get_xaxis_transform())
ax.axhline(0, color='0.7', lw=.7, ls=':')
ax.set_xticks(xs)
ax.set_xticklabels([f"{L.replace('ENTm', 'L')}\n{int(np.isfinite(PH['a'][i]).sum())}"
                    for i, L in enumerate(LAYERS)], fontsize=6.5,
                   linespacing=1.3)
ax.set_xlabel('entorhinal layer, and sessions with it', fontsize=7,
              labelpad=1)
ax.set_ylabel('theta phase re layer 2 (deg)', fontsize=7.5)
ax.set_title('the gradient is there and\ndoes not move', fontsize=8,
             color='0.25')
ax.set_ylim(-70, 78)
ax.legend(fontsize=6, frameon=False, loc='lower left')
ax.tick_params(labelsize=6.5)
ax.spines[['top', 'right']].set_visible(False)
lp(ax, 'H', x=-.26)

fig.suptitle(f'Where in the theta cycle each frequency peaks, and how little it '
             f'moves with the anchoring state ({NS} sessions; '
             f'{len(SESS)} with a laminar profile)',
             fontsize=8.6, x=.06, ha='left', y=.985)
plt.savefig(OUT, dpi=220, bbox_inches='tight')
plt.close(fig)
print(f'wrote {OUT}')
