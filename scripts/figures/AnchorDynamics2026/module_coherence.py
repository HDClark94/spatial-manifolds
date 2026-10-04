"""Does the grid attractor lose coherence in the non-anchored state, or only its anchor?

Position decoding already answered one half of this: the non-anchored state is a
DEGRADED code, not a re-referenced one (`position_decoding_by_anchoring`). N->N
buys nothing over A->N, and the drift component is small. So the map is not
simply moving.

That leaves two possibilities which decoding cannot separate, because both
degrade decoding:

  ATTRACTOR INTACT, UNANCHORED -- co-modular grid cells keep their fixed relative
    phase while the whole sheet wanders, within and between trials. Absolute
    phase is unstable; RELATIVE phase is not. Decoding to the track frame fails
    because the frame is lost, but the network's internal structure survives.

  ATTRACTOR DECOHERES -- the relative phases themselves scramble. There is no
    sheet to anchor. Decoding fails because the code fails.

Relative phase is the discriminating measure, and it is strictly more sensitive
than decoding: it never needs absolute position, never needs a decoder to be
well-fit, and is estimated pairwise so a handful of co-modular cells suffices.

PHASE, NOT CROSS-CORRELATION LAG. A grid cell's track response is periodic, so a
circular cross-correlation over the whole 200 cm ring has several near-equal
peaks and argmax picks among them arbitrarily, manufacturing variance. Phase at
the module's own spatial period is defined modulo exactly that period, which is
the modular structure a grid actually has:

    phi = angle( sum_x profile(x) exp(-2 pi i x / P) )

The period P is estimated per cell from its ANCHORED template and taken as the
module median, since co-modular cells share a period by construction.

THE FLOOR IS THE CONTROL. Noisier trials give noisier phase, and non-anchored
trials are noisier, so both relative and absolute concentration must fall for
reasons that have nothing to do with the attractor. Two floors are therefore
computed from the same trials, with the same noise:

  CROSS-MODULE pairs -- two grid cells from DIFFERENT modules in the same
    session. Independent attractors, so their relative phase is unconstrained by
    construction. This is what "no relative-phase structure" looks like at this
    noise level.
  TRIAL-SHUFFLED within-module pairs -- the same cells, phases paired across
    mismatched trials.

Within-module relative concentration landing at the floor in the non-anchored
state means decoherence; landing well above it means the sheet survived.

Outputs
  data/population_state/module_coherence.csv
  fig_module_coherence.pdf
"""
import json
import os
import sys
import warnings

import numpy as np
import pandas as pd

warnings.filterwarnings('ignore')
import matplotlib

matplotlib.use('Agg')
import matplotlib.pyplot as plt
import pynapple as nap
from scipy.stats import wilcoxon

sys.path.insert(0, '/Users/harryclark/Documents/spatial-manifolds/src')
from spatial_manifolds.anchoring import (ANCH_COLOR, NONANCH_COLOR,
                                         load_session_labels, smooth_nanaware)

plt.rcParams['font.family'] = 'Arial'
plt.rcParams['pdf.fonttype'] = 42

ROOT = '/Users/harryclark/Documents/spatial-manifolds'
FIG = f'{ROOT}/scripts/figures/AnchorDynamics2026'
PS = f'{ROOT}/data/population_state'
OUT_CSV = f'{PS}/module_coherence.csv'
OUT_PDF = f'{FIG}/fig_module_coherence.pdf'

SIGMA = 2.0
MIN_MOD_CELLS = 3        # co-modular grid cells before a module is used
MIN_STATE = 8            # trials in a state before its concentration is trusted
MIN_MOD_STRENGTH = .05   # a trial with no periodic modulation has no phase
P_RANGE = (25., 150.)    # cm; plausible track periods for a grid module
GRID_C = '#c04744'

# shared session plumbing (vr_paths, clip_trials, TL, NBIN) -- same trick as
# build_trial_maps.py, so this script cannot drift from the rest of the pipeline
_g = {}
for _c in json.load(open(f'{FIG}/lick_raster_by_trial_type.ipynb'))['cells']:
    if _c['cell_type'] != 'code':
        continue
    _s = ''.join(_c['source'])
    if _s.startswith('MOUSE, DAY') or 'INV = pd.DataFrame' in _s:
        continue
    exec(compile(_s, '<nb>', 'exec'), _g)
globals().update({k: v for k, v in _g.items() if not k.startswith('__')})

CLS = pd.read_csv(f'{ROOT}/data/cell_classifications.csv')
GC = CLS[(CLS.cell_type == 'GC') & (CLS.grid_module.notna())]


def dominant_period(profile, bin_cm):
    """Spatial period of the strongest Fourier component inside P_RANGE."""
    x = np.asarray(profile, float)
    x = x - np.nanmean(x)
    if not np.isfinite(x).all() or np.nanstd(x) == 0:
        return np.nan
    n = len(x)
    F = np.abs(np.fft.rfft(x))
    k = np.arange(len(F))
    with np.errstate(divide='ignore'):
        per = np.where(k > 0, n * bin_cm / np.maximum(k, 1), np.inf)
    ok = (per >= P_RANGE[0]) & (per <= P_RANGE[1])
    if not ok.any():
        return np.nan
    return float(per[ok][np.argmax(F[ok])])


def phase_and_strength(profile, period, bin_cm):
    """Matched-filter phase at `period`, and the modulation strength behind it."""
    x = np.asarray(profile, float)
    if not np.isfinite(x).all() or np.nansum(np.abs(x)) == 0:
        return np.nan, 0.
    pos = np.arange(len(x)) * bin_cm
    z = np.sum(x * np.exp(-2j * np.pi * pos / period))
    return float(np.angle(z)), float(np.abs(z) / np.sum(np.abs(x)))


def R(angles):
    """Circular resultant length."""
    a = np.asarray(angles, float)
    a = a[np.isfinite(a)]
    return np.nan if len(a) < MIN_STATE else float(np.abs(np.mean(np.exp(1j * a))))


def session(mo, dy):
    z = load_session_labels(mo, dy)
    if z is None:
        return []
    g = GC[(GC.mouse == mo) & (GC.day == dy)]
    if g.grid_module.nunique() == 0:
        return []
    bp, cp = vr_paths(mo, dy)
    if not (os.path.exists(bp) and os.path.exists(cp)):
        return []
    beh = nap.load_file(bp); clusters = nap.load_file(cp)
    trials_all = beh['trials'].as_dataframe()
    trials, orig = clip_trials(trials_all, clusters)
    keep = np.isin(trials_all.number.values.astype(int), orig)
    tn, trav = beh['trial_number'], beh['travel']
    dt = trav - (float(np.asarray(tn.values)[0]) - 1) * TL
    moving = beh['S'].threshold(3.0, method='above').time_support
    n_all = len(trials_all)
    bin_cm = TL / NBIN

    want = [int(c) for c in g.cluster_id]
    have = [c for c in want if c in set(int(x) for x in clusters.index)]
    if len(have) < MIN_MOD_CELLS:
        return []
    tc = nap.compute_1d_tuning_curves(clusters[have], dt, nb_bins=n_all * NBIN,
                                      minmax=[0, n_all * TL], ep=moving)
    # population state, not the per-cell label: the question is what the module
    # does while the POPULATION is in each state
    pop = np.asarray(z['frac_anch'], float) > .5

    mod = {int(r.cluster_id): int(r.grid_module) for _, r in g.iterrows()}
    S, PH, ST = {}, {}, {}
    for c in have:
        Mi = np.asarray(tc[c]).reshape(n_all, NBIN)[keep]
        S[c] = np.array([smooth_nanaware(r, sigma=SIGMA) for r in Mi])
    if len(pop) != S[have[0]].shape[0]:
        return []

    # period per cell from its anchored template, then the module median
    per_cell = {}
    for c in have:
        if pop.sum() >= MIN_STATE:
            per_cell[c] = dominant_period(np.nanmean(S[c][pop], axis=0), bin_cm)
    mods = {}
    for c, m in mod.items():
        if c in per_cell and np.isfinite(per_cell[c]):
            mods.setdefault(m, []).append(c)
    out = []
    mod_period = {m: float(np.nanmedian([per_cell[c] for c in cs]))
                  for m, cs in mods.items() if len(cs) >= MIN_MOD_CELLS}
    if not mod_period:
        return []

    for m, P in mod_period.items():
        for c in mods[m]:
            ph, st = zip(*[phase_and_strength(S[c][t], P, bin_cm)
                           for t in range(S[c].shape[0])])
            PH[c] = np.array(ph); ST[c] = np.array(st)

    rng = np.random.default_rng(abs(hash((mo, dy))) % 2 ** 32)

    def pair_rows(c1, c2, within, P):
        ok = (ST[c1] >= MIN_MOD_STRENGTH) & (ST[c2] >= MIN_MOD_STRENGTH)
        # relative phase is defined modulo the module period
        d = PH[c1] - PH[c2]
        rec = dict(mouse=mo, day=dy, c1=c1, c2=c2, within=within, period=P)
        for tag, msk in (('anch', pop & ok), ('non', ~pop & ok)):
            rec[f'n_{tag}'] = int(msk.sum())
            rec[f'Rrel_{tag}'] = R(d[msk])
            # trial-shuffled floor on the SAME trials, same noise
            sh = []
            for _ in range(20):
                i1 = np.where(msk)[0]
                if len(i1) < MIN_STATE:
                    sh.append(np.nan); continue
                sh.append(R(PH[c1][i1] - PH[c2][rng.permutation(i1)]))
            rec[f'Rrel_{tag}_shuf'] = float(np.nanmean(sh))
        return rec

    for m, cs in mods.items():
        if len(cs) < MIN_MOD_CELLS or m not in mod_period:
            continue
        P = mod_period[m]
        for i in range(len(cs)):
            for j in range(i + 1, len(cs)):
                out.append(pair_rows(cs[i], cs[j], 1, P))
    ms = [m for m in mod_period]
    if len(ms) >= 2:
        for a in range(len(ms)):
            for b in range(a + 1, len(ms)):
                # cross-module pairs scored at the FIRST module's period, so the
                # statistic is constructed identically to the within-module one
                P = mod_period[ms[a]]
                for c1 in mods[ms[a]]:
                    for c2 in mods[ms[b]]:
                        ph2, st2 = zip(*[phase_and_strength(S[c2][t], P, bin_cm)
                                         for t in range(S[c2].shape[0])])
                        PH[c2], ST[c2] = np.array(ph2), np.array(st2)
                        out.append(pair_rows(c1, c2, 0, P))

    # absolute phase, per cell: concentration against its own anchored mean
    for m, cs in mods.items():
        if len(cs) < MIN_MOD_CELLS or m not in mod_period:
            continue
        for c in cs:
            ok = ST[c] >= MIN_MOD_STRENGTH
            rec = dict(mouse=mo, day=dy, c1=c, c2=-1, within=2,
                       period=mod_period[m])
            for tag, msk in (('anch', pop & ok), ('non', ~pop & ok)):
                rec[f'n_{tag}'] = int(msk.sum())
                rec[f'Rrel_{tag}'] = R(PH[c][msk])
                rec[f'Rrel_{tag}_shuf'] = np.nan
            out.append(rec)
    return out


if __name__ == '__main__':
    sess = sorted(set(zip(GC.mouse, GC.day)))
    allr = []
    for k, (mo, dy) in enumerate(sess, 1):
        try:
            r = session(int(mo), int(dy))
        except Exception as e:
            print(f'  ! M{mo}D{dy}: {type(e).__name__}: {e}', flush=True)
            continue
        allr += r
        if r:
            print(f'[{k}/{len(sess)}] M{mo}D{dy}: {len(r)} rows', flush=True)
    D = pd.DataFrame(allr)
    D.to_csv(OUT_CSV, index=False)
    print(f'\nwrote {OUT_CSV}  ({len(D)} rows)')


# ============================== figure =========================================
def make_figure():
    D = pd.read_csv(OUT_CSV)
    W = D[D.within == 1].dropna(subset=['Rrel_anch', 'Rrel_non'])
    X = D[D.within == 0].dropna(subset=['Rrel_anch', 'Rrel_non'])
    A = D[D.within == 2].dropna(subset=['Rrel_anch', 'Rrel_non'])
    # sessions, not pairs: pairs inside a session share cells and a state sequence
    ws = W.groupby(['mouse', 'day'])[['Rrel_anch', 'Rrel_non']].mean()
    xs = X.groupby(['mouse', 'day'])[['Rrel_anch', 'Rrel_non']].mean()
    as_ = A.groupby(['mouse', 'day'])[['Rrel_anch', 'Rrel_non']].mean()
    sh = W.groupby(['mouse', 'day'])[['Rrel_anch_shuf', 'Rrel_non_shuf']].mean()

    fig = plt.figure(figsize=(9.6, 3.5))
    G = fig.add_gridspec(1, 3, width_ratios=[1, 1, 1.15], wspace=.44)

    def tidy(ax):
        ax.tick_params(labelsize=7)
        ax.spines[['top', 'right']].set_visible(False)

    def lp(ax, s, x=-.20):
        ax.text(x, 1.04, s, transform=ax.transAxes, fontsize=10, weight='bold',
                va='bottom', ha='right')

    def paired(ax, d, a, b, title, ylab):
        for _, r in d.iterrows():
            ax.plot([0, 1], [r[a], r[b]], color='0.78', lw=.7, zorder=1)
        ax.plot([0, 1], [d[a].mean(), d[b].mean()], 'o-', color=GRID_C, ms=6,
                lw=2.2, zorder=3)
        dd = (d[a] - d[b]).dropna()
        p = wilcoxon(dd).pvalue
        ax.set_xticks([0, 1])
        ax.set_xticklabels(['anchored', 'non-anch'], fontsize=7.5)
        ax.set_xlim(-.3, 1.3)
        ax.set_ylabel(ylab, fontsize=8)
        ax.set_title(f'{title}\n$\\Delta$ = {dd.mean():+.3f}, p = {p:.2g} '
                     f'({len(dd)} sessions)', fontsize=8.5)
        tidy(ax)
        return p

    # A -- absolute phase: the anchor is lost
    ax = fig.add_subplot(G[0])
    paired(ax, as_, 'Rrel_anch', 'Rrel_non',
           'ABSOLUTE phase, per cell', 'phase concentration $R$')
    lp(ax, 'A')

    # B -- relative phase: the sheet is not
    ax = fig.add_subplot(G[1])
    paired(ax, ws, 'Rrel_anch', 'Rrel_non',
           'RELATIVE phase, co-modular pairs', 'phase concentration $R$')
    lp(ax, 'B')

    # C -- and it stays above its floors in both states
    ax = fig.add_subplot(G[2])
    groups = [('within\nmodule', ws.Rrel_anch, ws.Rrel_non),
              ('cross\nmodule', xs.Rrel_anch, xs.Rrel_non),
              ('trial\nshuffled', sh.Rrel_anch_shuf, sh.Rrel_non_shuf)]
    w = .34
    for i, (nm, a, b) in enumerate(groups):
        ax.bar(i - w / 2, a.mean(), w, color=ANCH_COLOR, linewidth=0,
               label='anchored' if i == 0 else None)
        ax.bar(i + w / 2, b.mean(), w, color=NONANCH_COLOR, linewidth=0,
               label='non-anchored' if i == 0 else None)
        for x, v in ((i - w / 2, a), (i + w / 2, b)):
            se = v.std() / np.sqrt(v.notna().sum())
            ax.plot([x, x], [v.mean() - se, v.mean() + se], color='0.3', lw=1)
    ax.set_xticks(range(len(groups)))
    ax.set_xticklabels([g[0] for g in groups], fontsize=7.5)
    ax.set_ylabel('relative phase $R$', fontsize=8)
    ax.set_title('the sheet survives: co-modular pairs stay\nabove both floors in '
                 'BOTH states', fontsize=8.5)
    ax.legend(fontsize=7, frameon=False, loc='upper right')
    tidy(ax)
    lp(ax, 'C', x=-.17)

    fig.savefig(OUT_PDF, bbox_inches='tight')
    print(f'wrote {OUT_PDF}')


if os.path.exists(OUT_CSV):
    make_figure()
