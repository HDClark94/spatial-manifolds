"""Figure 5 statistics with the hierarchy modelled, instead of ignored.

Usage:  python3 fig6_lmm.py

WHAT WAS WRONG. Every number in Figure 5 came from a Wilcoxon signed-rank test
over sessions, which treats 19 sessions as 19 independent observations. They are
not: they come from a handful of mice, several sessions each, so sessions from
one animal are correlated and the effective n is smaller than 19. That inflates
significance, and it does so more the more unbalanced the per-mouse session
counts are.

WHAT THIS DOES INSTEAD. Linear mixed models with random intercepts for mouse and
for session nested within mouse:

    value ~ anchored + (1 | mouse) + (1 | mouse:session)

fitted on TRIAL-level data, so the model sees the real hierarchy -- trials
within sessions within mice -- rather than a pre-averaged summary. Random
intercepts for session absorb the pairing that the signed-rank test was using,
and the mouse term absorbs what it was ignoring.

THE SPEED CONFOUND, HANDLED PROPERLY. The states differ in running speed, and
the previous answer to that was to subsample trials into a speed-matched subset
-- which discarded 79% of the data and still left a residual speed difference.
A mixed model can take speed as a covariate instead:

    value ~ anchored + speed + (1 | mouse) + (1 | mouse:session)

This asks what is left of the state effect once speed is accounted for, on all
10,802 trials rather than 2,294. Both versions are reported, because they answer
different questions: without the covariate, "do anchored trials have lower theta
amplitude?"; with it, "lower than trials at the same running speed?".

THE GAIN PANELS ARE DIFFERENT. Those compare a within-state CORRELATION, which
is one number per session per state, so there is no trial level to model. They
get a mixed model on the session-level values with a random intercept for mouse,
which is the part the signed-rank test was missing.

Amplitude uses the session z-scored column: absolute LFP amplitude spans 50-180
across sessions with electrode impedance and reference, so a raw-amplitude model
would be dominated by between-session scale.
"""
import sys
import warnings

import numpy as np
import pandas as pd

warnings.filterwarnings('ignore')
import statsmodels.formula.api as smf
from scipy.stats import pearsonr, wilcoxon

LFP = '/Users/harryclark/Documents/spatial-manifolds/data/lfp'
MINTR = 15

T = pd.read_csv(f'{LFP}/theta_frequency.csv')
T['anch'] = T.anch.astype(bool)
keep = []
for (mo, dy), d in T.groupby(['mouse', 'day']):
    if (d.anch.sum() >= MINTR) and ((~d.anch).sum() >= MINTR):
        keep.append((mo, dy))
T = T[T.set_index(['mouse', 'day']).index.isin(keep)].copy()
T['session'] = T.mouse.astype(str) + 'D' + T.day.astype(str)
T['A'] = T.anch.astype(int)

n_mice = T.mouse.nunique()
per_mouse = T.groupby('mouse').session.nunique()
print(f'{len(T)} trials, {T.session.nunique()} sessions, {n_mice} mice')
print(f'sessions per mouse: {per_mouse.to_dict()}')
print('  -> the signed-rank test treated these as '
      f'{T.session.nunique()} independent units\n')


def lmm(df, dv, covar=None, warn=True):
    """value ~ anchored [+ speed] + (1|mouse) + (1|mouse:session).

    Returns the fixed effect plus the two variance components, because a
    variance estimated at the boundary (0) is the usual cause of the
    convergence warnings these models throw, and it changes how the SE should
    be read -- a mouse variance of 0 means the model has collapsed to one that
    ignores mouse, which is the very thing being corrected for.
    """
    f = f'{dv} ~ A' + (f' + {covar}' if covar else '')
    with warnings.catch_warnings(record=True) as w:
        warnings.simplefilter('always')
        r = smf.mixedlm(f, df, groups=df['mouse'],
                        vc_formula={'session': '0 + C(session)'}
                        ).fit(reml=True, method='lbfgs')
        conv = 'OK ' if not any('onvergence' in str(x.message) or
                                'oundary' in str(x.message) or
                                'Hessian' in str(x.message) for x in w) else 'WARN'
    cr = np.asarray(r.cov_re)
    v_mouse = float(cr.ravel()[0]) if cr.size else np.nan
    vc = np.asarray(getattr(r, 'vcomp', []))
    v_sess = float(vc.ravel()[0]) if vc.size else np.nan
    return r.params['A'], r.bse['A'], r.pvalues['A'], v_mouse, v_sess, conv


def lmm_session(S, col_a, col_n, dv):
    """Session-level: the signed-rank test's own unit, plus a mouse term.

    This is the minimal correction to what Figure 5 already does. The trial-level
    model has far more power but assumes trials are conditionally independent
    given session, which consecutive trials are not -- running speed and theta
    drift slowly across a session -- so its p-values are anti-conservative. Where
    the two disagree, this one is the conservative reading.
    """
    # Exactly two rows per session, so a session random intercept is confounded
    # with the residual and the fit is not identified (statsmodels fails
    # outright). The paired design collapses to the per-session DIFFERENCE,
    # which is the same test, and the mouse term then does the work that was
    # missing:  diff ~ 1 + (1 | mouse).
    D = pd.DataFrame(dict(mouse=S.mouse, d=S[col_a].values - S[col_n].values))
    try:
        r = smf.mixedlm('d ~ 1', D, groups=D['mouse']).fit(reml=True,
                                                           method='lbfgs')
        p_lmm = float(r.pvalues['Intercept'])
        b_lmm = float(r.params['Intercept'])
    except Exception:
        p_lmm, b_lmm = np.nan, float(D.d.mean())
    # The most conservative reading available: collapse to one number per
    # ANIMAL, then test across animals. n = 6, so the smallest attainable
    # Wilcoxon p is 0.031 -- this cannot show a small effect, but anything it
    # does show is not pseudoreplication.
    per = D.groupby('mouse').d.mean()
    p_mouse = wilcoxon(per.values).pvalue if len(per) >= 6 else np.nan
    return b_lmm, p_lmm, p_mouse, float(per.mean())


print(f'{"measure":30s} {"beta":>9s} {"LMM trial":>11s} {"Wilcoxon":>11s} '
      f'{"LMM sess":>11s} {"per-mouse":>11s}')
print(f'{"":30s} {"":>9s} {"(n=4205)":>11s} {"(n=19)":>11s} {"(n=19,+mouse)":>11s} '
      f'{"(n=6)":>11s}')
print('-' * 108)

# session-level values, for the signed-rank comparison
S = []
for (mo, dy), d in T.groupby(['mouse', 'day']):
    a, n = d[d.anch], d[~d.anch]
    S.append(dict(mouse=mo, day=dy,
                  z_amp_a=a.z_amp.mean(), z_amp_n=n.z_amp.mean(),
                  hz_a=a.inst_hz.mean(), hz_n=n.inst_hz.mean(),
                  spd_a=a.speed.mean(), spd_n=n.speed.mean(),
                  r_amp_a=pearsonr(a.speed, a.amp)[0],
                  r_amp_n=pearsonr(n.speed, n.amp)[0],
                  r_hz_a=pearsonr(a.speed, a.inst_hz)[0],
                  r_hz_n=pearsonr(n.speed, n.inst_hz)[0]))
S = pd.DataFrame(S)

for dv, nm, wa, wn in (('z_amp', 'theta amplitude (z)', 'z_amp_a', 'z_amp_n'),
                       ('inst_hz', 'theta frequency (Hz)', 'hz_a', 'hz_n'),
                       ('speed', 'running speed (cm/s)', 'spd_a', 'spd_n')):
    b, se, p, vm, vs, conv = lmm(T, dv)
    pw = wilcoxon(S[wa], S[wn]).pvalue
    bs, ps, pm, bm = lmm_session(S, wa, wn, dv)
    print(f'{nm:30s} {b:+9.4f} {p:11.3g} {pw:11.3g} {ps:11.3g} {pm:11.3g}   '
          f'var(mouse)={vm:.3g} {conv}')

print('\nwith running speed as a covariate (all trials, not a matched subset):')
for dv, nm in (('z_amp', 'theta amplitude (z) | speed'),
               ('inst_hz', 'theta frequency (Hz) | speed')):
    b, se, p, vm, vs, conv = lmm(T, dv, covar='speed')
    print(f'{nm:34s} {b:+18.4f} {se:8.4f} {p:10.3g}   '
          f'var(mouse)={vm:.4g} var(sess)={vs:.4g} {conv}')

# ── the gain panels: one correlation per session per state ───────────────────
print('\ngain panels — session-level r, random intercept for mouse:')
G = []
for _, r in S.iterrows():
    for st, tag in ((1, 'a'), (0, 'n')):
        G.append(dict(mouse=r.mouse, day=r.day, A=st,
                      r_amp=r[f'r_amp_{tag}'], r_hz=r[f'r_hz_{tag}']))
G = pd.DataFrame(G)
for dv, nm, wa, wn in (('r_amp', 'r(speed, amplitude)', 'r_amp_a', 'r_amp_n'),
                       ('r_hz', 'r(speed, frequency)', 'r_hz_a', 'r_hz_n')):
    m = smf.mixedlm(f'{dv} ~ A', G, groups=G['mouse']).fit(reml=True,
                                                           method='lbfgs')
    pw = wilcoxon(S[wa], S[wn]).pvalue
    _, _, pm, _ = lmm_session(S, wa, wn, dv)
    print(f'{nm:30s} {m.params["A"]:+9.4f} {"":>11s} {pw:11.3g} '
          f'{m.pvalues["A"]:11.3g} {pm:11.3g}')

print('\n(Wilcoxon column is the previous, non-hierarchical result)')


# ── the coupling correlations, done properly ─────────────────────────────────
# r is bounded on [-1, 1] and its sampling distribution is skewed, so the
# models are fitted on Fisher z = arctanh(r), which is approximately normal
# with variance 1/(n-3). Differences are reported back-transformed as well.
#
# There is no trial level here -- each session contributes ONE correlation per
# state -- so "mouse and day" means: day is the pairing unit (exactly two
# observations, anchored and non-anchored), which collapses to the per-day
# difference, and mouse is the random intercept over those differences.
print('\n' + '=' * 78)
print('COUPLING CORRELATIONS — r(speed, theta), Fisher-z, mouse + day levels')
print('=' * 78)
n_tr_a = T[T.anch].groupby(['mouse', 'day']).size()
n_tr_n = T[~T.anch].groupby(['mouse', 'day']).size()
for col, nm in (('r_amp', 'r(speed, amplitude)'), ('r_hz', 'r(speed, frequency)')):
    ra, rn = S[f'{col}_a'].values, S[f'{col}_n'].values
    za, zn = np.arctanh(np.clip(ra, -.999, .999)), np.arctanh(np.clip(rn, -.999, .999))
    D = pd.DataFrame(dict(mouse=S.mouse.values, d=za - zn))
    m = smf.mixedlm('d ~ 1', D, groups=D['mouse']).fit(reml=True, method='lbfgs')
    b, se, p = (float(m.params['Intercept']), float(m.bse['Intercept']),
                float(m.pvalues['Intercept']))
    per = D.groupby('mouse').d.mean()
    p_mouse = wilcoxon(per.values).pvalue
    p_wil_z = wilcoxon(za, zn).pvalue
    print(f'\n{nm}')
    print(f'  mean r      anchored {ra.mean():+.3f}   non-anchored {rn.mean():+.3f}'
          f'   ({(ra < rn).sum()}/{len(ra)} sessions lower when anchored)')
    print(f'  Fisher z    anchored {za.mean():+.3f}   non-anchored {zn.mean():+.3f}')
    print(f'  LMM  d_z ~ 1 + (1|mouse):  beta = {b:+.4f}  SE = {se:.4f}  '
          f'p = {p:.4g}')
    print(f'      back-transformed difference in r = {np.tanh(b):+.3f}')
    print(f'  Wilcoxon over sessions (no mouse term), on z:   p = {p_wil_z:.4g}')
    print(f'  per-mouse, averaged within animal first (n={len(per)}): p = {p_mouse:.4g}')
    print('      per-mouse mean Δz: ' +
          '  '.join(f'M{int(k)} {v:+.2f}' for k, v in per.items()))


# ── the same correlations on SPEED-MATCHED trials ────────────────────────────
# This control matters more for the coupling than for the means. A correlation
# is attenuated by restricting the range of its predictor, so if the two states
# sample different ranges of running speed, the state with the wider range gets
# a larger r for free -- and the gain result would be an artefact of that.
# Decile-matching equalises the speed distributions, which removes the artefact
# but also shrinks both correlations, so the matched r's are expected to be
# smaller in BOTH states; what matters is whether the DIFFERENCE survives.
MIN_MATCH = 15
print('\n' + '=' * 78)
print('COUPLING CORRELATIONS ON SPEED-MATCHED TRIALS')
print('=' * 78)
rows = []
for (mo, dy), d in T.groupby(['mouse', 'day']):
    m = d[d.speed_matched.astype(bool)]
    ma, mn = m[m.anch], m[~m.anch]
    if len(ma) < MIN_MATCH or len(mn) < MIN_MATCH:
        continue
    a, n = d[d.anch], d[~d.anch]
    rows.append(dict(
        mouse=mo, day=dy, n_a=len(ma), n_n=len(mn),
        r_amp_a=pearsonr(ma.speed, ma.amp)[0], r_amp_n=pearsonr(mn.speed, mn.amp)[0],
        r_hz_a=pearsonr(ma.speed, ma.inst_hz)[0],
        r_hz_n=pearsonr(mn.speed, mn.inst_hz)[0],
        sd_a=ma.speed.std(), sd_n=mn.speed.std(),
        sd_a_all=a.speed.std(), sd_n_all=n.speed.std()))
M = pd.DataFrame(rows)
print(f'{len(M)} of {len(S)} sessions keep >= {MIN_MATCH} matched trials per state '
      f'(median {M[["n_a","n_n"]].min(axis=1).median():.0f} per state)')
print(f'speed SD within state — unmatched: anchored {M.sd_a_all.mean():.2f}, '
      f'non-anchored {M.sd_n_all.mean():.2f} cm/s')
print(f'                          matched: anchored {M.sd_a.mean():.2f}, '
      f'non-anchored {M.sd_n.mean():.2f} cm/s')
print(f'  range-restriction check: unmatched SD ratio (non/anch) '
      f'{(M.sd_n_all / M.sd_a_all).mean():.3f}, matched {(M.sd_n / M.sd_a).mean():.3f}')

for col, nm in (('r_amp', 'r(speed, amplitude)'), ('r_hz', 'r(speed, frequency)')):
    ra, rn = M[f'{col}_a'].values, M[f'{col}_n'].values
    za = np.arctanh(np.clip(ra, -.999, .999))
    zn = np.arctanh(np.clip(rn, -.999, .999))
    D = pd.DataFrame(dict(mouse=M.mouse.values, d=za - zn))
    mm = smf.mixedlm('d ~ 1', D, groups=D['mouse']).fit(reml=True, method='lbfgs')
    per = D.groupby('mouse').d.mean()
    p_mouse = wilcoxon(per.values).pvalue if len(per) >= 6 else np.nan
    print(f'\n{nm}  [speed-matched]')
    print(f'  mean r      anchored {ra.mean():+.3f}   non-anchored {rn.mean():+.3f}'
          f'   ({(ra < rn).sum()}/{len(ra)} sessions lower when anchored)')
    print(f'  LMM  d_z ~ 1 + (1|mouse):  beta = {float(mm.params["Intercept"]):+.4f}'
          f'  SE = {float(mm.bse["Intercept"]):.4f}  '
          f'p = {float(mm.pvalues["Intercept"]):.4g}')
    print(f'      back-transformed difference in r = '
          f'{np.tanh(float(mm.params["Intercept"])):+.3f}')
    print(f'  Wilcoxon over sessions, on z:  p = {wilcoxon(za, zn).pvalue:.4g}')
    print(f'  per-mouse (n={len(per)}): p = {p_mouse:.4g}   ' +
          '  '.join(f'M{int(k)} {v:+.2f}' for k, v in per.items()))


# ── SLOPE, not correlation: the measure the panels are actually claiming ─────
# r conflates two things -- how steeply theta rises with speed, and how much
# speed varies. Restricting the range of speed attenuates r with the slope
# unchanged, which is why the r comparison needed speed matching at all.
#
# The panels are labelled "gain", and gain is the SLOPE: d(theta)/d(speed), in
# theta units per cm/s. A regression coefficient is invariant to the range of
# the predictor, so it needs no matching, uses every trial, and answers the
# mechanistic question directly. If the slope is unchanged while r falls, the
# transfer function is intact and only the input range shrank; if the slope
# falls, theta genuinely tracks speed less steeply.
print('\n' + '=' * 78)
print('GAIN AS SLOPE — no speed matching required')
print('=' * 78)
rows = []
for (mo, dy), d in T.groupby(['mouse', 'day']):
    a, n = d[d.anch], d[~d.anch]
    r = dict(mouse=mo, day=dy)
    for tag, g in (('a', a), ('n', n)):
        r[f'sl_amp_{tag}'] = np.polyfit(g.speed, g.z_amp, 1)[0]
        r[f'sl_hz_{tag}'] = np.polyfit(g.speed, g.inst_hz, 1)[0]
    rows.append(r)
SL = pd.DataFrame(rows)
for col, nm, unit in (('sl_amp', 'slope: amplitude on speed', 'z per cm/s'),
                      ('sl_hz', 'slope: frequency on speed', 'Hz per cm/s')):
    va, vn = SL[f'{col}_a'].values, SL[f'{col}_n'].values
    D = pd.DataFrame(dict(mouse=SL.mouse.values, d=va - vn))
    mm = smf.mixedlm('d ~ 1', D, groups=D['mouse']).fit(reml=True, method='lbfgs')
    per = D.groupby('mouse').d.mean()
    print(f'\n{nm}  ({unit})')
    print(f'  anchored {va.mean():+.5f}   non-anchored {vn.mean():+.5f}   '
          f'({(va < vn).sum()}/{len(va)} lower when anchored)')
    print(f'  relative change {100 * (va.mean() - vn.mean()) / abs(vn.mean()):+.1f}%')
    print(f'  LMM  d ~ 1 + (1|mouse):  beta = {float(mm.params["Intercept"]):+.5f}  '
          f'SE = {float(mm.bse["Intercept"]):.5f}  '
          f'p = {float(mm.pvalues["Intercept"]):.4g}')
    print(f'  Wilcoxon over sessions:  p = {wilcoxon(va, vn).pvalue:.4g}')
    print(f'  per-mouse (n={len(per)}): p = {wilcoxon(per.values).pvalue:.4g}   ' +
          '  '.join(f'M{int(k)} {v:+.4f}' for k, v in per.items()))
