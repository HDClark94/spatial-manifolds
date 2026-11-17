"""Can the anchoring state be read out from the pupil, trial by trial?

The paper shows the pupil is smaller when the MEC population is anchored. That
is a difference of means. A reviewer-style question is the harder one: given the
pupil alone, could the state -- and the moments it CHANGES -- be recovered?

Three things are measured, per session, and pooled over sessions:

1. DISCRIMINATION. Single-trial AUC for the neural state from the trial's mean
   pupil radius. Track luminance is NOT a confound for this: the measure is a
   trial mean, and every trial covers the same track, so the position-dependent
   component of pupil size is common to all trials and cancels. Luminance would
   only confound a within-trial, time-resolved readout.

2. STATE RECOVERY. The SAME two-state classifier the neural side uses -- median
   filter of 5, then 2-means -- applied to the trial series of pupil radius, and
   scored against the neural labels. Using the same estimator on both sides is
   what makes the comparison fair.

3. CHANGE DETECTION, which is what Matt asked about: the distance from each
   neural state switch to the nearest switch the pupil gives.

THE NULL MATTERS MORE THAN THE STATISTIC. Two-means applied to any smooth series
returns two states, and both the pupil and the neural state drift in long blocks,
so chance agreement is far above 0.5. The null here is a PHASE-RANDOMISED
surrogate of the pupil series, which preserves its autocorrelation exactly while
destroying its alignment to the neural state -- a circular shift preserves
autocorrelation too but samples only n distinct surrogates and leaves the
original's block structure intact at some shifts.

Writes data/eye_anchoring/pupil_decode_state.csv, one row per session.
"""
import os, sys, warnings
import numpy as np, pandas as pd
warnings.filterwarnings('ignore')
from scipy.ndimage import median_filter
from sklearn.cluster import KMeans
from sklearn.metrics import roc_auc_score, cohen_kappa_score

ROOT = '/Users/harryclark/Documents/spatial-manifolds'
TRIALS = f'{ROOT}/data/eye_anchoring/eye_anchoring_trials.csv'
OUT = f'{ROOT}/data/eye_anchoring/pupil_decode_state.csv'
MEDFILT = 5          # the neural classifier's median filter, matched here
N_NULL = 500
RNG = np.random.default_rng(7)


def phase_randomised(x, rng):
    """Surrogate with the same power spectrum -- so the same autocorrelation and
    block structure -- and no relation to anything else."""
    X = np.fft.rfft(x - x.mean())
    ph = rng.uniform(0, 2 * np.pi, len(X))
    ph[0] = 0
    if len(x) % 2 == 0:
        ph[-1] = 0
    y = np.fft.irfft(np.abs(X) * np.exp(1j * ph), n=len(x))
    return y / (y.std() + 1e-12) * x.std() + x.mean()


def two_state(x):
    """The neural classifier's trial-series step: median filter, then 2-means."""
    s = median_filter(x, size=MEDFILT, mode='nearest')
    if np.ptp(s) < 1e-9:
        return np.zeros(len(s), int)
    k = KMeans(n_clusters=2, n_init=10, random_state=0).fit(s.reshape(-1, 1))
    lab = k.labels_
    # label 1 = the LOW-pupil cluster, to match "anchored"
    return (lab == np.argmin(k.cluster_centers_.ravel())).astype(int)


def switches(lab):
    return np.where(np.diff(lab) != 0)[0] + 1


def switch_distance(a, b, n):
    """Median distance, in trials, from each switch in `a` to the nearest in `b`."""
    if len(a) == 0 or len(b) == 0:
        return np.nan
    return float(np.median([np.min(np.abs(b - i)) for i in a]))


def session(g):
    x = g.z_rad.values.astype(float)
    y = g.anch.values.astype(int)
    if np.isnan(x).any():
        ok = ~np.isnan(x)
        x = np.interp(np.arange(len(x)), np.where(ok)[0], x[ok])
    n = len(x)
    auc = roc_auc_score(y, -x)            # low pupil predicts anchored
    lab = two_state(x)
    acc = max((lab == y).mean(), (1 - lab == y).mean())
    kap = max(cohen_kappa_score(lab, y), cohen_kappa_score(1 - lab, y))
    sw_n, sw_p = switches(y), switches(lab)
    d = switch_distance(sw_n, sw_p, n)

    aucs, accs, kaps, ds = [], [], [], []
    for _ in range(N_NULL):
        xs = phase_randomised(x, RNG)
        aucs.append(roc_auc_score(y, -xs))
        l2 = two_state(xs)
        accs.append(max((l2 == y).mean(), (1 - l2 == y).mean()))
        kaps.append(max(cohen_kappa_score(l2, y), cohen_kappa_score(1 - l2, y)))
        ds.append(switch_distance(sw_n, switches(l2), n))
    aucs, accs, kaps = np.array(aucs), np.array(accs), np.array(kaps)
    ds = np.array(ds, float)

    def p_hi(obs, null):
        return float((np.sum(null >= obs) + 1) / (len(null) + 1))

    def p_lo(obs, null):
        null = null[np.isfinite(null)]
        if not len(null) or not np.isfinite(obs):
            return np.nan
        return float((np.sum(null <= obs) + 1) / (len(null) + 1))

    return dict(
        mouse=int(g.mouse.iloc[0]), day=int(g.day.iloc[0]), n_trials=n,
        frac_anch=float(y.mean()), n_sw_neural=len(sw_n), n_sw_pupil=len(sw_p),
        auc=auc, auc_null=float(aucs.mean()), auc_p=p_hi(auc, aucs),
        acc=acc, acc_null=float(accs.mean()), acc_p=p_hi(acc, accs),
        kappa=kap, kappa_null=float(kaps.mean()), kappa_p=p_hi(kap, kaps),
        sw_dist=d, sw_dist_null=float(np.nanmean(ds)), sw_dist_p=p_lo(d, ds))


if __name__ == '__main__':
    D = pd.read_csv(TRIALS)
    D['ok'] = (D.coverage >= 0.50) & D.z_rad.notna() & D.z_dev.notna()
    D['anch'] = D.frac_anch > 0.5
    cls = (D[D.ok].groupby(['mouse', 'day']).anch
             .agg(n_anch='sum', n_tot='size').reset_index())
    cls['minority'] = np.minimum(cls.n_anch, cls.n_tot - cls.n_anch)
    s = (D.groupby(['mouse', 'day'])
          .agg(blink=('blink_frac', 'first'), rad_sd=('rad_raw', 'std'),
               rad_med=('rad_raw', 'median'), n_ok=('ok', 'sum'))
          .reset_index().merge(cls, on=['mouse', 'day']))
    s['cv'] = s.rad_sd / s.rad_med
    s['inc'] = ((s.blink <= .20) & (s.n_ok >= 40) & (s.cv >= .015) &
                (s.minority >= 30))
    inc = set(zip(s[s.inc].mouse, s[s.inc].day))
    E = D[D.ok & [(m, d) in inc for m, d in zip(D.mouse, D.day)]]
    print(f'{E.groupby(["mouse","day"]).ngroups} sessions, {len(E)} trials')
    rows = [session(g) for _, g in E.groupby(['mouse', 'day'])]
    R = pd.DataFrame(rows)
    R.to_csv(OUT, index=False)
    print(R.to_string(index=False))
    print(f'\nwrote {OUT}')
