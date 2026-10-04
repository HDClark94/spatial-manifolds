"""Which mechanism? Three checks that separate the candidate accounts."""
import warnings; warnings.filterwarnings('ignore')
import numpy as np, pandas as pd
from scipy.stats import wilcoxon, pearsonr
T = pd.read_csv('/Users/harryclark/Documents/spatial-manifolds/data/lfp/theta_frequency.csv')
T['anch'] = T.anch.astype(bool)
MINTR = 15
ok = lambda d: len(d[d.anch]) >= MINTR and len(d[~d.anch]) >= MINTR

# --- 1. Is it just a multiplicative scaling of the whole LFP?
# If anchoring simply scales theta by k<1, slope and mean must fall by the SAME
# factor. Refit after dividing amplitude by its own within-state mean.
rows = []
for (mo, dy), d in T.groupby(['mouse', 'day']):
    if not ok(d): continue
    a, n = d[d.anch], d[~d.anch]
    sa, sn = np.polyfit(a.speed, a.amp, 1)[0], np.polyfit(n.speed, n.amp, 1)[0]
    na, nn = (np.polyfit(a.speed, a.amp / a.amp.mean(), 1)[0],
              np.polyfit(n.speed, n.amp / n.amp.mean(), 1)[0])
    rows.append(dict(mean_a=a.amp.mean(), mean_n=n.amp.mean(), sl_a=sa, sl_n=sn,
                     norm_a=na, norm_n=nn,
                     int_a=np.polyfit(a.speed, a.amp, 1)[1],
                     int_n=np.polyfit(n.speed, n.amp, 1)[1],
                     hz_int_a=np.polyfit(a.speed, a.inst_hz, 1)[1],
                     hz_int_n=np.polyfit(n.speed, n.inst_hz, 1)[1]))
R = pd.DataFrame(rows)
print(f'--- 1. multiplicative scaling? ({len(R)} sessions)')
print(f'  mean amplitude    anchored/non = {(R.mean_a / R.mean_n).median():.3f}')
print(f'  raw slope         anchored/non = {(R.sl_a / R.sl_n).median():.3f}')
print(f'  slope after within-state normalisation: {R.norm_a.median():+.5f} vs '
      f'{R.norm_n.median():+.5f}, p = {wilcoxon(R.norm_a, R.norm_n).pvalue:.3g}')
print('  -> a pure gain change predicts the first two ratios to match.\n')

print('--- 2. intercept: is low-speed theta preserved?')
print(f'  amplitude intercept  anchored {R.int_a.median():.2f} vs non {R.int_n.median():.2f}, '
      f'p = {wilcoxon(R.int_a, R.int_n).pvalue:.3g}')
print(f'  frequency intercept  anchored {R.hz_int_a.median():.3f} vs non {R.hz_int_n.median():.3f} Hz, '
      f'p = {wilcoxon(R.hz_int_a, R.hz_int_n).pvalue:.3g}')
cross = (R.int_a - R.int_n) / (R.sl_n - R.sl_a)
print(f'  implied crossover speed: median {cross.median():.1f} cm/s '
      f'(session speed range {T.speed.quantile(.05):.0f}-{T.speed.quantile(.95):.0f})\n')

# --- 3. Is the gain change specific to theta, or does broadband lose speed coupling too?
try:
    B = pd.read_csv('/Users/harryclark/Documents/spatial-manifolds/data/lfp/'
                    'lfp_power_by_anchoring.csv')
    print('--- 3. band power file columns:', list(B.columns))
    if 'speed' in B.columns:
        B['anch'] = B.anch.astype(bool)
        for band in sorted(B.band.unique()):
            out = []
            for (mo, dy), d in B[B.band == band].groupby(['mouse', 'day']):
                d = d.groupby(['trial', 'anch'], as_index=False).agg(
                    p=('z_abs' if 'z_abs' in B.columns else 'abs_power', 'mean'),
                    speed=('speed', 'first'))
                if not ok(d): continue
                a, n = d[d.anch], d[~d.anch]
                out.append((pearsonr(a.speed, a.p)[0], pearsonr(n.speed, n.p)[0]))
            o = np.array(out)
            print(f'  {band:12s} r(speed,power)  anchored {o[:,0].mean():+.3f}  '
                  f'non {o[:,1].mean():+.3f}  p = {wilcoxon(o[:,0], o[:,1]).pvalue:.2g}')
except FileNotFoundError:
    print('--- 3. band power file not found')
