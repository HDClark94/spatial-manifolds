"""Controls for the speed-theta gain result."""
import warnings; warnings.filterwarnings('ignore')
import numpy as np, pandas as pd
from scipy.stats import pearsonr, wilcoxon, ttest_1samp
rng = np.random.default_rng(0)
T = pd.read_csv('/Users/harryclark/Documents/spatial-manifolds/data/lfp/theta_frequency.csv')
T['anch'] = T.anch.astype(bool)
MINTR = 15

def slope(x, y):
    return np.polyfit(x, y, 1)[0]

# ---- 1. speed range by state (does r attenuation explain it?)
rows = []
for (mo, dy), d in T.groupby(['mouse', 'day']):
    a, n = d[d.anch], d[~d.anch]
    if len(a) < MINTR or len(n) < MINTR: continue
    rows.append(dict(sd_a=a.speed.std(), sd_n=n.speed.std(),
                     r_a=pearsonr(a.speed, a.amp)[0], r_n=pearsonr(n.speed, n.amp)[0],
                     sl_a=slope(a.speed, a.amp), sl_n=slope(n.speed, n.amp),
                     ra_hz=pearsonr(a.speed, a.inst_hz)[0], rn_hz=pearsonr(n.speed, n.inst_hz)[0],
                     sla_hz=slope(a.speed, a.inst_hz), sln_hz=slope(n.speed, n.inst_hz)))
R = pd.DataFrame(rows)
print(f'speed SD   anchored {R.sd_a.mean():.2f}  non {R.sd_n.mean():.2f}  '
      f'p = {wilcoxon(R.sd_a, R.sd_n).pvalue:.3g}  ({(R.sd_a > R.sd_n).sum()}/{len(R)} wider when anchored)')
print(f'raw slope  amp  anchored {R.sl_a.mean():+.4f}  non {R.sl_n.mean():+.4f}  '
      f'p = {wilcoxon(R.sl_a, R.sl_n).pvalue:.3g}')
print(f'raw slope  freq anchored {R.sla_hz.mean():+.5f}  non {R.sln_hz.mean():+.5f}  '
      f'p = {wilcoxon(R.sla_hz, R.sln_hz).pvalue:.3g}')
# correlation attenuation predicts dr proportional to d(sd); test residual
import scipy.stats as ss
dr, dsd = (R.r_a - R.r_n).values, (R.sd_a - R.sd_n).values
print(f'  dr vs dSD across sessions: rho = {ss.spearmanr(dr, dsd)[0]:+.3f}, '
      f'p = {ss.spearmanr(dr, dsd)[1]:.3g}')

# ---- 2. circular-shift null on the state vector (preserves run structure + drift)
def gain(d, mask):
    a, n = d[mask], d[~mask]
    if len(a) < MINTR or len(n) < MINTR: return np.nan
    return slope(a.speed, a.amp) - slope(n.speed, n.amp)

obs, nulls = [], []
for (mo, dy), d in T.groupby(['mouse', 'day']):
    d = d.sort_values('trial')
    m = d.anch.values
    if m.sum() < MINTR or (~m).sum() < MINTR: continue
    obs.append(gain(d, m))
    nulls.append([gain(d, np.roll(m, k)) for k in rng.integers(1, len(m), 200)])
obs = np.array(obs); nulls = np.array(nulls)
nm = np.nanmean(nulls, axis=0)
print(f'\ncircular-shift null (200x, state vector rolled within session):')
print(f'  observed mean gain difference {np.nanmean(obs):+.4f}')
print(f'  null mean {np.nanmean(nm):+.4f}  sd {np.nanstd(nm):.4f}  '
      f'p = {(np.sum(nm <= np.nanmean(obs)) + 1) / (len(nm) + 1):.4g}')

# ---- 3. does the gain survive equalising speed range? (quantile-trim to common support)
keep = []
for (mo, dy), d in T.groupby(['mouse', 'day']):
    a, n = d[d.anch], d[~d.anch]
    if len(a) < MINTR or len(n) < MINTR: continue
    lo = max(a.speed.quantile(.05), n.speed.quantile(.05))
    hi = min(a.speed.quantile(.95), n.speed.quantile(.95))
    a2, n2 = a[a.speed.between(lo, hi)], n[n.speed.between(lo, hi)]
    if len(a2) < MINTR or len(n2) < MINTR: continue
    keep.append((slope(a2.speed, a2.amp), slope(n2.speed, n2.amp),
                 pearsonr(a2.speed, a2.amp)[0], pearsonr(n2.speed, n2.amp)[0]))
K = np.array(keep)
print(f'\ncommon speed support ({len(K)} sessions, 5-95% overlap):')
print(f'  slope anchored {K[:,0].mean():+.4f}  non {K[:,1].mean():+.4f}  '
      f'p = {wilcoxon(K[:,0], K[:,1]).pvalue:.3g}')
print(f'  r     anchored {np.tanh(np.arctanh(K[:,2]).mean()):+.3f}  '
      f'non {np.tanh(np.arctanh(K[:,3]).mean()):+.3f}  '
      f'p = {wilcoxon(K[:,2], K[:,3]).pvalue:.3g}')
