"""Speed tuning by anchoring state: does the velocity-gain account survive?

Reads speed_tuning_cells.csv / speed_tuning_decode.csv (written by
speed_tuning.py) and asks three things:

  1. Do speed-tuned cells have flatter speed tuning when anchored?
  2. Is any such change a change in SPEED SENSITIVITY or just in firing rate?
     slope falls with rate even if tuning shape is untouched, so gain
     (slope / mean rate) is the measure that separates them.
  3. Is speed less decodable from the population when anchored, with bin count
     and speed distribution matched between states?

Sessions are the unit of analysis throughout -- cells within a session share a
behavioural state and are not independent, so a cell-level test would count the
same state change hundreds of times. Cell-level numbers are printed alongside
for reference only.
"""
import sys
import numpy as np, pandas as pd
from scipy.stats import wilcoxon

OUT = '/Users/harryclark/Documents/spatial-manifolds/data/lfp'
C = pd.read_csv(f'{OUT}/speed_tuning_cells.csv')
D = pd.read_csv(f'{OUT}/speed_tuning_decode.csv')

C['speed_cell'] = (C.p_pool < .05) & (C.r_pool > 0)
C['neg_cell'] = (C.p_pool < .05) & (C.r_pool < 0)
print(f'{len(C)} cells, {C.groupby(["mouse","day"]).ngroups} sessions')
print(f'  positively speed-modulated {C.speed_cell.mean()*100:.1f}%, '
      f'negatively {C.neg_cell.mean()*100:.1f}% '
      f'(shift null, p < 0.05)\n')


def paired(df, a, n, lab, by_session=True):
    d = df.dropna(subset=[a, n])
    if by_session:
        d = d.groupby(['mouse', 'day'])[[a, n]].median().dropna()
    if len(d) < 5:
        print(f'  {lab}: too few'); return
    w = wilcoxon(d[a], d[n])
    print(f'  {lab:34s} anchored {d[a].median():+.4f}  non {d[n].median():+.4f}  '
          f'({100*(d[a].median()-d[n].median())/abs(d[n].median()):+6.1f}%)  '
          f'p = {w.pvalue:.3g}  n = {len(d)}')


for tag, name in (('', 'all running bins'), ('cs_', 'common speed support')):
    print(f'--- speed cells only, {name} (session medians)')
    S = C[C.speed_cell]
    paired(S, f'{tag}slope_a', f'{tag}slope_n', 'slope (Hz per cm/s)')
    paired(S, f'{tag}gain_a', f'{tag}gain_n', 'gain (slope / mean rate)')
    paired(S, f'{tag}r_a', f'{tag}r_n', 'r (rate, speed)')
    paired(S, f'{tag}rate_a', f'{tag}rate_n', 'mean rate (Hz)')
    print()

print('--- all cells, all running bins (session medians)')
paired(C, 'slope_a', 'slope_n', 'slope (Hz per cm/s)')
paired(C, 'gain_a', 'gain_n', 'gain (slope / mean rate)')
paired(C, 'rate_a', 'rate_n', 'mean rate (Hz)')
print()

print('--- cell level, for reference only (not independent)')
S = C[C.speed_cell].dropna(subset=['slope_a', 'slope_n'])
print(f'  slope   anchored {S.slope_a.median():+.4f} vs non {S.slope_n.median():+.4f}, '
      f'p = {wilcoxon(S.slope_a, S.slope_n).pvalue:.3g}, '
      f'{(S.slope_a < S.slope_n).mean()*100:.0f}% of {len(S)} cells flatter\n')

print('--- population speed decoding, bins matched for count and speed')
d = D.dropna(subset=['dec_r_a', 'dec_r_n'])
print(f'  r(pred, true)   anchored {d.dec_r_a.median():.3f}  non {d.dec_r_n.median():.3f}  '
      f'p = {wilcoxon(d.dec_r_a, d.dec_r_n).pvalue:.3g}  n = {len(d)}')
print(f'  abs error cm/s  anchored {d.dec_mae_a.median():.2f}  non {d.dec_mae_n.median():.2f}  '
      f'p = {wilcoxon(d.dec_mae_a, d.dec_mae_n).pvalue:.3g}')
print(f'  ({(d.dec_r_a > d.dec_r_n).sum()}/{len(d)} sessions decode BETTER when anchored)')
print(f'  matched speed SD  anchored {d.speed_sd_a.median():.2f}  non {d.speed_sd_n.median():.2f}')
