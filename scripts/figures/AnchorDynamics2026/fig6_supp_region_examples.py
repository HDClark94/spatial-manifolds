"""Figure 6 supplement 2: the state outside MEC, in sessions MEC does not dominate.

The main region-example figure shows sessions where entorhinal cortex supplies
most of the cells, which is the usual case and is also the weakest place to
argue from: a state that MEC expresses strongly will look shared with any
structure sampled alongside it, and a reader is entitled to suspect that what is
being shown is MEC leaking into its neighbours.

This figure deliberately picks the other sessions. Each row is laid out exactly
as in the main figure -- the same `region_row.draw_row` -- so the comparison is
like for like.

Every row here has exactly TWO structures, which is deliberate: a row with three
divides the same page width between more panels, and the comparison that matters
is between a pair of components, not among three. Sessions with a third sampled
structure are in the main examples.

  M21 D20           the strongest entorhinal-visual agreement in the dataset,
                    +0.91 on 58 visual cells.
  M28 D20           a middling one, +0.65, visual cells ~720 um from MEC.
  M21 D25           VIS-DOMINANT -- visual cortex contributes MORE label-varying
                    cells than MEC (66 against 64) -- and ANTI-CORRELATED, -0.50.
                    M21 is the animal whose probe traversed visual cortex
                    properly rather than clipping its ventral edge, so its
                    visual cells sit 0.7-1.1 mm from the nearest entorhinal cell.
  M25 D21           SUB-DOMINANT, subicular complex 35 against MEC 31, +0.23.
  M26 D13           WITH CEREBELLUM, 18 cells, +0.58. The cerebellum was intended
                    as a negative control and is not one; showing it is the
                    honest way to say so.

Taken in order the visual rows run +0.91, +0.65, -0.50: the range is the point,
and a figure showing only the top of it would misrepresent Figure 6C.

WHAT IS AND IS NOT BEING CLAIMED. These are examples, chosen for their sampling
rather than their result, and the printed correlations are single sessions. The
cerebellar rows in particular rest on 15-18 cells, where an axis is barely
estimable -- they are shown so the claim "every structure recorded has a state"
can be inspected at its weakest point, not because two sessions establish it.

Writes fig6_supp_region_examples.pdf
"""
import os
import sys
import warnings

import numpy as np

warnings.filterwarnings('ignore')
import matplotlib

matplotlib.use('Agg')
import matplotlib.pyplot as plt

sys.path.insert(0, '/Users/harryclark/Documents/spatial-manifolds/src')
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import region_row as RR

plt.rcParams['font.family'] = 'Arial'
plt.rcParams['pdf.fonttype'] = 42

FIG = os.path.dirname(os.path.abspath(__file__))
OUT = f'{FIG}/fig6_supp_region_examples.pdf'
MIN_CELLS = 8          # lower than the main figure: cerebellum never reaches 20

# (mouse, day, what makes this session worth showing)
SESSIONS = [
    (21, 20, 'the strongest entorhinal-visual agreement in the dataset'),
    (28, 20, 'visual-rich; its visual cells sit ~720 um from MEC'),
    (21, 25, 'visual cortex contributes more cells than MEC — and disagrees'),
    (25, 21, 'subicular complex contributes more cells than MEC'),
    (26, 13, 'the probe reaches cerebellum'),
]

rows = []
for mo, dy, why in SESSIONS:
    d = RR.collect(mo, dy, min_cells=MIN_CELLS)
    if len(d) < 2:
        print(f'  ! M{mo}D{dy}: only {list(d)} reach {MIN_CELLS} cells')
        continue
    regs = RR.top_regions(d, n=3)
    if len(regs) != 2:
        print(f'  ! M{mo}D{dy}: {len(regs)} structures ({regs}), not 2 — skipped')
        continue
    rows.append((mo, dy, why, d, regs))
    print(f'M{mo} D{dy}: ' + ', '.join(f'{r} {len(d[r][0])}' for r in regs)
          + f'   [{why}]')

fig = plt.figure(figsize=(10, 2.05 * len(rows)))
OUTER = fig.add_gridspec(len(rows), 1, hspace=.88)

for i, (mo, dy, why, d, regs) in enumerate(rows):
    # the pairwise agreements go INTO the row headline rather than onto the
    # rightmost axis: with three structures the string is wider than that panel
    # and ran over the rate-map titles beside it
    txt = []
    for a in range(len(regs)):
        for b in range(a + 1, len(regs)):
            r = np.corrcoef(RR.region_axis(d[regs[a]][0])[0],
                            RR.region_axis(d[regs[b]][0])[0])[0, 1]
            txt.append(f'{regs[a][0]}–{regs[b][0]} {r:+.2f}')
    RR.draw_row(fig, OUTER[i], mo, dy, d, regs, letter='ABCDEF'[i],
                headline=f'M{mo} D{dy} — {why}   ({",  ".join(txt)})')

fig.suptitle('The state outside MEC: sessions where entorhinal cortex does not '
             'supply most of the cells', fontsize=8.6, y=1.0)
fig.savefig(OUT, bbox_inches='tight')
print(f'\nwrote {OUT}')
