"""Exploratory: every single-shank session in which MEC's own state axis is
reliable and switches, drawn as Figure 7A-D, to see how much visual cortex
changes with MEC. Not a paper figure.

Sessions: single shank (cells span < 100 um mediolaterally), MEC split-half
reliability >= 0.75 and at least one MEC switch, plus M21 D25 as the
counter-example (visual axis reliable, the two apparently anti-correlated).
Each panel's header adds both axes' split-half reliability and r(MEC, VIS)
against its ceiling sqrt(rel_MEC * rel_VIS).

Writes explore_single_shank_vis_mec.pdf
"""
import os

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

import fig7_supp_vis_examples as S

OUT = f'{S.F.FIG}/explore_single_shank_vis_mec.pdf'
SESS = [(21, 19), (21, 20), (21, 21), (21, 26), (26, 12), (26, 18),
        (27, 23), (27, 24), (28, 19), (28, 20), (29, 20), (21, 25)]

if __name__ == '__main__':
    SH = pd.read_csv(f'{S.F.PS}/s15_shared_sessions.csv').set_index(['mouse', 'day'])
    ex = S.REG[S.REG.set_index(['mouse', 'day']).index.isin(SESS)].excess.dropna()
    vlim = float(max(abs(np.percentile(ex, 2)), abs(np.percentile(ex, 98))))
    nr = (len(SESS) + 1) // 2
    fig = plt.figure(figsize=(10.4, 3.1 * nr))
    outer = fig.add_gridspec(nr, 2, hspace=.42, wspace=.12, left=.05, right=.985,
                             top=1 - .25 / nr, bottom=.02)
    for i, (mo, dy) in enumerate(SESS):
        cell = outer[i // 2, i % 2]
        S.example(fig, cell, mo, dy, chr(65 + i), vlim)
        r = SH.loc[(mo, dy)]
        ceil = np.sqrt(max(r.rel_mec, 0) * max(r.rel_vis, 0))
        p = cell.get_position(fig)
        fig.text(p.x0 + .30 * p.width, p.y1 + .012,
                 f'reliability MEC {r.rel_mec:.2f}, VIS {r.rel_vis:.2f};  r = {r.r_mv:+.2f} '
                 f'({r.r_mv / ceil:+.2f} of ceiling {ceil:.2f}), shift-null p = {r.p_mv:.2g}',
                 fontsize=6.3, color='0.25')
    fig.savefig(OUT, format='pdf', dpi=200)
    print('wrote', OUT)
