"""Figure S15: how the gross anatomy of each shank is estimated, on M26 and M20.

One page per mouse, the same panels on each. M20 is the second example
(2026-11-01): of the mice not already shown elsewhere, its DiI makes four
discrete tracks, one per shank (sections 5, 7, 10 and 13, 120-180 um apart, the
shanks' 180 um mediolateral spacing in that animal); M25 and M28 carry DiI in
nearly every section.

A  The DAPI / DiI sections of M26, lateral to medial, 60 um apart, at true aspect.
B  The DiI signal alone: red above the blue counterstain (R - 0.6 B - 40).
C  DiI per section, with the peaks each shank's track makes (orange), assigned to
   shanks in mediolateral order.
D  For each shank, medial to lateral: above, its peak section (DAPI / DiI, with
   the 500 um scale bar's calibration); below, the LFP spectra and curated cells
   of that shank's single-shank session, with the rule-based boundary and exit
   of anatomy_rules.py and the atlas border.

The section strips are data/histology/M{mouse}_sections.png, the image
embedded in ~/Desktop/Mouse_slices/M{mouse}.pdf (the full-resolution export;
the 500 um scale bar is 85 px in M26 and 114 px in M20).

Writes fig_supp_histology_example.pdf (M26) and fig_supp_histology_example_M20.pdf
"""
import os
import numpy as np, pandas as pd, matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from PIL import Image
from scipy.signal import find_peaks
import anatomy_rules as AR, boundary_unsupervised as BU

Image.MAX_IMAGE_PIXELS = None
plt.rcParams['font.family'] = 'Arial'
plt.rcParams['pdf.fonttype'] = 42
FIG = os.path.dirname(os.path.abspath(__file__))
SEC_UM = 60.
HIST = '/Users/harryclark/Documents/spatial-manifolds/data/histology'
# per mouse: sections, um per px (500 um bar), and the single-shank session of
# each shank, medial (0) -> lateral (3), the one with most curated cells where
# a shank was recorded alone more than once
MICE = {26: dict(nsec=17, um_px=500 / 85, sess={0: 17, 1: 18, 2: 11, 3: 12},
                 out=f'{FIG}/fig_supp_histology_example.pdf'),
        20: dict(nsec=16, um_px=500 / 114, sess={0: 25, 1: 20, 2: 21, 3: 22},
                 out=f'{FIG}/fig_supp_histology_example_M20.pdf')}
RC = {'VIS': '#3f9b4f', 'MEC': '0.15', 'other': '0.65'}
CX = pd.read_csv('/Users/harryclark/Documents/spatial-manifolds/data/cell_classifications_v2.csv')

def build(MO, nsec, um_px, sess, out):
    NSEC, UM_PX, SESS, OUT = nsec, um_px, sess, out
    a = np.asarray(Image.open(f'{HIST}/M{MO}_sections.png').convert('RGB')).astype(np.float32)
    H, Wt = a.shape[:2]; W = Wt / NSEC
    f_ = max(1, int(round(25 / UM_PX)))
    hb, wb = H // f_, Wt // f_
    ab = a[:hb * f_, :wb * f_].reshape(hb, f_, wb, f_, 3).mean((1, 3))
    ex = np.clip(ab[..., 0] - .6 * ab[..., 2] - 40, 0, None)
    Wb = wb / NSEC
    sig = np.array([ex[:, int(round(i * Wb)):int(round((i + 1) * Wb))].sum() for i in range(NSEC)]); sig /= sig.max()
    pk, pr = find_peaks(np.r_[0, sig, 0], prominence=.12); pk = list(pk - 1)
    if len(pk) > 4:                       # the four most prominent tracks
        pk = sorted(np.array(pk)[np.argsort(-pr['prominences'])[:4]])
    # the most medial shank leaves only a weak, broad track: its section is the
    # strongest one medial to the third peak
    if len(pk) == 3:
        rest = np.arange(pk[-1] + 1, NSEC)
        pk.append(int(rest[np.argmax(sig[rest])]))
    pk = sorted(pk)[:4]                                   # lateral -> medial
    shank_sec = {3 - j: p for j, p in enumerate(pk)}      # shank index medial(0) -> lateral(3)
    print('peak sections (1-based, lateral->medial):', [p + 1 for p in pk])

    fig = plt.figure(figsize=(10.4, 12.6))
    G = fig.add_gridspec(4, 1, height_ratios=[.55, .55, .55, 4.2], hspace=.28, left=.07, right=.99, top=.97, bottom=.05)
    disp = Image.fromarray(ab.astype(np.uint8))
    ext = [.5, NSEC + .5, hb / Wb, 0]
    axA = fig.add_subplot(G[0]); axA.imshow(disp, aspect='equal', extent=ext)
    axB = fig.add_subplot(G[1]); axB.imshow(ex, aspect='equal', cmap='magma', extent=ext, vmin=0,
                                           vmax=np.percentile(ex[ex > 0], 99))
    for ax_, lab, L in ((axA, 'DAPI / DiI', 'A'), (axB, 'DiI (blue\nsubtracted)', 'B')):
        ax_.set_yticks([]); ax_.set_xticks([]); ax_.set_anchor('W')
        ax_.set_ylabel(lab, fontsize=7.5, rotation=0, ha='right', va='center')
        for i in range(1, NSEC):
            ax_.axvline(i + .5, color='w', lw=.4, alpha=.5)
        ax_.text(-.07, 1.02, L, transform=ax_.transAxes, fontsize=10, weight='bold')
    axA.set_title(f'M{MO}: {NSEC} sections, {SEC_UM:.0f} um apart, lateral (left) to medial (right)', fontsize=8.5, loc='left')
    axC = fig.add_subplot(G[2])
    cols = ['#bbbbbb'] * NSEC
    for p in pk:
        cols[p] = '#e66101'
    axC.bar(np.arange(1, NSEC + 1), sig, color=cols, width=.8)
    for k, p in shank_sec.items():
        axC.annotate(f'shank {k + 1}', (p + 1, sig[p]), xytext=(0, 2), textcoords='offset points', ha='center', fontsize=6.5)
    axC.set_xlim(.5, NSEC + .5); axC.set_ylim(0, 1.3); axC.set_xticks(range(1, NSEC + 1)); axC.tick_params(labelsize=6.5)
    axC.set_ylabel('DiI\n(norm.)', fontsize=7.5, rotation=0, ha='right', va='center')
    axC.set_xlabel('section', fontsize=7.5, labelpad=1); axC.spines[['top', 'right']].set_visible(False)
    axC.text(-.07, 1.02, 'C', transform=axC.transAxes, fontsize=10, weight='bold')
    axA.apply_aspect(); axB.apply_aspect()
    pa = axA.get_position()
    for ax_ in (axB, axC):
        p_ = ax_.get_position(); ax_.set_position([pa.x0, p_.y0, pa.width, p_.height])

    gD = G[3].subgridspec(1, 4, wspace=.14)
    for k in range(4):
        dy = SESS[k]; sec = shank_sec[k]
        g = gD[k].subgridspec(2, 1, height_ratios=[1, 1.35], hspace=.18)
        # the shank's section, at true aspect, with a 500 um bar
        s0, s1 = int(round(sec * W)), int(round((sec + 1) * W))
        crop = a[:, s0:s1].astype(np.uint8)
        axS = fig.add_subplot(g[0]); axS.imshow(crop, aspect='equal')
        axS.plot([crop.shape[1] * .07, crop.shape[1] * .07 + 500 / UM_PX], [crop.shape[0] * .93] * 2, color='w', lw=2)
        axS.text(crop.shape[1] * .07, crop.shape[0] * .88, '500 um', color='w', fontsize=6)
        axS.set_xticks([]); axS.set_yticks([])
        axS.set_title(f'shank {k + 1} ({AR.SH[(MO, k)]:.0f} um ML)\nsection {sec + 1}', fontsize=7.5, pad=3)
        if k == 0:
            axS.text(-.12, 1.15, 'D', transform=axS.transAxes, fontsize=10, weight='bold')
        # its single-shank session: atlas strip, spectra, cells, rule lines
        C, f, LP = AR.load_spectra(MO, dy)
        Z = (LP - LP.mean(0)) / LP.std(0)
        rule = AR.session_rule(MO, dy)
        vis, mec = C.dv[C.region == 'VIS'], C.dv[C.region == 'MEC']
        b = (vis.max() + mec.min()) / 2 if len(vis) and len(mec) else np.nan
        ylim = (C.dv.max() + 60, C.dv.min() - 60)
        gl = g[1].subgridspec(1, 3, width_ratios=[.08, 1, .52], wspace=.05)
        a0 = fig.add_subplot(gl[0])
        for d, rg in zip(C.dv, C.region):
            a0.barh(d, 1, height=58, color=RC[rg])
        a0.set_ylim(ylim); a0.set_xticks([]); a0.tick_params(labelsize=6)
        if k == 0:
            a0.set_ylabel('dorsoventral (um)', fontsize=7.5)
        ax = fig.add_subplot(gl[1])
        ax.imshow(Z, aspect='auto', cmap='RdBu_r', vmin=-2.5, vmax=2.5, extent=[0, len(f), ylim[0], ylim[1]],
                  interpolation='nearest')
        ax.set_xticks([np.argmin(abs(f - x)) for x in (10, 60, 200)]); ax.set_xticklabels([10, 60, 200], fontsize=6)
        ax.set_xlabel('frequency (Hz)', fontsize=6.5, labelpad=1); ax.tick_params(labelleft=False)
        ax.set_title(f'D{dy}', fontsize=7.5, pad=2, loc='left')
        cc = CX[(CX.mouse == MO) & (CX.day == dy) & CX.cell_class_of1.isin(['GC', 'NGS', 'NS'])].dropna(subset=['coord_SCs_y'])
        a2 = fig.add_subplot(gl[2], sharey=ax)
        rng = np.random.default_rng(0)
        colj = AR.cell_column(cc)
        for j in range(4):
            m = cc[colj == j]
            a2.scatter(j + rng.uniform(-.3, .3, len(m)), m.coord_SCs_y, s=AR.rate_size([AR.vr_rates(MO, dy).get(int(c_), .1) for c_ in m.cluster_id]), c=[RC[BU.region(x)] for x in m.brain_region_atlas], lw=0, alpha=.7)
        a2.set_xticks([0, 1, 2, 3]); a2.set_xticklabels(['G', 'N', 'S', 'I'], fontsize=6); a2.set_xlim(-.6, 3.6)
        a2.tick_params(labelleft=False, labelsize=6); a2.spines[['top', 'right']].set_visible(False)
        a2.set_title(f'{len(cc)} cells', fontsize=6.5, pad=2)
        for a_ in (a0, ax, a2):
            if np.isfinite(b):
                a_.axhline(b, color='k', ls='--', lw=.8)
        for a_ in (ax, a2):
            a_.axhline(rule['boundary'], color='#e66101', lw=1.3)
            a_.axhline(rule['exit'], color='#5e3c99', lw=1.1)
            a_.axhspan(rule['exit'], ylim[0], color='0.6', alpha=.35, lw=0, zorder=3)
        print(f'shank {k + 1}: D{dy}, section {sec + 1}, boundary {rule["boundary"]:.0f}, exit {rule["exit"]:.0f}, '
              f'atlas {b:.0f}, cerebellum cells {rule["n_cerebellum"]}')
    from matplotlib.lines import Line2D
    fig.legend(handles=[Line2D([], [], color='#e66101', lw=1.3, label='VIS - MEC/PARA boundary (40-300 Hz block onset)'),
                        Line2D([], [], color='#5e3c99', lw=1.1, label='leaves cortex (grey: below)'),
                        Line2D([], [], color='k', ls='--', lw=.8, label='atlas VIS - MEC border'),
                        Line2D([], [], marker='o', ls='', color='#3f9b4f', ms=3, label='atlas VIS cell'),
                        Line2D([], [], marker='o', ls='', color='0.15', ms=3, label='atlas MEC cell'),
                        Line2D([], [], marker='o', ls='', color='0.65', ms=3, label='atlas other'),
                        Line2D([], [], marker='o', ls='', color='0.4', ms=1.6, label='5 Hz'),
                        Line2D([], [], marker='o', ls='', color='0.4', ms=5.8, label='100 Hz (size: firing rate)')],
               loc='lower center', ncol=8, fontsize=6.3, frameon=False, bbox_to_anchor=(.5, -.005))
    fig.savefig(OUT + '.partial', format='pdf', dpi=250); os.replace(OUT + '.partial', OUT)
    print('wrote', OUT)


if __name__ == '__main__':
    for mo_, kw in MICE.items():
        build(mo_, **kw)
