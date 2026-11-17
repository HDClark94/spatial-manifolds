'''Figure S16: more single-shank sessions in the format of Figure 7 A-D.

The sessions considered for Figure 7 and not used there: single-shank
recordings with clear switching in MEC and at least five switching visual cells,
chosen for many transitions, a blocky MEC PC1 and visual PC1 above its shift
null. Each row is one session, drawn exactly as a row of Figure 7 (example()
and maps_panel() of fig7_supp_vis_examples, the same agreement scale), with two
visual and three MEC example cells.

Each row also shows, as an inset beside the probe, the spike waveform of the most
dorsal curated unit on the shank (the arrowhead on the probe): the mean on each
of its 16 nearest contacts, laid out as the contacts are, from the VR sorting
(data/waveforms/clark2025_waveforms.npz), so that the cells at the top of the
recording can be judged as units.

Writes fig7_supp_more_examples.pdf
'''
import os
import sys

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import fig7_supp_vis_examples as S                                    # noqa: E402

SESSIONS = [(25, 19), (29, 19), (28, 20), (21, 26), (26, 12)]
WAVES = '/Users/harryclark/Documents/spatial-manifolds/data/waveforms/clark2025_waveforms.npz'
_W = np.load(WAVES, allow_pickle=True)


def waveform_inset(fig, axsl, axlf, axh, mo, dy):
    """The most dorsal curated unit's waveform footprint, beside the probe, and
    an arrowhead marking it on the probe."""
    c = axsl._cells
    j = int(np.nanargmax(c['pv']))
    cid, macro = int(c['cid'][j]), c['macro'][j]
    k = f'M{mo}D{dy}_VR'
    u = np.flatnonzero(_W[f'{k}/unit_ids'] == cid)
    if not len(u):
        print(f'  M{mo}D{dy}: no waveform for unit {cid}')
        return
    u = int(u[0])
    avg = _W[f'{k}/avg'][u].astype(float)                    # (samples, 16 contacts), uV
    sd = _W[f'{k}/std'][u].astype(float)
    xy = _W[f'{k}/channel_xy'][_W[f'{k}/channels'][u]]
    fs = float(_W[f'{k}/fs'])
    # arrowhead on the probe, left of the unit
    axsl.annotate('', xy=(_x_of(axsl, j) - 18, c['pv'][j]), xytext=(_x_of(axsl, j) - 120, c['pv'][j]),
                  arrowprops=dict(arrowstyle='-|>', color='k', lw=.8, mutation_scale=6), zorder=8)
    # inset: beside the probe, at its top; under the histology where there is one
    pl = axlf.get_position()
    w, h = .068, .052
    if axh is None:
        rect = [pl.x1 + .016, pl.y1 - h, w, h]
    else:
        # shrink the histology section to the top 42% of its slot, the legend
        # beneath it, and the inset under that
        ph = axh.get_position()
        f = .42
        axh.set_position([ph.x0 + ph.width * (1 - f) / 2, ph.y1 - ph.height * f,
                          ph.width * f, ph.height * f])
        rect = [ph.x0 + .004, pl.y0, w, h]
    ax = fig.add_axes(rect)
    t = np.arange(avg.shape[0]) / fs * 1000
    cols = np.unique(np.round(xy[:, 0]))
    peak = int(np.argmax(np.ptp(avg, axis=0)))
    gain = 30.0 / max(np.ptp(avg[:, peak]), 1e-6)            # peak spans two contact rows
    span = t[-1] * 1.25
    for ch in range(avg.shape[1]):
        x0 = np.searchsorted(cols, np.round(xy[ch, 0])) * span
        y0 = xy[ch, 1]
        col = 'k' if ch == peak else '0.55'
        if ch == peak:
            ax.fill_between(t + x0, y0 + gain * (avg[:, ch] - sd[:, ch]),
                            y0 + gain * (avg[:, ch] + sd[:, ch]), color='0.8', lw=0,
                            edgecolor='none', zorder=1)
        ax.plot(t + x0, y0 + gain * avg[:, ch], color=col, lw=.6 if ch == peak else .45, zorder=2)
    # scale bars: 10 uV, 1 ms
    xb, yb = -0.4, xy[:, 1].min() - 8
    ax.plot([xb, xb + 1], [yb, yb], color='k', lw=.8)
    ax.plot([xb, xb], [yb, yb + 10 * gain], color='k', lw=.8)
    ax.text(xb + .5, yb - 3, '1 ms', fontsize=4.6, ha='center', va='top')
    ax.text(xb - .15, yb + 5 * gain, '10 µV', fontsize=4.6, ha='right', va='center', rotation=90)
    ax.axis('off')
    ax.set_title(f'most dorsal unit\n{macro} {cid}, {np.ptp(avg[:, peak]):.0f} µV p-p',
                 fontsize=5.6, pad=1, linespacing=1.05)
    print(f'  M{mo}D{dy}: most dorsal curated unit {macro} {cid} at {c["pv"][j]:.0f} um, '
          f'{np.ptp(avg[:, peak]):.1f} uV peak-to-peak')


def _x_of(axsl, j):
    return axsl._cells_x[j]
OUT = f'{os.path.dirname(os.path.abspath(__file__))}/fig7_supp_more_examples.pdf'


def main():
    # Figure 7's agreement scale, so colours compare across the two figures
    ex = S.REG[S.REG.set_index(['mouse', 'day']).index.isin(S.SESSIONS)].excess.dropna()
    vlim = float(max(abs(np.percentile(ex, 2)), abs(np.percentile(ex, 98))))
    fig = plt.figure(figsize=(10.4, 3.0 * len(SESSIONS)))
    outer = fig.add_gridspec(len(SESSIONS), 1, hspace=.42, left=.06, right=.985,
                             top=1 - .25 / len(SESSIONS), bottom=.35 / len(SESSIONS))
    letters = 'ABCDEFGHIJ'
    for i, (mo, dy) in enumerate(SESSIONS):
        row = outer[i].subgridspec(1, 2, width_ratios=[1.05, 1], wspace=.14)
        picks = S.pick_followers(mo, dy, force=S.FORCE.get((mo, dy)))
        print(f'M{mo}D{dy} examples: ' + ', '.join(f'{k + 1}:{st} {c} r={a:+.2f}'
                                                  for k, (c, st, a, _) in enumerate(picks)))
        sc = S.example(fig, row[0], mo, dy, letters[2 * i], vlim,
                       marks=[(c, k + 1) for k, (c, *_) in enumerate(picks)])
        waveform_inset(fig, *fig._probe_axes, mo, dy)
        S.maps_panel(fig, row[1], mo, dy, picks, letters[2 * i + 1])
    cax = fig.add_axes([.07, .12 / len(SESSIONS), .16, .004])
    cb = fig.colorbar(sc, cax=cax, orientation='horizontal')
    cb.outline.set_visible(False); cb.ax.tick_params(labelsize=6, length=2, pad=1)
    cb.set_label('agreement above chance with the MEC PC1', fontsize=6.5, labelpad=2)
    fig.savefig(OUT, dpi=200, bbox_inches='tight')
    print('wrote', OUT)


if __name__ == '__main__':
    main()
