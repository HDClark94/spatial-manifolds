"""Unit cards from the packed waveform file (pack_waveforms.py), for curating
by eye units that pass bombcell but do not look like spikes of neural origin.

Each card: the average template (+-1 SD) on the 16 channels nearest the peak, drawn at
their probe positions; amplitude and spike depth over the session; the ISI histogram; and
bombcell label, SNR, rate, ISI violations, peak-to-trough duration and peak counts.

Usage:
    python view_waveforms.py M25D16_OF1                  # every unit bombcell did not call noise
    python view_waveforms.py M25D16_OF1 69 92 167        # these units
    python view_waveforms.py M25D16_OF1 --file other.npz
    python view_waveforms.py --cells cells.csv [--type VR] [--out cards.pdf]
        cells.csv has mouse, day, cluster_id and an optional note column; units are
        drawn from the OF1 session unless --type says otherwise
Writes waveform_cards_<session>.pdf (or --out), six units to an A4 page.
"""
import os
import sys

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt                                         # noqa: E402
import numpy as np                                                      # noqa: E402
from matplotlib.backends.backend_pdf import PdfPages                    # noqa: E402

FILE = '/Users/harryclark/Documents/spatial-manifolds/data/waveforms/clark2025_waveforms.npz'
LAB_C = {'good': '#1b998b', 'mua': '#e09f3e', 'non_soma': '#6d2e9e', 'noise': '0.5', '': '0.5'}


def load(ses, path=FILE):
    W = np.load(path)
    D = {k.split('/', 1)[1]: W[k] for k in W.files if k.startswith(ses + '/')}
    if not D:                                   # a single per-session part file
        D = dict(np.load(path)) if os.path.basename(path).startswith(ses) else {}
    if not D:
        raise SystemExit(f'{ses} not in {path}')
    D['first'] = np.r_[0, np.cumsum(D['spike_n'])[:-1]]
    return D


def card(fig, sub, D, i, note=''):
    g = sub.subgridspec(2, 2, width_ratios=[1, 1.3], hspace=.55, wspace=.35)
    lab = str(D['bombcell'][i])
    xy = D['channel_xy'][D['channels'][i]]
    avg, sd = D['avg'][i].astype(float), D['std'][i].astype(float)
    t = np.linspace(-.42, .42, avg.shape[0])                # 3 ms drawn 0.84 x-units wide
    amp = np.abs(avg).max()
    s = 12 / max(amp, 1)                                    # 1 y-unit per um; peak 12 um tall
    a = fig.add_subplot(g[:, 0])
    for c in range(avg.shape[1]):
        x0, y0 = xy[c, 0] / 32, xy[c, 1]
        a.fill_between(x0 + t, y0 + s * (avg[:, c] - sd[:, c]), y0 + s * (avg[:, c] + sd[:, c]),
                       color='0.8', lw=0, edgecolor='none')
        a.plot(x0 + t, y0 + s * avg[:, c], color='0.45' if c else 'k', lw=.7 if c else 1.2)
    a.set_title(f'{note}unit {D["unit_ids"][i]}  {lab or "unlabelled"}', fontsize=7, color=LAB_C[lab], loc='left')
    a.text(0, 1.0, f'{amp:.0f} uV peak', transform=a.transAxes, fontsize=5, va='top')
    a.axis('off')

    j = slice(D['first'][i], D['first'][i] + D['spike_n'][i])
    b = fig.add_subplot(g[0, 1])
    if D['spike_n'][i]:
        b.scatter(D['spike_t'][j] / 60, D['spike_amp'][j].astype(float), s=.5, c='k', lw=0, rasterized=True)
        b2 = b.twinx()
        b2.scatter(D['spike_t'][j] / 60, D['spike_y'][j].astype(float), s=.5, c='#e09f3e', lw=0, rasterized=True)
        b2.tick_params(labelsize=5, colors='#e09f3e'); b2.set_ylabel('depth (um)', fontsize=5, color='#e09f3e')
    b.set_xlabel('time (min)', fontsize=5); b.set_ylabel('amplitude (uV)', fontsize=5)
    b.tick_params(labelsize=5)

    c = fig.add_subplot(g[1, 1])
    bins = D['isi_bins']
    c.bar(bins[:-1], D['isi'][i], width=np.diff(bins), align='edge', color='0.3', lw=0)
    c.axvspan(0, 1.5, color='#e09f3e', alpha=.2, lw=0)
    c.set_xlabel('ISI (ms)', fontsize=5); c.tick_params(labelsize=5)
    q = lambda k: float(D[k][i]) if k in D else np.nan
    c.set_title(f'SNR {q("qm_snr"):.1f}  {q("qm_firing_rate"):.1f} Hz  ISI viol {q("qm_isi_violations_ratio"):.2f}\n'
                f'p-t {1e3 * q("tm_peak_to_trough_duration"):.2f} ms  peaks +{q("tm_num_positive_peaks"):.0f}/'
                f'-{q("tm_num_negative_peaks"):.0f}  presence {q("qm_presence_ratio"):.2f}', fontsize=5, loc='left')
    for ax in (b, c):
        ax.spines[['top', 'right']].set_visible(False)


def write(items, out, title=''):
    """items: (D, unit index, note) per card."""
    with PdfPages(out) as pdf:
        for p in range(0, len(items), 6):
            fig = plt.figure(figsize=(7, 10))
            gs = fig.add_gridspec(3, 2, hspace=.35, wspace=.25, left=.04, right=.95, top=.95, bottom=.04)
            fig.suptitle(title, fontsize=8)
            for k, (D, i, note) in enumerate(items[p:p + 6]):
                card(fig, gs[k // 2, k % 2], D, i, note)
            pdf.savefig(fig); plt.close(fig)
    print(f'{len(items)} units -> {out}')


def opt(name, default):
    return sys.argv[sys.argv.index(name) + 1] if name in sys.argv else default


def main():
    path = opt('--file', FILE)
    if '--cells' in sys.argv:
        import pandas as pd
        cells, ty = pd.read_csv(opt('--cells', None)), opt('--type', 'OF1')
        S, items = {}, []
        for _, r in cells.iterrows():
            ses = f'M{r.mouse}D{r.day}_{ty}'
            D = S.setdefault(ses, load(ses, path))
            note = f'{ses}  {r.note}\n' if 'note' in r and isinstance(r.note, str) else f'{ses}\n'
            items.append((D, list(D['unit_ids']).index(r.cluster_id), note))
        write(items, opt('--out', 'waveform_cards.pdf'))
        return
    args = [a for a in sys.argv[1:] if not a.startswith('--') and a != path]
    ses, units = args[0], [int(u) for u in args[1:]]
    D = load(ses, path)
    uid = list(D['unit_ids'])
    idx = [uid.index(u) for u in units] if units else \
        [k for k in range(len(uid)) if D['bombcell'][k] != 'noise']
    write([(D, i, '') for i in idx], opt('--out', f'waveform_cards_{ses}.pdf'), ses)


if __name__ == '__main__':
    main()
