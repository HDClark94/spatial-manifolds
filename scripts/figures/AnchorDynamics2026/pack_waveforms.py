"""
Pack the waveform information from every VR and OF1 SortingAnalyzer (.zarr) into one
small file, so units can be curated by eye on a laptop (templates that do not look like
spikes but still pass bombcell).

A full analyzer is ~170 MB and thousands of files per session, almost all of it dense
templates (units x 90 samples x 384 channels, float64) and per-spike amplitudes and
locations. Per unit this keeps only:
  - average and std template on the N_CHAN channels nearest the peak channel (uV, float16)
  - the peak channel, the positions of those channels, and the unit location
  - up to N_SPK spikes, chosen at random: time (s), amplitude (uV) and depth (um), for
    amplitude-over-time and drift plots; none for units bombcell labels noise, which
    are never used (their templates and metrics are kept)
  - the ISI histogram, quality metrics, template metrics, the bombcell label, and the
    N_SIM most similar units by template
Unit ids are the analyzer's unit_ids, which are the cluster_ids of the clusters.npz files
and of data/cell_classifications_v2.csv.

Usage (reads the analyzers in place on the mounted datastore; ~4.5 min a session over
SMB, so run several workers):
    python pack_waveforms.py w n          # worker w of n: one .npz per session in PARTS
    python pack_waveforms.py --merge      # once every session is packed: PARTS -> OUT
Packed sessions are skipped, so a stopped run resumes.

SRC holds M*/D*/{VR,OF1}/*_analyzer.zarr. In OUT every key is
'M25D16_VR/<field>'; np.load reads keys lazily, so one session loads without the rest:
    W = np.load('clark2025_waveforms.npz')
    W['M25D16_VR/unit_ids'], W['M25D16_VR/avg']           # avg: units x 90 x N_CHAN
"""
import glob
import json
import os
import sys
import time
import zipfile

import numpy as np
import zarr

N_CHAN = 16       # channels kept per unit, nearest the peak channel
N_SPK = 2000      # spikes kept per unit for amplitude/drift plots
N_SIM = 5         # most similar units kept per unit
TYPES = ('VR', 'OF1')
SRC = '/Volumes/INCR-NolanLab/ActiveProjects/Wolf/harry/data/clark2025'
PARTS = '/Users/harryclark/Documents/spatial-manifolds/data/waveforms/parts'
OUT = '/Users/harryclark/Documents/spatial-manifolds/data/waveforms/clark2025_waveforms.npz'


def pack_session(zpath, out):
    z = zarr.open(zpath, 'r')
    e = z['extensions']
    uid = z['sorting/unit_ids'][:]
    n = len(uid)
    fs = z['sorting'].attrs['sampling_frequency']

    probe = z['recording_info'].attrs['probegroup']['probes'][0]
    pos = np.asarray(probe['contact_positions'], float)
    dci = np.asarray(probe['device_channel_indices'])
    pos = pos[np.argsort(dci)]                         # rows in recording channel order
    shank = np.asarray(probe['shank_ids'], int)[np.argsort(dci)]

    avg = e['templates/average'][:]
    sd = e['templates/std'][:]
    peak = np.abs(avg).max(1).argmax(1)                # channel with the largest |template|
    near = np.argsort(np.linalg.norm(pos[None] - pos[peak][:, None], axis=2), 1)[:, :N_CHAN]
    rows = np.arange(n)[:, None]
    avg_n = avg.transpose(0, 2, 1)[rows, near].transpose(0, 2, 1).astype(np.float16)
    sd_n = sd.transpose(0, 2, 1)[rows, near].transpose(0, 2, 1).astype(np.float16)
    del avg, sd

    bc = glob.glob(zpath.replace('_analyzer.zarr', '_bombcell_curation.json'))
    lab = {}
    if bc:
        for m in json.load(open(bc[0]))['manual_labels']:
            lab[m['unit_id']] = m['labels']['bombcell_label'][0]
    label = np.array([lab.get(int(u), '') for u in uid])

    ui = z['sorting/spikes/unit_index'][:]
    t = z['sorting/spikes/sample_index'][:] / fs
    amp = e['spike_amplitudes/amplitudes'][:]
    loc_y = e['spike_locations/spike_locations'][:]['y']
    rng = np.random.default_rng(0)
    order = np.argsort(ui, kind='stable')
    bounds = np.searchsorted(ui[order], np.arange(n + 1))
    keep = [np.sort(rng.choice(order[a:b], 0 if label[i] == 'noise' else min(N_SPK, b - a),
                               replace=False))
            for i, (a, b) in enumerate(zip(bounds[:-1], bounds[1:]))]
    spk_n = np.array([len(k) for k in keep])
    keep = np.concatenate(keep)

    sim = e['template_similarity/similarity'][:]
    np.fill_diagonal(sim, -np.inf)
    top = np.argsort(-sim, 1)[:, :N_SIM]

    D = dict(unit_ids=uid, fs=np.float32(fs), peak_channel=peak, channels=near,
             channel_xy=pos.astype(np.float32), channel_shank=shank,
             avg=avg_n, std=sd_n, noise_levels=e['noise_levels/noise_levels'][:],
             unit_xyz=e['unit_locations/unit_locations'][:].astype(np.float32),
             spike_n_total=np.diff(bounds), spike_n=spk_n,
             spike_t=t[keep].astype(np.float32), spike_amp=amp[keep].astype(np.float16),
             spike_y=loc_y[keep].astype(np.float16),
             isi_bins=e['isi_histograms/bins'][:], isi=e['isi_histograms/isi_histograms'][:].astype(np.int32),
             similar_ids=uid[top], similar=np.take_along_axis(sim, top, 1).astype(np.float16))
    for ext in ('quality_metrics', 'template_metrics'):
        for k, v in e[f'{ext}/metrics'].items():
            if k != 'index':
                D[f'{ext[0]}m_{k}'] = v[:].astype(np.float32)
    for k in ('KSLabel', 'Amplitude', 'ContamPct'):
        if k in z['sorting/properties']:
            D[f'ks_{k}'] = z[f'sorting/properties/{k}'][:]

    D['bombcell'] = label

    np.savez_compressed(out + '.part.npz', **D)
    os.replace(out + '.part.npz', out)


def merge(parts, out):
    tmp = out + '.part.npz'
    with zipfile.ZipFile(tmp, 'w', zipfile.ZIP_STORED) as zo:
        for p in sorted(glob.glob(f'{parts}/M*D*_*.npz')):
            ses = os.path.basename(p)[:-4]
            with zipfile.ZipFile(p) as zi:
                for name in zi.namelist():             # members are already deflated .npy
                    zo.writestr(f'{ses}/{name}', zi.read(name), zipfile.ZIP_DEFLATED)
    os.replace(tmp, out)


if __name__ == '__main__':
    os.makedirs(PARTS, exist_ok=True)
    zs = sorted(p for p in glob.glob(f'{SRC}/M*/D*/*/*_analyzer.zarr') if p.split('/')[-2] in TYPES)
    name = lambda zp: '{}{}_{}'.format(*zp.split('/')[-4:-1])
    if not zs:
        raise SystemExit(f'no analyzers under {SRC}: is the datastore mounted?')
    if sys.argv[1:] == ['--merge']:
        missing = [name(zp) for zp in zs if not os.path.exists(f'{PARTS}/{name(zp)}.npz')]
        if missing:
            raise SystemExit(f'{len(missing)} sessions not packed, e.g. {missing[:3]}; not merging')
        merge(PARTS, OUT)
        print(f'{len(zs)} sessions -> {OUT}: {os.path.getsize(OUT) / 1e6:.0f} MB', flush=True)
    else:
        w, n = int(sys.argv[1]), int(sys.argv[2])
        print(f'{len(zs)} analyzers under {SRC}; worker {w} of {n}', flush=True)
        for zp in zs[w::n]:
            out = f'{PARTS}/{name(zp)}.npz'
            if os.path.exists(out):
                continue
            t0 = time.time()
            try:
                pack_session(zp, out)
                print(f'{name(zp)}: {os.path.getsize(out) / 1e6:.1f} MB, {time.time() - t0:.0f} s', flush=True)
            except Exception as ex:
                print(f'{name(zp)} failed: {ex!r}', flush=True)
