"""Do fast-spiking interneurons lock to gamma, and does it move with the state?

The model in Figure 8 has interneurons generating the gamma rhythm that nests
in theta, and Figure 2 has interneurons among the cells that follow the
anchoring state. Those two claims only meet if the interneurons in THIS dataset
are actually coupled to the gamma in this LFP. That is what this measures.

TWO THINGS DECIDE WHETHER THE ANSWER MEANS ANYTHING.

SPIKE COUNT. Fast-spiking cells fire several times more than principal cells,
and the mean resultant length rises as spike count falls, so an MRL comparison
between the two is mostly a comparison of firing rate -- in the direction that
makes interneurons look WORSE. Pairwise phase consistency is unbiased by count
(Vinck et al. 2010) and is used throughout here.

SPIKE CONTAMINATION, which is the trap specific to this question. A fast-spiking
cell's own waveform leaks into the local field at exactly the frequencies being
tested, so measuring a cell against the channel it was recorded on manufactures
the result: the cell is partly correlating with itself. Every number reported
here uses an LFP channel group at least MIN_SEP groups away from the cell's own.
The own-channel value is computed as well, purely to show the size of the
artifact that would otherwise be reported.

Bands are Figure 7's: slow gamma 30-48 Hz and 60-100 Hz, which the CA1
literature would call mid gamma. Trials are speed-matched between states as
everywhere else, and running samples only.

WHAT IT FOUND, over 4,308 entorhinal cells in 32 sessions.

There IS spike-field coupling to gamma here, and it is real rather than leak:
PPC falls from the cell's own group to a group 120 um away and then to near
zero at 1.2 mm, which is a local field, not a waveform.

But it does not single out the fast-spiking cells. Interneurons are no more
locked than principal cells in either band (per session, p = 0.60 slow and
p = 0.92 fast), and GRID cells are the most locked of the four identities --
about twice any other class in both bands. Firing rate does not explain this;
PPC is uncorrelated with spike count within every class.

The one result that does support the model: in 60-100 Hz gamma, interneurons
fire 32 degrees LATER in the cycle than principal cells (p = 9.6e-5, later in
25 of 31 sessions), which is the direction PING predicts, with the pyramidal
population leading and the fast-spiking cells following. There is no such lag
in 30-48 Hz gamma (p = 0.77), and in that band the interneurons have no
consistent preferred phase across cells at all (R = 0.05, against 0.22-0.28
for the principal classes) while in fast gamma every class is consistent
(R = 0.47-0.59).

The coupling does not change with the anchoring state. Four comparisons --
two bands by two cell classes -- give Holm-corrected p >= 0.17, the smallest
being principal cells locking slightly less to fast gamma when anchored
(0.0028 against 0.0042, raw p = 0.043), which is the same direction as the
theta-fast-gamma uncoupling in Figure 7G but does not survive correction.

So the interneuron-gamma link in the Figure 8 model is partly supported --
the phase lag is there -- but the privileged coupling it assumes is not, and
nothing here ties that coupling to the anchoring state.

Writes data/lfp/gamma_locking.csv
"""
import os
import sys
import time
import warnings

import h5py
import numpy as np
import pandas as pd

warnings.filterwarnings('ignore')
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, '/Users/harryclark/Documents/spatial-manifolds/src')
from scipy.fft import fft, ifft, next_fast_len

from lfp_batch import (FS, LFP_ROOT, clip_trials, nap, speed_matched, vr_paths)

ROOT = '/Users/harryclark/Documents/spatial-manifolds'
OUT = f'{ROOT}/data/lfp/gamma_locking.csv'

BANDS = {'slow': (30., 48.), 'fast': (60., 100.)}
RUN_SPEED = 3.0
MIN_SPIKES = 50            # per cell per state; PPC is unbiased but still noisy
MIN_SEP = 2                # channel groups between a cell and its LFP, ~120 um


def ppc(ph):
    """Pairwise phase consistency: unbiased by the number of spikes."""
    n = len(ph)
    if n < 2:
        return np.nan
    r = np.abs(np.sum(np.exp(1j * ph)))
    return float((r * r - n) / (n * (n - 1)))


def band_phase(x, lo, hi):
    """Analytic phase in one band, from a single transform."""
    n = len(x)
    nf = next_fast_len(n)
    X = fft(x, nf)
    freqs = np.fft.fftfreq(nf, 1 / FS)
    Y = np.zeros(nf, complex)
    m = (freqs >= lo) & (freqs <= hi)
    Y[m] = 2.0 * X[m]
    return np.angle(ifft(Y)[:n])


def run_session(mo, dy, state, ident, rng):
    lf = f'{LFP_ROOT}/M{mo}/D{dy}/VR/sub-M{mo}_ses-D{dy}_typ-VR_beh.nwb'
    if not os.path.exists(lf):
        return []
    fh = h5py.File(lf, 'r')
    if 'processing/ecephys/LFP/lfp/data' not in fh:
        fh.close(); return []
    names = [n.decode() for n in
             fh['general/extracellular_ephys/electrodes/channel_name'][:]]
    ch2g = {c: i for i, n in enumerate(names) for c in n.split('_')}
    ana = pd.read_csv(f'{LFP_ROOT}/M{mo}/D{dy}/anatomy_M{mo}_D{dy}.csv')
    ent = {c for c, r in zip(ana.channel_id, ana.brain_region.astype(str))
           if r.startswith('ENTm')}
    ent_groups = sorted({ch2g[c] for c in ent if c in ch2g})
    if len(ent_groups) < 2 * MIN_SEP + 1:
        fh.close(); return []

    bp, cp = vr_paths(mo, dy)
    beh = nap.load_file(bp); clusters = nap.load_file(cp)
    trials, _ = clip_trials(beh['trials'].as_dataframe(), clusters)
    ec = clusters.metadata['extremum_channel']
    cells = []
    for c in [int(x) for x in clusters.index]:
        k = (mo, dy, c)
        if k not in ident:
            continue
        g = ch2g.get(str(ec[c]), -1)
        if g in ent_groups:
            cells.append((c, g))
    if not cells:
        fh.close(); return []

    # Each cell's own group, the NEAREST group at least MIN_SEP away, and the
    # furthest one. Nearest-beyond-the-leak is the number to report: gamma is
    # local, with a coherence length of a few hundred microns, so scoring a
    # cell against the far end of the entorhinal span would manufacture a null
    # as surely as scoring it against its own channel manufactures a result.
    # The three separations together show whether coupling decays like a leak
    # or plateaus like a field.
    def near(g):
        cand = [h for h in ent_groups if abs(h - g) >= MIN_SEP]
        return min(cand, key=lambda h: abs(h - g)) if cand else -1

    def far(g):
        cand = [h for h in ent_groups if abs(h - g) >= MIN_SEP]
        return max(cand, key=lambda h: abs(h - g)) if cand else -1
    want = sorted(({g for _, g in cells} | {near(g) for _, g in cells}
                   | {far(g) for _, g in cells}) - {-1})
    d = fh['processing/ecephys/LFP/lfp/data']
    nT = d.shape[0]
    blk = (d.chunks[0] if d.chunks else 26041) * 8
    X = np.empty((nT, len(want)), np.float32)
    for i in range(0, nT, blk):
        X[i:i + blk] = d[i:i + blk, :][:, want]
    fh.close()
    col = {g: i for i, g in enumerate(want)}

    t = np.arange(nT) / FS
    S = beh['S']
    spd = np.asarray(S.values)[np.searchsorted(np.asarray(S.index), t)
                               .clip(0, len(S) - 1)]
    run = spd >= RUN_SPEED
    info = []
    for _, tr in trials.iterrows():
        u = int(tr.number)
        if u not in state:
            continue
        m = (t >= tr.start) & (t <= tr.end) & run
        if m.sum() < FS:
            continue
        info.append((bool(state[u]), float(spd[m].mean()), m))
    if len(info) < 10:
        return []
    km = speed_matched(np.array([i[1] for i in info]),
                       np.array([i[0] for i in info]), rng)
    masks = {True: np.zeros(nT, bool), False: np.zeros(nT, bool)}
    for i in km:
        masks[info[i][0]] |= info[i][2]

    PH = {}
    for bn, (lo, hi) in BANDS.items():
        for g in want:
            PH[(bn, g)] = band_phase(X[:, col[g]].astype(float), lo, hi)

    rows = []
    for c, g in cells:
        gn, gf = near(g), far(g)
        if gn < 0:
            continue
        sp = np.asarray(clusters[c].index, dtype=float)
        idx = np.round(sp * FS).astype(int)
        idx = idx[(idx >= 0) & (idx < nT)]
        r = dict(mouse=mo, day=dy, cluster_id=c, identity=ident[(mo, dy, c)],
                 group=g, near_group=gn, far_group=gf, n_spikes=len(idx),
                 sep_near=abs(gn - g), sep_far=abs(gf - g))
        ok = True
        for st, tag in ((True, 'a'), (False, 'n')):
            ii = idx[masks[st][idx]]
            r[f'n_{tag}'] = len(ii)
            if len(ii) < MIN_SPIKES:
                ok = False
            for bn in BANDS:
                r[f'ppc_{bn}_{tag}'] = ppc(PH[(bn, gn)][ii])     # the result
                r[f'own_{bn}_{tag}'] = ppc(PH[(bn, g)][ii])      # leak, for scale
                r[f'far_{bn}_{tag}'] = ppc(PH[(bn, gf)][ii])     # decay check
                r[f'phi_{bn}_{tag}'] = float(np.angle(
                    np.mean(np.exp(1j * PH[(bn, gn)][ii])))) if len(ii) else np.nan
        if ok:
            rows.append(r)
    return rows


if __name__ == '__main__':
    T = pd.read_csv(f'{ROOT}/data/population_state/anchoring_trials.csv')
    U = pd.read_csv(f'{ROOT}/data/population_state/unit_table.csv')
    ident = {(int(a), int(b), int(c)): d for a, b, c, d in
             U[['mouse', 'day', 'cluster_id', 'identity']].itertuples(index=False)
             if isinstance(d, str)}
    sess = sorted(set(zip(T.mouse.astype(int), T.day.astype(int))))
    out, t0 = [], time.time()
    for mo, dy in sess:
        st = T[(T.mouse == mo) & (T.day == dy)]
        if not len(st):
            continue
        state = dict(zip(st.trial.astype(int), st.frac_anch > 0.5))
        rng = np.random.default_rng(abs(hash((mo, dy))) % 2 ** 32)
        try:
            r = run_session(mo, dy, state, ident, rng)
        except Exception as e:
            print(f'  ! M{mo}D{dy}: {type(e).__name__}: {e}', flush=True); continue
        out += r
        if r:
            print(f'  M{mo}D{dy}: {len(r)} cells  ({time.time() - t0:.0f}s)',
                  flush=True)
    D = pd.DataFrame(out)
    D.to_csv(OUT, index=False)
    print(f'\nwrote {OUT}: {len(D)} cells, '
          f'{D.groupby(["mouse", "day"]).ngroups if len(D) else 0} sessions')
