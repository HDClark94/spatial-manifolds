"""Where in the theta and gamma cycle do the detected synapses actually act?

Figure 5 finds 8,834 putative monosynaptic connections and asks whether
connected pairs share the anchoring state; they do not. That is a question
about the state LABEL. This asks a different one: does the EFFICACY of those
connections -- how much a presynaptic spike actually changes the target's
firing -- depend on where in the theta cycle the presynaptic spike lands, on
where in the gamma cycle it lands, and on the anchoring state.

WHY IT IS WORTH ASKING. Figure 9's gate is a GAIN: it scales how much the
error population is allowed to influence the attractor. Every measurement
supporting that figure so far is a correlate of the state -- band power,
theta-gamma coupling, spike-field locking, population rate against band
amplitude -- and none of them measures a synaptic gain. This does. A state
dependence of inhibitory efficacy would be the first direct measurement of the
quantity the model posits; its absence constrains the model in a way another
power comparison cannot.

It also gives the theta-gamma nesting of Figure 8 and Figure S10 a cellular
referent, which it presently lacks, and it tests the one surviving piece of
the interneuron story from Figure S11 -- that interneurons fire a median 45
degrees later than principal cells in 60-100 Hz gamma -- against a synaptic
rather than a population observable.

WHAT CANNOT BE TESTED HERE, stated up front. The PING direction the lag
predicts is principal -> interneuron, and this dataset has 18 such excitatory
pairs in 12 sessions with a median of 3,144 presynaptic spikes. That is not
enough for a phase-resolved measurement, so the loop can only be probed from
the inhibitory side: interneuron -> grid, which has 303 pairs in 41 sessions
with a median of 47,342 presynaptic spikes.

FOUR TRAPS, and they are most of the work.

1. THE RATE CONFOUND IS THE NULL HERE, not a nuisance. The raw size of a CCG
   trough scales with presynaptic and postsynaptic rate, and both are strongly
   theta-modulated, so "efficacy varies with theta phase" falls out of firing
   rate alone. Efficacy is therefore taken against the pair's own
   hollow-Gaussian predictor computed WITHIN each phase bin, so the baseline
   carries that bin's rate structure, and reported as a ratio to it.

2. SPIKE COUNT. Any peak or trough estimate is count-biased, which is the same
   class of bug as using MRL instead of PPC for phase locking. Presynaptic
   spikes are subsampled to the smallest bin's count, so every bin's estimate
   is made from the same number of spikes.

3. GAMMA PHASE MUST NOT COME FROM THE CELL'S OWN CHANNEL. A fast-spiking
   waveform leaks into the local field at gamma frequencies, so gamma phase is
   read from a channel group at least MIN_SEP groups (~120 um) away. Theta is
   exempt: spike energy is at ~1 kHz and contributes nothing at 6-10 Hz, so
   theta phase is taken from the cell's own group, as in phase_locking.py.

4. THE NULL, WHICH TOOK THREE ATTEMPTS AND IS THE WHOLE DIFFICULTY.
   The postsynaptic cell has gamma rhythmicity of its own, and the predictor
   cannot remove it (see efficacy()). Conditioning on the presynaptic spike's
   gamma phase then rotates that residual with the conditioning phase, which
   mimics phase-dependent synaptic gain exactly.

   (a) A PERMUTATION of the phase labels fails: it destroys the co-oscillation
       along with the synapse, so the artifact stays in the observed value and
       never enters the null. Measured this way the modulation ran 1.07x the
       null in theta, 1.48x in slow gamma and 2.27x in fast gamma -- the exact
       inverse of the predictor's frequency response -- and scaled with the
       postsynaptic cell's own gamma PPC (rho = +0.22 to +0.27).

   (b) THE MIRROR WINDOW fails too, for a subtler reason. Co-oscillation
       between two rhythmic cells is symmetric in lag only when averaged over
       all phases. Conditioning on the presynaptic phase SHIFTS the
       oscillation in lag by an amount that depends on the conditioning phase,
       so the causal window and its reflection sample different points of it
       and the asymmetry is built in. Subtracting the mirror left the PPC
       correlation intact (+0.247 fast) and raised the ratio to 2.83x, which
       is what adding variance without removing bias looks like.

   (c) A ONE-CYCLE SHIFT is the control that separates them, and is what this
       script uses. Each presynaptic spike is moved forward by exactly one
       period of the band in question, found from the unwrapped analytic
       phase, so its phase is unchanged while its 3 ms alignment with the
       target is destroyed. The postsynaptic cell's gamma co-oscillation is
       preserved in the shifted version, because shifting by a whole period
       leaves the pre-post phase relationship alone; the synapse is not.
       Whatever modulation survives the shift is the artifact, and the
       difference is what can only be synaptic. Shifts of one, two and three
       periods give three null samples per pair per band.

Writes data/monosyn/monosyn_phase.csv
"""
import glob
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
from scipy.stats import norm as sp_norm

from lfp_batch import (FS, LFP_ROOT, clip_trials, nap, speed_matched,
                       vr_paths)
from spatial_manifolds.cross_correlograms import _CCH_inner

ROOT = '/Users/harryclark/Documents/spatial-manifolds'
OUT = f'{ROOT}/data/monosyn/monosyn_phase.csv'

BIN_S, MAX_LAG_S = 0.001, 0.025
# monosyn.py uses +-50 ms because its rejection rules examine the
# anticausal bins. Nothing is re-detected here, so the window only has to
# carry the predictor: a 5 ms-sigma hollow Gaussian is fully supported
# within +-25 ms, and mode='same' edge effects cannot reach a causal
# window at +0.7 to +4.7 ms. Halving it halves the cost per CCG.
SIGMA_MS, HOLLOW_FRAC = 5.0, 0.60
CAUSAL = (0.0007, 0.0047)      # English et al., as in monosyn.py
ANTI = (-0.0047, -0.0007)      # the mirror of it; see efficacy()
BANDS = {'slow': (30., 48.), 'fast': (60., 100.)}
NPH = 6                        # phase bins
MIN_SEP = 2                    # groups, ~120 um, for GAMMA phase only
RUN_SPEED = 3.0
MIN_PRE = 3000                 # presynaptic spikes inside the running mask
MIN_PER_BIN = 400              # after count matching
SHIFTS = (1, 2, 3)          # whole periods, for the cycle-shift null


def hollow_gaussian(lags_s):
    lags_ms = lags_s * 1000.0
    c = len(lags_ms) // 2
    half = int(round(sp_norm.ppf((1 + HOLLOW_FRAC) / 2) * SIGMA_MS))
    g = np.exp(-0.5 * (lags_ms / SIGMA_MS) ** 2)
    g[c - half: c + half + 1] = 0.0
    return g / g.sum()


HALF = int(np.floor(MAX_LAG_S / BIN_S))
EDGES = np.linspace(-HALF * BIN_S, HALF * BIN_S, 2 * HALF + 2)
LAGS = .5 * (EDGES[:-1] + EDGES[1:])
GK = hollow_gaussian(LAGS)
CZ = np.where((LAGS >= CAUSAL[0]) & (LAGS <= CAUSAL[1]))[0]
AZ = np.where((LAGS >= ANTI[0]) & (LAGS <= ANTI[1]))[0]


def efficacy(pre, post):
    """Counts in the causal window AND in its mirror image, with predictors.

    THE CAUSAL WINDOW ALONE IS NOT ENOUGH, and the first version of this
    script got it wrong. The hollow-Gaussian predictor is a 5 ms-sigma
    Gaussian, which passes 97% of an 8 Hz oscillation, 45% at 40 Hz and only
    4% at 80 Hz. So it tracks and removes theta co-modulation almost
    perfectly and fast-gamma co-modulation hardly at all, and whatever gamma
    rhythmicity the POSTSYNAPTIC cell has of its own survives in the residual
    at the causal window. Conditioning on the presynaptic spike's gamma phase
    then rotates that residual with the conditioning phase and produces a
    large, sharply phase-locked "efficacy modulation" with no change in
    synaptic gain anywhere. Measured that way the modulation came out 1.07x
    the permutation null in theta, 1.48x in slow gamma and 2.27x in fast
    gamma -- the exact inverse of the predictor's ability to remove the
    oscillation -- and its size scaled with the postsynaptic cell's own
    gamma PPC (rho = +0.22 to +0.27, within session in 22-23 of 27 sessions).
    That is the artifact, not a result.

    A permutation of the phase labels cannot catch it, because permuting
    destroys the co-oscillation along with everything else, so the artifact
    sits in the observed value and not in the null.

    The fix is the mirror window. Co-oscillation between two rhythmic cells is
    symmetric in lag; a monosynaptic effect is causal and one-sided. Taking
    the causal window against its own reflection cancels the symmetric part
    and keeps the part that can only be synaptic.
    """
    if len(pre) < 20:
        return np.nan, np.nan, np.nan, np.nan
    r = _CCH_inner(np.ascontiguousarray(np.sort(pre)),
                   np.ascontiguousarray(post), EDGES, MAX_LAG_S)
    b = np.convolve(r, GK, mode='same')
    return (float(r[CZ].sum()), float(b[CZ].sum()),
            float(r[AZ].sum()), float(b[AZ].sum()))


def eff_ratio(o, p_, oa, pa):
    """Causal ratio minus its mirror: the lag-asymmetric part only."""
    if not (np.isfinite(p_) and np.isfinite(pa)) or p_ <= 0 or pa <= 0:
        return np.nan
    return (o - p_) / p_ - (oa - pa) / pa


def band_phase(x, lo, hi):
    n = len(x)
    nf = next_fast_len(n)
    X = fft(x, nf)
    f = np.fft.fftfreq(nf, 1 / FS)
    Y = np.zeros(nf, complex)
    m = (f >= lo) & (f <= hi)
    Y[m] = 2.0 * X[m]
    a = np.angle(ifft(Y)[:n])
    return a, np.maximum.accumulate(np.unwrap(a))


def cycle_shift(idx, uph, t, k):
    """Move each spike forward by k whole periods, phase preserved.

    The unwrapped analytic phase is a monotone clock for the band, so the
    time at which a spike's phase next recurs is where the unwrapped phase
    reaches its value plus 2*pi*k. A band-filtered signal can dip backwards
    briefly, which np.interp will not accept, so the clock is forced
    monotone first; over a whole period that costs far less than it buys.
    """
    return np.interp(uph[idx] + 2.0 * np.pi * k, uph, t)


def modulation(eff, ph):
    """First-harmonic depth and preferred phase of efficacy against phase."""
    v = np.asarray(eff, float)
    ok = np.isfinite(v)
    if ok.sum() < NPH - 1:
        return np.nan, np.nan
    v = v[ok] - v[ok].mean()
    z = np.sum(v * np.exp(1j * np.asarray(ph)[ok]))
    return float(np.abs(z) / ok.sum()), float(np.angle(z))


def run_session(mo, dy, con, ident, state, rng):
    lf = f'{LFP_ROOT}/M{mo}/D{dy}/VR/sub-M{mo}_ses-D{dy}_typ-VR_beh.nwb'
    ana = f'{LFP_ROOT}/M{mo}/D{dy}/anatomy_M{mo}_D{dy}.csv'
    if not (os.path.exists(lf) and os.path.exists(ana)):
        return []
    fh = h5py.File(lf, 'r')
    if ('processing/ecephys/Processed/theta/data' not in fh
            or 'processing/ecephys/LFP/lfp/data' not in fh):
        fh.close(); return []
    names = [n.decode() for n in
             fh['general/extracellular_ephys/electrodes/channel_name'][:]]
    ch2g = {c: i for i, n in enumerate(names) for c in n.split('_')}
    A = pd.read_csv(ana)
    ent = {c for c, r in zip(A.channel_id, A.brain_region.astype(str))
           if r.startswith('ENTm')}
    eg = sorted({ch2g[c] for c in ent if c in ch2g})
    if len(eg) < 2 * MIN_SEP + 1:
        fh.close(); return []

    bp, cp = vr_paths(mo, dy)
    beh = nap.load_file(bp)
    clusters = nap.load_file(cp)
    trials, _ = clip_trials(beh['trials'].as_dataframe(), clusters)
    ec = clusters.metadata['extremum_channel']
    have = {int(c) for c in clusters.index}

    pairs = con[(con.pre.isin(have)) & (con.post.isin(have))]
    if not len(pairs):
        fh.close(); return []
    # the groups whose phase we will need: each presynaptic cell's own group
    # for theta, and its nearest group beyond the leak for gamma
    def near(g):
        cand = [h for h in eg if abs(h - g) >= MIN_SEP]
        return min(cand, key=lambda h: abs(h - g)) if cand else -1
    pre_g = {int(p): ch2g.get(str(ec[int(p)]), -1) for p in pairs.pre.unique()}
    pre_g = {k: v for k, v in pre_g.items() if v in eg}
    if not pre_g:
        fh.close(); return []
    want_th = sorted(set(pre_g.values()))
    want_gm = sorted({near(g) for g in pre_g.values()} - {-1})

    nT = fh['processing/ecephys/Processed/theta/data'].shape[0]
    TH = {g: fh['processing/ecephys/Processed/theta/data'][:, g].astype(np.float32)
          for g in want_th}
    d = fh['processing/ecephys/LFP/lfp/data']
    blk = (d.chunks[0] if d.chunks else 26041) * 8
    X = np.empty((nT, len(want_gm)), np.float32)
    for i in range(0, nT, blk):
        X[i:i + blk] = d[i:i + blk, :][:, want_gm]
    fh.close()
    gcol = {g: i for i, g in enumerate(want_gm)}
    GM, UN = {}, {}
    for bn, rg in BANDS.items():
        for g in want_gm:
            GM[(bn, g)], UN[(bn, g)] = band_phase(X[:, gcol[g]].astype(float), *rg)
    for g in want_th:
        th = np.asarray(TH[g], float)
        th = (th + np.pi) % (2 * np.pi) - np.pi
        UN[('theta', g)] = np.maximum.accumulate(np.unwrap(np.nan_to_num(th)))

    t = np.arange(nT) / FS
    S = beh['S']
    spd = np.asarray(S.values)[np.searchsorted(np.asarray(S.index), t)
                               .clip(0, len(S) - 1)]
    run = spd >= RUN_SPEED
    inside = np.zeros(nT, bool)
    st = {True: np.zeros(nT, bool), False: np.zeros(nT, bool)}
    info = []
    for _, tr in trials.iterrows():
        m = (t >= tr.start) & (t <= tr.end)
        inside |= m
        u = int(tr.number)
        if u in state and (m & run).sum() >= FS:
            info.append((bool(state[u]), float(spd[m & run].mean()), m))
    if info:
        km = speed_matched(np.array([i[1] for i in info]),
                           np.array([i[0] for i in info]), rng)
        for i in km:
            st[info[i][0]] |= info[i][2]
    keep = run & inside

    edges = np.linspace(-np.pi, np.pi, NPH + 1)
    centres = .5 * (edges[:-1] + edges[1:])
    rows = []
    for _, c in pairs.iterrows():
        pre, post = int(c.pre), int(c.post)
        if pre not in pre_g:
            continue
        sp_pre = np.asarray(clusters[pre].index, dtype=float)
        sp_post = np.asarray(clusters[post].index, dtype=float)
        i_pre = np.round(sp_pre * FS).astype(int)
        i_pre = i_pre[(i_pre >= 0) & (i_pre < nT)]
        i_pre = i_pre[keep[i_pre]]
        i_post = np.round(sp_post * FS).astype(int)
        i_post = i_post[(i_post >= 0) & (i_post < nT)]
        i_post = i_post[keep[i_post]]
        if len(i_pre) < MIN_PRE or len(i_post) < 500:
            continue
        tp = np.ascontiguousarray(i_post / FS)
        base = dict(mouse=mo, day=dy, pre=pre, post=post, kind=c.kind,
                    pre_id=ident.get((mo, dy, pre)),
                    post_id=ident.get((mo, dy, post)),
                    n_pre=len(i_pre), n_post=len(i_post))
        # ---- overall, and by anchoring state: the gate-as-a-gain test -------
        ia = i_pre[st[True][i_pre]]
        inn = i_pre[st[False][i_pre]]
        ks = min(len(ia), len(inn))
        o, p_, oa, pa = efficacy(i_pre / FS, tp)
        rows.append(dict(base, band='-', state='all', bin=-1, phase=np.nan,
                         n_bin=len(i_pre), obs=o, pred=p_, obs_anti=oa,
                         pred_anti=pa))
        if ks >= MIN_PER_BIN:
            for tag, ii in (('anch', rng.choice(ia, ks, replace=False)),
                            ('non', rng.choice(inn, ks, replace=False))):
                o, p_, oa, pa = efficacy(ii / FS, tp)
                rows.append(dict(base, band='-', state=tag, bin=-1,
                                 phase=np.nan, n_bin=ks, obs=o, pred=p_,
                                 obs_anti=oa, pred_anti=pa))
        # ---- phase-resolved, one band at a time -----------------------------
        for bn in ('theta',) + tuple(BANDS):
            ph = (TH[pre_g[pre]] if bn == 'theta'
                  else GM[(bn, near(pre_g[pre]))])
            a = np.asarray(ph)[i_pre]
            # The stored theta phase is on 0..2*pi and np.angle returns
            # -pi..pi. Wrapping both to -pi..pi means the bin edges do not
            # depend on which convention the band came from; without it every
            # theta bin but the top three is empty and the band is skipped.
            a = (a + np.pi) % (2 * np.pi) - np.pi
            good = np.isfinite(a)
            if good.sum() < MIN_PRE:
                continue
            a, ip = a[good], i_pre[good]
            b = np.clip(np.digitize(a, edges) - 1, 0, NPH - 1)
            cnt = np.bincount(b, minlength=NPH)
            k = int(cnt.min())
            if k < MIN_PER_BIN:
                continue
            sub = [rng.choice(ip[b == q], k, replace=False)
                   for q in range(NPH)]
            eff = []
            for q in range(NPH):
                o, p_, oa, pa = efficacy(sub[q] / FS, tp)
                eff.append(eff_ratio(o, p_, oa, pa))
                rows.append(dict(base, band=bn, state='all', bin=q,
                                 phase=centres[q], n_bin=k, obs=o, pred=p_,
                                 obs_anti=oa, pred_anti=pa))
            # the one-cycle-shift null, per trap 4(c)
            dep, pref = modulation(eff, centres)
            uph = UN[('theta', pre_g[pre]) if bn == 'theta'
                     else (bn, near(pre_g[pre]))]
            nd = np.empty(len(SHIFTS))
            for s, kc in enumerate(SHIFTS):
                e2 = []
                for q in range(NPH):
                    ts = cycle_shift(sub[q], uph, t, kc)
                    e2.append(eff_ratio(*efficacy(ts, tp)))
                nd[s] = modulation(e2, centres)[0]
            rows.append(dict(base, band=bn, state='all', bin=-2,
                             phase=pref, n_bin=k, obs=dep,
                             pred=float(np.nanmean(nd)),
                             null_sd=float(np.nanstd(nd)),
                             z=(dep - np.nanmean(nd)) / (np.nanstd(nd) or np.nan),
                             p=float((nd >= dep).mean()),
                             n_null=int(np.isfinite(nd).sum())))
    return rows


if __name__ == '__main__':
    wk, nw = (int(sys.argv[1]), int(sys.argv[2])) if len(sys.argv) > 2 else (0, 1)
    C = pd.concat([pd.read_csv(f) for f in
                   sorted(glob.glob(f'{ROOT}/data/monosyn/connections2_w*.csv'))],
                  ignore_index=True)
    U = pd.read_csv(f'{ROOT}/data/population_state/unit_table.csv')
    ident = {(int(a), int(b), int(c)): d for a, b, c, d in
             U[['mouse', 'day', 'cluster_id', 'identity']].itertuples(index=False)
             if isinstance(d, str)}
    T = pd.read_csv(f'{ROOT}/data/population_state/anchoring_trials.csv')
    # The inhibitory side is the only one with the spikes for a phase-resolved
    # measurement, and a putative interneuron presynaptic cell is the pathway
    # Figure 5 found strongest and the model is about.
    C = C[(C.kind == 'inh')].copy()
    C['pre_id'] = [ident.get((m, d, p)) for m, d, p in zip(C.mouse, C.day, C.pre)]
    C = C[C.pre_id == 'putative interneuron']
    sess = sorted(set(zip(C.mouse.astype(int), C.day.astype(int))))
    sess = [s for i, s in enumerate(sess) if i % nw == wk]
    out, t0 = [], time.time()
    for mo, dy in sess:
        s = T[(T.mouse == mo) & (T.day == dy)]
        state = dict(zip(s.trial.astype(int), s.frac_anch > 0.5))
        con = C[(C.mouse == mo) & (C.day == dy)]
        rng = np.random.default_rng(abs(hash((mo, dy))) % 2 ** 32)
        try:
            r = run_session(mo, dy, con, ident, state, rng)
        except Exception as e:
            print(f'  ! M{mo}D{dy}: {type(e).__name__}: {e}', flush=True)
            continue
        out += r
        if r:
            npair = len({(x['pre'], x['post']) for x in r})
            print(f'  M{mo}D{dy}: {npair} pairs  ({time.time() - t0:.0f}s)',
                  flush=True)
    D = pd.DataFrame(out)
    tag = '' if nw == 1 else f'_w{wk}'
    dest = OUT.replace('.csv', f'{tag}.csv')
    # Write to a temporary name and move it into place only on success. The
    # first version wrote straight to `dest`, so when the LFP volume
    # unmounted mid-run every worker truncated its own good output to a
    # header before failing, and two of three results were unrecoverable.
    if not len(D):
        print(f'\nNOTHING COLLECTED -- {dest} left untouched. Check that '
              f'{LFP_ROOT} is mounted.')
        sys.exit(1)
    tmp = dest + '.partial'
    D.to_csv(tmp, index=False)
    os.replace(tmp, dest)
    print(f'\nwrote {dest}: {len(D)} rows')
