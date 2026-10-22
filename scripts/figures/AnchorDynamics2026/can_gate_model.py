"""A continuous attractor with a gated landmark correction, simulated.

THE QUESTION. Figure 9 proposes that the anchoring state is the gain on a
landmark correction and not a change in the entorhinal circuit. That account
has to produce, from one parameter, four things measured separately: fields
that recur or do not; slow gamma up and fast gamma down with the difference
crossing zero at a fixed frequency; theta-fast coupling weakening while
theta-slow does not; and spike timing that does not move at all. This builds
the circuit and asks whether it does.

WHAT IS IMPOSED AND WHAT IS NOT. Imposed: the architecture, the two inhibitory
delays, and the gate, which scales ONLY the error population's drive onto the
attractor. Not imposed: any gamma amplitude, any coupling strength, any
preferred phase, any crossover frequency. Those are read out of the simulated
field potential afterwards with the estimators used on the recordings.

THE ARCHITECTURE.

    landmark phase --(delay)--> [error ring + fast global I] --(gate)--> [CAN
                                       ^                                  ring
                                       |                                  + slow
                                       +--- subtractive prediction --------+ global I]

Three things took several rebuilds to get right and are worth stating, because
each is a claim about what the circuit has to be like:

  1. THE BUMP LIVES IN A SLOW SYNAPTIC VARIABLE (60 ms, NMDA-like). A gamma
     rhythm strong enough to see in the field potential silences the network
     once a cycle, and a bump held in the firing rate alone has to re-form
     from noise each time -- at a random phase. Held in a slow synapse it
     survives its own rhythm. Without this the attractor dissolves whenever
     the oscillator is strong enough to measure.
  2. THE RHYTHM COMES FROM A GLOBAL INHIBITORY POOL WITH A DELAY, not from
     structured inhibition. Inhibition broad and strong enough to generate
     PING is also broad enough to flatten a Mexican hat; a global pool acts
     only on the uniform mode, so it sharpens the bump instead of erasing it,
     and the delay fixes the frequency at a circuit constant. 7 ms gives 42 Hz
     and 2 ms gives 86 Hz, measured.
  3. PATH INTEGRATION IS AN EXACT (SPECTRAL) TRANSLATION. A finite-difference
     advection diffuses the bump away within a lap, and the gate-shut
     condition then has no attractor left to be wrong about.

The error population receives a bump at the phase the landmark implies and a
subtractive, low-passed copy of the attractor. When the attractor sits where
the landmark says, the two cancel and it falls silent -- so the fast-gamma
generator is driven by MISMATCH, not by whether landmarks are present. That is
why opening the gate, the state in which landmarks are used, should LOWER fast
gamma. The landmark input itself arrives in both states.

RESULTS. Five of the measured results come out, three do not, and one of the
five is a prediction the model made first and the recordings then confirmed.
Read both lists before using this for anything.

REPRODUCED, with a healthy bump in BOTH states (population vector 0.70 and
0.68, so the gate-shut condition is a path-integrating attractor and not a
dead network):

  anchoring      lap-to-lap rate-map correlation +0.99 with the gate open
                 against -0.06 with it shut; mean |mismatch| 10 deg against
                 86, where 86 is the value for a phase drawn uniformly. The
                 phase is RE-DRAWN each lap rather than displaced to a
                 consistent offset -- what the drift test found, and what no
                 displaced-map account gives.
  slow gamma     UP when the gate opens, +0.228 log10 at 42 Hz (data +0.105 z)
  fast gamma     DOWN when the gate opens, -0.295 log10 at 86 Hz (data -0.108)
  theta-fast     coupling FALLS when the gate opens, -58.6% (data -15.9%).
                 Right sign, roughly four times too large.
  THE PREDICTION within the gate-shut state, fast-gamma amplitude scales with
                 the instantaneous |mismatch|. Tested in the recordings by
                 gamma_vs_decode_error.py: partial r = +0.032 over 41 sessions
                 and 133,583 bins, positive in 35 of 41, p = 1.4e-07, and
                 ABSENT in the slow band (19 of 41, p = 0.58). The model
                 generated this before it was looked for.

NOT REPRODUCED:

  theta-slow coupling  the model has it rise 105% when the gate opens; the
                       recordings have it unchanged (-7.9%, p = 0.71). It
                       rises because the slow pool inherits its theta
                       modulation through the bump, so a better-formed bump is
                       more deeply modulated. Giving the pool its own septal
                       theta, which is anatomically real, moves it only from
                       +105% to +84% and was left out.
  theta-speed decoupling  absent (slope 0.049 against 0.048). Theta here is an
                       imposed septal drive the network barely contributes to,
                       so nothing can change its speed gain. Architectural.
  spike-field invariances  cannot be scored: the surrogate spikes are drawn
                       from the population sum, which carries no gamma
                       structure, so PPC is zero in both states. The model is
                       therefore silent on Figures S10 and S11, which are
                       among the strongest constraints on the account.

WHAT MADE THE DIFFERENCE, beyond the three architectural facts below: the fast
band needs an ASYNCHRONOUS BACKGROUND that is not theta-gated. With the error
signal theta-gated and nothing else in the band, the residual when the gate is
open is pure theta-locked error and coupling goes UP. With a background, the
error collapses and the background becomes the larger share of what is left,
so coupling falls while the timing never moves.

NOT YET READY TO REPLACE FIGURE 8: theta-slow coupling goes the wrong way and
the spike-timing invariances cannot be checked at all.

Writes can_gate_model.pdf and prints every measured quantity.
"""
import os

import matplotlib
import numpy as np

matplotlib.use('Agg')
import matplotlib.pyplot as plt
from scipy.signal import butter, filtfilt, hilbert, welch

plt.rcParams['font.family'] = 'Arial'
plt.rcParams['pdf.fonttype'] = 42

FIG = os.path.dirname(os.path.abspath(__file__))
OUT = f'{FIG}/can_gate_model.pdf'

DT = 0.0005                  # s, 2 kHz
N = 128
TRACK, LAM = 200.0, 55.0
NLAP = 20
TAU_E, TAU_S = 0.005, 0.060  # membrane; slow recurrent synapse holding the bump
TAU_EF = 0.002               # error population: a faster membrane
TAU_G = 0.002                # inhibitory pools
D_SLOW, D_FAST = 0.007, 0.0035       # -> 42 Hz and 86 Hz
A_EE, A_G, DRIVE = 4.0, 2.0, 0.50    # attractor: bump PV 0.66, stable
F_EE, F_G, F_DR = 3.0, 5.0, 0.0      # error ring; F_G from the scan
F_BG, F_NOISE = 0.25, 0.10   # asynchronous background on the error ring
S_THETA = 0.0                # septal theta direct onto the slow pool.
#                              Tried at 0.15, 0.35 and 0.60 to flatten
#                              theta-slow coupling and it does not: the
#                              change is +105% at 0 and +84% at 0.60,
#                              still far from the measured null. Left at
#                              zero rather than carried as a knob that
#                              buys nothing.
DELAY_LAND = 0.012
F_THETA, THETA_DEPTH = 8.0, 0.35
# Septal theta need not reach the two populations at the same phase, so the
# error ring's theta gate carries an offset. It was swept over a full cycle,
# 0 to 5 radians, against Figure S10, which measures the two bands only about
# 40 degrees apart in the theta cycle. It does not help: the fast band's
# preferred phase stays within 25 degrees of the theta peak throughout
# (15, 357, 352, 354, 5, 24 degrees) and the slow band's within 10 degrees of
# the trough (197 to 201), so the separation never leaves 150-180 degrees.
# The two pools' envelopes are driven in antiphase by the construction itself
# -- one tracks how concentrated the bump is, the other how large the error is
# -- and no phase offset on the input changes that. Left at zero, and the
# failure is reported rather than hidden.
THETA_PHI_F = 0.0
TAU_PRED = 0.008
# G_OPEN and F_G were set by can_gate_scan.py, which scored the model against
# the measured band changes with the attractor's own results as constraints.
# They turn out to be orthogonal: G_OPEN sets the slow-gamma change and the
# sign of theta's, F_G sets the fast-gamma change, the theta-fast coupling and
# the prediction together, through a threshold between 4 and 5 at which the
# error ring starts to oscillate strongly. There is therefore no setting with
# a realistically small fast-gamma change AND the measured coupling drop: in
# this model they are the same event. 2.0 and 5.0 keep both and leave the
# fast-gamma overshoot as the one remaining spectral failure.
G_OPEN = 2.0
NOISE, RMAX = 0.05, 2.0

th = np.arange(N) * 2 * np.pi / N
KF = np.fft.fftfreq(N) * N
dth = (th[:, None] - th[None, :] + np.pi) % (2 * np.pi) - np.pi


def ring(k, amp):
    W = np.exp(k * (np.cos(dth) - 1.0))
    return amp * W / W.sum(1, keepdims=True)


W_EE, W_FF = ring(2.2, A_EE), ring(2.2, F_EE)
relu = lambda x: np.maximum(x, 0.0)


def bump(phase, k=2.2):
    b = np.exp(k * (np.cos(th - phase) - 1.0))
    return b / b.max()


def shift(v, s):
    return relu(np.real(np.fft.ifft(np.fft.fft(v) * np.exp(-2j * np.pi * KF * s / N))))


def run_blocks(seed=0, nblock=33):
    """One session that goes non-anchored, anchored, non-anchored.

    The recordings do not hand us one switch halfway: the population state
    comes in blocks of tens of trials with transitions in both directions, so
    the simulated session is built the same way. Nothing is reset between
    blocks, so the attractor carries straight through each transition.
    """
    global NLAP
    keep = NLAP
    sched = np.r_[np.zeros(nblock), np.full(nblock, G_OPEN), np.zeros(nblock)]
    try:
        NLAP = 3 * nblock
        return run(G_OPEN, seed, schedule=sched), sched
    finally:
        NLAP = keep


def run(gate, seed=0, switch_lap=None, schedule=None):
    rng = np.random.default_rng(seed)
    nstep = int(NLAP * TRACK / 30.0 / DT)
    x, lap = 0.0, 0
    s = bump(0.0); r_E = s.copy(); g_s = 0.0
    r_F = np.zeros(N); g_f = 0.0; pred = np.zeros(N)
    buf_s = [0.0] * int(D_SLOW / DT)
    buf_f = [0.0] * int(D_FAST / DT)
    land = [0.0] * max(1, int(DELAY_LAND / DT))
    rec = {k: [] for k in ('t', 'x', 'lap', 'spd', 'phi', 'mism',
                           'lfp_s', 'lfp_f', 'lfp_th', 'rE')}
    for i in range(nstep):
        t = i * DT
        spd = 30.0 + 9.0 * np.sin(2 * np.pi * x / TRACK) \
            - 11.0 * np.exp(-((x - 100.0) / 22.0) ** 2) \
            + 2.5 * rng.standard_normal()
        spd = max(spd, 3.0)
        x += spd * DT
        if x >= TRACK:                   # position resets; the bump does not
            x -= TRACK; lap += 1
        land.append(2 * np.pi * (x % LAM) / LAM)
        phi_land = land.pop(0)

        # septal theta, multiplicative on the recurrent drive and scaling with
        # speed; added as a current its trough rectifies the bump away
        a_th = THETA_DEPTH * (0.55 + 0.015 * spd)
        th_t = a_th * np.cos(2 * np.pi * F_THETA * t)
        th_f = a_th * np.cos(2 * np.pi * F_THETA * t + THETA_PHI_F)

        pred = pred + DT / TAU_PRED * (-pred + s)
        # theta GATES the transmission of the error rather than driving the
        # error population on its own. Added as an independent current it is
        # still there when the error is zero, so the residual fast gamma in the
        # anchored state is the theta-locked part and theta-fast coupling goes
        # UP when the gate opens -- the opposite of what is measured.
        # the error signal is theta-gated; the background is NOT. When the
        # gate is open the error collapses and the background becomes the
        # larger share of what is left in this band, so theta-fast coupling
        # falls even though nothing about the timing has moved. Without a
        # background the residual is pure theta-gated error and coupling goes
        # UP when the gate opens -- the opposite of the recordings.
        I_F = ((1.0 + 1.6 * th_f) * (1.6 * bump(phi_land) - 1.1 * pred)
               - F_G * buf_f[0] + F_DR + F_BG
               + F_NOISE * rng.standard_normal(N))
        r_F = r_F + DT / TAU_EF * (-r_F + np.clip(I_F, 0, RMAX))
        g_f = g_f + DT / TAU_G * (-g_f + relu((r_F ** 2).mean()))
        buf_f.append(g_f); buf_f.pop(0)

        corr = r_F - r_F.mean()          # zero-mean: moves the bump, adds no drive
        I_E = ((1.0 + th_t) * (W_EE @ s) - A_G * buf_s[0] + DRIVE
               + gate * corr + NOISE * rng.standard_normal(N))
        r_E = r_E + DT / TAU_E * (-r_E + np.clip(I_E, 0, RMAX))
        s = s + DT / TAU_S * (-s + r_E)
        # the medial septum projects onto interneurons directly, so the
        # slow pool's theta modulation need not be inherited through the
        # bump. Inherited, it scales with how well formed the bump is and
        # theta-slow coupling rises when the gate opens; the recordings
        # have it unchanged.
        g_s = g_s + DT / TAU_G * (-g_s + relu((r_E ** 2).mean()
                                              + S_THETA * th_t))
        buf_s.append(g_s); buf_s.pop(0)

        sh = spd * DT / LAM * N
        s = shift(s, sh); r_E = shift(r_E, sh)

        if schedule is not None:
            gate = float(schedule[min(lap, len(schedule) - 1)])
        elif switch_lap is not None and lap >= switch_lap:
            gate = 0.0
        rec['lfp_s'].append(g_s); rec['lfp_f'].append(g_f); rec['lfp_th'].append(th_t)
        if i % 4 == 0:
            phi = np.angle((s * np.exp(1j * th)).sum()) % (2 * np.pi)
            rec['t'].append(t); rec['x'].append(x); rec['lap'].append(lap)
            rec['spd'].append(spd); rec['phi'].append(phi)
            rec['mism'].append(abs(np.angle(np.exp(1j * (phi_land - phi)))))
            rec['rE'].append(s.copy())
    return {k: np.asarray(v) for k, v in rec.items()}


ANCH_C, NON_C = '#a8559e', '#3f9b8f'
SLOW_C, FAST_C, THETA_C = '#2f7d32', '#e8701a', '#7b4173'


def lfp_of(r, w_th=9.0):
    """One field potential: the two inhibitory pools plus the theta drive."""
    z = lambda v: v - v.mean()
    return z(r['lfp_s']) + z(r['lfp_f']) + w_th * z(r['lfp_th'])


def band(x, lo, hi, fs=1 / DT, order=3):
    b, a = butter(order, [lo / (fs / 2), hi / (fs / 2)], btype='band')
    return filtfilt(b, a, x)


def tort_mi(phase, amp, nb=18):
    e = np.linspace(-np.pi, np.pi, nb + 1)
    m = np.array([amp[(phase >= e[i]) & (phase < e[i + 1])].mean()
                  for i in range(nb)])
    p = m / m.sum()
    return float((np.log(nb) + (p * np.log(p + 1e-12)).sum()) / np.log(nb))


def ppc(ph):
    n = len(ph)
    if n < 50:
        return np.nan
    z = np.exp(1j * ph)
    return float((np.abs(z.sum()) ** 2 - n) / (n * (n - 1)))


def ratemap(r, unit, nb=40):
    L = int(r['lap'].max())
    M = np.full((L, nb), np.nan)
    b = ((r['x'] % LAM) / LAM * nb).astype(int) % nb
    for lp in range(L):
        m = r['lap'] == lp
        for k in range(nb):
            v = r['rE'][m][b[m] == k, unit]
            if len(v):
                M[lp, k] = v.mean()
    return M


def population_state(r, nbin=50):
    """The model's own anchoring raster and PC1, built as Figure 1 builds them.

    The simulated ring is passed through the recordings' own classifier --
    trial_cluster_labels on each unit's trial-by-position map, then the first
    principal component of the cell-by-trial label matrix -- so the model's
    population state is produced by the same code, and the same choices, as
    the measured one.
    """
    import sys
    sys.path.insert(0, '/Users/harryclark/Documents/spatial-manifolds/src')
    from spatial_manifolds.anchoring import trial_cluster_labels
    lap = np.asarray(r['lap'])
    pos = np.asarray(r['x']) % TRACK
    nlap = int(lap.max()) + 1
    b = np.clip((pos / TRACK * nbin).astype(int), 0, nbin - 1)
    rE = r['rE']
    M = np.full((rE.shape[1], nlap, nbin), np.nan)
    for L in range(nlap):
        m = lap == L
        if not m.any():
            continue
        for k in range(nbin):
            mm = m & (b == k)
            if mm.any():
                M[:, L, k] = rE[mm].mean(0)
    lab = np.stack([trial_cluster_labels(M[i])[0] for i in range(len(M))])
    P = lab[~np.isnan(lab).all(1)]
    frac = np.nanmean(P, axis=0)
    Pc = np.nan_to_num(P - np.nanmean(P, axis=1, keepdims=True))
    U, sv, Vt = np.linalg.svd(Pc, full_matrices=False)
    pc1, load = Vt[0], U[:, 0]
    if np.corrcoef(pc1, np.nan_to_num(frac))[0, 1] < 0:
        pc1, load = -pc1, -load
    return dict(labels=lab, pc1=pc1, pc1_load=load, frac_anch=frac, maps=M)


def lap_spectra(r, nper=2048):
    """One spectrum per lap, so the model can be summarised as Figure 8 is.

    Figure 8E plots two things the recordings allow because they have trials:
    a whitened spectrum with the session mean removed, and each frequency
    z-scored across trials. A single Welch estimate per state supports
    neither. The model's laps are its trials, so the field potential is cut
    at lap boundaries and a spectrum taken within each.
    """
    lfp = lfp_of(r)
    lap = np.repeat(np.asarray(r['lap']), 4)[:len(lfp)]
    f, rows = None, []
    for L in np.unique(lap):
        seg = lfp[lap == L]
        if len(seg) < nper:
            continue
        f, P = welch(seg, fs=1 / DT, nperseg=nper)
        rows.append(np.log10(P + 1e-30))
    return f, np.asarray(rows)


def analyse(r, seed=0):
    fs, out = 1 / DT, {}
    lfp = lfp_of(r)
    f, P = welch(lfp, fs=fs, nperseg=4096)
    out['f'], out['P'] = f, P
    for nm, lo, hi in (('theta', 6, 10), ('slow', 30, 48), ('fast', 60, 100)):
        out[nm] = float(np.log10(P[(f >= lo) & (f < hi)]).mean())
    ph_t = np.angle(hilbert(band(lfp, 5, 11)))
    rng = np.random.default_rng(seed)
    step = int(round(fs / 500))
    pop = r['rE'].sum(1)
    spk = rng.random(len(pop)) < np.clip(pop / pop.max() * .4, 0, 1)
    for nm, lo, hi in (('slow', 30, 48), ('fast', 60, 100)):
        a = np.abs(hilbert(band(lfp, lo, hi)))
        out[f'mi_{nm}'] = tort_mi(ph_t, a)
        out[f'amp_{nm}'] = a
        pg = np.angle(hilbert(band(lfp, lo, hi)))[::step][:len(pop)]
        out[f'ppc_{nm}'] = ppc(pg[spk])
        out[f'pref_{nm}'] = float(np.degrees(np.angle(
            np.mean(np.exp(1j * ph_t[::step][:len(pop)][spk])))) % 360)
        # where in theta each band peaks: phase of theta at that band's maxima
        env = a[::step][:len(pop)]
        pk = env > np.quantile(env, .9)
        out[f'thphase_{nm}'] = float(np.degrees(np.angle(
            np.mean(np.exp(1j * ph_t[::step][:len(pop)][pk])))) % 360)
    at = np.abs(hilbert(band(lfp, 5, 11)))[::step][:len(r['spd'])]
    out['theta_speed'] = float(np.polyfit(r['spd'], at, 1)[0])
    return out


def bump_excess(rE, rng, bin_ms=100.0, sigma=2.5, nperm=200):
    """The bump measured exactly as attractor_coherence.py measures it.

    The model's own bump statistic is the resultant of the rate profile over
    the ring, which is not what the recordings can report: there, each cell's
    phase offset is estimated and the population is projected onto it, and the
    answer is the excess over a shuffle of which offset belongs to which cell.
    To compare the two, the same measure is applied here -- rates binned to
    100 ms and smoothed as the recordings are, z-scored per neuron, projected
    onto the ring phase, and referred to the same phase shuffle.
    """
    from scipy.ndimage import gaussian_filter
    k = max(1, int(round(bin_ms / 1e3 / (DT * 4))))        # rE is every 4 dt
    T = (len(rE) // k) * k
    R = rE[:T].reshape(-1, k, rE.shape[1]).mean(1).T       # neurons x bins
    R = gaussian_filter(R, sigma=(0, sigma))
    sd = R.std(1, keepdims=True)
    R = (R - R.mean(1, keepdims=True)) / np.where(sd > 1e-12, sd, 1.0)
    pv = lambda ph: (np.abs(np.exp(1j * ph) @ R)
                     / (np.sqrt(len(ph) / 2.0)
                        * np.linalg.norm(R, axis=0) + 1e-9)).mean()
    nul = np.mean([pv(rng.permutation(th)) for _ in range(nperm)])
    return float(pv(th)), float(nul)


if __name__ == '__main__':
    print('simulating two states ...', flush=True)
    S = {True: run(G_OPEN, 0), False: run(0.0, 0)}
    A = {k: analyse(v) for k, v in S.items()}
    o, s_ = A[True], A[False]

    print('\n=== 1. does the field recur? ================================')
    for anch in (True, False):
        M = ratemap(S[anch], N // 4)
        C = np.corrcoef(np.nan_to_num(M))
        iu = np.triu_indices(len(M), 1)
        pv = np.mean(np.abs((S[anch]['rE'] * np.exp(1j * th)).sum(1))
                     / np.maximum(S[anch]['rE'].sum(1), 1e-9))
        print(f'  gate {"OPEN " if anch else "SHUT "}: lap-to-lap map r '
              f'{C[iu].mean():+.3f}   |mismatch| '
              f'{np.degrees(S[anch]["mism"]).mean():5.1f} deg   bump {pv:.3f}')
        S[anch]['_map'] = M

    print('  -- the same measure the recordings use (phase-shuffle excess):')
    _rng = np.random.default_rng(1)
    BX = {}
    for anch in (True, False):
        ob, nu = bump_excess(S[anch]['rE'], _rng)
        BX[anch] = (ob, nu)
        print(f'     gate {"OPEN " if anch else "SHUT "}: PV {ob:.3f}  '
              f'shuffled {nu:.3f}  excess {ob - nu:+.4f}')
    print(f'     the model keeps '
          f'{100 * (BX[False][0] - BX[False][1]) / (BX[True][0] - BX[True][1]):.0f}'
          f'% of the bump with the gate shut')

    # one continuous session with the gate closing halfway, passed through the
    # recordings' own anchoring classifier, so B and C can be read against
    # Figure 1 rather than against each other
    print('\n=== 1b. the population state, as Figure 1 measures it =======')
    NBLOCK = 33          # ~100 laps, as a recorded session has
    RS, SCHED = run_blocks(nblock=NBLOCK)
    PSW = population_state(RS)
    _n = PSW['labels'].shape[1]
    # the blocks are NBLOCK laps each BY CONSTRUCTION; thirds of the
    # recorded lap count are not the same thing, because the number of
    # laps actually run depends on the speed profile and falls short of
    # NLAP. Taking thirds put the boundary nine laps early and made the
    # state look as though it persisted after the gate shut.
    _edge = [min(NBLOCK, _n), min(2 * NBLOCK, _n)]
    _open = np.zeros(_n, bool); _open[_edge[0]:_edge[1]] = True
    _use = ~np.isnan(PSW['labels']).all(1)
    _bi = int(np.where(_use)[0][int(np.argmax(PSW['pc1_load']))])
    print(f'  {PSW["labels"].shape[0]} units, {_n} laps, gate open on laps '
          f'{_edge[0]}-{_edge[1] - 1}')
    print(f'  anchored fraction  open {np.nanmean(PSW["labels"][:, _open]):.2f}'
          f'  shut {np.nanmean(PSW["labels"][:, ~_open]):.2f}')
    print(f'  PC1 positive on {(PSW["pc1"][_open] > 0).sum()}/{_open.sum()} '
          f'open and {(PSW["pc1"][~_open] > 0).sum()}/{(~_open).sum()} shut laps',
          flush=True)
    SW = dict(sw_map=PSW['maps'][_bi], sw_labels=PSW['labels'],
              sw_pc1=PSW['pc1'], sw_load=PSW['pc1_load'],
              sw_edges=np.array(_edge))

    # the same summary the recordings get: whitened spectrum with the mean
    # removed, and each frequency z-scored across laps and averaged by state
    _fl, _Lo = lap_spectra(S[True])
    _, _Ls = lap_spectra(S[False])
    _All = np.vstack([_Lo, _Ls])
    _ok = (_fl >= 2) & (_fl <= 120)
    _Z = (_All - _All.mean(0)) / (_All.std(0) + 1e-12)
    _w = _All.mean(0) * 0
    _wo = _Lo.mean(0) + np.log10(np.maximum(_fl, 1e-9))
    _ws = _Ls.mean(0) + np.log10(np.maximum(_fl, 1e-9))
    _mu = np.concatenate([_wo[_ok], _ws[_ok]]).mean()
    SP = dict(sp_f=_fl[_ok], rel_open=(_wo - _mu)[_ok],
              rel_shut=(_ws - _mu)[_ok],
              z_open=_Z[:len(_Lo)].mean(0)[_ok],
              z_shut=_Z[len(_Lo):].mean(0)[_ok])
    print(f'  per-lap spectra: {len(_Lo)} open laps, {len(_Ls)} shut')

    print('\n=== 2. the field potential ==================================')
    for nm in ('theta', 'slow', 'fast'):
        print(f'  {nm:<6} log10 power  open {o[nm]:+.3f}  shut {s_[nm]:+.3f}  '
              f'open - shut {o[nm] - s_[nm]:+.3f}')
    f = o['f']
    d = np.log10(o['P']) - np.log10(s_['P'])
    m = (f >= 20) & (f <= 120)
    sgn = np.sign(d[m])
    cr = f[m][np.where(np.diff(sgn) != 0)[0]]
    print('  difference crosses zero at ' +
          (', '.join(f'{c:.0f}' for c in cr) + ' Hz' if len(cr) else 'nowhere')
          + ' (20-120 Hz)')

    print('\n=== 3. coupling and spike timing ============================')
    for nm in ('slow', 'fast'):
        print(f'  theta-{nm:<5} MI  open {o[f"mi_{nm}"]:.5f}  '
              f'shut {s_[f"mi_{nm}"]:.5f}  '
              f'{100 * (o[f"mi_{nm}"] - s_[f"mi_{nm}"]) / s_[f"mi_{nm}"]:+.1f}%')
    for nm in ('slow', 'fast'):
        print(f'  {nm:<5} PPC open {o[f"ppc_{nm}"]:.4f} shut {s_[f"ppc_{nm}"]:.4f}'
              f'   theta phase of band peak: open {o[f"thphase_{nm}"]:.0f}, '
              f'shut {s_[f"thphase_{nm}"]:.0f} deg')

    print('\n=== 4. theta against running speed ==========================')
    print(f'  slope  open {o["theta_speed"]:.4f}   shut {s_["theta_speed"]:.4f}')

    print('\n=== 5. the prediction: fast gamma against mismatch ==========')
    r = S[False]
    step = int(round((1 / DT) / 500))
    af = s_['amp_fast'][::step][:len(r['mism'])]
    rho = float(np.corrcoef(r['mism'], af)[0, 1])
    q = np.quantile(r['mism'], np.linspace(0, 1, 11))
    dec = [af[(r['mism'] >= q[i]) & (r['mism'] < q[i + 1])].mean() for i in range(10)]
    print(f'  within the SHUT state, r(|mismatch|, fast-gamma amplitude) '
          f'= {rho:+.3f}')
    print('  by decile: ' + ' '.join(f'{v:.3f}' for v in dec))
    np.savez_compressed(f'{FIG}/can_gate_model.npz', f=f, d=d,
                        P_open=o['P'], P_shut=s_['P'],
                        map_open=S[True]['_map'], map_shut=S[False]['_map'],
                        mism=r['mism'], ampfast=af, dec=np.array(dec),
                        bump_open=np.array(BX[True]),
                        bump_shut=np.array(BX[False]), **SP, **SW)
    print('\nwrote can_gate_model.npz')


def draw():
    """The figure, from the cached run."""
    z = np.load(f'{FIG}/can_gate_model.npz')
    fig = plt.figure(figsize=(8.6, 4.5))
    gs = fig.add_gridspec(2, 3, hspace=.62, wspace=.38, left=.07, right=.98,
                          top=.90, bottom=.11)

    for j, (key, nm, col) in enumerate(((('map_open'), 'gate OPEN', ANCH_C),
                                        (('map_shut'), 'gate SHUT', NON_C))):
        ax = fig.add_subplot(gs[j, 0])
        M = z[key]
        ax.imshow(M / np.nanmax(M), aspect='auto', cmap='viridis',
                  interpolation='nearest', origin='upper')
        ax.set_xticks([0, M.shape[1] - 1]); ax.set_xticklabels(['0', 'grid period'],
                                                              fontsize=6)
        ax.set_ylabel('lap', fontsize=7.5)
        ax.set_title(nm, fontsize=8, color=col, loc='left')
        ax.tick_params(labelsize=6.5)
        if j == 0:
            ax.text(-.30, 1.12, 'A', transform=ax.transAxes, fontsize=10,
                    weight='bold', va='bottom')

    ax = fig.add_subplot(gs[0, 1])
    ax.semilogy(z['f'], z['P_open'], color=ANCH_C, lw=1.1, label='open')
    ax.semilogy(z['f'], z['P_shut'], color=NON_C, lw=1.1, label='shut')
    ax.set_xlim(2, 120); ax.set_xlabel('frequency (Hz)', fontsize=7.5)
    ax.set_ylabel('power', fontsize=7.5)
    ax.legend(fontsize=6, frameon=False)
    ax.set_title('the two states as spectra', fontsize=8, loc='left')
    ax.tick_params(labelsize=6.5); ax.spines[['top', 'right']].set_visible(False)
    ax.text(-.26, 1.12, 'B', transform=ax.transAxes, fontsize=10, weight='bold',
            va='bottom')

    ax = fig.add_subplot(gs[1, 1])
    ax.axhline(0, color='0.6', lw=.8, ls=':')
    ax.axvspan(30, 48, color='#2f7d32', alpha=.10, lw=0)
    ax.axvspan(60, 100, color='#e8701a', alpha=.10, lw=0)
    ax.plot(z['f'], z['d'], color='0.25', lw=.9)
    ax.set_xlim(2, 120); ax.set_ylim(-.6, .6)
    ax.set_xlabel('frequency (Hz)', fontsize=7.5)
    ax.set_ylabel('open $-$ shut (log10)', fontsize=7.5)
    ax.set_title('slow band up, fast band down', fontsize=8, loc='left')
    ax.tick_params(labelsize=6.5); ax.spines[['top', 'right']].set_visible(False)
    ax.text(-.26, 1.12, 'C', transform=ax.transAxes, fontsize=10, weight='bold',
            va='bottom')

    ax = fig.add_subplot(gs[0, 2])
    f, d = z['f'], z['d']
    vals = [d[(f >= lo) & (f < hi)].mean() for lo, hi in ((6, 10), (30, 48), (60, 100))]
    ax.bar(range(3), vals, .62, color=[THETA_C, SLOW_C, FAST_C], lw=0)
    ax.axhline(0, color='0.3', lw=.9)
    ax.set_xticks(range(3)); ax.set_xticklabels(['theta', 'slow \u03b3', 'fast \u03b3'],
                                                fontsize=7)
    ax.set_ylabel('open $-$ shut (log10)', fontsize=7.5)
    ax.set_title('the model, by band', fontsize=8, loc='left')
    ax.tick_params(labelsize=6.5); ax.spines[['top', 'right']].set_visible(False)
    ax.text(-.30, 1.12, 'D', transform=ax.transAxes, fontsize=10, weight='bold',
            va='bottom')

    ax = fig.add_subplot(gs[1, 2])
    ax.plot(range(1, 11), z['dec'], color=FAST_C, marker='o', ms=3, lw=1.2)
    ax.set_xlabel('decile of |mismatch|, gate shut', fontsize=7.5)
    ax.set_ylabel('fast-\u03b3 amplitude', fontsize=7.5)
    ax.set_title('the prediction to test in the data', fontsize=8, loc='left')
    ax.tick_params(labelsize=6.5); ax.spines[['top', 'right']].set_visible(False)
    ax.text(-.30, 1.12, 'E', transform=ax.transAxes, fontsize=10, weight='bold',
            va='bottom')

    fig.savefig(OUT, dpi=300, bbox_inches='tight')
    print(f'wrote {OUT}')


def bump_excess(rE, rng, bin_ms=100.0, sigma=2.5, nperm=200):
    """The bump measured exactly as attractor_coherence.py measures it.

    The model's own bump statistic is the resultant of the rate profile over
    the ring, which is not what the recordings can report: there, each cell's
    phase offset is estimated and the population is projected onto it, and the
    answer is the excess over a shuffle of which offset belongs to which cell.
    To compare the two, the same measure is applied here -- rates binned to
    100 ms and smoothed as the recordings are, z-scored per neuron, projected
    onto the ring phase, and referred to the same phase shuffle.
    """
    from scipy.ndimage import gaussian_filter
    k = max(1, int(round(bin_ms / 1e3 / (DT * 4))))        # rE is every 4 dt
    T = (len(rE) // k) * k
    R = rE[:T].reshape(-1, k, rE.shape[1]).mean(1).T       # neurons x bins
    R = gaussian_filter(R, sigma=(0, sigma))
    sd = R.std(1, keepdims=True)
    R = (R - R.mean(1, keepdims=True)) / np.where(sd > 1e-12, sd, 1.0)
    pv = lambda ph: (np.abs(np.exp(1j * ph) @ R)
                     / (np.sqrt(len(ph) / 2.0)
                        * np.linalg.norm(R, axis=0) + 1e-9)).mean()
    nul = np.mean([pv(rng.permutation(th)) for _ in range(nperm)])
    return float(pv(th)), float(nul)


if __name__ == '__main__':
    draw()      # defined below main, so it is called once the module is whole
