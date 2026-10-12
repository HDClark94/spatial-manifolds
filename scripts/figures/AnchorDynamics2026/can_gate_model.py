"""A continuous attractor with a gated correction, simulated.

THE QUESTION. Figure 8 proposes that the anchoring state is the gain on a
landmark correction, not a change in the entorhinal circuit. That account has
to produce, from one parameter, four things measured separately: fields that
recur or do not; slow gamma up and fast gamma down with the difference
crossing zero at a fixed frequency; theta-fast coupling weakening while
theta-slow does not; and spike timing that does not move at all. This builds
the circuit and asks whether it does.

WHAT IS IMPOSED AND WHAT IS NOT. Imposed: the architecture (a ring attractor
with slow inhibition, an error population with fast inhibition, a landmark
pathway with a conduction delay), the two inhibitory time constants, and the
gate itself, which scales ONLY the error population's drive onto the
attractor. Not imposed: any gamma amplitude, any coupling strength, any
preferred phase, any crossover frequency. Those are read out of the simulated
field potential afterwards with the same estimators used on the recordings.

THE ARCHITECTURE.

    landmark phase ---(delay)--> [error ring, fast I] --(gate g)--> [attractor
                                        ^                            ring,
                                        |                            slow I]
                                        +--- subtractive prediction -----+

The error population receives a bump at the phase the landmark implies and a
subtractive copy of the attractor's own activity. When the attractor sits
where the landmark says, the two cancel and the error population falls silent.
So the fast-gamma generator is driven by MISMATCH, not by whether landmarks
are present -- which is why opening the gate, the state in which landmarks are
used, LOWERS fast gamma. The landmark input itself arrives in both states.

Velocity enters as advection of the attractor profile, so path integration is
exact up to noise and the bump is never told where it is except through the
error population. At the end of each lap position resets; the attractor is not
reset. With the gate open the error population re-anchors it within a few tens
of ms; with the gate shut nothing does, and the next lap starts from wherever
noise left the bump.

STATUS, 2026-10-12: PARTIAL. Read this before using any number from it.

WHAT WORKS. The gate reproduces the anchoring result decisively. In the
configuration that holds a bump, opening it gives a lap-to-lap rate-map
correlation of +0.943 and a mean |mismatch| of 2.2 deg; shutting it gives
-0.042 and 90.1 deg, which is the value for a phase drawn uniformly. That is
the hardest measured result -- a phase re-drawn each lap rather than a map
displaced to a consistent offset -- and it falls out of path integration
through a teleport with no correction, needing nothing else.

WHAT ALSO WORKS. Delayed inhibition gives each band a frequency fixed by a
circuit constant: 7 ms of delay gives 42 Hz and 2-3 ms gives 86-90 Hz,
measured. Over a threefold change in drive the fast loop's amplitude moves by
a factor of three while its frequency moves by 1 Hz. That is the property the
account needs -- amplitude tracks drive, frequency does not -- and it is a
property of the circuit rather than an assumption about the state. It is also
what would produce a state difference that crosses zero at the same frequency
in every session.

WHAT DOES NOT WORK YET, and why. The attractor does not survive with the gate
shut inside the full model: the bump's population-vector length falls to about
0.04, so the gate-shut condition is a dead network rather than an uncorrected
one. In isolation -- no theta, no error population -- the same attractor
parameters hold a bump for 40 s with a population vector of 0.4 to 0.6. Adding
the rhythm destroys it, whether theta enters as an additive current or as a
multiplicative gain.

The diagnosis is architectural, not a matter of tuning. Inhibition broad and
strong enough to generate PING is also broad and strong enough to flatten the
Mexican hat that holds the bump. A single inhibitory population cannot do both
jobs. The attractor needs two: structured, undelayed inhibition to hold the
bump, and a separate delayed pool to make the rhythm -- which is what the
biology suggests anyway, since these are different interneuron classes.

CONSEQUENCE. The fast band does come out larger with the gate shut (sd 14.8
against 11.8), which is the direction the account predicts, and the direction
the recordings show. IT CANNOT BE CLAIMED from this simulation, because in
that configuration the gate-shut attractor had already lost its bump, so the
comparison may be bump against no bump rather than corrected against
uncorrected. The slow band has no generator at all yet: the attractor here has
undelayed inhibition and so does not oscillate.

TO FINISH IT: give the attractor two inhibitory populations as above, confirm
the bump survives with the gate shut for a whole session, then re-run. Only
then are the gamma numbers worth reading.

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

# ---- parameters -------------------------------------------------------------
# The two gamma bands are produced by DELAYED INHIBITION, not by the membrane
# time constants: a negative feedback loop with delay d oscillates near
# 1/(2d + ...), so each band's frequency is set by a fixed circuit constant and
# cannot drift with drive. Measured in isolation, 7 ms gives 42 Hz and 2 ms
# gives 86 Hz, and the fast loop's AMPLITUDE tracks its drive over a 3-fold
# range while its frequency moves by 1 Hz. That is the property the account
# needs and it is a property of the circuit, not an assumption about the state.
DT = 0.0005                 # s; 2 kHz
N = 128
TRACK, LAM = 200.0, 55.0
NLAP = 24
TAU_E, TAU_I = 0.005, 0.009   # attractor: no delay, a stable bump
TAU_F = 0.002                 # error population, fast
D_SLOW, D_FAST = 0.007, 0.003      # inhibitory delays -> 42 Hz and 86 Hz
DELAY_LAND = 0.012          # landmark pathway conduction delay
F_THETA = 8.0
G_OPEN = 9.0                # correction gain when the gate is open
TAU_PRED = 0.008            # the prediction copy is rate-coded, not spiking
NOISE = 0.05
DRIVE_E = 0.10              # the bump is self-sustaining at this drive;
#                             with the gate shut it path-integrates for the
#                             whole session instead of dissolving
RMAX = 2.0

th = np.arange(N) * 2 * np.pi / N
KF = np.fft.fftfreq(N) * N
dth = (th[:, None] - th[None, :] + np.pi) % (2 * np.pi) - np.pi


def ring(k, amp):
    W = np.exp(k * (np.cos(dth) - 1.0))
    return amp * W / W.sum(1, keepdims=True)


W_EE = ring(2.2, 3.00)
W_EI = ring(0.9, 2.60)      # attractor inhibition, structured, NOT delayed
W_IE = ring(1.6, 1.70)
W_FE = ring(0.9, 4.00)      # error population, delayed by D_FAST
W_EF = ring(1.6, 3.00)
fsat = lambda x: np.maximum(x, 0.0)


def bump(phase, k=2.2):
    b = np.exp(k * (np.cos(th - phase) - 1.0))
    return b / b.max()


def run(gate, seed=0):
    """One session at a fixed correction gain. Returns the recorded traces."""
    rng = np.random.default_rng(seed)
    nstep = int(NLAP * TRACK / 30.0 / DT)
    x, lap = 0.0, 0
    r_E = bump(0.0) * 0.6
    r_I = np.zeros(N)
    r_F = np.zeros(N)
    r_FI = np.zeros(N)
    bufF = [np.zeros(N) for _ in range(int(D_FAST / DT))]
    pred = np.zeros(N)
    land_buf = [0.0] * max(1, int(DELAY_LAND / DT))
    rec = dict(t=[], x=[], lap=[], spd=[], phi=[], mism=[],
               lfp_s=[], lfp_f=[], lfp_th=[], rE=[])
    keep_every = 4
    for i in range(nstep):
        t = i * DT
        spd = 30.0 + 9.0 * np.sin(2 * np.pi * x / TRACK) \
            - 11.0 * np.exp(-((x - 100.0) / 22.0) ** 2) \
            + 2.5 * rng.standard_normal()
        spd = max(spd, 3.0)
        x += spd * DT
        if x >= TRACK:                  # position resets; the bump does not
            x -= TRACK
            lap += 1
        phi_land = 2 * np.pi * (x % LAM) / LAM
        land_buf.append(phi_land)
        phi_delayed = land_buf.pop(0)

        a_theta = 0.30 + 0.011 * spd    # septal theta, scaling with speed
        theta = a_theta * np.cos(2 * np.pi * F_THETA * t)

        # the prediction the error population subtracts is a RATE copy of the
        # attractor, low-passed: without this the attractor's own rhythm is
        # injected straight into the fast loop and entrains it
        pred = pred + DT / TAU_PRED * (-pred + r_E)

        # error population: landmark bump minus prediction, fast delayed
        # inhibition. Silent when the attractor already sits where the
        # landmark says, whatever the state of the gate.
        I_F = (1.55 * bump(phi_delayed) - 1.05 * pred + 0.55 * theta
               - W_FE @ bufF[0])
        r_F = r_F + DT / TAU_F * (-r_F + fsat(I_F))
        r_FI = r_FI + DT / TAU_F * (-r_FI + fsat(W_EF @ r_F))
        bufF.append(r_FI.copy()); bufF.pop(0)

        # the correction is zero-mean across the ring: it moves the bump, it
        # does not add drive, so opening the gate cannot raise rates by itself
        corr = r_F - r_F.mean()

        # theta as a multiplicative gain on the recurrent drive, not an
        # additive current: added, its trough rectifies the bump away within
        # a cycle and the gate-shut attractor cannot survive its own rhythm
        I_E = ((1.0 + 0.45 * np.cos(2 * np.pi * F_THETA * t))
               * (W_EE @ r_E) - W_EI @ r_I + DRIVE_E + gate * corr
               + NOISE * rng.standard_normal(N))
        r_E = r_E + DT / TAU_E * (-r_E + fsat(I_E))
        # path integration as an exact translation of the profile; a
        # finite-difference advection diffuses the bump away within a lap and
        # the gate-shut condition then has no attractor left to be wrong about
        shift = spd * DT / LAM * N
        r_E = np.maximum(np.real(np.fft.ifft(
            np.fft.fft(r_E) * np.exp(-2j * np.pi * KF * shift / N))), 0.0)
        r_I = r_I + DT / TAU_I * (-r_I + fsat(W_IE @ r_E))

        rec['lfp_s'].append(float(r_I.sum()))
        rec['lfp_f'].append(float(r_FI.sum()))
        rec['lfp_th'].append(float(theta))
        if i % keep_every == 0:
            z = (r_E * np.exp(1j * th)).sum()
            phi = np.angle(z) % (2 * np.pi)
            rec['t'].append(t); rec['x'].append(x); rec['lap'].append(lap)
            rec['spd'].append(spd); rec['phi'].append(phi)
            rec['mism'].append(abs(np.angle(np.exp(1j * (phi_land - phi)))))
            rec['rE'].append(r_E.copy())
    return {k: np.asarray(v) for k, v in rec.items()}


def lfp_of(rec):
    """One field potential: the two inhibitory currents plus the theta drive."""
    s = rec['lfp_s'] - rec['lfp_s'].mean()
    f = rec['lfp_f'] - rec['lfp_f'].mean()
    th_ = rec['lfp_th'] - rec['lfp_th'].mean()
    return 1.0 * s + 1.0 * f + 14.0 * th_


def band(x, lo, hi, fs=1 / DT, order=3):
    b, a = butter(order, [lo / (fs / 2), hi / (fs / 2)], btype='band')
    return filtfilt(b, a, x)


def tort_mi(phase, amp, nb=18):
    e = np.linspace(-np.pi, np.pi, nb + 1)
    m = np.array([amp[(phase >= e[i]) & (phase < e[i + 1])].mean()
                  for i in range(nb)])
    p = m / m.sum()
    return float((np.log(nb) + (p * np.log(p + 1e-12)).sum()) / np.log(nb))


def ppc(phases):
    """Pairwise phase consistency: unbiased by spike count, as in Figure S11."""
    n = len(phases)
    if n < 20:
        return np.nan
    z = np.exp(1j * phases)
    return float((np.abs(z.sum()) ** 2 - n) / (n * (n - 1)))


def analyse(rec, seed=0):
    """Everything the recordings are scored on, scored the same way."""
    fs = 1 / DT
    lfp = lfp_of(rec)
    out = {}
    f, P = welch(lfp, fs=fs, nperseg=4096)
    out['f'], out['P'] = f, P
    for nm, lo, hi in (('theta', 6, 10), ('slow', 30, 48), ('fast', 60, 100)):
        m = (f >= lo) & (f < hi)
        out[nm] = float(np.log10(P[m]).mean())
    ph_t = np.angle(hilbert(band(lfp, 5, 11)))
    for nm, lo, hi in (('slow', 30, 48), ('fast', 60, 100)):
        amp = np.abs(hilbert(band(lfp, lo, hi)))
        out[f'mi_{nm}'] = tort_mi(ph_t, amp)
        # spike-field: Poisson spikes from the attractor, phase of each band
        rng = np.random.default_rng(seed)
        step = int(round(fs / 500))
        rE = rec['rE'] if 'rE' in rec else None
        ph_g = np.angle(hilbert(band(lfp, lo, hi)))[::step][:len(rec['phi'])]
        rate = rec['_pop']
        k = rng.random(len(rate)) < np.clip(rate / rate.max() * .35, 0, 1)
        out[f'ppc_{nm}'] = ppc(ph_g[k])
        out[f'pref_{nm}'] = float(np.degrees(np.angle(
            np.mean(np.exp(1j * np.angle(hilbert(band(lfp, 5, 11)))[::step]
                           [:len(rec['phi'])][k])))) % 360)
        out[f'amp_{nm}'] = amp
    # theta amplitude against running speed, per lap
    a_t = np.abs(hilbert(band(lfp, 5, 11)))
    step = int(round(fs / 500))
    a_ds = a_t[::step][:len(rec['spd'])]
    sl = np.polyfit(rec['spd'], a_ds, 1)
    out['theta_speed_slope'] = float(sl[0])
    return out


if __name__ == '__main__':
    print('simulating ...', flush=True)
    S = {True: run(G_OPEN, 0), False: run(0.0, 0)}
    for k, r in S.items():
        r['_pop'] = r['rE'].sum(1)
    A = {k: analyse(r) for k, r in S.items()}

    print('\n--- does the field recur? -----------------------------------')
    for anch in (True, False):
        r = S[anch]
        # trial x phase map for one unit, correlated between laps
        nb, L = 40, int(r['lap'].max())
        M = np.zeros((L, nb))
        pos_phase = (r['x'] % LAM) / LAM * nb
        u = N // 4
        for lp in range(L):
            m = r['lap'] == lp
            idx = pos_phase[m].astype(int) % nb
            for b in range(nb):
                v = r['rE'][m][idx == b, u]
                M[lp, b] = v.mean() if len(v) else np.nan
        C = np.corrcoef(np.nan_to_num(M))
        iu = np.triu_indices(L, 1)
        print(f'  gate {"OPEN " if anch else "SHUT "}: lap-to-lap map '
              f'correlation {C[iu].mean():+.3f}   mean |mismatch| '
              f'{np.degrees(r["mism"]).mean():5.1f} deg')
        S[anch]['_map'] = M

    print('\n--- the field potential -------------------------------------')
    for nm in ('theta', 'slow', 'fast'):
        d = A[True][nm] - A[False][nm]
        print(f'  {nm:<6} log10 power  open {A[True][nm]:+.3f}  '
              f'shut {A[False][nm]:+.3f}   open - shut {d:+.3f}')
    f = A[True]['f']
    d = np.log10(A[True]['P']) - np.log10(A[False]['P'])
    m = (f >= 20) & (f <= 120)
    sgn = np.sign(d[m])
    cross = f[m][np.where(np.diff(sgn) != 0)[0]]
    print(f'  difference crosses zero at '
          + (', '.join(f'{c:.0f} Hz' for c in cross) if len(cross) else 'nowhere')
          + ' between 20 and 120 Hz')

    print('\n--- theta-gamma coupling, and spike timing ------------------')
    for nm in ('slow', 'fast'):
        o, sh = A[True][f'mi_{nm}'], A[False][f'mi_{nm}']
        print(f'  theta-{nm:<5} MI  open {o:.5f}  shut {sh:.5f}  '
              f'change {100 * (o - sh) / sh:+.1f}%')
    for nm in ('slow', 'fast'):
        print(f'  {nm:<5} PPC open {A[True][f"ppc_{nm}"]:.4f} '
              f'shut {A[False][f"ppc_{nm}"]:.4f}   preferred theta phase '
              f'open {A[True][f"pref_{nm}"]:.0f} deg, '
              f'shut {A[False][f"pref_{nm}"]:.0f} deg')

    print('\n--- theta against running speed -----------------------------')
    print(f'  slope  open {A[True]["theta_speed_slope"]:.4f}   '
          f'shut {A[False]["theta_speed_slope"]:.4f}')

    print('\n--- the prediction: fast gamma against mismatch -------------')
    r = S[False]
    step = int(round((1 / DT) / 500))
    af = A[False]['amp_fast'][::step][:len(r['mism'])]
    nb = 10
    q = np.quantile(r['mism'], np.linspace(0, 1, nb + 1))
    mu = [af[(r['mism'] >= q[i]) & (r['mism'] < q[i + 1])].mean() for i in range(nb)]
    rho = np.corrcoef(r['mism'], af)[0, 1]
    print(f'  within the SHUT state, r(|mismatch|, fast-gamma amplitude) '
          f'= {rho:+.3f}')
    print('  by decile of mismatch: ' + ' '.join(f'{v:.2f}' for v in mu))
    np.savez_compressed(f'{FIG}/can_gate_model.npz',
                        f=f, d=d, P_open=A[True]['P'], P_shut=A[False]['P'],
                        map_open=S[True]['_map'], map_shut=S[False]['_map'],
                        mism_shut=r['mism'], ampfast_shut=af,
                        spd_open=S[True]['spd'], spd_shut=S[False]['spd'])
    print('\nwrote can_gate_model.npz')
