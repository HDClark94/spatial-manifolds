"""The gated attractor again, but as a two-dimensional grid network.

WHY. The model in can_gate_model.py is a ring: one module's phase along the
track. That is enough for the spectra, but it assumes the thing a grid model
should produce. This builds the same circuit on a two-dimensional sheet whose
units are genuine grid cells -- hexagonal firing in an open arena -- and then
drives it along a straight line through that arena, teleporting back to the
start at the end of each lap, which is what a virtual linear track is.

WHY THE TELEPORT MATTERS. At the teleport the animal's position jumps back to
the start; its path integrator does not. The bump therefore arrives at the
start of lap n+1 carrying a whole lap of accumulated translation, and
something has to put it back. That something is the landmark correction, and
the gate decides whether it is applied. So on this track the gate is not an
abstraction: with it open the phase is reset every lap and fields recur in the
same places, with it shut the phase accumulates and the fields walk.

THE SHEET. Units tile the torus of two grid phases (theta1, theta2). Physical
displacement enters as d(theta_a) = k_a . dx with k1 and k2 sixty degrees
apart and |k| = 4 pi / (sqrt 3 lambda), which is what makes the spatial
pattern hexagonal rather than square. The recurrent kernel is symmetric in all
three lattice axes, cos a + cos b + cos(a+b), so the bump is hexagonal on the
sheet too. Everything else -- slow synaptic variable holding the bump, delayed
global inhibition setting each gamma band, theta-gated error sheet, zero-mean
gated correction -- is carried over unchanged from the ring.

Writes can_gate_model_2d.npz
"""
import os

import numpy as np
from scipy.signal import hilbert, welch

FIG = os.path.dirname(os.path.abspath(__file__))
# NOISE_SCALE multiplies both noise terms, so the same circuit can be run as a
# noisier version of itself and the two compared; the output is tagged to match
_NSC = float(os.environ.get('NOISE_SCALE', 1.0))
OUT = (f'{FIG}/can_gate_model_2d.npz' if _NSC == 1.0 else
       f'{FIG}/can_gate_model_2d_noisy.npz')

DT = 0.0005
NS = 36                       # sheet is NS x NS units
TRACK, LAM = 200.0, 42.0      # 1.2x the ring's 55 cm: a longer period
#                               means a larger slip per lap relative to the
#                               lattice, so laps drift further out of register
#                               and coincide with the anchored phase less often
NLAP = 20
TRACK_ANGLE = np.radians(12.0)        # the track's heading in the arena.
#   This, not the spacing, sets how many fields a unit has per lap: a track
#   that grazes a lattice row gives one long field however small the period
#   is. At 21 degrees only 27% of units had two or more fields per 200 cm;
#   at 32 degrees 97% do, with a mean of 2.4, at the same 66 cm spacing.
TAU_E, TAU_S, TAU_EF, TAU_G = 0.005, 0.060, 0.002, 0.002
D_SLOW, D_FAST = 0.007, 0.0035
A_EE, A_G, DRIVE = 4.0, 2.0, 0.50
F_EE, F_G, F_DR = 3.0, 5.0, 0.0
F_BG, F_NOISE = 0.25, 0.10 * _NSC
DELAY_LAND = 0.012
F_THETA, THETA_DEPTH = 8.0, 0.35
TAU_PRED = 0.008
# Scanned over G_OPEN, F_G and F_BG on the sheet, as the ring was scanned.
# F_G barely matters here (5 to 14 moves fast gamma from -0.005 to -0.007 and
# the coupling not at all): the error population's pool averages over 1296
# units rather than 128 and never reaches the strongly oscillating regime the
# ring has. G_OPEN=2 locks the phase best once the lattice is correct, mean
# mismatch 53 degrees, and puts fast gamma at -0.019 against a measured
# -0.016. Slow gamma then comes out with the wrong sign; that is reported
# rather than tuned away.
# The sheet needs a far stronger gate than the ring (150 against 2) to hold
# the phase: the correction is one zero-mean drive spread over 1296 units
# rather than 128, so per unit it is much weaker. Scanned 5 to 300; the open
# state's mean mismatch falls 54 -> 8 degrees between 5 and 70 and saturates
# by 150, where lap-to-lap map correlation in the open block is 0.89. Below
# that the open block does not repeat exactly from lap to lap and the fields
# recur on a several-lap cycle instead.
G_OPEN = 5.0                  # raised with the period: at LAM=66 the same
#                               gain left the phase unlocked, mean mismatch
#                               85 deg against 35 at LAM=55
NOISE, RMAX = 0.05 * _NSC, 2.0
# Path integration here is an exact Fourier shift, so with the gate shut the
# per-lap phase step is identical every lap and a unit's firing recurs on a
# fixed cycle -- closed blobs in the trial-by-position map rather than the
# scattered fields the recordings show. Real path integration accumulates
# noise, so the step varies from lap to lap. PI_DIFF adds that as a random
# walk on the sheet alongside the velocity shift. It was tried at 0.002 to
# 0.017 rad per step and makes the classification WORSE, not better: a small
# random walk keeps consecutive laps near each other, so the shut block gives
# MORE false positives (0.54-0.71 against 0.35 with none), while a large one
# outruns the correction and the open block stops being anchored (0.17).
# The deterministic step is what decorrelates laps. Left at zero.
PI_DIFF = 0.0 * _NSC
# How the landmark correction reaches the attractor. On the ring a zero-mean
# additive drive moves the bump cleanly, because it is concentrated on 128
# units. Spread over 1296 it cannot, and scanning its gain only trades one
# failure for the other:
#
#   G_OPEN      5     10     20     40     70
#   mismatch   54     43     40     34      8   degrees, gate open
#   bump PV  0.103  0.148  0.075  0.063  0.047  against 0.27 with it shut
#
# the phase locks only once the bump has been hollowed out. So on the sheet
# the error translates the bump instead, which is what a correction to a path
# integrator does: the phase error is driven towards zero with time constant
# 1 / G_SHIFT while the gate is open. That costs the bump nothing, because
# moving it does not deform it --
#
#   G_SHIFT     1      2      4      8     16
#   mismatch   22     11      6      3      2   degrees
#   lap r   0.926  0.964  0.979  0.989  0.994  between successive lap maps
#   bump PV 0.430  0.430  0.430  0.430  0.430  identical open and shut
#
# and 4 /s is taken, as a clearly anchored state rather than a perfect clamp.
#
# The quantity corrected is the mismatch the error population below already
# encodes -- it receives the landmark bump and subtracts a copy of the
# attractor -- but it is read out directly here rather than decoded from that
# population's rates, so the correction is noiseless in a way the circuit's
# own estimate would not be. It is stated plainly because it matters for what
# the open state can be claimed to show: that a gated correction holds the
# phase, not how accurately a population of this size could estimate it.
G_SHIFT = 4.0                 # per second, when the gate is open
# The error population's current onto the attractor, which is what the ring's
# gate acts on and is kept here at the same strength, so that the two
# geometries are the same circuit. With the shift above in place it is no
# longer harmful: it paints a bump at the LANDMARK phase, and that is why
# turning it up alone was so destructive -- at a 54 degree mismatch it put a
# second bump on the sheet beside the attractor's own, which is what the
# population vectors above are reporting and why a gain of 150 left a grid
# score of 0.29. Once the phase is locked the two coincide. It is also
# self-limiting: the error population reports the landmark minus a copy of the
# attractor, so a working correction drives it quiet (pool mean 0.017 open
# against 0.047 shut), and the current it delivers when anchored is small.
# That is the model's account of the fast-gamma fall.
G_CURR = 5.0
# A landmark drive onto the attractor was tried instead, to give the gate
# something that does not go quiet when the correction succeeds, and was
# dropped: taken zero-mean it is clipped negative over most of the sheet and
# so LOWERS activity rather than raising it, which is the same reason the
# error current lowers it (mean rate 0.699 open against 0.791 shut).
G_LAND = 0.0
KK = 1.6                      # width of the sheet kernel

# The two reciprocal lattice vectors are 120 degrees apart, not 60. The
# kernel below is symmetric in three axes, the third being k3 = -(k1 + k2),
# and only at 120 degrees does that third vector have the same length as the
# other two: at 60 it is sqrt(3) times too long, which is not a hexagonal
# lattice at all and gives fields that are elongated streaks rather than round
# blobs.
_KM = 4 * np.pi / (np.sqrt(3) * LAM)
K1 = _KM * np.array([1.0, 0.0])
K2 = _KM * np.array([np.cos(2 * np.pi / 3), np.sin(2 * np.pi / 3)])
KMAT = np.vstack([K1, K2])            # x (cm) -> (theta1, theta2)

_i = np.arange(NS)
TH1, TH2 = np.meshgrid(2 * np.pi * _i / NS, 2 * np.pi * _i / NS, indexing='ij')
FX, FY = np.meshgrid(np.fft.fftfreq(NS) * NS, np.fft.fftfreq(NS) * NS,
                     indexing='ij')


def hexkern(a, b, k=KK):
    """Symmetric in all three lattice axes, so the bump is hexagonal.

    The third axis is theta3 = -(theta1 + theta2); with k1 and k2 120 degrees
    apart all three have equal length and this is the standard hexagonal form.
    """
    return np.exp(k * (np.cos(a) + np.cos(b) + np.cos(a + b) - 3.0))


_W = hexkern(TH1, TH2)
_W = _W / _W.sum()
WF_E = np.fft.fft2(_W) * A_EE
WF_F = np.fft.fft2(_W) * F_EE
relu = lambda x: np.maximum(x, 0.0)


def bump_at(p1, p2):
    b = hexkern(TH1 - p1, TH2 - p2)
    return b / b.max()


def conv(v, WF):
    return np.real(np.fft.ifft2(np.fft.fft2(v) * WF))


def shift2(v, d1, d2):
    """Translate on the torus by (d1, d2) radians, exactly, in Fourier space."""
    ph = np.exp(-2j * np.pi * (FX * d1 / (2 * np.pi) + FY * d2 / (2 * np.pi)))
    return relu(np.real(np.fft.ifft2(np.fft.fft2(v) * ph)))


# Several modules, as a recording samples. Spacings in the ratio the
# literature reports, about 1.4 between neighbours, and orientations that
# differ, so that a phase coincidence in one module is not a coincidence in
# the others. Every module sees the same position and the same gate.
MODULES = [(40.0, np.radians(0.0)), (56.0, np.radians(14.0)),
           (78.0, np.radians(-11.0))]
NS_M = 24                      # units per side in a multi-module run


def module_setup(lam, ori, ns=NS_M, k=KK):
    """Everything one module needs: its lattice, kernels and sheet grids."""
    km = 4 * np.pi / (np.sqrt(3) * lam)
    rot = lambda a: np.array([np.cos(a), np.sin(a)])
    kmat = np.vstack([km * rot(ori), km * rot(ori + 2 * np.pi / 3)])
    i = np.arange(ns)
    t1, t2 = np.meshgrid(2 * np.pi * i / ns, 2 * np.pi * i / ns, indexing='ij')
    fx, fy = np.meshgrid(np.fft.fftfreq(ns) * ns, np.fft.fftfreq(ns) * ns,
                         indexing='ij')
    w = np.exp(k * (np.cos(t1) + np.cos(t2) + np.cos(t1 + t2) - 3.0))
    w = w / w.sum()
    return dict(lam=lam, ori=ori, kmat=kmat, ns=ns, th1=t1, th2=t2,
                fx=fx, fy=fy, wf_e=np.fft.fft2(w) * A_EE,
                wf_f=np.fft.fft2(w) * F_EE)


def m_bump(M, p1, p2):
    b = np.exp(KK * (np.cos(M['th1'] - p1) + np.cos(M['th2'] - p2)
                     + np.cos(M['th1'] - p1 + M['th2'] - p2) - 3.0))
    return b / b.max()


def m_conv(M, v, wf):
    return np.real(np.fft.ifft2(np.fft.fft2(v) * wf))


def m_shift(M, v, d1, d2):
    ph = np.exp(-2j * np.pi * (M['fx'] * d1 + M['fy'] * d2) / (2 * np.pi))
    return relu(np.real(np.fft.ifft2(np.fft.fft2(v) * ph)))


def run_multi(gate, seed=0, schedule=None, nlap=None, mods=None):
    """The same session, with several modules running side by side.

    The gate is one global signal and every module sees the same position, so
    the modules differ only in their lattice. That is the point: a lap on
    which a drifting phase happens to land where it sat when anchored is a
    coincidence in one module and not in the others, so the coincidences stop
    being shared and the population average can clear them.
    """
    rng = np.random.default_rng(seed)
    nlap = nlap or NLAP
    MM = [module_setup(*m) for m in (mods or MODULES)]
    nstep = int(nlap * TRACK / 30.0 / DT)
    heading = np.array([np.cos(TRACK_ANGLE), np.sin(TRACK_ANGLE)])
    x2 = np.array([12.0, 9.0])
    x, lap = 0.0, 0
    for M in MM:
        M['s'] = m_bump(M, 0, 0); M['rE'] = M['s'].copy(); M['gs'] = 0.0
        M['rF'] = np.zeros((M['ns'], M['ns'])); M['gf'] = 0.0
        M['pred'] = np.zeros((M['ns'], M['ns']))
        M['bs'] = [0.0] * int(D_SLOW / DT)
        M['bf'] = [0.0] * int(D_FAST / DT)
    land = [x2.copy()] * max(1, int(DELAY_LAND / DT))
    rec = {k: [] for k in ('t', 'x', 'lap', 'spd', 'mism',
                           'lfp_s', 'lfp_f', 'lfp_th', 'rE')}
    for i in range(nstep):
        t = i * DT
        spd = 30.0 + 9.0 * np.sin(2 * np.pi * x / TRACK) \
            - 11.0 * np.exp(-((x - 100.0) / 22.0) ** 2) \
            + 2.5 * rng.standard_normal()
        spd = max(spd, 3.0)
        x += spd * DT
        step = spd * DT * heading
        x2 = x2 + step
        if x >= TRACK:
            x -= TRACK; lap += 1
            x2 = x2 - TRACK * heading
        land.append(x2.copy()); xl = land.pop(0)
        if schedule is not None:
            gate = float(schedule[min(lap, len(schedule) - 1)])
        a_th = THETA_DEPTH * (0.55 + 0.015 * spd)
        th_t = a_th * np.cos(2 * np.pi * F_THETA * t)
        gs_tot = gf_tot = 0.0
        mism = []
        for M in MM:
            pl1, pl2 = M['kmat'] @ xl
            M['pred'] += DT / TAU_PRED * (-M['pred'] + M['s'])
            I_F = ((1.0 + 1.6 * th_t) * (1.6 * m_bump(M, pl1, pl2)
                                         - 1.1 * M['pred'])
                   - F_G * M['bf'][0] + F_DR + F_BG
                   + F_NOISE * rng.standard_normal((M['ns'], M['ns'])))
            M['rF'] += DT / TAU_EF * (-M['rF'] + np.clip(I_F, 0, RMAX))
            M['gf'] += DT / TAU_G * (-M['gf'] + relu((M['rF'] ** 2).mean()))
            M['bf'].append(M['gf']); M['bf'].pop(0)
            _lb = m_bump(M, pl1, pl2)
            I_E = ((1.0 + th_t) * m_conv(M, M['s'], M['wf_e'])
                   - A_G * M['bs'][0] + DRIVE
                   + (gate / G_OPEN) * (G_CURR * (M['rF'] - M['rF'].mean())
                                        + G_LAND * (_lb - _lb.mean()))
                   + NOISE * rng.standard_normal((M['ns'], M['ns'])))
            M['rE'] += DT / TAU_E * (-M['rE'] + np.clip(I_E, 0, RMAX))
            M['s'] += DT / TAU_S * (-M['s'] + M['rE'])
            M['gs'] += DT / TAU_G * (-M['gs'] + relu((M['rE'] ** 2).mean()))
            M['bs'].append(M['gs']); M['bs'].pop(0)
            d1, d2 = M['kmat'] @ step
            if gate > 0:                       # the landmark correction
                z1 = np.angle((M['s'] * np.exp(1j * M['th1'])).sum())
                z2 = np.angle((M['s'] * np.exp(1j * M['th2'])).sum())
                d1 += (gate / G_OPEN) * G_SHIFT * DT * (
                    (pl1 - z1 + np.pi) % (2 * np.pi) - np.pi)
                d2 += (gate / G_OPEN) * G_SHIFT * DT * (
                    (pl2 - z2 + np.pi) % (2 * np.pi) - np.pi)
            M['s'] = m_shift(M, M['s'], d1, d2)
            M['rE'] = m_shift(M, M['rE'], d1, d2)
            gs_tot += M['gs']; gf_tot += M['gf']
            if i % 4 == 0:
                z1 = (M['s'] * np.exp(1j * M['th1'])).sum()
                z2 = (M['s'] * np.exp(1j * M['th2'])).sum()
                dd = (np.array([np.angle(z1), np.angle(z2)])
                      - np.array([pl1, pl2]) + np.pi) % (2 * np.pi) - np.pi
                mism.append(np.hypot(*dd))
        rec['lfp_s'].append(gs_tot); rec['lfp_f'].append(gf_tot)
        rec['lfp_th'].append(th_t)
        if i % 4 == 0:
            rec['t'].append(t); rec['x'].append(x); rec['lap'].append(lap)
            rec['spd'].append(spd); rec['mism'].append(float(np.mean(mism)))
            rec['rE'].append(np.concatenate([M['rE'].ravel() for M in MM]))
    out = {k: np.asarray(v) for k, v in rec.items()}
    out['module'] = np.concatenate([np.full(M['ns'] ** 2, j)
                                    for j, M in enumerate(MM)])
    return out


def phase_of(x):
    """The grid phases of a position in the arena."""
    return KMAT @ np.asarray(x, float)


def run(gate, seed=0, switch_lap=None, nlap=None, walk=False,
        dur=None, schedule=None):
    """One session. walk=True foraged in 2-D instead of running the track."""
    rng = np.random.default_rng(seed)
    nlap = nlap or NLAP
    nstep = int((dur if dur else nlap * TRACK / 30.0) / DT)
    heading = np.array([np.cos(TRACK_ANGLE), np.sin(TRACK_ANGLE)])
    x2 = np.array([12.0, 9.0])            # where the track starts in the arena
    x, lap, hd = 0.0, 0, rng.uniform(0, 2 * np.pi)
    s = bump_at(0, 0); r_E = s.copy(); g_s = 0.0
    r_F = np.zeros((NS, NS)); g_f = 0.0; pred = np.zeros((NS, NS))
    buf_s = [0.0] * int(D_SLOW / DT)
    buf_f = [0.0] * int(D_FAST / DT)
    land = [(0.0, 0.0)] * max(1, int(DELAY_LAND / DT))
    rec = {k: [] for k in ('t', 'x', 'y', 'pos', 'lap', 'spd', 'mism',
                           'lfp_s', 'lfp_f', 'lfp_th', 'rE')}
    for i in range(nstep):
        t = i * DT
        if walk:
            hd += 0.9 * np.sqrt(DT) * rng.standard_normal()
            spd = 22.0 + 4.0 * rng.standard_normal()
            spd = float(np.clip(spd, 3.0, 45.0))
            step = spd * DT * np.array([np.cos(hd), np.sin(hd)])
            nxt = x2 + step
            for c in (0, 1):              # reflect at the arena wall
                if not (0.0 <= nxt[c] <= 150.0):
                    hd = np.pi - hd if c == 0 else -hd
                    step = spd * DT * np.array([np.cos(hd), np.sin(hd)])
                    nxt = x2 + step
            x2 = nxt
        else:
            spd = 30.0 + 9.0 * np.sin(2 * np.pi * x / TRACK) \
                - 11.0 * np.exp(-((x - 100.0) / 22.0) ** 2) \
                + 2.5 * rng.standard_normal()
            spd = max(spd, 3.0)
            x += spd * DT
            step = spd * DT * heading
            x2 = x2 + step
            if x >= TRACK:
                # the animal is put back at the start; the integrator is not
                x -= TRACK; lap += 1
                x2 = x2 - TRACK * heading
        land.append(tuple(phase_of(x2)))
        pl1, pl2 = land.pop(0)

        a_th = THETA_DEPTH * (0.55 + 0.015 * spd)
        th_t = a_th * np.cos(2 * np.pi * F_THETA * t)

        pred = pred + DT / TAU_PRED * (-pred + s)
        I_F = ((1.0 + 1.6 * th_t) * (1.6 * bump_at(pl1, pl2) - 1.1 * pred)
               - F_G * buf_f[0] + F_DR + F_BG
               + F_NOISE * rng.standard_normal((NS, NS)))
        r_F = r_F + DT / TAU_EF * (-r_F + np.clip(I_F, 0, RMAX))
        g_f = g_f + DT / TAU_G * (-g_f + relu((r_F ** 2).mean()))
        buf_f.append(g_f); buf_f.pop(0)

        _lb = bump_at(pl1, pl2)
        I_E = ((1.0 + th_t) * conv(s, WF_E) - A_G * buf_s[0] + DRIVE
               + (gate / G_OPEN) * (G_CURR * (r_F - r_F.mean())
                                    + G_LAND * (_lb - _lb.mean()))
               + NOISE * rng.standard_normal((NS, NS)))
        r_E = r_E + DT / TAU_E * (-r_E + np.clip(I_E, 0, RMAX))
        s = s + DT / TAU_S * (-s + r_E)
        g_s = g_s + DT / TAU_G * (-g_s + relu((r_E ** 2).mean()))
        buf_s.append(g_s); buf_s.pop(0)

        d1, d2 = KMAT @ step
        d1 += PI_DIFF * rng.standard_normal()     # path-integration noise
        d2 += PI_DIFF * rng.standard_normal()
        if gate > 0:
            # The landmark correction. The error population reports where the
            # bump sits against where the landmarks say it should, and that
            # error is driven towards zero with time constant 1 / G_SHIFT.
            # It translates the bump instead of being added to its input, so
            # the gate can lock the phase without broadening the bump.
            z1 = np.angle((s * np.exp(1j * TH1)).sum())
            z2 = np.angle((s * np.exp(1j * TH2)).sum())
            e1 = (pl1 - z1 + np.pi) % (2 * np.pi) - np.pi
            e2 = (pl2 - z2 + np.pi) % (2 * np.pi) - np.pi
            d1 += (gate / G_OPEN) * G_SHIFT * DT * e1
            d2 += (gate / G_OPEN) * G_SHIFT * DT * e2
        s = shift2(s, d1, d2)
        r_E = shift2(r_E, d1, d2)

        if schedule is not None:
            gate = float(schedule[min(lap, len(schedule) - 1)])
        elif switch_lap is not None and lap >= switch_lap:
            gate = 0.0
        rec['lfp_s'].append(g_s); rec['lfp_f'].append(g_f)
        rec['lfp_th'].append(th_t)
        if i % 4 == 0:
            z = (s * np.exp(1j * TH1)).sum(), (s * np.exp(1j * TH2)).sum()
            ph = np.array([np.angle(z[0]), np.angle(z[1])])
            mm = (ph - np.array([pl1, pl2]) + np.pi) % (2 * np.pi) - np.pi
            rec['t'].append(t); rec['x'].append(x); rec['lap'].append(lap)
            rec['y'].append(x2.copy()); rec['pos'].append(x2.copy())
            rec['spd'].append(spd)
            rec['mism'].append(float(np.hypot(*mm)))
            rec['rE'].append(r_E.ravel().copy())
    out = {k: np.asarray(v) for k, v in rec.items() if k != 'pos'}
    out['pos'] = np.asarray(rec['pos'])
    return out


def lfp_of(r, w_th=9.0):
    z = lambda v: v - v.mean()
    return z(r['lfp_s']) + z(r['lfp_f']) + w_th * z(r['lfp_th'])


def ratemaps_2d(r, nbin=30, box=150.0, sigma=1.2):
    """Each unit's firing against position in the arena."""
    from scipy.ndimage import gaussian_filter
    p = r['pos']
    bx = np.clip((p[:, 0] / box * nbin).astype(int), 0, nbin - 1)
    by = np.clip((p[:, 1] / box * nbin).astype(int), 0, nbin - 1)
    idx = bx * nbin + by
    occ = np.bincount(idx, minlength=nbin * nbin).astype(float)
    R = r['rE']
    tot = np.zeros((R.shape[1], nbin * nbin))
    for k in range(R.shape[1]):
        tot[k] = np.bincount(idx, weights=R[:, k], minlength=nbin * nbin)
    M = tot / np.maximum(occ, 1)[None, :]
    M[:, occ < 3] = np.nan
    M = M.reshape(-1, nbin, nbin)
    return np.stack([gaussian_filter(np.nan_to_num(m), sigma) for m in M])


if __name__ == '__main__':
    import importlib.util
    import sys
    _s = importlib.util.spec_from_file_location('cgm', f'{FIG}/can_gate_model.py')
    CGM = importlib.util.module_from_spec(_s); _s.loader.exec_module(CGM)
    sys.path.insert(0, '/Users/harryclark/Documents/spatial-manifolds/'
                       'GRID-PATTERN-FORMATION')
    from scores import GridScorer

    print('foraging, for the rate maps ...', flush=True)
    W = run(G_OPEN, seed=1, walk=True, dur=240.0)
    MAPS = ratemaps_2d(W, nbin=30)
    _sc = GridScorer(30, ((0, 1), (0, 1)),
                     list(zip([0.2] * 10, np.linspace(0.4, 1.0, 10))))
    GS = np.array([_sc.get_scores(MAPS[k])[0] for k in range(len(MAPS))])
    print(f'  grid score: median {np.nanmedian(GS):.2f}, '
          f'{100 * np.nanmean(GS > 0.4):.0f}% above 0.4', flush=True)

    print('the track, in blocks of non-anchored / anchored / non-anchored ...',
          flush=True)
    NBLOCK = 33          # ~100 laps, as a recorded session has
    SCHED = np.r_[np.zeros(NBLOCK), np.full(NBLOCK, G_OPEN), np.zeros(NBLOCK)]
    R = run(G_OPEN, seed=0, schedule=SCHED, nlap=3 * NBLOCK)
    lap = R['lap']
    OPENLAP = (lap >= NBLOCK) & (lap < 2 * NBLOCK)
    print(f'  {int(lap.max()) + 1} laps; mean |mismatch| open '
          f'{np.degrees(R["mism"][OPENLAP].mean()):.0f} deg, shut '
          f'{np.degrees(R["mism"][~OPENLAP].mean()):.0f} deg', flush=True)

    PSW = CGM.population_state(R)
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
    print(f'  anchored fraction  open {np.nanmean(PSW["labels"][:, _open]):.2f}'
          f'  shut {np.nanmean(PSW["labels"][:, ~_open]):.2f}')
    print(f'  PC1 positive on {(PSW["pc1"][_open] > 0).sum()}/{_open.sum()} '
          f'open and {(PSW["pc1"][~_open] > 0).sum()}/{(~_open).sum()} shut',
          flush=True)

    # the two states are the two halves of the one session
    lfp = lfp_of(R)
    lf = np.repeat(lap, 4)[:len(lfp)]
    LFOPEN = (lf >= NBLOCK) & (lf < 2 * NBLOCK)
    SP = {}
    for nm, msk in (('open', LFOPEN), ('shut', ~LFOPEN)):
        f, P = welch(lfp[msk], fs=1 / DT, nperseg=4096)
        SP[f'P_{nm}'] = P
        for b, lo, hi in (('theta', 6, 10), ('slow', 30, 48), ('fast', 60, 100)):
            SP[f'{b}_{nm}'] = float(np.log10(P[(f >= lo) & (f < hi)]).mean())
    SP['f'] = f
    print('  log10 power, open - shut: ' + '  '.join(
        f'{b} {SP[f"{b}_open"] - SP[f"{b}_shut"]:+.3f}'
        for b in ('theta', 'slow', 'fast')), flush=True)

    # per-lap spectra, so the figure can be drawn as Figure 8E is
    L = []
    for Lp in range(int(lap.max()) + 1):
        seg = lfp[lf == Lp]
        if len(seg) < 2048:
            continue
        fl, Pl = welch(seg, fs=1 / DT, nperseg=2048)
        L.append((Lp, np.log10(Pl + 1e-30)))
    idx = np.array([a for a, _ in L]); A = np.stack([b for _, b in L])
    ok = (fl >= 2) & (fl <= 120)
    Zs = (A - A.mean(0)) / (A.std(0) + 1e-12)
    IO = (idx >= NBLOCK) & (idx < 2 * NBLOCK)
    wo = A[IO].mean(0) + np.log10(np.maximum(fl, 1e-9))
    ws = A[~IO].mean(0) + np.log10(np.maximum(fl, 1e-9))
    mu = np.concatenate([wo[ok], ws[ok]]).mean()
    MI, AMP = {}, {}
    ph_t = np.angle(hilbert(CGM.band(lfp, 5, 11)))
    for b, lo, hi in (('slow', 30, 48), ('fast', 60, 100)):
        a = np.abs(hilbert(CGM.band(lfp, lo, hi)))
        AMP[b] = a
        for nm, msk in (('open', LFOPEN), ('shut', ~LFOPEN)):
            MI[f'{b}_{nm}'] = CGM.tort_mi(ph_t[msk], a[msk])
        print(f'  theta-{b} MI open {MI[f"{b}_open"]:.5f} shut '
              f'{MI[f"{b}_shut"]:.5f}  '
              f'{100 * (MI[f"{b}_open"] - MI[f"{b}_shut"]) / MI[f"{b}_shut"]:+.1f}%',
              flush=True)
    for b in ('slow', 'fast'):
        for nm, msk in (('open', LFOPEN), ('shut', ~LFOPEN)):
            e = AMP[b][msk]
            pk = e > np.quantile(e, .9)
            MI[f'ph_{b}_{nm}'] = float(np.degrees(np.angle(
                np.mean(np.exp(1j * ph_t[msk][pk])))) % 360)
        print(f'  theta phase of the {b} peak: open '
              f'{MI[f"ph_{b}_open"]:.0f} deg  shut {MI[f"ph_{b}_shut"]:.0f} deg',
              flush=True)
    af = AMP['fast'][::4][:len(R['mism'])]
    shut = ~OPENLAP
    rho = float(np.corrcoef(R['mism'][shut], af[shut])[0, 1])
    q = np.quantile(R['mism'][shut], np.linspace(0, 1, 11))
    dec = [af[shut][(R['mism'][shut] >= q[i]) & (R['mism'][shut] < q[i + 1])].mean()
           for i in range(10)]
    print(f'  prediction, gate shut: r(|mismatch|, fast gamma) = {rho:+.3f}',
          flush=True)

    np.savez_compressed(
        OUT, maps=MAPS[::7], grid_score=GS[::7], gs_all=GS,
        sw_map=PSW['maps'][_bi], sw_labels=PSW['labels'], sw_pc1=PSW['pc1'],
        sw_load=PSW['pc1_load'], sw_edges=np.array(_edge),
        sp_f=fl[ok], rel_open=(wo - mu)[ok], rel_shut=(ws - mu)[ok],
        z_open=Zs[IO].mean(0)[ok], z_shut=Zs[~IO].mean(0)[ok],
        f=SP['f'], P_open=SP['P_open'], P_shut=SP['P_shut'],
        d=np.log10(SP['P_open']) - np.log10(SP['P_shut']),
        dec=np.array(dec), pred_r=rho,
        mi=np.array([MI['slow_open'], MI['slow_shut'],
                     MI['fast_open'], MI['fast_shut']]),
        ph=np.array([MI['ph_slow_open'], MI['ph_slow_shut'],
                     MI['ph_fast_open'], MI['ph_fast_shut']]),
        band_d=np.array([SP[f'{b}_open'] - SP[f'{b}_shut']
                         for b in ('theta', 'slow', 'fast')]))
    print(f'wrote {OUT}')
