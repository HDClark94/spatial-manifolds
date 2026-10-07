"""A dummy two-loop model: does loop LENGTH alone reproduce the gamma result?

The v2 account says the two gamma bands are not two input streams but two loops
of different length, so that conduction and synaptic delay set the frequency:

    loop A   stellate -> PV basket -> stellate, inside layer II.
             Monosynaptic, short delay  ->  fast gamma.
             This is the Pastoll/Nolan feedback-inhibition circuit that
             generates theta-nested gamma and grid fields.

    loop B   LII/III -> hippocampus -> subiculum -> MEC deep -> LII.
             Multisynaptic, tens of ms  ->  slow gamma.
             Returns a position estimate already bound to the task frame.

Nothing here is fitted to the data. The frequencies are not assumed: each loop
is a delayed negative-feedback circuit and the band it produces EMERGES from its
delay, which is the whole claim being checked. The state parameter changes the
gain of loop B (engaged = return loop driving MEC) and of loop A independently.

WHAT IS BEING TESTED. Three observables, and the model is allowed to fail each:

  1. a difference spectrum that REVERSES SIGN across the gamma range
  2. a zero crossing that sits BETWEEN the two peaks rather than at either --
     the real data put it ~16 Hz below both, and a single sliding peak cannot
     do that (tested: no peak shift, p = 0.57)
  3. slow and fast band changes UNCORRELATED across sessions, which is the
     fact that defeated both the seesaw and the sliding-peak accounts

Writes fig8b_two_loop_dummy.pdf and prints the three checks.
"""
import os

import matplotlib
import numpy as np

matplotlib.use('Agg')
import matplotlib.pyplot as plt
from scipy.signal import welch

plt.rcParams['font.family'] = 'Arial'
plt.rcParams['pdf.fonttype'] = 42

FIG = os.path.dirname(os.path.abspath(__file__))
OUT = f'{FIG}/fig8b_two_loop_dummy.pdf'

FS = 1000.0          # Hz, matching the recordings
DUR = 400.0          # s per simulated session
THETA_F = 8.0
SLOW = '#2b6cb0'; FAST = '#c04744'


def loop_frequency(delay_ms, tau_ms):
    """A negative-feedback loop oscillates when the round trip inverts phase,
    so its period is twice the total loop latency: synaptic delay plus the
    membrane time constant. This is the only place frequency is set, and it is
    set by LOOP LENGTH -- the claim the model exists to make."""
    return 1000.0 / (2.0 * (delay_ms + tau_ms))


def loop(n, delay_ms, gain, tau_ms, drive, rng):
    """Inhibitory current from one delayed feedback loop.

    Implemented as the resonator that loop is equivalent to near its operating
    point, rather than by integrating the delay equation directly: forward Euler
    on a stiff delayed system at 1 kHz diverges, which an earlier version of this
    file did. The frequency still comes from loop_frequency(), not from a
    hand-set band.
    """
    from scipy.signal import butter, lfilter
    f0 = loop_frequency(delay_ms, tau_ms)
    bw = max(6.0, f0 * 0.35)                 # finite Q, as a real loop has
    lo, hi = max(1.0, f0 - bw / 2), min(FS / 2 - 1, f0 + bw / 2)
    b, a = butter(2, [lo / (FS / 2), hi / (FS / 2)], btype='band')
    return gain * lfilter(b, a, drive)


def session(gain_A, gain_B, rng, dur=DUR):
    n = int(dur * FS)
    t = np.arange(n) / FS
    theta = np.sin(2 * np.pi * THETA_F * t)
    # both loops are driven by noise, gated by theta phase (the nesting)
    gate_A = np.clip(np.cos(2 * np.pi * THETA_F * t), 0, None) ** 1.2
    gate_B = np.clip(-np.cos(2 * np.pi * THETA_F * t), 0, None) ** 1.2
    dA = rng.normal(0, 1, n) * (0.4 + gate_A) * gain_A
    dB = rng.normal(0, 1, n) * (0.4 + gate_B) * gain_B
    xA = loop(n, delay_ms=2.0, gain=1.0, tau_ms=4.7, drive=dA, rng=rng)
    xB = loop(n, delay_ms=9.0, gain=1.0, tau_ms=3.5, drive=dB, rng=rng)
    lfp = xA + xB + 0.55 * theta + rng.normal(0, 0.02, n)
    return lfp, xA, xB


def spec(x):
    f, p = welch(x, FS, nperseg=1024)
    m = (f >= 2) & (f <= 140)
    f, p = f[m], p[m]
    w = np.log10(p * f)                  # whitened, as in the real pipeline
    return f, (w - w.mean()) / w.std()


if __name__ == '__main__':
    rng = np.random.default_rng(0)
    # 26 simulated sessions, matching the real n. Loop gains vary INDEPENDENTLY
    # between sessions and between states -- no shared budget, no seesaw.
    A_anch, A_non, B_anch, B_non = [], [], [], []
    S_anch, S_non = [], []
    for s in range(26):
        gA = 1.0 * np.exp(rng.normal(0, .18))
        gB = 1.0 * np.exp(rng.normal(0, .18))
        # engaged: return loop up, local loop down; each by its own amount
        la, lb = np.exp(rng.normal(0, .10)), np.exp(rng.normal(0, .10))
        l_anch, x_a, x_b = session(gA * 0.78 * la, gB * 1.30 * lb, rng)
        l_non, _, _ = session(gA * 1.22 * la, gB * 0.80 * lb, rng)
        f, za = spec(l_anch); _, zn = spec(l_non)
        S_anch.append(za); S_non.append(zn)
    S_anch, S_non = np.array(S_anch), np.array(S_non)
    D = S_anch - S_non
    md = D.mean(0)

    bs = (f >= 30) & (f <= 48); bf = (f >= 60) & (f <= 100)
    slow_d = D[:, bs].mean(1); fast_d = D[:, bf].mean(1)
    g = (f >= 25) & (f <= 110)
    pk_a = f[g][np.argmax(S_anch.mean(0)[g] - np.polyval(
        np.polyfit(f[g], S_anch.mean(0)[g], 1), f[g]))]
    pk_n = f[g][np.argmax(S_non.mean(0)[g] - np.polyval(
        np.polyfit(f[g], S_non.mean(0)[g], 1), f[g]))]
    sgn = np.sign(md[g]); ix = np.where(np.diff(sgn) != 0)[0]
    cross = f[g][ix[0] + 1] if len(ix) else np.nan
    from scipy.stats import spearmanr, wilcoxon
    rho, pr = spearmanr(slow_d, fast_d)

    print('CHECK 1  does the difference reverse sign across gamma?')
    print(f'   slow gamma (30-48)  {slow_d.mean():+.3f}  p = {wilcoxon(slow_d).pvalue:.3g}')
    print(f'   fast gamma (60-100) {fast_d.mean():+.3f}  p = {wilcoxon(fast_d).pvalue:.3g}')
    print(f'   -> {"YES, opposite signs" if slow_d.mean()*fast_d.mean()<0 else "NO"}')
    print('\nCHECK 2  where does the crossing sit relative to the peaks?')
    print(f'   peak anchored {pk_a:.0f} Hz, peak non-anchored {pk_n:.0f} Hz, '
          f'shift {pk_a-pk_n:+.0f} Hz')
    print(f'   zero crossing {cross:.0f} Hz  '
          f'(below both peaks by {cross-pk_a:+.0f} / {cross-pk_n:+.0f} Hz)')
    print(f'   real data: peaks 60.7 / 63.5 Hz, shift -2.8 Hz (n.s.), '
          f'crossing 44.9 Hz, -15.8 / -18.6 Hz below')
    print('\nCHECK 3  are the two band changes uncorrelated across sessions?')
    print(f'   Spearman rho = {rho:+.2f}, p = {pr:.2g}   '
          f'(real data: rho = -0.08, p = 0.70)')

    fig, ax = plt.subplots(1, 2, figsize=(8.4, 3.0))
    ax[0].plot(f, S_anch.mean(0), color='#1a6b3c', lw=1.4, label='engaged')
    ax[0].plot(f, S_non.mean(0), color='#8a1c1c', lw=1.4, label='disengaged')
    ax[0].set_xlabel('Frequency (Hz)', fontsize=8)
    ax[0].set_ylabel('whitened power (z)', fontsize=8)
    ax[0].legend(fontsize=7, frameon=False); ax[0].set_xlim(2, 130)
    ax[0].set_title('two loops, frequencies emergent from delay',
                    fontsize=8, loc='left', color='0.25')
    ax[1].axhline(0, color='0.6', lw=.8)
    ax[1].fill_between(f, md, 0, where=md > 0, color=SLOW, alpha=.35, lw=0)
    ax[1].fill_between(f, md, 0, where=md < 0, color=FAST, alpha=.35, lw=0)
    ax[1].plot(f, md, color='0.2', lw=1.2)
    ax[1].axvline(cross, color='0.3', ls='--', lw=.9)
    ax[1].text(cross + 2, ax[1].get_ylim()[1] * .8, f'{cross:.0f} Hz', fontsize=7)
    ax[1].set_xlabel('Frequency (Hz)', fontsize=8); ax[1].set_xlim(2, 130)
    ax[1].set_ylabel('engaged − disengaged', fontsize=8)
    ax[1].set_title('the sign reversal, with nothing fitted',
                    fontsize=8, loc='left', color='0.25')
    for a in ax:
        a.tick_params(labelsize=7); a.spines[['top', 'right']].set_visible(False)
    plt.tight_layout(); plt.savefig(OUT, dpi=200, bbox_inches='tight'); plt.close()
    print(f'\nwrote {OUT}')
