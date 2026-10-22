"""A parameter scan over the gated-attractor model, scored against the data.

Figure S13J puts the model's band changes beside the measured ones in the same
units and the comparison is unflattering: the model gets the sign of both
gamma changes and overshoots them by about an order of magnitude, and theta
moves the wrong way. This asks whether that is a property of the mechanism or
of the particular parameters it was first written with.

WHAT IS SCORED. The three band changes against their measured values, with the
anchoring result and the direction of theta-fast coupling as constraints
rather than terms: a setting that matched the band changes by abolishing the
attractor would not be an improvement. Runs are shortened for the sweep and
the winner is re-run at full length.

Writes can_gate_scan.csv
"""
import itertools
import os
import sys
import time

import numpy as np
import pandas as pd

FIG = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, FIG)
import importlib.util                                           # noqa: E402

_s = importlib.util.spec_from_file_location('cgm', f'{FIG}/can_gate_model.py')
M = importlib.util.module_from_spec(_s)
_s.loader.exec_module(M)

# the measured values, in log10 power, anchored minus non-anchored, over the
# same bands with the mains lines excluded (28 sessions)
TARGET = dict(theta=-0.0132, slow=+0.0178, fast=-0.0160)

GRID = dict(
    F_BG=[0.25, 1.0, 3.0],        # asynchronous background on the error ring
    A_G=[1.0, 2.0, 3.5],          # inhibition gain on the attractor ring
    F_G=[3.0, 6.0, 10.0],         # inhibition gain on the error ring
    G_OPEN=[2.0, 4.5, 8.0],       # how hard the gate drives the correction
)


def one(nlap=10, **kw):
    """Both gate states under one parameter setting."""
    old = {k: getattr(M, k) for k in kw}
    old['NLAP'] = M.NLAP
    for k, v in kw.items():
        setattr(M, k, v)
    M.NLAP = nlap
    try:
        S = {True: M.run(M.G_OPEN, 0), False: M.run(0.0, 0)}
        A = {k: M.analyse(v) for k, v in S.items()}
        row = {f'd_{b}': A[True][b] - A[False][b]
               for b in ('theta', 'slow', 'fast')}
        for b in ('slow', 'fast'):
            o, s_ = A[True][f'mi_{b}'], A[False][f'mi_{b}']
            row[f'mi_{b}_pct'] = 100 * (o - s_) / max(s_, 1e-9)
        for st, nm in ((True, 'open'), (False, 'shut')):
            Mp = M.ratemap(S[st], M.N // 4)
            C = np.corrcoef(np.nan_to_num(Mp))
            iu = np.triu_indices(len(C), 1)
            row[f'r_{nm}'] = float(C[iu].mean())
            row[f'pv_{nm}'] = float(np.mean(
                np.abs((S[st]['rE'] * np.exp(1j * M.th)).sum(1))
                / np.maximum(S[st]['rE'].sum(1), 1e-9)))
    finally:
        for k, v in old.items():
            setattr(M, k, v)
    return row


def score(r):
    """Distance from the measured band changes, with the model's job intact.

    The band terms are relative, so that a change of the right size counts the
    same in each band; anchoring and the direction of theta-fast coupling are
    penalties, because a setting that fits the spectrum by killing the
    attractor is not a better model of these recordings.
    """
    sse = sum(((r[f'd_{b}'] - t) / abs(t)) ** 2 for b, t in TARGET.items())
    pen = 0.0
    if r['r_open'] < 0.80:
        pen += 50 * (0.80 - r['r_open'])
    if r['r_shut'] > 0.25:
        pen += 50 * (r['r_shut'] - 0.25)
    if r['mi_fast_pct'] > 0:
        pen += 5.0
    if r['pv_open'] < 0.45 or r['pv_shut'] < 0.45:
        pen += 10.0
    return sse + pen


def main():
    keys = list(GRID)
    combos = list(itertools.product(*(GRID[k] for k in keys)))
    print(f'{len(combos)} settings, two runs each', flush=True)
    rows, t0 = [], time.time()
    for i, c in enumerate(combos):
        kw = dict(zip(keys, c))
        try:
            r = one(**kw)
        except Exception as e:
            print(f'  ! {kw}: {type(e).__name__}: {e}', flush=True)
            continue
        r.update(kw); r['score'] = score(r)
        rows.append(r)
        print(f'  [{i + 1:3d}/{len(combos)}] ' +
              ' '.join(f'{k}={v:g}' for k, v in kw.items()) +
              f' | theta {r["d_theta"]:+.3f} slow {r["d_slow"]:+.3f} '
              f'fast {r["d_fast"]:+.3f} | r {r["r_open"]:+.2f}/{r["r_shut"]:+.2f}'
              f' | score {r["score"]:.1f}  [{time.time() - t0:.0f}s]',
              flush=True)
    D = pd.DataFrame(rows).sort_values('score')
    D.to_csv(f'{FIG}/can_gate_scan.csv', index=False)
    print('\nbest ten:')
    cols = keys + ['d_theta', 'd_slow', 'd_fast', 'r_open', 'r_shut',
                   'mi_fast_pct', 'score']
    print(D[cols].head(10).to_string(index=False))
    print(f'\nthe model as it stands: ' +
          ' '.join(f'{k}={getattr(M, k):g}' for k in keys))
    print(f'measured: theta {TARGET["theta"]:+.4f} slow {TARGET["slow"]:+.4f} '
          f'fast {TARGET["fast"]:+.4f}')


if __name__ == '__main__':
    main()
