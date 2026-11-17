"""How many grid cells does the bump measure need? Split-half reliability.

The bump in Figure S18 is read by projecting the module's rates onto its cells'
torus phases. With few cells that projection is noisy, and a session's bump
measures are then as much sampling noise as attractor. The direct test: split
the module into two disjoint random halves of k cells each, read the bump's
phase from each half separately, and ask how closely the two agree moment by
moment -- mean cos(psi_A - psi_B) over running 100 ms bins, averaged over the
two torus axes (1 = identical, 0 = unrelated). Agreement can only come from a
shared bump, since the halves share no cells. Done for every k up to half the
module, separately in each state, 100 random splits per k.

Also recorded per k: the bump above its phase shuffle (Figure S18G) read from
random k-cell subsets, to show how the measure itself depends on cell count.

Reads data/population_state/attractor_cache/ (attractor_cache.py)
Writes data/population_state/attractor_reliability.csv
"""
import glob
import os

import numpy as np
import pandas as pd

ROOT = '/Users/harryclark/Documents/spatial-manifolds'
PS = f'{ROOT}/data/population_state'
NSPLIT = 100
NPERM = 20


def pv(R, ph):
    z = np.abs(np.exp(1j * ph) @ R)
    return z / (np.sqrt(len(ph) / 2.0) * np.linalg.norm(R, axis=0) + 1e-9)


def run(f, rng):
    z = np.load(f)
    cells = list(z['cells'])
    mi = [cells.index(c) for c in z['module']]
    R, phi = z['R'][mi].astype(float), z['phi']
    a, n = z['anch'], z['non']
    N = len(mi)
    rows = []
    for k in range(2, N // 2 + 1):
        ag = {'a': [], 'n': []}
        for _ in range(NSPLIT):
            p = rng.permutation(N)
            A, B = p[:k], p[k:2 * k]
            c = []
            for j in range(2):
                pa = np.angle(np.exp(1j * phi[j, A]) @ R[A])
                pb = np.angle(np.exp(1j * phi[j, B]) @ R[B])
                c.append(np.cos(pa - pb))
            c = np.mean(c, 0)
            ag['a'].append(c[a].mean()); ag['n'].append(c[n].mean())
        rows.append(dict(k=k, agree_a=np.mean(ag['a']), agree_n=np.mean(ag['n']),
                         agree_a_sd=np.std(ag['a']), agree_n_sd=np.std(ag['n'])))
    # the S18G measure from k-cell subsets
    for k in sorted({4, 6, 8, 12, 16, 24, 32, N} & set(range(4, N + 1))):
        ea, en = [], []
        for _ in range(30 if k < N else 1):
            S = rng.permutation(N)[:k]
            o, s = [], []
            for j in range(2):
                o.append(pv(R[S], phi[j, S]))
                s.append(np.mean([pv(R[S], rng.permutation(phi[j, S])) for _ in range(NPERM)], 0))
            d = np.mean(o, 0) - np.mean(s, 0)
            ea.append(d[a].mean()); en.append(d[n].mean())
        rows.append(dict(k=k, sub_bump_a=np.mean(ea), sub_bump_n=np.mean(en),
                         sub_bump_a_sd=np.std(ea), sub_bump_n_sd=np.std(en)))
    D = pd.DataFrame(rows)
    D.insert(0, 'n_grid', N); D.insert(0, 'day', int(z['day'])); D.insert(0, 'mouse', int(z['mouse']))
    return D


if __name__ == '__main__':
    rng = np.random.default_rng(0)
    out = []
    for f in sorted(glob.glob(f'{PS}/attractor_cache/M*D*.npz')):
        D = run(f, rng)
        h = D.dropna(subset=['agree_a']).iloc[-1]
        print(f'  M{D.mouse[0]}D{D.day[0]}: {D.n_grid[0]:2d} grid; halves of {int(h.k)}: '
              f'agreement {h.agree_a:.2f} anchored, {h.agree_n:.2f} non-anchored', flush=True)
        out.append(D)
    pd.concat(out, ignore_index=True).to_csv(f'{PS}/attractor_reliability.csv', index=False)
    print('wrote attractor_reliability.csv')
