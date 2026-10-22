"""Theta-nested spectral components (tSCs) and the anchoring state.

Reads the per-cycle maps written by tsc_extract.py and asks whether anchored and
non-anchored trials are built from different KINDS of theta cycle.

NORMALISATION. Each session's maps are divided, frequency by frequency, by that
session's median amplitude at the frequency (removing the 1/f slope and
electrode differences between sessions). The log of that is taken, and each
cycle's own mean subtracted, so a map describes the SHAPE of the cycle -- where in
frequency and theta phase its power sits -- and not how strong it is. The
cycle's overall amplitude is kept separately (`level`).

DECOMPOSITION. Equal numbers of cycles from every session (so no session
dominates), PCA to the components carrying VAR_KEEP of the variance, then
FastICA for K components. Each component is a frequency x phase template, signed
so its largest absolute value is positive. Every cycle is then projected to give
its weight on each. Fitted on the INDUCED maps (theta-locked waveform removed --
see tsc_extract.py) and separately on the RAW maps; a raw component with no
induced counterpart is theta waveform shape, not gamma.

STABILITY. Templates are fitted twice on disjoint halves of the sessions and
matched; a component that does not reproduce is not reported as a finding.

STATE. Within each session, anchored and non-anchored cycles are matched on
running speed (deciles), and the mean weight on each component compared; the
session is the unit (Wilcoxon), Holm across components. Also the fraction of
cycles whose largest weight is each component.

Writes data/lfp/tsc_components.npz and data/lfp/tsc_state.csv
"""
import glob
import os
import sys

import numpy as np
import pandas as pd
from scipy.stats import wilcoxon
from sklearn.decomposition import PCA, FastICA

ROOT = '/Users/harryclark/Documents/spatial-manifolds'
K = 4                      # chosen by split-half stability: only K = 4 keeps
                           # every component reproducible (|r| >= 0.79); see report
VAR_KEEP = .90
N_PER_SESSION = 800
SEED = 0


def load_session(fn, kind):
    z = np.load(fn, allow_pickle=True)
    cnt = z['cnt'][:, None, :]
    A = z[f'{kind}_sum'] / np.maximum(cnt, 1)
    med = np.median(A, axis=(0, 2), keepdims=True)
    L = np.log(np.maximum(A / med, 1e-6))
    level = L.mean(axis=(1, 2))
    shape = L - level[:, None, None]
    meta = {k: z[k] for k in ('anch', 'speed', 'trial', 'theta_amp') if k in z.files}
    return shape.reshape(len(shape), -1).astype(np.float32), level, meta


def fit(X, k=K, seed=SEED):
    pca = PCA(n_components=VAR_KEEP, svd_solver='full', random_state=seed).fit(X)
    Z = pca.transform(X)
    k = min(k, Z.shape[1])
    ica = FastICA(n_components=k, random_state=seed, max_iter=2000, whiten='unit-variance').fit(Z)
    T = ica.mixing_.T @ pca.components_            # templates in map space
    s = np.sign(T[np.arange(len(T)), np.abs(T).argmax(1)])
    T = T * s[:, None]
    unmix = (ica.components_ * s[:, None])          # in PCA space

    def project(Y):
        return (pca.transform(Y) @ unmix.T)
    return T, project


def match(T1, T2):
    """Greedy best-correlation matching of two template sets."""
    C = np.corrcoef(T1, T2)[:len(T1), len(T1):]
    pairs, used = [], set()
    for i in np.argsort(-np.abs(C).max(1)):
        j = max((j for j in range(len(T2)) if j not in used), key=lambda j: abs(C[i, j]))
        used.add(j); pairs.append((i, j, C[i, j]))
    return sorted(pairs)


def speed_matched_idx(speed, anch, rng, nq=10):
    q = np.quantile(speed, np.linspace(0, 1, nq + 1)); q[0] -= 1; q[-1] += 1
    keep = []
    for lo, hi in zip(q[:-1], q[1:]):
        m = (speed > lo) & (speed <= hi)
        a = np.flatnonzero(m & anch); b = np.flatnonzero(m & ~anch)
        n = min(len(a), len(b))
        if n:
            keep += list(rng.choice(a, n, replace=False)) + list(rng.choice(b, n, replace=False))
    return np.array(sorted(keep), int)


def holm(p):
    p = np.asarray(p, float); o = np.argsort(p); out = np.empty_like(p); run = 0
    for r, i in enumerate(o):
        run = max(run, p[i] * (len(p) - r)); out[i] = min(run, 1.0)
    return out


def analyse(files, kind, k=K, seed=SEED, verbose=True):
    rng = np.random.default_rng(seed)
    S = [load_session(f, kind) for f in files]
    keys = [os.path.basename(f)[:-4] for f in files]
    fitX = np.concatenate([x[rng.choice(len(x), min(N_PER_SESSION, len(x)), replace=False)]
                           for x, _, _ in S])
    T, project = fit(fitX, k, seed)
    # split-half stability over sessions
    half = rng.permutation(len(S))
    h1, h2 = half[: len(S) // 2], half[len(S) // 2:]
    stab = None
    if len(h1) >= 2 and len(h2) >= 2:
        T1, _ = fit(np.concatenate([S[i][0][:N_PER_SESSION] for i in h1]), k, seed)
        T2, _ = fit(np.concatenate([S[i][0][:N_PER_SESSION] for i in h2]), k, seed)
        full1 = match(T, T1); full2 = match(T, T2)
        stab = [min(abs(a[2]), abs(b[2])) for a, b in zip(full1, full2)]
    rows = []
    for key, (x, level, meta) in zip(keys, S):
        W = project(x)
        if 'anch' not in meta or meta['anch'].all() or (~meta['anch']).all():
            continue
        idx = speed_matched_idx(meta['speed'], meta['anch'], rng)
        if len(idx) < 40:
            continue
        a = meta['anch'][idx]; Wm = W[idx]
        dom = Wm.argmax(1)
        r = dict(session=key, n=len(idx), level_anch=level[idx][a].mean(),
                 level_non=level[idx][~a].mean())
        for c in range(W.shape[1]):
            r[f'w{c}_anch'] = Wm[a, c].mean(); r[f'w{c}_non'] = Wm[~a, c].mean()
            r[f'occ{c}_anch'] = (dom[a] == c).mean(); r[f'occ{c}_non'] = (dom[~a] == c).mean()
        rows.append(r)
    D = pd.DataFrame(rows)
    res = []
    for c in range(T.shape[0]):
        d = D[f'w{c}_anch'] - D[f'w{c}_non']
        o = D[f'occ{c}_anch'] - D[f'occ{c}_non']
        res.append(dict(component=c, d_weight=float(d.median()),
                        p_weight=float(wilcoxon(d).pvalue) if len(d) >= 6 else np.nan,
                        d_occupancy=float(o.median()),
                        p_occupancy=float(wilcoxon(o).pvalue) if len(o) >= 6 else np.nan,
                        stability=None if stab is None else float(stab[c])))
    R = pd.DataFrame(res)
    R['holm_weight'] = holm(R.p_weight.fillna(1)); R['holm_occupancy'] = holm(R.p_occupancy.fillna(1))
    if verbose:
        print(f'[{kind}] {len(S)} sessions, {len(D)} with both states, K = {T.shape[0]}')
        print(R.round(4).to_string(index=False))
    return T, R, D


if __name__ == '__main__':
    files = sorted(glob.glob(sys.argv[1] if len(sys.argv) > 1 else f'{ROOT}/data/lfp/tsc/M*D*.npz'))
    out = {}
    for kind in ('ind', 'raw'):
        T, R, D = analyse(files, kind)
        out[kind] = (T, R, D)
    Ti, Tr = out['ind'][0], out['raw'][0]
    print('\nraw components against induced (|r| of best match): '
          + ', '.join(f'raw{i}->ind{j} {abs(c):.2f}' for i, j, c in match(Tr, Ti)))
    z0 = np.load(files[0])
    dest = sys.argv[2] if len(sys.argv) > 2 else f'{ROOT}/data/lfp/tsc_components.npz'
    np.savez_compressed(dest, induced=Ti, raw=Tr, freqs=z0['freqs'], nph=z0['nph'])
    out['ind'][1].assign(kind='induced').to_csv(dest.replace('.npz', '_state.csv'), index=False)
    print('wrote', dest)
