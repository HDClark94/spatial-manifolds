"""The MEC population's state space, anchored and non-anchored.

Reads data/manifold_supp/M{mo}D{dy}.npz (manifold_supp_build.py): every MEC
cell, 100 ms bins, running only, sqrt and z-scored per cell.

POINTS. Consecutive running bins of one lap are averaged in 0.5 s windows (time
only; position is never used to build a point). Each window keeps its lap,
track position and state.

THREE EMBEDDINGS, each fitted once on both states together so that they share a
space:
  PCA        on the windows (10 components kept, 3 drawn)
  UMAP       3D, on the 10 PCs of the windows
  CEBRA      CEBRA-Time (unsupervised: positives are bins close in time; it is
             never given position or state), 3D on the hypersphere, trained on
             the 100 ms bins and averaged into the same windows. Because it
             learns whatever changes slowly, and states come in blocks of laps,
             its separation of the states may include slow drift; it is shown,
             not tested.

IS IT A RING? Persistent homology (ripser, H1) on each state SEPARATELY, in the
6-PC space, with equal numbers of windows per state: each window replaced by the
mean of its K_DENOISE nearest neighbours, then FPS_N points by furthest-point
sampling. A ring gives one long H1 bar and short others. The null shifts each
cell's whole time series by its own random offset (breaking co-activity, keeping
every cell's statistics) and reruns the identical pipeline N_SHUF times.
Each state's statistic is the mean over N_SUB random subsamples. Reported per
state: the longest bar against the null's 95th percentile, and its dominance over
the second bar. Lifetimes are divided by the cloud's extent
(95th percentile of pairwise distances), so a more spread-out cloud does not
win on scale alone.

POSITIVE CONTROL. The same windows rebuilt as a ring: each replaced by the
anchored track template at its position plus the residual of another window of
the same state, so the noise is the data's own. If the pipeline finds one
dominant loop there and not in the data, the data are not a single ring.

IS THE MANIFOLD STILL ORGANISED BY THE TRACK? Position decoded from each
embedding within each state by k-nearest neighbours on (cos, sin) of track phase,
5-fold cross-validated with whole laps held out; median circular error, against
the same with each lap's positions circularly shifted by its own random offset.

Writes data/manifold_supp/results/M{mo}D{dy}.npz
"""
import glob
import os
import sys
import time
import warnings

import numpy as np

warnings.filterwarnings('ignore')
from sklearn.decomposition import PCA
from sklearn.model_selection import GroupKFold
from sklearn.neighbors import KNeighborsRegressor, NearestNeighbors

ROOT = '/Users/harryclark/Documents/spatial-manifolds'
IND = f'{ROOT}/data/manifold_supp'
OUTD = f'{IND}/results'
TL = 200.0
WIN = 5                    # bins per window (0.5 s)
MIN_WIN = 300              # windows per state
MIN_LAPS = 10              # laps per state
N_MATCH = 1000             # windows per state for the homology
K_DENOISE = 10
FPS_N = 300
N_SHUF = 20
N_SUB = 5                  # subsamples averaged per state (real and control)
K_DEC = 20
CEBRA_ITERS = 3000


def windows(X, t, tr):
    out, lab = [], []
    for k in np.unique(tr):
        i = np.flatnonzero(tr == k)
        for s in np.split(i, np.flatnonzero(np.diff(t[i]) > 0.15) + 1):
            for j in range(0, len(s) - WIN + 1, WIN):
                out.append(X[s[j:j + WIN]].mean(0)); lab.append(s[j + WIN // 2])
    return np.array(out), np.array(lab)


def fps(P, n, rng):
    i = [int(rng.integers(len(P)))]
    d = np.linalg.norm(P - P[i[0]], axis=1)
    for _ in range(n - 1):
        k = int(d.argmax()); i.append(k); d = np.minimum(d, np.linalg.norm(P - P[k], axis=1))
    return np.array(i)


def h1(P, rng):
    from ripser import ripser
    idx = NearestNeighbors(n_neighbors=K_DENOISE).fit(P).kneighbors(P)[1]
    Q = P[idx].mean(1)
    Q = Q[fps(Q, min(FPS_N, len(Q)), rng)]
    dg = ripser(Q, maxdim=1)['dgms'][1]
    # lifetimes as a fraction of the cloud's extent (95th percentile of pairwise
    # distances), so a more spread-out cloud does not win on scale alone
    from scipy.spatial.distance import pdist
    scale = np.percentile(pdist(Q), 95)
    life = np.sort(dg[:, 1] - dg[:, 0])[::-1] / scale
    return np.r_[life[:5], np.zeros(max(0, 5 - len(life)))], dg / scale


def ring_control(W, pos, a, rng, nb=50):
    """The same windows rebuilt AS a ring: the anchored track template at each
    window's position plus the residual of another window of the same state."""
    b = (pos / TL * nb).astype(int).clip(0, nb - 1)
    T = np.array([W[a & (b == k)].mean(0) if (a & (b == k)).any() else np.nan * W[0]
                  for k in range(nb)])
    T = np.where(np.isfinite(T), T, np.nanmean(T, 0))
    R = W - T[b]
    out = np.empty_like(W)
    for m in (a, ~a):
        i = np.flatnonzero(m)
        out[i] = T[b[i]] + R[rng.permutation(i)]
    return out


def homology(X, t, tr, anch, rng, keep_dgm=False, pos=None, control=False, n_sub=1):
    W, lab = windows(X, t, tr)
    if control:
        W = ring_control(W, pos[lab], anch[lab], rng)
    P = PCA(6, random_state=0).fit_transform(W)
    a = anch[lab]
    n = min(a.sum(), (~a).sum(), N_MATCH)
    out = {}
    for nm, m in (('anch', a), ('non', ~a)):
        lives = []
        for r_ in range(n_sub):
            i = rng.choice(np.flatnonzero(m), n, replace=False)
            life, dg = h1(P[i], rng)
            lives.append(life)
            if keep_dgm and r_ == 0:
                out[nm + '_dgm'] = dg
        out[nm] = np.mean(lives, 0)
    return out


def decode_pos(E, pos, lap, rng):
    """Median circular |error| (cm) of kNN position decoding, laps held out."""
    ph = 2 * np.pi * pos / TL
    Y = np.c_[np.cos(ph), np.sin(ph)]
    pred = np.zeros_like(Y)
    for trn, tst in GroupKFold(5).split(E, Y, lap):
        pred[tst] = KNeighborsRegressor(K_DEC).fit(E[trn], Y[trn]).predict(E[tst])
    d = np.angle(np.exp(1j * (np.arctan2(pred[:, 1], pred[:, 0]) - ph)))
    return float(np.median(np.abs(d)) * TL / (2 * np.pi))


def cebra_embed(X, seed=0):
    import torch
    from cebra import CEBRA
    torch.manual_seed(seed)
    dev = 'mps' if torch.backends.mps.is_available() else 'cpu'
    m = CEBRA(model_architecture='offset10-model', batch_size=512, learning_rate=3e-4,
              temperature=1, output_dimension=3, max_iterations=CEBRA_ITERS,
              distance='cosine', conditional='time', device=dev, verbose=False,
              time_offsets=10)
    m.fit(X)
    return m.transform(X)


def session(fn):
    import umap
    key = os.path.basename(fn)[:-4]
    z = np.load(fn)
    X, pos, tr, t = z['X'], z['pos'], z['trial'], z['t']
    anch = z['pc1'] > 0
    W, lab = windows(X, t, tr)
    a = anch[lab]
    laps_a, laps_n = len(np.unique(tr[anch])), len(np.unique(tr[~anch]))
    if min(a.sum(), (~a).sum()) < MIN_WIN or min(laps_a, laps_n) < MIN_LAPS:
        return None
    rng = np.random.default_rng(0)
    t0 = time.time()
    pca = PCA(10, random_state=0).fit(W)
    Pw = pca.transform(W)
    U = umap.UMAP(n_components=3, n_neighbors=40, min_dist=.3, random_state=0).fit_transform(Pw)
    Ec = cebra_embed(X)
    # CEBRA on 100 ms bins -> the same windows (mean, back onto the sphere)
    Cw, _ = windows(Ec, t, tr)
    Cw = Cw / np.linalg.norm(Cw, axis=1, keepdims=True)
    # homology, real and shuffled
    H = homology(X, t, tr, anch, rng, keep_dgm=True, n_sub=N_SUB)
    Hc = homology(X, t, tr, anch, rng, pos=pos, control=True, n_sub=N_SUB)
    Hs = {'anch': [], 'non': []}
    for _ in range(N_SHUF):
        Xs = np.column_stack([np.roll(X[:, j], rng.integers(500, len(X) - 500))
                              for j in range(X.shape[1])])
        h = homology(Xs, t, tr, anch, rng)
        Hs['anch'].append(h['anch']); Hs['non'].append(h['non'])
    # position decoding within each state, from each embedding
    dec = {}
    pw, lw = pos[lab], tr[lab]
    for enm, E in (('pca', Pw[:, :6]), ('umap', U), ('cebra', Cw)):
        for st, m in (('anch', a), ('non', ~a)):
            dec[f'{enm}_{st}'] = decode_pos(E[m], pw[m], lw[m], rng)
            # null: each lap's positions circularly shifted by its own offset
            sh = {k: rng.uniform(0, TL) for k in np.unique(lw[m])}
            pn = (pw[m] + np.array([sh[k] for k in lw[m]])) % TL
            dec[f'{enm}_{st}_null'] = decode_pos(E[m], pn, lw[m], rng)
    os.makedirs(OUTD, exist_ok=True)
    np.savez_compressed(
        f'{OUTD}/{key}.npz', P3=Pw[:, :3].astype(np.float32), U=U.astype(np.float32),
        C=Cw.astype(np.float32), pos=pw, lap=lw, anch=a, n_cells=X.shape[1],
        cls=z['cls'], var_expl=pca.explained_variance_ratio_,
        H_anch=H['anch'], H_non=H['non'], Hc_anch=Hc['anch'], Hc_non=Hc['non'], dgm_anch=H['anch_dgm'], dgm_non=H['non_dgm'],
        Hs_anch=np.array(Hs['anch']), Hs_non=np.array(Hs['non']),
        dec_keys=np.array(list(dec)), dec_vals=np.array(list(dec.values())))
    print(f'  {key}: {X.shape[1]} cells, {a.sum()}/{(~a).sum()} windows | H1 top anch '
          f'{H["anch"][0]:.2f} (null95 {np.percentile(np.array(Hs["anch"])[:, 0], 95):.2f}, '
          f'dom {H["anch"][0] / H["anch"][1]:.2f}) non {H["non"][0]:.2f} (null95 '
          f'{np.percentile(np.array(Hs["non"])[:, 0], 95):.2f}, dom {H["non"][0] / H["non"][1]:.2f}) | '
          f'ring control dom {Hc["anch"][0] / Hc["anch"][1]:.2f}/{Hc["non"][0] / Hc["non"][1]:.2f} | '
          f'pos err pca {dec["pca_anch"]:.0f}/{dec["pca_non"]:.0f} umap {dec["umap_anch"]:.0f}/'
          f'{dec["umap_non"]:.0f} cebra {dec["cebra_anch"]:.0f}/{dec["cebra_non"]:.0f} cm | '
          f'{time.time() - t0:.0f}s', flush=True)
    return True


if __name__ == '__main__':
    files = sorted(glob.glob(f'{IND}/M*D*.npz'))
    if len(sys.argv) > 1:
        files = [f for f in files if any(a_ in f for a_ in sys.argv[1:])]
    for fn in files:
        if os.path.exists(f'{OUTD}/{os.path.basename(fn)}') and len(sys.argv) == 1:
            continue
        try:
            session(fn)
        except Exception as e:
            print(f'  ! {os.path.basename(fn)}: {type(e).__name__}: {e}', flush=True)
