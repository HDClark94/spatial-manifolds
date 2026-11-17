"""Example cells for Figure S4, the open-field decomposition (of_xgb_decomp.py).

For each example cell the selected model is refitted fold by fold exactly as
the decomposition fits it, and its held-out predictions are kept, so that every
predicted tuning curve in the figure comes from bins the model never saw. All
seven models are scored for the walkthrough cell (Figure S4B), to show the
path forward selection takes. A ten-second stretch of that cell's variables,
spikes and held-out rate feeds the schematic (Figure S4A).

Writes data/of_xgb/of_xgb_examples.npz.
"""
import itertools
import sys

import numpy as np
import pandas as pd
import xgboost as xgb

HERE = '/Users/harryclark/Documents/spatial-manifolds/scripts/figures/AnchorDynamics2026'
sys.path.insert(0, HERE)
from of_xgb_decomp import DT, N_FOLD, OUT, XGB, fold_llh, select, session_bins  # noqa: E402

# (mouse, day, cluster): one MEC cell of each kind; the last is the walkthrough
EXAMPLES = [(21, 26, 160), (27, 18, 213), (26, 11, 225), (21, 24, 246)]
WALK = EXAMPLES[-1]
MODELS = [''.join(m) for k in (1, 2, 3) for m in itertools.combinations('PHS', k)]
TRACE_S = 10.
N_POS, N_HD, S_EDGES = 40, 30, np.arange(0, 42, 3.)
A = pd.read_csv(f'{OUT}/of_xgb_cells.csv', keep_default_na=False)


def cv_predict(y, X, fold):
    """Held-out rate per bin: each fold predicted by a model fitted, with early
    stopping, on the others (as fold_llh fits it)."""
    mu = np.full(len(y), np.nan)
    for k in range(N_FOLD):
        tr, te = fold != k, fold == k
        va = fold == (k + 1) % N_FOLD
        fi = tr & ~va
        m = xgb.XGBRegressor(max_depth=X.shape[1], **XGB).fit(
            X[fi], y[fi], eval_set=[(X[va], y[va])], verbose=False)
        mu[te] = m.predict(X[te])
    return mu / DT


def tuning(y, mu, F):
    """Observed (spikes) and predicted (held-out rate) tuning to each variable."""
    occ = lambda idx, n: np.bincount(idx, minlength=n) * DT
    out = {}
    xy = np.clip((F['P'] / 100 * N_POS).astype(int), 0, N_POS - 1)
    ip = xy[:, 1] * N_POS + xy[:, 0]
    hd = np.arctan2(F['H'][:, 0], F['H'][:, 1])
    ih = np.clip(((hd + np.pi) / (2 * np.pi) * N_HD).astype(int), 0, N_HD - 1)
    isp = np.digitize(F['S'][:, 0], S_EDGES) - 1
    ok = (isp >= 0) & (isp < len(S_EDGES) - 1)
    # minimum occupancy (s): a 2.5 cm position bin is visited far less than an
    # angle or speed bin
    for name, idx, n, m, t_min in (('P', ip, N_POS ** 2, slice(None), .1),
                                   ('H', ih, N_HD, slice(None), .5),
                                   ('S', isp, len(S_EDGES) - 1, ok, .5)):
        t = occ(idx[m], n)
        with np.errstate(invalid='ignore', divide='ignore'):
            o = np.bincount(idx[m], y[m], n) / t
            p = np.bincount(idx[m], mu[m] * DT, n) / t
        o[t < t_min], p[t < t_min] = np.nan, np.nan
        out[f'{name}_obs'], out[f'{name}_pred'] = o, p
    out['P_obs'] = out['P_obs'].reshape(N_POS, N_POS)
    out['P_pred'] = out['P_pred'].reshape(N_POS, N_POS)
    return out


res = {}
for i, (mo, dy, c) in enumerate(EXAMPLES):
    B = session_bins(mo, dy)
    y = np.histogram(np.asarray(B['cl'][c].index, float), B['edges'])[0].astype(float)
    row = A[(A.mouse == mo) & (A.day == dy) & (A.cluster_id == c)].iloc[0]
    model = row.model
    if (mo, dy, c) == WALK:
        model2, LL = select(y, B['F'], B['vars'], B['fold'])
        assert model2 == model, (model2, model)
        res['walk_visited'] = np.array([''.join(k) for k in LL])
        for m in MODELS:
            LL.setdefault(tuple(m), fold_llh(y, np.hstack([B['F'][v] for v in m]), B['fold']))
        res['walk_llh'] = np.array([LL[tuple(m)] for m in MODELS])
    X = np.hstack([B['F'][v] for v in model])
    mu = cv_predict(y, X, B['fold'])
    for k, v in tuning(y, mu, B['F']).items():
        res[f'c{i}_{k}'] = v
    res[f'c{i}_id'] = np.array([mo, dy, c])
    res[f'c{i}_model'] = np.array(model)
    res[f'c{i}_llh'] = float(row[f'llh_{model}'])
    res[f'c{i}_rate'] = float(row.rate)
    if (mo, dy, c) == WALK:
        t0 = int(np.argmax(np.convolve(y, np.ones(int(TRACE_S / DT)), 'valid')))
        sl = slice(t0, t0 + int(TRACE_S / DT))
        res['trace_y'] = y[sl]
        res['trace_mu'] = mu[sl]
        res['trace_F'] = np.hstack([B['F'][v] for v in 'PHS'])[sl]
        res['fold'] = B['fold']
    print(f'M{mo}D{dy} c{c}: {model}, {row.rate:.1f} Hz, {float(row[f"llh_{model}"]):.2f} bits/spike')

res['models'] = np.array(MODELS)
res['S_edges'] = S_EDGES
np.savez_compressed(f'{OUT}/of_xgb_examples.npz', **res)
print(f'wrote {OUT}/of_xgb_examples.npz')
