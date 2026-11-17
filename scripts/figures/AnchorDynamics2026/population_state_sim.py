"""Simulated sessions for Figure S2: how to summarise a population's labels into
one state per trial, the fraction of cells anchored or the z-scored PC1 score.

Built on the grid-cell simulation of Figure S1 (fig1_supp_classifier.py): the same
cells, rate maps and classifier. What S1 does not have, and recorded populations
do, is cells that do not follow the population state. Each session therefore
mixes three groups:

  followers    anchored on the trials the population is anchored (the true
               state), so all of them move together
  incoherent   each cell switching on its own, with the same epoch statistics
               as the true state but independently of it and of each other
  competing    one shared state of their own, independent of the true one: a
               second coherent group, the case where PC1 could lock on to the
               wrong group

The true state is a two-state Markov chain over N_TRIALS trials (mean epochs of
P_STAY lengths), starting anchored. Each cell is classified from the whole
track, the task region and the black box exactly as Figure 2 classifies them,
without the median filter. Outcomes are drawn from the true state on uncued
trials (half the trials): a hit with probability HIT[1] when anchored and HIT[0]
when not.

Cells are pooled per session (N_POOL of each group) and classified once; the
figure draws populations of different size and make-up from the pool, so the
expensive step is not repeated. Writes data/classifier_sim/population_state_sim.npz.
"""
import importlib.util
import os
import sys
import time

import numpy as np

HERE = '/Users/harryclark/Documents/spatial-manifolds/scripts/figures/AnchorDynamics2026'
sys.path.insert(0, '/Users/harryclark/Documents/spatial-manifolds/src')
from spatial_manifolds.anchoring import trial_cluster_labels  # noqa: E402

# the S1 simulation, imported without running its figure: its module-level code
# loads S1's caches, which exist after S1 has been built once
_spec = importlib.util.spec_from_file_location('s1', f'{HERE}/fig1_supp_classifier.py')
_src = open(f'{HERE}/fig1_supp_classifier.py').read()
_src = _src[:_src.index('# ============================ classification')]
S1 = importlib.util.module_from_spec(_spec)
exec(compile(_src, f'{HERE}/fig1_supp_classifier.py', 'exec'), S1.__dict__)

OUT = '/Users/harryclark/Documents/spatial-manifolds/data/classifier_sim/population_state_sim.npz'
N_SESS = 8
N_TRIALS = 120
N_POOL = {'followers': 80, 'incoherent': 60, 'competing': 60}
P_STAY = (.85, .93)                       # non-anchored, anchored: mean epochs ~7 and ~14 trials
HIT = (.3, .8)                            # uncued hit probability, non-anchored and anchored
STRETCH = {'whole': lambda tc: tc, 'task': lambda tc: tc[:, 25:55],
           'box': lambda tc: np.concatenate(
               [np.r_[np.full((1, 15), np.nan), tc[:-1, -15:]], tc[:, :15]], axis=1)}


def markov(rng, n=N_TRIALS, start=1):
    s = np.empty(n, int)
    s[0] = start
    for t in range(1, n):
        s[t] = s[t - 1] if rng.uniform() < P_STAY[s[t - 1]] else 1 - s[t - 1]
    return s


def classify(tc):
    return {k: trial_cluster_labels(f(tc), medfilt_size=0, nan_aware=True)[0]
            for k, f in STRETCH.items()}


if __name__ == '__main__':
    res = {}
    t0 = time.time()
    for s in range(N_SESS):
        rng = np.random.default_rng(500 + s)
        truth = markov(rng)
        other = markov(rng, start=int(rng.integers(0, 2)))
        uncued = rng.uniform(size=N_TRIALS) < .5
        hit = rng.uniform(size=N_TRIALS) < np.where(truth == 1, HIT[1], HIT[0])
        cells = {}
        units = rng.choice(S1.N_UNITS, sum(N_POOL.values()), replace=False)
        i = 0
        for grp, n in N_POOL.items():
            u = units[i:i + n]; i += n
            if grp == 'followers':
                tcs = S1.simulate(u, truth, rng)[0]
                states = np.tile(truth, (n, 1))
            elif grp == 'competing':
                tcs = S1.simulate(u, other, rng)[0]
                states = np.tile(other, (n, 1))
            else:                                   # each cell its own state and offsets
                states = np.array([markov(rng, start=int(rng.integers(0, 2))) for _ in u])
                tcs = np.concatenate([S1.simulate([uu], st, rng)[0] for uu, st in zip(u, states)])
            labs = [classify(tc) for tc in tcs]
            for k in STRETCH:
                res[f's{s}_{grp}_{k}'] = np.array([l[k] for l in labs])
            res[f's{s}_{grp}_state'] = states
            print(f'  session {s}: {grp} classified, {time.time() - t0:.0f}s', flush=True)
        res[f's{s}_truth'], res[f's{s}_other'] = truth, other
        res[f's{s}_uncued'], res[f's{s}_hit'] = uncued, hit
    np.savez_compressed(OUT, n_sess=N_SESS, **res)
    print(f'wrote {OUT}')
