"""Cache 2D open-field rate maps for a handful of net7 units.

fig1_supp_classifier needs them for the panel that shows the 1D track as a slice
through the 2D environment -- the point being that a grid unit's periodic firing
along the track is a chord through its hexagonal field arrangement, not a
separately specified 1D property. The simulation cache (sim_cache.npz) stores the
1D trial x position tuning curves and the grid scores but not the 2D maps, so
they are recomputed here once and cached, keeping torch out of the figure script.

Units cached: the top grid-score units (the ones the slice was chosen to cross
~3 fields of) plus a few near-zero-grid-score units for contrast.
"""
import os
import sys

import numpy as np
import torch

REPO = '/Users/harryclark/Documents/spatial-manifolds/GRID-PATTERN-FORMATION'
sys.path.insert(0, REPO)
sys.path.insert(0, '/Users/harryclark/Documents/spatial-manifolds/src')
from model import RNN
from place_cells import PlaceCells
from trajectory_generator import TrajectoryGenerator
from visualize import compute_ratemaps

NET = '/Users/harryclark/Documents/spatial-manifolds/data/rnn_xgboost/survey_s7_Ng1024'
CACHE = '/Users/harryclark/Documents/spatial-manifolds/data/rnn_xgboost/net7_1D_track_sim'
OUT = f'{CACHE}/of_ratemaps_cache.npz'
RES, N_AVG = 50, 250

DEVICE = ('mps' if torch.backends.mps.is_available()
          else 'cuda' if torch.cuda.is_available() else 'cpu')


class Options:
    pass


o = Options()
o.Np, o.Ng, o.sequence_length, o.batch_size = 512, 1024, 20, 200
o.learning_rate, o.weight_decay = 1e-4, 1e-4
o.place_cell_rf, o.surround_scale = 0.12, 2
o.pc_rf_seed = 55
o.RNN_type, o.activation, o.DoG, o.periodic = 'RNN', 'relu', True, False
o.box_width = o.box_height = 2.2
o.device = DEVICE

pc = PlaceCells(o)
pc.us = torch.tensor(np.load(f'{NET}/place_cells.npy'),
                     dtype=torch.float32).to(DEVICE)
model = RNN(o, pc).to(DEVICE)
model.load_state_dict(torch.load(f'{NET}/ckpt.pth',
                                 map_location=DEVICE)['model'])
model.eval()
tg = TrajectoryGenerator(o, pc)

gs = np.load(f'{NET}/multiscale_scores.npz')['grid_scores']
grid_order = np.argsort(gs)[::-1]
# The slice was chosen against the top-4 grid units, so those are the ones whose
# fields it actually crosses -- cache them, plus near-zero-score units for
# contrast, plus WALKTHROUGH_CANDIDATES.
#
# The walkthrough candidates matter. The top-scoring units (23, 290, 318, 596)
# go nearly silent during the blind condition, so their C-block trials fall below
# the classifier's activity floor and are force-labelled non-anchored before the
# correlation step is reached. They classify perfectly, but they classify for the
# wrong reason to illustrate with: the trial x trial correlation, the 2-means
# split and the null gate never decide anything. The candidates below keep firing
# through the blind block and are decided by decorrelation, which is the
# mechanism the figure is meant to show.
WALKTHROUGH_CANDIDATES = [413, 415, 403, 417, 865, 869]
units = np.concatenate([grid_order[:4], np.argsort(np.abs(gs))[:4],
                        WALKTHROUGH_CANDIDATES]).astype(int)
units = np.unique(units)
print(f'device={DEVICE}; caching {len(units)} units at res={RES}, n_avg={N_AVG}')
print('grid scores:', {int(u): round(float(gs[u]), 3) for u in units})

rm, _, _, _ = compute_ratemaps(model, tg, o, res=RES, n_avg=N_AVG,
                               Ng=len(units), idxs=units)
rm = np.asarray(rm).reshape(len(units), RES, RES)

np.savez_compressed(OUT, ratemaps=rm, units=units, grid_scores=gs[units],
                    res=RES, box_width=o.box_width)
print(f'wrote {OUT}  {rm.shape}')
