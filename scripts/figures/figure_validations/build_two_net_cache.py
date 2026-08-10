"""
Two-network disconnected-control cache builder (symmetric: two from-scratch Ng=1024 nets).

net 1 = independently-trained Ng=1024, seed 1234
net 2 = independently-trained Ng=1024, seed 5678
Both trained from scratch on the same place-cell targets; driven on the SAME seeded
trajectories, so cross-network cell pairs share position/velocity drive but have ZERO
coupling (block-diagonal ground-truth J).

Baseline = position + speed + head direction (from the shared velocity input).
Expected: ΔpR² = pR2(base+cell) - pR2(base) is elevated WITHIN each net (tracks |J|)
but ~0 ACROSS nets. Resumable: saves after every target.
"""
import sys, os, time
REPO_DIR = '/Users/harryclark/Documents/spatial-manifolds/GRID-PATTERN-FORMATION'
sys.path.insert(0, REPO_DIR)
sys.path.insert(0, '/Users/harryclark/Documents/spatial-manifolds/src')

import numpy as np
import torch
from place_cells import PlaceCells
from trajectory_generator import TrajectoryGenerator
from model import RNN
from visualize import compute_ratemaps
from scores import GridScorer
from spatial_manifolds.mlencoding import MLencoding

# ── Config ────────────────────────────────────────────────────────────────────
SUBSET_SEED = 0
K_PER_NET   = 40      # top grid cells taken from each network
N_BATCHES   = 12      # joint trajectory batches -> T = N_BATCHES*200*20 bins
RES, N_AVG  = 40, 50
N_CV        = 3
DATA_DIR    = '/Users/harryclark/Documents/spatial-manifolds/data/rnn_xgboost'
NETA_DIR    = DATA_DIR + '/second_net_seed1234_Ng1024'   # net 1
NETB_DIR    = DATA_DIR + '/second_net_seed5678_Ng1024'   # net 2
CACHE_FILE  = DATA_DIR + '/pr2_two_network_cache.npz'

DEVICE = 'mps' if torch.backends.mps.is_available() else ('cuda' if torch.cuda.is_available() else 'cpu')
print(f'device={DEVICE}', flush=True)

def build_net(ckpt_dir):
    class Options: pass
    o = Options()
    o.Np=512; o.Ng=1024; o.sequence_length=20; o.batch_size=200
    o.learning_rate=1e-4; o.weight_decay=1e-4; o.place_cell_rf=0.12; o.surround_scale=2
    o.RNN_type='RNN'; o.activation='relu'; o.DoG=True; o.periodic=False
    o.box_width=2.2; o.box_height=2.2; o.device=DEVICE
    pc = PlaceCells(o)
    pc.us = torch.tensor(np.load(os.path.join(REPO_DIR,'models','example_pc_centers.npy')),
                         dtype=torch.float32).to(DEVICE)
    m = RNN(o, pc).to(DEVICE)
    m.load_state_dict(torch.load(os.path.join(ckpt_dir,'ckpt.pth'), map_location=DEVICE)['model'])
    m.eval()
    return o, pc, m

o1, place_cells, model  = build_net(NETA_DIR)
o2, place_cells2, model2 = build_net(NETB_DIR)
trajectory_generator = TrajectoryGenerator(o1, place_cells)   # shared centers/trajectories

coord_range = ((-o1.box_width/2, o1.box_width/2), (-o1.box_height/2, o1.box_height/2))
scorer = GridScorer(RES, coord_range, zip([0.2]*10, np.linspace(0.4,1.0,10).tolist()))

def grid_scores(mdl, opts):
    a, _, _, _ = compute_ratemaps(mdl, trajectory_generator, opts, res=RES, n_avg=N_AVG, Ng=opts.Ng)
    return np.array([scorer.get_scores(a[ni])[0] for ni in range(opts.Ng)])

# ── Select top-K grid cells from each net (seeded) ────────────────────────────
torch.manual_seed(SUBSET_SEED); np.random.seed(SUBSET_SEED)
sc1 = grid_scores(model,  o1)
sc2 = grid_scores(model2, o2)
sel1 = np.argsort(sc1)[::-1][:K_PER_NET].astype(int)
sel2 = np.argsort(sc2)[::-1][:K_PER_NET].astype(int)
print(f'net1(seed1234) selected grid scores: {sc1[sel1].min():.2f}-{sc1[sel1].max():.2f} | '
      f'net2(seed5678): {sc2[sel2].min():.2f}-{sc2[sel2].max():.2f}', flush=True)

# ── Joint activation time series on SHARED trajectories (re-seed) ─────────────
torch.manual_seed(SUBSET_SEED+1); np.random.seed(SUBSET_SEED+1)
G1L, G2L, PL, VL = [], [], [], []
with torch.no_grad():
    for _ in range(N_BATCHES):
        inp, pos_b, _ = trajectory_generator.get_test_batch()
        g1 = model.g(inp).cpu().numpy().transpose(1,0,2).reshape(-1, o1.Ng)
        g2 = model2.g(inp).cpu().numpy().transpose(1,0,2).reshape(-1, o2.Ng)
        p  = pos_b.cpu().numpy().transpose(1,0,2).reshape(-1, 2)
        v  = inp[0].cpu().numpy().transpose(1,0,2).reshape(-1, 2)
        G1L.append(g1[:, sel1]); G2L.append(g2[:, sel2]); PL.append(p); VL.append(v)
G_comb = np.concatenate([np.concatenate(G1L,0), np.concatenate(G2L,0)], axis=1).astype(np.float32)
POS    = np.concatenate(PL, 0).astype(np.float32)
VEL    = np.concatenate(VL, 0).astype(np.float32)
speed  = np.linalg.norm(VEL, axis=1)
hd     = np.arctan2(VEL[:, 1], VEL[:, 0])
pos_n    = (POS   - POS.min(0))  / (POS.max(0)  - POS.min(0)  + 1e-8)
speed_n  = (speed - speed.min()) / (speed.max() - speed.min() + 1e-8)
baseline = np.column_stack([pos_n, speed_n, np.cos(hd), np.sin(hd)]).astype(np.float32)
BASE_COLS = np.array(['pos_x', 'pos_y', 'speed', 'cos_hd', 'sin_hd'])
T = len(G_comb); K = K_PER_NET; N = 2 * K
print(f'combined population: {N} units ({K}+{K}) | T={T:,} bins | '
      f'baseline={list(BASE_COLS)} | speed range {speed.min():.4f}-{speed.max():.4f}', flush=True)

# ── Block-diagonal ground-truth connectivity ─────────────────────────────────
J1 = model.RNN.weight_hh_l0.detach().cpu().numpy()[np.ix_(sel1, sel1)]
J2 = model2.RNN.weight_hh_l0.detach().cpu().numpy()[np.ix_(sel2, sel2)]
J_comb = np.zeros((N, N), np.float32)
J_comb[:K, :K] = J1; J_comb[K:, K:] = J2
net_id  = np.array([0]*K + [1]*K, dtype=int)
grid_sc = np.concatenate([sc1[sel1], sc2[sel2]]).astype(np.float32)

# ── Load / init cache ─────────────────────────────────────────────────────────
if os.path.exists(CACHE_FILE):
    c = np.load(CACHE_FILE)
    pr2_base = c['pr2_base'].copy(); pr2_cell = c['pr2_cell'].copy()
    pr2_full = c['pr2_full'].copy(); completed = c['completed'].copy()
    print(f'resumed: {int(completed.sum())}/{N} targets done', flush=True)
else:
    pr2_base = np.full((N,),   np.nan, np.float32)
    pr2_cell = np.full((N, N), np.nan, np.float32)
    pr2_full = np.full((N, N), np.nan, np.float32)
    completed = np.zeros(N, dtype=bool)

def save():
    np.savez(CACHE_FILE, pr2_base=pr2_base, pr2_cell=pr2_cell, pr2_full=pr2_full,
             completed=completed, J_comb=J_comb, net_id=net_id, grid_sc=grid_sc,
             sel1=sel1, sel2=sel2, K=K, base_cols=BASE_COLS,
             net_seeds=np.array([1234, 5678]))

save()

xgb = MLencoding(tunemodel='xgboost', cov_history=True, spike_history=False,
                 window=20, n_filters=5, max_time=200)

inc = np.where(~completed)[0]
if len(inc):
    tb = time.time(); ti_b = int(inc[0]); y_b = G_comb[:, ti_b]
    xgb.fit_cv(baseline, y_b, verbose=0, continuous_folds=True, n_cv=N_CV)
    for ci in range(min(3, N-1)):
        cb = ci if ci != ti_b else ci+1
        cov = G_comb[:, cb][:, None]
        xgb.fit_cv(cov, y_b, verbose=0, continuous_folds=True, n_cv=N_CV)
        xgb.fit_cv(np.column_stack([baseline, cov]), y_b, verbose=0, continuous_folds=True, n_cv=N_CV)
    t_tgt = (time.time()-tb) * (N-1) / 3
    print(f'ETA ~{t_tgt*len(inc)/3600:.1f} h ({t_tgt/60:.1f} min/target x {len(inc)})', flush=True)

t0 = time.time()
for ti in range(N):
    if completed[ti]: continue
    y = G_comb[:, ti]
    _, p_base = xgb.fit_cv(baseline, y, verbose=0, continuous_folds=True, n_cv=N_CV)
    pr2_base[ti] = float(np.nanmean(p_base))
    for ci in range(N):
        if ci == ti: continue
        cov = G_comb[:, ci][:, None]
        _, p_cell = xgb.fit_cv(cov, y, verbose=0, continuous_folds=True, n_cv=N_CV)
        _, p_full = xgb.fit_cv(np.column_stack([baseline, cov]), y, verbose=0, continuous_folds=True, n_cv=N_CV)
        pr2_cell[ci, ti] = float(np.nanmean(p_cell))
        pr2_full[ci, ti] = float(np.nanmean(p_full))
    completed[ti] = True
    save()
    d = pr2_full - pr2_base[None, :]
    within = net_id[:, None] == net_id[None, :]
    dw = np.nanmean(d[within & ~np.eye(N, dtype=bool)])
    dx = np.nanmean(d[~within])
    print(f'[{int(completed.sum())}/{N}] target {ti} (net{net_id[ti]})  '
          f'{(time.time()-t0)/60:.1f} min  mean dpR2 within={dw:.4f} cross={dx:.4f}', flush=True)

print('DONE' if completed.all() else f'{int(completed.sum())}/{N} done', flush=True)
