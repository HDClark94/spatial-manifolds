"""Pairwise XGBoost coupling (raw pR2 and delta pR2) for units WITHIN the
merged net7 dominance-test network (narrow-origin group 0 + wide-origin
group 1, single coupled Ng=2048 RNN, 50k-step joint fine-tune already done).

Adapted from build_two_net_cache_67.py, which ran the same assay across two
INDEPENDENT (zero cross-coupling) nets as a disconnected control. Here there
is only one model -- units are pulled from a single forward pass, and the
within/cross group split (ng_group: 0=narrow-origin, 1=wide-origin) tests
whether the two original grid scales remain functionally distinguishable
after being fused and fine-tuned together, despite the spacing drift already
seen (docs: spacing_before_after_dominance.png).
"""
import sys, os, time
REPO_DIR='/Users/harryclark/Documents/spatial-manifolds/GRID-PATTERN-FORMATION'
sys.path.insert(0, REPO_DIR); sys.path.insert(0, '/Users/harryclark/Documents/spatial-manifolds/src')
import numpy as np, torch
from place_cells import PlaceCells
from trajectory_generator import TrajectoryGenerator
from model import RNN
from spatial_manifolds.mlencoding import MLencoding

SUBSET_SEED=0; K_PER_GROUP=50; N_BATCHES=12; N_CV=3
DATA='/Users/harryclark/Documents/spatial-manifolds/data/rnn_xgboost'
NET=DATA+'/merged_net7_dominance_Ng2048'
CACHE=DATA+'/pr2_merged_net7_cache.npz'
DEVICE='mps' if torch.backends.mps.is_available() else ('cuda' if torch.cuda.is_available() else 'cpu')
print(f'device={DEVICE}', flush=True)

class Options: pass
o=Options()
o.Np=1024; o.Ng=2048; o.sequence_length=20; o.batch_size=200
o.learning_rate=1e-4; o.weight_decay=1e-4; o.place_cell_rf=[0.12,0.1944]; o.surround_scale=2
o.pc_rf_seed=0
o.RNN_type='RNN'; o.activation='relu'; o.DoG=True; o.periodic=False
o.box_width=2.2; o.box_height=2.2; o.device=DEVICE

pc = PlaceCells(o)
pc.us = torch.tensor(np.load(os.path.join(NET,'place_cells.npy')), dtype=torch.float32).to(DEVICE)
sigma_group = np.load(os.path.join(NET,'sigma_group.npy'))
pc.sigma_group = sigma_group
widths = np.array([0.12,0.1944], dtype=np.float32)
pc.sigma = torch.tensor(widths[sigma_group], dtype=torch.float32, device=DEVICE)

model = RNN(o, pc).to(DEVICE)
model.load_state_dict(torch.load(os.path.join(NET,'ckpt.pth'), map_location=DEVICE)['model'])
model.eval()
tg = TrajectoryGenerator(o, pc)

# ── reuse cached grid scores / ng_group from validate_multiscale.py ──────────
scores_npz = np.load(os.path.join(NET,'multiscale_scores.npz'))
grid_scores = scores_npz['grid_scores']
ng_group = scores_npz['ng_group']
GC_THRESH=0.3

def select_unbiased(sc, ng, group, K, rng):
    gc = np.where((sc > GC_THRESH) & (ng == group))[0]
    return np.sort(rng.choice(gc, size=min(K, len(gc)), replace=False)).astype(int)

_rng = np.random.RandomState(SUBSET_SEED)
sel0 = select_unbiased(grid_scores, ng_group, 0, K_PER_GROUP, _rng)
sel1 = select_unbiased(grid_scores, ng_group, 1, K_PER_GROUP, _rng)
sel = np.concatenate([sel0, sel1])
K0, K1 = len(sel0), len(sel1)
Jfull = model.RNN.weight_hh_l0.detach().cpu().numpy()
print(f'group0(narrow-origin) {int(((grid_scores>GC_THRESH)&(ng_group==0)).sum())} GCs, {K0} selected, '
      f'mean|J| within={np.abs(Jfull[np.ix_(sel0,sel0)]).mean():.4f} | '
      f'group1(wide-origin) {int(((grid_scores>GC_THRESH)&(ng_group==1)).sum())} GCs, {K1} selected, '
      f'mean|J| within={np.abs(Jfull[np.ix_(sel1,sel1)]).mean():.4f} | '
      f'mean|J| cross={np.abs(Jfull[np.ix_(sel0,sel1)]).mean():.4f}', flush=True)

# ── single-model joint activations on generated trajectories ────────────────
torch.manual_seed(SUBSET_SEED+1); np.random.seed(SUBSET_SEED+1)
GL,PL,VL=[],[],[]
with torch.no_grad():
    for _ in range(N_BATCHES):
        traj = tg.generate_trajectory(o.box_width, o.box_height, o.batch_size)
        v = np.stack([traj['ego_v']*np.cos(traj['target_hd']), traj['ego_v']*np.sin(traj['target_hd'])], axis=-1)
        v = torch.tensor(v, dtype=torch.float32).transpose(0,1).to(DEVICE)
        pos = np.stack([traj['target_x'], traj['target_y']], axis=-1)
        pos = torch.tensor(pos, dtype=torch.float32).transpose(0,1).to(DEVICE)
        ip = np.stack([traj['init_x'], traj['init_y']], axis=-1)
        ip = torch.tensor(ip, dtype=torch.float32).to(DEVICE)
        ia = pc.get_activation(ip).squeeze()
        g = model.g((v, ia)).cpu().numpy().transpose(1,0,2).reshape(-1, o.Ng)
        p = pos.cpu().numpy().transpose(1,0,2).reshape(-1,2)
        vv = v.cpu().numpy().transpose(1,0,2).reshape(-1,2)
        GL.append(g[:, sel]); PL.append(p); VL.append(vv)
G_comb = np.concatenate(GL,0).astype(np.float32)
POS = np.concatenate(PL,0).astype(np.float32); VEL = np.concatenate(VL,0).astype(np.float32)
speed = np.linalg.norm(VEL,axis=1); hd = np.arctan2(VEL[:,1],VEL[:,0])
pos_n = (POS-POS.min(0))/(POS.max(0)-POS.min(0)+1e-8); speed_n=(speed-speed.min())/(speed.max()-speed.min()+1e-8)
baseline = np.column_stack([pos_n, speed_n, np.cos(hd), np.sin(hd)]).astype(np.float32)
T = len(G_comb); N = K0+K1
print(f'combined: {N} units ({K0}+{K1}) | T={T:,} bins | baseline=pos+speed+hd | speed {speed.min():.4f}-{speed.max():.4f}', flush=True)

J_comb = Jfull[np.ix_(sel,sel)]
group_id = np.array([0]*K0 + [1]*K1, dtype=int)
grid_sc_sel = grid_scores[sel].astype(np.float32)

if os.path.exists(CACHE):
    c=np.load(CACHE); pr2_base=c['pr2_base'].copy();pr2_cell=c['pr2_cell'].copy();pr2_full=c['pr2_full'].copy();completed=c['completed'].copy()
    print(f'resumed: {int(completed.sum())}/{N}', flush=True)
else:
    pr2_base=np.full((N,),np.nan,np.float32); pr2_cell=np.full((N,N),np.nan,np.float32)
    pr2_full=np.full((N,N),np.nan,np.float32); completed=np.zeros(N,dtype=bool)

def save():
    np.savez(CACHE, pr2_base=pr2_base, pr2_cell=pr2_cell, pr2_full=pr2_full, completed=completed,
             J_comb=J_comb, group_id=group_id, grid_sc=grid_sc_sel, sel=sel, K0=K0, K1=K1,
             base_cols=np.array(['pos_x','pos_y','speed','cos_hd','sin_hd']))
save()
xgb = MLencoding(tunemodel='xgboost', cov_history=True, spike_history=False, window=20, n_filters=5, max_time=200)

inc = np.where(~completed)[0]
if len(inc):
    tb=time.time(); tib=int(inc[0]); yb=G_comb[:,tib]
    xgb.fit_cv(baseline,yb,verbose=0,continuous_folds=True,n_cv=N_CV)
    for ci in range(min(3,N-1)):
        cb = ci if ci!=tib else ci+1; cov=G_comb[:,cb][:,None]
        xgb.fit_cv(cov,yb,verbose=0,continuous_folds=True,n_cv=N_CV)
        xgb.fit_cv(np.column_stack([baseline,cov]),yb,verbose=0,continuous_folds=True,n_cv=N_CV)
    tt=(time.time()-tb)*(N-1)/3
    print(f'ETA ~{tt*len(inc)/3600:.1f} h ({tt/60:.1f} min/target x {len(inc)})', flush=True)

t0=time.time()
for ti in range(N):
    if completed[ti]: continue
    y = G_comb[:,ti]
    _,pb = xgb.fit_cv(baseline,y,verbose=0,continuous_folds=True,n_cv=N_CV); pr2_base[ti]=float(np.nanmean(pb))
    for ci in range(N):
        if ci==ti: continue
        cov = G_comb[:,ci][:,None]
        _,pcl = xgb.fit_cv(cov,y,verbose=0,continuous_folds=True,n_cv=N_CV)
        _,pfl = xgb.fit_cv(np.column_stack([baseline,cov]),y,verbose=0,continuous_folds=True,n_cv=N_CV)
        pr2_cell[ci,ti]=float(np.nanmean(pcl)); pr2_full[ci,ti]=float(np.nanmean(pfl))
    completed[ti]=True; save()
    d = pr2_full-pr2_base[None,:]; within = group_id[:,None]==group_id[None,:]
    dw = np.nanmean(d[within & ~np.eye(N,dtype=bool)]); dx = np.nanmean(d[~within])
    print(f'[{int(completed.sum())}/{N}] target {ti} (group{group_id[ti]})  {(time.time()-t0)/60:.1f} min  '
          f'mean dpR2 within={dw:.4f} cross={dx:.4f}', flush=True)
print('DONE' if completed.all() else f'{int(completed.sum())}/{N} done', flush=True)
