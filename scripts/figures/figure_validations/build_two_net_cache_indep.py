"""
INDEPENDENT-trajectory control (net 1 + net 7). Same cells as the 1+7 run, but the two
nets are driven on DIFFERENT trajectories. Cross-net pairs then share NOTHING (different
positions at each time bin), so if the 1+7 cross floor (0.035) was shared-input leakage
it should collapse to ~0 here; within-net coupling is preserved (each net on its own traj).
Definitive test that the cross floor is common-input leakage, not coupling/estimator bias.
"""
import sys, os, time
REPO_DIR='/Users/harryclark/Documents/spatial-manifolds/GRID-PATTERN-FORMATION'
sys.path.insert(0, REPO_DIR); sys.path.insert(0,'/Users/harryclark/Documents/spatial-manifolds/src')
import numpy as np, torch
from place_cells import PlaceCells
from trajectory_generator import TrajectoryGenerator
from model import RNN
from spatial_manifolds.mlencoding import MLencoding

N_BATCHES=12; N_CV=3
DATA='/Users/harryclark/Documents/spatial-manifolds/data/rnn_xgboost'
NETA=DATA+'/diffPC_A_Ng1024'; NETB=DATA+'/survey_s7_Ng1024'
CACHE17=DATA+'/pr2_two_net_17_cache.npz'; CACHE=DATA+'/pr2_two_net_indep_cache.npz'
DEVICE='mps' if torch.backends.mps.is_available() else ('cuda' if torch.cuda.is_available() else 'cpu')
print(f'device={DEVICE}', flush=True)

# same cell selection + J as the 1+7 shared-trajectory run
c17=np.load(CACHE17,allow_pickle=True)
sel1=c17['sel1']; sel2=c17['sel2']; J_comb=c17['J_comb']; net_id=c17['net_id']; K=int(c17['K']); N=len(net_id)

def build_net(d):
    class O: pass
    o=O(); o.Np=512;o.Ng=1024;o.sequence_length=20;o.batch_size=200
    o.learning_rate=1e-4;o.weight_decay=1e-4;o.place_cell_rf=0.12;o.surround_scale=2
    o.RNN_type='RNN';o.activation='relu';o.DoG=True;o.periodic=False
    o.box_width=2.2;o.box_height=2.2;o.device=DEVICE
    pc=PlaceCells(o); pc.us=torch.tensor(np.load(d+'/place_cells.npy'),dtype=torch.float32).to(DEVICE)
    m=RNN(o,pc).to(DEVICE); m.load_state_dict(torch.load(d+'/ckpt.pth',map_location=DEVICE)['model']); m.eval()
    return o,pc,m
o1,pcA,mA=build_net(NETA); o2,pcB,mB=build_net(NETB)
tgA=TrajectoryGenerator(o1,pcA); tgB=TrajectoryGenerator(o2,pcB)

def gen(m,o,tg,sel,seed):
    torch.manual_seed(seed); np.random.seed(seed)
    GL,PL,VL=[],[],[]
    with torch.no_grad():
        for _ in range(N_BATCHES):
            inp,pos_b,_=tg.get_test_batch()
            GL.append(m.g(inp).cpu().numpy().transpose(1,0,2).reshape(-1,o.Ng)[:,sel])
            PL.append(pos_b.cpu().numpy().transpose(1,0,2).reshape(-1,2))
            VL.append(inp[0].cpu().numpy().transpose(1,0,2).reshape(-1,2))
    return (np.concatenate(GL,0).astype(np.float32),np.concatenate(PL,0),np.concatenate(VL,0))

# INDEPENDENT trajectories: different seeds
G1,POSa,VELa = gen(mA,o1,tgA,sel1,seed=100)
G7,POSb,VELb = gen(mB,o2,tgB,sel2,seed=200)
G_comb=np.concatenate([G1,G7],axis=1).astype(np.float32); T=len(G_comb)

def make_baseline(POS,VEL):
    sp=np.linalg.norm(VEL,axis=1); hd=np.arctan2(VEL[:,1],VEL[:,0])
    pn=(POS-POS.min(0))/(POS.max(0)-POS.min(0)+1e-8); sn=(sp-sp.min())/(sp.max()-sp.min()+1e-8)
    return np.column_stack([pn,sn,np.cos(hd),np.sin(hd)]).astype(np.float32)
baseA=make_baseline(POSa,VELa); baseB=make_baseline(POSb,VELb)
print(f'INDEPENDENT trajectories | {N} units | T={T:,} | net1 seed100, net7 seed200', flush=True)

if os.path.exists(CACHE):
    c=np.load(CACHE); pr2_base=c['pr2_base'].copy();pr2_cell=c['pr2_cell'].copy();pr2_full=c['pr2_full'].copy();completed=c['completed'].copy()
    print(f'resumed {int(completed.sum())}/{N}', flush=True)
else:
    pr2_base=np.full((N,),np.nan,np.float32); pr2_cell=np.full((N,N),np.nan,np.float32)
    pr2_full=np.full((N,N),np.nan,np.float32); completed=np.zeros(N,dtype=bool)
def save():
    np.savez(CACHE,pr2_base=pr2_base,pr2_cell=pr2_cell,pr2_full=pr2_full,completed=completed,
             J_comb=J_comb,net_id=net_id,K=K,sel1=sel1,sel2=sel2)
save()
xgb=MLencoding(tunemodel='xgboost',cov_history=True,spike_history=False,window=20,n_filters=5,max_time=200)

t0=time.time()
for ti in range(N):
    if completed[ti]: continue
    base = baseA if net_id[ti]==0 else baseB     # each target uses ITS net's trajectory position
    y=G_comb[:,ti]
    _,pb=xgb.fit_cv(base,y,verbose=0,continuous_folds=True,n_cv=N_CV); pr2_base[ti]=float(np.nanmean(pb))
    for ci in range(N):
        if ci==ti: continue
        cov=G_comb[:,ci][:,None]
        _,pcl=xgb.fit_cv(cov,y,verbose=0,continuous_folds=True,n_cv=N_CV)
        _,pfl=xgb.fit_cv(np.column_stack([base,cov]),y,verbose=0,continuous_folds=True,n_cv=N_CV)
        pr2_cell[ci,ti]=float(np.nanmean(pcl)); pr2_full[ci,ti]=float(np.nanmean(pfl))
    completed[ti]=True; save()
    d=pr2_full-pr2_base[None,:]; within=net_id[:,None]==net_id[None,:]
    dw=np.nanmean(d[within & ~np.eye(N,dtype=bool)]); dx=np.nanmean(d[~within])
    print(f'[{int(completed.sum())}/{N}] target {ti} (net{net_id[ti]})  {(time.time()-t0)/60:.1f} min  '
          f'within={dw:.4f} cross={dx:.4f}', flush=True)
print('DONE' if completed.all() else f'{int(completed.sum())}/{N} done', flush=True)
