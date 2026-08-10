"""
Two-network disconnected control — DIFFERENT place-cell tilings variant.

net A (seed 1234, pc_seed 101) and net B (seed 5678, pc_seed 202): both Ng=1024, trained
from scratch, each targeting its OWN random place-cell population. Driven on the SAME
trajectories (shared self-motion / true position) but each net uses its own place cells for
the initial state. Cross-network pairs still share the common input (position) but have zero
coupling AND less representational alignment -> tests whether de-aligning the target basis
lowers the cross-net ΔpR² floor.
"""
import sys, os, time
REPO_DIR='/Users/harryclark/Documents/spatial-manifolds/GRID-PATTERN-FORMATION'
sys.path.insert(0, REPO_DIR); sys.path.insert(0, '/Users/harryclark/Documents/spatial-manifolds/src')
import numpy as np, torch
from place_cells import PlaceCells
from trajectory_generator import TrajectoryGenerator
from model import RNN
from visualize import compute_ratemaps
from scores import GridScorer
from spatial_manifolds.mlencoding import MLencoding

SUBSET_SEED=0; K_PER_NET=40; N_BATCHES=12; RES,N_AVG=40,50; N_CV=3
DATA='/Users/harryclark/Documents/spatial-manifolds/data/rnn_xgboost'
NETA=DATA+'/diffPC_A_Ng1024'; NETB=DATA+'/survey_s7_Ng1024'
CACHE=DATA+'/pr2_two_net_17_cache.npz'
DEVICE='mps' if torch.backends.mps.is_available() else ('cuda' if torch.cuda.is_available() else 'cpu')
print(f'device={DEVICE}', flush=True)

def build_net(ckpt_dir):
    class Options: pass
    o=Options()
    o.Np=512;o.Ng=1024;o.sequence_length=20;o.batch_size=200
    o.learning_rate=1e-4;o.weight_decay=1e-4;o.place_cell_rf=0.12;o.surround_scale=2
    o.RNN_type='RNN';o.activation='relu';o.DoG=True;o.periodic=False
    o.box_width=2.2;o.box_height=2.2;o.device=DEVICE
    pc=PlaceCells(o)
    pc.us=torch.tensor(np.load(os.path.join(ckpt_dir,'place_cells.npy')),dtype=torch.float32).to(DEVICE)
    m=RNN(o,pc).to(DEVICE); m.load_state_dict(torch.load(os.path.join(ckpt_dir,'ckpt.pth'),map_location=DEVICE)['model']); m.eval()
    return o,pc,m

o1,pcA,model  = build_net(NETA)
o2,pcB,model2 = build_net(NETB)
tgA=TrajectoryGenerator(o1,pcA); tgB=TrajectoryGenerator(o2,pcB)
coord=((-o1.box_width/2,o1.box_width/2),(-o1.box_height/2,o1.box_height/2))
scorer=GridScorer(RES,coord,zip([0.2]*10,np.linspace(0.4,1.0,10).tolist()))

def grid_scores(mdl,opts,tg):
    a,_,_,_=compute_ratemaps(mdl,tg,opts,res=RES,n_avg=N_AVG,Ng=opts.Ng)
    return np.array([scorer.get_scores(a[i])[0] for i in range(opts.Ng)])

def joint_batch(tg,o):
    """Shared trajectory; each net gets its own place-cell-based initial state."""
    traj=tg.generate_trajectory(o.box_width,o.box_height,o.batch_size)
    v=np.stack([traj['ego_v']*np.cos(traj['target_hd']),traj['ego_v']*np.sin(traj['target_hd'])],axis=-1)
    v=torch.tensor(v,dtype=torch.float32).transpose(0,1).to(DEVICE)
    pos=np.stack([traj['target_x'],traj['target_y']],axis=-1)
    pos=torch.tensor(pos,dtype=torch.float32).transpose(0,1).to(DEVICE)
    ip=np.stack([traj['init_x'],traj['init_y']],axis=-1)
    ip=torch.tensor(ip,dtype=torch.float32).to(DEVICE)
    iaA=pcA.get_activation(ip).squeeze(); iaB=pcB.get_activation(ip).squeeze()
    return (v,iaA),(v,iaB),pos

# ── select K grid cells per net — UNBIASED w.r.t. connectivity ────────────────
# Random sample among grid cells (score>GC_THRESH), so the within-net |J| range
# is representative, not cherry-picked (in real data you can't know the strong
# connections ahead of time). Seeded for reproducibility.
torch.manual_seed(SUBSET_SEED); np.random.seed(SUBSET_SEED)
sc1=grid_scores(model,o1,tgA); sc2=grid_scores(model2,o2,tgB)
GC_THRESH=0.3
Jf1=model.RNN.weight_hh_l0.detach().cpu().numpy(); Jf2=model2.RNN.weight_hh_l0.detach().cpu().numpy()
def select_unbiased(sc,K,rng):
    gc=np.where(sc>GC_THRESH)[0]
    return np.sort(rng.choice(gc, size=min(K,len(gc)), replace=False)).astype(int)
_rng=np.random.RandomState(0)
sel1=select_unbiased(sc1,K_PER_NET,_rng); sel2=select_unbiased(sc2,K_PER_NET,_rng)
print(f'netA(net1:1234/pc101) {len(np.where(sc1>GC_THRESH)[0])} GCs, random {len(sel1)} selected, '
      f'mean|J|={np.abs(Jf1[np.ix_(sel1,sel1)]).mean():.4f} | '
      f'netB(net7:5555/pc55) {len(np.where(sc2>GC_THRESH)[0])} GCs, random {len(sel2)} selected, '
      f'mean|J|={np.abs(Jf2[np.ix_(sel2,sel2)]).mean():.4f}', flush=True)

# ── joint activations on shared trajectories ─────────────────────────────────
torch.manual_seed(SUBSET_SEED+1); np.random.seed(SUBSET_SEED+1)
G1L,G2L,PL,VL=[],[],[],[]
with torch.no_grad():
    for _ in range(N_BATCHES):
        inpA,inpB,pos_b=joint_batch(tgA,o1)
        g1=model.g(inpA).cpu().numpy().transpose(1,0,2).reshape(-1,o1.Ng)
        g2=model2.g(inpB).cpu().numpy().transpose(1,0,2).reshape(-1,o2.Ng)
        p=pos_b.cpu().numpy().transpose(1,0,2).reshape(-1,2)
        vv=inpA[0].cpu().numpy().transpose(1,0,2).reshape(-1,2)
        G1L.append(g1[:,sel1]); G2L.append(g2[:,sel2]); PL.append(p); VL.append(vv)
G_comb=np.concatenate([np.concatenate(G1L,0),np.concatenate(G2L,0)],axis=1).astype(np.float32)
POS=np.concatenate(PL,0).astype(np.float32); VEL=np.concatenate(VL,0).astype(np.float32)
speed=np.linalg.norm(VEL,axis=1); hd=np.arctan2(VEL[:,1],VEL[:,0])
pos_n=(POS-POS.min(0))/(POS.max(0)-POS.min(0)+1e-8); speed_n=(speed-speed.min())/(speed.max()-speed.min()+1e-8)
baseline=np.column_stack([pos_n,speed_n,np.cos(hd),np.sin(hd)]).astype(np.float32)
T=len(G_comb); K=K_PER_NET; N=2*K
print(f'combined: {N} units ({K}+{K}) | T={T:,} bins | baseline=pos+speed+hd | speed {speed.min():.4f}-{speed.max():.4f}', flush=True)

J1=model.RNN.weight_hh_l0.detach().cpu().numpy()[np.ix_(sel1,sel1)]
J2=model2.RNN.weight_hh_l0.detach().cpu().numpy()[np.ix_(sel2,sel2)]
J_comb=np.zeros((N,N),np.float32); J_comb[:K,:K]=J1; J_comb[K:,K:]=J2
net_id=np.array([0]*K+[1]*K,dtype=int); grid_sc=np.concatenate([sc1[sel1],sc2[sel2]]).astype(np.float32)

if os.path.exists(CACHE):
    c=np.load(CACHE); pr2_base=c['pr2_base'].copy();pr2_cell=c['pr2_cell'].copy();pr2_full=c['pr2_full'].copy();completed=c['completed'].copy()
    print(f'resumed: {int(completed.sum())}/{N}', flush=True)
else:
    pr2_base=np.full((N,),np.nan,np.float32); pr2_cell=np.full((N,N),np.nan,np.float32)
    pr2_full=np.full((N,N),np.nan,np.float32); completed=np.zeros(N,dtype=bool)

def save():
    np.savez(CACHE,pr2_base=pr2_base,pr2_cell=pr2_cell,pr2_full=pr2_full,completed=completed,
             J_comb=J_comb,net_id=net_id,grid_sc=grid_sc,sel1=sel1,sel2=sel2,K=K,
             base_cols=np.array(['pos_x','pos_y','speed','cos_hd','sin_hd']),net_seeds=np.array([1234,5555]))
save()
xgb=MLencoding(tunemodel='xgboost',cov_history=True,spike_history=False,window=20,n_filters=5,max_time=200)

inc=np.where(~completed)[0]
if len(inc):
    tb=time.time(); tib=int(inc[0]); yb=G_comb[:,tib]
    xgb.fit_cv(baseline,yb,verbose=0,continuous_folds=True,n_cv=N_CV)
    for ci in range(min(3,N-1)):
        cb=ci if ci!=tib else ci+1; cov=G_comb[:,cb][:,None]
        xgb.fit_cv(cov,yb,verbose=0,continuous_folds=True,n_cv=N_CV)
        xgb.fit_cv(np.column_stack([baseline,cov]),yb,verbose=0,continuous_folds=True,n_cv=N_CV)
    tt=(time.time()-tb)*(N-1)/3
    print(f'ETA ~{tt*len(inc)/3600:.1f} h ({tt/60:.1f} min/target x {len(inc)})', flush=True)

t0=time.time()
for ti in range(N):
    if completed[ti]: continue
    y=G_comb[:,ti]
    _,pb=xgb.fit_cv(baseline,y,verbose=0,continuous_folds=True,n_cv=N_CV); pr2_base[ti]=float(np.nanmean(pb))
    for ci in range(N):
        if ci==ti: continue
        cov=G_comb[:,ci][:,None]
        _,pcl=xgb.fit_cv(cov,y,verbose=0,continuous_folds=True,n_cv=N_CV)
        _,pfl=xgb.fit_cv(np.column_stack([baseline,cov]),y,verbose=0,continuous_folds=True,n_cv=N_CV)
        pr2_cell[ci,ti]=float(np.nanmean(pcl)); pr2_full[ci,ti]=float(np.nanmean(pfl))
    completed[ti]=True; save()
    d=pr2_full-pr2_base[None,:]; within=net_id[:,None]==net_id[None,:]
    dw=np.nanmean(d[within & ~np.eye(N,dtype=bool)]); dx=np.nanmean(d[~within])
    print(f'[{int(completed.sum())}/{N}] target {ti} (net{net_id[ti]})  {(time.time()-t0)/60:.1f} min  '
          f'mean dpR2 within={dw:.4f} cross={dx:.4f}', flush=True)
print('DONE' if completed.all() else f'{int(completed.sum())}/{N} done', flush=True)
