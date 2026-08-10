"""
Two-network control — net7 (grid solution) + net10 (solves task but did NOT form grids: a
different, non-grid solution). RANDOM 40 cells per net (NOT grid-filtered) — sample units
regardless of gridness (per user). Shared trajectory, per-net place-cell init, different place
cells. STANDARD 200 ms covariate history (comparable to diffPC/67/17 pairings).
Both nets: ~5 cm path-integration, near-identical mean|J| — matched task/scale, different
representation. Tests whether the non-grid net shows a distinct within/cross ΔpR² signature,
and whether the maximally dissimilar (grid vs non-grid) pairing gives a low cross floor.
Cache: pr2_two_net_710_cache.npz.
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

SUBSET_SEED=0; K_PER_NET=40; N_BATCHES=12; RES,N_AVG=40,50; N_CV=3; MAX_TIME=200
DATA='/Users/harryclark/Documents/spatial-manifolds/data/rnn_xgboost'
NETA=DATA+'/survey_s7_Ng1024'; NETB=DATA+'/survey_s10_Ng1024'
CACHE=DATA+'/pr2_two_net_710_cache.npz'
DEVICE='mps' if torch.backends.mps.is_available() else ('cuda' if torch.cuda.is_available() else 'cpu')
print(f'device={DEVICE} | MAX_TIME={MAX_TIME}ms (standard) | RANDOM (non-grid) selection', flush=True)

def build_net(ckpt_dir):
    class Options: pass
    o=Options(); o.Np=512;o.Ng=1024;o.sequence_length=20;o.batch_size=200
    o.learning_rate=1e-4;o.weight_decay=1e-4;o.place_cell_rf=0.12;o.surround_scale=2
    o.RNN_type='RNN';o.activation='relu';o.DoG=True;o.periodic=False;o.box_width=2.2;o.box_height=2.2;o.device=DEVICE
    pc=PlaceCells(o); pc.us=torch.tensor(np.load(os.path.join(ckpt_dir,'place_cells.npy')),dtype=torch.float32).to(DEVICE)
    m=RNN(o,pc).to(DEVICE); m.load_state_dict(torch.load(os.path.join(ckpt_dir,'ckpt.pth'),map_location=DEVICE)['model']); m.eval()
    return o,pc,m

o1,pcA,model  = build_net(NETA)
o2,pcB,model2 = build_net(NETB)
tgA=TrajectoryGenerator(o1,pcA); tgB=TrajectoryGenerator(o2,pcB)
coord=((-o1.box_width/2,o1.box_width/2),(-o1.box_height/2,o1.box_height/2))
scorer=GridScorer(RES,coord,zip([0.2]*10,np.linspace(0.4,1.0,10).tolist()))

# rate maps: used only to (a) drop any near-dead unit, (b) grid-score the SELECTED cells for the record
a1,_,_,_=compute_ratemaps(model,tgA,o1,res=RES,n_avg=N_AVG,Ng=o1.Ng)
a2,_,_,_=compute_ratemaps(model2,tgB,o2,res=RES,n_avg=N_AVG,Ng=o2.Ng)

def joint_batch(tg,o):
    traj=tg.generate_trajectory(o.box_width,o.box_height,o.batch_size)
    v=np.stack([traj['ego_v']*np.cos(traj['target_hd']),traj['ego_v']*np.sin(traj['target_hd'])],axis=-1)
    v=torch.tensor(v,dtype=torch.float32).transpose(0,1).to(DEVICE)
    pos=np.stack([traj['target_x'],traj['target_y']],axis=-1); pos=torch.tensor(pos,dtype=torch.float32).transpose(0,1).to(DEVICE)
    ip=np.stack([traj['init_x'],traj['init_y']],axis=-1); ip=torch.tensor(ip,dtype=torch.float32).to(DEVICE)
    return (v,pcA.get_activation(ip).squeeze()),(v,pcB.get_activation(ip).squeeze()),pos

# ── RANDOM selection (NOT grid-filtered): 40 random cells among active units ───
torch.manual_seed(SUBSET_SEED); np.random.seed(SUBSET_SEED)
Jf1=model.RNN.weight_hh_l0.detach().cpu().numpy(); Jf2=model2.RNN.weight_hh_l0.detach().cpu().numpy()
def select_random(a,K,rng):
    std=a.reshape(a.shape[0],-1).std(1)
    active=np.argsort(std)[int(0.1*len(std)):]     # drop least-active 10% (dead units); NO grid filter
    return np.sort(rng.choice(active,size=K,replace=False)).astype(int)
_rng=np.random.RandomState(0)
sel1=select_random(a1,K_PER_NET,_rng); sel2=select_random(a2,K_PER_NET,_rng)
gs1=np.array([scorer.get_scores(a1[i])[0] for i in sel1]); gs2=np.array([scorer.get_scores(a2[i])[0] for i in sel2])
print(f'net7 (grids): {K_PER_NET} RANDOM cells, {(gs1>0.3).sum()}/{K_PER_NET} are grid (score>0.3), mean|J|={np.abs(Jf1[np.ix_(sel1,sel1)]).mean():.4f} | '
      f'net10 (non-grid): {K_PER_NET} RANDOM cells, {(gs2>0.3).sum()}/{K_PER_NET} are grid, mean|J|={np.abs(Jf2[np.ix_(sel2,sel2)]).mean():.4f}', flush=True)

# ── joint activations on shared trajectories ─────────────────────────────────
torch.manual_seed(SUBSET_SEED+1); np.random.seed(SUBSET_SEED+1)
G1L,G2L,PL,VL=[],[],[],[]
with torch.no_grad():
    for _ in range(N_BATCHES):
        inpA,inpB,pos_b=joint_batch(tgA,o1)
        g1=model.g(inpA).cpu().numpy().transpose(1,0,2).reshape(-1,o1.Ng)
        g2=model2.g(inpB).cpu().numpy().transpose(1,0,2).reshape(-1,o2.Ng)
        p=pos_b.cpu().numpy().transpose(1,0,2).reshape(-1,2); vv=inpA[0].cpu().numpy().transpose(1,0,2).reshape(-1,2)
        G1L.append(g1[:,sel1]); G2L.append(g2[:,sel2]); PL.append(p); VL.append(vv)
G_comb=np.concatenate([np.concatenate(G1L,0),np.concatenate(G2L,0)],axis=1).astype(np.float32)
POS=np.concatenate(PL,0).astype(np.float32); VEL=np.concatenate(VL,0).astype(np.float32)
speed=np.linalg.norm(VEL,axis=1); hd=np.arctan2(VEL[:,1],VEL[:,0])
pos_n=(POS-POS.min(0))/(POS.max(0)-POS.min(0)+1e-8); speed_n=(speed-speed.min())/(speed.max()-speed.min()+1e-8)
baseline=np.column_stack([pos_n,speed_n,np.cos(hd),np.sin(hd)]).astype(np.float32)
T=len(G_comb); K=K_PER_NET; N=2*K
print(f'combined: {N} units ({K}+{K}) | T={T:,} bins | baseline=pos+speed+hd', flush=True)

J1=Jf1[np.ix_(sel1,sel1)]; J2=Jf2[np.ix_(sel2,sel2)]
J_comb=np.zeros((N,N),np.float32); J_comb[:K,:K]=J1; J_comb[K:,K:]=J2
net_id=np.array([0]*K+[1]*K,dtype=int); grid_sc=np.concatenate([gs1,gs2]).astype(np.float32)

if os.path.exists(CACHE):
    c=np.load(CACHE); pr2_base=c['pr2_base'].copy();pr2_cell=c['pr2_cell'].copy();pr2_full=c['pr2_full'].copy();completed=c['completed'].copy()
    print(f'resumed: {int(completed.sum())}/{N}', flush=True)
else:
    pr2_base=np.full((N,),np.nan,np.float32); pr2_cell=np.full((N,N),np.nan,np.float32)
    pr2_full=np.full((N,N),np.nan,np.float32); completed=np.zeros(N,dtype=bool)

def save():
    np.savez(CACHE,pr2_base=pr2_base,pr2_cell=pr2_cell,pr2_full=pr2_full,completed=completed,
             J_comb=J_comb,net_id=net_id,grid_sc=grid_sc,sel1=sel1,sel2=sel2,K=K,
             base_cols=np.array(['pos_x','pos_y','speed','cos_hd','sin_hd']),
             net_seeds=np.array(['survey_s7','survey_s10']),max_time=MAX_TIME)
save()
xgb=MLencoding(tunemodel='xgboost',cov_history=True,spike_history=False,window=20,n_filters=5,max_time=MAX_TIME)

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
    d=pr2_full-pr2_base[None,:]
    wl=[(d[np.ix_(net_id==g,net_id==g)][~np.eye((net_id==g).sum(),dtype=bool)]) for g in (0,1)]
    xl=d[np.ix_(net_id==0,net_id==1)]
    print(f'[{int(completed.sum())}/{N}] base={pr2_base[ti]:.3f} | net7 within={np.nanmean(wl[0]):.4f} '
          f'net10 within={np.nanmean(wl[1]):.4f} cross={np.nanmean(xl):.4f} ({(time.time()-t0)/60:.1f} min)', flush=True)
print('NET710 DONE', flush=True)
