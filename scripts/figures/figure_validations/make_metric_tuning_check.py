"""
Does the product ΔpR2·pR2(cell) re-introduce tuning-dependence?
Compare each metric's correlation with rate-map correlation (tuning) vs with |J| (coupling),
on the diffPC pair. ΔpR2 is tuning-orthogonal (~0.10); question is where the product lands.
compute_ratemaps only (no XGBoost) -> won't disturb the running pre7 XGBoost loop.
"""
import sys, os
REPO_DIR='/Users/harryclark/Documents/spatial-manifolds/GRID-PATTERN-FORMATION'
sys.path.insert(0, REPO_DIR); sys.path.insert(0,'/Users/harryclark/Documents/spatial-manifolds/src')
import numpy as np, torch
from scipy.stats import spearmanr
from place_cells import PlaceCells
from trajectory_generator import TrajectoryGenerator
from model import RNN
from visualize import compute_ratemaps
DATA='/Users/harryclark/Documents/spatial-manifolds/data/rnn_xgboost'
NETA=DATA+'/diffPC_A_Ng1024'; NETB=DATA+'/diffPC_B_Ng1024'; CACHE=DATA+'/pr2_two_net_diffPC_cache.npz'
DEVICE='mps' if torch.backends.mps.is_available() else 'cpu'; RES,N_AVG=40,50; EPS=0.02
c=np.load(CACHE,allow_pickle=True)
pr2_base=c['pr2_base']; pr2_cell=c['pr2_cell']; pr2_full=c['pr2_full']; net_id=c['net_id']; J=c['J_comb']; sel1=c['sel1']; sel2=c['sel2']; N=len(net_id)
def build_net(d):
    class O: pass
    o=O(); o.Np=512;o.Ng=1024;o.sequence_length=20;o.batch_size=200
    o.learning_rate=1e-4;o.weight_decay=1e-4;o.place_cell_rf=0.12;o.surround_scale=2
    o.RNN_type='RNN';o.activation='relu';o.DoG=True;o.periodic=False;o.box_width=2.2;o.box_height=2.2;o.device=DEVICE
    pc=PlaceCells(o); pc.us=torch.tensor(np.load(d+'/place_cells.npy'),dtype=torch.float32).to(DEVICE)
    m=RNN(o,pc).to(DEVICE); m.load_state_dict(torch.load(d+'/ckpt.pth',map_location=DEVICE)['model']); m.eval(); return o,pc,m
o1,pcA,mA=build_net(NETA); o2,pcB,mB=build_net(NETB); tgA=TrajectoryGenerator(o1,pcA); tgB=TrajectoryGenerator(o2,pcB)
a1,_,_,_=compute_ratemaps(mA,tgA,o1,res=RES,n_avg=N_AVG,Ng=o1.Ng); a2,_,_,_=compute_ratemaps(mB,tgB,o2,res=RES,n_avg=N_AVG,Ng=o2.Ng)
rmcorr=np.corrcoef(np.concatenate([a1[sel1],a2[sel2]],0).reshape(N,-1))
low=pr2_base<0.3
aj=[];rc=[];B=[];C=[];F=[];wi=[]
for ti in range(N):
    if low[ti]: continue
    for ci in range(N):
        if ci==ti or np.isnan(pr2_full[ci,ti]): continue
        aj.append(abs(float(J[ti,ci]))); rc.append(rmcorr[ci,ti]); wi.append(net_id[ci]==net_id[ti])
        B.append(pr2_base[ti]); C.append(pr2_cell[ci,ti]); F.append(pr2_full[ci,ti])
aj=np.array(aj);rc=np.array(rc);wi=np.array(wi,dtype=bool);B=np.array(B);C=np.array(C);F=np.array(F)
dpr2=F-B; prod=dpr2*np.clip(C,0,None); gm=np.sign(dpr2)*np.sqrt(np.abs(dpr2)*np.clip(C,0,None))
cands={'ΔpR²':dpr2,'pR²(cell)':C,'product Δ·cell':prod,'geomean(Δ,cell)':gm}
print(f'{"metric":16s} | ρ(|J|) within (coupling) | ρ(|rate-map corr|) within (tuning) | ρ(rmcorr signed) within')
for name,v in cands.items():
    m=np.isfinite(v)
    rj=spearmanr(aj[wi&m],v[wi&m])[0]
    rt_abs=spearmanr(np.abs(rc[wi&m]),v[wi&m])[0]
    rt=spearmanr(rc[wi&m],v[wi&m])[0]
    print(f'{name:16s} |   {rj:+.3f}                |   {rt_abs:+.3f}                          |   {rt:+.3f}')
print('DONE')
