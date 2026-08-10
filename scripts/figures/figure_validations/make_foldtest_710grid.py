"""
Subset test of the 'paired-fold' idea: for each source->target pair, run 10-fold CV, collect the
per-fold baseline and baseline+cell pR2, and compare 3 statistics as J estimators:
  ΔpR2 (mean gain) | paired t-stat (mean/SE across folds) | fold-consistency (frac folds improved).
Compare |J|-recovery (within-net Spearman) and within/cross separation. Net7(grid)+net10 (710grid sel).
CAVEAT: CV folds are dependent -> t p-values are anti-conservative; here we use t only as a RANKING.
"""
import sys, os, time
REPO='/Users/harryclark/Documents/spatial-manifolds/GRID-PATTERN-FORMATION'
sys.path.insert(0,REPO); sys.path.insert(0,'/Users/harryclark/Documents/spatial-manifolds/src')
import numpy as np, torch
from scipy.stats import spearmanr
from place_cells import PlaceCells
from trajectory_generator import TrajectoryGenerator
from model import RNN
from spatial_manifolds.mlencoding import MLencoding
DATA='/Users/harryclark/Documents/spatial-manifolds/data/rnn_xgboost'
CACHE=DATA+'/pr2_two_net_710grid_cache.npz'; DEVICE='mps' if torch.backends.mps.is_available() else 'cpu'
N_BATCHES=12; N_CV=10; N_PER_NET=4
c=np.load(CACHE,allow_pickle=True)
J=c['J_comb']; net_id=c['net_id']; sel1=c['sel1']; sel2=c['sel2']; K=int(c['K']); N=len(net_id); pr2_base0=c['pr2_base']
def build(d):
    class O: pass
    o=O(); o.Np=512;o.Ng=1024;o.sequence_length=20;o.batch_size=200
    o.learning_rate=1e-4;o.weight_decay=1e-4;o.place_cell_rf=0.12;o.surround_scale=2
    o.RNN_type='RNN';o.activation='relu';o.DoG=True;o.periodic=False;o.box_width=2.2;o.box_height=2.2;o.device=DEVICE
    pc=PlaceCells(o); pc.us=torch.tensor(np.load(d+'/place_cells.npy'),dtype=torch.float32).to(DEVICE)
    m=RNN(o,pc).to(DEVICE); m.load_state_dict(torch.load(d+'/ckpt.pth',map_location=DEVICE)['model']); m.eval(); return o,pc,m
o1,pcA,mA=build(DATA+'/survey_s7_Ng1024'); o2,pcB,mB=build(DATA+'/survey_s10_Ng1024'); tgA=TrajectoryGenerator(o1,pcA)
def jb():
    t=tgA.generate_trajectory(o1.box_width,o1.box_height,o1.batch_size)
    v=np.stack([t['ego_v']*np.cos(t['target_hd']),t['ego_v']*np.sin(t['target_hd'])],-1); v=torch.tensor(v,dtype=torch.float32).transpose(0,1).to(DEVICE)
    pos=np.stack([t['target_x'],t['target_y']],-1); pos=torch.tensor(pos,dtype=torch.float32).transpose(0,1).to(DEVICE)
    ip=np.stack([t['init_x'],t['init_y']],-1); ip=torch.tensor(ip,dtype=torch.float32).to(DEVICE)
    return (v,pcA.get_activation(ip).squeeze()),(v,pcB.get_activation(ip).squeeze()),pos
torch.manual_seed(1); np.random.seed(1)
G1,G2,PL,VL=[],[],[],[]
with torch.no_grad():
    for _ in range(N_BATCHES):
        a,b,pos=jb()
        G1.append(mA.g(a).cpu().numpy().transpose(1,0,2).reshape(-1,o1.Ng)[:,sel1])
        G2.append(mB.g(b).cpu().numpy().transpose(1,0,2).reshape(-1,o2.Ng)[:,sel2])
        PL.append(pos.cpu().numpy().transpose(1,0,2).reshape(-1,2)); VL.append(a[0].cpu().numpy().transpose(1,0,2).reshape(-1,2))
G=np.concatenate([np.concatenate(G1,0),np.concatenate(G2,0)],1).astype(np.float32)
POS=np.concatenate(PL,0); VEL=np.concatenate(VL,0)
sp=np.linalg.norm(VEL,axis=1); hd=np.arctan2(VEL[:,1],VEL[:,0])
pn=(POS-POS.min(0))/(POS.max(0)-POS.min(0)+1e-8); sn=(sp-sp.min())/(sp.max()-sp.min()+1e-8)
baseline=np.column_stack([pn,sn,np.cos(hd),np.sin(hd)]).astype(np.float32)
xgb=MLencoding(tunemodel='xgboost',cov_history=True,spike_history=False,window=20,n_filters=5,max_time=200)
# subset targets: first N_PER_NET per net with base>0.4
tgts=[]
for g in (0,1):
    idx=[ti for ti in np.where(net_id==g)[0] if pr2_base0[ti]>0.4][:N_PER_NET]; tgts+=idx
print(f'subset targets: {tgts} | {N_CV}-fold, {len(tgts)}x{N-1} pairs', flush=True)

aj=[];dp=[];tt=[];cons=[];wi=[]; basef=[];fullf=[];ci_all=[];ti_all=[]; t0=time.time()
for k,ti in enumerate(tgts):
    y=G[:,ti]; _,bf=xgb.fit_cv(baseline,y,verbose=0,continuous_folds=True,n_cv=N_CV); bf=np.asarray(bf)
    for ci in range(N):
        if ci==ti: continue
        _,ff=xgb.fit_cv(np.column_stack([baseline,G[:,ci]]),y,verbose=0,continuous_folds=True,n_cv=N_CV); ff=np.asarray(ff)
        d=ff-bf                                            # paired per-fold improvement
        aj.append(abs(float(J[ti,ci]))); wi.append(net_id[ci]==net_id[ti])
        dp.append(d.mean())
        tt.append(d.mean()/(d.std(ddof=1)/np.sqrt(len(d))+1e-9))
        cons.append((d>0).mean())
        basef.append(bf); fullf.append(ff); ci_all.append(ci); ti_all.append(ti)   # raw per-fold pR² for example scatters
    print(f'  target {k+1}/{len(tgts)} (#{ti}) done ({(time.time()-t0)/60:.1f} min)', flush=True)
aj=np.array(aj); dp=np.array(dp); tt=np.array(tt); cons=np.array(cons); wi=np.array(wi,dtype=bool)
np.savez(DATA+'/foldtest_710grid.npz',aj=aj,dpr2=dp,tstat=tt,cons=cons,wi=wi,tgts=np.array(tgts),
         base_folds=np.array(basef),full_folds=np.array(fullf),ci=np.array(ci_all),ti=np.array(ti_all))  # raw per-fold pR²
print('\n=== paired-fold statistics as J estimators (subset) ===', flush=True)
print(f'{"measure":14s} | within ρ(|J|) |  p     | within mean | cross mean | cross/within', flush=True)
for name,v in [('ΔpR² (mean)',dp),('paired t-stat',tt),('fold-consistency',cons)]:
    rho,p=spearmanr(aj[wi],v[wi]); wm=np.nanmean(v[wi]); xm=np.nanmean(v[~wi])
    print(f'{name:14s} | {rho:+.3f}       | {p:.0e} | {wm:+.3f}      | {xm:+.3f}     | {xm/wm:+.2f}', flush=True)
print('FOLDTEST DONE', flush=True)
