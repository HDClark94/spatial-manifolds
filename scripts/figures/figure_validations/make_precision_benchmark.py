"""
Benchmark coupling measures against ground-truth |J| on the diffPC pair.
All use the SAME behavioral baseline (pos+speed+HD, XGBoost) so only the MEASURE differs:
  - coactivity        : raw Pearson r of activity            (marginal, no conditioning)
  - ΔpR²              : predictive gain over baseline        (marginal, behavior-conditioned)
  - residual corr     : Pearson r of behavior-residuals      (pairwise partial)
  - precision partial : partial corr from inv-cov of residuals (conditional on ALL cells) <- proposed
Reports within-net Spearman(|J|, measure) [recovery] and cross/within ratio [floor].
"""
import sys, os, time
REPO_DIR='/Users/harryclark/Documents/spatial-manifolds/GRID-PATTERN-FORMATION'
sys.path.insert(0, REPO_DIR); sys.path.insert(0,'/Users/harryclark/Documents/spatial-manifolds/src')
import numpy as np, torch
import matplotlib; matplotlib.use('Agg'); import matplotlib.pyplot as plt
from scipy.stats import spearmanr
from place_cells import PlaceCells
from trajectory_generator import TrajectoryGenerator
from model import RNN
from spatial_manifolds.mlencoding import MLencoding

SUBSET_SEED=0; N_BATCHES=12; N_CV=3
DATA='/Users/harryclark/Documents/spatial-manifolds/data/rnn_xgboost'
FIG='/Users/harryclark/Documents/spatial-manifolds/scripts/figures/figure_validations'
NETA=DATA+'/diffPC_A_Ng1024'; NETB=DATA+'/diffPC_B_Ng1024'; CACHE=DATA+'/pr2_two_net_diffPC_cache.npz'
DEVICE='mps' if torch.backends.mps.is_available() else 'cpu'

c=np.load(CACHE,allow_pickle=True)
pr2_base=c['pr2_base']; pr2_full=c['pr2_full']; net_id=c['net_id']; J=c['J_comb']; K=int(c['K']); sel1=c['sel1']; sel2=c['sel2']; N=len(net_id)

def build_net(d):
    class O: pass
    o=O(); o.Np=512;o.Ng=1024;o.sequence_length=20;o.batch_size=200
    o.learning_rate=1e-4;o.weight_decay=1e-4;o.place_cell_rf=0.12;o.surround_scale=2
    o.RNN_type='RNN';o.activation='relu';o.DoG=True;o.periodic=False
    o.box_width=2.2;o.box_height=2.2;o.device=DEVICE
    pc=PlaceCells(o); pc.us=torch.tensor(np.load(d+'/place_cells.npy'),dtype=torch.float32).to(DEVICE)
    m=RNN(o,pc).to(DEVICE); m.load_state_dict(torch.load(d+'/ckpt.pth',map_location=DEVICE)['model']); m.eval()
    return o,pc,m
o1,pcA,mA=build_net(NETA); o2,pcB,mB=build_net(NETB); tgA=TrajectoryGenerator(o1,pcA)

def joint_batch(tg,o):
    traj=tg.generate_trajectory(o.box_width,o.box_height,o.batch_size)
    v=np.stack([traj['ego_v']*np.cos(traj['target_hd']),traj['ego_v']*np.sin(traj['target_hd'])],axis=-1)
    v=torch.tensor(v,dtype=torch.float32).transpose(0,1).to(DEVICE)
    pos=np.stack([traj['target_x'],traj['target_y']],axis=-1); pos=torch.tensor(pos,dtype=torch.float32).transpose(0,1).to(DEVICE)
    ip=np.stack([traj['init_x'],traj['init_y']],axis=-1); ip=torch.tensor(ip,dtype=torch.float32).to(DEVICE)
    return (v,pcA.get_activation(ip).squeeze()),(v,pcB.get_activation(ip).squeeze()),pos
torch.manual_seed(SUBSET_SEED+1); np.random.seed(SUBSET_SEED+1)
G1L,G2L,PL,VL=[],[],[],[]
with torch.no_grad():
    for _ in range(N_BATCHES):
        inpA,inpB,pos_b=joint_batch(tgA,o1)
        G1L.append(mA.g(inpA).cpu().numpy().transpose(1,0,2).reshape(-1,o1.Ng)[:,sel1])
        G2L.append(mB.g(inpB).cpu().numpy().transpose(1,0,2).reshape(-1,o2.Ng)[:,sel2])
        PL.append(pos_b.cpu().numpy().transpose(1,0,2).reshape(-1,2)); VL.append(inpA[0].cpu().numpy().transpose(1,0,2).reshape(-1,2))
G=np.concatenate([np.concatenate(G1L,0),np.concatenate(G2L,0)],axis=1).astype(np.float32)
POS=np.concatenate(PL,0); VEL=np.concatenate(VL,0)
sp=np.linalg.norm(VEL,axis=1); hd=np.arctan2(VEL[:,1],VEL[:,0])
pn=(POS-POS.min(0))/(POS.max(0)-POS.min(0)+1e-8); sn=(sp-sp.min())/(sp.max()-sp.min()+1e-8)
baseline=np.column_stack([pn,sn,np.cos(hd),np.sin(hd)]).astype(np.float32)
print(f'activations {G.shape}; residualizing {N} cells against baseline...', flush=True)

# residuals: same XGBoost baseline as ΔpR² (CV predictions), so only the MEASURE differs
xgb=MLencoding(tunemodel='xgboost',cov_history=True,spike_history=False,window=20,n_filters=5,max_time=200)
R=np.zeros_like(G)
t0=time.time()
for i in range(N):
    Yhat,_=xgb.fit_cv(baseline,G[:,i],verbose=0,continuous_folds=True,n_cv=N_CV)
    R[:,i]=G[:,i]-np.asarray(Yhat).ravel()
    if i%20==0: print(f'  residual {i}/{N} ({(time.time()-t0)/60:.1f} min)', flush=True)

coact=np.corrcoef(G.T)
rcorr=np.corrcoef(R.T)
C=np.cov(R.T)+1e-6*np.eye(N); Th=np.linalg.inv(C)
d=np.sqrt(np.diag(Th)); prec=-Th/np.outer(d,d)   # partial correlation matrix
dpr2=(pr2_full-pr2_base[None,:]).T                # [target,source]  (directed)

low=pr2_base<0.3
def longform(M, directed=False):
    aj=[]; val=[]; wi=[]
    for ti in range(N):
        if low[ti]: continue
        for ci in range(N):
            if ci==ti: continue
            v = M[ci,ti] if directed else M[ti,ci]
            if np.isnan(v): continue
            aj.append(abs(float(J[ti,ci]))); val.append(abs(float(v))); wi.append(net_id[ci]==net_id[ti])
    return np.array(aj),np.array(val),np.array(wi,dtype=bool)

measures=[('coactivity',coact,False),('ΔpR²',dpr2,True),('residual corr',rcorr,False),('precision partial',prec,False)]
print('\n=== BENCHMARK (diffPC pair) ===', flush=True)
print(f'{"measure":18s} | within ρ(|J|) | within mean | cross mean | cross/within', flush=True)
rows=[]
for name,M,dr in measures:
    aj,val,wi=longform(M,dr)
    rho=spearmanr(aj[wi],val[wi])[0]
    wm=val[wi].mean(); xm=val[~wi].mean(); ratio=xm/wm
    rows.append((name,rho,wm,xm,ratio,aj,val,wi))
    print(f'{name:18s} | {rho:+.3f}        | {wm:.4f}      | {xm:.4f}    | {ratio:.2f}', flush=True)

# figure
fig,ax=plt.subplots(1,3,figsize=(14,4.4))
names=[r[0] for r in rows]; cols=['#888','#c04744','#e0a25a','#2e8b57']
ax[0].bar(range(4),[r[1] for r in rows],color=cols)
ax[0].set_xticks(range(4)); ax[0].set_xticklabels(names,rotation=20,ha='right',fontsize=8)
ax[0].set_ylabel('within-net Spearman(|J|, measure)'); ax[0].set_title('coupling recovery\n(higher=better)',fontsize=9,fontweight='bold')
for i,r in enumerate(rows): ax[0].text(i,r[1]+0.005,f'{r[1]:.2f}',ha='center',fontsize=8)
ax[0].spines[['top','right']].set_visible(False)
ax[1].bar(range(4),[r[4] for r in rows],color=cols)
ax[1].set_xticks(range(4)); ax[1].set_xticklabels(names,rotation=20,ha='right',fontsize=8)
ax[1].set_ylabel('cross / within'); ax[1].set_title('cross-net floor\n(lower=better)',fontsize=9,fontweight='bold')
for i,r in enumerate(rows): ax[1].text(i,r[4]+0.01,f'{r[4]:.2f}',ha='center',fontsize=8)
ax[1].spines[['top','right']].set_visible(False)
# scatter: precision partial vs |J|
name,M,dr=('precision partial',prec,False); aj,val,wi=longform(M,dr)
ax[2].scatter(aj[~wi],val[~wi],s=5,alpha=0.3,color='#3171ae',label='cross (J=0)')
ax[2].scatter(aj[wi],val[wi],s=5,alpha=0.3,color='#c04744',label='within (J≠0)')
ax[2].set_xlabel('|J| (ground truth)'); ax[2].set_ylabel('|precision partial corr|')
ax[2].set_title(f'precision partial vs |J|\nwithin ρ={spearmanr(aj[wi],val[wi])[0]:+.2f}',fontsize=9,fontweight='bold')
ax[2].legend(fontsize=8,frameon=False,markerscale=2); ax[2].spines[['top','right']].set_visible(False)
fig.suptitle('Coupling measures vs ground-truth |J| (diffPC pair) — same baseline, different measure',fontsize=12,fontweight='bold')
plt.tight_layout(rect=[0,0,1,0.95])
plt.savefig(FIG+'/precision_benchmark.png',dpi=120,bbox_inches='tight'); plt.savefig(FIG+'/precision_benchmark.pdf',bbox_inches='tight')
print('saved precision_benchmark.png/pdf', flush=True); print('BENCHMARK_DONE', flush=True)
