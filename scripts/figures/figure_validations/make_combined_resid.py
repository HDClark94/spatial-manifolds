"""
Combined figure WITH pairwise residual correlation added as a 5th measure.
Columns: J | coactivity | pR2(cell) | ΔpR2 | residual corr.
Rows: matrices (top) | vs |J| recovery (middle) | vs rate-map corr / tuning (bottom).
Pair + cache set below (currently nets 6+7).
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
from visualize import compute_ratemaps
from spatial_manifolds.mlencoding import MLencoding

# ── pair config ──
DATA='/Users/harryclark/Documents/spatial-manifolds/data/rnn_xgboost'
FIG='/Users/harryclark/Documents/spatial-manifolds/scripts/figures/figure_validations'
NETA=DATA+'/survey_s6_Ng1024'; NETB=DATA+'/survey_s7_Ng1024'
CACHE=DATA+'/pr2_two_net_67_cache.npz'; OUT=FIG+'/combined_matrices_ratemapcorr_67'
LAB1,LAB2='net6','net7'
SUBSET_SEED=0; N_BATCHES=12; RES=40; N_AVG=50; N_CV=3
DEVICE='mps' if torch.backends.mps.is_available() else 'cpu'

c=np.load(CACHE,allow_pickle=True)
pr2_base=c['pr2_base']; pr2_cell=c['pr2_cell']; pr2_full=c['pr2_full']
net_id=c['net_id']; J=c['J_comb']; K=int(c['K']); sel1=c['sel1']; sel2=c['sel2']; N=len(net_id)

def build_net(d):
    class O: pass
    o=O(); o.Np=512;o.Ng=1024;o.sequence_length=20;o.batch_size=200
    o.learning_rate=1e-4;o.weight_decay=1e-4;o.place_cell_rf=0.12;o.surround_scale=2
    o.RNN_type='RNN';o.activation='relu';o.DoG=True;o.periodic=False
    o.box_width=2.2;o.box_height=2.2;o.device=DEVICE
    pc=PlaceCells(o); pc.us=torch.tensor(np.load(d+'/place_cells.npy'),dtype=torch.float32).to(DEVICE)
    m=RNN(o,pc).to(DEVICE); m.load_state_dict(torch.load(d+'/ckpt.pth',map_location=DEVICE)['model']); m.eval()
    return o,pc,m
o1,pcA,mA=build_net(NETA); o2,pcB,mB=build_net(NETB); tgA=TrajectoryGenerator(o1,pcA); tgB=TrajectoryGenerator(o2,pcB)

a1,_,_,_=compute_ratemaps(mA,tgA,o1,res=RES,n_avg=N_AVG,Ng=o1.Ng)
a2,_,_,_=compute_ratemaps(mB,tgB,o2,res=RES,n_avg=N_AVG,Ng=o2.Ng)
rmcorr=np.corrcoef(np.concatenate([a1[sel1],a2[sel2]],0).reshape(N,-1))

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
coact=np.corrcoef(G.T)

# residualize each cell against behavior (same baseline as ΔpR²), then correlate leftovers
sp=np.linalg.norm(VEL,axis=1); hd=np.arctan2(VEL[:,1],VEL[:,0])
pn=(POS-POS.min(0))/(POS.max(0)-POS.min(0)+1e-8); sn=(sp-sp.min())/(sp.max()-sp.min()+1e-8)
baseline=np.column_stack([pn,sn,np.cos(hd),np.sin(hd)]).astype(np.float32)
xgb=MLencoding(tunemodel='xgboost',cov_history=True,spike_history=False,window=20,n_filters=5,max_time=200)
Rr=np.zeros_like(G); t0=time.time()
for i in range(N):
    Yhat,_=xgb.fit_cv(baseline,G[:,i],verbose=0,continuous_folds=True,n_cv=N_CV); Rr[:,i]=G[:,i]-np.asarray(Yhat).ravel()
    if i%20==0: print(f'residual {i}/{N} ({(time.time()-t0)/60:.1f} min)',flush=True)
rcorr_resid=np.corrcoef(Rr.T)
low=pr2_base<0.3
W,X='#c04744','#3171ae'

# matrices [target,source]
Jm=J.copy(); CO=coact.copy(); CELL=pr2_cell.T.copy(); DE=(pr2_full-pr2_base[None,:]).T.copy(); RC=rcorr_resid.copy()
for M in (CELL,DE): np.fill_diagonal(M,np.nan)
np.fill_diagonal(CO,np.nan); np.fill_diagonal(RC,np.nan); DE[low,:]=np.nan
def blocks(ax,title):
    ax.axhline(K-0.5,color='k',lw=1.1); ax.axvline(K-0.5,color='k',lw=1.1); ax.set_xticks([]);ax.set_yticks([])
    for f,l,cc in [(0.25,LAB1,W),(0.75,LAB2,X)]:
        ax.text(N*f,N+3,l,ha='center',va='top',fontsize=6,fontweight='bold',color=cc)
        ax.text(-3,N*f,l,ha='center',va='center',rotation=90,fontsize=6,fontweight='bold',color=cc)
    ax.set_title(title,fontsize=8.5,fontweight='bold',pad=5)

# long-form
rc=[];co=[];ce=[];de=[];rv=[];wi=[];aj=[]
for ti in range(N):
    if low[ti]: continue
    for ci in range(N):
        if ci==ti or np.isnan(pr2_full[ci,ti]): continue
        rc.append(rmcorr[ci,ti]); co.append(coact[ci,ti]); ce.append(pr2_cell[ci,ti])
        de.append(pr2_full[ci,ti]-pr2_base[ti]); rv.append(rcorr_resid[ci,ti]); wi.append(net_id[ci]==net_id[ti]); aj.append(abs(float(J[ti,ci])))
rc,co,ce,de,rv,aj=map(np.array,(rc,co,ce,de,rv,aj)); wi=np.array(wi,dtype=bool)

MEAS=[('coactivity (r)',CO,co,'RdBu_r'),('pR²(cell)',CELL,ce,'viridis'),('ΔpR²',DE,de,'inferno'),('residual corr',RC,rv,'PuOr_r')]
fig=plt.figure(figsize=(19,12.2))
gs=fig.add_gridspec(3,5,height_ratios=[1.15,1.0,1.0],hspace=0.36,wspace=0.24)
# row 0: J + 4 measure matrices
axJ=fig.add_subplot(gs[0,0]); vj=np.percentile(np.abs(Jm[Jm!=0]),99)
fig.colorbar(axJ.imshow(Jm,cmap='RdBu_r',vmin=-vj,vmax=vj,interpolation='nearest'),ax=axJ,fraction=0.046,pad=0.02); blocks(axJ,'J (ground-truth coupling)')
for col,(name,M,_,cmap) in enumerate(MEAS,start=1):
    ax=fig.add_subplot(gs[0,col]); cm=matplotlib.colormaps[cmap].copy(); cm.set_bad('lightgrey')
    if cmap in ('RdBu_r','PuOr_r'): vm=np.nanpercentile(np.abs(M),99); lo,hi=-vm,vm
    else: lo,hi=(np.nanpercentile(M,2),np.nanpercentile(M,98)) if name=='pR²(cell)' else (0,np.nanpercentile(M,95))
    fig.colorbar(ax.imshow(M,cmap=cm,vmin=lo,vmax=hi,interpolation='nearest'),ax=ax,fraction=0.046,pad=0.02); blocks(ax,name)
# row 1: vs |J|
axl=fig.add_subplot(gs[1,0]); axl.axis('off'); axl.text(0.0,0.8,'vs |J| — recovery of\nground-truth coupling',fontsize=10,va='top',fontweight='bold')
for col,(name,_,y,_) in enumerate(MEAS,start=1):
    ax=fig.add_subplot(gs[1,col]); yy=np.abs(y) if name!='ΔpR²' else y
    ax.scatter(aj[~wi],yy[~wi],s=4,alpha=0.3,color=X); ax.scatter(aj[wi],yy[wi],s=4,alpha=0.3,color=W)
    ax.set_xlabel('|J|'); ax.set_ylabel('|'+name+'|' if name!='ΔpR²' else name)
    ax.set_title(f'within ρ={spearmanr(aj[wi],yy[wi])[0]:+.2f}',fontsize=8.5,fontweight='bold')
    ax.axhline(0,color='grey',lw=0.5,ls=':'); ax.spines[['top','right']].set_visible(False)
# row 2: vs rate-map corr
axl2=fig.add_subplot(gs[2,0]); axl2.axis('off'); axl2.text(0.0,0.8,'vs rate-map corr —\ntuning / phase similarity',fontsize=10,va='top',fontweight='bold')
for col,(name,_,y,_) in enumerate(MEAS,start=1):
    ax=fig.add_subplot(gs[2,col])
    ax.scatter(rc[wi],y[wi],s=4,alpha=0.3,color=W,label='within'); ax.scatter(rc[~wi],y[~wi],s=4,alpha=0.3,color=X,label='cross')
    rw=spearmanr(rc[wi],y[wi])[0]; rx=spearmanr(rc[~wi],y[~wi])[0]
    ax.set_xlabel('rate-map correlation'); ax.set_ylabel(name)
    ax.set_title(f'ρ within={rw:+.2f} cross={rx:+.2f}',fontsize=8.5,fontweight='bold')
    ax.axhline(0,color='grey',lw=0.5,ls=':'); ax.spines[['top','right']].set_visible(False)
    if col==1: ax.legend(fontsize=7,frameon=False,markerscale=2)
fig.suptitle(f'Coupling measures — structure (top) · |J| recovery (middle) · tuning (bottom) — {LAB1}+{LAB2}',fontsize=12,fontweight='bold')
plt.savefig(OUT+'.png',dpi=115,bbox_inches='tight'); plt.savefig(OUT+'.pdf',bbox_inches='tight')
print('saved '+OUT+'.png/pdf', flush=True); print('COMBINED_DONE', flush=True)
