"""
Combined figure: top row = matrices (J | coactivity | pR2(cell) | ΔpR2), network-ordered;
bottom row = each measure vs rate-map correlation (tuning similarity), within vs cross.
Shows structure (top) + why coactivity is just tuning while ΔpR2 isn't (bottom). diffPC cache.
"""
import sys, os
REPO_DIR='/Users/harryclark/Documents/spatial-manifolds/GRID-PATTERN-FORMATION'
sys.path.insert(0, REPO_DIR); sys.path.insert(0,'/Users/harryclark/Documents/spatial-manifolds/src')
import numpy as np, torch
import matplotlib; matplotlib.use('Agg'); import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from scipy.stats import spearmanr
from place_cells import PlaceCells
from trajectory_generator import TrajectoryGenerator
from model import RNN
from visualize import compute_ratemaps

SUBSET_SEED=0; N_BATCHES=12; RES=40; N_AVG=50
DATA='/Users/harryclark/Documents/spatial-manifolds/data/rnn_xgboost'
FIG='/Users/harryclark/Documents/spatial-manifolds/scripts/figures/figure_validations'
NETA=DATA+'/diffPC_A_Ng1024'; NETB=DATA+'/survey_s7_Ng1024'; CACHE=DATA+'/pr2_two_net_17_cache.npz'
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
    ip=np.stack([traj['init_x'],traj['init_y']],axis=-1); ip=torch.tensor(ip,dtype=torch.float32).to(DEVICE)
    return (v,pcA.get_activation(ip).squeeze()),(v,pcB.get_activation(ip).squeeze())
torch.manual_seed(SUBSET_SEED+1); np.random.seed(SUBSET_SEED+1)
G1L,G2L=[],[]
with torch.no_grad():
    for _ in range(N_BATCHES):
        inpA,inpB=joint_batch(tgA,o1)
        G1L.append(mA.g(inpA).cpu().numpy().transpose(1,0,2).reshape(-1,o1.Ng)[:,sel1])
        G2L.append(mB.g(inpB).cpu().numpy().transpose(1,0,2).reshape(-1,o2.Ng)[:,sel2])
G_comb=np.concatenate([np.concatenate(G1L,0),np.concatenate(G2L,0)],axis=1).astype(np.float32)
coact=np.corrcoef(G_comb.T)
low=pr2_base<0.3
W,X='#c04744','#3171ae'

# matrices [target,source]
Jm=J.copy(); CO=coact.copy(); CELL=pr2_cell.T.copy(); DE=(pr2_full-pr2_base[None,:]).T.copy()
for M in (CELL,DE): np.fill_diagonal(M,np.nan)
np.fill_diagonal(CO,np.nan); DE[low,:]=np.nan
def blocks(ax,title):
    ax.axhline(K-0.5,color='k',lw=1.1); ax.axvline(K-0.5,color='k',lw=1.1); ax.set_xticks([]);ax.set_yticks([])
    for f,l,cc in [(0.25,'net1',W),(0.75,'net7',X)]:
        ax.text(N*f,N+3,l,ha='center',va='top',fontsize=6,fontweight='bold',color=cc)
        ax.text(-3,N*f,l,ha='center',va='center',rotation=90,fontsize=6,fontweight='bold',color=cc)
    ax.set_title(title,fontsize=9,fontweight='bold',pad=5)

# long-form for scatters
rc=[];co=[];ce=[];de=[];wi=[];aj=[]
for ti in range(N):
    if low[ti]: continue
    for ci in range(N):
        if ci==ti or np.isnan(pr2_full[ci,ti]): continue
        rc.append(rmcorr[ci,ti]); co.append(coact[ci,ti]); ce.append(pr2_cell[ci,ti])
        de.append(pr2_full[ci,ti]-pr2_base[ti]); wi.append(net_id[ci]==net_id[ti]); aj.append(abs(float(J[ti,ci])))
rc=np.array(rc);co=np.array(co);ce=np.array(ce);de=np.array(de);wi=np.array(wi,dtype=bool);aj=np.array(aj)

fig=plt.figure(figsize=(16,12.2))
gs=fig.add_gridspec(3,4,height_ratios=[1.15,1.0,1.0],hspace=0.36,wspace=0.22)
# ── row 0: matrices ──
axJ=fig.add_subplot(gs[0,0]); vj=np.percentile(np.abs(Jm[Jm!=0]),99)
fig.colorbar(axJ.imshow(Jm,cmap='RdBu_r',vmin=-vj,vmax=vj,interpolation='nearest'),ax=axJ,fraction=0.046,pad=0.02); blocks(axJ,'J  (ground-truth coupling)')
for col,(M,title,cmap,lo,hi) in zip(range(1,4),[
    (CO,'coactivity (Pearson r)','RdBu_r',-np.nanpercentile(np.abs(CO),99),np.nanpercentile(np.abs(CO),99)),
    (CELL,'pR²(cell)','viridis',0,np.nanpercentile(CELL,99)),
    (DE,'ΔpR²','inferno',0,np.nanpercentile(DE,95))]):
    ax=fig.add_subplot(gs[0,col]); cm=matplotlib.colormaps[cmap].copy(); cm.set_bad('lightgrey')
    fig.colorbar(ax.imshow(M,cmap=cm,vmin=lo,vmax=hi,interpolation='nearest'),ax=ax,fraction=0.046,pad=0.02); blocks(ax,title)
# ── row 1: measure vs |J| (recovery of ground-truth coupling) ──
axL1=fig.add_subplot(gs[1,0]); axL1.axis('off')
axL1.text(0.0,0.85,'vs |J| — recovery of\nground-truth coupling',fontsize=10,va='top',fontweight='bold')
axL1.text(0.0,0.45,'within pairs span |J|;\ncross pairs all at |J|=0\n(so ρ is within-only).',fontsize=8.5,va='top',color='#555')
for col,(y,name) in zip(range(1,4),[(co,'coactivity (r)'),(ce,'pR²(cell)'),(de,'ΔpR²')]):
    ax=fig.add_subplot(gs[1,col])
    ax.scatter(aj[~wi],y[~wi],s=4,alpha=0.3,color=X,label='cross (J=0)')
    ax.scatter(aj[wi],y[wi],s=4,alpha=0.3,color=W,label='within (J≠0)')
    rw=spearmanr(aj[wi],y[wi])[0]
    ax.set_xlabel('|J|  (ground-truth coupling)'); ax.set_ylabel(name)
    ax.set_title(f'{name} vs |J|\nwithin ρ={rw:+.2f}',fontsize=8.5,fontweight='bold')
    ax.axhline(0,color='grey',lw=0.6,ls=':'); ax.spines[['top','right']].set_visible(False)
    if col==1: ax.legend(fontsize=7,frameon=False,markerscale=2)
# ── row 2: measure vs rate-map correlation (tuning similarity) ──
axL2=fig.add_subplot(gs[2,0]); axL2.axis('off')
axL2.text(0.0,0.85,'vs rate-map corr —\ntuning / phase similarity',fontsize=10,va='top',fontweight='bold')
axL2.text(0.0,0.48,'Coactivity ≈ tuning (ρ=0.96):\njust re-reads rate-map overlap.',fontsize=8.5,va='top',color=X)
axL2.text(0.0,0.12,'ΔpR² ⊥ tuning (ρ=0.11):\nisolates coupling, not tuning.',fontsize=8.5,va='top',color=W)
for col,(y,name) in zip(range(1,4),[(co,'coactivity (r)'),(ce,'pR²(cell)'),(de,'ΔpR²')]):
    ax=fig.add_subplot(gs[2,col])
    ax.scatter(rc[wi],y[wi],s=4,alpha=0.3,color=W,label='within')
    ax.scatter(rc[~wi],y[~wi],s=4,alpha=0.3,color=X,label='cross')
    rw=spearmanr(rc[wi],y[wi])[0]; rx=spearmanr(rc[~wi],y[~wi])[0]
    ax.set_xlabel('rate-map correlation'); ax.set_ylabel(name)
    ax.set_title(f'{name} vs rate-map corr\nρ within={rw:+.2f} cross={rx:+.2f}',fontsize=8.5,fontweight='bold')
    ax.axhline(0,color='grey',lw=0.6,ls=':'); ax.spines[['top','right']].set_visible(False)
fig.suptitle('Coupling structure (top) · recovery of ground-truth |J| (middle) · relationship to tuning similarity (bottom)',
             fontsize=12,fontweight='bold')
plt.savefig(FIG+'/combined_matrices_ratemapcorr_17.png',dpi=120,bbox_inches='tight')
plt.savefig(FIG+'/combined_matrices_ratemapcorr_17.pdf',bbox_inches='tight')
print('saved combined_matrices_ratemapcorr_17.png/pdf', flush=True); print('COMBINED_DONE', flush=True)
