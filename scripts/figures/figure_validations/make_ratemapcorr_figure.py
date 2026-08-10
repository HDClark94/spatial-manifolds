"""
Rate-map correlation (tuning/phase similarity) vs each coupling measure, colored
within-net (J≠0) vs cross-net (J=0). Tests whether a measure just re-reads tuning
(coactivity should track rate-map-corr, incl. cross-net) or captures coupling beyond
it (ΔpR² should stay low for cross-net pairs even at high rate-map similarity).
Uses the completed diffPC cache.
"""
import sys, os
REPO_DIR='/Users/harryclark/Documents/spatial-manifolds/GRID-PATTERN-FORMATION'
sys.path.insert(0, REPO_DIR); sys.path.insert(0,'/Users/harryclark/Documents/spatial-manifolds/src')
import numpy as np, torch
import matplotlib; matplotlib.use('Agg'); import matplotlib.pyplot as plt
from scipy.stats import spearmanr
from place_cells import PlaceCells
from trajectory_generator import TrajectoryGenerator
from model import RNN
from visualize import compute_ratemaps

SUBSET_SEED=0; N_BATCHES=12; RES=40; N_AVG=50
DATA='/Users/harryclark/Documents/spatial-manifolds/data/rnn_xgboost'
FIG='/Users/harryclark/Documents/spatial-manifolds/scripts/figures/figure_validations'
NETA=DATA+'/diffPC_A_Ng1024'; NETB=DATA+'/diffPC_B_Ng1024'; CACHE=DATA+'/pr2_two_net_diffPC_cache.npz'
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

# rate maps (spatial tuning) for the selected cells
a1,_,_,_=compute_ratemaps(mA,tgA,o1,res=RES,n_avg=N_AVG,Ng=o1.Ng)
a2,_,_,_=compute_ratemaps(mB,tgB,o2,res=RES,n_avg=N_AVG,Ng=o2.Ng)
rm=np.concatenate([a1[sel1],a2[sel2]],axis=0).reshape(N,-1)   # (N, RES*RES)
rmcorr=np.corrcoef(rm)                                        # rate-map correlation (N,N)
print('rate maps + corr done', flush=True)

# coactivity (temporal correlation over shared trajectory)
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
print('coactivity done', flush=True)

# long-form over ordered pairs (base-filtered targets)
low=pr2_base<0.3
rc=[]; co=[]; ce=[]; de=[]; wi=[]
for ti in range(N):
    if low[ti]: continue
    for ci in range(N):
        if ci==ti or np.isnan(pr2_full[ci,ti]): continue
        rc.append(rmcorr[ci,ti]); co.append(coact[ci,ti])
        ce.append(pr2_cell[ci,ti]); de.append(pr2_full[ci,ti]-pr2_base[ti])
        wi.append(net_id[ci]==net_id[ti])
rc=np.array(rc); co=np.array(co); ce=np.array(ce); de=np.array(de); wi=np.array(wi,dtype=bool)
W,X='#c04744','#3171ae'

fig,axes=plt.subplots(1,3,figsize=(15,4.6))
for ax,y,name in [(axes[0],co,'coactivity (Pearson r)'),(axes[1],ce,'pR²(cell)'),(axes[2],de,'ΔpR²')]:
    ax.scatter(rc[wi],y[wi],s=5,alpha=0.3,color=W,label='within (J≠0)')
    ax.scatter(rc[~wi],y[~wi],s=5,alpha=0.3,color=X,label='cross (J=0)')
    rw=spearmanr(rc[wi],y[wi])[0]; rx=spearmanr(rc[~wi],y[~wi])[0]
    ax.set_xlabel('rate-map correlation  (tuning / phase similarity)')
    ax.set_ylabel(name)
    ax.set_title(f'{name}  vs  rate-map corr\nSpearman: within={rw:+.2f}  cross={rx:+.2f}',fontsize=9,fontweight='bold')
    ax.axhline(0,color='grey',lw=0.6,ls=':')
    ax.legend(fontsize=8,frameon=False,markerscale=2)
    ax.spines[['top','right']].set_visible(False)
fig.suptitle('Does the measure just re-read tuning? coupling vs rate-map similarity (within vs cross)',
             fontsize=12,fontweight='bold')
plt.tight_layout(rect=[0,0,1,0.95])
plt.savefig(FIG+'/ratemapcorr_vs_coupling.png',dpi=120,bbox_inches='tight')
plt.savefig(FIG+'/ratemapcorr_vs_coupling.pdf',bbox_inches='tight')
print(f"Spearman(rmcorr, measure): coact within={spearmanr(rc[wi],co[wi])[0]:+.2f}/cross={spearmanr(rc[~wi],co[~wi])[0]:+.2f} | "
      f"pR2cell within={spearmanr(rc[wi],ce[wi])[0]:+.2f}/cross={spearmanr(rc[~wi],ce[~wi])[0]:+.2f} | "
      f"dpr2 within={spearmanr(rc[wi],de[wi])[0]:+.2f}/cross={spearmanr(rc[~wi],de[~wi])[0]:+.2f}", flush=True)
print(f"cross pairs high rate-map-corr (>0.5): mean coact={co[~wi][rc[~wi]>0.5].mean():.3f} "
      f"mean dpr2={de[~wi][rc[~wi]>0.5].mean():.4f} (n={(rc[~wi]>0.5).sum()})", flush=True)
print('RMCORR_DONE', flush=True)
