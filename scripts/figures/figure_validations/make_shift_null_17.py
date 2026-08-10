"""
Circular-shift NULL for the diffPC two-network control.

Builds the empirical 'chance' distribution of ΔpR²: for a random sample of pairs,
circularly shift the source cell's time series (breaking its temporal/positional
alignment with the target) and recompute ΔpR² = pR²(base+shifted) - pR²(base).
A source with no true relationship should give ΔpR² ~ null. Comparing the cross-net
floor to this null shows whether the floor is real common-input leakage or just chance.

Runs post-hoc on the finished cache; regenerates activations deterministically
(same seeds + saved cell selection as build_two_net_cache_diffPC.py).
"""
import sys, os, time
REPO_DIR='/Users/harryclark/Documents/spatial-manifolds/GRID-PATTERN-FORMATION'
sys.path.insert(0, REPO_DIR); sys.path.insert(0, '/Users/harryclark/Documents/spatial-manifolds/src')
import numpy as np, torch
import matplotlib; matplotlib.use('Agg'); import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from place_cells import PlaceCells
from trajectory_generator import TrajectoryGenerator
from model import RNN
from spatial_manifolds.mlencoding import MLencoding

SUBSET_SEED=0; N_BATCHES=12; N_CV=3; N_NULL=500
DATA='/Users/harryclark/Documents/spatial-manifolds/data/rnn_xgboost'
FIG='/Users/harryclark/Documents/spatial-manifolds/scripts/figures/figure_validations'
NETA=DATA+'/diffPC_A_Ng1024'; NETB=DATA+'/survey_s7_Ng1024'; CACHE=DATA+'/pr2_two_net_17_cache.npz'
DEVICE='mps' if torch.backends.mps.is_available() else ('cuda' if torch.cuda.is_available() else 'cpu')

c=np.load(CACHE, allow_pickle=True)
pr2_base=c['pr2_base']; pr2_cell=c['pr2_cell']; pr2_full=c['pr2_full']
net_id=c['net_id']; J=c['J_comb']; K=int(c['K']); sel1=c['sel1']; sel2=c['sel2']; N=len(net_id)
print(f'device={DEVICE} | cache N={N}', flush=True)

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

# regenerate activations exactly as the builder (deterministic)
torch.manual_seed(SUBSET_SEED+1); np.random.seed(SUBSET_SEED+1)
G1L,G2L,PL,VL=[],[],[],[]
with torch.no_grad():
    for _ in range(N_BATCHES):
        inpA,inpB,pos_b=joint_batch(tgA,o1)
        g1=mA.g(inpA).cpu().numpy().transpose(1,0,2).reshape(-1,o1.Ng)
        g2=mB.g(inpB).cpu().numpy().transpose(1,0,2).reshape(-1,o2.Ng)
        p=pos_b.cpu().numpy().transpose(1,0,2).reshape(-1,2); vv=inpA[0].cpu().numpy().transpose(1,0,2).reshape(-1,2)
        G1L.append(g1[:,sel1]); G2L.append(g2[:,sel2]); PL.append(p); VL.append(vv)
G_comb=np.concatenate([np.concatenate(G1L,0),np.concatenate(G2L,0)],axis=1).astype(np.float32)
POS=np.concatenate(PL,0).astype(np.float32); VEL=np.concatenate(VL,0).astype(np.float32)
speed=np.linalg.norm(VEL,axis=1); hd=np.arctan2(VEL[:,1],VEL[:,0])
pos_n=(POS-POS.min(0))/(POS.max(0)-POS.min(0)+1e-8); speed_n=(speed-speed.min())/(speed.max()-speed.min()+1e-8)
baseline=np.column_stack([pos_n,speed_n,np.cos(hd),np.sin(hd)]).astype(np.float32)
T=len(G_comb)
# sanity: regenerated activations should reproduce a cached pr2 (spot check target 0's baseline)
print(f'T={T} | regenerated activations (spot pr2_base[0] cached={pr2_base[0]:.3f})', flush=True)

xgb=MLencoding(tunemodel='xgboost',cov_history=True,spike_history=False,window=20,n_filters=5,max_time=200)

# ── shifted-source null on a random sample of pairs ──────────────────────────
rng=np.random.RandomState(123); null=[]; t0=time.time()
tries=0
while len(null)<N_NULL and tries<N_NULL*3:
    tries+=1
    ti=rng.randint(N); ci=rng.randint(N)
    if ci==ti or np.isnan(pr2_base[ti]) or pr2_base[ti]<0.3: continue
    off=rng.randint(T//10, 9*T//10)
    shifted=np.roll(G_comb[:,ci], off)[:,None]
    _,pf=xgb.fit_cv(np.column_stack([baseline,shifted]), G_comb[:,ti], verbose=0, continuous_folds=True, n_cv=N_CV)
    null.append(float(np.nanmean(pf))-pr2_base[ti])
    if len(null)%100==0: print(f'  null {len(null)}/{N_NULL}  ({(time.time()-t0)/60:.1f} min)', flush=True)
null=np.array(null)

# ── actual within/cross ΔpR² (base-filtered) from cache ──────────────────────
within=[]; cross=[]
for ti in range(N):
    if pr2_base[ti]<0.3: continue
    for ci in range(N):
        if ci==ti or np.isnan(pr2_full[ci,ti]): continue
        d=pr2_full[ci,ti]-pr2_base[ti]
        (within if net_id[ci]==net_id[ti] else cross).append(d)
within=np.array(within); cross=np.array(cross)

nm,ns=null.mean(),null.std(); n95=np.percentile(null,95)
print(f'\n=== SHIFT-NULL RESULT ===', flush=True)
print(f'null (shifted source):  mean={nm:+.4f}  std={ns:.4f}  95th={n95:.4f}', flush=True)
print(f'cross-net ΔpR²:         mean={cross.mean():+.4f}  ({(cross.mean()-nm)/ns:+.1f} SD above null)', flush=True)
print(f'within-net ΔpR²:        mean={within.mean():+.4f}  ({(within.mean()-nm)/ns:+.1f} SD above null)', flush=True)
frac_cross_above=(cross.mean()>n95)
print(f'cross floor above null 95th percentile? {frac_cross_above}', flush=True)

# ── figure: null vs cross vs within ──────────────────────────────────────────
fig,ax=plt.subplots(figsize=(6,4.5))
parts=ax.violinplot([null,cross,within],positions=[0,1,2],widths=0.7,showmeans=True)
for b,cc in zip(parts['bodies'],['#999','#3171ae','#c04744']): b.set_facecolor(cc); b.set_alpha(0.55)
ax.axhline(0,color='k',lw=0.8,ls=':')
ax.axhline(n95,color='#999',lw=1,ls='--',label='null 95th pct')
ax.set_xticks([0,1,2]); ax.set_xticklabels(['null\n(shifted source)','cross-net\n(J=0)','within-net\n(J≠0)'])
ax.set_ylabel('ΔpR²  (base+cell − base)')
ax.set_title('diffPC control: chance null vs cross vs within\n'
             f'cross = {(cross.mean()-nm)/ns:.0f} SD above chance; within = {(within.mean()-nm)/ns:.0f} SD',fontsize=10,fontweight='bold')
ax.legend(fontsize=8,frameon=False); ax.spines[['top','right']].set_visible(False)
plt.tight_layout(); plt.savefig(FIG+'/two_network_17_shiftnull.png',dpi=120,bbox_inches='tight'); plt.savefig(FIG+'/two_network_17_shiftnull.pdf',bbox_inches='tight')
print('saved two_network_17_shiftnull.png/pdf', flush=True); print('NULL_DONE', flush=True)
