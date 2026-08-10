"""
'Why not just use coactivity?' figure.
Regenerates the diffPC activations, computes the raw pairwise coactivity (Pearson corr)
matrix, and puts J (ground truth) | coactivity | pR2(cell) | delta pR2 side by side,
network-ordered, plus the within-vs-cross comparison showing coactivity's cross-net
false-positive floor vs delta pR2's near-zero cross floor.
"""
import sys, os
REPO_DIR='/Users/harryclark/Documents/spatial-manifolds/GRID-PATTERN-FORMATION'
sys.path.insert(0, REPO_DIR); sys.path.insert(0,'/Users/harryclark/Documents/spatial-manifolds/src')
import numpy as np, torch
import matplotlib; matplotlib.use('Agg'); import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from place_cells import PlaceCells
from trajectory_generator import TrajectoryGenerator
from model import RNN

SUBSET_SEED=0; N_BATCHES=12
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
o1,pcA,mA=build_net(NETA); o2,pcB,mB=build_net(NETB); tgA=TrajectoryGenerator(o1,pcA)

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
print(f'regenerated activations G_comb {G_comb.shape}', flush=True)

# ── coactivity (Pearson correlation) matrix ──────────────────────────────────
coact=np.corrcoef(G_comb.T)              # (N,N) symmetric
low=pr2_base<0.3

# matrices oriented [target,source]
Jm=J.copy(); CO=coact.copy(); CELL=pr2_cell.T.copy(); DE=(pr2_full-pr2_base[None,:]).T.copy()
for M in (CELL,DE): np.fill_diagonal(M,np.nan)
np.fill_diagonal(CO,np.nan); DE[low,:]=np.nan
W,X='#c04744','#3171ae'
def blocks(ax,title):
    ax.axhline(K-0.5,color='k',lw=1.2); ax.axvline(K-0.5,color='k',lw=1.2); ax.set_xticks([]);ax.set_yticks([])
    for f,l,cc in [(0.25,'net1',W),(0.75,'net2',X)]:
        ax.text(N*f,N+3,l,ha='center',va='top',fontsize=7,fontweight='bold',color=cc)
        ax.text(-3,N*f,l,ha='center',va='center',rotation=90,fontsize=7,fontweight='bold',color=cc)
    ax.set_title(title,fontsize=9,fontweight='bold',pad=6)

fig=plt.figure(figsize=(16,7.2))
gs=fig.add_gridspec(2,4,height_ratios=[1.35,1.0],hspace=0.32,wspace=0.16)
# row 1: matrices
axJ=fig.add_subplot(gs[0,0]); vmaxJ=np.percentile(np.abs(Jm[Jm!=0]),99)
imJ=axJ.imshow(Jm,cmap='RdBu_r',vmin=-vmaxJ,vmax=vmaxJ,interpolation='nearest'); blocks(axJ,'J  (ground-truth coupling)')
fig.colorbar(imJ,ax=axJ,fraction=0.046,pad=0.02)
for ax_i,(M,title,cmap,lo,hi) in zip(range(1,4),[
    (CO,'coactivity  (Pearson r)','RdBu_r',-np.nanpercentile(np.abs(CO),99),np.nanpercentile(np.abs(CO),99)),
    (CELL,'pR²(cell)','viridis',0,np.nanpercentile(CELL,99)),
    (DE,'ΔpR²  (base+cell − base)','inferno',0,np.nanpercentile(DE,95))]):
    ax=fig.add_subplot(gs[0,ax_i]); cm=matplotlib.colormaps[cmap].copy(); cm.set_bad('lightgrey')
    im=ax.imshow(M,cmap=cm,vmin=lo,vmax=hi,interpolation='nearest'); blocks(ax,title)
    fig.colorbar(im,ax=ax,fraction=0.046,pad=0.02)

# row 2: within/cross for each measure (|coact|, pR2cell, ΔpR2) + cross/within ratio
def wc(mat, directed):
    wi,cr=[],[]
    for ti in range(N):
        if low[ti]: continue
        for ci in range(N):
            if ci==ti: continue
            val = mat[ci,ti] if directed else mat[ti,ci]
            if np.isnan(val): continue
            (wi if net_id[ci]==net_id[ti] else cr).append(abs(val) if not directed else float(val))
    return np.array(wi),np.array(cr)
co_w,co_x = wc(coact,False)                 # |correlation|
ce_w,ce_x = np.array([pr2_cell[ci,ti] for ti in range(N) if not low[ti] for ci in range(N) if ci!=ti and net_id[ci]==net_id[ti] and not np.isnan(pr2_cell[ci,ti])]),\
            np.array([pr2_cell[ci,ti] for ti in range(N) if not low[ti] for ci in range(N) if ci!=ti and net_id[ci]!=net_id[ti] and not np.isnan(pr2_cell[ci,ti])])
de_w,de_x = np.array([pr2_full[ci,ti]-pr2_base[ti] for ti in range(N) if not low[ti] for ci in range(N) if ci!=ti and net_id[ci]==net_id[ti] and not np.isnan(pr2_full[ci,ti])]),\
            np.array([pr2_full[ci,ti]-pr2_base[ti] for ti in range(N) if not low[ti] for ci in range(N) if ci!=ti and net_id[ci]!=net_id[ti] and not np.isnan(pr2_full[ci,ti])])

measures=[('coactivity |r|',co_w,co_x),('pR²(cell)',ce_w,ce_x),('ΔpR²',de_w,de_x)]
# panel A: within vs cross violins (normalized per measure to its within-mean, so scales comparable)
axV=fig.add_subplot(gs[1,0:2]); pos=0; ticks=[]
for name,wi,cr in measures:
    norm=wi.mean()
    p=axV.violinplot([wi/norm,cr/norm],positions=[pos,pos+0.55],widths=0.45,showmeans=True)
    for b,cc in zip(p['bodies'],[W,X]): b.set_facecolor(cc); b.set_alpha(0.5)
    ticks.append((pos+0.27,name)); pos+=1.6
axV.axhline(0,color='k',lw=0.6,ls=':'); axV.axhline(1,color='grey',lw=0.6,ls='--')
axV.set_xticks([t for t,_ in ticks]); axV.set_xticklabels([n for _,n in ticks])
axV.set_ylabel('value / within-mean'); axV.set_title('within (red) vs cross (blue), normalised',fontsize=9,fontweight='bold')
axV.legend([mpatches.Patch(color=W,alpha=.5),mpatches.Patch(color=X,alpha=.5)],['within (J≠0)','cross (J=0)'],fontsize=8,frameon=False)
axV.spines[['top','right']].set_visible(False)
# panel B: cross/within ratio (the false-positive floor) — lower is better
axR=fig.add_subplot(gs[1,2])
ratios=[cr.mean()/wi.mean() for _,wi,cr in measures]
axR.bar(range(3),ratios,color=['#888','#888',W])
axR.set_xticks(range(3)); axR.set_xticklabels([n for n,_,_ in measures],fontsize=8)
axR.set_ylabel('cross / within  (false-positive floor)')
for i,r in enumerate(ratios): axR.text(i,r+0.01,f'{r:.2f}',ha='center',fontsize=9)
axR.set_title('cross-net leakage\n(lower = better)',fontsize=9,fontweight='bold'); axR.spines[['top','right']].set_visible(False)
# panel C: text takeaway
axT=fig.add_subplot(gs[1,3]); axT.axis('off')
axT.text(0.0,0.95,'Coactivity shows strong cross-network\nstructure — but those pairs have J=0.',fontsize=8,va='top',color=X)
axT.text(0.0,0.62,f'cross/within:\n  coactivity |r| = {ratios[0]:.2f}\n  pR²(cell)   = {ratios[1]:.2f}\n  ΔpR²        = {ratios[2]:.2f}',fontsize=8,va='top',family='monospace')
axT.text(0.0,0.18,'ΔpR² conditions out the shared drive,\ncollapsing the cross-net floor.',fontsize=8,va='top',color=W)

fig.suptitle('Why not just use coactivity? — J vs coactivity vs pR²(cell) vs ΔpR² (two disconnected grid nets)',fontsize=12,fontweight='bold')
plt.savefig(FIG+'/coactivity_vs_dpr2.png',dpi=120,bbox_inches='tight')
plt.savefig(FIG+'/coactivity_vs_dpr2.pdf',bbox_inches='tight')
print(f'cross/within — coactivity={ratios[0]:.3f} pR2cell={ratios[1]:.3f} dpr2={ratios[2]:.3f}', flush=True)
print('saved coactivity_vs_dpr2.png/pdf', flush=True); print('COACT_DONE', flush=True)
