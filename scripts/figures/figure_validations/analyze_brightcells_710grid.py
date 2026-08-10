"""Identify & characterize the 'predicted-by-everything' target cells (bright rows) in the
net7+net10 710grid pR2(cell) matrix. Hypothesis: they are the most cleanly position-tuned,
low-noise cells -> any position-coding cell (either net) predicts them."""
import sys, os
REPO='/Users/harryclark/Documents/spatial-manifolds/GRID-PATTERN-FORMATION'
sys.path.insert(0,REPO); sys.path.insert(0,'/Users/harryclark/Documents/spatial-manifolds/src')
import numpy as np, torch
import matplotlib; matplotlib.use('Agg'); import matplotlib.pyplot as plt
from place_cells import PlaceCells
from trajectory_generator import TrajectoryGenerator
from model import RNN
from visualize import compute_ratemaps
from scores import GridScorer
DATA='/Users/harryclark/Documents/spatial-manifolds/data/rnn_xgboost'; FIG=DATA.replace('/data/rnn_xgboost','/scripts/figures/figure_validations')
CACHE=DATA+'/pr2_two_net_710grid_cache.npz'; DEVICE='mps' if torch.backends.mps.is_available() else 'cpu'; RES,N_AVG=40,50
c=np.load(CACHE,allow_pickle=True)
pr2_base=c['pr2_base']; pr2_cell=c['pr2_cell']; net_id=c['net_id']; grid_sc=c['grid_sc']; sel1=c['sel1']; sel2=c['sel2']; K=int(c['K']); N=len(net_id)

# mean pR2(cell) each target is predicted with — over all / within-net / cross-net sources
def m(ti,mask):
    idx=mask&(np.arange(N)!=ti); v=pr2_cell[idx,ti]; return np.nanmean(v)
allm=np.array([m(ti,np.ones(N,bool)) for ti in range(N)])
winm=np.array([m(ti,net_id==net_id[ti]) for ti in range(N)])
crom=np.array([m(ti,net_id!=net_id[ti]) for ti in range(N)])

def build(d):
    class O: pass
    o=O(); o.Np=512;o.Ng=1024;o.sequence_length=20;o.batch_size=200
    o.learning_rate=1e-4;o.weight_decay=1e-4;o.place_cell_rf=0.12;o.surround_scale=2
    o.RNN_type='RNN';o.activation='relu';o.DoG=True;o.periodic=False;o.box_width=2.2;o.box_height=2.2;o.device=DEVICE
    pc=PlaceCells(o); pc.us=torch.tensor(np.load(d+'/place_cells.npy'),dtype=torch.float32).to(DEVICE)
    mm=RNN(o,pc).to(DEVICE); mm.load_state_dict(torch.load(d+'/ckpt.pth',map_location=DEVICE)['model']); mm.eval(); return o,pc,mm
o1,pcA,mA=build(DATA+'/survey_s7_Ng1024'); o2,pcB,mB=build(DATA+'/survey_s10_Ng1024')
a1,_,_,_=compute_ratemaps(mA,TrajectoryGenerator(o1,pcA),o1,res=RES,n_avg=N_AVG,Ng=1024)
a2,_,_,_=compute_ratemaps(mB,TrajectoryGenerator(o2,pcB),o2,res=RES,n_avg=N_AVG,Ng=1024)
RM={0:a1[sel1],1:a2[sel2]}  # rate maps of the SELECTED cells, per net

fig,ax=plt.subplots(2,4,figsize=(15,7.5))
for row,(g,lab,A) in enumerate([(0,'net7 (grid)',a1),(1,'net10 (non-grid)',a2)]):
    idx=np.where(net_id==g)[0]
    ti=idx[np.nanargmax(allm[idx])]; local=ti-g*K            # brightest target
    lo=idx[np.nanargmin(allm[idx])]; locallo=lo-g*K          # least-predicted target (contrast)
    unit=(sel1 if g==0 else sel2)[local]
    def stats(rm):
        r=rm.ravel(); r=r-r.min(); mean=r.mean(); return dict(peak=rm.max(),mean=rm.mean(),
            sparsity=(mean**2)/ (np.mean(r**2)+1e-9))   # ~1 = uniform/dense, low = sparse
    st=stats(RM[g][local])
    baserank=int((pr2_base[idx]>pr2_base[ti]).sum())     # 0 = highest base pR2 in its net
    print(f'\n=== {lab}: BRIGHT cell = selected#{local} (unit {unit}) ===')
    print(f'  predicted-with mean pR2(cell): all={allm[ti]:.3f}  within-net-sources={winm[ti]:.3f}  cross-net-sources={crom[ti]:.3f}')
    print(f'  base pR2 (pos+speed+HD): {pr2_base[ti]:.3f}  (rank {baserank}/{K} in {lab}; net median={np.median(pr2_base[idx]):.3f})')
    print(f'  grid score: {grid_sc[ti]:.2f}   rate: mean={st["mean"]:.3f} peak={st["peak"]:.3f} density(1=uniform)={st["sparsity"]:.2f}')
    print(f'  vs LEAST-predicted cell #{locallo}: all pR2={allm[lo]:.3f}, base pR2={pr2_base[lo]:.3f}, grid={grid_sc[lo]:.2f}')
    # panels: bright rate map | least-predicted rate map | base pR2 vs mean-pred scatter | mean-pred bar
    ax[row,0].imshow(RM[g][local],cmap='jet'); ax[row,0].set_title(f'{lab}\nBRIGHT cell (unit {unit})\nbasepR²={pr2_base[ti]:.2f} grid={grid_sc[ti]:.2f}',fontsize=8,fontweight='bold'); ax[row,0].axis('off')
    ax[row,1].imshow(RM[g][locallo],cmap='jet'); ax[row,1].set_title(f'least-predicted cell #{locallo}\nbasepR²={pr2_base[lo]:.2f} grid={grid_sc[lo]:.2f}',fontsize=8); ax[row,1].axis('off')
    axs=ax[row,2]; axs.scatter(pr2_base[idx],allm[idx],s=18,color='#888'); axs.scatter(pr2_base[ti],allm[ti],s=60,color='r',zorder=3,label='bright')
    from scipy.stats import spearmanr; rho=spearmanr(pr2_base[idx],allm[idx])[0]
    axs.set_xlabel('base pR² (position-tuned-ness)'); axs.set_ylabel('mean pR²(cell) it is predicted with'); axs.set_title(f'{lab}: ρ={rho:+.2f}',fontsize=8,fontweight='bold'); axs.legend(fontsize=7,frameon=False)
    axb=ax[row,3]; order=np.argsort(allm[idx]); axb.bar(range(K),allm[idx][order],color=['r' if idx[o]==ti else '#888' for o in order]); axb.set_title(f'{lab}: mean pR²(cell) per target',fontsize=8); axb.set_xlabel('cells (sorted)'); axb.set_ylabel('mean pR²(cell)')
plt.suptitle("'Predicted-by-everything' cells: the most cleanly position-tuned, low-noise units",fontsize=12,fontweight='bold')
plt.tight_layout(rect=[0,0,1,0.96]); plt.savefig(FIG+'/brightcells_710grid.png',dpi=120,bbox_inches='tight')
print('\nsaved brightcells_710grid.png'); print('BRIGHT_DONE')
