"""What makes a cell 'predicted by any cell' (bright row in pR2(cell))? Correlate each cell's
row-brightness (mean pR2(cell) it is predicted with) against mean rate, broadness (Treves-Rolls
sparsity), base pR2, grid score — across all 80 cells, per net."""
import sys, os
REPO='/Users/harryclark/Documents/spatial-manifolds/GRID-PATTERN-FORMATION'
sys.path.insert(0,REPO); sys.path.insert(0,'/Users/harryclark/Documents/spatial-manifolds/src')
import numpy as np, torch
import matplotlib; matplotlib.use('Agg'); import matplotlib.pyplot as plt
from scipy.stats import spearmanr
from place_cells import PlaceCells
from trajectory_generator import TrajectoryGenerator
from model import RNN
from visualize import compute_ratemaps
DATA='/Users/harryclark/Documents/spatial-manifolds/data/rnn_xgboost'
FIG='/Users/harryclark/Documents/spatial-manifolds/scripts/figures/figure_validations'
CACHE=DATA+'/pr2_two_net_710grid_cache.npz'; DEVICE='mps' if torch.backends.mps.is_available() else 'cpu'; RES,N_AVG=40,50
c=np.load(CACHE,allow_pickle=True)
pr2_base=c['pr2_base']; pr2_cell=c['pr2_cell']; net_id=c['net_id']; grid_sc=c['grid_sc']; sel1=c['sel1']; sel2=c['sel2']; K=int(c['K']); N=len(net_id)
rowbright=np.array([np.nanmean(pr2_cell[np.arange(N)!=ti,ti]) for ti in range(N)])

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
RM=np.concatenate([a1[sel1],a2[sel2]],0).reshape(N,-1)     # rate maps of the 80 selected cells
meanrate=RM.mean(1)
r=RM-RM.min(1,keepdims=True)
broadness=(r.mean(1)**2)/(np.mean(r**2,1)+1e-9)            # Treves-Rolls: ~1 dense/broad, low = sparse

print('Per-net Spearman of row-brightness (predicted-by-any) vs cell properties:')
print(f'{"net":16s} | mean rate | broadness | base pR² | grid score')
for g,lab in [(0,'net7 (grid)'),(1,'net10 (nongrid)')]:
    idx=net_id==g
    print(f'{lab:16s} | {spearmanr(rowbright[idx],meanrate[idx])[0]:+.2f}      | '
          f'{spearmanr(rowbright[idx],broadness[idx])[0]:+.2f}      | '
          f'{spearmanr(rowbright[idx],pr2_base[idx])[0]:+.2f}     | {spearmanr(rowbright[idx],grid_sc[idx])[0]:+.2f}')
# how graded? top cells
for g,lab,sel in [(0,'net7',sel1),(1,'net10',sel2)]:
    idx=np.where(net_id==g)[0]; order=idx[np.argsort(-rowbright[idx])]
    thr=rowbright[idx].mean()+rowbright[idx].std()
    print(f'\n{lab}: {int((rowbright[idx]>thr).sum())}/{K} cells above mean+1SD | top5 rowbright='
          +', '.join(f'{rowbright[o]:.2f}(rate{meanrate[o]:.2f},broad{broadness[o]:.2f})' for o in order[:5]))

fig,ax=plt.subplots(1,2,figsize=(11,4.6))
for g,col,lab in [(0,'#c04744','net7'),(1,'#3171ae','net10')]:
    idx=net_id==g
    ax[0].scatter(meanrate[idx],rowbright[idx],color=col,label=lab,s=22)
    ax[1].scatter(broadness[idx],rowbright[idx],color=col,label=lab,s=22)
ax[0].set_xlabel('mean firing rate'); ax[0].set_ylabel('row-brightness (mean pR²(cell) predicted-with)'); ax[0].set_title('vs mean rate',fontweight='bold'); ax[0].legend(frameon=False)
ax[1].set_xlabel('broadness (Treves-Rolls, 1=dense)'); ax[1].set_ylabel('row-brightness'); ax[1].set_title('vs broadness',fontweight='bold'); ax[1].legend(frameon=False)
for a in ax: a.spines[['top','right']].set_visible(False)
plt.tight_layout(); plt.savefig(FIG+'/rowstructure_710grid.png',dpi=120,bbox_inches='tight')
print('\nsaved rowstructure_710grid.png'); print('ROW_DONE')
