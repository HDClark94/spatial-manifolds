"""Grid emergence across training for net 3: grid-score-vs-step curve + rate-map montage."""
import sys, os, glob
REPO_DIR='/Users/harryclark/Documents/spatial-manifolds/GRID-PATTERN-FORMATION'
sys.path.insert(0, REPO_DIR)
import numpy as np, torch
import matplotlib; matplotlib.use('Agg'); import matplotlib.pyplot as plt
from place_cells import PlaceCells
from trajectory_generator import TrajectoryGenerator
from model import RNN
from scores import GridScorer
from visualize import compute_ratemaps

D='/Users/harryclark/Documents/spatial-manifolds/data/rnn_xgboost/survey_s3_Ng1024'
FIG='/Users/harryclark/Documents/spatial-manifolds/scripts/figures/figure_validations'
RES,N_AVG,N_SHOW=40,50,6
DEVICE='mps' if torch.backends.mps.is_available() else 'cpu'

snaps=sorted(glob.glob(D+'/snap_*.pth'),
             key=lambda p:int(os.path.basename(p).replace('snap_','').replace('.pth','')))
print('snapshots:', [os.path.basename(s) for s in snaps], flush=True)

class O: pass
o=O(); o.Np=512;o.Ng=1024;o.sequence_length=20;o.batch_size=200
o.learning_rate=1e-4;o.weight_decay=1e-4;o.place_cell_rf=0.12;o.surround_scale=2
o.RNN_type='RNN';o.activation='relu';o.DoG=True;o.periodic=False
o.box_width=2.2;o.box_height=2.2;o.device=DEVICE
pc=PlaceCells(o); pc.us=torch.tensor(np.load(D+'/place_cells.npy'),dtype=torch.float32).to(DEVICE)
m=RNN(o,pc).to(DEVICE); tg=TrajectoryGenerator(o,pc)
scorer=GridScorer(RES,((-1.1,1.1),(-1.1,1.1)),zip([0.2]*10,np.linspace(0.4,1.0,10).tolist()))

steps=[]; med=[]; frac=[]; mx=[]; top_rms=[]
for s in snaps:
    step=int(os.path.basename(s).replace('snap_','').replace('.pth',''))
    m.load_state_dict(torch.load(s,map_location=DEVICE)); m.eval()
    a,_,_,_=compute_ratemaps(m,tg,o,res=RES,n_avg=N_AVG,Ng=o.Ng)
    sc=np.array([scorer.get_scores(a[i])[0] for i in range(o.Ng)])
    v=sc[~np.isnan(sc)]
    steps.append(step); med.append(np.median(v)); frac.append((v>0.3).mean()); mx.append(v.max())
    order=np.flip(np.argsort(np.nan_to_num(sc,nan=-1)))
    top_rms.append((a[order[:N_SHOW]].copy(), sc[order[:N_SHOW]].copy()))
    print(f"step {step}: med={np.median(v):+.2f} frac>0.3={(v>0.3).mean():.2f} max={v.max():.2f}", flush=True)

nrow=len(snaps)
fig=plt.figure(figsize=(2+N_SHOW*1.5, 1.4*nrow+1.4))
gs=fig.add_gridspec(nrow+1, N_SHOW, height_ratios=[1.6]+[1]*nrow, hspace=0.35, wspace=0.05)
# top: metrics vs step
axm=fig.add_subplot(gs[0,:])
axm.plot(steps,med,'o-',color='#c04744',label='grid score median')
axm.plot(steps,mx,'s--',color='#e0a0a0',label='max grid score',alpha=0.8)
axm.axhline(0.3,color='grey',ls=':',lw=1,label='grid threshold 0.3')
axm.set_xlabel('training step'); axm.set_ylabel('grid score')
ax2=axm.twinx(); ax2.plot(steps,frac,'^-',color='#3171ae',label='frac cells >0.3')
ax2.set_ylabel('frac grid cells',color='#3171ae'); ax2.tick_params(axis='y',labelcolor='#3171ae')
axm.legend(fontsize=7,loc='upper left',frameon=False); ax2.legend(fontsize=7,loc='lower right',frameon=False)
axm.set_title('Net 3 (1111/pc11): grid-cell emergence vs training',fontsize=11,fontweight='bold')
axm.spines[['top']].set_visible(False)
# rows: top ratemaps per snapshot
for r,(rms,scs) in enumerate(top_rms):
    for c in range(N_SHOW):
        ax=fig.add_subplot(gs[r+1,c]); ax.imshow(rms[c].T,origin='lower',cmap='jet',interpolation='gaussian')
        ax.set_xticks([]);ax.set_yticks([]); ax.set_title(f'{scs[c]:.2f}',fontsize=5,pad=1)
        if c==0: ax.set_ylabel(f'{steps[r]//1000}k',fontsize=7,fontweight='bold')
plt.savefig(FIG+'/grid_emergence_net3.png',dpi=120,bbox_inches='tight')
plt.savefig(FIG+'/grid_emergence_net3.pdf',bbox_inches='tight')
print('saved grid_emergence_net3.png/pdf', flush=True); print('EMERGENCE_DONE', flush=True)
