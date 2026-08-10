"""Confirm both DIFFERENT-place-cell Ng=1024 nets form grids and path-integrate."""
import sys, os
REPO_DIR='/Users/harryclark/Documents/spatial-manifolds/GRID-PATTERN-FORMATION'
sys.path.insert(0, REPO_DIR)
import numpy as np, torch
import matplotlib; matplotlib.use('Agg'); import matplotlib.pyplot as plt
from place_cells import PlaceCells
from trajectory_generator import TrajectoryGenerator
from model import RNN
from scores import GridScorer
from visualize import compute_ratemaps

DATA='/Users/harryclark/Documents/spatial-manifolds/data/rnn_xgboost'
FIG='/Users/harryclark/Documents/spatial-manifolds/scripts/figures/figure_validations'
RES,N_AVG=40,50; DEVICE='mps' if torch.backends.mps.is_available() else 'cpu'
NETS={'A':(DATA+'/diffPC_A_Ng1024','1234/pc101'), 'B':(DATA+'/diffPC_B_Ng1024','5678/pc202')}

def build(d):
    class O: pass
    o=O(); o.Np=512;o.Ng=1024;o.sequence_length=20;o.batch_size=200
    o.learning_rate=1e-4;o.weight_decay=1e-4;o.place_cell_rf=0.12;o.surround_scale=2
    o.RNN_type='RNN';o.activation='relu';o.DoG=True;o.periodic=False
    o.box_width=2.2;o.box_height=2.2;o.device=DEVICE
    pc=PlaceCells(o); pc.us=torch.tensor(np.load(d+'/place_cells.npy'),dtype=torch.float32).to(DEVICE)
    m=RNN(o,pc).to(DEVICE); m.load_state_dict(torch.load(d+'/ckpt.pth',map_location=DEVICE)['model']); m.eval()
    return o,pc,m

built={k:build(d) for k,(d,_) in NETS.items()}
coord=((-1.1,1.1),(-1.1,1.1)); scorer=GridScorer(RES,coord,zip([0.2]*10,np.linspace(0.4,1.0,10).tolist()))
info={}; acts={}
for k,(o,pc,m) in built.items():
    tg=TrajectoryGenerator(o,pc)
    a,_,_,_=compute_ratemaps(m,tg,o,res=RES,n_avg=N_AVG,Ng=o.Ng)
    sc=np.array([scorer.get_scores(a[i])[0] for i in range(o.Ng)])
    with torch.no_grad():
        inp,pos,_=tg.get_test_batch(); pred=pc.get_nearest_cell_pos(m.predict(inp)).cpu()
    err=torch.sqrt(((pos.cpu()-pred)**2).sum(-1)).mean().item()*100
    v=sc[~np.isnan(sc)]; info[k]=dict(err=err,med=np.median(v),frac=(v>0.3).mean(),n=(v>0.3).sum(),tg=tg)
    acts[k]=(a,sc)
    print(f'net {k} ({NETS[k][1]}): decode={err:.1f} cm | grid median={np.median(v):.2f} | frac>0.3={(v>0.3).mean():.2f} | n>0.3={(v>0.3).sum()}/1024', flush=True)

fig=plt.figure(figsize=(15,7)); gs=fig.add_gridspec(3,10,height_ratios=[1,1,1.4],hspace=0.35,wspace=0.05)
for row,k in enumerate(['A','B']):
    a,sc=acts[k]; order=np.flip(np.argsort(np.nan_to_num(sc,nan=-1)))
    for c in range(10):
        ax=fig.add_subplot(gs[row,c]); rm=a[order[c]]
        ax.imshow(rm.T,origin='lower',cmap='jet',interpolation='gaussian'); ax.set_xticks([]);ax.set_yticks([])
        ax.set_title(f'{sc[order[c]]:.2f}',fontsize=6)
        if c==0: ax.set_ylabel(f'net {"1" if k=="A" else "2"}\n({NETS[k][1]})',fontsize=8,fontweight='bold')
for i,k in enumerate(['A','B']):
    o,pc,m=built[k]; tg=info[k]['tg']
    with torch.no_grad():
        inp,pos,_=tg.get_test_batch(); pred=pc.get_nearest_cell_pos(m.predict(inp)).cpu()
    pos=pos.cpu(); ax=fig.add_subplot(gs[2,i*5:i*5+5])
    for j in range(5):
        ax.plot(pos[:,j,0],pos[:,j,1],c='k',lw=2,label='true' if j==0 else None)
        ax.plot(pred[:,j,0],pred[:,j,1],'.-',c='C1',label='decoded' if j==0 else None)
    ax.scatter(pc.us.cpu()[:,0],pc.us.cpu()[:,1],s=8,alpha=0.3,c='lightgrey')
    ax.set_xlim(-1.1,1.1);ax.set_ylim(-1.1,1.1);ax.set_xticks([]);ax.set_yticks([])
    ax.set_title(f'net {"1" if k=="A" else "2"} ({NETS[k][1]}) — {info[k]["err"]:.1f} cm',fontsize=9); ax.legend(fontsize=7,frameon=False)
fig.suptitle('DIFFERENT place-cell nets (Ng=1024): grid formation (top) + path integration (bottom)',fontsize=11,fontweight='bold')
plt.savefig(FIG+'/two_network_diffPC_validation.png',dpi=120,bbox_inches='tight'); plt.savefig(FIG+'/two_network_diffPC_validation.pdf',bbox_inches='tight')
print('saved two_network_diffPC_validation.png/pdf', flush=True); print('VALIDATION_DONE', flush=True)
