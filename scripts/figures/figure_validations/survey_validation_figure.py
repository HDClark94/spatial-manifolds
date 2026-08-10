"""10-seed survey: per-net grid cells (left) + path-integration decoding (right)."""
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
RES,N_AVG,N_SHOW=40,50,5
DEVICE='mps' if torch.backends.mps.is_available() else 'cpu'

NETS=[
 ('net 1','1234/pc101', DATA+'/diffPC_A_Ng1024'),
 ('net 2','5678/pc202', DATA+'/diffPC_B_Ng1024'),
 ('net 3','1111/pc11',  DATA+'/survey_s3_Ng1024'),
 ('net 4','2222/pc22',  DATA+'/survey_s4_Ng1024'),
 ('net 5','3333/pc33',  DATA+'/survey_s5_Ng1024'),
 ('net 6','4444/pc44',  DATA+'/survey_s6_Ng1024'),
 ('net 7','5555/pc55',  DATA+'/survey_s7_Ng1024'),
 ('net 8','6666/pc66',  DATA+'/survey_s8_Ng1024'),
 ('net 9','7777/pc77',  DATA+'/survey_s9_Ng1024'),
 ('net 10','8888/pc88', DATA+'/survey_s10_Ng1024'),
]

def build(d):
    class O: pass
    o=O(); o.Np=512;o.Ng=1024;o.sequence_length=20;o.batch_size=200
    o.learning_rate=1e-4;o.weight_decay=1e-4;o.place_cell_rf=0.12;o.surround_scale=2
    o.RNN_type='RNN';o.activation='relu';o.DoG=True;o.periodic=False
    o.box_width=2.2;o.box_height=2.2;o.device=DEVICE
    pc=PlaceCells(o); pc.us=torch.tensor(np.load(d+'/place_cells.npy'),dtype=torch.float32).to(DEVICE)
    m=RNN(o,pc).to(DEVICE); m.load_state_dict(torch.load(d+'/ckpt.pth',map_location=DEVICE)['model']); m.eval()
    return o,pc,m

scorer=GridScorer(RES,((-1.1,1.1),(-1.1,1.1)),zip([0.2]*10,np.linspace(0.4,1.0,10).tolist()))
rows=[]
for label,seed,d in NETS:
    if not os.path.exists(d+'/ckpt.pth'):
        print(f'skip {label}: no ckpt at {d}', flush=True); continue
    o,pc,m=build(d); tg=TrajectoryGenerator(o,pc)
    a,_,_,_=compute_ratemaps(m,tg,o,res=RES,n_avg=N_AVG,Ng=o.Ng)
    sc=np.array([scorer.get_scores(a[i])[0] for i in range(o.Ng)])
    with torch.no_grad():
        inp,pos,_=tg.get_test_batch(); pred=pc.get_nearest_cell_pos(m.predict(inp)).cpu()
    err=torch.sqrt(((pos.cpu()-pred)**2).sum(-1)).mean().item()*100
    v=sc[~np.isnan(sc)]
    # GC×GC J block (phase-sorted) + mean-J vs phase-difference curve
    gc=np.where(sc>0.3)[0]; Jw=m.RNN.weight_hh_l0.detach().cpu().numpy()
    Jgc=None; ph_c=None; ph_exc=None; ph_inh=None
    if len(gc)>=8:
        rm_fft=np.array([np.fft.fft2(a[gi]) for gi in gc])
        mp=(np.abs(rm_fft)**2).mean(0); mp[0,0]=0
        half=RES//2; kp=np.unravel_index(mp[:half,:half].argmax(),(half,half))
        ph=np.angle(rm_fft[:,kp[0],kp[1]]); srt=np.argsort(ph)
        gc_sorted=gc[srt]; ph_sorted=ph[srt]
        Jgc=Jw[np.ix_(gc_sorted,gc_sorted)]                       # Jgc[i,j] = weight source j -> target i
        pd=(ph_sorted[:,None]-ph_sorted[None,:]+np.pi)%(2*np.pi)-np.pi   # phase_target - phase_source
        offdiag=~np.eye(len(gc_sorted),dtype=bool)
        edges=np.linspace(-np.pi,np.pi,21); ph_c=(edges[:-1]+edges[1:])/2
        ph_exc=[]; ph_inh=[]
        for lo,hi in zip(edges[:-1],edges[1:]):
            vals=Jgc[offdiag&(pd>=lo)&(pd<hi)]
            e=vals[vals>0]; ii=vals[vals<0]
            ph_exc.append(e.mean() if len(e) else np.nan)
            ph_inh.append(ii.mean() if len(ii) else np.nan)
        ph_exc=np.array(ph_exc); ph_inh=np.array(ph_inh)
    rows.append(dict(label=label,seed=seed,pc=pc,pos=pos.cpu(),pred=pred,a=a,sc=sc,
                     err=err,med=np.median(v),frac=(v>0.3).mean(),
                     Jgc=Jgc,ph_c=ph_c,ph_exc=ph_exc,ph_inh=ph_inh,n_gc=len(gc)))
    print(f'{label} ({seed}): decode={err:.1f}cm grid_med={np.median(v):.2f} frac>0.3={(v>0.3).mean():.2f}', flush=True)

nrow=len(rows)
fig=plt.figure(figsize=(13.5, 1.3*nrow+0.7))
gs=fig.add_gridspec(nrow, N_SHOW+3, width_ratios=[1]*N_SHOW+[1.7,1.7,2.6], hspace=0.28, wspace=0.06)
for r,R in enumerate(rows):
    order=np.flip(np.argsort(np.nan_to_num(R['sc'],nan=-1)))
    for cidx in range(N_SHOW):
        ax=fig.add_subplot(gs[r,cidx]); rm=R['a'][order[cidx]]
        ax.imshow(rm.T,origin='lower',cmap='jet',interpolation='gaussian'); ax.set_xticks([]);ax.set_yticks([])
        ax.set_title(f"{R['sc'][order[cidx]]:.2f}",fontsize=5,pad=1)
        if cidx==0:
            ax.set_ylabel(f"{R['label']}\n{R['seed']}\nmed {R['med']:.2f} | {R['err']:.0f}cm",
                          fontsize=6,fontweight='bold')
    # GC×GC J block, phase-sorted (zoomed to grid cells)
    axJ=fig.add_subplot(gs[r,N_SHOW])
    if R['Jgc'] is not None:
        vmax=np.percentile(np.abs(R['Jgc']),99)
        axJ.imshow(R['Jgc'],cmap='RdBu_r',vmin=-vmax,vmax=vmax,aspect='auto',interpolation='nearest')
    axJ.set_xticks([]);axJ.set_yticks([]); axJ.set_ylabel(f"{R['n_gc']} GC",fontsize=5)
    if r==0: axJ.set_title('J: GC×GC\n(phase-sorted)',fontsize=6)
    # mean excitatory / inhibitory J vs phase difference (Δφ≈0 = similar phase)
    axC=fig.add_subplot(gs[r,N_SHOW+1])
    if R['ph_exc'] is not None:
        axC.plot(R['ph_c'],R['ph_exc'],color='#c04744',lw=1.2,label='exc J>0')
        axC.plot(R['ph_c'],R['ph_inh'],color='#3171ae',lw=1.2,label='inh J<0')
        axC.axhline(0,color='grey',lw=0.5,ls=':'); axC.axvline(0,color='grey',lw=0.5,ls=':')
        if r==0: axC.legend(fontsize=4,frameon=False,loc='center right')
    axC.set_xticks([-np.pi,0,np.pi]); axC.set_xticklabels(['-π','0','π'],fontsize=5)
    axC.tick_params(labelsize=4.5); axC.spines[['top','right']].set_visible(False)
    if r==0: axC.set_title('mean J vs Δphase\n(exc / inh)',fontsize=6)
    # decoding
    ax=fig.add_subplot(gs[r,N_SHOW+2]); us=R['pc'].us.cpu()
    ax.scatter(us[:,0],us[:,1],s=4,alpha=0.25,c='lightgrey')
    for j in range(5):
        ax.plot(R['pos'][:,j,0],R['pos'][:,j,1],c='k',lw=1.1)
        ax.plot(R['pred'][:,j,0],R['pred'][:,j,1],'.-',c='C1',ms=2,lw=0.7)
    ax.set_xlim(-1.1,1.1);ax.set_ylim(-1.1,1.1);ax.set_xticks([]);ax.set_yticks([])
    if r==0: ax.set_title('place cells (grey) · true (black) · decoded (orange)',fontsize=6)
fig.suptitle('10-seed survey (Ng=1024, 50k steps): grid cells (left) + path integration (right)',
             fontsize=11,fontweight='bold')
plt.savefig(FIG+'/two_network_diffPC_validation.png',dpi=130,bbox_inches='tight')
plt.savefig(FIG+'/two_network_diffPC_validation.pdf',bbox_inches='tight')

print('\n=== SURVEY SUMMARY ===', flush=True)
for R in rows:
    print(f"{R['label']:7s} {R['seed']:12s}: decode={R['err']:5.1f}cm  grid_med={R['med']:.2f}  frac>0.3={R['frac']:.2f}", flush=True)
print('SURVEY_FIGURE_DONE', flush=True)
