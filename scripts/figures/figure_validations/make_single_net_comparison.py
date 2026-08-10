"""
SINGLE-NETWORK comparison: within one network (J known), does ΔpR² beat coactivity & pR2(cell)?
Uses the WITHIN-net pairs of the 710grid cache (net7 grid, net10 non-grid) as two single-net examples.
For each net: |J|-recovery (Spearman of each measure with |J|) and tuning-dependence (Spearman with
|rate-map correlation|). The expected story: coactivity/pR2(cell) recover |J| better but ride tuning;
ΔpR² recovers weaker but is tuning-independent.
"""
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
DATA='/Users/harryclark/Documents/spatial-manifolds/data/rnn_xgboost'; FIG=DATA.replace('data/rnn_xgboost','scripts/figures/figure_validations')
CACHE=DATA+'/pr2_two_net_710grid_cache.npz'; DEVICE='mps' if torch.backends.mps.is_available() else 'cpu'; RES,N_AVG,N_BATCHES=40,50,12
c=np.load(CACHE,allow_pickle=True)
pr2_base=c['pr2_base']; pr2_cell=c['pr2_cell']; pr2_full=c['pr2_full']; net_id=c['net_id']; J=c['J_comb']; sel1=c['sel1']; sel2=c['sel2']; K=int(c['K']); N=len(net_id)
def build(d):
    class O: pass
    o=O(); o.Np=512;o.Ng=1024;o.sequence_length=20;o.batch_size=200
    o.learning_rate=1e-4;o.weight_decay=1e-4;o.place_cell_rf=0.12;o.surround_scale=2
    o.RNN_type='RNN';o.activation='relu';o.DoG=True;o.periodic=False;o.box_width=2.2;o.box_height=2.2;o.device=DEVICE
    pc=PlaceCells(o); pc.us=torch.tensor(np.load(d+'/place_cells.npy'),dtype=torch.float32).to(DEVICE)
    m=RNN(o,pc).to(DEVICE); m.load_state_dict(torch.load(d+'/ckpt.pth',map_location=DEVICE)['model']); m.eval(); return o,pc,m
o1,pcA,mA=build(DATA+'/survey_s7_Ng1024'); o2,pcB,mB=build(DATA+'/survey_s10_Ng1024'); tgA=TrajectoryGenerator(o1,pcA)
a1,_,_,_=compute_ratemaps(mA,tgA,o1,res=RES,n_avg=N_AVG,Ng=1024); a2,_,_,_=compute_ratemaps(mB,TrajectoryGenerator(o2,pcB),o2,res=RES,n_avg=N_AVG,Ng=1024)
rmcorr=np.corrcoef(np.concatenate([a1[sel1],a2[sel2]],0).reshape(N,-1))
def jb():
    t=tgA.generate_trajectory(o1.box_width,o1.box_height,o1.batch_size)
    v=np.stack([t['ego_v']*np.cos(t['target_hd']),t['ego_v']*np.sin(t['target_hd'])],-1); v=torch.tensor(v,dtype=torch.float32).transpose(0,1).to(DEVICE)
    ip=np.stack([t['init_x'],t['init_y']],-1); ip=torch.tensor(ip,dtype=torch.float32).to(DEVICE)
    return (v,pcA.get_activation(ip).squeeze()),(v,pcB.get_activation(ip).squeeze())
torch.manual_seed(1); np.random.seed(1)
G1,G2=[],[]
with torch.no_grad():
    for _ in range(N_BATCHES):
        a,b=jb(); G1.append(mA.g(a).cpu().numpy().transpose(1,0,2).reshape(-1,o1.Ng)[:,sel1]); G2.append(mB.g(b).cpu().numpy().transpose(1,0,2).reshape(-1,o2.Ng)[:,sel2])
coact=np.corrcoef(np.concatenate([np.concatenate(G1,0),np.concatenate(G2,0)],1).T)
low=pr2_base<0.3

nets=[(0,'net7 (grid)'),(1,'net10 (non-grid)')]; measures=['coactivity','pR²(cell)','ΔpR²']
recJ={}; recT={}
print(f'{"network":18s} | {"":14s} coactivity | pR²(cell) | ΔpR²')
for g,lab in nets:
    idx=np.where(net_id==g)[0]; aj=[];co=[];ce=[];de=[];tu=[]
    for ti in idx:
        if low[ti]: continue
        for ci in idx:
            if ci==ti or np.isnan(pr2_full[ci,ti]): continue
            aj.append(abs(float(J[ti,ci]))); co.append(abs(coact[ci,ti])); ce.append(pr2_cell[ci,ti]); de.append(pr2_full[ci,ti]-pr2_base[ti]); tu.append(abs(rmcorr[ci,ti]))
    aj,co,ce,de,tu=map(np.array,(aj,co,ce,de,tu))
    rJ=[spearmanr(aj,v)[0] for v in (co,ce,de)]; rT=[spearmanr(tu,v)[0] for v in (co,ce,de)]
    recJ[lab]=rJ; recT[lab]=rT
    print(f'{lab:18s} | ρ(|J|)  recovery : {rJ[0]:+.2f}      | {rJ[1]:+.2f}    | {rJ[2]:+.2f}   (n={len(aj)})')
    print(f'{"":18s} | ρ(tuning) depend.: {rT[0]:+.2f}      | {rT[1]:+.2f}    | {rT[2]:+.2f}')

fig,ax=plt.subplots(1,2,figsize=(12,4.6)); x=np.arange(3); w=0.36; cols=['#c04744','#3171ae']
for j,(g,lab) in enumerate(nets):
    ax[0].bar(x+(j-0.5)*w,recJ[lab],w,color=cols[j],label=lab)
    ax[1].bar(x+(j-0.5)*w,recT[lab],w,color=cols[j],label=lab)
for a,ttl,ylab in [(ax[0],'|J|-recovery (higher = better)\nSpearman(measure, |J|)','ρ with |J|'),(ax[1],'tuning-dependence (lower = cleaner)\nSpearman(measure, |rate-map corr|)','ρ with |tuning|')]:
    a.set_xticks(x); a.set_xticklabels(measures); a.set_ylabel(ylab); a.set_title(ttl,fontsize=10,fontweight='bold'); a.axhline(0,color='grey',lw=0.6); a.legend(frameon=False,fontsize=8); a.spines[['top','right']].set_visible(False)
plt.suptitle('Single-network comparison: does ΔpR² beat coactivity & pR²(cell)?',fontsize=12,fontweight='bold')
plt.tight_layout(rect=[0,0,1,0.95]); plt.savefig(FIG+'/single_net_comparison.png',dpi=120,bbox_inches='tight')
print('\nsaved single_net_comparison.png'); print('SINGLE_DONE')
