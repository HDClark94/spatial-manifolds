"""
Combined figure for net7(grid/position) + HD(head-direction).
Matrices: J | coactivity | pR2(cell) | ΔpR2 (net-ordered).
Row 1: vs |J| (recovery, linear LMM + 95% CI, per net; cross in grey).
Row 2: vs tuning — WITHIN-NET only (net7=spatial rate-map corr, HD=heading tuning corr), QUADRATIC LMM
       (measure ~ tuning + tuning^2 + (1|target)) since the tuning axis goes negative / U-shaped.
Colors: net7=tab:blue, HD=tab:orange, cross=grey.
"""
import sys, os
REPO_DIR='/Users/harryclark/Documents/spatial-manifolds/GRID-PATTERN-FORMATION'
sys.path.insert(0, REPO_DIR); sys.path.insert(0,'/Users/harryclark/Documents/spatial-manifolds/src')
import numpy as np, torch, pandas as pd
import matplotlib; matplotlib.use('Agg'); import matplotlib.pyplot as plt
import statsmodels.formula.api as smf
from place_cells import PlaceCells
from trajectory_generator import TrajectoryGenerator
from model import RNN
from visualize import compute_ratemaps

DATA='/Users/harryclark/Documents/spatial-manifolds/data/rnn_xgboost'
FIG='/Users/harryclark/Documents/spatial-manifolds/scripts/figures/figure_validations'
NETA=DATA+'/survey_s7_Ng1024'; HD_DIR=DATA+'/hd_net_Ng1024'; CACHE=DATA+'/pr2_two_net_hd7_cache.npz'
OUT=FIG+'/combined_matrices_hd7'; LAB1,LAB2='net7 (grid)','HD net'
SUBSET_SEED=0; N_BATCHES=12; RES,N_AVG=40,50; KAPPA=4.0; NBIN=24
DEVICE='mps' if torch.backends.mps.is_available() else 'cpu'
N7,N7D,HDc,HDd,CRSc='#1f77b4','#10517d','#ff7f0e','#b35900','#9e9e9e'   # net7 blue, HD orange, cross grey

class HDCells:
    def __init__(self, us, kappa, device):
        self.us=torch.tensor(us,dtype=torch.float32).to(device); self.kappa=kappa
    def get_activation(self, hd): return torch.softmax(self.kappa*torch.cos(hd - self.us), dim=-1)

c=np.load(CACHE,allow_pickle=True)
pr2_base=c['pr2_base']; pr2_cell=c['pr2_cell']; pr2_full=c['pr2_full']
net_id=c['net_id']; J=c['J_comb']; K=int(c['K']); sel1=c['sel1']; sel2=c['sel2']; N=len(net_id)

def build_grid(d):
    class O: pass
    o=O(); o.Np=512;o.Ng=1024;o.sequence_length=20;o.batch_size=200
    o.learning_rate=1e-4;o.weight_decay=1e-4;o.place_cell_rf=0.12;o.surround_scale=2
    o.RNN_type='RNN';o.activation='relu';o.DoG=True;o.periodic=False;o.box_width=2.2;o.box_height=2.2;o.device=DEVICE
    pc=PlaceCells(o); pc.us=torch.tensor(np.load(d+'/place_cells.npy'),dtype=torch.float32).to(DEVICE)
    m=RNN(o,pc).to(DEVICE); m.load_state_dict(torch.load(d+'/ckpt.pth',map_location=DEVICE)['model']); m.eval(); return o,pc,m
def build_hd(d):
    us=np.load(d+'/hd_cells.npy'); hd=HDCells(us,KAPPA,DEVICE)
    class O: pass
    o=O(); o.Np=len(us);o.Ng=1024;o.sequence_length=20;o.batch_size=200
    o.learning_rate=1e-4;o.weight_decay=1e-4;o.place_cell_rf=0.12;o.surround_scale=2
    o.RNN_type='RNN';o.activation='relu';o.DoG=True;o.periodic=False;o.box_width=2.2;o.box_height=2.2;o.device=DEVICE
    m=RNN(o,hd).to(DEVICE); m.load_state_dict(torch.load(d+'/ckpt.pth',map_location=DEVICE)['model']); m.eval(); return o,hd,m
o1,pcA,mA=build_grid(NETA); o2,hdB,mB=build_hd(HD_DIR); tgA=TrajectoryGenerator(o1,pcA)
a1,_,_,_=compute_ratemaps(mA,tgA,o1,res=RES,n_avg=N_AVG,Ng=o1.Ng); rm7=np.corrcoef(a1[sel1].reshape(K,-1))

def joint_batch():
    traj=tgA.generate_trajectory(o1.box_width,o1.box_height,o1.batch_size)
    ego=traj['ego_v']; hdir=traj['target_hd']
    v=np.stack([ego*np.cos(hdir),ego*np.sin(hdir)],-1); v=torch.tensor(v,dtype=torch.float32).transpose(0,1).to(DEVICE)
    phi=np.stack([traj['phi_x'],traj['phi_y']],-1); phi=torch.tensor(phi,dtype=torch.float32).transpose(0,1).to(DEVICE)
    ip=np.stack([traj['init_x'],traj['init_y']],-1); ip=torch.tensor(ip,dtype=torch.float32).to(DEVICE)
    ihd=torch.tensor(traj['init_hd'],dtype=torch.float32).to(DEVICE)[:,None,:]
    return (v,pcA.get_activation(ip).squeeze()),(phi,hdB.get_activation(ihd).squeeze(1))
torch.manual_seed(SUBSET_SEED+1); np.random.seed(SUBSET_SEED+1)
G1L,G2L,HDL=[],[],[]
with torch.no_grad():
    for _ in range(N_BATCHES):
        (v,p0),(phi,h0)=joint_batch()
        G1L.append(mA.g((v,p0)).cpu().numpy().transpose(1,0,2).reshape(-1,o1.Ng)[:,sel1])
        G2L.append(mB.g((phi,h0)).cpu().numpy().transpose(1,0,2).reshape(-1,o2.Ng)[:,sel2])
        HDL.append(np.arctan2(v.cpu().numpy().transpose(1,0,2).reshape(-1,2)[:,1],v.cpu().numpy().transpose(1,0,2).reshape(-1,2)[:,0]))
G=np.concatenate([np.concatenate(G1L,0),np.concatenate(G2L,0)],axis=1).astype(np.float32)
hd=np.concatenate(HDL,0); coact=np.corrcoef(G.T)
# HD heading tuning curves -> within-HD tuning similarity
edges=np.linspace(-np.pi,np.pi,NBIN+1); wbin=np.clip(np.digitize(hd,edges)-1,0,NBIN-1)
G2=G[:,K:]; tc=np.array([[G2[wbin==b,i].mean() if (wbin==b).any() else 0 for b in range(NBIN)] for i in range(K)])
rmHD=np.corrcoef(tc)
low=pr2_base<0.3

Jm=J.copy(); CO=coact.copy(); CELL=pr2_cell.T.copy(); DE=(pr2_full-pr2_base[None,:]).T.copy()
for M in (CELL,DE): np.fill_diagonal(M,np.nan)
np.fill_diagonal(CO,np.nan); DE[low,:]=np.nan
TUN=np.full((N,N),np.nan); TUN[:K,:K]=rm7; TUN[K:,K:]=rmHD
def blocks(ax,title):
    ax.axhline(K-0.5,color='k',lw=1.1); ax.axvline(K-0.5,color='k',lw=1.1); ax.set_xticks([]);ax.set_yticks([])
    for f,l,cc in [(0.25,LAB1,N7),(0.75,LAB2,HDc)]:
        ax.text(N*f,N+3,l,ha='center',va='top',fontsize=6,fontweight='bold',color=cc)
        ax.text(-3,N*f,l,ha='center',va='center',rotation=90,fontsize=6,fontweight='bold',color=cc)
    ax.set_title(title,fontsize=8.5,fontweight='bold',pad=5)

co=[];ce=[];de=[];aj=[];tu=[];grp=[];tg=[]   # grp: 0=net7 within,1=HD within,2=cross
for ti in range(N):
    if low[ti]: continue
    for ci in range(N):
        if ci==ti or np.isnan(pr2_full[ci,ti]): continue
        co.append(coact[ci,ti]); ce.append(pr2_cell[ci,ti]); de.append(pr2_full[ci,ti]-pr2_base[ti])
        aj.append(abs(float(J[ti,ci]))); tu.append(TUN[ci,ti]); tg.append(ti)
        grp.append(0 if (net_id[ci]==0 and net_id[ti]==0) else (1 if (net_id[ci]==1 and net_id[ti]==1) else 2))
co,ce,de,aj,tu=map(np.array,(co,ce,de,aj,tu)); grp=np.array(grp); tg=np.array(tg)

def lmm(x,y,g):
    m=np.isfinite(x)&np.isfinite(y); x,y,g=x[m],y[m],g[m]
    if len(np.unique(g))<3 or x.std()==0: return np.nan,np.nan,np.nan,np.nan,None
    df=pd.DataFrame({'x':x,'y':y,'g':g})
    try:
        r=smf.mixedlm('y ~ x',df,groups=df['g']).fit(method='lbfgs',maxiter=300)
        cov=r.cov_params().loc[['Intercept','x'],['Intercept','x']].values
        return r.params['x']*x.std()/y.std(), r.pvalues['x'], r.params['Intercept'], r.params['x'], cov
    except Exception: return np.nan,np.nan,np.nan,np.nan,None
def fitline(ax,xs,b0,b1,cov,lc,bc):
    if not np.isfinite(b1): return
    yh=b0+b1*xs
    if cov is not None:
        se=np.sqrt(np.clip(cov[0,0]+xs**2*cov[1,1]+2*xs*cov[0,1],0,None)); ax.fill_between(xs,yh-1.96*se,yh+1.96*se,color=bc,alpha=0.30,lw=0,zorder=2)
    ax.plot(xs,yh,color=lc,lw=2,zorder=3)
def lmm2(x,y,g):
    m=np.isfinite(x)&np.isfinite(y); x,y,g=x[m],y[m],g[m]
    if len(np.unique(g))<3 or x.std()==0: return np.nan,np.nan,np.nan,np.nan,np.nan,np.nan,None
    df=pd.DataFrame({'x':x,'x2':x**2,'y':y,'g':g})
    try:
        r=smf.mixedlm('y ~ x + x2',df,groups=df['g']).fit(method='lbfgs',maxiter=400)
        cov=r.cov_params().loc[['Intercept','x','x2'],['Intercept','x','x2']].values
        return r.params['x']*x.std()/y.std(), r.params['x2']*(x**2).std()/y.std(), r.pvalues['x2'], r.params['Intercept'], r.params['x'], r.params['x2'], cov
    except Exception: return np.nan,np.nan,np.nan,np.nan,np.nan,np.nan,None
def fitcurve(ax,xs,b0,b1,b2,cov,lc,bc):
    if not np.isfinite(b1): return
    yh=b0+b1*xs+b2*xs**2
    if cov is not None:
        X=np.column_stack([np.ones_like(xs),xs,xs**2]); se=np.sqrt(np.clip(np.einsum('ij,jk,ik->i',X,cov,X),0,None))
        ax.fill_between(xs,yh-1.96*se,yh+1.96*se,color=bc,alpha=0.30,lw=0,zorder=2)
    ax.plot(xs,yh,color=lc,lw=2,zorder=3)

MEAS=[('coactivity (r)',CO,co,True),('pR²(cell)',CELL,ce,False),('ΔpR²',DE,de,False)]  # 4th=signed?
fig=plt.figure(figsize=(15.5,12.2)); gs=fig.add_gridspec(3,4,height_ratios=[1.15,1.0,1.0],hspace=0.36,wspace=0.26)
axJ=fig.add_subplot(gs[0,0]); vj=np.percentile(np.abs(Jm[Jm!=0]),99)
fig.colorbar(axJ.imshow(Jm,cmap='RdBu_r',vmin=-vj,vmax=vj,interpolation='nearest'),ax=axJ,fraction=0.046,pad=0.02); blocks(axJ,'J (ground-truth coupling)')
for col,(name,M,_,signed) in enumerate(MEAS,start=1):
    cmap='RdBu_r' if signed else 'inferno'; ax=fig.add_subplot(gs[0,col]); cm=matplotlib.colormaps[cmap].copy(); cm.set_bad('lightgrey')
    if signed: vm=np.nanpercentile(np.abs(M),99); lo,hi=-vm,vm
    else: lo,hi=0,np.nanpercentile(M,98)
    fig.colorbar(ax.imshow(M,cmap=cm,vmin=lo,vmax=hi,interpolation='nearest'),ax=ax,fraction=0.046,pad=0.02); blocks(ax,name)
# row 1: vs |J| — linear LMM per net (cross grey)
axl=fig.add_subplot(gs[1,0]); axl.axis('off'); axl.text(0.0,0.8,'vs |J| — recovery\n\nLMM: measure ~ |J|\n+ (1|target)\n\nnet7=blue  HD=orange\ncross=grey',fontsize=9,va='top',fontweight='bold')
for col,(name,_,y,_) in enumerate(MEAS,start=1):
    ax=fig.add_subplot(gs[1,col]); yy=np.abs(y) if name!='ΔpR²' else y
    ax.scatter(aj[grp==2],yy[grp==2],s=4,alpha=0.25,color=CRSc); ax.scatter(aj[grp==0],yy[grp==0],s=4,alpha=0.35,color=N7); ax.scatter(aj[grp==1],yy[grp==1],s=4,alpha=0.35,color=HDc)
    b7,p7,i7,s7,c7=lmm(aj[grp==0],yy[grp==0],tg[grp==0]); bh,ph,ih,sh,ch=lmm(aj[grp==1],yy[grp==1],tg[grp==1])
    fitline(ax,np.linspace(0,np.nanpercentile(aj[grp==0],99),50),i7,s7,c7,N7D,N7); fitline(ax,np.linspace(0,np.nanpercentile(aj[grp==1],99),50),ih,sh,ch,HDd,HDc)
    ax.set_xlabel('|J|'); ax.set_ylabel('|'+name+'|' if name!='ΔpR²' else name)
    ax.set_title(f'net7 β*={b7:+.2f} · HD β*={bh:+.2f}',fontsize=8,fontweight='bold'); ax.axhline(0,color='grey',lw=0.5,ls=':'); ax.spines[['top','right']].set_visible(False)
# row 2: vs tuning — QUADRATIC LMM per net (within-net only)
axl2=fig.add_subplot(gs[2,0]); axl2.axis('off'); axl2.text(0.0,0.8,'vs tuning (within-net)\nnet7=spatial, HD=heading\n\nQUADRATIC LMM:\nmeasure ~ t + t² + (1|target)',fontsize=9,va='top',fontweight='bold')
for col,(name,_,y,_) in enumerate(MEAS,start=1):
    ax=fig.add_subplot(gs[2,col])
    ax.scatter(tu[grp==0],y[grp==0],s=5,alpha=0.35,color=N7,label='net7 (spatial)'); ax.scatter(tu[grp==1],y[grp==1],s=5,alpha=0.35,color=HDc,label='HD (heading)')
    l7,q7,pq7,i7,s7,s27,c7=lmm2(tu[grp==0],y[grp==0],tg[grp==0]); lh,qh,pqh,ih,sh,s2h,ch=lmm2(tu[grp==1],y[grp==1],tg[grp==1])
    x7=np.linspace(np.nanmin(tu[grp==0]),np.nanmax(tu[grp==0]),60); xh=np.linspace(np.nanmin(tu[grp==1]),np.nanmax(tu[grp==1]),60)
    fitcurve(ax,x7,i7,s7,s27,c7,N7D,N7); fitcurve(ax,xh,ih,sh,s2h,ch,HDd,HDc)
    ax.set_xlabel('within-net tuning corr'); ax.set_ylabel(name)
    ax.set_title(f'net7 quad β*={q7:+.2f} (p={pq7:.0e})',fontsize=7.5,fontweight='bold'); ax.axhline(0,color='grey',lw=0.5,ls=':'); ax.spines[['top','right']].set_visible(False)
    if col==1: ax.legend(fontsize=7,frameon=False,markerscale=2)
fig.suptitle('net7 (grid) + HD (head-direction): structure (top) · |J| recovery (mid) · tuning (bottom, within-net, quadratic) — cross ≈ 0',fontsize=11.5,fontweight='bold')
plt.savefig(OUT+'.png',dpi=115,bbox_inches='tight'); plt.savefig(OUT+'.pdf',bbox_inches='tight')
print('saved '+OUT+'.png/pdf',flush=True); print('HD7FIG DONE',flush=True)
