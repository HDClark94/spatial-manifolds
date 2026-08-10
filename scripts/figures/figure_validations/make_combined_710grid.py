"""
Combined figure — J | coactivity | pR2(cell) | ΔpR2 (net7 phase-sorted).
Rows: matrices (top) | vs |J| recovery (middle) | vs rate-map corr / tuning (bottom).
Each scatter is annotated with a linear MIXED model (measure ~ predictor + random intercept per
TARGET cell) to account for pair non-independence: reports standardized slope β* and its p-value,
plus the fitted fixed-effect line. within=green, cross=orange. Cache: 710grid (net7+net10).
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
NETA=DATA+'/survey_s7_Ng1024'; NETB=DATA+'/survey_s10_Ng1024'
CACHE=DATA+'/pr2_two_net_710grid_cache.npz'; OUT=FIG+'/combined_matrices_710grid'
LAB1,LAB2='net7 (grid)','net10 (non-grid)'
SUBSET_SEED=0; N_BATCHES=12; RES=40; N_AVG=50
DEVICE='mps' if torch.backends.mps.is_available() else 'cpu'
WIN,CRS='#1f77b4','#ff7f0e'; WIND,CRSD='#10517d','#b35900'   # within=tab:blue, cross=tab:orange (+dark for lines)
NL1,NL2='#7b3294','#008080'                                  # net block labels (purple/teal, distinct from within/cross)

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
a1,_,_,_=compute_ratemaps(mA,tgA,o1,res=RES,n_avg=N_AVG,Ng=o1.Ng)
a2,_,_,_=compute_ratemaps(mB,tgB,o2,res=RES,n_avg=N_AVG,Ng=o2.Ng)
rmcorr=np.corrcoef(np.concatenate([a1[sel1],a2[sel2]],0).reshape(N,-1))

# phase-sort net7's grid cells by grid phase (net10 non-grid -> left in place)
_F=np.fft.fft2(a1[sel1]); _ps=np.fft.fftshift((np.abs(_F)**2).mean(0)); _c=RES//2
_ps[_c-2:_c+3,_c-2:_c+3]=0
_pk=np.unravel_index(np.argmax(_ps),_ps.shape); _kx,_ky=(_pk[0]-_c)%RES,(_pk[1]-_c)%RES
perm=np.concatenate([np.argsort(np.angle(_F[:,_kx,_ky])),np.arange(K,N)]).astype(int)

def joint_batch(tg,o):
    traj=tg.generate_trajectory(o.box_width,o.box_height,o.batch_size)
    v=np.stack([traj['ego_v']*np.cos(traj['target_hd']),traj['ego_v']*np.sin(traj['target_hd'])],axis=-1)
    v=torch.tensor(v,dtype=torch.float32).transpose(0,1).to(DEVICE)
    pos=np.stack([traj['target_x'],traj['target_y']],axis=-1); pos=torch.tensor(pos,dtype=torch.float32).transpose(0,1).to(DEVICE)
    ip=np.stack([traj['init_x'],traj['init_y']],axis=-1); ip=torch.tensor(ip,dtype=torch.float32).to(DEVICE)
    return (v,pcA.get_activation(ip).squeeze()),(v,pcB.get_activation(ip).squeeze()),pos
torch.manual_seed(SUBSET_SEED+1); np.random.seed(SUBSET_SEED+1)
G1L,G2L=[],[]
with torch.no_grad():
    for _ in range(N_BATCHES):
        inpA,inpB,_=joint_batch(tgA,o1)
        G1L.append(mA.g(inpA).cpu().numpy().transpose(1,0,2).reshape(-1,o1.Ng)[:,sel1])
        G2L.append(mB.g(inpB).cpu().numpy().transpose(1,0,2).reshape(-1,o2.Ng)[:,sel2])
G=np.concatenate([np.concatenate(G1L,0),np.concatenate(G2L,0)],axis=1).astype(np.float32)
coact=np.corrcoef(G.T)
low=pr2_base<0.3

# matrices [target,source] (phase-sorted net7 block)
Jm=J.copy(); CO=coact.copy(); CELL=pr2_cell.T.copy(); DE=(pr2_full-pr2_base[None,:]).T.copy()
for M in (CELL,DE): np.fill_diagonal(M,np.nan)
np.fill_diagonal(CO,np.nan); DE[low,:]=np.nan
Jm=Jm[perm][:,perm];CO=CO[perm][:,perm];CELL=CELL[perm][:,perm];DE=DE[perm][:,perm]
def blocks(ax,title):
    ax.axhline(K-0.5,color='k',lw=1.1); ax.axvline(K-0.5,color='k',lw=1.1); ax.set_xticks([]);ax.set_yticks([])
    for f,l,cc in [(0.25,LAB1,NL1),(0.75,LAB2,NL2)]:
        ax.text(N*f,N+3,l,ha='center',va='top',fontsize=6,fontweight='bold',color=cc)
        ax.text(-3,N*f,l,ha='center',va='center',rotation=90,fontsize=6,fontweight='bold',color=cc)
    ax.set_title(title,fontsize=8.5,fontweight='bold',pad=5)

# long-form (rc=rate-map corr, co/ce/de=measures, aj=|J|, tg=target idx for random effect, wi=within)
rc=[];co=[];ce=[];de=[];wi=[];aj=[];tg=[]
for ti in range(N):
    if low[ti]: continue
    for ci in range(N):
        if ci==ti or np.isnan(pr2_full[ci,ti]): continue
        rc.append(rmcorr[ci,ti]); co.append(coact[ci,ti]); ce.append(pr2_cell[ci,ti])
        de.append(pr2_full[ci,ti]-pr2_base[ti]); wi.append(net_id[ci]==net_id[ti]); aj.append(abs(float(J[ti,ci]))); tg.append(ti)
rc,co,ce,de,aj=map(np.array,(rc,co,ce,de,aj)); wi=np.array(wi,dtype=bool); tg=np.array(tg)

def lmm(x,y,g):
    """measure ~ x + (1|target); returns standardized slope β*, p, raw (intercept, slope), and the
    2x2 fixed-effect covariance (for the CI band)."""
    m=np.isfinite(x)&np.isfinite(y); x,y,g=x[m],y[m],g[m]
    if len(np.unique(g))<3 or x.std()==0: return np.nan,np.nan,np.nan,np.nan,None
    df=pd.DataFrame({'x':x,'y':y,'g':g})
    try:
        r=smf.mixedlm('y ~ x',df,groups=df['g']).fit(method='lbfgs',maxiter=300)
        b0,b1,p=r.params['Intercept'],r.params['x'],r.pvalues['x']
        cov=r.cov_params().loc[['Intercept','x'],['Intercept','x']].values
        return b1*x.std()/y.std(), p, b0, b1, cov
    except Exception: return np.nan,np.nan,np.nan,np.nan,None
def fitline(ax,xs,b0,b1,cov,linecol,bandcol):
    """draw the fixed-effect line + 95% CI band from the parameter covariance."""
    if not np.isfinite(b1): return
    yh=b0+b1*xs
    if cov is not None:
        se=np.sqrt(np.clip(cov[0,0]+xs**2*cov[1,1]+2*xs*cov[0,1],0,None))
        ax.fill_between(xs,yh-1.96*se,yh+1.96*se,color=bandcol,alpha=0.30,lw=0,zorder=2)
    ax.plot(xs,yh,color=linecol,lw=2,zorder=3)
def lmm2(x,y,g):
    """QUADRATIC: measure ~ x + x^2 + (1|target). Returns std linear β1*, std quad β2*, quad p,
    raw (b0,b1,b2) and 3x3 cov (for the U-shaped curve + band)."""
    m=np.isfinite(x)&np.isfinite(y); x,y,g=x[m],y[m],g[m]
    if len(np.unique(g))<3 or x.std()==0: return np.nan,np.nan,np.nan,np.nan,np.nan,np.nan,None
    df=pd.DataFrame({'x':x,'x2':x**2,'y':y,'g':g})
    try:
        r=smf.mixedlm('y ~ x + x2',df,groups=df['g']).fit(method='lbfgs',maxiter=400)
        b0,b1,b2=r.params['Intercept'],r.params['x'],r.params['x2']; p2=r.pvalues['x2']
        cov=r.cov_params().loc[['Intercept','x','x2'],['Intercept','x','x2']].values
        return b1*x.std()/y.std(), b2*(x**2).std()/y.std(), p2, b0, b1, b2, cov
    except Exception: return np.nan,np.nan,np.nan,np.nan,np.nan,np.nan,None
def fitcurve(ax,xs,b0,b1,b2,cov,linecol,bandcol):
    if not np.isfinite(b1): return
    yh=b0+b1*xs+b2*xs**2
    if cov is not None:
        X=np.column_stack([np.ones_like(xs),xs,xs**2]); se=np.sqrt(np.clip(np.einsum('ij,jk,ik->i',X,cov,X),0,None))
        ax.fill_between(xs,yh-1.96*se,yh+1.96*se,color=bandcol,alpha=0.30,lw=0,zorder=2)
    ax.plot(xs,yh,color=linecol,lw=2,zorder=3)

MEAS=[('coactivity (r)',CO,co,True),('pR²(cell)',CELL,ce,False),('ΔpR²',DE,de,False)]  # 4th = signed?
fig=plt.figure(figsize=(15.5,12.2))
gs=fig.add_gridspec(3,4,height_ratios=[1.15,1.0,1.0],hspace=0.36,wspace=0.26)
# row 0: J + 3 measure matrices
axJ=fig.add_subplot(gs[0,0]); vj=np.percentile(np.abs(Jm[Jm!=0]),99)
fig.colorbar(axJ.imshow(Jm,cmap='RdBu_r',vmin=-vj,vmax=vj,interpolation='nearest'),ax=axJ,fraction=0.046,pad=0.02); blocks(axJ,'J (ground-truth coupling)')
for col,(name,M,_,signed) in enumerate(MEAS,start=1):
    cmap='RdBu_r' if signed else 'inferno'
    ax=fig.add_subplot(gs[0,col]); cm=matplotlib.colormaps[cmap].copy(); cm.set_bad('lightgrey')
    if signed: vm=np.nanpercentile(np.abs(M),99); lo,hi=-vm,vm
    else: lo,hi=0,np.nanpercentile(M,98)
    fig.colorbar(ax.imshow(M,cmap=cm,vmin=lo,vmax=hi,interpolation='nearest'),ax=ax,fraction=0.046,pad=0.02); blocks(ax,name)
# row 1: vs |J|  (LMM on within-net pairs; cross all sit at |J|=0)
axl=fig.add_subplot(gs[1,0]); axl.axis('off'); axl.text(0.0,0.8,'vs |J| — recovery of\nground-truth coupling\n\nLMM: measure ~ |J|\n+ (1|target)',fontsize=9.5,va='top',fontweight='bold')
for col,(name,_,y,_) in enumerate(MEAS,start=1):
    ax=fig.add_subplot(gs[1,col]); yy=np.abs(y) if name!='ΔpR²' else y
    ax.scatter(aj[~wi],yy[~wi],s=4,alpha=0.25,color=CRS); ax.scatter(aj[wi],yy[wi],s=4,alpha=0.3,color=WIN)
    bstar,p,b0,b1,cov=lmm(aj[wi],yy[wi],tg[wi])
    fitline(ax,np.linspace(0,np.nanpercentile(aj[wi],99),50),b0,b1,cov,WIND,WIN)
    ax.set_xlabel('|J|'); ax.set_ylabel('|'+name+'|' if name!='ΔpR²' else name)
    ax.set_title(f'β*={bstar:+.2f}  p={p:.1g}',fontsize=8.5,fontweight='bold')
    ax.axhline(0,color='grey',lw=0.5,ls=':'); ax.spines[['top','right']].set_visible(False)
# row 2: vs rate-map corr (LMM within & cross separately)
axl2=fig.add_subplot(gs[2,0]); axl2.axis('off'); axl2.text(0.0,0.8,'vs rate-map corr —\ntuning / phase similarity\n\nLMM: measure ~ tuning\n+ (1|target)',fontsize=9.5,va='top',fontweight='bold')
for col,(name,_,y,_) in enumerate(MEAS,start=1):
    ax=fig.add_subplot(gs[2,col])
    ax.scatter(rc[wi],y[wi],s=4,alpha=0.3,color=WIN,label='within'); ax.scatter(rc[~wi],y[~wi],s=4,alpha=0.3,color=CRS,label='cross')
    b1w,b2w,p2w,b0w,r1w,r2w,covw=lmm2(rc[wi],y[wi],tg[wi]); b1x,b2x,p2x,b0x,r1x,r2x,covx=lmm2(rc[~wi],y[~wi],tg[~wi])
    xs=np.linspace(rc.min(),rc.max(),60)
    fitcurve(ax,xs,b0w,r1w,r2w,covw,WIND,WIN); fitcurve(ax,xs,b0x,r1x,r2x,covx,CRSD,CRS)
    ax.set_xlabel('rate-map correlation'); ax.set_ylabel(name)
    ax.set_title(f'within: lin β*={b1w:+.2f} · quad β*={b2w:+.2f} (p={p2w:.0e})',fontsize=7.5,fontweight='bold')
    ax.axhline(0,color='grey',lw=0.5,ls=':'); ax.spines[['top','right']].set_visible(False)
    if col==1: ax.legend(fontsize=7,frameon=False,markerscale=2)
fig.suptitle(f'Coupling measures — structure (top) · |J| recovery (middle) · tuning (bottom) — {LAB1}+{LAB2}  [LMM, random intercept per target]',fontsize=11.5,fontweight='bold')
plt.savefig(OUT+'.png',dpi=115,bbox_inches='tight'); plt.savefig(OUT+'.pdf',bbox_inches='tight')
print('saved '+OUT+'.png/pdf', flush=True); print('COMBINED_DONE', flush=True)
