"""Generate the two-network control figures from the cache: 4-matrix panel + analysis."""
import numpy as np, matplotlib
matplotlib.use('Agg'); import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from scipy.stats import spearmanr
FIG='/Users/harryclark/Documents/spatial-manifolds/scripts/figures/figure_validations'
c=np.load('/Users/harryclark/Documents/spatial-manifolds/data/rnn_xgboost/pr2_two_network_cache.npz',allow_pickle=True)
pr2_base=c['pr2_base']; pr2_cell=c['pr2_cell']; pr2_full=c['pr2_full']
net_id=c['net_id']; J=c['J_comb']; K=int(c['K']); N=len(net_id)
W,X='#c04744','#3171ae'

# ── 4-matrix panel ────────────────────────────────────────────────────────────
Jmat=J.copy(); Cell=pr2_cell.T.copy(); Full=pr2_full.T.copy(); Delta=(pr2_full-pr2_base[None,:]).T.copy()
low=np.where(pr2_base<0.3)[0]
for M in (Cell,Full,Delta): np.fill_diagonal(M,np.nan)
Delta[low,:]=np.nan
def blocks(ax,title):
    ax.axhline(K-0.5,color='k',lw=1.2); ax.axvline(K-0.5,color='k',lw=1.2); ax.set_xticks([]);ax.set_yticks([])
    for f,l,cc in [(0.25,'net1',W),(0.75,'net2',X)]:
        ax.text(N*f,N+3,l,ha='center',va='top',fontsize=7,fontweight='bold',color=cc)
        ax.text(-3,N*f,l,ha='center',va='center',rotation=90,fontsize=7,fontweight='bold',color=cc)
    ax.set_title(title,fontsize=9,fontweight='bold',pad=6)
fig,axes=plt.subplots(1,4,figsize=(16,4.3))
vmaxJ=np.percentile(np.abs(Jmat[Jmat!=0]),99)
im=axes[0].imshow(Jmat,cmap='RdBu_r',vmin=-vmaxJ,vmax=vmaxJ,interpolation='nearest'); blocks(axes[0],'J  (ground-truth coupling)')
fig.colorbar(im,ax=axes[0],fraction=0.046,pad=0.02)
for ax,M,title,cmap,lo,hi in [
    (axes[1],Cell,'pR²(cell)','viridis',0,np.nanpercentile(Cell,99)),
    (axes[2],Full,'pR²(base + cell)','viridis',np.nanpercentile(Full,2),np.nanpercentile(Full,98)),
    (axes[3],Delta,'ΔpR²  (base+cell − base)','inferno',0,np.nanpercentile(Delta,95))]:
    cm=matplotlib.colormaps[cmap].copy(); cm.set_bad('lightgrey')
    im=ax.imshow(M,cmap=cm,vmin=lo,vmax=hi,interpolation='nearest'); blocks(ax,title)
    fig.colorbar(im,ax=ax,fraction=0.046,pad=0.02)
fig.text(0.5,0.02,'rows = target (post-synaptic)   |   cols = source (pre-synaptic)   |   base = position + speed + head direction   |   grey = self or excluded (pr2_base<0.3)',ha='center',fontsize=8,color='#555')
fig.suptitle('Two-network control (two from-scratch 1024-unit nets): J vs XGBoost predictability matrices',fontsize=11,fontweight='bold')
plt.tight_layout(rect=[0,0.03,1,0.96])
plt.savefig(FIG+'/two_network_matrices.pdf',bbox_inches='tight'); plt.savefig(FIG+'/two_network_matrices.png',dpi=120,bbox_inches='tight')
plt.close()

# ── analysis (base-filtered) ──────────────────────────────────────────────────
rows=[]
for ti in range(N):
    if pr2_base[ti]<0.3: continue
    for ci in range(N):
        if ci==ti or np.isnan(pr2_full[ci,ti]): continue
        rows.append((abs(float(J[ti,ci])),float(pr2_cell[ci,ti]),float(pr2_full[ci,ti]-pr2_base[ti]),bool(net_id[ci]==net_id[ti])))
absJ,pcell,dpr2,within=(np.array([r[k] for r in rows]) for k in range(4)); within=within.astype(bool)
fig,ax=plt.subplots(1,3,figsize=(14,4.3))
wj,wd=absJ[within],dpr2[within]; q=np.quantile(wj,np.linspace(0,1,6))
cy=[wd[(wj>=q[i])&(wj<=q[i+1])].mean() for i in range(5)]
ce=[wd[(wj>=q[i])&(wj<=q[i+1])].std()/np.sqrt(max(((wj>=q[i])&(wj<=q[i+1])).sum(),1)) for i in range(5)]
ax[0].errorbar(range(5),cy,yerr=ce,marker='o',color=W,lw=2,capsize=3,label='within-net (J≠0)')
floor=dpr2[~within].mean(); ax[0].axhline(floor,color=X,lw=2,ls='--',label='cross-net floor (J=0)')
ax[0].set_xticks(range(5));ax[0].set_xticklabels(['Q1','Q2','Q3','Q4','Q5']);ax[0].set_xlabel('|J| quintile (within-net)');ax[0].set_ylabel('ΔpR²')
rho,pv=spearmanr(wj,wd); ax[0].set_title(f'ΔpR² rises with |J|\nSpearman={rho:+.2f}, p={pv:.0e}',fontsize=9,fontweight='bold')
ax[0].legend(fontsize=8,frameon=False); ax[0].spines[['top','right']].set_visible(False)
data=[dpr2[within],dpr2[~within],pcell[within],pcell[~within]]
parts=ax[1].violinplot(data,positions=[0,0.55,1.7,2.25],widths=0.45,showmeans=True)
for b,cc in zip(parts['bodies'],[W,X,W,X]): b.set_facecolor(cc); b.set_alpha(0.5)
ax[1].axhline(0,color='k',lw=0.8,ls=':');ax[1].set_xticks([0.27,1.97]);ax[1].set_xticklabels(['ΔpR²','pR²(cell)']);ax[1].set_ylabel('value')
ax[1].set_title(f'cross ΔpR²={dpr2[~within].mean():.3f} (≈0)\nvs cross pR²(cell)={pcell[~within].mean():.3f}',fontsize=9,fontweight='bold')
ax[1].legend([mpatches.Patch(color=W,alpha=.5),mpatches.Patch(color=X,alpha=.5)],['within','cross'],fontsize=8,frameon=False)
ax[1].spines[['top','right']].set_visible(False)
r_d=spearmanr(absJ[within],dpr2[within])[0]; r_c=spearmanr(absJ[within],pcell[within])[0]
ax[2].bar([0,1],[r_d,r_c],color=[W,'#888'],width=0.6);ax[2].set_xticks([0,1]);ax[2].set_xticklabels(['ΔpR²','pR²(cell)'])
ax[2].set_ylabel('within-net Spearman(|J|, ·)')
for xx,vv in [(0,r_d),(1,r_c)]: ax[2].text(xx,vv+0.008,f'{vv:.2f}',ha='center',fontsize=9)
hi=within&(absJ>=np.quantile(absJ[within],0.9))
ax[2].set_title(f'strong (top-10% |J|) within ΔpR²\n= {dpr2[hi].mean()/max(floor,1e-9):.1f}× cross floor',fontsize=9,fontweight='bold')
ax[2].spines[['top','right']].set_visible(False); ax[2].set_ylim(0,max(r_d,r_c)*1.25)
plt.tight_layout(); plt.savefig(FIG+'/two_network_control.pdf',bbox_inches='tight'); plt.savefig(FIG+'/two_network_control.png',dpi=120,bbox_inches='tight')
plt.close()

print(f'FIGURES_DONE | within ΔpR²={dpr2[within].mean():.4f} cross={dpr2[~within].mean():.4f} '
      f'ratio={dpr2[within].mean()/max(dpr2[~within].mean(),1e-9):.2f}x | '
      f'Spearman(|J|,ΔpR²)={r_d:+.3f} Spearman(|J|,pR²cell)={r_c:+.3f}', flush=True)
