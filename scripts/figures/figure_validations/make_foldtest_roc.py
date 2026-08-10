"""Can the paired-fold statistic classify within (coupled) vs cross (uncoupled) pairs — i.e. recover
network membership? Reads foldtest_710grid.npz; reports AUC (within=positive) + TPR/FPR at
significance thresholds for ΔpR2 / paired t-stat / fold-consistency, plus within-vs-cross histograms."""
import numpy as np
import matplotlib; matplotlib.use('Agg'); import matplotlib.pyplot as plt
from scipy.stats import mannwhitneyu, t as tdist
DATA='/Users/harryclark/Documents/spatial-manifolds/data/rnn_xgboost'
FIG='/Users/harryclark/Documents/spatial-manifolds/scripts/figures/figure_validations'
d=np.load(DATA+'/foldtest_710grid.npz')
aj=d['aj']; dpr2=d['dpr2']; tstat=d['tstat']; cons=d['cons']; wi=d['wi'].astype(bool)
n1=int(wi.sum()); n0=int((~wi).sum())
print(f'within pairs (coupled): {n1} | cross pairs (uncoupled): {n0}\n')
def auc(v):
    m=np.isfinite(v); U=mannwhitneyu(v[wi&m],v[(~wi)&m],alternative='greater').statistic
    return U/((wi&m).sum()*((~wi)&m).sum())
print(f'{"measure":16s} | AUC (within>cross) | within med | cross med')
for name,v in [('ΔpR²',dpr2),('paired t-stat',tstat),('fold-consistency',cons)]:
    print(f'{name:16s} | {auc(v):.3f}              | {np.nanmedian(v[wi]):+.3f}     | {np.nanmedian(v[~wi]):+.3f}')

# classification at significance thresholds (t-stat), one-sided df=9
tcrit05=tdist.ppf(0.95,9)                         # p<0.05 uncorrected
tcrit_bonf=tdist.ppf(1-0.05/len(tstat),9)         # Bonferroni over all pairs
print('\n--- t-stat as a significance classifier (within should fire +, cross should stay -) ---')
for lab,thr in [('p<0.05 uncorrected',tcrit05),('Bonferroni',tcrit_bonf)]:
    tpr=(tstat[wi]>thr).mean(); fpr=(tstat[~wi]>thr).mean()
    print(f'{lab:20s} (t>{thr:.2f}): TPR within-flagged={tpr:.2f}  FPR cross-flagged={fpr:.2f}')

fig,ax=plt.subplots(1,3,figsize=(15,4.2))
for a,(name,v) in zip(ax,[('ΔpR²',dpr2),('paired t-stat',tstat),('fold-consistency',cons)]):
    lo,hi=np.nanpercentile(v,1),np.nanpercentile(v,99); b=np.linspace(lo,hi,40)
    a.hist(v[wi],bins=b,alpha=0.55,color='#c04744',density=True,label=f'within (coupled) n={n1}')
    a.hist(v[~wi],bins=b,alpha=0.55,color='#3171ae',density=True,label=f'cross (uncoupled) n={n0}')
    a.set_title(f'{name}  AUC={auc(v):.2f}',fontweight='bold'); a.set_xlabel(name); a.legend(fontsize=7,frameon=False); a.spines[['top','right']].set_visible(False)
    if name=='paired t-stat': a.axvline(tcrit05,color='k',ls='--',lw=1,label='p<0.05')
plt.suptitle('Within (coupled) vs cross (uncoupled) separation — can the statistic recover network membership?',fontweight='bold')
plt.tight_layout(rect=[0,0,1,0.95]); plt.savefig(FIG+'/foldtest_roc.png',dpi=120,bbox_inches='tight')
print('\nsaved foldtest_roc.png'); print('ROC_DONE')
