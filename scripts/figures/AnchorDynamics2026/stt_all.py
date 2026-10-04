"""Per-session sliding-window coupling: does r track the state within sessions?"""
import numpy as np, pandas as pd
from scipy.stats import spearmanr, wilcoxon
rng = np.random.default_rng(0)
T = pd.read_csv('/Users/harryclark/Documents/spatial-manifolds/data/lfp/theta_frequency.csv')
T['anch'] = T.anch.astype(bool)
WIN, NSHIFT = 25, 500

def sliding(sp, am, an, win=WIN):
    c=[];r=[];f=[]
    for i in range(len(sp)-win+1):
        s,a,k = sp[i:i+win], am[i:i+win], an[i:i+win]
        m = np.isfinite(s)&np.isfinite(a)
        if m.sum()<win*.6 or np.std(s[m])==0: continue
        c.append(i+win//2); r.append(np.corrcoef(s[m],a[m])[0,1]); f.append(np.nanmean(k))
    return np.asarray(c),np.asarray(r),np.asarray(f)

rows=[]
for (mo,dy),d in T.groupby(['mouse','day']):
    d=d.sort_values('trial'); tr=d.trial.values
    if d.anch.sum()<15 or (~d.anch).sum()<15: continue
    full=np.arange(tr.min(),tr.max()+1); idx=tr-tr.min()
    sp=np.full(len(full),np.nan); am=sp.copy(); an=sp.copy()
    sp[idx],am[idx],an[idx]=d.speed.values,d.amp.values,d.anch.values.astype(float)
    c,r,f=sliding(sp,am,an)
    if len(r)<30 or np.std(f)==0: continue
    rho=spearmanr(r,f)[0]
    null=np.array([spearmanr(r,sliding(sp,am,np.roll(an,k))[2])[0]
                   for k in rng.integers(1,len(an),NSHIFT)])
    null=null[np.isfinite(null)]
    p=(np.sum(np.abs(null)>=abs(rho))+1)/(len(null)+1)
    # how many independent state blocks does this session actually contain?
    st=np.nan_to_num(an)>.5
    nblk=int(np.sum(np.diff(st.astype(int))!=0))+1
    rows.append(dict(mouse=mo,day=dy,rho=rho,p=p,nblk=nblk,nwin=len(r),
                     r_anch=np.mean(r[f>.5]) if (f>.5).any() else np.nan,
                     r_non=np.mean(r[f<.5]) if (f<.5).any() else np.nan))
R=pd.DataFrame(rows)
R.to_csv('/Users/harryclark/Documents/spatial-manifolds/data/lfp/speed_theta_timecourse.csv',index=False)
neg=(R.rho<0).sum()
from scipy.stats import binomtest
print(f'{len(R)} sessions, window {WIN} trials')
print(f'  per-session rho: median {R.rho.median():+.3f}, {neg}/{len(R)} negative, '
      f'sign test p = {binomtest(neg,len(R)).pvalue:.3g}')
print(f'  vs 0 (Wilcoxon): p = {wilcoxon(R.rho).pvalue:.3g}')
print(f'  individually significant against own shift null: {(R.p<.05).sum()}/{len(R)} '
      f'({(R[(R.p<.05)].rho<0).sum()} of them negative)')
P = R.dropna(subset=['r_anch', 'r_non'])      # keep the pairing
print(f'  window r when mostly anchored {P.r_anch.mean():+.3f} vs mostly non {P.r_non.mean():+.3f}, '
      f'p = {wilcoxon(P.r_anch, P.r_non).pvalue:.3g}  (n = {len(P)})')
print(f'  state blocks per session: median {R.nblk.median():.0f}, range {R.nblk.min()}-{R.nblk.max()}')
print(f'  rho vs n blocks: rho = {spearmanr(R.rho,R.nblk)[0]:+.3f}, p = {spearmanr(R.rho,R.nblk)[1]:.3g}')
