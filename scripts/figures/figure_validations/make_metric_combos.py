"""
Can pR2(cell) be combined with ΔpR2 into a better coupling metric?
Pure arithmetic on cached pr2_base[ti], pr2_cell[ci,ti], pr2_full[ci,ti] (no XGBoost).
Variance partitioning:  pR2(cell) = ΔpR2 (unique/coupling) + commonality (shared/tuning).
Benchmark each candidate on within-net Spearman(|J|) [recovery] and cross/within [unbiasedness].
"""
import numpy as np
from scipy.stats import spearmanr
DATA='/Users/harryclark/Documents/spatial-manifolds/data/rnn_xgboost'
CACHES=[('diffPC','pr2_two_net_diffPC_cache.npz'),('6+7','pr2_two_net_67_cache.npz'),('1+7','pr2_two_net_17_cache.npz')]
EPS=0.02  # regularizer for ratio denominators (pR2 can be ~0 / slightly negative under CV)

def metrics(base,cell,full):
    dpr2  = full - base
    common= cell + base - full                      # tuning redundancy (b+c - b = c)
    uf_c  = dpr2 / np.clip(cell, EPS, None)         # uniqueness fraction (÷ cell total)
    uf_f  = dpr2 / np.clip(full, EPS, None)         # ÷ full total
    gm    = np.sign(dpr2)*np.sqrt(np.abs(dpr2)*np.clip(cell,0,None))  # geo-mean of coupling & cell
    prod  = dpr2 * np.clip(cell,0,None)             # product gate
    return {'ΔpR² (unique)':dpr2,'pR²(cell) [=Δ+common]':cell,'commonality (tuning)':common,
            'uniq frac Δ/cell':uf_c,'uniq frac Δ/full':uf_f,'geomean(Δ,cell)':gm,'product Δ·cell':prod}

for tag,fn in CACHES:
    c=np.load(DATA+'/'+fn,allow_pickle=True)
    pr2_base=c['pr2_base']; pr2_cell=c['pr2_cell']; pr2_full=c['pr2_full']; net_id=c['net_id']; J=c['J_comb']; N=len(net_id)
    low=pr2_base<0.3
    aj=[];wi=[];B=[];C=[];F=[]
    for ti in range(N):
        if low[ti]: continue
        for ci in range(N):
            if ci==ti or np.isnan(pr2_full[ci,ti]) or np.isnan(pr2_cell[ci,ti]): continue
            aj.append(abs(float(J[ti,ci]))); wi.append(net_id[ci]==net_id[ti])
            B.append(pr2_base[ti]); C.append(pr2_cell[ci,ti]); F.append(pr2_full[ci,ti])
    aj=np.array(aj); wi=np.array(wi,dtype=bool); B=np.array(B); C=np.array(C); F=np.array(F)
    M=metrics(B,C,F)
    print(f'\n===== {tag}  (n_within={wi.sum()}, n_cross={(~wi).sum()}, base≈{np.nanmean(pr2_base[~low]):.2f}) =====')
    print(f'{"metric":22s} | within ρ(|J|) |  p    | within mean | cross mean | cross/within')
    for name,v in M.items():
        m=np.isfinite(v)
        rho,p=spearmanr(aj[wi&m], v[wi&m])
        wm=np.nanmean(v[wi&m]); xm=np.nanmean(v[(~wi)&m]); r=xm/wm if wm!=0 else np.nan
        print(f'{name:22s} | {rho:+.3f}       | {p:.0e} | {wm:+.4f}    | {xm:+.4f}   | {r:+.2f}')
print('\nDONE')
