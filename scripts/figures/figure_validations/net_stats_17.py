"""Per-net within ΔpR² + cross ΔpR² from the (partial) nets-6+7 cache. net_id 0=net1, 1=net7."""
import numpy as np
c=np.load('/Users/harryclark/Documents/spatial-manifolds/data/rnn_xgboost/pr2_two_net_17_cache.npz', allow_pickle=True)
pr2_base=c['pr2_base']; pr2_full=c['pr2_full']; net_id=c['net_id']; completed=c['completed']; N=len(net_id)
low=pr2_base<0.3
w={0:[],1:[]}; cross=[]
for ti in range(N):
    if not completed[ti] or low[ti]: continue
    for ci in range(N):
        if ci==ti or np.isnan(pr2_full[ci,ti]): continue
        d=float(pr2_full[ci,ti]-pr2_base[ti])
        (w[int(net_id[ti])] if net_id[ci]==net_id[ti] else cross).append(d)
mn=lambda x: (sum(x)/len(x)) if len(x) else float('nan')
print(f"{int(completed.sum())}/{N} done | net1 within={mn(w[0]):+.4f} | net7 within={mn(w[1]):+.4f} | cross={mn(cross):+.4f}")
