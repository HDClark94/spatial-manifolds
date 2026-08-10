"""One-line running stats from the partial net7+net10 cache: within (each net) + cross ΔpR2
over COMPLETED target columns (base-filtered), robust to mid-write reads."""
import numpy as np, sys, time
CACHE='/Users/harryclark/Documents/spatial-manifolds/data/rnn_xgboost/pr2_two_net_710_cache.npz'
c=None
for _ in range(4):
    try: c=np.load(CACHE,allow_pickle=True); break
    except Exception: time.sleep(3)
if c is None: print('[stats: cache busy, retry next tick]'); sys.exit(0)
pr2_base=c['pr2_base']; pr2_full=c['pr2_full']; net_id=c['net_id']; completed=c['completed']; N=len(net_id)
w0=[]; w1=[]; cr=[]
for ti in np.where(completed)[0]:
    if pr2_base[ti]<0.3: continue
    for ci in range(N):
        if ci==ti or np.isnan(pr2_full[ci,ti]): continue
        v=float(pr2_full[ci,ti]-pr2_base[ti])
        if net_id[ci]==net_id[ti]: (w0 if net_id[ti]==0 else w1).append(v)
        else: cr.append(v)
m=lambda a:(np.mean(a) if a else float('nan'))
print(f"[{int(completed.sum())}/{N}] net7 within={m(w0):+.4f} (n={len(w0)}) | "
      f"net10 within={m(w1):+.4f} (n={len(w1)}) | cross={m(cr):+.4f} (n={len(cr)})")
