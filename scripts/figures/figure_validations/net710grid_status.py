"""Phase-aware status for net7(grid)+net10 run: waiting-for-HD -> building (within/cross)."""
import numpy as np, os, sys, time
CACHE='/Users/harryclark/Documents/spatial-manifolds/data/rnn_xgboost/pr2_two_net_710grid_cache.npz'
if not os.path.exists(CACHE):
    print('waiting for HD pipeline to finish before net7(grid)+net10 build...'); sys.exit(0)
c=None
for _ in range(4):
    try: c=np.load(CACHE,allow_pickle=True); break
    except Exception: time.sleep(3)
if c is None: print('BUILD [cache busy, retry next tick]'); sys.exit(0)
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
print(f"BUILD [{int(completed.sum())}/{N}] net7(grid) within={m(w0):+.4f}(n={len(w0)}) | net10 within={m(w1):+.4f}(n={len(w1)}) | cross={m(cr):+.4f}(n={len(cr)})")
