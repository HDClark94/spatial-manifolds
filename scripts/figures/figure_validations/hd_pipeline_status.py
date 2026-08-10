"""Phase-aware one-line status for the HD pipeline monitor:
waiting-for-net710  ->  HD training (hd_err)  ->  hd7 cache build (within/cross)."""
import numpy as np, os, sys, time
CACHE='/Users/harryclark/Documents/spatial-manifolds/data/rnn_xgboost/pr2_two_net_hd7_cache.npz'
LOG='/Users/harryclark/Documents/spatial-manifolds/data/rnn_xgboost/hd_pipeline.log'
if os.path.exists(CACHE):
    c=None
    for _ in range(4):
        try: c=np.load(CACHE,allow_pickle=True); break
        except Exception: time.sleep(3)
    if c is None: print('BUILD [hd7 cache busy, retry next tick]'); sys.exit(0)
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
    print(f"BUILD [{int(completed.sum())}/{N}] net7 within={m(w0):+.4f}(n={len(w0)}) | HD within={m(w1):+.4f}(n={len(w1)}) | cross={m(cr):+.4f}(n={len(cr)})")
else:
    try: lines=open(LOG).read().splitlines()
    except Exception: lines=[]
    hd=[l for l in lines if 'hd_err' in l]
    if any('HD_TRAIN DONE' in l for l in lines): print('TRAIN done; starting hd7 cache build...')
    elif hd: print('TRAIN '+hd[-1])
    else: print('waiting for net710 to finish before HD training...')
