"""Does the SPATIAL net7+net10 pairing show 'coactivity fooled (high cross) but ΔpR2 clean
(low cross)'? Regenerate activations, compute coactivity within/cross, compare to cached ΔpR2."""
import sys, os
REPO='/Users/harryclark/Documents/spatial-manifolds/GRID-PATTERN-FORMATION'
sys.path.insert(0,REPO); sys.path.insert(0,'/Users/harryclark/Documents/spatial-manifolds/src')
import numpy as np, torch
from place_cells import PlaceCells
from trajectory_generator import TrajectoryGenerator
from model import RNN
DATA='/Users/harryclark/Documents/spatial-manifolds/data/rnn_xgboost'
NETA=DATA+'/survey_s7_Ng1024'; NETB=DATA+'/survey_s10_Ng1024'; CACHE=DATA+'/pr2_two_net_710_cache.npz'
DEVICE='mps' if torch.backends.mps.is_available() else 'cpu'; N_BATCHES=10
c=np.load(CACHE,allow_pickle=True)
pr2_base=c['pr2_base']; pr2_full=c['pr2_full']; net_id=c['net_id']; sel1=c['sel1']; sel2=c['sel2']; K=int(c['K']); N=len(net_id)
def build(d):
    class O: pass
    o=O(); o.Np=512;o.Ng=1024;o.sequence_length=20;o.batch_size=200
    o.learning_rate=1e-4;o.weight_decay=1e-4;o.place_cell_rf=0.12;o.surround_scale=2
    o.RNN_type='RNN';o.activation='relu';o.DoG=True;o.periodic=False;o.box_width=2.2;o.box_height=2.2;o.device=DEVICE
    pc=PlaceCells(o); pc.us=torch.tensor(np.load(d+'/place_cells.npy'),dtype=torch.float32).to(DEVICE)
    m=RNN(o,pc).to(DEVICE); m.load_state_dict(torch.load(d+'/ckpt.pth',map_location=DEVICE)['model']); m.eval(); return o,pc,m
o1,pcA,mA=build(NETA); o2,pcB,mB=build(NETB); tgA=TrajectoryGenerator(o1,pcA)
def jb():
    t=tgA.generate_trajectory(o1.box_width,o1.box_height,o1.batch_size)
    v=np.stack([t['ego_v']*np.cos(t['target_hd']),t['ego_v']*np.sin(t['target_hd'])],-1)
    v=torch.tensor(v,dtype=torch.float32).transpose(0,1).to(DEVICE)
    ip=np.stack([t['init_x'],t['init_y']],-1); ip=torch.tensor(ip,dtype=torch.float32).to(DEVICE)
    return (v,pcA.get_activation(ip).squeeze()),(v,pcB.get_activation(ip).squeeze())
torch.manual_seed(1); np.random.seed(1)
G1,G2=[],[]
with torch.no_grad():
    for _ in range(N_BATCHES):
        a,b=jb(); G1.append(mA.g(a).cpu().numpy().transpose(1,0,2).reshape(-1,o1.Ng)[:,sel1]); G2.append(mB.g(b).cpu().numpy().transpose(1,0,2).reshape(-1,o2.Ng)[:,sel2])
G=np.concatenate([np.concatenate(G1,0),np.concatenate(G2,0)],1)
co=np.corrcoef(G.T); np.fill_diagonal(co,np.nan)
d=pr2_full-pr2_base[None,:]; low=pr2_base<0.3
def wx(M,use_abs,directed):
    wv=[];xv=[]
    for ti in range(N):
        if low[ti]: continue
        for ci in range(N):
            if ci==ti: continue
            v=M[ci,ti] if directed else M[ti,ci]
            if np.isnan(v): continue
            v=abs(v) if use_abs else v
            (wv if net_id[ci]==net_id[ti] else xv).append(v)
    return np.mean(wv),np.mean(xv)
cw,cx=wx(co,True,False); dw,dxx=wx(d,False,True)
print('=== SPATIAL pairing net7+net10 ===')
print(f'coactivity |r|:  within={cw:.3f}  cross={cx:.3f}  cross/within={cx/cw:.2f}')
print(f'ΔpR²:           within={dw:.4f} cross={dxx:.4f} cross/within={dxx/dw:.2f}')
print(f'-> coactivity separation {cx/cw:.2f} (≈1 = FOOLED) vs ΔpR² separation {dxx/dw:.2f} (<1 = distinguishes)')
print('DONE')
