"""
Train a HEAD-DIRECTION integrator RNN (a 'different task' partner net).
Same Sorscher RNN architecture (2D input, Ng hidden, ReLU, no bias), but:
  - input  = angular velocity [cos(dθ), sin(dθ)] = trajectory phi_x, phi_y  (not translational velocity)
  - target = HD-cell code: von Mises tuning over heading θ (1D circular), NOT 2D place cells
So the net integrates angular velocity into heading -> hidden units become HEAD-DIRECTION cells,
a representation ~independent of position. Saved like the grid nets (ckpt.pth + hd_cells.npy) so the
two-net cache builder can load it identically. Usage: --save_dir DIR [--steps N --rnn_seed S].
"""
import sys, os, argparse, time, numpy as np, torch
REPO='/Users/harryclark/Documents/spatial-manifolds/GRID-PATTERN-FORMATION'
sys.path.insert(0, REPO)
from trajectory_generator import TrajectoryGenerator
from model import RNN

ap=argparse.ArgumentParser()
ap.add_argument('--rnn_seed', type=int, default=7010)
ap.add_argument('--save_dir', required=True)
ap.add_argument('--steps', type=int, default=100000)
ap.add_argument('--Nhd', type=int, default=256)
ap.add_argument('--kappa', type=float, default=4.0)
args=ap.parse_args()
DEVICE='mps' if torch.backends.mps.is_available() else ('cuda' if torch.cuda.is_available() else 'cpu')
os.makedirs(args.save_dir, exist_ok=True)
print(f'device={DEVICE} | HD integrator | Nhd={args.Nhd} kappa={args.kappa} steps={args.steps}', flush=True)

class HDCells:
    """von Mises HD tuning over heading; drop-in for PlaceCells (get_activation)."""
    def __init__(self, Nhd, kappa, device):
        self.Nhd=Nhd; self.kappa=kappa; self.device=device
        self.us=torch.linspace(-np.pi, np.pi, Nhd+1)[:-1].to(device)   # preferred headings
    def get_activation(self, hd):        # hd (...,1) -> (...,Nhd) softmax von Mises
        return torch.softmax(self.kappa*torch.cos(hd - self.us), dim=-1)
    def get_nearest(self, act, k=3):     # circular mean of top-k preferred headings
        _,idx=torch.topk(act,k,dim=-1); mu=self.us[idx]
        return torch.atan2(torch.sin(mu).mean(-1), torch.cos(mu).mean(-1))

class O: pass
o=O(); o.Np=args.Nhd; o.Ng=1024; o.sequence_length=20; o.batch_size=200
o.learning_rate=1e-4; o.weight_decay=1e-4; o.place_cell_rf=0.12; o.surround_scale=2
o.RNN_type='RNN'; o.activation='relu'; o.DoG=True; o.periodic=False
o.box_width=2.2; o.box_height=2.2; o.device=DEVICE

torch.manual_seed(args.rnn_seed); np.random.seed(args.rnn_seed)
hd=HDCells(args.Nhd, args.kappa, DEVICE)
model=RNN(o, hd).to(DEVICE)
opt=torch.optim.Adam(model.parameters(), lr=o.learning_rate)
tg=TrajectoryGenerator(o, hd)

def make_batch():
    traj=tg.generate_trajectory(o.box_width,o.box_height,o.batch_size)
    phi=np.stack([traj['phi_x'],traj['phi_y']],-1)                                  # (batch,seq,2)
    phi=torch.tensor(phi,dtype=torch.float32).transpose(0,1).to(DEVICE)             # (seq,batch,2)
    thd=torch.tensor(traj['target_hd'],dtype=torch.float32).to(DEVICE)[...,None]    # (batch,seq,1)
    ihd=torch.tensor(traj['init_hd'],dtype=torch.float32).to(DEVICE)[:,None,:]      # (batch,1,1)
    return phi, hd.get_activation(thd), hd.get_activation(ihd).squeeze(1), thd.squeeze(-1)

t0=time.time()
for step in range(args.steps):
    phi,hd_code,init_code,thd=make_batch()
    preds=model.predict((phi, init_code)).transpose(0,1)          # (batch,seq,Nhd)
    loss=-(hd_code*torch.log_softmax(preds,dim=-1)).sum(-1).mean()
    J=model.RNN.weight_hh_l0
    loss=loss + o.weight_decay*(J**2).sum()
    opt.zero_grad(); loss.backward(); opt.step()
    if step%1000==0 or step==args.steps-1:
        with torch.no_grad():
            pred_hd=hd.get_nearest(torch.softmax(preds,dim=-1))
            err=torch.atan2(torch.sin(pred_hd-thd),torch.cos(pred_hd-thd)).abs().mean()*180/np.pi
        print(f'step {step} loss {loss.item():.4f} hd_err {err.item():.1f} deg ({(time.time()-t0)/60:.1f} min)', flush=True)
    if step%5000==0 or step==args.steps-1:
        torch.save({'model':model.state_dict()}, args.save_dir+'/ckpt.pth')
        np.save(args.save_dir+'/hd_cells.npy', hd.us.cpu().numpy())

# ── final validation: heading integration + HD tuning of hidden units ─────────
with torch.no_grad():
    phi,hd_code,init_code,thd=make_batch()
    G=model.g((phi, init_code)).transpose(0,1).reshape(-1,o.Ng).cpu().numpy()       # (T,Ng)
    pred_hd=hd.get_nearest(torch.softmax(model.predict((phi,init_code)).transpose(0,1),dim=-1))
    err=torch.atan2(torch.sin(pred_hd-thd),torch.cos(pred_hd-thd)).abs().mean()*180/np.pi
    hdv=thd.reshape(-1).cpu().numpy()
    # HD tuning: modulation of each unit across 24 heading bins
    bins=np.linspace(-np.pi,np.pi,25); which=np.digitize(hdv,bins)-1
    tc=np.array([[G[which==b,i].mean() if (which==b).any() else 0 for b in range(24)] for i in range(o.Ng)])
    mod=(tc.max(1)-tc.min(1))/(tc.mean(1)+1e-6)
    print(f'FINAL hd_err {err.item():.1f} deg | HD-modulated units (mod>1): {(mod>1).sum()}/{o.Ng} | active {(G.std(0)>1e-3).sum()}', flush=True)
print('HD_TRAIN DONE', flush=True)
