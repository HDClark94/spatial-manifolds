"""Train one Ng=1024 grid RNN from scratch with a UNIQUE place-cell tiling."""
import sys, os, time, argparse
REPO_DIR='/Users/harryclark/Documents/spatial-manifolds/GRID-PATTERN-FORMATION'
sys.path.insert(0, REPO_DIR)
import numpy as np, torch
from place_cells import PlaceCells
from trajectory_generator import TrajectoryGenerator
from model import RNN

p=argparse.ArgumentParser()
p.add_argument('--rnn_seed', type=int, required=True)
p.add_argument('--pc_seed',  type=int, required=True)
p.add_argument('--save_dir', required=True)
p.add_argument('--steps',    type=int, default=50000)
p.add_argument('--activation', default='relu', choices=['relu','tanh'])  # tanh removes the non-negativity constraint -> non-grid spatial solution
p.add_argument('--lr', type=float, default=1e-4)
p.add_argument('--Ng', type=int, default=1024)
p.add_argument('--optimizer', default='adam', choices=['adam','rmsprop'])  # paper uses RMSProp
p.add_argument('--pc_rf', type=float, default=0.12)                         # paper σ1 = 0.2
args=p.parse_args()

DEVICE='mps' if torch.backends.mps.is_available() else ('cuda' if torch.cuda.is_available() else 'cpu')
SAVE_EVERY=1000
SNAP_EVERY=10000   # keep model-only weight snapshots (snap_<step>.pth) for across-training analysis
class Options: pass
o=Options()
o.Np=512;o.Ng=args.Ng;o.sequence_length=20;o.batch_size=200
o.learning_rate=args.lr;o.weight_decay=1e-4;o.place_cell_rf=args.pc_rf;o.surround_scale=2
o.RNN_type='RNN';o.activation=args.activation;o.DoG=True;o.periodic=False
o.box_width=2.2;o.box_height=2.2;o.device=DEVICE

os.makedirs(args.save_dir, exist_ok=True)
CKPT=os.path.join(args.save_dir,'ckpt.pth'); PC=os.path.join(args.save_dir,'place_cells.npy')
print(f'device={DEVICE} rnn_seed={args.rnn_seed} pc_seed={args.pc_seed} steps={args.steps} dir={args.save_dir}', flush=True)

# ── unique place-cell tiling for this network ────────────────────────────────
us=np.random.RandomState(args.pc_seed).uniform(-o.box_width/2, o.box_width/2, (o.Np,2)).astype(np.float32)
np.save(PC, us)
torch.manual_seed(args.rnn_seed); np.random.seed(args.rnn_seed)
pc=PlaceCells(o); pc.us=torch.tensor(us).to(DEVICE)
tg=TrajectoryGenerator(o,pc)
model=RNN(o,pc).to(DEVICE)
opt=(torch.optim.RMSprop if args.optimizer=='rmsprop' else torch.optim.Adam)(model.parameters(), lr=o.learning_rate)

history={'step':0,'loss':[],'err':[]}
if os.path.isfile(CKPT):
    ck=torch.load(CKPT,map_location=DEVICE)
    model.load_state_dict(ck['model']); opt.load_state_dict(ck['optim']); history=ck['history']
    print(f"resumed at {history['step']}", flush=True)

gen=tg.get_generator(); model.train(); t0=time.time(); start=history['step']
for step in range(start, args.steps):
    inp,pco,pos=next(gen); model.zero_grad()
    loss,err=model.compute_loss(inp,pco,pos); loss.backward(); opt.step()
    history['loss'].append(loss.item()); history['err'].append(err.item()); history['step']=step+1
    if step%500==0:
        ms=(time.time()-t0)/max(step-start+1,1)*1000
        print(f"step {step+1:6d}/{args.steps} loss {loss.item():.3f} err {err.item()*100:6.1f} cm ({ms:.0f} ms/step)", flush=True)
    if (step+1)%SAVE_EVERY==0 or (step+1)==args.steps:
        torch.save({'model':model.state_dict(),'optim':opt.state_dict(),'history':history}, CKPT)
    if (step+1)%SNAP_EVERY==0:
        torch.save(model.state_dict(), os.path.join(args.save_dir, f'snap_{step+1}.pth'))
print(f"DONE step={history['step']} final err {np.mean(history['err'][-50:])*100:.1f} cm ({(time.time()-t0)/60:.1f} min)", flush=True)
