"""Variant of merge_and_finetune.py aimed at HELPING the network converge on
a genuinely two-module solution, rather than letting joint fine-tuning erode
one scale into the other (which is what happened in both plain merges:
narrow drifted ~15-17% toward wide while cross-block recurrent coupling grew
from 0 to roughly half the within-block norm).

Mechanism: after backward(), gradients on the WITHIN-block portions of
encoder.weight, RNN.weight_hh_l0, RNN.weight_ih_l0 and decoder.weight (i.e.
each subnet's own, already-converged internal solution) are scaled down by
--subnet_lr_scale, while the CROSS-block portions (recurrent + encoder +
decoder off-diagonal blocks, all zero-initialized) keep the full learning
rate. This lets the two populations learn how to talk to each other quickly
without erasing what each one already learned to do on its own -- a direct
test of whether stable two-module coexistence is possible, or whether some
drift is unavoidable even when protected.
"""
import sys, os, time, argparse
REPO_DIR='/Users/harryclark/Documents/spatial-manifolds/GRID-PATTERN-FORMATION'
sys.path.insert(0, REPO_DIR)
import numpy as np, torch
from place_cells import PlaceCells
from trajectory_generator import TrajectoryGenerator
from model import RNN

p = argparse.ArgumentParser()
p.add_argument('--dir_a', required=True, help='narrow/first source net directory')
p.add_argument('--dir_b', required=True, help='wide/second source net directory')
p.add_argument('--pc_rf_a', type=float, required=True)
p.add_argument('--pc_rf_b', type=float, required=True)
p.add_argument('--save_dir', required=True)
p.add_argument('--steps', type=int, default=25000)
p.add_argument('--lr', type=float, default=1e-4)
p.add_argument('--subnet_lr_scale', type=float, default=0.05,
                help='multiplier applied to gradients on each subnet\'s own within-block '
                     'weights; cross-block (previously-zero) weights keep the full rate')
p.add_argument('--sequence_length', type=int, default=20)
p.add_argument('--box_width', type=float, default=2.2)
p.add_argument('--grad_clip', type=float, default=1.0)
args = p.parse_args()

DEVICE = 'mps' if torch.backends.mps.is_available() else ('cuda' if torch.cuda.is_available() else 'cpu')
os.makedirs(args.save_dir, exist_ok=True)

us_a = np.load(os.path.join(args.dir_a, 'place_cells.npy'))
us_b = np.load(os.path.join(args.dir_b, 'place_cells.npy'))
assert np.allclose(us_a, us_b), 'source nets must share the same place-cell centres (same pc_seed)'

ck_a = torch.load(os.path.join(args.dir_a, 'ckpt.pth'), map_location='cpu')['model']
ck_b = torch.load(os.path.join(args.dir_b, 'ckpt.pth'), map_location='cpu')['model']
Ng_a = ck_a['RNN.weight_hh_l0'].shape[0]
Ng_b = ck_b['RNN.weight_hh_l0'].shape[0]
Np_each = us_a.shape[0]
Ng_total = Ng_a + Ng_b
Np_total = Np_each * 2
print(f'source A ({args.dir_a}): Ng={Ng_a}, pc_rf={args.pc_rf_a}', flush=True)
print(f'source B ({args.dir_b}): Ng={Ng_b}, pc_rf={args.pc_rf_b}', flush=True)
print(f'merged: Ng_total={Ng_total}, Np_total={Np_total}, subnet_lr_scale={args.subnet_lr_scale}', flush=True)

class Options: pass
o = Options()
o.Np=Np_total; o.Ng=Ng_total; o.sequence_length=args.sequence_length; o.batch_size=200
o.learning_rate=args.lr; o.weight_decay=1e-4
o.place_cell_rf=[args.pc_rf_a, args.pc_rf_b]; o.surround_scale=2
o.pc_rf_seed=0
o.RNN_type='RNN'; o.activation='relu'; o.DoG=True; o.periodic=False
o.box_width=args.box_width; o.box_height=args.box_width; o.device=DEVICE

pc = PlaceCells(o)
us_merged = np.concatenate([us_a, us_b], axis=0).astype(np.float32)
pc.us = torch.tensor(us_merged, device=DEVICE)
pc.sigma_group = np.concatenate([np.zeros(Np_each, dtype=int), np.ones(Np_each, dtype=int)])
sigma_vals = np.array([args.pc_rf_a, args.pc_rf_b], dtype=np.float32)[pc.sigma_group]
pc.sigma = torch.tensor(sigma_vals, dtype=torch.float32, device=DEVICE)
np.save(os.path.join(args.save_dir, 'place_cells.npy'), us_merged)
np.save(os.path.join(args.save_dir, 'sigma_group.npy'), pc.sigma_group)

ng_group = np.concatenate([np.zeros(Ng_a, dtype=int), np.ones(Ng_b, dtype=int)])
np.save(os.path.join(args.save_dir, 'ng_group.npy'), ng_group)

tg = TrajectoryGenerator(o, pc)
model = RNN(o, pc).to(DEVICE)

with torch.no_grad():
    model.encoder.weight.zero_()
    model.RNN.weight_hh_l0.zero_()
    model.RNN.weight_ih_l0.zero_()
    model.decoder.weight.zero_()

    model.encoder.weight[:Ng_a, :Np_each] = ck_a['encoder.weight']
    model.encoder.weight[Ng_a:, Np_each:] = ck_b['encoder.weight']

    model.RNN.weight_hh_l0[:Ng_a, :Ng_a] = ck_a['RNN.weight_hh_l0']
    model.RNN.weight_hh_l0[Ng_a:, Ng_a:] = ck_b['RNN.weight_hh_l0']

    model.RNN.weight_ih_l0[:Ng_a] = ck_a['RNN.weight_ih_l0']
    model.RNN.weight_ih_l0[Ng_a:] = ck_b['RNN.weight_ih_l0']

    model.decoder.weight[:Np_each, :Ng_a] = ck_a['decoder.weight']
    model.decoder.weight[Np_each:, Ng_a:] = ck_b['decoder.weight']

# ── gradient masks: within-block (each subnet's own weights) scaled down,
# cross-block (zero-initialized, mediates inter-population coupling) full rate
scale = args.subnet_lr_scale
mask_encoder = torch.ones(Ng_total, Np_total, device=DEVICE)
mask_encoder[:Ng_a, :Np_each] = scale; mask_encoder[Ng_a:, Np_each:] = scale
mask_recurrent = torch.ones(Ng_total, Ng_total, device=DEVICE)
mask_recurrent[:Ng_a, :Ng_a] = scale; mask_recurrent[Ng_a:, Ng_a:] = scale
mask_ih = torch.full((Ng_total, 2), scale, device=DEVICE)  # no cross-structure; whole tensor protected
mask_decoder = torch.ones(Np_total, Ng_total, device=DEVICE)
mask_decoder[:Np_each, :Ng_a] = scale; mask_decoder[Np_each:, Ng_a:] = scale

sr = torch.linalg.eigvals(model.RNN.weight_hh_l0.detach().cpu()).abs().max().item()
cross0 = (model.RNN.weight_hh_l0[:Ng_a, Ng_a:].norm() + model.RNN.weight_hh_l0[Ng_a:, :Ng_a].norm()).item()
print(f'merged network built: initial spectral radius={sr:.3f}, initial cross-block norm={cross0:.4f} (should be ~0)', flush=True)

opt = torch.optim.Adam(model.parameters(), lr=o.learning_rate)
history = {'step':0, 'loss':[], 'err':[], 'cross_norm':[], 'within_norm':[]}
gen = tg.get_generator(); model.train(); t0 = time.time()

for step in range(args.steps):
    inp, pco, pos = next(gen); model.zero_grad()
    loss, err = model.compute_loss(inp, pco, pos); loss.backward()
    with torch.no_grad():
        model.encoder.weight.grad *= mask_encoder
        model.RNN.weight_hh_l0.grad *= mask_recurrent
        model.RNN.weight_ih_l0.grad *= mask_ih
        model.decoder.weight.grad *= mask_decoder
    gnorm = torch.nn.utils.clip_grad_norm_(model.parameters(), args.grad_clip)
    opt.step()
    history['loss'].append(loss.item()); history['err'].append(err.item()); history['step']=step+1
    if step % 500 == 0:
        with torch.no_grad():
            cross_norm = (model.RNN.weight_hh_l0[:Ng_a, Ng_a:].norm() + model.RNN.weight_hh_l0[Ng_a:, :Ng_a].norm()).item()
            within_norm = (model.RNN.weight_hh_l0[:Ng_a, :Ng_a].norm() + model.RNN.weight_hh_l0[Ng_a:, Ng_a:].norm()).item()
        history['cross_norm'].append(cross_norm); history['within_norm'].append(within_norm)
        ms = (time.time()-t0)/max(step+1, 1)*1000
        print(f"step {step+1:6d}/{args.steps} loss {loss.item():.3f} err {err.item()*100:6.1f} cm "
             f"grad_norm {gnorm.item():7.2f} cross_norm {cross_norm:8.3f} within_norm {within_norm:8.3f} "
             f"({ms:.0f} ms/step)", flush=True)
    if (step+1) % 1000 == 0 or (step+1) == args.steps:
        torch.save({'model': model.state_dict(), 'optim': opt.state_dict(), 'history': history},
                   os.path.join(args.save_dir, 'ckpt.pth'))

print(f"DONE final err {np.mean(history['err'][-50:])*100:.1f} cm ({(time.time()-t0)/60:.1f} min)", flush=True)
