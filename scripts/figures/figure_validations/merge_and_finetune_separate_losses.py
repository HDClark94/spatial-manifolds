"""Variant of merge_and_finetune.py testing whether the joint/competing
softmax loss -- not the learning rate -- is what actually drove the drift
seen in the plain merge (narrow spacing shifted ~15-17% toward wide, in both
net6 and net7 pairs). The plain merge calls model.compute_loss() unmodified:
a SINGLE softmax across the full combined Np population, so narrow and wide
place cells compete inside one shared classification target -- exactly the
same objective structure as the from-scratch mixed-width attempts that
always collapsed to one scale. A single accurate position code already
satisfies that one shared target, so nothing in the LOSS itself rewards
keeping two distinct scales; drift is the objective eroding "unrewarded"
structure over training, just starting from a strongly pre-structured
initialization so it happens partially rather than outright.

Here the merge/block-diagonal initialization is identical to
merge_and_finetune.py, but fine-tuning uses TWO INDEPENDENT per-population
losses (same grouped-softmax trick as train_two_population_scratch.py),
summed, with a UNIFORM learning rate (no gradient masking) -- isolating the
loss-function change as the only intervention, to test whether removing the
cross-population competition alone is enough to prevent drift.
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
print(f'merged: Ng_total={Ng_total}, Np_total={Np_total} -- SEPARATE per-population losses, uniform LR', flush=True)

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

# ── grouped (per-population, non-competing) activation -- monkey-patched onto
# this pc instance only, same trick as train_two_population_scratch.py
group_idx = [np.where(pc.sigma_group == g)[0] for g in range(2)]
group_idx_t = [torch.tensor(gi, dtype=torch.long, device=DEVICE) for gi in group_idx]
_softmax = torch.nn.Softmax(dim=-1)

def grouped_activation(pos):
    d = torch.abs(pos[:, :, None, :] - pc.us[None, None, ...]).float()
    norm2 = (d ** 2).sum(-1)
    out = torch.zeros_like(norm2)
    for gi in group_idx_t:
        sigma_g = pc.sigma[gi]
        og = _softmax(-norm2[..., gi] / (2 * sigma_g ** 2))
        if pc.DoG:
            og -= _softmax(-norm2[..., gi] / (2 * pc.surround_scale * sigma_g ** 2))
            mn, _ = og.min(-1, keepdims=True)
            og = og + torch.abs(mn)
            og = og / og.sum(-1, keepdims=True)
        out[..., gi] = og
    return out

pc.get_activation = grouped_activation

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

def grouped_loss(inp, pco, pos):
    preds = model.predict(inp)
    yhat = torch.zeros_like(preds)
    for gi in group_idx_t:
        yhat[..., gi] = _softmax(preds[..., gi])
    eps = 1e-8
    loss = -(pco * torch.log(yhat + eps)).sum(-1).mean()
    loss = loss + model.weight_decay * (model.RNN.weight_hh_l0 ** 2).sum()
    pred_pos = pc.get_nearest_cell_pos(preds)
    err = torch.sqrt(((pos - pred_pos) ** 2).sum(-1)).mean()
    return loss, err

sr = torch.linalg.eigvals(model.RNN.weight_hh_l0.detach().cpu()).abs().max().item()
cross0 = (model.RNN.weight_hh_l0[:Ng_a, Ng_a:].norm() + model.RNN.weight_hh_l0[Ng_a:, :Ng_a].norm()).item()
print(f'merged network built: initial spectral radius={sr:.3f}, initial cross-block norm={cross0:.4f} (should be ~0)', flush=True)

opt = torch.optim.Adam(model.parameters(), lr=o.learning_rate)
history = {'step':0, 'loss':[], 'err':[], 'cross_norm':[], 'within_norm':[]}
gen = tg.get_generator(); model.train(); t0 = time.time()

for step in range(args.steps):
    inp, pco, pos = next(gen); model.zero_grad()
    loss, err = grouped_loss(inp, pco, pos); loss.backward()
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
