"""Train a SINGLE RNN from scratch (random init, no pretraining/merging) with
Ng units arbitrarily split into two halves ("subnetworks" A and B), each
outputting onto its OWN independently-normalized place-cell population
(narrow vs wide widths, same centers). Unlike train_two_population_scratch.py
(uniform learning rate everywhere -> collapsed to one scale) this run gives
WITHIN-subnetwork weights (A-to-A, B-to-B: recurrent, encoder, decoder) the
normal learning rate, while CROSS-subnetwork weights (A-to-B, B-to-A) get a
much smaller rate (--cross_lr_scale). The idea: let each half rapidly
develop its own specialized dynamics against its own place-cell population,
while inter-half coupling grows more slowly -- avoiding the premature
homogenization that may have driven the earlier uniform-LR run to collapse
onto a single shared scale. No pretrained checkpoints are used anywhere.
"""
import sys, os, time, argparse
REPO_DIR='/Users/harryclark/Documents/spatial-manifolds/GRID-PATTERN-FORMATION'
sys.path.insert(0, REPO_DIR)
import numpy as np, torch
from place_cells import PlaceCells
from trajectory_generator import TrajectoryGenerator
from model import RNN

p = argparse.ArgumentParser()
p.add_argument('--rnn_seed', type=int, required=True)
p.add_argument('--pc_seed', type=int, default=0)
p.add_argument('--save_dir', required=True)
p.add_argument('--steps', type=int, default=50000)
p.add_argument('--lr', type=float, default=1e-4)
p.add_argument('--Ng', type=int, default=2048)
p.add_argument('--Np', type=int, default=1024)
p.add_argument('--pc_rf', type=str, default='0.12,0.1944')
p.add_argument('--sequence_length', type=int, default=20)
p.add_argument('--box_width', type=float, default=2.2)
p.add_argument('--grad_clip', type=float, default=1.0)
p.add_argument('--irnn_scale', type=float, default=0.99)
p.add_argument('--cross_lr_scale', type=float, default=0.05,
                help='multiplier applied to gradients on CROSS-subnetwork weights '
                     '(A<->B); within-subnetwork weights (A-A, B-B) keep the full rate')
args = p.parse_args()

DEVICE = 'mps' if torch.backends.mps.is_available() else ('cuda' if torch.cuda.is_available() else 'cpu')
os.makedirs(args.save_dir, exist_ok=True)
torch.manual_seed(args.rnn_seed); np.random.seed(args.rnn_seed)

pc_rf_list = [float(x) for x in args.pc_rf.split(',')]
assert len(pc_rf_list) == 2, 'this script tests exactly two subnetworks / two output populations'
assert args.Ng % 2 == 0 and args.Np % 2 == 0

class Options: pass
o = Options()
o.Np=args.Np; o.Ng=args.Ng; o.sequence_length=args.sequence_length; o.batch_size=200
o.learning_rate=args.lr; o.weight_decay=1e-4; o.place_cell_rf=pc_rf_list; o.surround_scale=2
o.pc_rf_seed=args.pc_seed
o.RNN_type='RNN'; o.activation='relu'; o.DoG=True; o.periodic=False
o.box_width=args.box_width; o.box_height=args.box_width; o.device=DEVICE

pc = PlaceCells(o)
# force a clean (non-shuffled) block structure for the place-cell groups too,
# so "population A/B" lines up cleanly with "subnetwork A/B" for reporting
Np_half = args.Np // 2
pc.sigma_group = np.concatenate([np.zeros(Np_half, dtype=int), np.ones(Np_half, dtype=int)])
widths = np.array(pc_rf_list, dtype=np.float32)
pc.sigma = torch.tensor(widths[pc.sigma_group], dtype=torch.float32, device=DEVICE)

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

if args.irnn_scale > 0:
    with torch.no_grad():
        model.RNN.weight_hh_l0.copy_(args.irnn_scale * torch.eye(o.Ng, device=DEVICE))
    sr = torch.linalg.eigvals(model.RNN.weight_hh_l0.detach().cpu()).abs().max().item()
    print(f'irnn_init ON: weight_hh_l0 = {args.irnn_scale}*I, spectral radius={sr:.3f} '
          f'(identity -> cross-subnet recurrent blocks start at exactly 0)', flush=True)

Ng_half = args.Ng // 2
ng_group = np.concatenate([np.zeros(Ng_half, dtype=int), np.ones(Ng_half, dtype=int)])
np.save(os.path.join(args.save_dir, 'place_cells.npy'), pc.us.cpu().numpy())
np.save(os.path.join(args.save_dir, 'sigma_group.npy'), pc.sigma_group)
np.save(os.path.join(args.save_dir, 'ng_group.npy'), ng_group)

# ── gradient masks: WITHIN-subnet (A-A, B-B) full rate, CROSS-subnet (A-B, B-A) slowed
scale = args.cross_lr_scale
mask_encoder = torch.full((args.Ng, args.Np), scale, device=DEVICE)
mask_encoder[:Ng_half, :Np_half] = 1.0; mask_encoder[Ng_half:, Np_half:] = 1.0
mask_recurrent = torch.full((args.Ng, args.Ng), scale, device=DEVICE)
mask_recurrent[:Ng_half, :Ng_half] = 1.0; mask_recurrent[Ng_half:, Ng_half:] = 1.0
mask_decoder = torch.full((args.Np, args.Ng), scale, device=DEVICE)
mask_decoder[:Np_half, :Ng_half] = 1.0; mask_decoder[Np_half:, Ng_half:] = 1.0
# weight_ih_l0 (velocity input) has no cross-subnet structure -- always "within", full rate

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

opt = torch.optim.Adam(model.parameters(), lr=o.learning_rate)
history = {'step': 0, 'loss': [], 'err': [], 'cross_norm': [], 'within_norm': []}
start_step = 0
ckpt_path = os.path.join(args.save_dir, 'ckpt.pth')
if os.path.isfile(ckpt_path):
    ck = torch.load(ckpt_path, map_location=DEVICE)
    model.load_state_dict(ck['model']); opt.load_state_dict(ck['optim']); history = ck['history']
    start_step = history['step']
    print(f'resumed at {start_step}', flush=True)

print(f'device={DEVICE} rnn_seed={args.rnn_seed} pc_seed={args.pc_seed} steps={args.steps} '
      f'pc_rf={pc_rf_list} seq_len={args.sequence_length} box_width={args.box_width} '
      f'Np={args.Np} Ng={args.Ng} cross_lr_scale={args.cross_lr_scale} dir={args.save_dir}', flush=True)
print(f'place-cell width groups: {pc_rf_list}, counts={[len(gi) for gi in group_idx]}, '
      f'ng_group counts={np.bincount(ng_group).tolist()}', flush=True)

gen = tg.get_generator(); model.train(); t0 = time.time()
for step in range(start_step, args.steps):
    inp, pco, pos = next(gen); model.zero_grad()
    loss, err = grouped_loss(inp, pco, pos); loss.backward()
    with torch.no_grad():
        model.encoder.weight.grad *= mask_encoder
        model.RNN.weight_hh_l0.grad *= mask_recurrent
        model.decoder.weight.grad *= mask_decoder
    gnorm = torch.nn.utils.clip_grad_norm_(model.parameters(), args.grad_clip)
    opt.step()
    history['loss'].append(loss.item()); history['err'].append(err.item()); history['step'] = step + 1
    if step % 500 == 0:
        with torch.no_grad():
            cross_norm = (model.RNN.weight_hh_l0[:Ng_half, Ng_half:].norm() + model.RNN.weight_hh_l0[Ng_half:, :Ng_half].norm()).item()
            within_norm = (model.RNN.weight_hh_l0[:Ng_half, :Ng_half].norm() + model.RNN.weight_hh_l0[Ng_half:, Ng_half:].norm()).item()
        history['cross_norm'].append(cross_norm); history['within_norm'].append(within_norm)
        ms = (time.time() - t0) / max(step - start_step + 1, 1) * 1000
        print(f"step {step+1:6d}/{args.steps} loss {loss.item():.3f} err {err.item()*100:6.1f} cm "
              f"grad_norm {gnorm.item():7.2f} cross_norm {cross_norm:8.3f} within_norm {within_norm:8.3f} "
              f"({ms:.0f} ms/step)", flush=True)
    if (step + 1) % 1000 == 0 or (step + 1) == args.steps:
        torch.save({'model': model.state_dict(), 'optim': opt.state_dict(), 'history': history}, ckpt_path)

print(f"DONE step={args.steps} final err {np.mean(history['err'][-50:])*100:.1f} cm "
      f"({(time.time()-t0)/60:.1f} min)", flush=True)
