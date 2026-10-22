"""Train a SINGLE RNN from scratch (no pretrain/merge) against TWO
independently-normalized place-cell populations (different widths), to test
whether two grid modules emerge spontaneously when the two populations are
never forced to compete in one shared softmax.

This differs from train_multiscale.py's mixture support in exactly one way:
there, PlaceCells.get_activation applies ONE joint softmax across the whole
combined Np population, so narrow- and wide-width cells compete inside a
single classification target -- a single accurate position code already
satisfies that one task, so nothing pressures the network to grow two
distinct internal scales (which is likely why all from-scratch mixture/
sequence-length/encoder-mask attempts this session converged on one scale).
Here each width group gets its OWN softmax (its own independent read-out
task), summed into the loss -- a real multi-task objective instead of one
merged one. The RNN, encoder and decoder stay fully dense/unconstrained: any
modularity has to emerge on its own, not be architecturally imposed.

Implementation note: PlaceCells.get_activation is monkey-patched with a
group-wise version on the constructed instance only (no edits to the shared
place_cells.py). TrajectoryGenerator is otherwise used unmodified, since it
just calls place_cells.get_activation(pos) internally. The joint decoder
softmax in model.compute_loss is bypassed by a local grouped-loss function;
model.g/model.predict (encoder/RNN/decoder) are reused unchanged.
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
args = p.parse_args()

DEVICE = 'mps' if torch.backends.mps.is_available() else ('cuda' if torch.cuda.is_available() else 'cpu')
os.makedirs(args.save_dir, exist_ok=True)
torch.manual_seed(args.rnn_seed); np.random.seed(args.rnn_seed)

pc_rf_list = [float(x) for x in args.pc_rf.split(',')]
assert len(pc_rf_list) == 2, 'this script tests exactly two output populations'

class Options: pass
o = Options()
o.Np=args.Np; o.Ng=args.Ng; o.sequence_length=args.sequence_length; o.batch_size=200
o.learning_rate=args.lr; o.weight_decay=1e-4; o.place_cell_rf=pc_rf_list; o.surround_scale=2
o.pc_rf_seed=args.pc_seed
o.RNN_type='RNN'; o.activation='relu'; o.DoG=True; o.periodic=False
o.box_width=args.box_width; o.box_height=args.box_width; o.device=DEVICE

pc = PlaceCells(o)
group_idx = [np.where(pc.sigma_group == g)[0] for g in range(len(pc_rf_list))]
group_idx_t = [torch.tensor(gi, dtype=torch.long, device=DEVICE) for gi in group_idx]
_softmax = torch.nn.Softmax(dim=-1)

def grouped_activation(pos):
    '''Same DoG-difference-of-softmax formula as PlaceCells.get_activation,
    but each width group is normalized (softmax'd, DoG-shifted-and-rescaled)
    independently over its OWN indices only -- no cross-group competition.'''
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

pc.get_activation = grouped_activation  # monkey-patch: TrajectoryGenerator calls this internally

tg = TrajectoryGenerator(o, pc)
model = RNN(o, pc).to(DEVICE)

if args.irnn_scale > 0:
    with torch.no_grad():
        model.RNN.weight_hh_l0.copy_(args.irnn_scale * torch.eye(o.Ng, device=DEVICE))
    sr = torch.linalg.eigvals(model.RNN.weight_hh_l0.detach().cpu()).abs().max().item()
    print(f'irnn_init ON: weight_hh_l0 = {args.irnn_scale}*I, spectral radius={sr:.3f}', flush=True)

np.save(os.path.join(args.save_dir, 'place_cells.npy'), pc.us.cpu().numpy())
np.save(os.path.join(args.save_dir, 'sigma_group.npy'), pc.sigma_group)

def grouped_loss(inp, pco, pos):
    preds = model.predict(inp)  # raw decoder logits, shape [T,B,Np]
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
history = {'step': 0, 'loss': [], 'err': []}
start_step = 0
ckpt_path = os.path.join(args.save_dir, 'ckpt.pth')
if os.path.isfile(ckpt_path):
    ck = torch.load(ckpt_path, map_location=DEVICE)
    model.load_state_dict(ck['model']); opt.load_state_dict(ck['optim']); history = ck['history']
    start_step = history['step']
    print(f'resumed at {start_step}', flush=True)

print(f'device={DEVICE} rnn_seed={args.rnn_seed} pc_seed={args.pc_seed} steps={args.steps} '
      f'pc_rf={pc_rf_list} seq_len={args.sequence_length} box_width={args.box_width} '
      f'Np={args.Np} Ng={args.Ng} dir={args.save_dir}', flush=True)
print(f'place-cell width groups: {pc_rf_list}, counts={[len(gi) for gi in group_idx]}', flush=True)

gen = tg.get_generator(); model.train(); t0 = time.time()
for step in range(start_step, args.steps):
    inp, pco, pos = next(gen); model.zero_grad()
    loss, err = grouped_loss(inp, pco, pos); loss.backward()
    gnorm = torch.nn.utils.clip_grad_norm_(model.parameters(), args.grad_clip)
    opt.step()
    history['loss'].append(loss.item()); history['err'].append(err.item()); history['step'] = step + 1
    if step % 500 == 0:
        ms = (time.time() - t0) / max(step - start_step + 1, 1) * 1000
        print(f"step {step+1:6d}/{args.steps} loss {loss.item():.3f} err {err.item()*100:6.1f} cm "
              f"grad_norm {gnorm.item():7.2f} ({ms:.0f} ms/step)", flush=True)
    if (step + 1) % 1000 == 0 or (step + 1) == args.steps:
        torch.save({'model': model.state_dict(), 'optim': opt.state_dict(), 'history': history}, ckpt_path)

print(f"DONE step={args.steps} final err {np.mean(history['err'][-50:])*100:.1f} cm "
      f"({(time.time()-t0)/60:.1f} min)", flush=True)
