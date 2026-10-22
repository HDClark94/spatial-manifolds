"""Post-hoc decode-error evaluation for a two-population network: decodes
position separately from each population's own top-k RAW decoder logits
(same method as PlaceCells.get_nearest_cell_pos, just restricted to each
group's own indices instead of mixed across both populations).

Note: an earlier version of this script tried a probability-weighted
centroid decode using the group-wise-softmax'd predictions instead of raw
logits. That turned out to be a strictly worse metric for this codebase's
place-cell code: even the TRUE target distribution (from PlaceCells'
DoG-softmax-shift-rescale construction) is only weakly peaked (~0.02 max
probability over 512 cells, barely above the ~0.002 uniform baseline) --
the shift-and-renormalize step needed for the DoG surround subtraction adds
a near-uniform offset that swamps the actual spatial signal. A weighted
average over such a barely-peaked distribution collapses to roughly the
centroid of all cells (~box center) regardless of true position. Raw
decoder logits avoid this because the trained network can push them
arbitrarily far apart pre-softmax -- that's why top-k-of-raw-logits was the
right choice all along; the only real fix needed was doing it per-group
instead of mixed across both populations.
"""
import sys, os, argparse
REPO_DIR='/Users/harryclark/Documents/spatial-manifolds/GRID-PATTERN-FORMATION'
sys.path.insert(0, REPO_DIR)
import numpy as np, torch
from place_cells import PlaceCells
from trajectory_generator import TrajectoryGenerator
from model import RNN

p = argparse.ArgumentParser()
p.add_argument('--dir', required=True)
p.add_argument('--Ng', type=int, default=2048)
p.add_argument('--Np', type=int, default=1024)
p.add_argument('--pc_rf', type=str, required=True)
p.add_argument('--pc_seed', type=int, required=True)
p.add_argument('--sequence_length', type=int, default=20)
p.add_argument('--box_width', type=float, default=2.2)
p.add_argument('--n_batches', type=int, default=10)
p.add_argument('--activation_mode', type=str, default='grouped', choices=['grouped', 'joint'],
               help='grouped = per-population independent softmax (matches networks trained with '
                    'separate losses, e.g. merge_and_finetune_separate_losses.py / '
                    'train_two_population_scratch.py). joint = single softmax across the whole '
                    'combined population (matches merge_and_finetune.py / plain mixture training). '
                    'This MUST match how the network was actually trained -- it also determines the '
                    'trajectory generator\'s initial-state encoding, not just the loss target, so a '
                    'mismatch here degrades decode error as an artifact, not a real effect.')
args = p.parse_args()

pc_rf_list = [float(x) for x in args.pc_rf.split(',')]
assert len(pc_rf_list) == 2

DEVICE = 'mps' if torch.backends.mps.is_available() else ('cuda' if torch.cuda.is_available() else 'cpu')

class Options: pass
o = Options()
o.Np=args.Np; o.Ng=args.Ng; o.sequence_length=args.sequence_length; o.batch_size=200
o.learning_rate=1e-4; o.weight_decay=1e-4; o.place_cell_rf=pc_rf_list; o.surround_scale=2
o.pc_rf_seed=args.pc_seed
o.RNN_type='RNN'; o.activation='relu'; o.DoG=True; o.periodic=False
o.box_width=args.box_width; o.box_height=args.box_width; o.device=DEVICE

pc = PlaceCells(o)
pc.us = torch.tensor(np.load(os.path.join(args.dir, 'place_cells.npy')), dtype=torch.float32).to(DEVICE)
saved_sigma_group_path = os.path.join(args.dir, 'sigma_group.npy')
if os.path.isfile(saved_sigma_group_path):
    pc.sigma_group = np.load(saved_sigma_group_path)
widths = np.asarray(pc_rf_list, dtype=np.float32)
pc.sigma = torch.tensor(widths[pc.sigma_group], dtype=torch.float32, device=DEVICE)

group_idx = [np.where(pc.sigma_group == g)[0] for g in range(2)]
group_idx_t = [torch.tensor(gi, dtype=torch.long, device=DEVICE) for gi in group_idx]
us_group = [pc.us[gi] for gi in group_idx_t]
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

if args.activation_mode == 'grouped':
    pc.get_activation = grouped_activation
# else: leave PlaceCells' default joint (single-softmax-across-all-Np) get_activation in place

model = RNN(o, pc).to(DEVICE)
model.load_state_dict(torch.load(os.path.join(args.dir, 'ckpt.pth'), map_location=DEVICE)['model'])
model.eval()
tg = TrajectoryGenerator(o, pc)

def topk_decode(logits_sub, us_sub, k=3):
    kk = min(k, logits_sub.shape[-1])
    _, idxs = torch.topk(logits_sub, k=kk, dim=-1)
    return us_sub[idxs].mean(-2)

errs_narrow, errs_wide, errs_combined, errs_raw = [], [], [], []
with torch.no_grad():
    gen = tg.get_generator()
    for _ in range(args.n_batches):
        inp, pco, pos = next(gen)
        preds = model.predict(inp)

        pred_pos_narrow = topk_decode(preds[..., group_idx_t[0]], us_group[0])
        pred_pos_wide = topk_decode(preds[..., group_idx_t[1]], us_group[1])
        pred_pos_combined = (pred_pos_narrow + pred_pos_wide) / 2
        pred_pos_raw = pc.get_nearest_cell_pos(preds)

        errs_narrow.append(torch.sqrt(((pos - pred_pos_narrow) ** 2).sum(-1)).mean().item())
        errs_wide.append(torch.sqrt(((pos - pred_pos_wide) ** 2).sum(-1)).mean().item())
        errs_combined.append(torch.sqrt(((pos - pred_pos_combined) ** 2).sum(-1)).mean().item())
        errs_raw.append(torch.sqrt(((pos - pred_pos_raw) ** 2).sum(-1)).mean().item())

print(f'{args.dir}', flush=True)
print(f'  err_narrow (top-3 raw logits, narrow group only) = {np.mean(errs_narrow)*100:.2f} cm', flush=True)
print(f'  err_wide   (top-3 raw logits, wide group only)   = {np.mean(errs_wide)*100:.2f} cm', flush=True)
print(f'  err_combined (avg of narrow+wide estimates)      = {np.mean(errs_combined)*100:.2f} cm', flush=True)
print(f'  err_raw_topk (old metric, top-3 mixed across both)= {np.mean(errs_raw)*100:.2f} cm', flush=True)
print('EVAL_DONE', flush=True)
