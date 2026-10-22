"""Train one grid RNN from scratch against a MIXTURE of place-cell widths,
to test whether a single, architecturally undivided network spontaneously
differentiates into multiple grid scales/modules (H1/H4-style emergence)
rather than needing pre-specified separate circuits (H2-style).

Same pipeline as train_diffPC.py, but --pc_rf accepts a comma-separated list
of widths (e.g. "0.06,0.16") instead of one float: the Np place cells are
split into equal-sized groups, one per width, group membership shuffled
across place-cell index/position (see place_cells.py).
"""
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
p.add_argument('--activation', default='relu', choices=['relu','tanh'])
p.add_argument('--lr', type=float, default=1e-4)
p.add_argument('--Ng', type=int, default=1024)
p.add_argument('--optimizer', default='adam', choices=['adam','rmsprop'])
p.add_argument('--pc_rf', type=str, default='0.12',
               help='single float (uniform, original behaviour) or comma-separated '
                    'list of floats for a heterogeneous mixture, e.g. "0.06,0.16"')
p.add_argument('--sequence_length', type=int, default=20,
               help='steps per training trajectory (dt=0.02s each) -- longer sequences '
                    'accumulate more real-world path length within one training example, '
                    'creating genuine long-range path-integration pressure')
p.add_argument('--sequence_length_mix', type=str, default='',
               help='comma-separated list of sequence lengths, e.g. "20,100" -- if set, '
                    'OVERRIDES --sequence_length: a fresh length is drawn uniformly at '
                    'random from this list every training step, so different horizons '
                    '(and, if scale tracks horizon, different favoured grid scales) are '
                    'each trained on some fraction of the time within one network.')
p.add_argument('--box_width', type=float, default=2.2)
p.add_argument('--Np', type=int, default=512)
p.add_argument('--grad_clip', type=float, default=1.0,
               help='max gradient norm (this pipeline had none; longer sequence_length '
                    'makes exploding gradients over deeper BPTT much more likely). 0 disables.')
p.add_argument('--irnn_scale', type=float, default=0.0,
               help='if >0, reinitialize the recurrent weight matrix as scale*identity '
                    '(IRNN trick, Le/Jaitly/Hinton 2015) instead of PyTorch default init. '
                    'Default RNN init has spectral radius ~0.6 -> vanishes to ~0 over '
                    '40+ step BPTT for a ReLU RNN; identity init keeps spectral radius ~1.')
p.add_argument('--encoder_mask', action='store_true',
               help='hard-mask the ENCODER (input, Np->Ng) weights so each RNN unit can '
                    'only ever receive input from place cells matching its own assigned '
                    'width group. The recurrent layer stays fully shared/unconstrained -- '
                    'this biases input wiring only, not the pattern-forming dynamics '
                    'themselves. Requires --pc_rf to be a mixture (>1 width).')
args=p.parse_args()

pc_rf_list = [float(x) for x in args.pc_rf.split(',')]
pc_rf = pc_rf_list[0] if len(pc_rf_list) == 1 else pc_rf_list
seqlen_mix = [int(x) for x in args.sequence_length_mix.split(',')] if args.sequence_length_mix else None

DEVICE='mps' if torch.backends.mps.is_available() else ('cuda' if torch.cuda.is_available() else 'cpu')
SAVE_EVERY=1000
SNAP_EVERY=10000
class Options: pass
o=Options()
o.Np=args.Np;o.Ng=args.Ng;o.sequence_length=args.sequence_length;o.batch_size=200
o.learning_rate=args.lr;o.weight_decay=1e-4;o.place_cell_rf=pc_rf;o.surround_scale=2
o.pc_rf_seed=args.pc_seed
o.RNN_type='RNN';o.activation=args.activation;o.DoG=True;o.periodic=False
o.box_width=args.box_width;o.box_height=args.box_width;o.device=DEVICE
if seqlen_mix:
    o.sequence_length = max(seqlen_mix)  # RNN itself is length-agnostic; this is just for any code that reads it at init

os.makedirs(args.save_dir, exist_ok=True)
CKPT=os.path.join(args.save_dir,'ckpt.pth'); PC=os.path.join(args.save_dir,'place_cells.npy')
SIGMA_GROUP=os.path.join(args.save_dir,'sigma_group.npy')
print(f'device={DEVICE} rnn_seed={args.rnn_seed} pc_seed={args.pc_seed} steps={args.steps} '
     f'pc_rf={pc_rf} seq_len={"mix"+str(seqlen_mix) if seqlen_mix else args.sequence_length} '
     f'box_width={args.box_width} Np={args.Np} dir={args.save_dir}', flush=True)

# ── unique place-cell tiling for this network ────────────────────────────────
us=np.random.RandomState(args.pc_seed).uniform(-o.box_width/2, o.box_width/2, (o.Np,2)).astype(np.float32)
np.save(PC, us)
torch.manual_seed(args.rnn_seed); np.random.seed(args.rnn_seed)
pc=PlaceCells(o); pc.us=torch.tensor(us).to(DEVICE)
np.save(SIGMA_GROUP, pc.sigma_group)
print(f'place-cell width groups: {[float(w) for w in pc_rf_list]}, '
     f'counts={np.bincount(pc.sigma_group).tolist()}', flush=True)
tg=TrajectoryGenerator(o,pc)
model=RNN(o,pc).to(DEVICE)
if args.irnn_scale > 0:
    with torch.no_grad():
        model.RNN.weight_hh_l0.copy_(args.irnn_scale * torch.eye(o.Ng, device=DEVICE))
    sr = torch.linalg.eigvals(model.RNN.weight_hh_l0.detach().cpu()).abs().max().item()
    print(f'irnn_init ON: weight_hh_l0 = {args.irnn_scale}*I, spectral radius={sr:.3f}', flush=True)
opt=(torch.optim.RMSprop if args.optimizer=='rmsprop' else torch.optim.Adam)(model.parameters(), lr=o.learning_rate)

# ── optional: hard-mask encoder (input) connectivity by width group ─────────
# Each RNN unit is assigned to one width group (equal split, shuffled); its
# encoder row can only have nonzero weight on place cells from that SAME
# group. Re-applied after every optimizer step since a zero-valued weight
# still receives nonzero gradient in a linear layer (dL/dW[i,j] = grad_i *
# x_j, independent of W[i,j]) -- masking only at init would not hold.
encoder_mask = None
if args.encoder_mask:
    if len(pc_rf_list) < 2:
        raise ValueError('--encoder_mask requires a multi-width --pc_rf mixture')
    ng_rng = np.random.RandomState(args.rnn_seed + 999)
    ng_group = np.resize(np.arange(len(pc_rf_list)), o.Ng)
    ng_rng.shuffle(ng_group)
    np.save(os.path.join(args.save_dir, 'ng_group.npy'), ng_group)
    mask_np = (ng_group[:, None] == pc.sigma_group[None, :]).astype(np.float32)
    encoder_mask = torch.tensor(mask_np, device=DEVICE)
    with torch.no_grad():
        model.encoder.weight.data *= encoder_mask
    print(f'encoder_mask ON: Ng unit groups counts={np.bincount(ng_group).tolist()}, '
         f'kept {encoder_mask.sum().item():.0f}/{encoder_mask.numel()} encoder weights', flush=True)

history={'step':0,'loss':[],'err':[],'seqlen':[]}
if os.path.isfile(CKPT):
    ck=torch.load(CKPT,map_location=DEVICE)
    model.load_state_dict(ck['model']); opt.load_state_dict(ck['optim']); history=ck['history']
    history.setdefault('seqlen', [])
    print(f"resumed at {history['step']}", flush=True)

gen=tg.get_generator(); model.train(); t0=time.time(); start=history['step']
seq_rng=np.random.RandomState(args.rnn_seed + 7777)
for step in range(start, args.steps):
    if seqlen_mix:
        o.sequence_length = int(seq_rng.choice(seqlen_mix))
    inp,pco,pos=next(gen); model.zero_grad()
    loss,err=model.compute_loss(inp,pco,pos); loss.backward()
    gnorm=torch.nn.utils.clip_grad_norm_(model.parameters(), args.grad_clip if args.grad_clip>0 else 1e9)
    opt.step()
    if encoder_mask is not None:
        with torch.no_grad():
            model.encoder.weight.data *= encoder_mask
    history['loss'].append(loss.item()); history['err'].append(err.item()); history['step']=step+1
    history['seqlen'].append(o.sequence_length)
    if step%500==0:
        ms=(time.time()-t0)/max(step-start+1,1)*1000
        sl_tag = f"sl={o.sequence_length} " if seqlen_mix else ""
        print(f"step {step+1:6d}/{args.steps} {sl_tag}loss {loss.item():.3f} err {err.item()*100:6.1f} cm "
             f"grad_norm {gnorm.item():7.2f} ({ms:.0f} ms/step)", flush=True)
    if (step+1)%SAVE_EVERY==0 or (step+1)==args.steps:
        torch.save({'model':model.state_dict(),'optim':opt.state_dict(),'history':history}, CKPT)
    if (step+1)%SNAP_EVERY==0:
        torch.save(model.state_dict(), os.path.join(args.save_dir, f'snap_{step+1}.pth'))
print(f"DONE step={history['step']} final err {np.mean(history['err'][-50:])*100:.1f} cm ({(time.time()-t0)/60:.1f} min)", flush=True)
