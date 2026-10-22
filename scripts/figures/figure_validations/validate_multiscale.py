"""Validate a train_multiscale.py run: does a single RNN trained against a
MIXTURE of place-cell widths form grid cells at more than one spatial scale?

For every Ng unit: grid_score (GridScorer, same as survey_validation_figure.py)
and field spacing (mean distance of the 6 nearest autocorrelogram peaks to the
centre, converted to cm) computed directly here -- the existing GridScorer has
no spacing method, and the real-data spacing estimator in
spatial_manifolds.tuning_scores.grid_score is written against pynapple
session objects, not a bare ratemap array, so it isn't a drop-in fit either.

Bimodality check: GMM with k=1 vs k=2 on log(spacing) for grid-scoring units,
picked by BIC -- k=1 winning means "no evidence for more than one scale",
consistent with how model selection was used elsewhere in this project.
"""
import sys, os, argparse
REPO_DIR='/Users/harryclark/Documents/spatial-manifolds/GRID-PATTERN-FORMATION'
sys.path.insert(0, REPO_DIR)
import numpy as np, torch
import matplotlib; matplotlib.use('Agg'); import matplotlib.pyplot as plt
from skimage.feature.peak import peak_local_max
from sklearn.mixture import GaussianMixture
from place_cells import PlaceCells
from trajectory_generator import TrajectoryGenerator
from model import RNN
from scores import GridScorer
from visualize import compute_ratemaps

p = argparse.ArgumentParser()
p.add_argument('--dir', required=True)
p.add_argument('--Ng', type=int, default=1024)
p.add_argument('--pc_rf', type=str, required=True,
               help='same --pc_rf value used at training time (single float or comma list)')
p.add_argument('--pc_seed', type=int, required=True)
p.add_argument('--res', type=int, default=40)
p.add_argument('--n_avg', type=int, default=50)
p.add_argument('--grid_thresh', type=float, default=0.3)
p.add_argument('--sequence_length', type=int, default=20)
p.add_argument('--box_width', type=float, default=2.2)
p.add_argument('--Np', type=int, default=512)
args = p.parse_args()

pc_rf_list = [float(x) for x in args.pc_rf.split(',')]
pc_rf = pc_rf_list[0] if len(pc_rf_list) == 1 else pc_rf_list

DEVICE = 'mps' if torch.backends.mps.is_available() else ('cuda' if torch.cuda.is_available() else 'cpu')


def field_spacing_cm(sac, res, box_width):
    """Mean distance of the 6 nearest autocorrelogram peaks to the centre,
    converted from ratemap-bin units to cm. NaN if fewer than 6 peaks found."""
    center = np.array(sac.shape) / 2.0
    peaks = peak_local_max(np.nan_to_num(sac), min_distance=3, exclude_border=True)
    if len(peaks) < 7:  # centre peak + >= 6 surrounding
        return np.nan
    d = np.linalg.norm(peaks - center, axis=1)
    order = np.argsort(d)[1:7]  # drop the centre peak itself
    if len(order) < 6:
        return np.nan
    mean_bins = d[order].mean()
    cm_per_bin = (box_width / res) * 100
    return mean_bins * cm_per_bin


class Options: pass
o = Options()
o.Np=args.Np; o.Ng=args.Ng; o.sequence_length=args.sequence_length; o.batch_size=200
o.learning_rate=1e-4; o.weight_decay=1e-4; o.place_cell_rf=pc_rf; o.surround_scale=2
o.pc_rf_seed=args.pc_seed
o.RNN_type='RNN'; o.activation='relu'; o.DoG=True; o.periodic=False
o.box_width=args.box_width; o.box_height=args.box_width; o.device=DEVICE

pc = PlaceCells(o)
pc.us = torch.tensor(np.load(os.path.join(args.dir, 'place_cells.npy')), dtype=torch.float32).to(DEVICE)

# PlaceCells.__init__ reconstructs sigma_group via its own RNG shuffle (seeded by
# --pc_seed), which only reproduces the ORIGINAL training-time assignment when the
# same pc_seed was used to train. Some runs (e.g. merge_and_finetune.py) build a
# deliberate, non-shuffled sigma_group and save it directly -- always prefer that
# saved file when present, rather than silently risk a mismatched reconstruction.
saved_sigma_group_path = os.path.join(args.dir, 'sigma_group.npy')
if len(pc_rf_list) > 1 and os.path.isfile(saved_sigma_group_path):
    saved_group = np.load(saved_sigma_group_path)
    if not np.array_equal(saved_group, pc.sigma_group):
        print('sigma_group.npy on disk differs from the freshly-reconstructed shuffle '
             '-- using the SAVED (training-time) assignment', flush=True)
    pc.sigma_group = saved_group
    widths = np.asarray(pc_rf_list, dtype=np.float32)
    pc.sigma = torch.tensor(widths[saved_group], dtype=torch.float32, device=DEVICE)

model = RNN(o, pc).to(DEVICE)
model.load_state_dict(torch.load(os.path.join(args.dir, 'ckpt.pth'), map_location=DEVICE)['model'])
model.eval()
tg = TrajectoryGenerator(o, pc)

print(f'place-cell width groups: {pc_rf_list}, counts={np.bincount(pc.sigma_group).tolist()}', flush=True)

RES = args.res
a, _, _, _ = compute_ratemaps(model, tg, o, res=RES, n_avg=args.n_avg, Ng=o.Ng)

scorer = GridScorer(RES, ((-1.1, 1.1), (-1.1, 1.1)), zip([0.2]*10, np.linspace(0.4, 1.0, 10).tolist()))

grid_scores = np.full(o.Ng, np.nan)
spacings = np.full(o.Ng, np.nan)
for i in range(o.Ng):
    res = scorer.get_scores(a[i])
    grid_scores[i] = res[0]
    spacings[i] = field_spacing_cm(res[4], RES, o.box_width)

with torch.no_grad():
    inp, pos, _ = tg.get_test_batch()
    pred = pc.get_nearest_cell_pos(model.predict(inp)).cpu()
err_cm = torch.sqrt(((pos.cpu() - pred) ** 2).sum(-1)).mean().item() * 100

is_grid = grid_scores > args.grid_thresh
n_grid = int(is_grid.sum())
print(f'\ndecode err = {err_cm:.1f} cm, {n_grid}/{o.Ng} units grid_score>{args.grid_thresh}', flush=True)

ng_group_path = os.path.join(args.dir, 'ng_group.npy')
ng_group = np.load(ng_group_path) if os.path.isfile(ng_group_path) else None
if ng_group is not None:
    print(f'encoder_mask was ON -- ng_group counts={np.bincount(ng_group).tolist()}', flush=True)
    for g in sorted(set(ng_group)):
        sel = is_grid & (ng_group == g) & ~np.isnan(spacings)
        sp_g = spacings[sel]
        if len(sp_g):
            print(f'  Ng group {g} (matched to pc width {pc_rf_list[g]}): '
                 f'{sel.sum()} grid cells, spacing median={np.median(sp_g):.1f}cm '
                 f'range=[{sp_g.min():.1f},{sp_g.max():.1f}]cm', flush=True)
        else:
            print(f'  Ng group {g} (matched to pc width {pc_rf_list[g]}): 0 grid cells', flush=True)

valid_spacing = spacings[is_grid & ~np.isnan(spacings)]
if len(valid_spacing) >= 10:
    log_sp = np.log(valid_spacing).reshape(-1, 1)
    bics = []
    for k in (1, 2, 3):
        gm = GaussianMixture(n_components=k, covariance_type='full', n_init=10, random_state=0)
        gm.fit(log_sp)
        bics.append(gm.bic(log_sp))
    best_k = int(np.argmin(bics)) + 1
    print(f'spacing (grid cells only, n={len(valid_spacing)}): '
         f'median={np.median(valid_spacing):.1f}cm range=[{valid_spacing.min():.1f},{valid_spacing.max():.1f}]cm', flush=True)
    print(f'GMM on log(spacing): BIC(k=1,2,3)={[round(b) for b in bics]} -> best_k={best_k}', flush=True)
    if best_k > 1:
        gm = GaussianMixture(n_components=best_k, covariance_type='full', n_init=10, random_state=0).fit(log_sp)
        means_cm = np.exp(gm.means_.ravel())
        print(f'{best_k}-component means: {sorted(np.round(means_cm, 1).tolist())} cm '
             f'-- MULTIPLE SCALES DETECTED', flush=True)
    else:
        print('single component wins -- NO evidence for multiple scales', flush=True)
else:
    print(f'only {len(valid_spacing)} grid cells with valid spacing -- too few to test bimodality', flush=True)

np.savez(os.path.join(args.dir, 'multiscale_scores.npz'),
        grid_scores=grid_scores, spacings=spacings, sigma_group=pc.sigma_group, err_cm=err_cm,
        ng_group=ng_group if ng_group is not None else np.array([]))

# ── figure: spacing histogram + top-scoring ratemaps ─────────────────────────
fig, axes = plt.subplots(2, 6, figsize=(14, 5))
axes[0, 0].remove(); axes[1, 0].remove()
axh = fig.add_subplot(2, 6, (1, 7))
if len(valid_spacing):
    axh.hist(valid_spacing, bins=20, color='#3171ae', alpha=0.85)
axh.set_xlabel('field spacing (cm)'); axh.set_ylabel(f'grid cells (score>{args.grid_thresh})')
axh.set_title(f'n_grid={n_grid}/{o.Ng}\ndecode={err_cm:.1f}cm', fontsize=9)
axh.spines[['top', 'right']].set_visible(False)

order = np.flip(np.argsort(np.nan_to_num(grid_scores, nan=-1)))
flat_axes = [axes[0, c] for c in range(1, 6)] + [axes[1, c] for c in range(1, 6)]
for ax, idx in zip(flat_axes, order[:10]):
    ax.imshow(a[idx].T, origin='lower', cmap='jet', interpolation='gaussian')
    sp = spacings[idx]
    ax.set_title(f'gs={grid_scores[idx]:.2f}\n{sp:.0f}cm' if not np.isnan(sp) else f'gs={grid_scores[idx]:.2f}',
                fontsize=7)
    ax.set_xticks([]); ax.set_yticks([])
fig.suptitle(f'Multiscale RNN validation -- {args.dir}', fontsize=10, fontweight='bold')
plt.tight_layout()
plt.savefig(os.path.join(args.dir, 'multiscale_validation.png'), dpi=140, bbox_inches='tight')
print('VALIDATION_DONE', flush=True)
