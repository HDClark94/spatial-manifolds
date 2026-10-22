"""Before/after spacing-drift figures for the two dominance-test merges:
net6 pair (large pre-merge yield gap, 64.7% vs 39.6%) and net7 pair
(matched pre-merge yield, 45.4% vs 38.2%). Reuses multiscale_scores.npz
saved by validate_multiscale.py for every source/merged checkpoint --
no recomputation needed.
"""
import os
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

BASE = '/Users/harryclark/Documents/spatial-manifolds/data/rnn_xgboost'
GRID_THRESH = 0.3


def load_spacings(subdir, group=None):
    d = np.load(os.path.join(BASE, subdir, 'multiscale_scores.npz'))
    is_grid = d['grid_scores'] > GRID_THRESH
    sp = d['spacings']
    sel = is_grid & ~np.isnan(sp)
    if group is not None and d['ng_group'].size:
        sel &= (d['ng_group'] == group)
    n_total = len(d['grid_scores']) if group is None else int((d['ng_group'] == group).sum())
    return sp[sel], n_total


def make_figure(pair_name, narrow_pre_dir, wide_pre_dir, merged_dir, title_suffix, out_path):
    narrow_pre, n_narrow_pre = load_spacings(narrow_pre_dir)
    wide_pre, n_wide_pre = load_spacings(wide_pre_dir)
    narrow_post, n_narrow_post = load_spacings(merged_dir, group=0)
    wide_post, n_wide_post = load_spacings(merged_dir, group=1)

    fig, axes = plt.subplots(1, 2, figsize=(10, 3.5), sharey=False)

    ax = axes[0]
    bins = np.linspace(60, 200, 29)
    ax.hist(narrow_pre, bins=bins, color='#3171ae', alpha=0.7, label=f'narrow (0.12), n={len(narrow_pre)}/{n_narrow_pre}')
    ax.hist(wide_pre, bins=bins, color='#c0392b', alpha=0.7, label=f'wide (0.1944), n={len(wide_pre)}/{n_wide_pre}')
    ax.axvline(np.median(narrow_pre), color='#3171ae', ls='--', lw=1.5)
    ax.axvline(np.median(wide_pre), color='#c0392b', ls='--', lw=1.5)
    ax.set_title(f'pre-merge (independent nets)\nmedians: {np.median(narrow_pre):.0f}cm / {np.median(wide_pre):.0f}cm', fontsize=9)
    ax.set_xlabel('field spacing (cm)')
    ax.set_ylabel('grid cells')
    ax.legend(fontsize=7, frameon=False)
    ax.spines[['top', 'right']].set_visible(False)

    ax = axes[1]
    ax.hist(narrow_post, bins=bins, color='#3171ae', alpha=0.7, label=f'narrow-origin, n={len(narrow_post)}/{n_narrow_post}')
    ax.hist(wide_post, bins=bins, color='#c0392b', alpha=0.7, label=f'wide-origin, n={len(wide_post)}/{n_wide_post}')
    ax.axvline(np.median(narrow_post), color='#3171ae', ls='--', lw=1.5)
    ax.axvline(np.median(wide_post), color='#c0392b', ls='--', lw=1.5)
    ax.set_title(f'post-merge (50k joint fine-tune)\nmedians: {np.median(narrow_post):.0f}cm / {np.median(wide_post):.0f}cm', fontsize=9)
    ax.set_xlabel('field spacing (cm)')
    ax.legend(fontsize=7, frameon=False)
    ax.spines[['top', 'right']].set_visible(False)

    fig.suptitle(f'{pair_name}: spacing before vs after merge ({title_suffix})', fontsize=10, fontweight='bold')
    plt.tight_layout()
    plt.savefig(out_path + '.png', dpi=150, bbox_inches='tight')
    plt.savefig(out_path + '.pdf', bbox_inches='tight')
    plt.close(fig)

    narrow_shift = np.median(narrow_post) - np.median(narrow_pre)
    wide_shift = np.median(wide_post) - np.median(wide_pre)
    print(f'{pair_name}: narrow shift {narrow_shift:+.1f}cm, wide shift {wide_shift:+.1f}cm '
          f'({out_path}.png)', flush=True)


make_figure(
    'net6 pair', 'survey_s6_Ng1024', 'merge_source_net6companion_wide_Ng1024',
    'merged_net6_dominance_Ng2048', 'yield 64.7% vs 39.6%',
    os.path.join(BASE, 'merged_net6_dominance_Ng2048', 'spacing_before_after_dominance'))

make_figure(
    'net7 pair', 'survey_s7_Ng1024', 'merge_source_net7companion_wide_Ng1024',
    'merged_net7_dominance_Ng2048', 'yield 45.4% vs 38.2%',
    os.path.join(BASE, 'merged_net7_dominance_Ng2048', 'spacing_before_after_dominance'))

print('ALL_FIGURES_DONE', flush=True)
