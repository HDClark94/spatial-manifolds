"""Decode-error-vs-training-step curves: pretraining of the two source nets
(narrow, wide) followed by the merged network's joint fine-tuning, on a single
timeline per pair. All values come from the 'history' dict saved in each
run's ckpt.pth -- no log parsing needed.
"""
import os
import torch
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

BASE = '/Users/harryclark/Documents/spatial-manifolds/data/rnn_xgboost'


def err_curve(subdir, smooth=200):
    ck = torch.load(os.path.join(BASE, subdir, 'ckpt.pth'), map_location='cpu')
    err = np.array(ck['history']['err']) * 100  # m -> cm
    if smooth > 1:
        kernel = np.ones(smooth) / smooth
        err = np.convolve(err, kernel, mode='valid')
    return err


def make_figure(pair_name, narrow_dir, wide_dir, merged_dir, yield_label, out_path):
    narrow_err = err_curve(narrow_dir)
    wide_err = err_curve(wide_dir)
    merged_err = err_curve(merged_dir)
    pretrain_steps = 50000
    x_narrow = np.arange(len(narrow_err))
    x_wide = np.arange(len(wide_err))
    x_merged = pretrain_steps + np.arange(len(merged_err))

    fig, ax = plt.subplots(1, 1, figsize=(9, 3.5))
    ax.plot(x_narrow, narrow_err, color='#3171ae', lw=1.2, label='narrow (0.12) pretraining')
    ax.plot(x_wide, wide_err, color='#c0392b', lw=1.2, label='wide (0.1944) pretraining')
    ax.plot(x_merged, merged_err, color='#4a4a4a', lw=1.4, label='merged joint fine-tune')
    ax.axvline(pretrain_steps, color='k', ls=':', lw=1)
    ax.text(pretrain_steps, ax.get_ylim()[1] * 0.9, ' merge point', fontsize=8, va='top')
    ax.set_xlabel('training step')
    ax.set_ylabel('decode error (cm)')
    ax.set_title(f'{pair_name}: decode error, independent pretraining -> merged fine-tune ({yield_label})',
                 fontsize=10, fontweight='bold')
    ax.legend(fontsize=8, frameon=False)
    ax.spines[['top', 'right']].set_visible(False)
    ax.set_ylim(bottom=0)
    plt.tight_layout()
    plt.savefig(out_path + '.png', dpi=150, bbox_inches='tight')
    plt.savefig(out_path + '.pdf', bbox_inches='tight')
    plt.close(fig)
    print(f'{pair_name}: saved {out_path}.png', flush=True)


make_figure(
    'net6 pair', 'survey_s6_Ng1024', 'merge_source_net6companion_wide_Ng1024',
    'merged_net6_dominance_Ng2048', 'yield 64.7% vs 39.6%',
    os.path.join(BASE, 'merged_net6_dominance_Ng2048', 'decode_error_training_curve'))

make_figure(
    'net7 pair', 'survey_s7_Ng1024', 'merge_source_net7companion_wide_Ng1024',
    'merged_net7_dominance_Ng2048', 'yield 45.4% vs 38.2%',
    os.path.join(BASE, 'merged_net7_dominance_Ng2048', 'decode_error_training_curve'))

print('ALL_CURVES_DONE', flush=True)
