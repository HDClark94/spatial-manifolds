"""
XGBoost shank-reconstruction of MEC task anchoring.

For one session, for every ENTm ('MEC') cell as a target:
  * fit an xgboost history model per shank, using that shank's ENTm cells'
    time-binned activity as covariates (the target is excluded from its own shank),
  * convert the cross-validated predicted spikes into a trial x position rate map,
  * recompute anchoring labels on the predicted map (task_anchoring_catalogue criteria),
  * record pR2 overall and split by the target's anchored / non-anchored epochs.

Produces, per target, up to 4 predicted rate maps + anchoring label vectors (one per
shank that carries ENTm cells).  Cached as a pickle for the plotting notebook.

Usage:
  python run_shank_reconstruction.py --mouse 26 --day 19 --data_path <cache_dir>
  optional: --max_targets N   --targets 12,34   --history_length 1000   --n_cv 10
"""
import os
import argparse
import pickle
import warnings
warnings.filterwarnings('ignore')

import numpy as np
import pandas as pd
import pynapple as nap

from spatial_manifolds.mlencoding import MLencoding, poisson_pseudoR2
from spatial_manifolds.anaylsis_parameters import time_bs

import anchoring_common as ac
from anchoring_common import (bs, tl, bpt, compute_vr_tcs, compute_time_vars,
                              get_mec_cells, get_shank_pools, stratified_sample,
                              compute_anchoring_labels, predicted_spikes_to_tc, labels_to_time)


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--mouse', type=int, required=True)
    p.add_argument('--day', type=int, required=True)
    p.add_argument('--data_path', type=str,
                   default='/Users/harryclark/Documents/spatial-manifolds/data/xgboost_shank_anchoring')
    p.add_argument('--source_path', type=str, default=ac.SOURCE_PATH)
    p.add_argument('--history_length', type=int, default=1000, help='covariate history span in ms')
    p.add_argument('--n_filters', type=int, default=5, help='history filters over the span (Figure4 uses 5)')
    p.add_argument('--n_cov', type=int, default=16,
                   help='subsample each shank pool to this many covariate cells (Figure4 default 16); '
                        'pass a large number for all cells')
    p.add_argument('--n_cv', type=int, default=10)
    p.add_argument('--seed', type=int, default=0, help='rng seed for covariate subsampling')
    p.add_argument('--max_targets', type=int, default=None, help='cap number of target cells (debug)')
    p.add_argument('--targets', type=str, default=None, help='comma-separated target cluster_ids (debug)')
    args = p.parse_args()

    mouse, day = args.mouse, args.day
    os.makedirs(args.data_path, exist_ok=True)

    print(f'Loading M{mouse}D{day} ...')
    tcs, tcs_time, _, last_ephys_bin, beh, clusters = compute_vr_tcs(
        mouse, day, apply_zscore=False, apply_guassian_filter=False, source_path=args.source_path)
    tv = compute_time_vars(beh, clusters, source_path=args.source_path)

    n_trials_ref = last_ephys_bin // bpt
    valid_ids    = [int(c) for c in clusters.index if c in tcs_time]
    mec = get_mec_cells(mouse, day, valid_ids=valid_ids, source_path=args.source_path)
    if len(mec) == 0:
        print('No ENTm cells with tuning curves — nothing to do.'); return
    shank_of = dict(zip(mec['cluster_id'], mec['shank_id']))

    # covariate pool per shank = dominant region (ENTm or PAR) on that shank
    pools = get_shank_pools(mouse, day, valid_ids=valid_ids, source_path=args.source_path)
    shanks_present = sorted(pools.keys())
    print(f'  ENTm targets: {len(mec)} on shanks {sorted(mec["shank_id"].unique().tolist())} '
          f'({mec["shank_id"].value_counts().sort_index().to_dict()})')
    print('  covariate pool per shank (ENTm + PAR, sampled representatively):')
    for sh in shanks_present:
        pl = pools[sh]
        print(f'    shank {sh}: ENTm={pl["n_ent"]}, PAR={pl["n_par"]} '
              f'(total {pl["n_ent"] + pl["n_par"]})')

    # target list
    targets = list(mec['cluster_id'])
    if args.targets:
        want = set(int(x) for x in args.targets.split(','))
        targets = [t for t in targets if t in want]
    if args.max_targets:
        targets = targets[:args.max_targets]
    print(f'  targets: {len(targets)}')

    import random
    rng = random.Random(args.seed)
    # Figure4-aligned encoder: coarse history (few filters over 1000 ms), NOT one filter
    # per time bin — a per-bin history explodes the design matrix for large populations.
    xgb = MLencoding(tunemodel='xgboost', cov_history=True, spike_history=False,
                     window=time_bs, n_filters=args.n_filters, max_time=args.history_length)

    # behaviour for the hit-rate panel
    trial_type = np.array(beh['trials']['type'])[:n_trials_ref]
    trial_perf = np.array(beh['trials']['performance'])[:n_trials_ref]

    L = min(len(tv['dt_in_time']), len(tcs_time[targets[0]]))
    dt_in_time  = tv['dt_in_time'][:L]
    moving_mask = tv['moving_mask'][:L]
    trial_number_in_time = tv['trial_number_in_time'][:L]

    out = dict(mouse=mouse, day=day, n_trials_ref=n_trials_ref, bpt=bpt, tl=tl, bs=bs,
               last_ephys_bin=last_ephys_bin, shanks_present=shanks_present,
               trial_type=trial_type, trial_perf=trial_perf, shank_of=shank_of,
               config=dict(n_filters=args.n_filters, history_length=args.history_length,
                           n_cv=args.n_cv, n_cov=args.n_cov, mec_prefix=ac.MEC_PREFIX),
               targets={})

    for ti, target_id in enumerate(targets):
        t_shank = int(shank_of[target_id])
        obs_tc  = np.asarray(tcs[target_id])[:last_ephys_bin]
        obs_lab = compute_anchoring_labels(obs_tc)
        if obs_lab is None:
            print(f'  [{ti+1}/{len(targets)}] cell {target_id}: observed labels failed, skipping'); continue
        lab_in_time = labels_to_time(obs_lab, trial_number_in_time)

        y = np.asarray(tcs_time[target_id])[:L]
        tinfo = dict(shank=t_shank, observed_tc=obs_tc, observed_labels=obs_lab, shanks={})

        for sh in shanks_present:
            ent_ids = [int(c) for c in pools[sh]['ent_ids'] if int(c) != target_id and c in tcs_time]
            par_ids = [int(c) for c in pools[sh]['par_ids'] if int(c) != target_id and c in tcs_time]
            if len(ent_ids) + len(par_ids) == 0:
                continue
            cov_ids, n_ent_used, n_par_used = stratified_sample(ent_ids, par_ids, args.n_cov, rng)
            X = np.vstack([np.asarray(tcs_time[c])[:L] for c in cov_ids]).T

            try:
                Y_hat, pR2_cv = xgb.fit_cv(X, y, n_cv=args.n_cv, verbose=0, continuous_folds=True)
            except Exception as e:
                print(f'    shank {sh}: fit failed ({e})'); continue
            Y_hat = np.asarray(Y_hat)[:L]

            pred_tc  = predicted_spikes_to_tc(Y_hat, dt_in_time, moving_mask, tv['max_bound'], last_ephys_bin)
            pred_lab = compute_anchoring_labels(pred_tc)

            def _mode_pr2(mode):
                m = (lab_in_time == mode) & np.isfinite(y) & np.isfinite(Y_hat)
                if m.sum() < 5:
                    return np.nan
                return float(poisson_pseudoR2(y[m], Y_hat[m], ynull=np.nanmean(y[m])))

            tinfo['shanks'][int(sh)] = dict(
                pred_tc=pred_tc, pred_labels=pred_lab,
                n_cov_cells=len(cov_ids), cov_ids=cov_ids,
                n_ent_used=n_ent_used, n_par_used=n_par_used,
                is_same_shank=bool(sh == t_shank),
                pr2_overall=float(np.nanmean(pR2_cv)),
                pr2_anchored=_mode_pr2(1.0),
                pr2_nonanchored=_mode_pr2(0.0),
            )
            s = tinfo['shanks'][int(sh)]
            print(f'  [{ti+1}/{len(targets)}] cell {target_id} (sh{t_shank}) <- sh{sh} '
                  f'({n_ent_used}E/{n_par_used}P{"*" if s["is_same_shank"] else ""}): '
                  f'pR2={s["pr2_overall"]:.3f}  anch={s["pr2_anchored"]:.3f}  nonanch={s["pr2_nonanchored"]:.3f}')

        out['targets'][int(target_id)] = tinfo

    fpath = os.path.join(args.data_path, f'M{mouse}D{day}_shank_reconstruction.pkl')
    with open(fpath, 'wb') as f:
        pickle.dump(out, f)
    print(f'Saved {fpath}  ({len(out["targets"])} targets)')


if __name__ == '__main__':
    main()
