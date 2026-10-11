"""Anchored / non-anchored trial classification.

Single source of truth for the classifier described in the Methods
(`scripts/figures/AnchorDynamics2026/PAPER.md`, "Classification of anchored and
non-anchored trials"). Previously this lived as duplicated inline definitions in
`trial_cluster_anchoring.ipynb` and `anchoring_classifier_validation.ipynb`,
which had already drifted apart on NaN handling.

The classifier asks whether a trial's spatial firing profile resembles the
cell's other trials -- self-consistency, not periodicity. It therefore applies
unchanged to grid cells, non-grid spatial cells and ramping cells, which is why
it replaced the spectral classifier (see Methods for that rationale). The
spectral classifier is not used anywhere in the paper.

It is also independent of recording modality. Trials are compared by Pearson
correlation of their spatial profiles, which is invariant to scale and offset,
so the units of the activity measure do not matter: binned spike counts,
calcium fluorescence and simulated network activations all work with the same
parameters. Verified explicitly -- scaling by 1e-3 or 1e3, or adding a constant
offset, returns bit-identical labels. In this project the same code is applied
to RNN unit activations (with ground truth), to MEC spike trains, and to
two-photon calcium imaging of hippocampus. The one scale-dependent parameter is
`MIN_ACTIVITY`, the silent-trial guard; set it relative to the units in use for
signals on a very different scale, or that can go negative.

Validated against recurrent-network ground truth in
`anchoring_classifier_validation.ipynb`: median accuracy 0.990 across 1024
simulated units, sensitivity 0.993, specificity 0.884.

    from spatial_manifolds.anchoring import trial_cluster_labels
    labels, mean_corr, smoothed, silhouette = trial_cluster_labels(rate_maps)
"""
import hashlib
import os
import numpy as np
from scipy.ndimage import gaussian_filter, median_filter
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score

# Defaults validated against RNN ground truth. MEDFILT_SIZE sets the temporal
# resolution floor: a size-5 median filter erases runs shorter than 3 trials, so
# 3 consecutive trials is the shortest detectable state change (recall 0.02 at a
# 1-trial epoch, 0.07 at 2, 0.64 at 3, plateauing ~0.81 from 5).
SMOOTH_SIGMA = 1.5
MEDFILT_SIZE = 5
N_NULL_SHIFTS = 200      # circular shifts for the per-cell gate null
NULL_PCT = 95            # percentile of that null used as the threshold
MIN_ACTIVITY = 1e-6
MIN_VALID_TRIALS = 6


def smooth_nanaware(row, sigma=SMOOTH_SIGMA):
    """Gaussian smoothing that interpolates over unvisited position bins.

    Unvisited bins are missing data, not zero firing. Filling them with zero
    (``np.nan_to_num``) asserts a measurement that was never made, and because
    the unvisited pattern is set by the animal's occupancy it is shared across
    simultaneously recorded cells -- so zero-filling can make trials correlate
    through their trajectory rather than through their firing, which is exactly
    the quantity this classifier clusters on. Normalising a smoothed map by a
    smoothed validity mask spreads real data into the gaps instead.
    """
    v = np.nan_to_num(np.asarray(row, float), nan=0.0)
    w = np.isfinite(np.asarray(row, float)).astype(float)
    vs = gaussian_filter(v, sigma=sigma, mode='nearest')
    ws = gaussian_filter(w, sigma=sigma, mode='nearest')
    with np.errstate(invalid='ignore', divide='ignore'):
        out = vs / ws
    out[ws < 1e-8] = 0.0
    return out


LABEL_DIR = ('/Users/harryclark/Documents/spatial-manifolds/data/'
             'population_state/labels')


def load_session_labels(mouse, day, label_dir=LABEL_DIR):
    """The precomputed anchoring labels for one session, or None if absent.

    Written by scripts/figures/AnchorDynamics2026/build_anchoring_labels.py.
    Figures read this rather than reclassifying, so every figure in the paper is
    built from one set of labels, and a change to the classifier invalidates
    them visibly (rebuild the directory) instead of silently.

    Returns a dict with labels (n_cells, n_trials), cluster_id, trial,
    frac_anch, pc1, pc1_load, pc1_var, null_thr and gate.
    """
    p = f'{label_dir}/M{mouse}D{day}.npz'
    if not os.path.exists(p):
        return None
    z = np.load(p, allow_pickle=True)
    return {k: z[k] for k in z.files}


NULL_CACHE_PATH = ('/Users/harryclark/Documents/spatial-manifolds/data/'
                   'population_state/null_thresholds.npz')
_NULL_CACHE = None          # lazily loaded {key: threshold}
_NULL_DIRTY = set()         # keys computed this process and not yet persisted


def _null_key(smoothed_valid, n_shift, pct, context=None):
    """Content hash of the profiles being shuffled, plus the null's parameters.

    Keying on the data rather than on (mouse, day, cluster) means no call site
    has to change to benefit from the cache, and a threshold cannot be reused
    for a cell whose rate maps have been recomputed. Rounding first keeps the
    key stable against float noise from a rebuild.
    """
    h = hashlib.sha1(np.ascontiguousarray(
        np.round(smoothed_valid, 6), dtype=np.float64).tobytes())
    h.update(f'|{n_shift}|{pct}|{SMOOTH_SIGMA}'.encode())
    if context is not None:
        # the threshold calibrates a within-context statistic, so a pooled
        # threshold must never be served for a context-aware call
        h.update(b'|ctx|')
        h.update(np.asarray(context).astype(str).tobytes())
    return h.hexdigest()


def load_null_cache(path=NULL_CACHE_PATH):
    """Load the shared null-threshold cache, or start an empty one."""
    global _NULL_CACHE
    if _NULL_CACHE is None:
        _NULL_CACHE = {}
        if os.path.exists(path):
            z = np.load(path, allow_pickle=True)
            _NULL_CACHE = dict(zip(z['keys'].tolist(), z['thr'].tolist()))
    return _NULL_CACHE


def save_null_cache(path=NULL_CACHE_PATH):
    """Persist thresholds computed this process, merging with what is on disk.

    Merge rather than overwrite: several workers may be populating the cache at
    once, and this project has already lost a full results table once to a
    single-session run overwriting it.
    """
    cache = load_null_cache(path)
    if os.path.exists(path):
        z = np.load(path, allow_pickle=True)
        merged = dict(zip(z['keys'].tolist(), z['thr'].tolist()))
        merged.update(cache)
    else:
        merged = dict(cache)
    os.makedirs(os.path.dirname(path), exist_ok=True)
    # np.savez_compressed appends .npz unless the name already ends in it, so
    # the temp name has to carry the extension or the rename target is wrong
    tmp = path.replace('.npz', f'.{os.getpid()}.tmp.npz')
    np.savez_compressed(tmp, keys=np.array(list(merged.keys())),
                        thr=np.array(list(merged.values()), dtype=float))
    os.replace(tmp, path)
    _NULL_DIRTY.clear()
    return len(merged)


def _mean_corr(C, context=None):
    """Each trial's mean correlation with the OTHER trials of its own context.

    With one context this is the mean over every other trial, which is what the
    single-context datasets have always used -- so the context-aware form is a
    strict generalisation, not a different method.

    Averaging across contexts would be wrong wherever a cell remaps between
    them: its within-context correlations are high and its across-context ones
    near zero, so pooling halves every trial's score equally, pushes the whole
    distribution below the cell's null, and labels a cell that is perfectly
    anchored in both contexts as anchored in neither.
    """
    if context is None:
        return (C.sum(axis=1) - 1) / (C.shape[0] - 1)
    c = np.asarray(context)
    same = c[:, None] == c[None, :]
    np.fill_diagonal(same, False)
    n = same.sum(axis=1)
    out = np.full(C.shape[0], np.nan)
    ok = n > 0
    out[ok] = (C * same).sum(axis=1)[ok] / n[ok]
    return out


def null_mean_corr(smoothed_valid, rng=None, n_shift=N_NULL_SHIFTS, pct=NULL_PCT,
                   use_cache=True, context=None):
    """What mean trial-trial correlation this cell reaches by chance.

    Each trial's smoothed profile is circularly shifted by an independent random
    offset, preserving every trial's shape, rate and spatial extent while
    destroying only the alignment BETWEEN trials -- the one thing the classifier
    clusters on. Returns the `pct` percentile of the resulting mean-correlation
    values, i.e. this cell's own "what chance looks like".

    A per-cell null is required rather than a fixed constant: calibrated on the
    RNN validation set the separating value is 0.58, but real entorhinal cells
    have their ANCHORED cluster centre at 0.08-0.10, so a shared constant would
    label every recorded trial non-anchored.
    """
    key = _null_key(smoothed_valid, n_shift, pct, context) if use_cache else None
    if key is not None:
        cache = load_null_cache()
        if key in cache:
            return float(cache[key])
    rng = rng or np.random.default_rng(0)
    nb = smoothed_valid.shape[1]
    vals = []
    for _ in range(n_shift):
        S = np.array([np.roll(r, int(rng.integers(nb))) for r in smoothed_valid])
        vals.append(_mean_corr(np.corrcoef(S), context))
    thr = float(np.nanpercentile(np.concatenate(vals), pct))
    if key is not None:
        load_null_cache()[key] = thr
        _NULL_DIRTY.add(key)
    return thr


def trial_cluster_labels(raw_trials, min_activity=MIN_ACTIVITY,
                         min_valid_trials=MIN_VALID_TRIALS,
                         smooth_sigma=SMOOTH_SIGMA, medfilt_size=MEDFILT_SIZE,
                         nan_aware=True, gate=True, rng=None, null_thr=None,
                         context=None):
    """Classify each trial of one cell as anchored (1) or non-anchored (0).

    Parameters
    ----------
    raw_trials : (n_trials, n_bins) trial-by-position firing rate map. NaN marks
        unvisited bins.
    nan_aware : if True (default) unvisited bins are interpolated over; if False
        they are zero-filled, reproducing the original implementation. On
        simulated ground truth both give median accuracy 0.990, but the
        NaN-aware treatment is the correct one and may matter more on real
        recordings with less uniform coverage.

    Returns
    -------
    labels : (n_trials,) 1.0 anchored, 0.0 non-anchored, NaN if unclassifiable.
    mean_corr : (n_trials,) each trial's mean correlation with all others.
    smoothed : (n_trials, n_bins) the smoothed profiles actually clustered.
    silhouette : separation of the 2-means partition, or NaN.

    Notes
    -----
    The median filter is applied to the LABEL SEQUENCE, not to the firing data.
    It removes isolated false-positive anchored labels inside non-anchored runs,
    which arise because a periodic cell can align in phase with other
    non-anchored trials by chance. It raises specificity from 0.808 to 0.884 at
    negligible cost to sensitivity, but imposes the 3-trial resolution floor
    noted above. Pass ``medfilt_size=0`` to disable it where brief anchoring
    events matter, accepting the lower specificity.
    """
    raw_trials = np.asarray(raw_trials, float)
    n_trials = raw_trials.shape[0]

    if nan_aware:
        smoothed = np.array([smooth_nanaware(raw_trials[t], smooth_sigma)
                             for t in range(n_trials)])
    else:
        smoothed = np.array([gaussian_filter(np.nan_to_num(raw_trials[t]),
                                             sigma=smooth_sigma)
                             for t in range(n_trials)])

    activity = smoothed.sum(axis=1)
    valid = activity > min_activity
    labels = np.full(n_trials, np.nan)
    mean_corr_full = np.full(n_trials, np.nan)
    sil = np.nan
    if valid.sum() < min_valid_trials:
        return labels, mean_corr_full, smoothed, sil

    C = np.corrcoef(smoothed[valid])
    ctx_valid = None if context is None else np.asarray(context)[valid]
    mean_corr = _mean_corr(C, ctx_valid)
    if not np.isfinite(mean_corr).any():
        return labels, mean_corr_full, smoothed, sil
    keep_mc = np.isfinite(mean_corr)
    km = KMeans(n_clusters=2, n_init=10, random_state=0).fit(
        mean_corr[keep_mc].reshape(-1, 1))
    anchored_cluster = int(np.argmax(km.cluster_centers_.ravel()))
    if len(np.unique(km.labels_)) > 1:
        sil = silhouette_score(mean_corr[keep_mc].reshape(-1, 1), km.labels_)
    lab_valid = np.zeros(valid.sum())
    lab_valid[keep_mc] = (km.labels_ == anchored_cluster).astype(float)
    labels[valid] = lab_valid
    labels[~valid] = 0.0

    # ABSOLUTE GATE. 2-means always returns two clusters, so a cell with no
    # anchoring at all is still split roughly in half: on surrogates with a field
    # at a random position every trial, 52% of trials came back anchored, and the
    # median filter below then launders the salt-and-pepper into runs that look
    # like real state structure. The per-cell silhouette does not detect this --
    # it is HIGHER for a never-anchored cell (0.666) than a perfectly anchored
    # one (0.620) -- because it measures how cleanly the split was made, not
    # whether splitting was warranted.
    #
    # So compare both cluster centres against what this cell reaches by chance.
    # Both below: nothing here resembles an anchored trial, label them all
    # non-anchored. Both above: every trial is anchored, which 2-means otherwise
    # caps at 0.84. Straddling: the split is real, keep it.
    #
    # On blocky surrogates this takes the mean error in reported anchored
    # fraction from 0.191 to 0.045 and per-trial accuracy from 0.808 to 0.947,
    # and it leaves RNN ground-truth accuracy unchanged at 0.990. A residual
    # over-report of about +0.05 remains in the mid-range; the measure is
    # monotonic in the true fraction and exact at the extremes, so the absolute
    # value should not be read as "this fraction of trials was anchored".
    if gate:
        thr = (null_mean_corr(smoothed[valid], rng, context=ctx_valid)
               if null_thr is None else null_thr)
        lo, hi = np.sort(km.cluster_centers_.ravel())
        if hi < thr:
            labels[:] = 0.0
        elif lo >= thr:
            labels[valid] = 1.0
    if medfilt_size:
        labels = median_filter(labels, size=medfilt_size, mode='nearest')
    mean_corr_full[valid] = mean_corr
    return labels, mean_corr_full, smoothed, sil


# anchored / non-anchored colours used throughout AnchorDynamics2026
MIN_EPOCH = 3            # trials; shorter runs of one sign are not epochs


def major_transitions(pc1, min_epoch=MIN_EPOCH):
    """Trials on which the population axis changes state.

    A transition is a sign change of PC1 -- that is the state variable the
    figures plot, so its zero crossing is the boundary, and no threshold or
    filter width has to be chosen. The one parameter is a floor on how short
    an epoch may be: runs of fewer than `min_epoch` trials are absorbed into
    the preceding epoch, because a single trial on the far side of zero is a
    crossing of the axis rather than a state the population entered. Without
    it M27 D23 gives 22 transitions over 393 trials; with it, 12.

    Returns the trial indices at which the sign changes, as a list.
    """
    sg = np.sign(np.nan_to_num(np.asarray(pc1, float)))
    sg[sg == 0] = 1.0
    if min_epoch > 1:
        changed = True
        while changed:
            changed = False
            edges = np.r_[0, np.where(np.diff(sg) != 0)[0] + 1, len(sg)]
            for a, b in zip(edges[:-1], edges[1:]):
                if b - a < min_epoch:
                    sg[a:b] = sg[a - 1] if a > 0 else sg[b]
                    changed = True
                    break
    return list(np.where(np.diff(sg) != 0)[0] + 1)


ANCH_COLOR = '#9d5391'
NONANCH_COLOR = '#5f9ea0'


def anchoring_cmap():
    """(cmap, norm) for 0/1 anchoring label rasters."""
    from matplotlib.colors import ListedColormap, BoundaryNorm
    cmap = ListedColormap([NONANCH_COLOR, ANCH_COLOR])
    return cmap, BoundaryNorm([-0.5, 0.5, 1.5], cmap.N)


# ── open-field variant ──────────────────────────────────────────────────────
# The trial-based classifier needs repeated traversals of the same track. In an
# open field there are no trials, and any time window samples a different,
# partial subset of the arena -- so a window-against-window correlation
# confounds "did the code drift" with "did the animal go somewhere else". The
# fix is to score each window against a WHOLE-SESSION reference map, which has
# full coverage, restricting the comparison to bins the window actually visited.
# This is also the true analogue of the trial version, where each trial is
# scored by its mean correlation with all others, i.e. with the consensus.
#
# Validated against RNN ground truth (a 2D cue-loss manipulation) in
# `open_field_anchoring.ipynb`: perfect accuracy for 30-90 s windows, and
# detection down to ~17 cm of drift in a 2.2 m arena. Accuracy holds for
# non-grid units, so the generality of the trial-based classifier survives.

def occupancy_ratemap(activity, pos, n_bins, box_width, min_occupancy=2):
    """2-D mean activity per spatial bin; NaN where occupancy is too low.

    NaN means "not sampled", and is propagated rather than zero-filled for the
    same reason as in the trial-based classifier.
    """
    hw = box_width / 2.0
    ix = np.clip(((pos[:, 0] + hw) / box_width * n_bins).astype(int), 0, n_bins - 1)
    iy = np.clip(((pos[:, 1] + hw) / box_width * n_bins).astype(int), 0, n_bins - 1)
    flat = ix * n_bins + iy
    cnt = np.bincount(flat, minlength=n_bins ** 2).astype(float)
    tot = np.bincount(flat, weights=np.asarray(activity, float), minlength=n_bins ** 2)
    out = np.full(n_bins ** 2, np.nan)
    ok = cnt >= min_occupancy
    out[ok] = tot[ok] / cnt[ok]
    return out.reshape(n_bins, n_bins)


def open_field_anchoring(activity, pos, window, n_bins=12, box_width=1.0,
                         min_occupancy=2, min_bins=30, medfilt_size=MEDFILT_SIZE):
    """Anchored / non-anchored classification for an open-field session.

    Parameters
    ----------
    activity : (n_samples,) firing rate or unit activity, sampled with `pos`.
    pos : (n_samples, 2) position, centred on the arena, same units as
        `box_width`.
    window : window length in SAMPLES. 30-90 s gave perfect accuracy against
        simulated ground truth; below ~30 s spatial coverage becomes the
        limiting factor, above ~90 s state changes are smeared.
    min_bins : a window is scored only if it shares at least this many visited
        bins with the reference; sparse windows would otherwise read as
        non-anchored purely from noise.

    Returns
    -------
    labels : (n_windows,) 1.0 anchored, 0.0 non-anchored, NaN if unscorable.
    scores : (n_windows,) correlation with the session reference map.
    starts : (n_windows,) sample index at which each window begins.
    """
    activity = np.asarray(activity, float)
    pos = np.asarray(pos, float)
    n = len(activity)
    starts = np.arange(0, n - window + 1, window)
    ref = occupancy_ratemap(activity, pos, n_bins, box_width, min_occupancy).ravel()

    scores = np.full(len(starts), np.nan)
    for i, s in enumerate(starts):
        sl = slice(s, s + window)
        w = occupancy_ratemap(activity[sl], pos[sl], n_bins, box_width, min_occupancy).ravel()
        m = np.isfinite(w) & np.isfinite(ref)
        if m.sum() < min_bins or w[m].std() < 1e-9 or ref[m].std() < 1e-9:
            continue
        scores[i] = np.corrcoef(w[m], ref[m])[0, 1]

    labels = np.full(len(starts), np.nan)
    good = np.isfinite(scores)
    if good.sum() >= MIN_VALID_TRIALS:
        v = np.where(good, scores, np.nanmin(scores)).reshape(-1, 1)
        km = KMeans(n_clusters=2, n_init=10, random_state=0).fit(v)
        labels = (km.labels_ == int(np.argmax(km.cluster_centers_.ravel()))).astype(float)
        if medfilt_size:
            labels = median_filter(labels, size=medfilt_size, mode='nearest')
    return labels, scores, starts


def labels_to_samples(labels, starts, window, n_samples):
    """Expand per-window labels to a per-sample array (NaN outside any window)."""
    out = np.full(n_samples, np.nan)
    for lab, s in zip(labels, starts):
        out[s:s + window] = lab
    return out
