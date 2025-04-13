import os
from argparse import ArgumentParser
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import pynapple as nap
from einops import rearrange
from scipy.ndimage import gaussian_filter1d
from tensortools import Ensemble, plot_objective, plot_similarity
from tensortools.cpwarp import fit_shifted_cp

from spatial_manifolds.data.loading import load_session
from spatial_manifolds.TCA import plot_factors
from spatial_manifolds.util import get_bounds

if __name__ == "__main__":
    parser = ArgumentParser(
        description="Computing TCA for Nolan lab Ephys data for use with this package."
    )
    parser.add_argument("--session", required=True, type=Path)
    parser.add_argument(
        "--bin_size",
        default=2.0,
        type=float,
    )
    parser.add_argument(
        "--smooth_sigma",
        default=1,
        type=int,
    )
    parser.add_argument(
        "--num_factors",
        default=20,
        type=int,
    )
    args = parser.parse_args()

    # Setup output path
    output_path = (
        Path("output")
        / "TCA"
        / "/".join(args.session.parts[-4:])
        / f"{args.bin_size}_{args.smooth_sigma}"
    )
    os.makedirs(output_path, exist_ok=True)

    # Load session
    neurons, behaviour, theta_phase = load_session(Path("."), args.session, "vr", curate=True)
    starts, stops = get_bounds(behaviour["trial_number"])
    trial_types = behaviour["trial_type"][starts[1:-1]].values
    rewarded = np.array(
        [
            np.any(behaviour["stopped_in_rz"][start:end].values)
            for start, end in zip(starts[1:-1], starts[2:])
        ]
    )
    trials = nap.IntervalSet(
        start=behaviour.times()[starts[1:-1]], end=behaviour.times()[starts[2:]]
    )

    # Bin
    binned = np.stack(
        [
            pd.DataFrame(
                nap.compute_1d_tuning_curves(
                    neurons,
                    behaviour["P"],
                    ep=trial,
                    nb_bins=int(200.0 / args.bin_size),
                )
            )
            .interpolate(axis=0)
            .bfill()
            .ffill()
            .pipe(
                lambda df: (
                    gaussian_filter1d(df.to_numpy(), sigma=args.smooth_sigma, axis=0)
                    if args.smooth_sigma > 0
                    else df.to_numpy()
                )
            )
            for trial in trials
        ]
    )
    binned = rearrange(binned, "trials position_bins neurons -> neurons position_bins trials")
    binned = (binned - np.min(binned, axis=(1, 2), keepdims=True)) / np.ptp(
        binned, axis=(1, 2), keepdims=True
    )

    # Normal TCA
    ensemble = Ensemble(fit_method="ncp_hals", nonneg=True)
    ensemble.fit(binned, ranks=range(1, args.num_factors + 1), replicates=5)

    # Plot metrics
    fig, axes = plt.subplots(1, 2, figsize=(8, 2))
    plot_objective(
        ensemble, ax=axes[0]
    )  # plot reconstruction error as a function of num components.
    plot_similarity(ensemble, ax=axes[1])  # plot model similarity as a function of num components.
    plt.savefig(output_path / "metrics.pdf", dpi=300)
    plt.close()

    # Plot
    for color, color_type, labels in [
        (rewarded, "rewarded", {0: "M", 1: "H"}),
        (trial_types, "trial_type", {0: "B", 1: "NB", 2: "P"}),
    ]:
        for i in range(1, args.num_factors + 1):
            os.makedirs(output_path / str(i), exist_ok=True)
            plot_factors(
                ensemble.factors(i)[0].factors,
                behaviour,
                color,
                labels,
                title=f"loss = {ensemble.objectives(i)[0]:.2f}",
            )
            plt.savefig(
                output_path / str(i) / f"{color_type}.pdf",
                bbox_inches="tight",
                dpi=300,
            )
            plt.close()

    # Shifted TCA
    losses = []
    for num_factors in range(5, args.num_factors):
        model = fit_shifted_cp(
            rearrange(binned, "neurons position_bins trials -> trials neurons position_bins"),
            num_factors,
            boundary="wrap",
            n_restarts=2,
            max_shift_axis0=0.2,
            max_shift_axis1=None,
            u_nonneg=True,
            v_nonneg=True,
        )
        factors = [model.factors[1], model.factors[2], model.factors[0]]
        factors = [factor.swapaxes(0, 1) for factor in factors]
        losses.append(model.loss_hist[-1])

        # Plot
        for color, color_type, labels in [
            (rewarded, "rewarded", {0: "M", 1: "H"}),
            (trial_types, "trial_type", {0: "B", 1: "NB", 2: "P"}),
        ]:
            os.makedirs(output_path / "shifted" / str(num_factors), exist_ok=True)
            plot_factors(
                factors, behaviour, color, labels, title=f"loss = {model.loss_hist[-1]:.2f}"
            )
            plt.savefig(
                output_path / "shifted" / str(num_factors) / f"{color_type}.pdf",
                bbox_inches="tight",
                dpi=300,
            )
            plt.close()

    plt.plot(losses)
    plt.scatter(losses)
    plt.xlabel("model rank")
    plt.ylabel("objective")
    plt.savefig(output_path / "shifted" / "losses.pdf", dpi=300)
