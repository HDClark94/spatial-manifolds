from argparse import ArgumentParser
from pathlib import Path

import matplotlib.gridspec as gridspec
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import pynapple as nap
from matplotlib.ticker import FixedLocator, MaxNLocator
from scipy.ndimage import gaussian_filter
from scipy.stats import sem

from spatial_manifolds.data.loading import load_session
from spatial_manifolds.tuning_curves import TuningCurveTypes
from spatial_manifolds.tuning_scores.grid_score import correlate2d_ignore_nan
from spatial_manifolds.util import get_speed_limit


def get_trials(behaviour, task):
    if task == "vr":
        times = behaviour.times()
        starts = pd.Series(behaviour["trial_number"].values).drop_duplicates(keep="first").index
        return nap.IntervalSet(start=times[starts[1:-1]], end=times[starts[2:]])
    elif "of" in task:
        splits = np.array_split(behaviour[len(behaviour) % 10 :], 10)
        return nap.IntervalSet(
            start=[split.time_support.start for split in splits],
            end=[split.time_support.end for split in splits],
        )


def plot_vr(ax1, ax2, neuron_name, tcs):
    ax1.plot(tcs.tcs_mean[neuron_name], linewidth=0.5)
    ax1.axvspan(0, 30.0, color="gray", alpha=0.5)
    ax1.axvspan(90.0, 110.0, color="green", alpha=0.5)
    ax1.axvspan(170, 200.0, color="gray", alpha=0.5)
    ax1.set_xlim(0.0, 200.0)
    ax1.axis("off")

    ax2.imshow(
        tcs.tcs_trial[:, :, np.where(tcs.neuron_index == neuron_name)[0][0]],
        aspect="auto",
        cmap="viridis",
    )
    ax2.axis("off")


def plot_of(ax1, ax2, neuron_name, tcs):
    tc = tcs.tcs_mean[0][neuron_name]
    mask = np.isnan(tc)
    tc = gaussian_filter(np.nan_to_num(tc).astype(np.float64), sigma=1.5)
    ax1.imshow(tc, aspect="equal", cmap="viridis")
    ax1.axis("off")

    tc[mask] = np.nan
    autocorr = correlate2d_ignore_nan(tc)
    ax2.imshow(autocorr, aspect="equal", cmap="viridis")
    ax2.axis("off")


def plot_neuron(axs, neuron_name, tcs, label_top=True):
    position_tcs_of1 = tcs["of1"]["position"]
    neuron_index_of1 = np.where(position_tcs_of1.neuron_index == neuron_name)[0][0]
    label_of1 = (
        f"ID:{neuron_name}"
        f"\n{position_tcs_of1.mean_rates[neuron_index_of1]:.2f}Hz"
        f"\nGS:{position_tcs_of1.scores['grid_score'].values[neuron_index_of1]:.2f}"
    )
    position_tcs_of2 = tcs["of2"]["position"]
    neuron_index_of2 = np.where(position_tcs_of2.neuron_index == neuron_name)[0][0]
    label_of2 = (
        f"\n{position_tcs_of2.mean_rates[neuron_index_of2]:.2f}Hz"
        f"\nGS:{position_tcs_of2.scores['grid_score'].values[neuron_index_of2]:.2f}"
    )
    position_tcs_vr = tcs["vr"]["position"]
    neuron_index_vr = np.where(position_tcs_vr.neuron_index == neuron_name)[0][0]
    label_vr = (
        f"\n{position_tcs_vr.mean_rates[neuron_index_vr]:.2f}Hz"
        f"\nSI:{position_tcs_vr.scores['spatial_information'].values[neuron_index_vr]:.2f}"
    )
    if label_top:
        ax_of1, ax_of1_corr, ax_of2, ax_of2_corr, ax_vr, ax_vr_trial = axs
    else:
        (
            ax_of1_label,
            ax_of1,
            ax_of1_corr,
            ax_of2_label,
            ax_of2,
            ax_of2_corr,
            ax_vr_label,
            ax_vr,
            ax_vr_trial,
        ) = axs
        ax_of1_label.text(0.5, 0.5, label_of1, fontsize=4, ha="center", va="center")
        ax_of1_label.axis("off")
        ax_of2_label.text(0.5, 0.5, label_of2, fontsize=4, ha="center", va="center")
        ax_of2_label.axis("off")
        ax_vr_label.text(0.5, 0.5, label_vr, fontsize=4, ha="center", va="center")
        ax_vr_label.axis("off")

    plot_of(ax_of1, ax_of1_corr, neuron_name, position_tcs_of1)
    if label_top:
        ax_of1.set_title(label_of1, fontsize=4)

    plot_of(ax_of2, ax_of2_corr, neuron_name, position_tcs_of2)
    if label_top:
        ax_of2.set_title(label_of2, fontsize=4)

    plot_vr(ax_vr, ax_vr_trial, neuron_name, position_tcs_vr)
    if label_top:
        ax_vr.set_title(label_vr, fontsize=4)


if __name__ == "__main__":
    parser = ArgumentParser("Computing cross correlation for Nolan lab ephys data.")
    parser.add_argument(
        "--storage",
        type=Path,
        default=".",
        help="Path to the storage.",
    )
    parser.add_argument(
        "--session",
        required=True,
        type=Path,
        help="Path to session folder from storage/data/",
    )
    parser.add_argument(
        "--bin_size_sec",
        type=float,
        default=0.001,
    )
    parser.add_argument(
        "--window_size_sec",
        type=float,
        default=0.01,
    )
    args = parser.parse_args()

    # Load data
    data = {task: load_session(args.storage, args.session, task) for task in ["of1", "of2", "vr"]}

    # Trials
    data = {
        task: (
            task_neurons,
            get_speed_limit(task_derivatives["behaviour"], task),
            get_trials(task_derivatives["behaviour"], task),
        )
        for task, (task_neurons, task_derivatives) in data.items()
    }

    # Setup output directory
    output_path = args.storage / "output" / "crosscorr" / "/".join(args.session.parts[1:])
    output_path.mkdir(parents=True, exist_ok=True)

    # Get classses
    tcs = TuningCurveTypes.load(args.storage / args.session)
    classes = TuningCurveTypes.get_classes(tcs)

    # Get grid cells
    grid = [
        neuron_name
        for neuron_name, neuron_classes in classes.items()
        if "GS_of1" in neuron_classes and "GS_of2" in neuron_classes
    ]
    assert len(grid) > 0

    # Get ramp cells
    ramp = [
        neuron_name
        for neuron_name, neuron_classes in classes.items()
        if "RS_vr" in neuron_classes
        and "GS_of1" not in neuron_classes
        and "GS_of2" not in neuron_classes
    ]
    assert len(ramp) > 0
    ramp_classes = np.array(
        [
            tcs["vr"]["position"]
            .scores["ramp_score"]
            .classes[np.where(tcs["vr"]["position"].neuron_index == neuron_name)[0][0]]
            for neuron_name in ramp
        ]
    )

    # Compute cross corr
    cross_corr = {
        task: nap.compute_crosscorrelogram(
            (neurons[ramp + grid], neurons[ramp + grid]),
            binsize=args.bin_size_sec,
            windowsize=args.window_size_sec,
            ep=speed_limit,
        )
        for task, (neurons, speed_limit, trials) in data.items()
    }

    # Plot average per ramp class
    unique = np.unique(ramp_classes)
    fig, axs = plt.subplots(3, len(unique), figsize=(12, 12), layout="constrained")
    for task, axs_task in zip(["of1", "of2", "vr"], axs):
       axs_task[0].set_ylabel(task.upper(), rotation=0, labelpad=20)
       for ramp_class, ax in zip(unique, axs_task):
           stacked = np.stack(
               [
                   cross_corr[task][(grid_cell, ramp_cell)]
                   for grid_cell in grid
                   for ramp_cell_idx, ramp_cell in enumerate(ramp)
                   if ramp_classes[ramp_cell_idx] == ramp_class
               ],
               axis=-1,
           )
           avg = np.nanmean(stacked, axis=-1)
           se = sem(stacked, axis=-1)
           ax.plot(cross_corr[task].index, avg)
           ax.fill_between(cross_corr[task].index, avg - se, avg + se, alpha=0.5)
           ax.axvline(
               cross_corr[task].index[np.argmax(avg)],
               color="red",
               linestyle="--",
               linewidth=1,
           )
           if task == "of1":
               ax.set_title(
                   f"{ramp_class}\n(N={len(ramp_classes[ramp_classes == ramp_class])})",
                   fontweight="bold",
               )
    axs[-1, 0].set_xlabel("time (s)")
    axs[-1, 1].set_xlabel("time (s)")
    plt.suptitle("average grid-ramp cross-correlation per ramp class")
    plt.savefig(output_path / "average_per_ramp_class.jpg", dpi=300)
    plt.close()

    # Plot
    for group1, group2, name in [
        (grid, grid, "grid-grid"),
        (ramp, ramp, "ramp-ramp"),
        (grid, ramp, "grid-ramp"),
    ]:
        fig = plt.figure(figsize=(len(group1) * 5, len(group2)))
        gs = gridspec.GridSpec(
            len(group2) + 1, len(group1) * 6 + 9, figure=fig, wspace=0.2, hspace=0.2
        )
        # group1 cells
        for i, cell_name in zip(range(0, len(group1) * 6, 6), group1):
            plot_neuron([fig.add_subplot(gs[0, 9 + i + j]) for j in range(6)], cell_name, tcs)
        # group2 cells
        for i, cell_name2 in enumerate(group2, 1):
            plot_neuron(
                [fig.add_subplot(gs[i, j]) for j in range(9)], cell_name2, tcs, label_top=False
            )

            # cross corr
            for j, (task, (neurons, speed_limit, trials)) in zip(
                range(0, len(data) * 2, 2), data.items()
            ):
                for idx, cell_name1 in enumerate(group1):
                    ax_cross = fig.add_subplot(gs[i, 9 + j + idx * 6 : 9 + 2 + j + idx * 6])
                    ax_cross.bar(
                        cross_corr[task].index.values,
                        cross_corr[task][(cell_name1, cell_name2)].values,
                        width=args.bin_size_sec,
                        color="grey",
                        linewidth=1,
                    )
                    ax_cross.tick_params(
                        axis="x", labelsize=2, direction="in", pad=-3, length=1, width=0.3
                    )
                    ax_cross.tick_params(
                        axis="y", labelsize=2, direction="in", pad=-1.5, length=1, width=0.3
                    )
                    ax_cross.set_xlim(
                        cross_corr[task].index.min() * 1.2, cross_corr[task].index.max() * 1.2
                    )
                    ax_cross.xaxis.set_major_locator(MaxNLocator(nbins=5))
                    ax_cross.yaxis.set_major_locator(MaxNLocator(nbins=5))
                    # Customize y-tick labels to exclude zero
                    y_max = cross_corr[task][(cell_name1, cell_name2)].max()
                    ax_cross.set_ylim(0, y_max)
                    y_ticks = ax_cross.get_yticks()
                    y_tick_labels = [f"{tick:.1f}" if tick != 0 else "" for tick in y_ticks]
                    ax_cross.yaxis.set_major_locator(FixedLocator(y_ticks))
                    ax_cross.set_yticklabels(y_tick_labels)
                    # Set horizontal alignment of y-tick labels to be left
                    for label in ax_cross.get_yticklabels():
                        label.set_ha("left")

        fig.savefig(output_path / f"{name}.jpg", dpi=300, bbox_inches="tight")
        plt.close()
