import os
import pickle
from argparse import ArgumentParser
from pathlib import Path

import matplotlib.pyplot as plt
import nemos as nmo
import numpy as np
import pandas as pd
import pynapple as nap
from scipy.ndimage import gaussian_filter

from spatial_manifolds.data.loading import load_session
from spatial_manifolds.tuning_curves import TuningCurveTypes
from spatial_manifolds.tuning_scores.grid_score import correlate2d_ignore_nan
from spatial_manifolds.util import get_speed_limit, get_trials


def plot_coupling(
    responses,
    tuning,
    color_tun,
    cmap_name="seismic",
    figsize=(10, 8),
    fontsize=15,
    alpha=0.5,
):

    # plot heatmap
    sum_resp = np.sum(responses, axis=2)
    # normalize by cols (for fixed receiver neuron, scale all responses
    # so that the strongest peaks to 1)
    sum_resp_n = (sum_resp.T / sum_resp.max(axis=1)).T

    # scale to 0,1
    color = -0.5 * (sum_resp_n - sum_resp_n.min()) / sum_resp_n.min()

    cmap = plt.get_cmap(cmap_name)
    n_row, n_col, n_tp = responses.shape
    time = np.arange(n_tp)
    fig, axs = plt.subplots(n_row + 1, n_col + 1, figsize=figsize, sharey="row")
    for rec, rec_resp in enumerate(responses):
        for send, resp in enumerate(rec_resp):
            axs[rec, send].plot(time, responses[rec, send], color="k")
            axs[rec, send].spines["left"].set_visible(False)
            axs[rec, send].spines["bottom"].set_visible(False)
            axs[rec, send].set_xticks([])
            axs[rec, send].set_yticks([])
            axs[rec, send].axhline(0, color="k", lw=0.5)
            if rec == n_row - 1:
                axs[n_row, send].remove()  # Remove the original axis
                axs[n_row, send] = fig.add_subplot(
                    n_row + 1,
                    n_col + 1,
                    np.ravel_multi_index((n_row, send + 1), (n_row + 1, n_col + 1)),
                )  # Add new axis

                axs[n_row, send].plot(
                    tuning.iloc[:, send].index,
                    tuning.iloc[:, send].values,
                    color=color_tun[send],
                    alpha=0.5,
                )
                axs[n_row, send].set_xticks([])
                axs[n_row, send].set_yticks([])

        axs[rec, send + 1].remove()  # Remove the original axis
        axs[rec, send + 1] = fig.add_subplot(
            n_row + 1,
            n_col + 1,
            np.ravel_multi_index((rec, send + 1), (n_row + 1, n_col + 1)) + 1,
        )  # Add new axis

        axs[rec, send + 1].plot(
            tuning.iloc[:, rec].index,
            tuning.iloc[:, rec].values,
            color=color_tun[rec],
            alpha=0.5,
        )
        axs[rec, send + 1].set_xticks([])
        axs[rec, send + 1].set_yticks([])
    axs[rec + 1, send + 1].set_xticks([])
    axs[rec + 1, send + 1].set_yticks([])
    axs[rec + 1, send + 1].spines["left"].set_visible(False)
    axs[rec + 1, send + 1].spines["bottom"].set_visible(False)
    for rec, rec_resp in enumerate(responses):
        for send, resp in enumerate(rec_resp):
            xlim = axs[rec, send].get_xlim()
            ylim = axs[rec, send].get_ylim()
            rect = plt.Rectangle(
                (xlim[0], ylim[0]),
                xlim[1] - xlim[0],
                ylim[1] - ylim[0],
                alpha=alpha,
                color=cmap(color[rec, send]),
                zorder=1,
            )
            axs[rec, send].add_patch(rect)
            axs[rec, send].set_xlim(xlim)
            axs[rec, send].set_ylim(ylim)
    axs[n_row // 2, 0].set_ylabel("receiver\n", fontsize=fontsize)
    axs[n_row, n_col // 2].set_xlabel("\nsender", fontsize=fontsize)

    plt.suptitle("Pairwise Interaction", fontsize=fontsize)
    return fig


def plot_vr(ax1, ax2, neuron_name):
    ax1.plot(tcs["vr"]["position"].tcs_mean[neuron_name])
    ax1.set_xlabel("position (cm)")
    ax1.set_ylabel("spikes/s")
    ax1.axvspan(0, 30.0, color="gray", alpha=0.5)
    ax1.axvspan(90.0, 110.0, color="green", alpha=0.5)
    ax1.axvspan(170, 200.0, color="gray", alpha=0.5)
    ax1.set_xticks([0, 200])
    ax1.set_xlim(0.0, 200.0)
    ax1.xaxis.set_visible(False)
    ax1.yaxis.set_visible(False)

    ax2.imshow(
        tcs["vr"]["position"].tcs_trial[
            :, :, np.where(tcs["vr"]["position"].neuron_index == neuron_name)[0][0]
        ],
        aspect="auto",
        cmap="viridis",
    )
    ax2.set_xticks([0, len(tcs["vr"]["position"].tcs_mean.index) - 1], ["0", "200"])
    ax2.set_xlabel("position (cm)")
    ax2.set_ylabel("trial")
    ax2.xaxis.set_visible(False)
    ax2.yaxis.set_visible(False)


def plot_of(ax1, ax2, neuron_name, tcs):
    tc = tcs.tcs_mean[0][neuron_name]
    mask = np.isnan(tc)
    tc = gaussian_filter(np.nan_to_num(tc).astype(np.float64), sigma=1.5)
    ax1.imshow(tc, aspect="equal", cmap="viridis")
    ax1.xaxis.set_visible(False)
    ax1.yaxis.set_visible(False)

    tc[mask] = np.nan
    autocorr = correlate2d_ignore_nan(tc)
    ax2.imshow(autocorr, aspect="equal", cmap="viridis")
    ax2.xaxis.set_visible(False)
    ax2.yaxis.set_visible(False)


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
        type=int,
        default=0.02,
    )
    parser.add_argument(
        "--window_size_sec",
        type=int,
        default=1.0,
    )
    args = parser.parse_args()

    # Load data
    data = {task: load_session(args.storage, args.session, task) for task in ["of1", "of2", "vr"]}
    data = {
        task: (
            task_neurons,
            task_derivatives,
            get_speed_limit(task_derivatives["behaviour"], task),
        )
        for task, (task_neurons, task_derivatives) in data.items()
    }

    # Setup output directory
    output_path = args.storage / args.session / "crosscorr"
    os.makedirs(output_path, exist_ok=True)

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

    # Fit population GLM
    model = nmo.glm.PopulationGLM()
    window_size = int(args.window_size_sec * 1 / args.bin_size_sec)
    basis = nmo.basis.RaisedCosineBasisLinear(
        n_basis_funcs=9, mode="conv", window_size=window_size, predictor_causality="causal"
    )
    time, basis_kernels = basis.evaluate_on_grid(window_size)
    for task, (task_neurons, task_derivatives, task_speed_limit) in data.items():
        if task == "vr":
            spike_counts = nap.TsdFrame(
                task_neurons[grid + ramp]
                .count(args.bin_size_sec)
                .as_dataframe()
                .reindex(grid + ramp, axis=1)
            )
            spike_counts = task_neurons[grid + ramp].count(args.bin_size_sec)
            X = basis.compute_features(spike_counts).restrict(task_speed_limit)
            spike_counts = spike_counts.restrict(task_speed_limit)
            model = nmo.glm.PopulationGLM(
                regularizer="Ridge", solver_name="LBFGS", regularizer_strength=0.1
            ).fit(X, spike_counts)

            predicted_firing_rate = model.predict(X) * X.rate
            if task == "vr":
                position = (
                    task_derivatives["behaviour"]["P"]
                    .bin_average(args.bin_size_sec)
                    .restrict(task_speed_limit)
                )
                tuning = nap.compute_1d_tuning_curves_continuous(
                    predicted_firing_rate,
                    feature=position,
                    nb_bins=100,
                    minmax=(0.0, 200.0),
                    ep=task_speed_limit,
                )
            else:
                position = (
                    task_derivatives["behaviour"][["P_x", "P_y"]]
                    .bin_average(args.bin_size_sec)
                    .restrict(task_speed_limit)
                )
                tuning = nap.compute_2d_tuning_curves_continuous(
                    predicted_firing_rate,
                    features=position,
                    nb_bins=50,
                    minmax=(0.0, 100.0, 0.0, 100.0),
                    ep=task_speed_limit,
                )
            weights = model.coef_.reshape(
                spike_counts.shape[1], basis.n_basis_funcs, spike_counts.shape[1]
            )
            responses = np.einsum("jki,tk->ijt", weights, basis_kernels)
            plot_coupling(
                responses, tuning, color_tun=["green"] * len(grid) + ["purple"] * len(ramp)
            )
            plt.show()
