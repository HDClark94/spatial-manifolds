import matplotlib.gridspec as gridspec
import matplotlib.pyplot as plt
import numpy as np


def plot_factors(factors, color, color_labels, bin_size, pos, title=None):
    plt.figure(figsize=(8, factors[0].shape[1] * 0.75))
    gs = gridspec.GridSpec(factors[0].shape[1], 3)

    # Define a colormap
    colors = plt.get_cmap("Set1", len(color_labels))
    color_map = {color_type: colors(i) for i, color_type in enumerate(color_labels.keys())}

    # Plot
    trial_min = np.min(factors[2])
    trial_max = np.max(factors[2])
    for factor in range(factors[0].shape[1]):

        # neuron factors
        sorted_indices = np.argsort(factors[0][:, factor])
        ax_neuron = plt.subplot(gs[factor, 0])
        ax_neuron.bar(
            np.arange(factors[0].shape[0]),
            factors[0][sorted_indices, factor],
            color="blue",
            edgecolor="none",
        )
        ax_neuron.xaxis.set_visible(False)
        ax_neuron.yaxis.set_visible(False)
        if factor == 0:
            ax_neuron.set_title("NEURON FACTORS")

        # temporal factors
        ax_temporal = plt.subplot(gs[factor, 1])
        if pos:
            ax_temporal.plot(
                np.linspace(0, 200.0, len(factors[1][:, factor])), factors[1][:, factor]
            )
            ax_temporal.axvspan(0, 30.0, color="gray", alpha=0.5)
            ax_temporal.axvspan(90.0, 110.0, color="green", alpha=0.5)
            ax_temporal.axvspan(170, 200.0, color="gray", alpha=0.5)
            ax_temporal.set_xlim(0.0, 200.0)
        else:
            end = len(factors[1][:, factor]) * bin_size
            ax_temporal.plot(
                np.linspace(0, end, len(factors[1][:, factor])), factors[1][:, factor]
            )
            ax_temporal.set_xlim(0, end)
        ax_temporal.xaxis.set_visible(False)
        ax_temporal.yaxis.set_visible(False)
        if factor == 0:
            if title is None:
                ax_temporal.set_title("TEMPORAL FACTOR")
            else:
                ax_temporal.set_title(f"{title}\n TEMPORAL FACTOR")

        # trial factors
        ax_trial = plt.subplot(gs[factor, 2])
        if factor == 0:
            handles = [
                plt.Line2D(
                    [0],
                    [0],
                    marker="o",
                    color="w",
                    markerfacecolor=color_map[color],
                    markersize=5,
                    label=color_labels[color],
                )
                for color in color_map.keys()
            ]
            ax_trial.legend(
                loc="upper center",
                bbox_to_anchor=(0.5, 2.0),
                ncol=len(color_map),
                handles=handles,
                frameon=False,
                fontsize="small",
                borderaxespad=0.05,
                handletextpad=0.05,
                labelspacing=0.05,
                columnspacing=0.1,
            )

        ax_trial.scatter(
            np.arange(len(factors[2][:, factor])),
            factors[2][:, factor],
            s=5,
            c=[color_map[c] for c in color.astype(int)],
            alpha=0.6,
        )
        ax_trial.set_ylim(trial_min, trial_max)
        ax_trial.xaxis.set_visible(False)
        ax_trial.yaxis.set_visible(False)
        if factor == 0:
            ax_trial.set_title("TRIAL FACTORS")

    ax_neuron.xaxis.set_visible(True)
    ax_neuron.set_xlabel("neuron")
    ax_temporal.set_xlabel("position (cm)" if pos else "time (s)")
    ax_temporal.xaxis.set_visible(True)
    ax_trial.set_xlabel("trial")
    ax_trial.xaxis.set_visible(True)
