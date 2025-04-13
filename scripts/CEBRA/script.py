import itertools
from mpl_toolkits.axes_grid1.inset_locator import inset_axes
from matplotlib import cm
from pycolormap_2d import ColorMap2DSteiger
import plotly.graph_objects as go
from argparse import ArgumentParser
from pathlib import Path

import cebra.integrations.plotly
import matplotlib.pyplot as plt
import numpy as np
import pynapple as nap
from cebra import CEBRA
from sklearn.model_selection import train_test_split

from spatial_manifolds.data.loading import load_session


def plot_embedding(embedding, labels, colors, training_name, legend_name, path):
    if colors == "cont":
        colormap = plt.get_cmap("viridis")
        norm = plt.Normalize(vmin=np.min(labels), vmax=np.max(labels))
        color_array = colormap(norm(labels))
    elif colors == "disc":
        unique_values = labels.unique()
        colormap = plt.get_cmap("tab10", len(unique_values))
        colormap = {val: colormap(i) for i, val in enumerate(unique_values)}
        color_array = [colormap[val] for val in labels]
    elif colors == "2d":
        colormap = ColorMap2DSteiger(
            range_x=(float(np.min(labels[:, 0])), float(np.max(labels[:, 0]))),
            range_y=(float(np.min(labels[:, 1])), float(np.max(labels[:, 1]))),
        )
        color_array = (
            np.array([colormap(float(px), float(py)) for px, py in labels.to_numpy()]) / 255
        )

    if embedding.shape[1] == 3:
        if colors == "cont":
            fig = go.Figure(
                data=[
                    go.Scatter3d(
                        x=embedding[:, 0],
                        y=embedding[:, 1],
                        z=embedding[:, 2],
                        mode="markers",
                        marker=dict(
                            size=5,
                            color=labels,
                            colorscale="Viridis",
                            opacity=0.4,
                            colorbar=dict(title=legend_name),
                        ),
                    )
                ]
            )
            fig.update_layout(showlegend=False)
        elif colors == "disc":
            fig = go.Figure()
            for val, color in colormap.items():
                mask = labels == val
                fig.add_trace(
                    go.Scatter3d(
                        x=embedding[mask, 0],
                        y=embedding[mask, 1],
                        z=embedding[mask, 2],
                        mode="markers",
                        marker=dict(
                            size=5,
                            color=f"rgba({color[0]*255}, {color[1]*255}, {color[2]*255}, {color[3]})",
                            opacity=0.4,
                        ),
                        name=str(val),
                    )
                )
            fig.update_layout(showlegend=True, legend=dict(title=dict(text=legend_name)))
        elif colors == "2d":
            fig = go.Figure()
            for val, color in colors.items():
                mask = labels == val
                fig.add_trace(
                    go.Scatter3d(
                        x=embedding[mask, 0],
                        y=embedding[mask, 1],
                        z=embedding[mask, 2],
                        mode="markers",
                        marker=dict(
                            size=5,
                            color=f"rgba({color[0]*255}, {color[1]*255}, {color[2]*255}, {color[3]})",
                            opacity=0.4,
                        ),
                        name=str(val),
                    )
                )
            fig.update_layout(showlegend=False)
        fig.update_layout(
            title=f"{''.join(args.session.parts[2:])}\nCEBRA-behaviour: {training_name} (legend: {legend_name})",
            paper_bgcolor="white",
            plot_bgcolor="white",
            scene=dict(
                xaxis=dict(
                    backgroundcolor="white",
                    title="CEBRA1",
                    showbackground=True,
                    gridcolor="lightgrey",
                ),
                yaxis=dict(
                    backgroundcolor="white",
                    title="CEBRA2",
                    showbackground=True,
                    gridcolor="lightgrey",
                ),
                zaxis=dict(
                    backgroundcolor="white",
                    title="CEBRA3",
                    showbackground=True,
                    gridcolor="lightgrey",
                ),
            ),
        )
        fig.write_html(
            path / f"{legend_name}.html",
            include_plotlyjs="cdn",
            full_html=False,
        )
    elif embedding.shape[1] == 2:
        if colors == "2d":
            fig, ax1 = plt.subplots(figsize=(4, 3))
            ax1.scatter(embedding[:, 0], embedding[:, 1], c=color_array, s=5, alpha=0.4)
            ax1.set_title(f"{''.join(args.session.parts[2:])}\nCEBRA-{training_name}")
            ax1.axis("off")

            # Plot 2D colormap
            ax2 = inset_axes(
                ax1,
                width="45%",
                height="50%",
                loc="upper right",
                bbox_to_anchor=(0.75, 0.5, 0.5, 0.5),
                bbox_transform=ax1.transAxes,
                borderpad=0,
            )
            resolution = 40
            x_range = (np.min(labels[:, 0]), np.max(labels[:, 0]))
            y_range = (np.min(labels[:, 1]), np.max(labels[:, 1]))
            x = np.linspace(x_range[0], x_range[1], resolution)
            y = np.linspace(y_range[0], y_range[1], resolution)
            gradient = np.zeros((resolution, resolution, 3))

            for i in range(resolution):
                for j in range(resolution):
                    gradient[i, j] = colormap(x[i], y[j]) / 255
            X, Y = np.meshgrid(x, y)
            ax2.pcolormesh(X, Y, gradient, shading="auto")
            ax2.set_title("2D POSITION", fontsize=8)
            ax2.axis("off")
            plt.tight_layout()
            plt.subplots_adjust(right=0.75)
        else:
            fig, ax = plt.subplots(figsize=(4, 3))
            if colors == "cont":
                ax.scatter(
                    embedding[:, 0],
                    embedding[:, 1],
                    c=color_array,
                    s=5,
                    alpha=0.4,
                    edgecolors="none",
                )
                cbar = plt.colorbar(cm.ScalarMappable(norm=norm, cmap=colormap), ax=ax)
                cbar.set_label(legend_name, fontsize=8)
                cbar.ax.tick_params(labelsize=8)
            else:
                for val, color in colormap.items():
                    mask = labels == val
                    ax.scatter(
                        embedding[mask, 0],
                        embedding[mask, 1],
                        color=color,
                        s=5,
                        alpha=0.4,
                        label=val,
                        edgecolors="none",
                    )
                if len(colormap) > 3:
                    plt.legend(
                        title=legend_name,
                        fontsize=8,
                        ncol=3,
                        frameon=False,
                        bbox_to_anchor=(1.05, 1),
                        loc="upper left",
                        handletextpad=-0.5,
                    )
                else:
                    plt.legend(
                        title=legend_name,
                        fontsize=8,
                        frameon=False,
                        bbox_to_anchor=(1.05, 1),
                        loc="upper left",
                        handletextpad=-0.5,
                    )
            plt.title(f"{''.join(args.session.parts[2:])}\nCEBRA-{training_name}")
            plt.axis("off")
            plt.tight_layout()
        plt.savefig(path / f"{legend_name}.png", dpi=300)
        plt.close()


def make_plots(embedding, behaviour, vars_disc, vars_cont, var, path):
    path.mkdir(parents=True, exist_ok=True)

    # Time figure
    plot_embedding(embedding, behaviour.times(), "cont", var, "time", path)
    for color_var in vars_disc:
        plot_embedding(embedding, behaviour[var], "disc", var, color_var, path)
    for color_var in vars_cont:
        plot_embedding(embedding, behaviour[var], "cont", var, color_var, path)
    if "P_x" in behaviour.columns:
        plot_embedding(embedding, behaviour[("P_x", "P_y")], "2d", var, "PxPy", path)


def generate_stratified_train_test_split(data_length, labels, test_size=0.2, random_state=42):
    indices = np.arange(data_length)
    train_indices, test_indices = train_test_split(
        indices, test_size=test_size, random_state=random_state, stratify=labels
    )
    train_mask = np.zeros(data_length, dtype=bool)
    test_mask = np.zeros(data_length, dtype=bool)
    train_mask[train_indices] = True
    test_mask[test_indices] = True
    return train_mask, test_mask


def generate_edge_train_test_split(data_length, labels, test_size=0.2, random_state=42):
    # Compute the size of the test set
    test_count = int(data_length * test_size)
    test_mask = np.zeros(data_length, dtype=bool)
    test_mask[-test_count:] = True  # Test chunk at the end
    train_mask = ~test_mask  # Complement of the test mask for training
    return train_mask, test_mask


if __name__ == "__main__":
    parser = ArgumentParser("Computing CEBRA manifolds for Nolan lab ephys data.")
    parser.add_argument(
        "--storage",
        type=Path,
        default=".",
        help="Path to the storage.",
    )
    parser.add_argument(
        "--task",
        type=str,
        default="vr",
        help="Task to run the analysis for.",
        choices=["vr", "of1", "of2", "img", "imgseq"],
    )
    parser.add_argument(
        "--session",
        required=True,
        type=Path,
        help="Path to session folder from storage/data/",
    )
    parser.add_argument("--bin_size_sec", type=float, default=0.04)
    parser.add_argument("--seed", type=int, default=420)
    args = parser.parse_args()

    # Load session
    neurons, derivatives = load_session(args.storage, args.session, args.task, curate=True)

    # Setup output directory
    output_path = args.storage / "output" / "CEBRA" / "/".join(args.session.parts[1:]) / args.task
    output_path.mkdir(parents=True, exist_ok=True)

    # Restriction
    restriction = nap.IntervalSet(
        start=neurons.time_support.start + 10, end=neurons.time_support.end - 10
    )

    # Variables
    if "of" in args.task:
        vars_cont = ["P_x", "P_y"]
        vars_disc = []
    elif args.task == "vr":
        vars_cont = ["P", "S"]
        vars_disc = ["trial_number", "trial_type"]
        vars_cont = ["P", "S"]
        vars_disc = []
        iterations = 3000
        dim = 3
    elif args.task == "imgseq":
        vars_cont = []
        vars_disc = ["image", "image_index"]

        # just a tryout
        # behaviour = derivatives["behaviour"].as_dataframe()
        # behaviour.loc[behaviour["image_index"] > 7, "image"] = 200
        # behaviour.loc[behaviour["image_index"] > 7, "image_index"] = 200
        # derivatives["behaviour"] = nap.TsdFrame(
        #    behaviour, time_support=derivatives["behaviour"].time_support
        # )
        iterations = 20000
        dim = 10
    elif args.task == "img":
        vars_cont = []
        vars_disc = ["image"]
        iterations = 20000
        dim = 10

    behaviour = derivatives["behaviour"][vars_cont].bin_average(args.bin_size_sec, ep=restriction)
    behaviour = nap.TsdFrame(
        behaviour.as_dataframe().interpolate(), time_support=behaviour.time_support
    )
    for var in vars_disc:
        behaviour = nap.TsdFrame(
            behaviour.as_dataframe().assign(
                **{
                    var: np.abs(
                        behaviour.value_from(derivatives["behaviour"][var])
                        .to_numpy()
                        .astype(np.int32)
                    )
                }
            ),
            time_support=behaviour.time_support,
        )
    spike_counts = neurons.count(args.bin_size_sec, ep=behaviour.time_support).to_numpy()

    # TIME
    path = output_path / "time"
    path.mkdir(parents=True, exist_ok=True)
    model = CEBRA(
        model_architecture="offset10-model",
        batch_size=2048,
        learning_rate=3e-4,
        temperature_mode="auto",
        output_dimension=dim,
        max_iterations=iterations,
        num_hidden_units=32,
        distance="cosine",
        conditional="time",
        device="cuda_if_available",
        verbose=True,
        time_offsets=10,
    )
    model.fit(
        spike_counts,
    )
    if dim < 4:
        make_plots(model.transform(spike_counts), behaviour, vars_disc, vars_cont, "time", path)
    embedding = nap.TsdFrame(
        d=model.transform(spike_counts), t=behaviour.times(), time_support=behaviour.time_support
    )
    embedding.save(path / "embedding.npz")
    cebra.plot_loss(model)
    plt.savefig(path / "loss.pdf")
    plt.close()
    cebra.plot_temperature(model)
    plt.savefig(path / "temperature.pdf")
    plt.close()
    quit()

    # BEHAVIOUR
    for var in vars_disc + vars_cont:
        train_mask, test_mask = generate_edge_train_test_split(
            len(spike_counts), labels=behaviour[var], test_size=0.3
        )
        path = output_path / "behaviour" / var
        (path / "train").mkdir(parents=True, exist_ok=True)
        (path / "test").mkdir(parents=True, exist_ok=True)
        (path / "shuffled").mkdir(parents=True, exist_ok=True)

        # First train shuffled data
        shuffled_model = CEBRA(
            model_architecture="offset10-model-mse",
            batch_size=2048,
            learning_rate=3e-4,
            temperature_mode="auto",
            output_dimension=dim,
            max_iterations=iterations,
            num_hidden_units=32,
            distance="euclidean",
            conditional="time_delta",
            device="cuda_if_available",
            verbose=True,
            time_offsets=10,
            optimizer_kwargs=(
                ("betas", (0.9, 0.999)),
                ("eps", 1e-08),
                ("weight_decay", 0.001),
                ("amsgrad", False),
            ),
        )
        shuffled_model.fit(spike_counts, np.random.permutation(behaviour[var].to_numpy()))

        make_plots(
            shuffled_model.transform(spike_counts),
            behaviour,
            vars_disc,
            vars_cont,
            var,
            path / "shuffled",
        )
        cebra.plot_loss(shuffled_model)
        plt.savefig(path / "shuffled" / "loss.pdf")
        cebra.plot_temperature(shuffled_model)
        plt.savefig(path / "shuffled" / "temperature.pdf")

        model = CEBRA(
            model_architecture="offset10-model-mse",
            batch_size=2048,
            learning_rate=3e-4,
            temperature_mode="auto",
            output_dimension=dim,
            max_iterations=iterations,
            num_hidden_units=32,
            distance="euclidean",
            conditional="time_delta",
            device="cuda_if_available",
            verbose=True,
            time_offsets=10,
            optimizer_kwargs=(
                ("betas", (0.9, 0.999)),
                ("eps", 1e-08),
                ("weight_decay", 0.001),
                ("amsgrad", False),
            ),
        )
        model.fit(
            spike_counts[train_mask],
            np.abs(behaviour[var].to_numpy()[train_mask]),
        )
        train_embedding = model.transform(spike_counts[train_mask])
        if dim < 4:
            make_plots(
                train_embedding,
                behaviour[train_mask],
                vars_disc,
                vars_cont,
                var,
                path / "train",
            )
        cebra.plot_loss(model)
        plt.savefig(path / "loss.pdf")
        plt.close()
        cebra.plot_temperature(model)
        plt.savefig(path / "temperature.pdf")
        plt.close()
        np.savez_compressed(path / "train" / "embedding.npz", train_embedding)
        for var in behaviour.columns:
            np.savez_compressed(
                path / "train" / f"{var}.npz", behaviour[train_mask][var].to_numpy()
            )

        test_embedding = model.transform(spike_counts[test_mask])
        if dim < 4:
            make_plots(
                test_embedding,
                behaviour[test_mask],
                vars_disc,
                vars_cont,
                var,
                path / "test",
            )
        np.savez_compressed(path / "test" / "embedding.npz", test_embedding)
        for var in behaviour.columns:
            np.savez_compressed(path / "test" / f"{var}.npz", behaviour[test_mask][var].to_numpy())
