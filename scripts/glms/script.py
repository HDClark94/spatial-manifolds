import os
from argparse import ArgumentParser
from pathlib import Path

import pynapple as nap

from spatial_manifolds.data.loading import load_session
from spatial_manifolds.glms import PoissonGLM
from spatial_manifolds.util import get_speed_limit

if __name__ == "__main__":
    args = parser = ArgumentParser("Fitting Poisson GLMs for Nolan lab ephys data.")
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
        choices=["vr", "of1", "of2"],
    )
    parser.add_argument(
        "--session",
        required=True,
        type=Path,
        help="Path to session folder from storage/data/",
    )
    args = parser.parse_args()

    # Load session
    neurons, derivatives = load_session(args.storage, args.session, args.task, curate=True)
    theta_phase = derivatives["theta_phase"].bin_average(0.02).dropna()
    behaviour = derivatives["behaviour"].interpolate(nap.Ts(theta_phase.times()))

    # Setup output directory
    output_path = args.storage / args.session / args.task / "glms"
    os.makedirs(output_path, exist_ok=True)

    # Speed limit
    speed_limit = get_speed_limit(behaviour["S"], args.task)
    neurons = neurons[neurons.restrict(speed_limit).rates > 0.5]
    behaviour = behaviour.restrict(speed_limit).as_dataframe()
    theta_phase = theta_phase.restrict(speed_limit)

    # Create and fit models
    spike_counts = neurons.count(0.02, ep=theta_phase.time_support)
    for neuron, extremum_channel in enumerate(neurons["extremum_channel"]):
        behaviour["T"] = theta_phase[extremum_channel].values
        X = PoissonGLM.apply_bases(
            nap.TsdFrame(behaviour),
            args.task,
        )
        glm = PoissonGLM(X, spike_counts[:, neuron].values, neurons.index[neuron])
