from collections import defaultdict
from pathlib import Path
import pickle
import matplotlib.pyplot as plt
import numpy as np
import scienceplots
import random

plt.style.use(["science", "seaborn-v0_8-talk"])


def main() -> None:
    output_dir = Path("outputs")
    visuals_dir = output_dir.joinpath("visuals")
    visuals_dir.mkdir(exist_ok=True)
    benchmark_dir = output_dir.joinpath("benchmark")
    with open(benchmark_dir.joinpath("metadata.pkl"), "rb") as f:
        metadata = pickle.load(f)

    fig = plt.figure(figsize=(6.5, 5))
    ax = fig.add_subplot(111)

    times = defaultdict(list)
    rows = defaultdict(list)
    residual_errors = defaultdict(list)
    forward_errors = defaultdict(list)
    backward_errors = defaultdict(list)

    for method in metadata.keys():
        trials = metadata[method]
        sorted_trials = sorted(trials, key=lambda x: x["m"])
        for trial in sorted_trials:
            times[method].append(trial["time_elapsed"])
            rows[method].append(trial["m"])
            residual_errors[method].append(trial["residual_error"])
            forward_errors[method].append(trial["forward_error"])
            backward_errors[method].append(trial["backward_error"])

    ax.plot(
        rows["lstsq"],
        times["lstsq"],
        marker="o",
        markersize=7.5,
        label="QR",
    )
    ax.plot(
        rows["sketch_and_precondition"],
        times["sketch_and_precondition"],
        marker="o",
        markersize=7.5,
        label="SAP-SAS",
    )
    ax.plot(
        rows["sketch_and_apply"],
        times["sketch_and_apply"],
        marker="o",
        markersize=7.5,
        label="SAA-SAS",
    )
    ax.legend(loc="upper left")
    ax.set_xscale("log")
    ax.set_xlabel(r"$m$")
    ax.set_ylabel("Time (sec)")
    ax.set_title(r"$n=10^3$")
    plt.savefig(
        visuals_dir.joinpath("benchmark_times.png"), dpi=600, bbox_inches="tight"
    )

    fig = plt.figure(figsize=(6.5, 5))
    ax = fig.add_subplot(111)

    ax.plot(
        rows["lstsq"],
        residual_errors["lstsq"],
        marker="o",
        markersize=7.5,
        label="QR",
    )
    ax.plot(
        rows["sketch_and_precondition"],
        residual_errors["sketch_and_precondition"],
        marker="o",
        markersize=7.5,
        label="SAP-SAS",
    )
    ax.plot(
        rows["sketch_and_apply"],
        np.array(residual_errors["sketch_and_apply"]) + 0.1,
        marker="o",
        markersize=7.5,
        label="SAA-SAS",
    )
    ax.legend(loc="upper left")
    ax.set_xscale("log")
    ax.set_ylim(-0.5, 10)
    ax.set_xlabel(r"$m$")
    ax.set_ylabel(r"Residual Error $\frac{\|Ax - b\|_2}{\|b\|_2}$")
    ax.set_title(r"$n=10^3$")

    plt.savefig(
        visuals_dir.joinpath("benchmark_residual_error.png"),
        dpi=600,
        bbox_inches="tight",
    )

    fig = plt.figure(figsize=(6.5, 5))
    ax = fig.add_subplot(111)

    ax.plot(
        rows["lstsq"],
        forward_errors["lstsq"],
        marker="o",
        markersize=7.5,
        label="QR",
    )
    ax.plot(
        rows["sketch_and_precondition"],
        forward_errors["sketch_and_precondition"],
        marker="o",
        markersize=7.5,
        label="SAP-SAS",
    )
    ax.plot(
        rows["sketch_and_apply"],
        np.array(forward_errors["sketch_and_apply"]) + 0.1,
        marker="o",
        markersize=7.5,
        label="SAA-SAS",
    )
    ax.legend(loc="upper left")
    ax.set_xscale("log")
    ax.set_ylim(-0.5, 10)
    ax.set_xlabel(r"$m$")
    ax.set_ylabel(r"Forward Error $\frac{\|x - \hat{x}\|_2}{\|x\|_2}$")
    ax.set_title(r"$n=10^3$")
    plt.savefig(
        visuals_dir.joinpath("benchmark_forward_error.png"),
        dpi=600,
        bbox_inches="tight",
    )

    fig = plt.figure(figsize=(6.5, 5))
    ax = fig.add_subplot(111)

    ax.plot(
        rows["lstsq"],
        backward_errors["lstsq"],
        marker="o",
        markersize=7.5,
        label="QR",
    )
    ax.plot(
        rows["sketch_and_precondition"],
        backward_errors["sketch_and_precondition"],
        marker="o",
        markersize=7.5,
        label="SAP-SAS",
    )
    ax.plot(
        rows["sketch_and_apply"],
        np.array(backward_errors["sketch_and_apply"]) + 0.1,
        marker="o",
        markersize=7.5,
        label="SAA-SAS",
    )
    ax.legend(loc="upper left")
    ax.set_xscale("log")
    ax.set_ylim(-0.5, 10)
    ax.set_xlabel(r"$m$")
    ax.set_ylabel(r"Backward Error $\frac{\|Ax - A\hat{x}\|_2}{\|A\|_2\|\hat{x}\|_2}$")
    ax.set_title(r"$n=10^3$")
    plt.savefig(
        visuals_dir.joinpath("benchmark_forward_error.png"),
        dpi=600,
        bbox_inches="tight",
    )


if __name__ == "__main__":
    main()
