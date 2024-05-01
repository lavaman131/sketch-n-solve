from collections import defaultdict
from pathlib import Path
import pickle
from matplotlib import ticker
import matplotlib.pyplot as plt
import numpy as np
import scienceplots

plt.style.use(["science", "seaborn-v0_8-talk"])


def format_func(value, pos):
    if value != 0:
        exp = int(np.floor(np.log10(abs(value))))
        base = value / 10**exp
        return r"${:.1f} \times 10^{{{}}}$".format(base, exp)
    else:
        return r"$0$"


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

    for method in metadata.keys():
        trials = metadata[method]
        sorted_trials = sorted(trials, key=lambda x: x["m"])
        for trial in sorted_trials:
            times[method].append(trial["time_elapsed"])
            rows[method].append(trial["m"])
            residual_errors[method].append(trial["residual_error"])
            forward_errors[method].append(trial["forward_error"])
    ax.plot(
        rows["lstsq"],
        times["lstsq"],
        marker="o",
        markersize=7.5,
        label="LSQR",
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
        residual_errors["lstsq"][-1],
        label="LSQR",
    )
    ax.plot(
        residual_errors["sketch_and_precondition"][-1],
        label="SAP-SAS",
    )
    ax.plot(
        residual_errors["sketch_and_apply"][-1],
        label="SAA-SAS",
    )
    ax.legend(loc="upper left")
    ax.set_yscale("log")
    ax.set_xlabel("Iterations")
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
        forward_errors["lstsq"][-1],
        label="LSQR",
    )
    ax.plot(
        forward_errors["sketch_and_precondition"][-1],
        label="SAP-SAS",
    )
    ax.plot(
        forward_errors["sketch_and_apply"][-1],
        label="SAA-SAS",
    )
    ax.legend(loc="upper left")
    # Create a FuncFormatter object with the formatting function
    formatter = ticker.FuncFormatter(format_func)

    # Set the formatter for the y-axis tick labels
    ax.yaxis.set_major_formatter(formatter)
    ax.set_xlabel("Iterations")
    ax.set_ylabel(r"Forward Error $\frac{\|x - \hat{x}\|_2}{\|x\|_2}$")
    ax.set_title(r"$n=10^3$")
    plt.savefig(
        visuals_dir.joinpath("benchmark_forward_error.png"),
        dpi=600,
        bbox_inches="tight",
    )


if __name__ == "__main__":
    main()
