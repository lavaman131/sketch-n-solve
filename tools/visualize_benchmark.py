from collections import defaultdict
from pathlib import Path
import pickle
from matplotlib import ticker
import matplotlib.pyplot as plt
import numpy as np
import scienceplots  # noqa: F401

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
    kappa = "10^{10}"
    beta = "10^{-10}"
    n = "10^3"
    step_size = 10
    with open(benchmark_dir.joinpath("metadata.pkl"), "rb") as f:
        metadata = pickle.load(f)

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
            residual_errors[method].extend(trial["residual_error"])
            forward_errors[method].extend(trial["forward_error"])
            if "backward_error" in trial:
                backward_errors[method].extend(trial["backward_error"])

    fig = plt.figure(figsize=(6.5, 5))
    ax = fig.add_subplot(111)

    ax.plot(
        rows["lstsq"],
        times["lstsq"],
        marker="o",
        markersize=7.5,
        label="LSQR",
        linewidth=2,
    )
    ax.plot(
        rows["sketch_and_precondition"],
        times["sketch_and_precondition"],
        marker="s",  # square marker
        markersize=7.5,
        label="SAP-SAS",
        linewidth=2,
    )
    ax.plot(
        rows["sketch_and_apply"],
        times["sketch_and_apply"],
        marker="^",  # triangle marker
        markersize=7.5,
        label="SAA-SAS",
        linewidth=2,
    )
    ax.legend(loc="best")
    ax.set_xscale("log")
    ax.set_xlabel(r"$m$")
    ax.set_ylabel("Time (sec)")
    ax.set_title(rf"$n={n}, \kappa(A) = {kappa}, \| Ax^* - b \|_2 = {beta}$")
    plt.savefig(
        visuals_dir.joinpath("benchmark_times.png"), dpi=600, bbox_inches="tight"
    )

    fig = plt.figure(figsize=(6.5, 5))
    ax = fig.add_subplot(111)

    ax.plot(
        np.arange(0, len(residual_errors["lstsq"]), step_size),
        residual_errors["lstsq"][::step_size],
        marker="o",
        markersize=7.5,
        label="LSQR",
        linewidth=2,
    )
    ax.plot(
        np.arange(0, len(residual_errors["sketch_and_precondition"]), step_size),
        residual_errors["sketch_and_precondition"][::step_size],
        marker="s",
        markersize=7.5,
        label="SAP-SAS",
        linewidth=2,
    )
    ax.plot(
        np.arange(0, len(residual_errors["sketch_and_apply"]), step_size),
        residual_errors["sketch_and_apply"][::step_size],
        marker="^",
        markersize=7.5,
        label="SAA-SAS",
        linewidth=2,
    )
    ax.legend(loc="best")
    ax.set_yscale("log")
    ax.set_xlabel("Iterations")
    ax.set_ylabel(r"Residual Error $\frac{\|Ax - b\|_2}{\|b\|_2}$")
    ax.set_title(rf"$n={n}, \kappa(A) = {kappa}, \| Ax^* - b \|_2 = {beta}$")

    plt.savefig(
        visuals_dir.joinpath("benchmark_residual_error.png"),
        dpi=600,
        bbox_inches="tight",
    )

    fig = plt.figure(figsize=(6.5, 5))
    ax = fig.add_subplot(111)

    ax.plot(
        np.arange(0, len(forward_errors["lstsq"]), step_size),
        forward_errors["lstsq"][::step_size],
        marker="o",
        markersize=7.5,
        label="LSQR",
        linewidth=2,
    )
    ax.plot(
        np.arange(0, len(forward_errors["sketch_and_precondition"]), step_size),
        forward_errors["sketch_and_precondition"][::step_size],
        marker="s",
        markersize=7.5,
        label="SAP-SAS",
        linewidth=2,
    )
    ax.plot(
        np.arange(0, len(forward_errors["sketch_and_apply"]), step_size),
        forward_errors["sketch_and_apply"][::step_size],
        marker="^",
        markersize=7.5,
        label="SAA-SAS",
        linewidth=2,
    )
    ax.legend(loc="best")
    # Create a FuncFormatter object with the formatting function
    formatter = ticker.FuncFormatter(format_func)

    # Set the formatter for the y-axis tick labels
    ax.yaxis.set_major_formatter(formatter)
    ax.set_xlabel("Iterations")
    ax.set_ylabel(r"Forward Error $\frac{\|x - \hat{x}\|_2}{\|x\|_2}$")
    ax.set_title(rf"$n={n}, \kappa(A) = {kappa}, \| Ax^* - b \|_2 = {beta}$")
    plt.savefig(
        visuals_dir.joinpath("benchmark_forward_error.png"),
        dpi=600,
        bbox_inches="tight",
    )

    # fig = plt.figure(figsize=(6.5, 5))
    # ax = fig.add_subplot(111)

    # ax.plot(
    #     backward_errors["lstsq"][-1],
    #     label="LSQR",
    # )
    # ax.plot(
    #     backward_errors["sketch_and_precondition"][-1],
    #     label="SAP-SAS",
    # )
    # ax.plot(
    #     backward_errors["sketch_and_apply"][-1],
    #     label="SAA-SAS",
    # )
    # ax.legend(loc="best")
    # # Create a FuncFormatter object with the formatting function
    # formatter = ticker.FuncFormatter(format_func)

    # # Set the formatter for the y-axis tick labels
    # ax.yaxis.set_major_formatter(formatter)
    # ax.set_xlabel("Iterations")
    # ax.set_ylabel(r"Backward Error")
    # ax.set_title(rf"$n={n}, \kappa(A) = {kappa}, \| Ax^* - b \|_2 = {beta}$")
    # plt.savefig(
    #     visuals_dir.joinpath("benchmark_backward_error.png"),
    #     dpi=600,
    #     bbox_inches="tight",
    # )


if __name__ == "__main__":
    main()
