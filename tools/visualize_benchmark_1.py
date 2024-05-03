from collections import defaultdict
from pathlib import Path
import pickle
from matplotlib import ticker
import matplotlib.pyplot as plt
import numpy as np
import scienceplots

plt.style.use(["science", "seaborn-v0_8-talk"])


def main() -> None:
    output_dir = Path("outputs")
    visuals_dir = output_dir.joinpath("visuals")
    visuals_dir.mkdir(exist_ok=True)
    benchmark_dir = output_dir.joinpath("benchmark")
    with open(benchmark_dir.joinpath("metadata.pkl"), "rb") as f:
        metadata = pickle.load(f)

    fig, axs = plt.subplots(1, 3, figsize=(10, 5), sharex=True, sharey=True)
    plt.subplots_adjust(top=0.85)

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
            if "backward_error" in trial:
                backward_errors[method].append(trial["backward_error"])

    axs[0].scatter(
        np.arange(len(residual_errors["lstsq"][-1])),
        residual_errors["lstsq"][-1],
        label="LSQR",
    )
    axs[0].scatter(
        np.arange(len(residual_errors["sketch_and_precondition"][-1])),
        residual_errors["sketch_and_precondition"][-1],
        label="SAP-SAS",
    )
    axs[0].scatter(
        np.arange(len(residual_errors["sketch_and_apply"][-1])),
        residual_errors["sketch_and_apply"][-1],
        label="SAA-SAS",
    )
    # axs[0].legend(loc="best")
    axs[0].set_ylabel(r"Residual Error $\frac{\|Ax - b\|_2}{\|b\|_2}$")
    axs[0].set_xlabel("Iterations")

    axs[1].scatter(
        np.arange(len(forward_errors["lstsq"][-1])),
        forward_errors["lstsq"][-1],
        # label="LSQR",
    )
    axs[1].scatter(
        np.arange(len(forward_errors["sketch_and_precondition"][-1])),
        forward_errors["sketch_and_precondition"][-1],
        # label="SAP-SAS",
    )
    print(forward_errors["sketch_and_apply"][-1])
    axs[1].scatter(
        np.arange(len(forward_errors["sketch_and_apply"][-1])),
        forward_errors["sketch_and_apply"][-1],
        # label="SAA-SAS",
    )
    # axs[1].legend(loc="best")
    axs[1].set_ylabel(r"Forward Error $\frac{\|x - \hat{x}\|_2}{\|x\|_2}$")
    axs[1].set_xlabel("Iterations")

    axs[2].scatter(
        np.arange(len(backward_errors["lstsq"][-1])),
        backward_errors["lstsq"][-1],
        # label="LSQR",
    )
    axs[2].scatter(
        np.arange(len(backward_errors["sketch_and_precondition"][-1])),
        backward_errors["sketch_and_precondition"][-1],
        # label="SAP-SAS",
    )
    axs[2].scatter(
        np.arange(len(backward_errors["sketch_and_apply"][-1])),
        backward_errors["sketch_and_apply"][-1],
        # label="SAA-SAS",
    )
    # axs[2].legend(loc="best")
    axs[2].set_ylabel(r"Backward Error")
    axs[2].set_xlabel("Iterations")

    fig.suptitle(
        r"$A$ is $20000 \times 100, \kappa(A) = 10^{10}, \| Ax^* - b \|_2 = 10^{-12}$",
        y=0.98,
        fontsize=16,
    )
    fig.legend(
        loc="upper center",
        bbox_to_anchor=(0.5, 0.925),
        ncol=3,
        frameon=True,
        fontsize="medium",
    )

    # for ax in axs.ravel():
    #     ax.set_ylim(-0.5, 1)

    plt.savefig(
        visuals_dir.joinpath("fig1.png"),
        dpi=600,
        bbox_inches="tight",
    )


if __name__ == "__main__":
    main()
