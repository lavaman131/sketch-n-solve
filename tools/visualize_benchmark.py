from collections import defaultdict
from pathlib import Path
import pickle
import matplotlib.pyplot as plt
import scienceplots

plt.style.use(["science", "seaborn-v0_8-talk"])


def main() -> None:
    output_dir = Path("outputs")
    visuals_dir = output_dir.joinpath("visuals")
    visuals_dir.mkdir(exist_ok=True)
    benchmark_dir = output_dir.joinpath("benchmark")
    with open(benchmark_dir.joinpath("metrics.pkl"), "rb") as f:
        metrics = pickle.load(f)
    with open(benchmark_dir.joinpath("metadata.pkl"), "rb") as f:
        metadata = pickle.load(f)

    fig = plt.figure(figsize=(6.5, 5))
    ax = fig.add_subplot(111)

    times = defaultdict(list)
    conds = defaultdict(list)

    for method in metadata.keys():
        trials = metadata[method]
        sorted_trials = sorted(trials, key=lambda x: x["cond"])
        for trial in sorted_trials:
            times[method].append(trial["time_elapsed"])
            conds[method].append(trial["cond"])

    print()
    ax.plot(
        conds["lstsq"],
        times["lstsq"],
        marker="o",
        markersize=7.5,
        label="Least Squares (Deterministic)",
    )
    ax.plot(
        conds["sketch_and_precondition"],
        times["sketch_and_precondition"],
        marker="o",
        markersize=7.5,
        label="Sketch and Precondition",
    )
    ax.plot(
        conds["sketch_and_apply"],
        times["sketch_and_apply"],
        marker="o",
        markersize=7.5,
        label="Sketch and Apply",
    )
    ax.legend(loc="upper right")
    ax.set_ylim(-0.1, 0.5)
    ax.set_xscale("log")
    ax.set_xlabel(r"Condition Number ($\kappa$)")
    ax.set_ylabel("Time (sec)")
    ax.set_title(
        "Comparison of Least Squares Methods for 4000 x 50 Matrix", pad=20, fontsize=17
    )
    plt.savefig(visuals_dir.joinpath("benchmark.png"), dpi=600, bbox_inches="tight")


if __name__ == "__main__":
    main()
