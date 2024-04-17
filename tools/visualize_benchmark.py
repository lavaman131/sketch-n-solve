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
    with open(benchmark_dir.joinpath("metadata.pkl"), "rb") as f:
        metadata = pickle.load(f)

    fig = plt.figure(figsize=(6.5, 5))
    ax = fig.add_subplot(111)

    times = defaultdict(list)
    conds = defaultdict(list)

    for method in metadata.keys():
        trials = metadata[method]
        sorted_trials = sorted(trials, key=lambda x: x["m"])
        for trial in sorted_trials:
            times[method].append(trial["time_elapsed"])
            conds[method].append(trial["m"])

    ax.plot(
        conds["lstsq"],
        times["lstsq"],
        marker="o",
        markersize=7.5,
        label="QR",
    )
    ax.plot(
        conds["sketch_and_precondition"],
        times["sketch_and_precondition"],
        marker="o",
        markersize=7.5,
        label="SAP-SAS",
    )
    ax.plot(
        conds["sketch_and_apply"],
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
    plt.savefig(visuals_dir.joinpath("benchmark.png"), dpi=600, bbox_inches="tight")


if __name__ == "__main__":
    main()
