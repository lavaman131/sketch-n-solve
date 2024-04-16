from collections import defaultdict
import pickle
import matplotlib.pyplot as plt
import scienceplots

plt.style.use(["science", "seaborn-v0_8-talk"])


def main() -> None:
    with open("metrics.pkl", "rb") as f:
        metrics = pickle.load(f)
    with open("metadata.pkl", "rb") as f:
        metadata = pickle.load(f)

    fig = plt.figure(figsize=(6.5, 5))
    ax = fig.add_subplot(111)

    times = defaultdict(list)
    condition_numbers = [1, 5, 10, 100, 1e3, 1e4, 1e5]

    for method in metadata.keys():
        for trial in metadata[method]:
            times[method].append(trial["time_elapsed"])

    ax.plot(
        condition_numbers,
        times["lstsq"],
        marker="o",
        markersize=7.5,
        label="Least Squares (Deterministic)",
    )
    ax.plot(
        condition_numbers,
        times["sketch_and_precondition"],
        marker="o",
        markersize=7.5,
        label="Sketch and Precondition",
    )
    ax.plot(
        condition_numbers,
        times["sketch_and_apply"],
        marker="o",
        markersize=7.5,
        label="Sketch and Apply",
    )
    ax.legend(loc="upper right")
    ax.set_ylim(0, 0.5)
    ax.set_xscale("log")
    ax.set_xlabel(r"Condition Number ($\kappa$)")
    ax.set_ylabel("Time (sec)")
    ax.set_title(
        "Comparison of Least Squares Methods for 4000 x 50 Matrix", pad=20, fontsize=17
    )
    plt.savefig("benchmark.png", dpi=600, bbox_inches="tight")


if __name__ == "__main__":
    main()
