from sketch_n_solve.metrics import calculate_least_squares_metrics
from sketch_n_solve.solve.least_squares import LeastSquares
from sketch_n_solve.utils.least_squares import (
    LeastSquaresProblemConfig,
)
import pickle


def main() -> None:
    seed = 42
    config = [
        {"m": 4000, "n": 50, "cond": 1},
        {"m": 4000, "n": 50, "cond": 5},
        {"m": 4000, "n": 50, "cond": 10},
        {"m": 4000, "n": 50, "cond": 100},
        {"m": 4000, "n": 50, "cond": 1e3},
        {"m": 4000, "n": 50, "cond": 1e4},
        {"m": 4000, "n": 50, "cond": 1e5},
    ]
    trials = [LeastSquaresProblemConfig(**kwarg) for kwarg in config]
    sketch_fn = "sparse_sign"
    lsq = LeastSquares(sketch_fn, seed)
    metrics, metadata = calculate_least_squares_metrics(trials, lsq)
    with open("metrics.pkl", "wb") as f:
        pickle.dump(metrics, f)
    with open("metadata.pkl", "wb") as f:
        pickle.dump(metadata, f)


if __name__ == "__main__":
    main()
