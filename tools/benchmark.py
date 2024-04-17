from pathlib import Path
from sketch_n_solve.metrics import (
    LeastSquaresMetricCallback,
)
from sketch_n_solve.solve.least_squares import LeastSquares
import pickle


def main() -> None:
    seed = 42
    output_dir = Path("outputs")
    benchmark_dir = output_dir.joinpath("benchmark")
    benchmark_dir.mkdir(exist_ok=True)
    precomputed_problems_dir = output_dir.joinpath("precomputed_problems")
    problem_paths = list(precomputed_problems_dir.glob("*.h5"))
    sketch_fn = "sparse_sign"
    lsq = LeastSquares(sketch_fn, seed)
    metric_callback = LeastSquaresMetricCallback()
    metadata = metric_callback(problem_paths, lsq)
    with open(benchmark_dir.joinpath("metadata.pkl"), "wb") as f:
        pickle.dump(metadata, f)


if __name__ == "__main__":
    main()
