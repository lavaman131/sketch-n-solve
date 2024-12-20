from collections import defaultdict
from pathlib import Path

import numpy as np
from sketch_n_solve.metrics import (
    LeastSquaresMetricCallback,
)
from sketch_n_solve.solve.least_squares import LeastSquares
import pickle


def main() -> None:
    seed = 42
    # cpu warmup
    _ = np.empty((10000, 10000))
    output_dir = Path("outputs")
    benchmark_dir = output_dir.joinpath("benchmark")
    benchmark_dir.mkdir(exist_ok=True)
    precomputed_problems_dir = output_dir.joinpath("precomputed_problems")
    problem_paths = list(precomputed_problems_dir.glob("*.npz"))
    sketch_fn = "clarkson_woodruff"
    lsq = LeastSquares(sketch_fn, seed)

    metadata = defaultdict(list)

    metric_callback = LeastSquaresMetricCallback(metadata=metadata)

    metadata = metric_callback(
        method="lstsq",
        problem_paths=problem_paths,
        lsq=lsq,
        # calculate_backward_error=True,
    )
    metadata = metric_callback(
        method="sketch_and_precondition",
        problem_paths=problem_paths,
        lsq=lsq,
        # calculate_backward_error=True,
    )
    metadata = metric_callback(
        method="sketch_and_apply",
        problem_paths=problem_paths,
        lsq=lsq,
        # calculate_backward_error=True,
    )

    with open(benchmark_dir.joinpath("metadata.pkl"), "wb") as f:
        pickle.dump(metadata, f)


if __name__ == "__main__":
    main()
