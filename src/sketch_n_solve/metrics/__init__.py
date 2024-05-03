from collections import defaultdict
from pathlib import Path
import time
from typing import Any, Dict, List, Optional
import numpy as np
from sketch_n_solve.metrics.least_squares import (
    backward_error,
    forward_error,
    residual_error,
)
from typing import TypedDict
import numpy.linalg as LA
from sketch_n_solve.solve.least_squares import LeastSquares
from tqdm import tqdm
import h5py
from sketch_n_solve.solve.least_squares.utils import lsqr
from scipy.sparse import csc_array


class LeastSquaresMetaData(TypedDict):
    forward_error: List[float]
    residual_error: List[float]
    backward_error: List[float]
    norm_r: float
    time_elapsed: float
    cond: float
    beta: float


class LeastSquaresMetricCallback:
    def __init__(self, metadata: Dict[str, Any] = defaultdict(list)) -> None:
        self.metadata = metadata

    @staticmethod
    def calculate_least_squares_error_metrics_batch(
        A: np.ndarray,
        b: np.ndarray,
        x: np.ndarray,
        x_hats: List[np.ndarray],
        calculate_backward_error: bool = False,
    ) -> Dict[str, List[float]]:
        assert x_hats, "No computed solutions."
        metrics = defaultdict(list)
        for x_hat in x_hats:
            metrics["forward_error"].append(forward_error(x, x_hat))
            metrics["residual_error"].append(residual_error(A, b, x_hat))
            if calculate_backward_error:
                metrics["backward_error"].append(backward_error(A, b, x_hat))
        return metrics

    def __call__(
        self,
        method: str,
        problem_paths: List[Path],
        lsq: LeastSquares,
    ) -> List[LeastSquaresMetaData]:
        for problem_path in tqdm(problem_paths):
            with h5py.File(problem_path, "r") as f:
                problem = {key: np.array(f[key]) for key in f.keys()}
                A, b, x, r_x, cond, beta = (
                    problem["A"],
                    problem["b"],
                    problem["x"],
                    problem["r_x"],
                    problem["cond"],
                    problem["beta"],
                )
                x = x.squeeze()
                b = b.squeeze()
                default_metadata = {
                    "norm_r": LA.norm(r_x),
                    "cond": cond,
                    "beta": beta,
                    "m": A.shape[0],
                    "n": A.shape[1],
                }

                if method == "sketch_and_apply":
                    x, time_elapsed, x_hats, _ = lsq.sketch_and_apply(
                        A, b, sparsity_parameter=None, log_x_hat=True
                    )

                    self.metadata["sketch_and_apply"].append(
                        {
                            "time_elapsed": time_elapsed,
                            **default_metadata,
                            **LeastSquaresMetricCallback.calculate_least_squares_error_metrics_batch(
                                A, b, x, x_hats, calculate_backward_error=False
                            ),
                        }
                    )
                elif method == "sketch_and_precondition":
                    x, time_elapsed, x_hats, _ = lsq.sketch_and_precondition(
                        A, b, log_x_hat=True
                    )
                    self.metadata["sketch_and_precondition"].append(
                        {
                            "time_elapsed": time_elapsed,
                            **default_metadata,
                            **LeastSquaresMetricCallback.calculate_least_squares_error_metrics_batch(
                                A, b, x, x_hats, calculate_backward_error=False
                            ),
                        }
                    )
                elif method == "lstsq":
                    start_time = time.perf_counter()
                    x, x_hats, *_ = lsqr(A, b, log_x_hat=True, iter_lim=100)
                    end_time = time.perf_counter()
                    time_elapsed = end_time - start_time
                    self.metadata["lstsq"].append(
                        {
                            "time_elapsed": time_elapsed,
                            **default_metadata,
                            **LeastSquaresMetricCallback.calculate_least_squares_error_metrics_batch(
                                A, b, x, x_hats, calculate_backward_error=False
                            ),
                        }
                    )
                else:
                    raise NotImplementedError(f"Unknown method: {method}")

        return self.metadata  # type: ignore
