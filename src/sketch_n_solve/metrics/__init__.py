from collections import defaultdict
from pathlib import Path
from typing import Dict, List
import numpy as np
from sketch_n_solve.metrics.least_squares import (
    backward_error,
    forward_error,
    residual_error,
)
from typing import TypedDict
import scipy.linalg as SLA
from sketch_n_solve.solve.least_squares import LeastSquares
from sketch_n_solve.utils import timer
from tqdm import tqdm
import h5py
from sketch_n_solve.solve.least_squares.utils import lsqr


class LeastSquaresMetaData(TypedDict):
    forward_error: List[float]
    residual_error: List[float]
    backward_error: List[float]
    norm_r: float
    time_elapsed: float
    cond: float
    beta: float


class LeastSquaresMetricCallback:
    def __init__(self) -> None:
        self.x_hat = []

    @staticmethod
    def calculate_least_squares_error_metrics(
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

    def x_hat_callback(self, x_hat: np.ndarray):
        self.x_hat.append(x_hat)

    def cleanup_callback(self):
        self.x_hat = []

    def __call__(
        self,
        problem_paths: List[Path],
        lsq: LeastSquares,
    ) -> List[LeastSquaresMetaData]:
        metadata = defaultdict(list)
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
                default_metadata = {
                    "norm_r": SLA.norm(r_x),
                    "cond": cond,
                    "beta": beta,
                    "m": A.shape[0],
                    "n": A.shape[1],
                }
                _, time_elapsed_lstq = timer(lsqr)(A, b, callback=self.x_hat_callback)
                metadata["lstsq"].append(
                    {
                        "time_elapsed": time_elapsed_lstq,
                        **default_metadata,
                        **LeastSquaresMetricCallback.calculate_least_squares_error_metrics(
                            A, b, x, self.x_hat
                        ),
                    }
                )
                self.cleanup_callback()
                _, time_elapsed_sketch_and_precondition = timer(
                    lsq.sketch_and_precondition
                )(A, b, callback=self.x_hat_callback)
                metadata["sketch_and_precondition"].append(
                    {
                        "time_elapsed": time_elapsed_sketch_and_precondition,
                        **default_metadata,
                        **LeastSquaresMetricCallback.calculate_least_squares_error_metrics(
                            A, b, x, self.x_hat
                        ),
                    }
                )
                self.cleanup_callback()
                _, time_elapsed_sketch_and_apply = timer(lsq.sketch_and_apply)(
                    A, b, sparsity_parameter=None, callback=self.x_hat_callback
                )

                metadata["sketch_and_apply"].append(
                    {
                        "time_elapsed": time_elapsed_sketch_and_apply,
                        **default_metadata,
                        **LeastSquaresMetricCallback.calculate_least_squares_error_metrics(
                            A, b, x, self.x_hat
                        ),
                    }
                )

                self.cleanup_callback()
        return metadata  # type: ignore
