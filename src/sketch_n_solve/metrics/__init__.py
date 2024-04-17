from collections import defaultdict
from pathlib import Path
from typing import Dict, List, Tuple, Union
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
from sketch_n_solve.utils.least_squares import (
    LeastSquaresProblemConfig,
    generate_least_squares_problem,
)
from tqdm import tqdm
import h5py


class LeastSquaresMetaData(TypedDict):
    forward_error: float
    residual_error: float
    backward_error: float
    norm_r: float
    time_elapsed: float
    cond: float
    beta: float


def calculate_least_squares_error_metrics(
    A: np.ndarray, b: np.ndarray, x: np.ndarray, x_hat: np.ndarray
) -> Dict[str, float]:
    metrics = dict()
    metrics["forward_error"] = forward_error(x, x_hat)
    metrics["residual_error"] = residual_error(A, b, x_hat)
    # metrics["backward_error"] = backward_error(A, b, x_hat)
    return metrics


def calculate_least_squares_metrics(
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
            lstsq_output, time_elapsed_lstq = timer(SLA.lstsq)(A, b)
            x_hat_lstsq = lstsq_output[0]
            metadata["lstsq"].append(
                {
                    "time_elapsed": time_elapsed_lstq,
                    **default_metadata,
                    **calculate_least_squares_error_metrics(A, b, x, x_hat_lstsq),
                }
            )
            x_hat_sketch_and_precondition, time_elapsed_sketch_and_precondition = timer(
                lsq.sketch_and_precondition
            )(A, b)
            metadata["sketch_and_precondition"].append(
                {
                    "time_elapsed": time_elapsed_sketch_and_precondition,
                    **default_metadata,
                    **calculate_least_squares_error_metrics(
                        A, b, x, x_hat_sketch_and_precondition
                    ),
                }
            )
            x_hat_sketch_and_apply, time_elapsed_sketch_and_apply = timer(
                lsq.sketch_and_apply
            )(A, b, sparsity_parameter=None)
            metadata["sketch_and_apply"].append(
                {
                    "time_elapsed": time_elapsed_sketch_and_apply,
                    **default_metadata,
                    **calculate_least_squares_error_metrics(
                        A, b, x, x_hat_sketch_and_apply
                    ),
                }
            )
    return metadata  # type: ignore
