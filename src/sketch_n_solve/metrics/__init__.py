from collections import defaultdict
from typing import Dict, List, Tuple
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


class LeastSquaresMetrics(TypedDict):
    forward_error: float
    residual_error: float
    backward_error: float


class MetaData(TypedDict):
    norm_r: float
    time_elapsed: float


def calculate_least_squares_error_metrics(
    A: np.ndarray, b: np.ndarray, x: np.ndarray, x_hat: np.ndarray
) -> LeastSquaresMetrics:
    metrics = dict()
    metrics["forward_error"] = forward_error(x, x_hat)
    metrics["residual_error"] = residual_error(A, b, x_hat)
    metrics["backward_error"] = backward_error(A, b, x_hat)
    return metrics  # type: ignore


def calculate_least_squares_metrics(
    trials: List[LeastSquaresProblemConfig],
    lsq: LeastSquares,
) -> Tuple[Dict[str, List[LeastSquaresMetrics]], Dict[str, List[MetaData]]]:
    metrics = defaultdict(list)
    metadata = defaultdict(list)
    for config in tqdm(trials):
        problem = generate_least_squares_problem(
            m=config.m,
            n=config.n,
            cond=config.cond,
            beta=config.beta,
            seed=lsq.seed,
        )
        A, b, x, r_x = problem["A"], problem["b"], problem["x"], problem["r_x"]
        norm_r = SLA.norm(r_x)
        lstsq_output, time_elapsed_lstq = timer(SLA.lstsq)(A, b)
        x_hat_lstsq = lstsq_output[0]
        metadata["lstsq"].append({"norm_r": norm_r, "time_elapsed": time_elapsed_lstq})
        x_hat_sketch_and_precondition, time_elapsed_sketch_and_precondition = timer(
            lsq.sketch_and_precondition
        )(A, b)
        metadata["sketch_and_precondition"].append(
            {"norm_r": norm_r, "time_elapsed": time_elapsed_sketch_and_precondition}
        )
        x_hat_sketch_and_apply, time_elapsed_sketch_and_apply = timer(
            lsq.sketch_and_apply
        )(A, b, sparsity_parameter=None)
        metadata["sketch_and_apply"].append(
            {"norm_r": norm_r, "time_elapsed": time_elapsed_sketch_and_apply}
        )
        metrics["lstsq"].append(
            calculate_least_squares_error_metrics(A, b, x, x_hat_lstsq)
        )
        metrics["sketch_and_precondition"].append(
            calculate_least_squares_error_metrics(
                A, b, x, x_hat_sketch_and_precondition
            )
        )
        metrics["sketch_and_apply"].append(
            calculate_least_squares_error_metrics(A, b, x, x_hat_sketch_and_apply)
        )
    return metrics, metadata
