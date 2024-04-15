from typing import Dict
import numpy as np
from sketch_n_solve.metrics.least_squares import (
    backward_error,
    forward_error,
    residual_error,
)


def calculate_least_squares_metrics(
    A: np.ndarray, b: np.ndarray, x: np.ndarray, x_hat: np.ndarray
) -> Dict[str, float]:
    metrics = dict()
    metrics["forward_error"] = forward_error(x, x_hat)
    metrics["residual_error"] = residual_error(A, b, x_hat)
    metrics["backward_error"] = backward_error(A, b, x_hat)
    return metrics
