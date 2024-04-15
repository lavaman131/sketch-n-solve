from typing import Optional
import numpy as np
import scipy.linalg as SLA


def forward_error(x: np.ndarray, x_hat: np.ndarray) -> np.floating:
    r"""Compute the forward error. The forward error quantifies how close the computed solution :math:`\hat{x}` is to the true solution :math:`x`.

    Parameters
    ----------
    x : np.ndarray
        The true solution.
    x_hat : np.ndarray
        The computed solution.

    Returns
    -------
    float
        The forward error.
    """
    return SLA.norm(x - x_hat) / SLA.norm(x)


def residual_error(A: np.ndarray, y: np.ndarray, x_hat: np.ndarray) -> float:
    r"""Compute the residual error. The residual error measures the suboptimality of :math:`\hat{x}` as a solution to the least-squares minimization problem.

    Parameters
    ----------
    y : np.ndarray
        The target vector.
    x_hat : np.ndarray
        The computed solution.

    Returns
    -------
    residual_error : float
        The residual error.
    """
    y_hat = A @ x_hat
    r = SLA.norm(y)
    error = SLA.norm(y - y_hat) / r
    residual_error = np.sqrt(1 + error**2) * r
    return residual_error


def backward_error(
    A: np.ndarray, y: np.ndarray, x: np.ndarray, theta: Optional[float] = None
) -> float:
    r"""Compute the backward error. If the backward error is small, then :math:`\hat{x}` is the true solution to nearly the right least-squares problem.

    Parameters
    ----------
    A : np.ndarray
        The input matrix.
    y : np.ndarray
        The target vector.
    x : np.ndarray
        The true solution.
    theta : float, optional
        by default np.inf

    Returns
    -------
    backward_error : float
        The backward error.
    """
    norm_x = SLA.norm(x)
    r = y - A @ x
    norm_r = SLA.norm(r)

    if theta:
        mu = theta**2 * norm_x**2
        mu = mu / (1 + mu)
    else:
        mu = 1

    phi = np.sqrt(mu) * norm_r / norm_x

    identity = np.eye(A.shape[0])
    outer_product = np.outer(r, r) / norm_r**2
    matrix = np.hstack((A, phi * (identity - outer_product)))

    backward_error = np.min(phi, np.min(SLA.svd(matrix, compute_uv=False)))

    return backward_error
