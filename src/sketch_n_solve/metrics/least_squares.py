from typing import Optional
import numpy as np
import numpy.linalg as LA


def forward_error(x: np.ndarray, x_hat: np.ndarray) -> float:
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
    assert x.ndim == 1, "The true solution should be a vector."
    return float(LA.norm(x - x_hat) / LA.norm(x))


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
    assert y.ndim == 1, "The target vector should be a vector."
    y_hat = A @ x_hat
    r = LA.norm(y)
    residual_error = LA.norm(y - y_hat) / r
    return float(residual_error)


def backward_error(
    A: np.ndarray, y: np.ndarray, x: np.ndarray, theta: Optional[float] = None
) -> float:
    r"""Compute the backward error.
    If the backward error is small, then :math:`\\hat{x}` is the true solution to nearly the right least-squares problem.
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
    assert y.ndim == 1, "The target vector should be a vector."
    norm_x = LA.norm(x)
    r = y - A @ x
    norm_r = LA.norm(r)

    if theta:
        mu = theta**2 * norm_x**2 / (1 + theta**2 * norm_x**2)
    else:
        mu = 1

    phi = np.sqrt(mu) * norm_r / norm_x
    outer_product = np.outer(r, r) / norm_r**2
    matrix = np.hstack((A, phi * (np.eye(A.shape[0]) - outer_product)))
    backward_error = np.minimum(phi, SLA.svd(matrix, compute_uv=False).min())  # type: ignore

    return backward_error
