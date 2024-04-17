from typing import Any, Callable, Optional
import numpy as np
import scipy.linalg as SLA
from scipy.linalg.lapack import dtrtrs as triangular_solve
import scipy.sparse
from sketch_n_solve.solve.least_squares.utils import lsqr


def _sketch_and_precondition(
    A: np.ndarray,
    b: np.ndarray,
    S: np.ndarray,
    use_sketch_and_solve_x_0: bool = True,
    tolerance: float = 1e-6,
    num_iters: Optional[int] = None,
    callback: Optional[Callable[[np.ndarray], None]] = None,
    **kwargs: Any,
) -> np.ndarray:
    """Solves the least squares problem using sketch and preconditioning as described in https://arxiv.org/pdf/2302.07202.pdf.

    Parameters
    ----------
    A : (n, d) np.ndarray
        The input matrix.
    b : (n, 1) np.ndarray
        The target vector.
    S : np.ndarray
        The sketch matrix.
    use_sketch_and_solve_x_0 : bool, optional
        Whether to use x_0 from sketch and solve as the initial guess for the least squares solver rather than the zero vector, by default True.
    tolerance : float, optional
        Error tolerance. Controls the number of iterations if num_iters is not specified, by default 1e-6.
    num_iters : int, optional
        Maximum number of iterations for least-squares QR solver, by default None.
    callback : Optional[Callable[[np.ndarray], None]], optional
        Callback function to be called after each iteration of LSQR, by default None.
    **kwargs : Any
        Additional required arguments depending on the sketch function.

    Returns
    -------
    x : (d, 1) np.ndarray
        The solution to the least squares problem.
    """
    assert b.ndim == 2, "The target vector should be a column vector."
    assert (
        A.shape[0] == b.shape[0]
    ), "The number of rows of the input matrix and the target vector should be the same."
    assert (
        num_iters is None or num_iters > 0
    ), "Number of iterations should be greater than 0."
    B = S @ A
    Q, R = SLA.qr(B, mode="economic", pivoting=False)  # type: ignore

    if use_sketch_and_solve_x_0:
        y_0 = triangular_solve(R, Q.T @ S @ b, lower=False)[0]  # type: ignore
        x_0 = y_0.squeeze()
    else:
        x_0 = None

    A_precond = A @ triangular_solve(R, np.eye(R.shape[0]), lower=False)[0]

    y = lsqr(
        A_precond,
        b,
        x0=x_0,
        iter_lim=num_iters,
        atol=tolerance,
        btol=tolerance,
        callback=callback,
    )[0]

    x = triangular_solve(R, y, lower=False)[0]
    return x.reshape(-1, 1)


def _sketch_and_apply(
    A: np.ndarray,
    b: np.ndarray,
    S: np.ndarray,
    tolerance: float = 1e-6,
    num_iters: Optional[int] = None,
    callback: Optional[Callable[[np.ndarray], None]] = None,
    **kwargs: Any,
) -> np.ndarray:
    """Solves the least squares problem using sketch-and-apply as described in https://arxiv.org/pdf/2302.07202.pdf.

    Parameters
    ----------
    A : (n, d) np.ndarray
        The input matrix.
    b : (n, 1) np.ndarray
        The target vector.
    S : np.ndarray
        The sketch matrix.
    tolerance : float, optional
        Error tolerance. Controls the number of iterations if num_iters is not specified, by default 1e-6.
    num_iters : int, optional
        Maximum number of iterations for least-squares QR solver, by default None.
    callback : Optional[Callable[[np.ndarray], None]], optional
        Callback function to be called after each iteration of LSQR, by default None.
    **kwargs : Any
        Additional required arguments depending on the sketch function.

    Returns
    -------
    x : (d, 1) np.ndarray
        The solution to the least squares problem.
    """
    assert b.ndim == 2, "The target vector should be a column vector."
    assert (
        A.shape[0] == b.shape[0]
    ), "The number of rows of the input matrix and the target vector should be the same."
    assert tolerance > 0, "Error tolerance should be greater than 0."
    assert (
        num_iters is None or num_iters > 0
    ), "Number of iterations should be greater than 0."

    B = S @ A
    Q, R = SLA.qr(B, mode="economic", pivoting=False)  # type: ignore
    z_0 = Q.T @ S @ b  # type: ignore
    z = lsqr(
        Q,  # type: ignore
        S @ b,
        x0=z_0.squeeze(),
        iter_lim=num_iters,
        atol=tolerance,
        btol=tolerance,
        callback=callback,
    )[0]
    x = triangular_solve(R, z, lower=False)[0]

    return x.reshape(-1, 1)


def _smoothed_sketch_and_apply(
    A: np.ndarray,
    b: np.ndarray,
    S: np.ndarray,
    tolerance: float = 1e-6,
    num_iters: Optional[int] = None,
    seed: Optional[int] = 42,
    callback: Optional[Callable[[np.ndarray], None]] = None,
    **kwargs: Any,
) -> np.ndarray:
    """Solves the least squares problem using smoothed sketch-and-apply as described in https://arxiv.org/pdf/2302.07202.pdf.

    Parameters
    ----------
    A : (n, d) np.ndarray
        The input matrix.
    b : (n, 1) np.ndarray
        The target vector.
    S : np.ndarray
        The sketch matrix.
    tolerance : float
        Error tolerance. Controls the number of iterations if num_iters is not specified.
    num_iters : int, optional
        Maximum number of iterations for least-squares QR solver, by default None. If specified will overwrite tolerance parameter for error tolerance.
    seed : int, optional
        Random seed for generation of G, by default 42.
    callback : Optional[Callable[[np.ndarray], None]], optional
        Callback function to be called after each iteration of LSQR, by default None.
    **kwargs : Any
        Additional required arguments depending on the sketch function.

    Returns
    -------
    x : (d, 1) np.ndarray
        The solution to the least squares problem.
    """
    assert b.ndim == 2, "The target vector should be a column vector."
    assert (
        A.shape[0] == b.shape[0]
    ), "The number of rows of the input matrix and the target vector should be the same."
    assert tolerance > 0, "Error tolerance should be greater than 0."
    assert (
        num_iters is None or num_iters > 0
    ), "Number of iterations should be greater than 0."
    rng = np.random.default_rng(seed)
    m, n = A.shape
    sigma = 10 * SLA.norm(A) * np.finfo(float).eps
    G = rng.standard_normal(size=(m, n))
    A_tilde = A + sigma * G / np.sqrt(m)
    x = _sketch_and_apply(A_tilde, b, S, tolerance, num_iters, callback)
    return x.reshape(-1, 1)
