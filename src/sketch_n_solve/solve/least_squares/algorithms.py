import time
from typing import Any, List, Optional, Tuple
import numpy as np
import numpy.linalg as LA
import scipy.linalg as SLA
from scipy.linalg.lapack import dtrtrs as triangular_solve
from sketch_n_solve.solve.least_squares.utils import lsqr
from scipy.sparse.linalg import LinearOperator


def _sketch_and_precondition(
    A: LinearOperator,
    b: np.ndarray,
    S: LinearOperator,
    use_sketch_and_solve_x_0: bool = True,
    tolerance: float = 1e-12,
    iter_lim: Optional[int] = 100,
    log_x_hat: bool = False,
    **kwargs: Any,
) -> Tuple[np.ndarray, float, List[np.ndarray]]:
    """Solves the least squares problem using sketch and preconditioning as described in https://arxiv.org/pdf/2302.07202.pdf.

    Parameters
    ----------
    A : LinearOperator
        The input matrix as a LinearOperator.
    b : (n, 1) np.ndarray
        The target vector.
    S : LinearOperator
        The sketch matrix as a LinearOperator.
    use_sketch_and_solve_x_0 : bool, optional
        Whether to use x_0 from sketch and solve as the initial guess for the least squares solver rather than the zero vector, by default True.
    tolerance : float, optional
        Error tolerance. Controls the number of iterations if iter_lim is not specified, by default 1e-12.
    iter_lim : int, optional
        Maximum number of iterations for least-squares QR solver, by default 100.
    log_x_hat : bool, optional
        Whether to log the intermediate solutions, by default False.
    **kwargs : Any
        Additional required arguments depending on the sketch function.

    Returns
    -------
    x : (d, 1) np.ndarray
        The solution to the least squares problem.
    time_elapsed : float
        Time taken to solve the least squares problem.
    x_hats : List[np.ndarray]
        List of intermediate solutions if log_x_hat is True.
    """
    assert b.ndim == 1, "The target vector should be a vector."
    assert (
        A.shape[0] == b.shape[0]
    ), "The number of rows of the input matrix and the target vector should be the same."
    assert (
        iter_lim is None or iter_lim > 0
    ), "Number of iterations should be greater than 0."
    start_time = time.perf_counter()
    B = S.matmat(A)
    c = S.matvec(b)
    Q, R = SLA.qr(B, mode="economic", pivoting=False)  # type: ignore

    if use_sketch_and_solve_x_0:
        x_0 = triangular_solve(R, Q.T @ c, lower=False)[0].squeeze()  # type: ignore
    else:
        x_0 = None

    A_precond = A @ triangular_solve(R, np.eye(R.shape[0]), lower=False)[0]

    y, y_hats, *_ = lsqr(
        A=A_precond,
        b=b,
        x0=x_0,
        atol=tolerance,
        btol=tolerance,
        iter_lim=iter_lim,
        log_x_hat=log_x_hat,
    )
    x = triangular_solve(R, y, lower=False)[0]
    end_time = time.perf_counter()

    time_elapsed = end_time - start_time
    x_hats = []
    if log_x_hat:
        x_hats = [triangular_solve(R, y_hat, lower=False)[0] for y_hat in y_hats]
    return x, time_elapsed, x_hats


def _sketch_and_apply(
    A: LinearOperator,
    b: np.ndarray,
    S: LinearOperator,
    tolerance: float = 1e-12,
    iter_lim: Optional[int] = 100,
    log_x_hat: bool = False,
    **kwargs: Any,
) -> Tuple[np.ndarray, float, List[np.ndarray]]:
    """Solves the least squares problem using sketch-and-apply as described in https://arxiv.org/pdf/2302.07202.pdf.

    Parameters
    ----------
    A : LinearOperator
        The input matrix as a LinearOperator.
    b : (n, 1) np.ndarray
        The target vector.
    S : LinearOperator
        The sketch matrix as a LinearOperator.
    tolerance : float, optional
        Error tolerance. Controls the number of iterations if iter_lim is not specified, by default 1e-12.
    iter_lim : int, optional
        Maximum number of iterations for least-squares QR solver, by default 100.
    log_x_hat : bool, optional
        Whether to log the intermediate solutions, by default False.
    **kwargs : Any
        Additional required arguments depending on the sketch function.

    Returns
    -------
    x : (d, 1) np.ndarray
        The solution to the least squares problem.
    time_elapsed : float
        Time taken to solve the least squares problem.
    x_hats : List[np.ndarray]
        List of intermediate solutions if log_x_hat is True.
    """
    assert b.ndim == 1, "The target vector should be a vector."
    assert (
        A.shape[0] == b.shape[0]
    ), "The number of rows of the input matrix and the target vector should be the same."
    assert tolerance > 0, "Error tolerance should be greater than 0."
    assert (
        iter_lim is None or iter_lim > 0
    ), "Number of iterations should be greater than 0."

    start_time = time.perf_counter()
    B = S.matmat(A)
    c = S.matvec(b)
    Q, R = SLA.qr(B, mode="economic", pivoting=False)  # type: ignore
    z_0 = (Q.T @ c).squeeze()  # type: ignore
    z, z_hats, *_ = lsqr(
        A=Q,  # type: ignore
        b=c,
        x0=z_0,
        atol=tolerance,
        btol=tolerance,
        iter_lim=iter_lim,
        log_x_hat=log_x_hat,
    )
    x = triangular_solve(R, z, lower=False)[0]
    end_time = time.perf_counter()

    time_elapsed = end_time - start_time
    x_hats = []
    if log_x_hat:
        x_hats = [triangular_solve(R, z_hat, lower=False)[0] for z_hat in z_hats]

    return x, time_elapsed, x_hats


def _smoothed_sketch_and_apply(
    A: LinearOperator,
    b: np.ndarray,
    S: LinearOperator,
    tolerance: float = 1e-12,
    iter_lim: Optional[int] = 100,
    seed: Optional[int] = 42,
    log_x_hat: bool = False,
    **kwargs: Any,
) -> Tuple[np.ndarray, float, List[np.ndarray]]:
    """Solves the least squares problem using smoothed sketch-and-apply as described in https://arxiv.org/pdf/2302.07202.pdf.

    Parameters
    ----------
    A : LinearOperator
        The input matrix as a LinearOperator.
    b : (n, 1) np.ndarray
        The target vector.
    S : LinearOperator
        The sketch matrix as a LinearOperator.
    tolerance : float
        Error tolerance. Controls the number of iterations if iter_lim is not specified.
    iter_lim : int, optional
        Maximum number of iterations for least-squares QR solver, by default 100. If specified will overwrite tolerance parameter for error tolerance.
    seed : int, optional
        Random seed for generation of G, by default 42.
    log_x_hat : bool, optional
        Whether to log the intermediate solutions, by default False.
    **kwargs : Any
        Additional required arguments depending on the sketch function.

    Returns
    -------
    x : (d, 1) np.ndarray
        The solution to the least squares problem.
    time_elapsed : float
        Time taken to solve the least squares problem.
    x_hats : List[np.ndarray]
        List of intermediate solutions if log_x_hat is True.
    """
    assert b.ndim == 1, "The target vector should be a vector."
    assert (
        A.shape[0] == b.shape[0]
    ), "The number of rows of the input matrix and the target vector should be the same."
    assert tolerance > 0, "Error tolerance should be greater than 0."
    assert (
        iter_lim is None or iter_lim > 0
    ), "Number of iterations should be greater than 0."
    rng = np.random.default_rng(seed)
    m, n = A.shape
    start_time = time.perf_counter()
    sigma = 10 * LA.norm(A) * np.finfo(float).eps
    G = rng.standard_normal(size=(m, n))
    A_tilde = A + sigma * G / np.sqrt(m)
    x, _, x_hats = _sketch_and_apply(
        A_tilde, b, S, tolerance, iter_lim, log_x_hat=log_x_hat
    )
    end_time = time.perf_counter()
    time_elapsed = end_time - start_time

    return x, time_elapsed, x_hats
