import time
from typing import Any, List, Optional, Tuple
import numpy as np
import numpy.linalg as LA

from scipy.linalg.lapack import dtrtrs as triangular_solve
from sketch_n_solve.solve.least_squares.utils import lsqr
from scipy.sparse.linalg import LinearOperator, aslinearoperator
from scipy.linalg import lstsq


def _sketch_and_precondition(
    A: np.ndarray,
    b: np.ndarray,
    S: LinearOperator,
    use_sketch_and_solve_x_0: bool = True,
    tolerance: float = 1e-12,
    iter_lim: Optional[int] = 100,
    log_x_hat: bool = False,
    **kwargs: Any,
) -> Tuple[np.ndarray, float, List[np.ndarray], int]:
    # ...
    assert (
        iter_lim is None or iter_lim > 0
    ), "Number of iterations should be greater than 0."

    start_time = time.perf_counter()
    B = S.matmat(A)
    c = S.matvec(b)
    Q, R = LA.qr(B)

    Q_tranpose = aslinearoperator(Q.transpose())

    R_inv = aslinearoperator(triangular_solve(R, np.eye(R.shape[1]), lower=False)[0])

    if use_sketch_and_solve_x_0:
        x_0 = R_inv.matvec(Q_tranpose.matvec(c))
    else:
        x_0 = None

    y, y_hats, istop, *_ = lsqr(
        A=A @ R_inv,
        b=b,
        x0=x_0,
        atol=tolerance,
        btol=tolerance,
        iter_lim=iter_lim,
        log_x_hat=log_x_hat,
    )
    x = R_inv.matvec(y)

    end_time = time.perf_counter()
    time_elapsed = end_time - start_time

    x_hats = []
    if log_x_hat:
        x_hats = [R_inv.matvec(y_hat) for y_hat in y_hats]

    return x, time_elapsed, x_hats, istop


def solve(A: np.ndarray, b: np.ndarray) -> np.ndarray:
    """Solves the least squares problem using the normal equation.

    Parameters
    ----------
    A : (m, n) np.ndarray
        The input matrix.
    b : (n, 1) np.ndarray
        The target vector.

    Returns
    -------
    x : (d, 1) np.ndarray
        The solution to the least squares problem.
    """
    if A.shape[0] == A.shape[1]:
        x = triangular_solve(A, b, lower=False)[0]
    else:
        x = lstsq(A, b)[0]
    return x


def _sketch_and_apply(
    A: np.ndarray,
    b: np.ndarray,
    S: LinearOperator,
    tolerance: float = 1e-12,
    iter_lim: Optional[int] = 100,
    log_x_hat: bool = False,
    **kwargs: Any,
) -> Tuple[np.ndarray, float, List[np.ndarray], int]:
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
    Q, R = LA.qr(B)
    z_0 = (Q.T @ c).squeeze()  # type: ignore
    z, z_hats, istop, *_ = lsqr(
        A=Q,  # type: ignore
        b=c,
        x0=z_0,
        atol=tolerance,
        btol=tolerance,
        iter_lim=iter_lim,
        log_x_hat=log_x_hat,
    )
    x = solve(R, z)
    end_time = time.perf_counter()

    time_elapsed = end_time - start_time
    x_hats = []
    if log_x_hat:
        x_hats = [solve(R, z_hat) for z_hat in z_hats]

    return x, time_elapsed, x_hats, istop


def _smoothed_sketch_and_apply(
    A: np.ndarray,
    b: np.ndarray,
    S: LinearOperator,
    tolerance: float = 1e-12,
    iter_lim: Optional[int] = 100,
    seed: Optional[int] = 42,
    log_x_hat: bool = False,
    **kwargs: Any,
) -> Tuple[np.ndarray, float, List[np.ndarray], int]:
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
    x, time_elapsed, x_hats, istop = _sketch_and_apply(
        A_tilde, b, S, tolerance, iter_lim, log_x_hat=log_x_hat
    )
    end_time = time.perf_counter()
    time_elapsed = end_time - start_time

    return x, time_elapsed, x_hats, istop
