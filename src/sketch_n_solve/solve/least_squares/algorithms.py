from typing import Optional
import numpy as np
import scipy.sparse.linalg as SPLA
import scipy.linalg as SLA


def _sketch_and_precondition(
    A: np.ndarray,
    b: np.ndarray,
    S: np.ndarray,
    use_sketch_and_solve_x_0: bool = True,
    delta: float = 1e-12,
    num_iters: Optional[int] = None,
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
    delta : float, optional
        Error tolerance. Controls the number of iterations if num_iters is not specified, by default 1e-12.
    num_iters : int, optional
        Maximum number of iterations for least-squares QR solver, by default None.
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
    R_inv, _ = SLA.lapack.dtrtri(R)
    x_0 = (R_inv @ Q.T @ S @ b).squeeze() if use_sketch_and_solve_x_0 else None  # type: ignore
    A_precond = A @ R_inv  # right preconditioning
    y = SPLA.lsqr(
        A_precond,
        b,
        x0=x_0,
        iter_lim=num_iters,
        atol=delta,
        btol=delta,
    )[0]
    x = SLA.solve_triangular(
        R, y, lower=False
    )  # solve the original least squares problem
    return x.reshape(-1, 1)


def _sketch_and_apply(
    A: np.ndarray,
    b: np.ndarray,
    S: np.ndarray,
    delta: float = 1e-12,
    num_iters: Optional[int] = None,
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
    delta : float, optional
        Error tolerance. Controls the number of iterations if num_iters is not specified, by default 1e-12.
    num_iters : int, optional
        Maximum number of iterations for least-squares QR solver, by default None.
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
    assert delta > 0, "Error tolerance should be greater than 0."
    assert (
        num_iters is None or num_iters > 0
    ), "Number of iterations should be greater than 0."

    B = S @ A
    Q, R = SLA.qr(B, mode="economic", pivoting=False)  # type: ignore
    z_0 = Q.T @ S @ b  # type: ignore
    z = SPLA.lsqr(
        Q,
        S @ b,
        x0=z_0.squeeze(),
        iter_lim=num_iters,
        atol=delta,
        btol=delta,
    )[0]
    x = SLA.solve_triangular(R, z, lower=False)

    return x.reshape(-1, 1)


def _smoothed_sketch_and_apply(
    A: np.ndarray,
    b: np.ndarray,
    G: np.ndarray,
    delta: float = 1e-12,
    num_iters: Optional[int] = None,
) -> np.ndarray:
    """Solves the least squares problem using smoothed sketch-and-apply as described in https://arxiv.org/pdf/2302.07202.pdf.

    Parameters
    ----------
    A : (n, d) np.ndarray
        The input matrix.
    b : (n, 1) np.ndarray
        The target vector.
    G : np.ndarray
        The sketch matrix, i.e., standard Gaussian matrix.
    delta : float
        Error tolerance. Controls the number of iterations if num_iters is not specified.
    num_iters : int, optional
        Maximum number of iterations for least-squares QR solver, by default None. If specified will overwrite delta parameter for error tolerance.
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
    assert delta > 0, "Error tolerance should be greater than 0."
    assert (
        num_iters is None or num_iters > 0
    ), "Number of iterations should be greater than 0."
    m, n = A.shape
    sigma = 10 * SLA.norm(A) * np.finfo(float).eps
    A_tilde = A + sigma * G / np.sqrt(m)
    x = _sketch_and_apply(A_tilde, b, G, delta, num_iters)
    return x.reshape(-1, 1)
