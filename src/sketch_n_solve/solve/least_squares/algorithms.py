from typing import Optional, Union
import numpy as np
import scipy.sparse
import scipy.sparse.linalg as SPLA
import scipy.linalg as SLA
import scipy.special
import numpy.linalg as LA


def _sketch_and_precondition(
    A: np.ndarray,
    b: np.ndarray,
    S: np.ndarray,
    q: Optional[int] = None,
) -> np.ndarray:
    """Solves the least squares problem using sketch and preconditioning as described in https://arxiv.org/pdf/2311.04362.pdf.

    Parameters
    ----------
    A : (n, d) np.ndarray
        The input matrix.
    b : (n, 1) np.ndarray
        The target vector.
    S : np.ndarray
        The sketch matrix.
    q : int, optional
        Maximum number of iterations for least-squares QR solver, by default None.
    **kwargs : Any
        Additional required arguments depending on the sketch function.

    Returns
    -------
    x_i : (d, 1) np.ndarray
        The solution to the least squares problem.
    """
    assert b.ndim == 2, "The target vector should be a column vector."
    assert (
        A.shape[0] == b.shape[0]
    ), "The number of rows of the input matrix and the target vector should be the same."
    B = S @ A
    Q, R = SLA.qr(B, mode="economic", pivoting=False)  # type: ignore
    R_inv = SLA.inv(R)
    x_0 = R_inv @ Q.T @ S @ b  # type: ignore
    A_precond = A @ R_inv  # right preconditioning
    y = SPLA.lsqr(A_precond, b, x0=x_0.squeeze(), iter_lim=q)[0]
    x_i = SLA.solve_triangular(
        R, y, lower=False
    )  # solve the original least squares problem
    return x_i.reshape(-1, 1)


# def _iterative_sketching(
#     A: np.ndarray,
#     b: np.ndarray,
#     S: np.ndarray,
#     d: Optional[int] = None,
#     q: np.float_ = np.finfo(float).eps,
#     damping: Union[str, float] = "optimal",
#     momentum: Union[str, float] = "optimal",
# ) -> np.ndarray:
#     """Solves the least squares problem using iterative sketching.

#     Parameters
#     ----------
#     A : (n, d) np.ndarray
#         The input matrix.
#     b : (n, 1) np.ndarray
#         The target vector.
#     S : np.ndarray
#         The sketch matrix.
#     d : int, optional
#         Sketching dimension (default values are described in the paper).
#     q : np.float_, optional
#         Number of steps or tolerance. If q >= 1, run for q steps. If q < 1,
#         run until the norm change in residual is less than
#         q*(Anorm * norm(x) + 0.01*Acond*norm(r)), where Anorm and Acond are
#         estimates of the norm and condition number of A. Defaults to eps.
#     damping : Union[str, float], optional
#         Damping coefficient (default 'optimal'). If 'optimal', the optimal coefficient
#         will be computed as a function of d and n.
#     momentum : Union[str, float], optional
#         Momentum coefficient (default 'optimal'). If 'optimal', the optimal coefficient
#         will be computed as a function of d and n.
#     **kwargs : Any
#         Additional required arguments depending on the sketch function.

#     Returns
#     -------
#     x : (d, 1) np.ndarray
#         The solution to the least squares problem.
#     """
#     m, n = A.shape

#     if d is None:
#         if scipy.sparse.issparse(A):
#             d = 20 * n
#         else:
#             if damping == "optimal":
#                 if momentum == "optimal":
#                     C = 1
#                 else:
#                     C = np.sqrt(2)
#                 min_ratio = 4
#             else:
#                 C = 2 + np.sqrt(2)
#                 min_ratio = 20
#             d = int(
#                 np.maximum(
#                     np.ceil(
#                         C**2
#                         * n
#                         * np.exp(
#                             np.real(
#                                 scipy.special.lambertw(
#                                     4
#                                     * m
#                                     / n**2
#                                     * np.log(1 / np.finfo(float).eps)
#                                     / C**2,
#                                     k=0,
#                                 )
#                             )
#                         )
#                     ),
#                     min_ratio * n,
#                 )
#             )

#     if damping == "optimal":
#         assert d, "d must be specified for optimal damping"
#         if momentum == "optimal":
#             momentum = n / d
#             damping = (1 - momentum) ** 2
#         elif momentum == 0:
#             r = n / d
#             damping = (1 - r) ** 2 / (1 + r)
#         else:
#             raise ValueError("Optimal damping for nonzero momentum is not implemented")

#     B = S @ A
#     Q, R = SLA.qr(B, mode="economic", pivoting=False)  # type: ignore
#     x = SLA.solve_triangular(R, Q.T @ S @ b)  # type: ignore
#     x_prev = x
#     r_prev = b - A @ x

#     A_cond = LA.cond(R)  # Estimating the condition number of A using R
#     if q < 1:
#         z = np.random.randn(n, 1)
#         for _ in range(int(np.ceil(np.log(n)))):
#             z = z / SLA.norm(z)
#             z = SLA.solve_triangular(R.T, z, lower=True)
#             z = z / SLA.norm(z)
#             z = R @ z
#         A_norm = SLA.norm(z)

#     resest = SLA.norm(r_prev) / SLA.norm(b)
#     if A_cond >= 5e-3 / np.finfo(float).eps and resest >= np.sqrt(np.finfo(float).eps):
#         print(
#             f"Condition number (est = {A_cond:.1e}) and relative residual (est = {resest:.1e}) both appear to be large"
#         )

#     iter = 1
#     while True:
#         x_copy = x
#         y = SLA.solve_triangular(R.T, A.T @ r_prev, lower=True)
#         d_i = SLA.solve_triangular(R, y, lower=False)
#         x = x + damping * d_i + momentum * (x - x_prev)
#         x_prev = x_copy
#         r = b - A @ x
#         updated_norm = SLA.norm(r - r_prev)
#         r_prev = r
#         iter += 1
#         if (q >= 1 and iter > q) or (
#             q < 1
#             and updated_norm <= q * (A_norm * SLA.norm(x) + 0.01 * A_cond * SLA.norm(r))
#         ):
#             break
#         elif q < 1 and iter >= 100:
#             print("Iterative sketching failed to meet tolerance")
#             break

#     return x


def _iterative_sketching(
    A: np.ndarray,
    b: np.ndarray,
    S: np.ndarray,
    delta: float = 1e-10,
    num_iters: Optional[int] = None,
) -> np.ndarray:
    """Solves the least squares problem using modified iterative sketching than described in https://arxiv.org/pdf/2311.04362.pdf.

    Parameters
    ----------
    A : (n, d) np.ndarray
        The input matrix.
    b : (n, 1) np.ndarray
        The target vector.
    S : np.ndarray
        The sketch matrix.
    delta : float
        Error tolerance. Controls the number of iterations if num_iters is not specified.
    num_iters : int, optional
        Maximum number of iterations for least-squares QR solver, by default None. If specified will overwrite delta parameter for error tolerance.
    **kwargs : Any
        Additional required arguments depending on the sketch function.

    Returns
    -------
    x_i : (d, 1) np.ndarray
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

    if num_iters:
        q = num_iters
    else:
        q = int(np.ceil(np.log(1 / delta)))

    B = S @ A
    Q, R = SLA.qr(B, mode="economic", pivoting=False)  # type: ignore
    x_i = SLA.solve_triangular(R, Q.T @ S @ b, lower=False)  # type: ignore

    for _ in range(q):
        r_i = b - A @ x_i
        d_i = SLA.solve_triangular(R, Q.T @ S @ r_i, lower=False)  # type: ignore
        x_i = x_i + d_i

    return x_i
