import numpy as np
import numpy.linalg as LA
from typing import List, Optional, Tuple


def check_convergence(
    tolerance: float, A: np.ndarray, x: np.ndarray, b: np.ndarray
) -> bool:
    b_hat = A @ x
    r = b - b_hat
    cond_1 = LA.norm(A.T @ r) / (LA.norm(A) * LA.norm(r)) <= tolerance
    cond_2 = LA.norm(r) / LA.norm(b) <= tolerance
    return cond_1 and cond_2  # type: ignore


def lsqr(
    A: np.ndarray,
    b: np.ndarray,
    tol: float = 1e-6,
    iter_lim: Optional[int] = None,
    x0: Optional[np.ndarray] = None,
    log_x_hat: bool = False,
) -> Tuple[np.ndarray, List[np.ndarray]]:
    m, n = A.shape
    if x0 is None:
        x = np.zeros((n, 1))
    else:
        x = x0.reshape(-1, 1)

    b = np.atleast_1d(b)
    if b.ndim == 1:
        b = b.reshape(-1, 1)

    x_hats = []
    if log_x_hat:
        x_hats.append(x.copy())

    b = b - A @ x
    bnorm = LA.norm(b)
    beta = bnorm
    u = b / beta
    v = A.T @ u
    alpha = LA.norm(v)
    v = v / alpha
    w = v.copy()
    phibar = beta
    rhobar = alpha
    resnorm = bnorm

    if iter_lim is None:
        iter_lim = 2 * n

    for _ in range(iter_lim):
        u = A @ v - alpha * u
        beta = LA.norm(u)
        u = u / beta
        v = A.T @ u - beta * v
        alpha = LA.norm(v)
        v = v / alpha
        rho = np.sqrt(rhobar**2 + beta**2)
        c = rhobar / rho
        s = beta / rho
        theta = s * alpha
        rhobar = -c * alpha
        phi = c * phibar
        phibar = s * phibar
        x += (phi / rho) * w
        w = v - (theta / rho) * w

        if log_x_hat:
            x_hats.append(x.copy())

        if check_convergence(tol, A, x, b):
            break

    return x, x_hats
