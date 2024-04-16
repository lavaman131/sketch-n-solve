from dataclasses import dataclass
from typing import Dict, Optional
import numpy as np
import scipy.linalg as SLA
from scipy.stats import ortho_group


@dataclass
class LeastSquaresProblemConfig:
    m: int
    n: int
    cond: float = 1e10
    beta: float = 1e-12


def generate_least_squares_problem(
    m: int, n: int, cond: float, beta: float, seed: Optional[int] = 42
) -> Dict[str, np.ndarray]:
    """Generate a least squares problem.

    Parameters
    ----------
    m : int
        number of rows of matrix A
    n : int
        number of columns of matrix A
    cond : float
        condition number of matrix A
    beta : float
        noise level
    seed : int, optional
        random seed, by default 42

    Returns
    -------
    problem : Dict[str, np.ndarray]
        matrix A, vector b, vector x, and noise vector r_x
    """
    assert cond >= 1, "Condition number must be greater than or equal to 1."
    assert beta >= 0, "Noise level must be greater than or equal to 0."
    m, n = max(m, n), min(m, n)
    U = ortho_group.rvs(m, random_state=seed)
    U_1 = U[:, :n]
    U_2 = U[:, n:]
    V = ortho_group.rvs(n, random_state=seed)
    Sigma = np.diag(np.logspace(1, 1 / cond, n))
    A = U_1 @ Sigma @ V.T

    rng = np.random.default_rng(seed)
    w = rng.standard_normal((n, 1))
    z = rng.standard_normal((m - n, 1))
    x = w / SLA.norm(w)
    u2z = U_2 @ z
    r_x = beta * u2z / SLA.norm(u2z)
    b = A @ x + r_x

    problem = {"A": A, "b": b, "x": x, "r_x": r_x}
    return problem
