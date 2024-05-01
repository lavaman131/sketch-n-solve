from dataclasses import dataclass
from pathlib import Path
from typing import Optional, Union
import numpy as np
import numpy.linalg as LA
from scipy.stats import ortho_group
import h5py


@dataclass
class LeastSquaresProblemConfig:
    m: int
    n: int
    cond: float = 1e10
    beta: float = 1e-10


def generate_least_squares_problem(
    m: int,
    n: int,
    cond: float,
    beta: float,
    save_dir: Union[str, Path],
    seed: Optional[int] = 42,
) -> None:
    """
    Generate a random least squares problem with dimensions m x n,
    condition number `cond`, and noise level `beta`.
    """
    assert cond >= 1, "Condition number must be greater than or equal to 1."
    assert beta >= 0, "Noise level must be greater than or equal to 0."

    save_dir = Path(save_dir) if isinstance(save_dir, str) else save_dir
    save_dir.mkdir(exist_ok=True, parents=True)

    rng = np.random.default_rng(seed)
    k = min(m, n)

    # Generate singular values
    singular_values = np.logspace(0, -np.log10(cond), k)

    # Generate random orthogonal matrices U_small and V_small
    U_small, _ = LA.qr(np.random.randn(m, k))
    V_small, _ = LA.qr(np.random.randn(n, k))

    # Construct the diagonal matrix Sigma with the singular values
    Sigma = np.zeros((k, k))
    np.fill_diagonal(Sigma, singular_values)

    # Compute the matrix A
    A = U_small @ Sigma @ V_small.T

    # Generate noise and true solution
    x = rng.standard_normal(n)
    r_x = beta * LA.norm(A @ x) * rng.standard_normal(m)
    b = A @ x + r_x

    problem = {"A": A, "b": b, "x": x, "r_x": r_x, "cond": cond, "beta": beta}
    cond_scientific_notation = f"{cond:.0e}".replace("+", "")
    fname = f"{m}x{n}_{cond_scientific_notation}.h5"

    with h5py.File(save_dir.joinpath(fname), "w") as f:
        for key, value in problem.items():
            f.create_dataset(key, data=value)


def generate_ortho_least_squares_problem(
    m: int,
    n: int,
    cond: float,
    beta: float,
    save_dir: Union[str, Path],
    seed: Optional[int] = 42,
) -> None:
    r"""Generate a least squares problem.

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
    save_dir : Union[str, Path]
        directory to save the problem where `problem` is:
            problem : Dict[str, np.ndarray]
                matrix A, vector b, vector x, and noise vector r_x
    seed : int, optional
        random seed, by default 42
    """
    assert cond >= 1, "Condition number must be greater than or equal to 1."
    assert beta >= 0, "Noise level must be greater than or equal to 0."
    save_dir = Path(save_dir) if isinstance(save_dir, str) else save_dir
    save_dir.mkdir(exist_ok=True, parents=True)
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
    x = w / LA.norm(w)
    u2z = U_2 @ z
    r_x = beta * u2z / LA.norm(u2z)
    b = A @ x + r_x

    problem = {"A": A, "b": b, "x": x, "r_x": r_x, "cond": cond, "beta": beta}
    cond_scientific_notation = f"{cond:.0e}".replace("+", "")
    fname = f"{m}x{n}_{cond_scientific_notation}.h5"
    with h5py.File(save_dir.joinpath(fname), "w") as f:
        for key, value in problem.items():
            f.create_dataset(key, data=value)
