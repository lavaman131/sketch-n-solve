from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Optional, Union
import numpy as np
import scipy.linalg as SLA
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
    x = w / SLA.norm(w)
    u2z = U_2 @ z
    r_x = beta * u2z / SLA.norm(u2z)
    b = A @ x + r_x

    problem = {"A": A, "b": b, "x": x, "r_x": r_x, "cond": cond, "beta": beta}
    cond_scientific_notation = f"{cond:.0e}".replace("+", "")
    fname = f"{m}x{n}_{cond_scientific_notation}.h5"
    with h5py.File(save_dir.joinpath(fname), "w") as f:
        for key, value in problem.items():
            f.create_dataset(key, data=value)
