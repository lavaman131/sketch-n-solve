from dataclasses import dataclass
from pathlib import Path
from typing import Optional, Union
import numpy as np
import numpy.linalg as LA
from scipy.stats import ortho_group


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

    # Generate singular values more efficiently
    singular_values = np.logspace(0, -np.log10(cond), k)

    # Use rng directly instead of np.random for consistency
    # Use more efficient matrix generation
    U_small = rng.standard_normal((m, k))
    V_small = rng.standard_normal((n, k))
    U_small, _ = LA.qr(U_small)
    V_small, _ = LA.qr(V_small)

    # Compute A more efficiently using einsum
    A = np.einsum("ik,k,jk->ij", U_small, singular_values, V_small)

    # Generate noise and solution more efficiently
    x = rng.standard_normal(n)
    Ax = A @ x
    r_x = beta * LA.norm(Ax) * rng.standard_normal(m)
    b = Ax + r_x

    problem = {
        "A": A,
        "b": b.squeeze(),
        "x": x.squeeze(),
        "r_x": r_x,
        "cond": cond,
        "beta": beta,
    }
    cond_scientific_notation = f"{cond:.0e}".replace("+", "")
    fname = f"{m}x{n}_{cond_scientific_notation}.npz"

    # Replace h5py with numpy save
    np.savez(save_dir.joinpath(fname), **problem)


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
    rng = np.random.default_rng(seed)

    # Generate orthogonal matrices more efficiently using QR
    U = rng.standard_normal((m, m))
    U, _ = LA.qr(U)
    V = rng.standard_normal((n, n))
    V, _ = LA.qr(V)

    U_1 = U[:, :n]
    U_2 = U[:, n:]

    # Generate Sigma more efficiently
    Sigma = np.diag(np.logspace(1, 1 / cond, n))

    # Compute A more efficiently using einsum
    A = np.einsum("ik,kj,jl->il", U_1, Sigma, V.T)

    # Generate x and noise more efficiently
    w = rng.standard_normal(n)
    z = rng.standard_normal(m - n)
    x = w / LA.norm(w)
    u2z = U_2 @ z
    r_x = beta * u2z / LA.norm(u2z)
    b = A @ x + r_x

    problem = {
        "A": A,
        "b": b.squeeze(),
        "x": x.squeeze(),
        "r_x": r_x,
        "cond": cond,
        "beta": beta,
    }
    cond_scientific_notation = f"{cond:.0e}".replace("+", "")
    fname = f"{m}x{n}_{cond_scientific_notation}.npz"

    # Replace h5py with numpy save
    np.savez(save_dir.joinpath(fname), **problem)
