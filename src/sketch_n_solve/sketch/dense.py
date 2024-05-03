from typing import Any, Optional, Tuple
import numpy as np
import scipy.linalg as SLA
import math
from scipy.sparse.linalg import LinearOperator, aslinearoperator


def uniform_dense(
    A: np.ndarray, k: Optional[int] = None, seed: Optional[int] = 42, **kwargs: Any
) -> Tuple[np.ndarray, LinearOperator]:
    """Implements uniform sketch as described in https://arxiv.org/pdf/2302.11474.pdf.

    Parameters
    ----------
    A : (m, n) np.ndarray
        The input matrix.
    k : int, optional
        The number of rows in the sketch matrix. If not provided, it is set to number of columns of A.
    seed : int, optional
        Random seed, by default 42.

    Returns
    -------
    A : (m, n) np.ndarray
        The input matrix.
    S : (d, m) np.ndarray
        The sketch matrix.
    """

    m, n = A.shape

    if k:
        assert k < m, "k should be less than the number of rows of the matrix."
        assert k > 0, "k should be greater than 0."
    else:
        k = n

    rng = np.random.default_rng(seed)

    S = rng.uniform(-1, 1, size=(k, m))

    return A, aslinearoperator(S)


def normal(
    A: np.ndarray, k: Optional[int] = None, seed: Optional[int] = 42, **kwargs: Any
) -> Tuple[np.ndarray, LinearOperator]:
    """Implements normal sketch as described in https://arxiv.org/pdf/2201.00450.pdf.

    Parameters
    ----------
    A : (m, n) np.ndarray
        The input matrix.
    k : int, optional
        The number of rows in the sketch matrix. If not provided, it is set to number of columns of A.
    seed : int, optional
        Random seed, by default 42.

    Returns
    -------
    A : (m, n) np.ndarray
        The input matrix.
    S : (d, m) np.ndarray
        The sketch matrix.
    """

    m, n = A.shape

    if k:
        assert k < m, "k should be less than the number of rows of the matrix."
        assert k > 0, "k should be greater than 0."
    else:
        k = n

    rng = np.random.default_rng(seed)

    S = rng.normal(loc=0, scale=(1 / k) ** 0.5, size=(k, m))

    return A, aslinearoperator(S)


def hadamard(
    A: np.ndarray, k: int = 500, seed: Optional[int] = 42, **kwargs: Any
) -> Tuple[np.ndarray, LinearOperator]:
    r"""Implements Hadamard sketch as described in https://arxiv.org/pdf/2201.00450.pdf.

    Parameters
    ----------
    A : (n, d) np.ndarray
        The input matrix.
    k : int
        The number of rows in the sketch matrix.
    seed : int, optional
        Random seed, by default 42.

    Returns
    -------
    A_padded (n_padded, d) : np.ndarray
        The input matrix padded to the nearest power of 2 with zeros where n_padded = :math:`2^{\lceil \log_2 n \rceil}`.
    S : (d, n_padded) np.ndarray
        The sketch matrix where n_padded = :math:`2^{\lceil \log_2 n \rceil}`.
    """

    n, d = A.shape
    assert k < n, "k should be less than the number of rows of the matrix."
    assert k > 0, "k should be greater than 0."
    rng = np.random.default_rng(seed)
    n_padded = 2 ** math.ceil(np.log2(n))
    A_padded = np.pad(A, ((0, n_padded - n), (0, 0)), "constant")
    H = SLA.hadamard(n=n_padded, dtype=np.float64)  # type: ignore
    D = np.diag(np.random.choice([-1, 1], size=n_padded))
    r = rng.integers(n_padded, size=k)
    S = H[r] @ H @ D / np.sqrt(k)
    return A_padded, aslinearoperator(S)
