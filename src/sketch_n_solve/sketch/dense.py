from typing import Optional, Tuple
import numpy as np
import scipy
import scipy.linalg
import scipy.sparse
import math


def uniform(
    A: np.ndarray, k: int, seed: Optional[int] = 42
) -> Tuple[np.ndarray, np.ndarray]:
    """Implements uniform sketch as described in https://arxiv.org/pdf/2201.00450.pdf.

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
    A : (n, d) np.ndarray
        The input matrix.
    S : (k, n) np.ndarray
        The sketch matrix.
    """

    n, d = A.shape

    assert k < n, "k should be less than the number of rows of the matrix."
    assert k > 0, "k should be greater than 0."

    rng = np.random.default_rng(seed)

    r = np.arange(k)
    i = rng.integers(n, size=k)
    S = scipy.sparse.csr_matrix((np.ones(k), (r, i)), shape=(k, n))

    scale = np.sqrt(n / k)
    return A, S * scale


def normal(
    A: np.ndarray, k: int, seed: Optional[int] = 42
) -> Tuple[np.ndarray, np.ndarray]:
    """Implements normal sketch as described in https://arxiv.org/pdf/2201.00450.pdf.

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
    A : (n, d) np.ndarray
        The input matrix.
    S : (k, n) np.ndarray
        The sketch matrix.
    """

    n, d = A.shape

    assert k < n, "k should be less than the number of rows of the matrix."
    assert k > 0, "k should be greater than 0."

    rng = np.random.default_rng(seed)

    S = rng.normal(loc=0, scale=(1 / k) ** 0.5, size=(k, n))

    return A, S


def hadamard(
    A: np.ndarray, k: int, seed: Optional[int] = 42
) -> Tuple[np.ndarray, np.ndarray]:
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
    S : (k, n_padded) np.ndarray
        The sketch matrix where n_padded = :math:`2^{\lceil \log_2 n \rceil}`.
    """

    n, d = A.shape
    assert k < n, "k should be less than the number of rows of the matrix."
    assert k > 0, "k should be greater than 0."
    rng = np.random.default_rng(seed)
    n_padded = 2 ** math.ceil(np.log2(n))
    A_padded = np.pad(A, ((0, n_padded - n), (0, 0)), "constant")
    H = scipy.linalg.hadamard(n=n_padded, dtype=np.float64)  # type: ignore
    D = np.diag(np.random.choice([-1, 1], size=n_padded))
    r = rng.integers(n_padded, size=k)
    S = H[r] @ H @ D / np.sqrt(k)
    return A_padded, S
