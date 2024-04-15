from typing import Optional, Tuple
import numpy as np
import scipy
import scipy.linalg


def uniform(
    A: np.ndarray, k: int, seed: Optional[int] = 42
) -> Tuple[np.ndarray, np.ndarray]:
    """Implements uniform sketch.

    Parameters
    ----------
    A : np.ndarray
        The input matrix.
    k : int
        The number of rows in the sketch matrix.
    seed : int, optional
        Random seed, by default 42.

    Returns
    -------
    A : np.ndarray
        The input matrix.
    S : np.ndarray
        The sketch matrix.
    """

    n, d = A.shape

    assert k < n, "k should be less than the number of rows of the matrix."
    assert k > 0, "k should be greater than 0."

    rng = np.random.default_rng(seed)

    S = np.zeros((k, n))

    r = np.arange(k)
    i = rng.integers(n, size=k)
    S[r] = A[i].copy()
    S[r, i] = 1.0

    scale = np.sqrt(n / k)
    return A, S * scale


def normal(
    A: np.ndarray, k: int, seed: Optional[int] = 42
) -> Tuple[np.ndarray, np.ndarray]:
    """Implements normal sketch.

    Parameters
    ----------
    A : np.ndarray
        The input matrix.
    k : int
        The number of rows in the sketch matrix.
    seed : int, optional
        Random seed, by default 42.

    Returns
    -------
    A : np.ndarray
        The input matrix.
    S : np.ndarray
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
    """Implements Hadamard sketch.

    Parameters
    ----------
    A : np.ndarray
        The input matrix.
    k : int
        The number of rows in the sketch matrix.
    seed : int, optional
        Random seed, by default 42.

    Returns
    -------
    A_padded : np.ndarray
        The input matrix padded to the nearest power of 2 with zeros.
    S : np.ndarray
        The sketch matrix.
    """

    n, d = A.shape
    assert k < n, "k should be less than the number of rows of the matrix."
    assert k > 0, "k should be greater than 0."
    rng = np.random.default_rng(seed)
    padded_n = 2 ** round(np.log2(n))
    A_padded = np.pad(A, ((0, padded_n - n), (0, 0)), "constant")
    H = scipy.linalg.hadamard(n=padded_n, dtype=np.float64)  # type: ignore
    D = np.diag(np.random.choice([-1, 1], size=padded_n))
    r = rng.integers(padded_n, size=k)
    S = H[r] @ H @ D / np.sqrt(k)
    return A_padded, S
