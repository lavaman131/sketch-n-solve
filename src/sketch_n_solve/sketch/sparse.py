from typing import Optional, Tuple
import numpy as np
import scipy.sparse


def clarkson_woodruff(
    A: np.ndarray, k: int, seed: Optional[int] = 42
) -> Tuple[np.ndarray, np.ndarray]:
    """Implements Clarkson-Woodruff sketch.

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

    T = scipy.sparse.csr_array(
        (np.ones(n), (rng.integers(k, size=n), np.arange(n))), shape=(k, n)
    )
    D = scipy.sparse.diags_array(rng.choice([-1, 1], n))
    S = T @ D

    return A, S
