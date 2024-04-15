from typing import Optional, Tuple
import numpy as np
import scipy.sparse


def clarkson_woodruff(
    A: np.ndarray, k: int, seed: Optional[int] = 42
) -> Tuple[np.ndarray, np.ndarray]:
    """Implements Clarkson-Woodruff sketch as described in https://arxiv.org/pdf/2201.00450.pdf.

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

    T = scipy.sparse.csr_array(
        (np.ones(n), (rng.integers(k, size=n), np.arange(n))), shape=(k, n)
    )
    D = scipy.sparse.diags_array(rng.choice([-1.0, 1.0], n))
    S = T @ D

    return A, S


def sparse_sign(
    A: np.ndarray,
    sparsity_parameter: Optional[int] = None,
    seed: Optional[int] = 42,
) -> Tuple[np.ndarray, np.ndarray]:
    r"""Implements sparse sign sketch as described in https://arxiv.org/pdf/2002.01387.pdf.

    Parameters
    ----------
    A : (n, d) np.ndarray
        The input matrix.
    sparsity_parameter : int, optional
        Recommended to set to :math:`2 \leq \zeta \leq d`. If not provided, it is set to :math:`\zeta = \min\{d, 8\}`.
    seed : int, optional
        Random seed, by default 42.

    Returns
    -------
    A : (n, d) np.ndarray
        The input matrix.
    S : (d, n) np.ndarray
        The sketch matrix.
    """

    n, d = A.shape

    rng = np.random.default_rng(seed)

    if sparsity_parameter:
        zeta = sparsity_parameter
    else:
        zeta = min(d, 8)

    size = zeta * n
    data = rng.choice([-1.0, 1.0], size=size)
    rows = rng.integers(d, size=size)
    cols = np.repeat(np.arange(n), zeta)
    S = scipy.sparse.csc_array((data, (rows, cols)), shape=(d, n))
    S *= 1 / np.sqrt(zeta)
    return A, S
