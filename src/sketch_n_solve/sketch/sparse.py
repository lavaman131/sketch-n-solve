from typing import Any, Optional, Tuple
import numpy as np
from scipy.sparse import csr_array, csc_array, diags_array
from scipy.sparse.linalg import LinearOperator, aslinearoperator
from scipy.linalg._sketches import cwt_matrix


def uniform_sparse(
    A: np.ndarray, k: Optional[int] = None, seed: Optional[int] = 42, **kwargs: Any
) -> Tuple[np.ndarray, LinearOperator]:
    """Implements uniform sketch as described in https://arxiv.org/pdf/2201.00450.pdf.

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
        k = n * 2

    rng = np.random.default_rng(seed)

    r = np.arange(k)
    i = rng.integers(m, size=k)
    scale = np.sqrt(m / k)
    S = csr_array((np.ones(k) * scale, (r, i)), shape=(k, m))
    return A, aslinearoperator(S)


def clarkson_woodruff(
    A: np.ndarray, k: Optional[int] = None, seed: Optional[int] = 42, **kwargs: Any
) -> Tuple[np.ndarray, LinearOperator]:
    """Implements Clarkson-Woodruff sketch as described in https://arxiv.org/pdf/2201.00450.pdf.

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
    S : (k, m) np.ndarray
        The sketch matrix.
    """

    m, n = A.shape

    if k:
        assert k < m, "k should be less than the number of rows of the matrix."
        assert k > 0, "k should be greater than 0."
    else:
        k = n

    rng = np.random.default_rng(seed)

    S = cwt_matrix(n, m, rng)

    return A, aslinearoperator(S)


def sparse_sign(
    A: np.ndarray,
    sparsity_parameter: Optional[int] = None,
    seed: Optional[int] = 42,
    **kwargs: Any,
) -> Tuple[np.ndarray, LinearOperator]:
    r"""Implements sparse sign sketch as described in https://arxiv.org/pdf/2002.01387.pdf.

    Parameters
    ----------
    A : (m, n) np.ndarray
        The input matrix.
    sparsity_parameter : int, optional
        Recommended to set to :math:`2 \leq \zeta \leq d`. If not provided, it is set to :math:`\zeta = \min\{d, 8\}`.
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
    d = 20 * n

    rng = np.random.default_rng(seed)

    if sparsity_parameter:
        zeta = sparsity_parameter
    else:
        zeta = 8

    size = zeta * m
    data = rng.choice([-1 / np.sqrt(zeta), 1 / np.sqrt(zeta)], size=size)
    rows = rng.integers(d, size=size)
    cols = np.repeat(np.arange(m), zeta)
    S = csc_array((data, (rows, cols)), shape=(d, m))
    return A, aslinearoperator(S)
