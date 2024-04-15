import numpy as np
import scipy.linalg as SLA
from sketch_n_solve.sketch import Sketch
from sketch_n_solve.solve.least_squares import LeastSquares
import time


def timer(func):
    def wrapper(*args, **kwargs):
        start = time.perf_counter()
        res = func(*args, **kwargs)
        print(f"Execution time: {time.perf_counter() - start}")
        return res

    return wrapper


if __name__ == "__main__":
    n, d = 4000, 50
    A = np.hstack([np.random.randn(n, d - 1), np.ones((n, 1))])
    coefs = np.random.randn(d, 1)
    b = A @ coefs

    sol1 = timer(SLA.lstsq)(A, b)[0]  # type: ignore
    print(np.allclose(sol1, coefs))
    sketch_fn = "sparse_sign"
    seed = 42
    sketch = Sketch(sketch_fn, seed)
    lsq = LeastSquares(sketch)
    # sol2 = timer(lsq.sketch_and_precondition)(A, b)
    # print(np.allclose(sol2, coefs))
    # sol3 = timer(lsq.iterative_sketching)(A, b, num_iters=1, sparsity_parameter=None)
    # print(np.allclose(sol3, coefs))
