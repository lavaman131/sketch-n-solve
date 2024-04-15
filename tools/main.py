import numpy as np
import scipy.linalg as SLA
from sketch_n_solve.metrics import calculate_least_squares_metrics
from sketch_n_solve.sketch import Sketch
from sketch_n_solve.solve.least_squares import LeastSquares
from sketch_n_solve.utils import timer
from sketch_n_solve.utils.least_squares import generate_least_squares_problem

if __name__ == "__main__":
    n, d = 4000, 50
    cond = 10**10
    beta = 10**-6
    A, b, x = generate_least_squares_problem(m=n, n=d, cond=cond, beta=beta)

    x_hat = timer(SLA.lstsq)(A, b)[0]  # type: ignore
    metrics = calculate_least_squares_metrics(A, b, x, x_hat)
    print(metrics)
    # sketch_fn = "sparse_sign"
    # seed = 42
    # sketch = Sketch(sketch_fn, seed)
    # lsq = LeastSquares(sketch)
    # x_hat2 = timer(lsq.sketch_and_precondition)(A, b)
    # x_hat3 = timer(lsq.iterative_sketching)(A, b, num_iters=1, sparsity_parameter=None)
