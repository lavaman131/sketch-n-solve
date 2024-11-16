import time
import numpy as np
import numpy.linalg as LA
from sketch_n_solve.solve.least_squares import LeastSquares
from scipy.sparse.linalg import lsqr


def main() -> None:
    # Recommended to use either "clarkson_woodruff" or "uniform_sparse"
    # for best results out of the box
    sketch_fn = "clarkson_woodruff"
    seed = 42
    # so lqsr assertion passes
    atol = 0.1
    rng = np.random.default_rng(seed)
    A = rng.standard_normal((1000000, 1000))
    x = rng.standard_normal(1000)
    b = A @ x
    lsq = LeastSquares(sketch_fn, seed)
    
    x_hat, _, istop, time_elapsed = lsq(A, b)

    print("residual", LA.norm(A @ x_hat - A @ x))

    is_close = np.allclose(x_hat, x, atol=atol)
    assert is_close, "x_hat is not close to x"

    print(f"Sketch and solve finished in {time_elapsed} seconds.")
    
    # Now let's try the same thing with the lsqr function
    start_time = time.perf_counter()
    x_hat, istop, *_ = lsqr(A, b)
    end_time = time.perf_counter()
    time_elapsed = end_time - start_time
    
    print("residual", LA.norm(A @ x_hat - A @ x))
    
    is_close = np.allclose(x_hat, x, atol=atol)
    assert is_close, "x_hat is not close to x"
    
    print(f"lsqr finished in {time_elapsed} seconds.")


if __name__ == "__main__":
    main()
