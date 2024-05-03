import numpy as np
import numpy.linalg as LA
from sketch_n_solve.solve.least_squares import LeastSquares
from sketch_n_solve.solve.least_squares.utils import lsqr

sketch_fn = "clarkson_woodruff"
seed = 42
rng = np.random.default_rng(seed)
A = rng.standard_normal((1000000, 1000))
x = rng.standard_normal(1000)
b = A @ x
lsq = LeastSquares(sketch_fn, seed)

x_hat, time_elapsed, x_hats, istop = lsq.sketch_and_apply(A, b)

print("residual", LA.norm(A @ x_hat - A @ x))

is_close = np.allclose(x_hat, x)
print(f"x_hat is close to x => {is_close}")

assert is_close, "Something went wrong!"

print(f"Finished in {time_elapsed} seconds.")
