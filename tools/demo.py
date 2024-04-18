import numpy as np

from sketch_n_solve.solve.least_squares import LeastSquares

sketch_fn = "sparse_sign"
seed = 42
A = np.random.randn(10000, 10)
x = np.random.randn(10, 1)
b = A @ x
lsq = LeastSquares(sketch_fn, seed)

print(lsq.sketch_and_apply.__doc__)

x_hat, time_elapsed, x_hats = lsq.sketch_and_apply(A, b, sparsity_parameter=None)

is_close = np.allclose(x_hat, x)
print(f"x_hat is close to x => {is_close}")

assert is_close, "Something went wrong!"
