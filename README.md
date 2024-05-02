# Sketch 'n Solve

Sketch 'n Solve is a Python library that implements basic randomized numerical linear algebra (RandNLA) techniques for solving large-scale linear systems of equations. The library is designed to be user-friendly and easy to use, with a focus on simplicity and efficiency. The library is built on top of NumPy and SciPy, and provides a simple interface for solving linear systems using randomized algorithms.

<p align="center">
    <img src="./assets/logo.jpg" style="width: 40%">
</p>

# 🚀 Getting Started 

## 📦 Installation

```bash
python -m venv .venv
source .venv/bin/activate
python -m pip install poetry
poetry install
```

## 🛠️ Usage

```python
import numpy as np
import numpy.linalg as LA
from sketch_n_solve.solve.least_squares import LeastSquares

sketch_fn = "sparse_sign"
seed = 42
A = np.random.randn(10000, 10)
x = np.random.randn(10, 1)
b = A @ x
lsq = LeastSquares(sketch_fn, seed)

print(lsq.sketch_and_apply.__doc__)

x_hat, time_elapsed, x_hats, istop = lsq.sketch_and_apply(A, b, sparsity_parameter=None)

print(x, x_hat)
print("residual", LA.norm(A @ x_hat - A @ x))

is_close = np.allclose(x_hat, x)
print(f"x_hat is close to x => {is_close}")

assert is_close, "Something went wrong!"
```
