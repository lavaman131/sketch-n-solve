# Sketch 'n Solve

Sketch 'n Solve is a Python library that implements basic randomized numerical linear algebra (RandNLA) techniques for solving large-scale linear algebra problems. The library is designed to be user-friendly and easy to use, with a focus on simplicity and efficiency. The library is built on top of NumPy and SciPy, and provides a simple interface for creating sketched matrices and solving linear systems using randomized numerical linear algebra algorithms.

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

### ⚡️ Fast Sparse Sketch Operators

```python
import numpy as np
import numpy.linalg as LA
from sketch_n_solve.solve.least_squares import LeastSquares

# Recommended to use either "clarkson_woodruff" or "uniform_sparse"
# for best results out of the box
sketch_fn = "clarkson_woodruff"

rng = np.random.default_rng(seed)
A = rng.standard_normal((1000000, 1000))
x = rng.standard_normal(1000)
b = A @ x

seed = 42
lsq = LeastSquares(sketch_fn, seed)

x_hat, time_elapsed, x_hats, istop = lsq.sketch_and_apply(A, b)

print("residual", LA.norm(A @ x_hat - A @ x))

is_close = np.allclose(x_hat, x)
print(f"x_hat is close to x => {is_close}")

assert is_close, "Something went wrong!"

print(f"Finished in {time_elapsed} seconds.")
```

### 🎨 Generate Sketch Matrix

You can also generate the sketch matrix and apply it to the input matrix `A` and the right-hand side `b` separately or for other downstream linear algebra tasks!

```python
import numpy as np
from sketch_n_solve.sketch import Sketch

sketch_fn = "normal"
seed = 42
sketch = Sketch(sketch_fn, seed)

rng = np.random.default_rng(seed)
A = rng.standard_normal((10000, 10))
b = rng.standard_normal(10000)

# Generate sketch matrix S
A, S = sketch.sketch(A)

# Use sketch matrix S to sketch A and b
SA = S @ A
Sb = S @ b
```

