import numpy as np

from sketch_n_solve.sketch import Sketch


if __name__ == "__main__":
    sketch_fn = "sparse_sign"
    seed = 42
    sketch = Sketch(sketch_fn, seed)
    A = np.random.randn(100, 10)
    sketched_matrix = sketch(A)
    print(sketched_matrix)
