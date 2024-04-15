import numpy as np

from sketch_n_solve.sketch import Sketch


if __name__ == "__main__":
    print(__package__)
    A = np.random.randn(100, 10)
    k = 5
    sketch_fn = "uniform"
    seed = 42
    sketch = Sketch(A, k, sketch_fn, seed)
    print(sketch.sketched_matrix)
    print(sketch.sketch_matrix)
    print(sketch.A)
