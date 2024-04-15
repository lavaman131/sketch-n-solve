import numpy as np
from sketch_n_solve.sketch import Sketch
from . import Solver


class LeastSquares(Solver):
    def __init__(self, A: np.ndarray, b: np.ndarray, sketch: Sketch) -> None:
        super().__init__(A, b, sketch)

    def solve(self):
        pass
