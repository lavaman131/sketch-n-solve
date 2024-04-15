import numpy as np
from sketch_n_solve.sketch import Sketch
from . import Solver


class LeastSquares(Solver):
    def __init__(self, sketch: Sketch) -> None:
        super().__init__(sketch)

    def __call__(self, A: np.ndarray, b: np.ndarray):
        pass
