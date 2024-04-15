import numpy as np
from sketch_n_solve.sketch import Sketch
from abc import ABC, abstractmethod


class Solver(ABC):
    def __init__(self, A: np.ndarray, b: np.ndarray, sketch: Sketch) -> None:
        self.A = A
        self.b = b
        self.sketch = sketch

    @abstractmethod
    def solve(self):
        pass
