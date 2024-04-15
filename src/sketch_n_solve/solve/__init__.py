from abc import ABC, abstractmethod
from typing import Any

import numpy as np
from sketch_n_solve.sketch import Sketch


class Solver(ABC):
    def __init__(self, sketch: Sketch) -> None:
        """Solver class.

        Parameters
        ----------
        sketch : Sketch
            The sketch object.
        """
        self.sketch = sketch

    @abstractmethod
    def __call__(self, A: np.ndarray, b: np.ndarray) -> Any:
        pass
