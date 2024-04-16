from typing import Any
from sketch_n_solve.sketch import Sketch


class Solver:
    def __init__(self, sketch_fn: str, seed: int, **kwargs: Any) -> None:
        """Solver class.

        Parameters
        ----------
        sketch : Sketch
            The sketch object.
        """
        self.sketch = Sketch(sketch_fn, seed)
