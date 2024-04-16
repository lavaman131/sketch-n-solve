from typing import Any
from sketch_n_solve.sketch import Sketch


class Solver:
    def __init__(self, sketch_fn: str, seed: int, **kwargs: Any) -> None:
        """Solver class.

        Parameters
        ----------
        sketch_fn : str
            The sketch function.
        seed : int
            The seed for the random number generator.
        **kwargs : Any
            Additional arguments for the sketch function.
        """
        self.seed = seed
        self.sketch_fn = sketch_fn
        self.sketch = Sketch(self.sketch_fn, self.seed, **kwargs)

    def __repr__(self) -> str:
        return (
            f"{self.__class__.__name__}(sketch_fn={self.sketch_fn}, seed={self.seed})"
        )
