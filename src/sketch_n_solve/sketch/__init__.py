# from sketch_n_solve import __package__
from typing import Callable, Optional, Tuple, TypeAlias, Union
import numpy as np
import importlib

sketch_fn_type: TypeAlias = Callable[
    [np.ndarray, int, Optional[int]], Tuple[np.ndarray, np.ndarray]
]


class Sketch:
    def __init__(
        self,
        sketch_fn: str,
        seed: Optional[int] = 42,
    ) -> None:
        self.sketch_fn = self._get_sketch_fn(sketch_fn)
        self.seed = seed

    def __call__(self, A: np.ndarray, k: int) -> np.ndarray:
        A, sketch_matrix = self.sketch_fn(A, k, self.seed)
        return sketch_matrix @ A

    def _get_sketch_fn(self, sketch_fn: str) -> sketch_fn_type:
        # Import the module based on the sketch function name
        if sketch_fn in {"uniform", "normal", "hadamard"}:
            module_name = ".dense"
        elif sketch_fn in {"clarkson_woodruff"}:
            module_name = ".sparse"
        else:
            raise ValueError(f"Unknown sketch function: {self.sketch_fn}")

        module = importlib.import_module(module_name, package=__package__)

        # Get the sketch function from the imported module
        return getattr(module, sketch_fn)

    def __repr__(self) -> str:
        return f"Sketch(sketch_fn={self.sketch_fn.__name__}, seed={self.seed})"
