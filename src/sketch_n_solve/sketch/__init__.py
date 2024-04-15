# from sketch_n_solve import __package__
from typing import Any, Callable, Optional, Tuple, TypeAlias, Union
import numpy as np
import importlib

sketch_fn_type: TypeAlias = Callable[..., Tuple[np.ndarray, np.ndarray]]


class Sketch:
    def __init__(
        self,
        sketch_fn: str,
        seed: Optional[int] = 42,
    ) -> None:
        """Sketch class.

        Parameters
        ----------
        sketch_fn : str
            The name of the sketch function. The available sketch functions are:
            - "dense"
                - "uniform"
                - "normal"
                - "hadamard"
            - "sparse"
                - "clarkson_woodruff"
        seed : Optional[int], optional
            Random seed, by default 42.
        """
        self.sketch_fn = self._get_sketch_fn(sketch_fn)
        self.seed = seed

    def __call__(self, A: np.ndarray, **kwargs: Any) -> np.ndarray:
        """Sketches the input matrix.

        Parameters
        ----------
        A : np.ndarray
            The input matrix.
        **kwargs : Any
            Additional required arguments depending on the sketch function.

        Returns
        -------
        np.ndarray
            The sketched matrix.
        """
        A, sketch_matrix = self.sketch_fn(A, seed=self.seed, **kwargs)
        return sketch_matrix @ A

    def _get_sketch_fn(self, sketch_fn: str) -> sketch_fn_type:
        """Get the sketch function based on the sketch function name.

        Parameters
        ----------
        sketch_fn : str
            The name of the sketch function.

        Returns
        -------
        sketch_fn_type
            The sketch function.

        Raises
        ------
        ValueError
            If the sketch function is not available.
        """
        # Import the module based on the sketch function name
        if sketch_fn in {"uniform", "normal", "hadamard"}:
            module_name = ".dense"
        elif sketch_fn in {"clarkson_woodruff", "sparse_sign"}:
            module_name = ".sparse"
        else:
            raise ValueError(f"Unknown sketch function: {self.sketch_fn}")

        module = importlib.import_module(module_name, package=__package__)

        # Get the sketch function from the imported module
        return getattr(module, sketch_fn)

    def __repr__(self) -> str:
        return f"Sketch(sketch_fn={self.sketch_fn.__name__}, seed={self.seed})"
