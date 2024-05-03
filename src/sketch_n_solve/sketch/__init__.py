# from sketch_n_solve import __package__
from typing import Any, Callable, Optional, Set, Tuple, TypeAlias
import numpy as np
import importlib

sketch_fn_type: TypeAlias = Callable[..., Tuple[np.ndarray, np.ndarray]]


class Sketch:
    dense_sketch_fns: Set[str] = {"uniform_dense", "normal", "hadamard"}
    sparse_sketch_fns: Set[str] = {"uniform_sparse", "clarkson_woodruff", "sparse_sign"}

    def __init__(self, sketch_fn: str, seed: Optional[int] = 42) -> None:
        """Sketch class.

        Parameters
        ----------
        sketch_fn : str
            The name of the sketch function. The available sketch functions are listed by using `self.list_available_sketch_fns`.
        seed : Optional[int], optional
            Random seed, by default 42.
        """
        self.sketch_fn = self._get_sketch_fn(sketch_fn)
        self.seed = seed

    def __call__(self, A: np.ndarray, **kwargs: Any) -> Tuple[np.ndarray, np.ndarray]:
        """Sketches the input matrix.

        Parameters
        ----------
        A : np.ndarray
            The input matrix.
        **kwargs : Any
            Additional required arguments depending on the sketch function.

        Returns
        -------
        A, S : Tuple[np.ndarray, np.ndarray]
            The input matrix, padded with zeros if needed, and the sketch matrix.
        """
        A, S = self.sketch_fn(A, seed=self.seed, **kwargs)
        return A, S

    @staticmethod
    def list_available_sketch_fns() -> None:
        """List the available sketch functions."""
        print("Dense sketch functions:")
        for sketch_fn in Sketch.dense_sketch_fns:
            print(f"- {sketch_fn}")
        print("\nSparse sketch functions:")
        for sketch_fn in Sketch.sparse_sketch_fns:
            print(f"- {sketch_fn}")

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
        if sketch_fn in Sketch.dense_sketch_fns:
            module_name = ".dense"
        elif sketch_fn in Sketch.sparse_sketch_fns:
            module_name = ".sparse"
        else:
            raise ValueError(f"Unknown sketch function: {self.sketch_fn}")

        module = importlib.import_module(module_name, package=__package__)

        # Get the sketch function from the imported module
        return getattr(module, sketch_fn)

    def __repr__(self) -> str:
        return f"Sketch(sketch_fn={self.sketch_fn.__name__}, seed={self.seed})"
