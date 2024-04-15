from typing import Any, Optional
import numpy as np
from sketch_n_solve.sketch import Sketch
from sketch_n_solve.solve.least_squares.algorithms import (
    _iterative_sketching,
    _sketch_and_precondition,
)
from .. import Solver


class LeastSquares(Solver):
    def __init__(self, sketch: Sketch) -> None:
        """Least squares solver.

        Parameters
        ----------
        sketch : Sketch
            The sketch object.
        """
        super().__init__(sketch)

    def __call__(self, A: np.ndarray, b: np.ndarray):
        pass

    def sketch_and_precondition(
        self, A: np.ndarray, b: np.ndarray, **kwargs: Any
    ) -> np.ndarray:
        """Solves the least squares problem using sketch and preconditioning as described in https://arxiv.org/pdf/2311.04362.pdf.

        Parameters
        ----------
        A : (n, d) np.ndarray
            The input matrix.
        b : (n, 1) np.ndarray
            The target vector.
        **kwargs : Any
            Additional required arguments depending on the sketch function.

        Returns
        -------
        x_i : (d, 1) np.ndarray
            The solution to the least squares problem.
        """
        A, S = self.sketch(A, **kwargs)
        return _sketch_and_precondition(A, b, S)

    def iterative_sketching(
        self,
        A: np.ndarray,
        b: np.ndarray,
        delta: float = 1e-10,
        num_iters: Optional[int] = None,
        **kwargs: Any,
    ) -> np.ndarray:
        """Solves the least squares problem using iterative sketching as described in https://arxiv.org/pdf/2311.04362.pdf.

        Parameters
        ----------
        A : (n, d) np.ndarray
            The input matrix.
        b : (n, 1) np.ndarray
            The target vector.
        delta : float
            Error tolerance. Controls the number of iterations if num_iters is not specified.
        num_iters : int, optional
            Maximum number of iterations for least-squares QR solver, by default None. If specified will overwrite delta parameter for error tolerance.
        **kwargs : Any
            Additional required arguments depending on the sketch function.

        Returns
        -------
        x_i : (d, 1) np.ndarray
            The solution to the least squares problem.
        """
        A, S = self.sketch(A, **kwargs)
        return _iterative_sketching(A, b, S)
