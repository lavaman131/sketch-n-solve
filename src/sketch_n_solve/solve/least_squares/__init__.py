from typing import Any, Optional
import numpy as np
from sketch_n_solve.solve.least_squares.algorithms import (
    _sketch_and_apply,
    _smoothed_sketch_and_apply,
    _sketch_and_precondition,
)
from .. import Solver
import scipy.linalg as SLA


class LeastSquares(Solver):
    def __init__(self, sketch: str, seed: int, **kwargs: Any) -> None:
        """Least squares solver.

        Parameters
        ----------
        sketch : Sketch
            The sketch object.
        """
        super().__init__(sketch, seed, **kwargs)

    def sketch_and_precondition(
        self,
        A: np.ndarray,
        b: np.ndarray,
        use_sketch_and_solve_x_0: bool = True,
        tolerance: float = 1e-6,
        num_iters: Optional[int] = 100,
        **kwargs: Any,
    ) -> np.ndarray:
        """Solves the least squares problem using sketch and preconditioning as described in https://arxiv.org/pdf/2302.07202.pdf.

        Parameters
        ----------
        A : (n, d) np.ndarray
            The input matrix.
        b : (n, 1) np.ndarray
            The target vector.
        use_sketch_and_solve_x_0 : bool, optional
            Whether to use x_0 from sketch and solve as the initial guess for the least squares solver rather than the zero vector, by default True.
        tolerance : float, optional
            Error tolerance. Controls the number of iterations if num_iters is not specified, by default 1e-6.
        num_iters : int, optional
            Maximum number of iterations for least-squares QR solver, by default 100.
        **kwargs : Any
            Additional required arguments depending on the sketch function.

        Returns
        -------
        x : (d, 1) np.ndarray
            The solution to the least squares problem.
        """
        A, S = self.sketch(A, **kwargs)
        x = _sketch_and_precondition(
            A, b, S, use_sketch_and_solve_x_0, tolerance, num_iters
        )
        return x

    def sketch_and_apply(
        self,
        A: np.ndarray,
        b: np.ndarray,
        tolerance: float = 1e-6,
        num_iters: Optional[int] = 100,
        **kwargs: Any,
    ) -> np.ndarray:
        """Solves the least squares problem using sketch-and-apply as described in https://arxiv.org/pdf/2302.07202.pdf.

        Parameters
        ----------
        A : (n, d) np.ndarray
            The input matrix.
        b : (n, 1) np.ndarray
            The target vector.
        tolerance : float
            Error tolerance. Controls the number of iterations if num_iters is not specified.
        num_iters : int, optional
            Maximum number of iterations for least-squares QR solver, by default 100. If specified will overwrite tolerance parameter for error tolerance.
        **kwargs : Any
            Additional required arguments depending on the sketch function.

        Returns
        -------
        x : (d, 1) np.ndarray
            The solution to the least squares problem.
        """
        A, S = self.sketch(A, **kwargs)
        x = _sketch_and_apply(A, b, S, tolerance, num_iters)

        if SLA.norm(A @ x - b) <= tolerance:
            return x
        return _smoothed_sketch_and_apply(A, b, S, tolerance, num_iters)
