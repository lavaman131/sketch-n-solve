from typing import Any, List, Optional, Tuple
import numpy as np
from .algorithms import (
    _sketch_and_apply,
    _smoothed_sketch_and_apply,
    _sketch_and_precondition,
)
from .. import Solver


class LeastSquares(Solver):
    def __init__(self, sketch: str, seed: int, **kwargs: Any) -> None:
        """Least squares solver.

        Parameters
        ----------
        sketch : Sketch
            The sketch object.
        """
        super().__init__(sketch, seed, **kwargs)

    # def sketch_and_precondition(
    #     self,
    #     A: np.ndarray,
    #     b: np.ndarray,
    #     use_sketch_and_solve_x_0: bool = True,
    #     tolerance: float = 1e-12,
    #     iter_lim: Optional[int] = 100,
    #     log_x_hat: bool = False,
    #     **kwargs: Any,
    # ) -> Tuple[np.ndarray, float, List[np.ndarray], int]:
    #     """Solves the least squares problem using sketch and preconditioning as described in https://arxiv.org/pdf/2302.07202.pdf.

    #     Parameters
    #     ----------
    #     A : (n, d) np.ndarray
    #         The input matrix.
    #     b : (n, 1) np.ndarray
    #         The target vector.
    #     use_sketch_and_solve_x_0 : bool, optional
    #         Whether to use x_0 from sketch and solve as the initial guess for the least squares solver rather than the zero vector, by default True.
    #     tolerance : float, optional
    #         Error tolerance. Controls the number of iterations if iter_lim is not specified, by default 1e-12.
    #     iter_lim : int, optional
    #         Maximum number of iterations for least-squares QR solver, by default 100.
    #     callback : Optional[Callable[[np.ndarray], None]], optional
    #         Callback function to be called after each iteration of LSQR, by default None.
    #     **kwargs : Any
    #         Additional required arguments depending on the sketch function.

    #     Returns
    #     -------
    #     x : (d, 1) np.ndarray
    #         The solution to the least squares problem.
    #     time_elapsed : float
    #         Time taken to solve the least squares problem.
    #     x_hats : List[np.ndarray]
    #         List of intermediate solutions if log_x_hat is True.
    #     """
    #     A, S = self.sketch(A, **kwargs)
    #     x, time_elapsed, x_hats, istop = _sketch_and_precondition(
    #         A, b, S, use_sketch_and_solve_x_0, tolerance, iter_lim, log_x_hat
    #     )
    #     return x, time_elapsed, x_hats, istop

    def __call__(
        self,
        A: np.ndarray,
        b: np.ndarray,
        tolerance: float = 1e-12,
        iter_lim: Optional[int] = 100,
        log_x_hat: bool = False,
        **kwargs: Any,
    ) -> Tuple[np.ndarray, float, List[np.ndarray], int]:
        """Solves the least squares problem using sketch-and-apply as described in https://arxiv.org/pdf/2302.07202.pdf.

        Parameters
        ----------
        A : (m, n) np.ndarray
            The input matrix.
        b : (m, 1) np.ndarray
            The target vector.
        tolerance : float
            Error tolerance. Controls the number of iterations if iter_lim is not specified.
        iter_lim : int, optional
            Maximum number of iterations for least-squares QR solver, by default 100. If specified will overwrite tolerance parameter for error tolerance.
        **kwargs : Any
            Additional required arguments depending on the sketch function.

        Returns
        -------
        x : (n, 1) np.ndarray
            The solution to the least squares problem.
        time_elapsed : float
            Time taken to solve the least squares problem.
        x_hats : List[np.ndarray]
            List of intermediate solutions if log_x_hat is True.
        """
        return self.sketch_and_apply(A, b, tolerance, iter_lim, log_x_hat, **kwargs)

    def sketch_and_apply(
        self,
        A: np.ndarray,
        b: np.ndarray,
        tolerance: float = 1e-12,
        iter_lim: Optional[int] = 100,
        log_x_hat: bool = False,
        **kwargs: Any,
    ) -> Tuple[np.ndarray, float, List[np.ndarray], int]:
        """Solves the least squares problem using sketch-and-apply as described in https://arxiv.org/pdf/2302.07202.pdf.

        Parameters
        ----------
        A : (m, n) np.ndarray
            The input matrix.
        b : (m, 1) np.ndarray
            The target vector.
        tolerance : float
            Error tolerance. Controls the number of iterations if iter_lim is not specified.
        iter_lim : int, optional
            Maximum number of iterations for least-squares QR solver, by default 100. If specified will overwrite tolerance parameter for error tolerance.
        **kwargs : Any
            Additional required arguments depending on the sketch function.

        Returns
        -------
        x : (n, 1) np.ndarray
            The solution to the least squares problem.
        time_elapsed : float
            Time taken to solve the least squares problem.
        x_hats : List[np.ndarray]
            List of intermediate solutions if log_x_hat is True.
        """
        A, S = self.sketch(A, **kwargs)
        x, time_elapsed_apply, x_hats_apply, istop = _sketch_and_apply(
            A, b, S, tolerance, iter_lim // 2, log_x_hat
        )

        if istop != 0:
            return x, time_elapsed_apply, x_hats_apply, istop
        x, time_elapsed_smoothed, x_hats_smoothed, istop = _smoothed_sketch_and_apply(
            A, b, S, tolerance, iter_lim // 2, self.seed, log_x_hat
        )
        return (
            x,
            time_elapsed_apply + time_elapsed_smoothed,
            x_hats_apply + x_hats_smoothed,
            istop,
        )
