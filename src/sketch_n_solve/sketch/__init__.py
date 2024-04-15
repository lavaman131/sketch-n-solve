from typing import Callable, Optional, Tuple, Union
import numpy as np
import scipy.sparse


class Sketch:
    def __init__(
        self,
        A: np.ndarray,
        k: int,
        sketch_fn: Callable[
            [np.ndarray, int, Optional[int]],
            Tuple[np.ndarray, Union[np.ndarray, scipy.sparse.csc_array]],
        ],
        seed: Optional[int] = 42,
    ) -> None:
        self.A = A
        self.k = k
        self.sketch_fn = sketch_fn
        self.seed = seed

        self.sketch_matrix = self.sketch_fn(self.A, self.k, self.seed)
        self.sketched_matrix = self.sketch_matrix @ self.A
