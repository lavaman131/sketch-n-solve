from pathlib import Path

from tqdm import tqdm
from sketch_n_solve.utils.least_squares import (
    LeastSquaresProblemConfig,
    generate_least_squares_problem,
    generate_ortho_least_squares_problem,
)


def main() -> None:
    seed = 42
    output_dir = Path("outputs")
    output_dir.mkdir(exist_ok=True)
    default_kwargs = {"cond": 1e10, "beta": 1e-10}
    dims = [(100000, 1000)]
    config = [{**default_kwargs, "m": m, "n": n} for m, n in dims]
    for kwarg in tqdm(config):
        lsq = LeastSquaresProblemConfig(**kwarg)
        generate_least_squares_problem(
            lsq.m,
            lsq.n,
            lsq.cond,
            lsq.beta,
            output_dir.joinpath("precomputed_problems"),
            seed=seed,
        )


if __name__ == "__main__":
    main()
