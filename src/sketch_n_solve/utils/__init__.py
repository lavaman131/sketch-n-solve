import time
from typing import Any, Callable, Sequence, Tuple

__all__ = ["timer"]


def timer(func: Callable[..., Any]) -> Callable[..., Sequence[Any]]:
    def wrapper(*args: Any, **kwargs: Any) -> Sequence[Any]:
        start = time.perf_counter()
        res = func(*args, **kwargs)
        end = time.perf_counter()
        time_elapsed = end - start
        return res, time_elapsed

    return wrapper
