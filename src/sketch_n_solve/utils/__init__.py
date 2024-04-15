import time
from typing import Any, Callable

__all__ = ["timer"]


def timer(func: Callable[..., Any]) -> Callable[..., Any]:
    def wrapper(*args: Any, **kwargs: Any) -> Any:
        start = time.perf_counter()
        res = func(*args, **kwargs)
        print(f"Execution time: {time.perf_counter() - start}")
        return res

    return wrapper
