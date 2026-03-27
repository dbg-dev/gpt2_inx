import functools
import time
from collections.abc import Callable
from typing import ParamSpec, TypeVar

from loguru import logger

P = ParamSpec("P")
R = TypeVar("R")


def timeit(func: Callable[P, R]):
    """
    Timer decorator
    """

    @functools.wraps(func)
    def wrapper(*args, **kwargs):
        start = time.perf_counter()
        result = func(*args, **kwargs)
        end = time.perf_counter()
        elapsed = end - start
        logger.info(f"[timeit] {func.__name__} executed in:\t{elapsed:.6f}")
        return result

    return wrapper
