"""Genotype-first alignment; no module inference or fixed output contract."""

from .limiter import Limiter
from .recursive import Alignment


def edit_path(parent1, parent2, *, collapse_corners=False, limiter=None):
    """Return ordered reference histories and dependency-bearing operations.

    The reference renumbers the second parent in place. This is not an immutable
    canonical path and must not be cached across reconstruction or parent mutation.
    """
    if limiter is None:
        limiter = parent1.limiter
    return Alignment(parent1, parent2, collapse_corners=collapse_corners, limiter=limiter)


def distance(parent1, parent2):
    """Match rcswx_distance's no-op shortcut and independently created limiter."""
    if parent1.serialise() == parent2.serialise():
        same = True
        for op1, op2 in zip(parent1.serialise(), parent2.serialise()):
            if op1.operation.name != op2.operation.name:
                same = False
                break
        if same:
            return 0
    limiter = Limiter(
        limits={
            "time": 60,
            "restart_time": 300,
            "max_id": 10000,
            "depth": 20,
            "memory": 8196,
            "memory_crossover": 65536,
            "individual_memory": 1024,
            "batch_pass_seconds": 0.1,
        }
    )
    return Alignment(parent1, parent2, limiter=limiter).distance
