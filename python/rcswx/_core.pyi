"""Owned primitive boundary for the native reference-compatible kernels."""

from collections.abc import Callable

type TokenRecord = tuple[int, str, list[str], int]
type StepRecord = tuple[int, str, int | None, int | None, int, int, float, bool, bool]
type SelectionRecord = tuple[int, float, list[list[int]], list[list[int]]]

def recursive_align(
    tokens1: list[TokenRecord],
    tokens2: list[TokenRecord],
    collapse_corners: bool = False,
    memory_check: Callable[[], bool] | None = None,
) -> tuple[float, list[list[StepRecord]], dict[str, int]]: ...
def valid_combinations(records: list[SelectionRecord]) -> tuple[list[str], list[float]]: ...
