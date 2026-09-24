"""Framework-free public structural API."""

from __future__ import annotations

from .portable import Architecture
from .recursive import Alignment, Selection, _same_raw_parent, apply_edits


def edit_path(
    parent1, parent2, *, collapse_corners=False, limiter=None, profile=False, limits=None
) -> Alignment:
    """Return a retained native alignment plan for two portable or legacy trees.

    Legacy preparation retains the reference-visible renumbering of the second
    parent.  Portable values are immutable and have no host runtime dependency.
    """
    return Alignment(
        parent1,
        parent2,
        collapse_corners=collapse_corners,
        limiter=limiter,
        profile=profile,
        limits=limits,
    )


def distance(parent1, parent2):
    """Return the native structural distance, preserving the legacy no-op shortcut."""
    if _same_raw_parent(parent1, parent2):
        return 0
    if isinstance(parent1, Architecture) or isinstance(parent2, Architecture):
        return edit_path(parent1, parent2).distance
    # Retain the historical independent limiter only on the legacy integration path.
    from .limiter import Limiter

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
    return edit_path(parent1, parent2, limiter=limiter).distance


__all__ = ["Alignment", "Selection", "apply_edits", "distance", "edit_path"]
