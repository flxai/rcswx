"""Raw crossover conveniences over retained native structural plans."""

from __future__ import annotations

from dataclasses import dataclass

from .recursive import raw_crossover


@dataclass(frozen=True, slots=True)
class CrossoverResult:
    child: object
    report: dict


def _rng_label(sampler: str) -> str:
    if sampler == "native":
        return "chacha12-v1"
    if sampler == "reference":
        return "numpy-ambient"
    raise ValueError("sampler must be 'native' or 'reference'")


def crossover_with_report(
    parent1,
    parent2,
    *,
    skewness=0,
    limiter=None,
    sampler="native",
    seed=None,
    rng=None,
):
    """Return a child and the historical selected-operation distance report."""
    child, selected, operations, distance1, distance2, between = raw_crossover(
        parent1,
        parent2,
        skewness=skewness,
        limiter=limiter,
        sampler=sampler,
        seed=seed,
        rng=rng,
    )
    return CrossoverResult(
        child,
        {
            "crossover_operations": selected,
            "crossover_all_operations": operations,
            "crossover_distance_to_parent1": distance1,
            "crossover_distance_to_parent2": distance2,
            "crossover_distance_between_parents": between,
            "crossover_skewness": skewness,
            "crossover_sampler": sampler,
            "crossover_rng": _rng_label(sampler),
            "crossover_numeric_policy": (
                "native-skew-normal-v1" if sampler == "native" else "scipy-reference"
            ),
        },
    )


def crossover(parent1, parent2, *, skewness=0, limiter=None, sampler="native", seed=None, rng=None):
    """Return only the native-applied offspring, preserving raw no-op aliasing."""
    return raw_crossover(
        parent1,
        parent2,
        skewness=skewness,
        limiter=limiter,
        sampler=sampler,
        seed=seed,
        rng=rng,
    )[0]
