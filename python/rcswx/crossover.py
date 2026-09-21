"""Raw genotype crossover conveniences; construction is an explicit next layer."""

from dataclasses import dataclass

from .recursive import raw_crossover


@dataclass(frozen=True, slots=True)
class CrossoverResult:
    child: object
    report: dict


def crossover_with_report(parent1, parent2, *, skewness=0, limiter=None):
    """Return a genotype and reference distance/selection metadata.

    Consumes the caller's NumPy RNG exactly where the reference selector does.
    No-op crossover may alias parent1. No weights, task head, or model are copied.
    """
    child, selected, operations, distance1, distance2, between = raw_crossover(
        parent1, parent2, skewness=skewness, limiter=limiter
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
        },
    )


def crossover(parent1, parent2, *, skewness=0, limiter=None):
    """Return only the raw offspring genotype, preserving original ownership."""
    return raw_crossover(parent1, parent2, skewness=skewness, limiter=limiter)[0]
