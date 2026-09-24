"""Portable structural alignment without Torch, NumPy, or SciPy.

Run from a source checkout after ``uv sync --locked`` or from an installed
minimal wheel. The explicit empty selection demonstrates deterministic replay;
the seeded sample uses RCSWX's native ChaCha12 stream, not a framework RNG.
"""

from rcswx import Architecture, apply_edits, edit_path


def main() -> None:
    first = Architecture.from_tree(("sequential", ("relu",)))
    second = Architecture.from_tree(("sequential", ("sigmoid",)))

    explicit_plan = edit_path(first, second)
    unchanged = apply_edits(explicit_plan, explicit_plan.select([]))

    seeded_plan = edit_path(first, second)
    sampled = apply_edits(seeded_plan, seeded_plan.sample(seed=42))

    print(f"distance: {explicit_plan.distance}")
    print(f"explicit root: {unchanged.to_dict()['nodes'][unchanged.to_dict()['root']]['name']}")
    print(f"seeded root: {sampled.to_dict()['nodes'][sampled.to_dict()['root']]['name']}")


if __name__ == "__main__":
    main()
