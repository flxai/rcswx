"""Portable structural alignment without Torch, NumPy, or SciPy.

Run from a source checkout after ``uv sync --locked`` or from an installed
minimal wheel. Inspect the weighted edits and choose a genuinely mixed child;
the seeded sample uses RCSWX's native ChaCha12 stream, not a framework RNG.
"""

from rcswx import Architecture, apply_edits, edit_path


def architecture(width: int, activation: str) -> Architecture:
    return Architecture.from_tree(
        (
            "sequential",
            ("computation", (f"linear({width})",)),
            ("computation", (activation,)),
        )
    )


def main() -> None:
    first = architecture(16, "relu")
    second = architecture(32, "softmax")
    plan = edit_path(first, second)
    for bit, edit in enumerate(plan.nontrivial_ops):
        print(f"Bit {bit}: {edit}")

    selection = plan.select("10")
    child = apply_edits(plan, selection)
    unchanged = apply_edits(plan, plan.select([]))
    sampled = apply_edits(plan, plan.sample(seed=0))

    print("Full distance:", plan.distance, "Selected cost:", selection.cost)
    print("Child operations:", [node["name"] for node in child.to_dict()["nodes"]])
    print("Empty-selection child:", [node["name"] for node in unchanged.to_dict()["nodes"]])
    print("Sampled child:", [node["name"] for node in sampled.to_dict()["nodes"]])


if __name__ == "__main__":
    main()
