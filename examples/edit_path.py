"""Inspect the alignment, edit costs, and crossover edit dependencies."""

from parents import describe, make_parents
from rcswx import edit_path


def main() -> None:
    a, b = make_parents()
    print("Parent A:", describe(a))
    print("Parent B:", describe(b))

    alignment = edit_path(a, b)
    print("Edit distance:", alignment.distance)

    print("\nOrdered alignment operations:")
    for operation in alignment.operations:
        print(" ", operation)

    print("\nNontrivial edits available for crossover:")
    for operation in alignment.nontrivial_ops:
        # __str__ includes the operation cost and any dependency constraints.
        print(" ", operation)


if __name__ == "__main__":
    main()
