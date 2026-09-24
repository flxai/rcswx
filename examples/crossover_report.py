"""Inspect the selected edits and alignment-based crossover distances."""

from parents import describe, make_parents
from rcswx import crossover_with_report


def main() -> None:
    a, b = make_parents()
    result = crossover_with_report(a, b, seed=0)
    report = result.report

    print("Offspring:", describe(result.child))
    print("Distance between parents:", report["crossover_distance_between_parents"])
    # These two values come from selected edit costs, not fresh alignments.
    print("Reported distance to A:", report["crossover_distance_to_parent1"])
    print("Reported distance to B:", report["crossover_distance_to_parent2"])
    print("Available edits:", len(report["crossover_all_operations"]))
    print("Selected edits:", len(report["crossover_operations"]))
    for operation in report["crossover_operations"]:
        print(" ", operation)


if __name__ == "__main__":
    main()
