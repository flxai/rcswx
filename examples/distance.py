"""Compute architecture edit distances; no model training or inference."""

from parents import describe, make_parents
from rcswx import distance


def main() -> None:
    a, b = make_parents()
    print("Parent A:", describe(a))
    print("Parent B:", describe(b))
    print("Distance A → B:", distance(a, b))
    print("Distance A → A:", distance(a, a))


if __name__ == "__main__":
    main()
