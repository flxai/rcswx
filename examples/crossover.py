"""Cross two parents, reconstruct the child, then build and run its model."""

import numpy as np
import torch
from parents import INPUT_SHAPE, describe, make_builder, make_parents
from rcswx import crossover


def main() -> None:
    a, b = make_parents()
    print("Parent A:", describe(a))
    print("Parent B:", describe(b))

    np.random.seed(0)
    child = crossover(a, b)
    print("Offspring:", describe(child))

    # Refresh metadata before model construction. This creates fresh weights.
    child = make_builder().re_id(child)
    model = child.build(child)
    with torch.no_grad():
        output = model(torch.zeros(INPUT_SHAPE))
    print("Output shape:", tuple(output.shape))


if __name__ == "__main__":
    main()
