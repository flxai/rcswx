"""Use reconstruction and a test forward pass to validate an offspring."""

import torch
from parents import INPUT_SHAPE, describe, make_builder, make_parents
from rcswx import validated_crossover


def main() -> None:
    a, b = make_parents()
    builder = make_builder()
    # Weight initialization and structural selection use separate streams.
    torch.manual_seed(0)

    child, report = validated_crossover(
        a,
        b,
        rebuild=builder.re_id,
        batch_shape=INPUT_SHAPE,
        max_tries=10,
        seed=0,
    )
    print("Validated offspring:", describe(child))
    print("Selected edits:", len(report["crossover_operations"]))

    # Validation used a temporary model. Build the model to use afterward.
    model = child.build(child)
    with torch.no_grad():
        output = model(torch.zeros(INPUT_SHAPE))
    print("Output shape:", tuple(output.shape))


if __name__ == "__main__":
    main()
