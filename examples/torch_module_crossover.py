"""Build, capture, and cross two RCSWX PyTorch modules.

Run from the repository root with the torch extra:
    uv run --locked --extra torch python examples/torch_module_crossover.py
"""

import torch
from rcswx.portable import Architecture
from rcswx.torch import build, capture, crossover_with_report

INPUT_SPEC = {
    "shape": [4, 8],
    "mode": "col",
    "other_shape": None,
    "other_mode": None,
    "branching_factor": 1,
    "last_im_shape": None,
}


def parent(width: int, activation: str) -> Architecture:
    return Architecture.from_tree(
        (
            "sequential",
            ("computation", (f"linear({width})",)),
            ("computation", (activation,)),
        ),
        grammar="einspace",
        grammar_version="1",
        input_spec=INPUT_SPEC,
    )


def main() -> None:
    torch.manual_seed(0)
    first = build(parent(16, "relu"), build_options={"dtype": torch.float32})
    second = build(parent(32, "softmax"), build_options={"dtype": torch.float32})
    result = crossover_with_report(first, second, seed=0)

    print("Captured grammar:", capture(result.child).architecture.to_dict()["grammar"])
    print("Selected operations:", result.report["crossover_operations"])
    print("Child is fresh:", result.child is not first and result.child is not second)
    print(result.child)
    result.child.eval()
    with torch.no_grad():
        print("Child output shape:", tuple(result.child(torch.randn(4, 8)).shape))


if __name__ == "__main__":
    main()
