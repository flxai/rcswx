"""Register and use the intentionally narrow Linear/ReLU converter.

Run from the repository root with the torch extra:
    uv run --locked --extra torch python examples/torch_converter.py
"""

import torch
from rcswx.torch import (
    build,
    capture,
    linear_relu_sequential_converter,
    register_importer,
    unregister_importer,
)
from torch import nn

INPUT_SPEC = {
    "shape": [4, 8],
    "mode": "col",
    "other_shape": None,
    "other_mode": None,
    "branching_factor": 1,
    "last_im_shape": None,
}


def main() -> None:
    registration = register_importer(
        "linear-relu-sequential",
        "1",
        [nn.Sequential],
        linear_relu_sequential_converter,
    )
    try:
        source = nn.Sequential(nn.Linear(8, 16), nn.ReLU())
        captured = capture(source, input_spec=INPUT_SPEC)
        rebuilt = build(captured)
        print("Importer:", captured.metadata["importer"])
        with torch.no_grad():
            print("Rebuilt output shape:", tuple(rebuilt(torch.zeros(4, 8)).shape))
    finally:
        unregister_importer(registration.name, registration.version)


if __name__ == "__main__":
    main()
