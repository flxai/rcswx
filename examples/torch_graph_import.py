"""Import ordinary models, validate a mixed child, train it, and save/load it.

Run with the torch extra:
    uv run --locked --extra torch python examples/torch_graph_import.py
"""

from pathlib import Path
from tempfile import TemporaryDirectory

import torch
from rcswx.torch import capture, crossover_imported, import_model, load, save
from torch import nn


def main() -> None:
    torch.manual_seed(0)
    inputs = (torch.randn(4, 8),)
    sources = (
        nn.Sequential(nn.Linear(8, 16), nn.ReLU()),
        nn.Sequential(nn.Linear(8, 32), nn.Softmax(dim=-1)),
    )
    parents = tuple(
        import_model(model, example_inputs=inputs, grammar="einspace", grammar_version="1")
        for model in sources
    )
    result = crossover_imported(*parents, validation_inputs=inputs, seed=0)
    assert result.architecture not in tuple(parent.architecture for parent in parents)
    print("Selected operations:", result.report["crossover_operations"])
    print("Validated mixed child:", result.report["validation"])

    # Fresh child state: no learned tensors are inherited from either source.
    child = result.child
    optimizer = torch.optim.SGD(child.parameters(), lr=0.1)
    before = child(*inputs).detach().clone()
    loss = child(*inputs).square().mean()
    loss.backward()
    optimizer.step()
    child.eval()
    trained = child(*inputs).detach()
    assert not torch.equal(before, trained)
    print("Training loss:", loss.item())

    with TemporaryDirectory(prefix="rcswx-import-") as directory:
        path = Path(directory) / "trained.pt"
        save(child, path)
        restored = load(path)
        torch.testing.assert_close(restored(*inputs), trained, rtol=0, atol=0)
        assert capture(restored).architecture == result.architecture
    print("Checkpoint restored trained predictions exactly.")


if __name__ == "__main__":
    main()
