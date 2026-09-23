"""Construct two small einspace parents; shared by the other examples.

Run from the repository root: python examples/parents.py
"""

import torch
from rcswx import PCFG, DerivationTreeNode, Limiter, Operation, Reconstructor
from rcswx.grammars import einspace as g

INPUT_SHAPE = (4, 8)


def make_builder() -> Reconstructor:
    """Use the bundled grammar and explicit CPU resource limits."""
    limiter = Limiter(
        {
            "time": 60,
            "restart_time": 300,
            "depth": 20,
            "max_id": 10000,
            "memory": 8196,
            "memory_crossover": 65536,
            "individual_memory": 1024,
            "batch_pass_seconds": 0.1,
        }
    )
    limiter.timer.start()
    return Reconstructor(PCFG(g.grammar, limiter), limiter, mode="iterative")


def make_parent(width: int, activation: Operation) -> DerivationTreeNode:
    """Build a Linear → activation tree using operations from the grammar.

    The width and activation must be available in g.grammar.
    """
    builder = make_builder()
    return builder.sample(
        input_params={
            "shape": torch.Size(INPUT_SHAPE),
            "mode": "col",
            "other_shape": None,
            "other_mode": None,
            "branching_factor": 1,
            "last_im_shape": None,
        },
        # Preorder grammar operations: sequential(computation, computation).
        operations=[
            g.sequential_module,
            g.computation_module,
            g.linear(width),
            g.computation_module,
            activation,
        ],
    )


def make_parents() -> tuple[DerivationTreeNode, DerivationTreeNode]:
    """Return fresh trees: Linear(8,16) → ReLU and Linear(8,32) → Softmax."""
    return make_parent(16, g.relu), make_parent(32, g.softmax)


def describe(root: DerivationTreeNode) -> str:
    """Show terminal operations for these sequential examples."""
    return " → ".join(
        node.operation.name
        for node in root.serialise()
        if node.operation.is_terminal()
    )


def main() -> None:
    a, b = make_parents()
    for label, parent in (("A", a), ("B", b)):
        print(f"Parent {label}: {describe(parent)}")
        print("  Grammar operations:", [node.operation.name for node in parent.serialise()])
        model = parent.build(parent)
        with torch.no_grad():
            output = model(torch.zeros(INPUT_SHAPE))
        print("  Output shape:", tuple(output.shape))


if __name__ == "__main__":
    main()
