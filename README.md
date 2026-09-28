# rcswx

**Compare neural architectures and generate offspring by recombining their
structure.** RCSWX provides a Python API for structural edit distances, alignment
plans, and crossover in grammar-based neural architecture search.

Use it to inspect how two architectures differ, choose or sample edits, and
build a fresh PyTorch model from the resulting architecture. Training, evaluation,
and population management remain part of your own research code.

The core consumes architecture trees. The PyTorch integration can **import
ordinary `nn.Module` computations that fit the existing `einspace` grammar**,
verify their reconstruction, and return a validated fresh child. RCSWX-built
models and explicit converters remain supported. No workflow inherits weights.

The Rust core implements Recursive Constrained Smith–Waterman crossover from
[Evolutionary Architecture Search through Grammar-Based Sequence Alignment](https://arxiv.org/abs/2512.04992).
For the original implementation and paper experiments, see
[rcswx-paper](https://github.com/flxai/rcswx-paper).

## Installation

Requires Python 3.12–3.14. Start with the minimal package:

```sh
python -m pip install rcswx
```

For a uv-managed project, use `uv add rcswx` instead. The minimal package does
not install or import Torch, NumPy, or SciPy. Add `rcswx[torch]` when you want
executable models, as in the PyTorch example below.
Source builds require Rust/Cargo; see the [development guide](docs/development.md)
for working from a checkout and [Nix usage](examples/README.md#nix-flakes).

`help(rcswx)` works without extras. Root introspection lists the portable API
and already-loaded attributes; optional legacy exports remain available through
explicit imports.

## How it works

```text
Two parent architecture trees
             ↓
    Align their structures
             ↓
          Edit plan
             ↓
    Select or sample edits
             ↓
    Offspring architecture
             ↓
  Build a fresh PyTorch model (optional)
```

The edit plan describes changes from the second parent toward the first.
Selecting no edits starts from the second parent. Edits can have dependencies,
so selection is constrained rather than an arbitrary mix of layers.
The distance measures structural edits—not differences in weights, predictions,
or accuracy.

## Portable example

Start with two small networks: `Linear(…, 16) → ReLU` and
`Linear(…, 32) → Softmax`. No Torch installation or input tensors are needed to
compare and recombine their structure.

```python
from rcswx import Architecture, apply_edits, edit_path


def architecture(width, activation):
    return Architecture.from_tree(
        (
            "sequential",
            ("computation", (f"linear({width})",)),
            ("computation", (activation,)),
        )
    )


first = architecture(16, "relu")
second = architecture(32, "softmax")
plan = edit_path(first, second)

for bit, edit in enumerate(plan.nontrivial_ops):
    print(f"Bit {bit}: {edit}")
selection = plan.select("10")  # Replace the activation, keep parent two's width.
child = apply_edits(plan, selection)

print("Full distance:", plan.distance, "Selected cost:", selection.cost)
print("Child operations:", [node["name"] for node in child.to_dict()["nodes"]])
```

Each tuple is `(operation_name, *children)`. The bundled `einspace` grammar
(version `"1"`, the default) uses `sequential` for composition and `computation`
to wrap a layer. Here, `linear(16)` names a linear layer with 16 output features.

This plan has two weighted edits: activation replacement costs `0.5`, and width
replacement costs `0.25`. The full distance is `0.75`, **not a count of changed
layers**. Selecting only the first edit costs `0.5` and produces
`['sequential', 'computation', 'linear(32)', 'computation', 'relu']`:
**parent two's width with parent one's activation**.

The mask is specific to this plan's displayed edit order. Use `plan.sample(seed=0)`
instead to sample a valid selection. Portable values contain no learned tensors,
and both parents remain unchanged. See the [portable guide](docs/portable.md)
for selection, serialization, reproducibility, and resource limits.

## PyTorch example

Install the optional integration:

```sh
python -m pip install 'rcswx[torch]'
```

Start from ordinary PyTorch modules; no hand-written architecture tree is needed:

```python
import torch
from torch import nn
from rcswx.torch import crossover_imported, import_model

torch.manual_seed(0)  # Weight initialization; separate from structural sampling.
examples = (torch.randn(4, 8),)
first_model = nn.Sequential(nn.Linear(8, 16), nn.ReLU())
second_model = nn.Sequential(nn.Linear(8, 32), nn.Softmax(dim=-1))
first = import_model(
    first_model, example_inputs=examples, grammar="einspace", grammar_version="1"
)
second = import_model(
    second_model, example_inputs=examples, grammar="einspace", grammar_version="1"
)
result = crossover_imported(first, second, validation_inputs=examples, seed=0)
child_model = result.child

print("Selected edits:", result.report["crossover_operations"])
print("Validation:", result.report["validation"])
print(child_model)
print("Output shape:", tuple(child_model(*examples).shape))
```

For this pair, `seed=0` selects the activation edit: the child is
`Linear(8, 32) → ReLU`, with output shape `(4, 32)`. It has fresh parameters, not
inherited weights. Import verifies source/reconstruction behavior using
corresponding state on disposable copies; crossover validates the actual sampled
child's grammar, shape, and runtime contract without changing native selection.
Unsupported models and invalid offspring raise: there is no approximation,
repair, or retry.

The initial backend is unchanged `einspace` version `"1"`: supported dense and
convolutional configurations, normalization, and structured binary add/cat
branches—not arbitrary PyTorch graphs. See the
[support matrix and validation limits](docs/pytorch.md#import-an-ordinary-pytorch-model).
The [complete runnable example](examples/torch_graph_import.py) trains the mixed
child and restores its trained predictions from a managed checkpoint.
For existing architecture trees or legacy no-forward module crossover, see the
[PyTorch guide](docs/pytorch.md).

## Important semantics

- **Supported representations:** this is not a general importer for arbitrary
  PyTorch, TensorFlow, or Keras models.
- **Fresh model state:** module crossover does not transfer learned weights,
  buffer values, or optimizer state.
- **Validate for your task:** `import_model` and `crossover_imported` explicitly
  verify reconstruction or offspring validity on supplied inputs. The existing
  `capture`, `build`, `crossover`, and `crossover_with_report` APIs remain
  no-forward operations. The new entrypoint never repairs or retries an invalid child.
- **Reproducibility:** native sampling has its own seeded RNG. Seed PyTorch
  separately for weight initialization; see [sampling and replay](docs/portable.md#sampling-and-replay).
- **Reference compatibility:** RCSWX is a corrected port, not a bug-for-bug
  replica. The optional reference sampler does not restore the original
  operator's bugs or promise historical seed-to-child identity. See
  [compatibility details](docs/compatibility.md).

## Documentation

- [Portable guide](docs/portable.md) — represent architectures, choose edits, and replay or save results.
- [PyTorch guide](docs/pytorch.md) — build and cross models, import a supported module, and save trained state.
- [Runnable examples](examples/README.md) — small CPU scripts; no dataset or training required.
- [Reference compatibility](docs/compatibility.md) — intentional corrections and comparison scope.
- [Development](docs/development.md) — source setup, builds, and verification.

## Cite

```bibtex
@inproceedings{gomez2026evolutionary,
  author    = {Gómez Martín, Adri and Möller, Felix and McDonagh, Steven and
               Abella, Monica and Desco, Manuel and Crowley, Elliot J. and
               Klein, Aaron and Ericsson, Linus},
  title     = {Evolutionary Architecture Search through Grammar-Based Sequence Alignment},
  booktitle = {International Conference on Automated Machine Learning (AutoML)},
  year      = {2026},
  url       = {https://arxiv.org/abs/2512.04992}
}
```

## License

[MIT](LICENSE.einsearch)
