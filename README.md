# rcswx

**Compare neural architectures and generate offspring by recombining their
structure.** RCSWX provides a Python API for structural edit distances, alignment
plans, and crossover in grammar-based neural architecture search.

Use it to inspect how two architectures differ, choose or sample edits, and
build a fresh PyTorch model from the resulting architecture. Training, evaluation,
and population management remain part of your own research code.

**Inputs are architecture trees, not arbitrary trained models.** Each tree
records the operations and composition rules used to construct a network.
PyTorch integration supports RCSWX-built models and explicitly registered
converters; crossover does **not** inherit learned weights.

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

**Continue with `first` and `second` from above.** Build them as PyTorch modules,
cross the modules, then inspect and run the child:

```python
import torch
from rcswx.torch import build, crossover_with_report

input_spec = {
    "shape": [4, 8],  # A batch of four examples with eight input features.
    "mode": "col",   # Features occupy the last dimension.
    "other_shape": None,
    "other_mode": None,
    "branching_factor": 1,
    "last_im_shape": None,
}

torch.manual_seed(0)  # Weight initialization; separate from structural sampling.
first_model = build(first, input_spec=input_spec)
second_model = build(second, input_spec=input_spec)
result = crossover_with_report(first_model, second_model, seed=0)
child_model = result.child

print("Selected edits:", result.report["crossover_operations"])
print(child_model)
child_model.eval()
with torch.no_grad():
    predictions = child_model(torch.randn(4, 8))
print("Output shape:", tuple(predictions.shape))
```

For this pair, `seed=0` selects the activation edit: the printed module contains
`Linear(in_features=8, out_features=32, bias=True)` followed by `ReLU()`, and the
output shape is `(4, 32)`. This is a fresh `torch.nn.Module` with newly initialized
parameters—not inherited weights. Train it with your usual PyTorch loop.

The remaining `input_spec` fields describe branch and image-layout bookkeeping;
the [input specification guide](docs/pytorch.md#adapt-the-input-specification)
explains every field and how to adapt the example. For an existing
`nn.Sequential` model, see the
[explicit converter recipe](docs/pytorch.md#import-a-supported-nnsequential-model);
arbitrary modules are not automatically importable. The
[standalone module example](examples/torch_module_crossover.py) includes all setup.

## Important semantics

- **Supported representations:** this is not a general importer for arbitrary
  PyTorch, TensorFlow, or Keras models.
- **Fresh model state:** module crossover does not transfer learned weights,
  buffer values, or optimizer state.
- **Validate for your task:** structural validity is not a universal guarantee
  of tensor-shape validity. Module crossover does not run a hidden validation
  forward pass or retry loop.
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
