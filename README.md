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

Requires Python 3.12–3.14. For the PyTorch workflow, install in your environment:

```sh
python -m pip install 'rcswx[torch]'
```

For a uv-managed project, use `uv add 'rcswx[torch]'` instead. If you only need
structural alignment and crossover on data, install `rcswx` without the extra:
the minimal package does not install or import Torch, NumPy, or SciPy.
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

## PyTorch example

Build two small networks using the bundled einspace grammar, cross their
architectures, and run the child on a batch. Both parents map eight input
features to sixteen output features; they differ in their activation.

```python
import torch
from rcswx import Architecture
from rcswx.torch import build, crossover

input_spec = {
    "shape": [4, 8],
    "mode": "col",
    "other_shape": None,
    "other_mode": None,
    "branching_factor": 1,
    "last_im_shape": None,
}


def parent(activation):
    architecture = Architecture.from_tree(
        (
            "sequential",
            ("computation", ("linear(16)",)),
            ("computation", (activation,)),
        ),
        grammar="einspace",
        grammar_version="1",
        input_spec=input_spec,
    )
    return build(architecture, build_options={"dtype": torch.float32})


first = parent("relu")
second = parent("softmax")
child = crossover(first, second, seed=17)

with torch.no_grad():
    print("Child output shape:", tuple(child(torch.zeros(4, 8)).shape))
print("Fresh model:", child is not first and child is not second)
```

The output shape is `(4, 16)`, and the child is a fresh model. Its parameters
are newly initialized, not copied from either parent. `seed=17` controls the
structural choice, not PyTorch's weight initialization.

See the [PyTorch guide](docs/pytorch.md) for supported models, capture,
converters, device/dtype choices, and checkpoints. The
[report example](examples/torch_module_crossover.py) also exposes the selected
operations and captured architecture.

## Portable example

Work directly with architecture data when you do not need a model or Torch.
This small structural example aligns two activation trees, selects the activation
replacement, and prints the resulting child's operations:

```python
from rcswx import Architecture, apply_edits, edit_path

first = Architecture.from_tree(("computation", ("relu",)))
second = Architecture.from_tree(("computation", ("sigmoid",)))
plan = edit_path(first, second)
selection = plan.select("1")  # Apply this pair's single nontrivial edit.
child = apply_edits(plan, selection)

print("Edit distance:", plan.distance)
print("Child operations:", [node["name"] for node in child.to_dict()["nodes"]])
```

Here the edit distance is `0.5` and the child operations are
`['computation', 'relu']`. Use `plan.sample(seed=42)` instead of `plan.select(...)`
to draw a selection rather than choosing one explicitly.

Portable values contain structure and metadata, not tensors or executable
modules. The parents remain unchanged. See the [portable guide](docs/portable.md)
for explicit selection, serialization, reproducibility, and resource limits.

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

- [PyTorch guide](docs/pytorch.md) — model construction, crossover, capture, and persistence.
- [Portable guide](docs/portable.md) — architecture data, edit plans, sampling, and limits.
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
