# Examples

Small CPU examples using the bundled einspace grammar. Each script constructs fresh parents; no dataset, training, GPU, or `einsearch` checkout is required.

## Examples at a glance

| File | Demonstrates |
| --- | --- |
| [parents.py](parents.py) | Construct two parent trees, build their PyTorch models, and check output shapes. Also provides shared setup for the other scripts. |
| [distance.py](distance.py) | Compute the edit distance between parents and the distance from a parent to itself. |
| [edit_path.py](edit_path.py) | Inspect the alignment's ordered operations and the nontrivial edits available for crossover. |
| [crossover.py](crossover.py) | Generate an offspring tree, reconstruct its metadata, and build and run its PyTorch model. |
| [crossover_report.py](crossover_report.py) | Inspect the selected edits and the distances reported by `crossover_with_report`. |
| [validated_crossover.py](validated_crossover.py) | Reconstruct and check offspring with `validated_crossover` before building a model. |
| [nix/flake.nix](nix/flake.nix) | Add `rcswx` to a consuming project's Python environment. |

## Run

With `rcswx` installed in your active Python environment, run these commands from the repository root:

```sh
python examples/parents.py
python examples/distance.py
python examples/edit_path.py
python examples/crossover.py
python examples/crossover_report.py
python examples/validated_crossover.py
```

Keep `parents.py` beside the other scripts: they import its construction helpers. The scripts live in this repository; installing the library alone does not provide these example files.

### From source

Requires Python 3.12–3.14, Rust/Cargo, and [uv](https://docs.astral.sh/uv/). In a checkout of this repository:

```sh
uv sync --locked
uv run --locked python examples/distance.py
```

Replace `distance.py` with any example above. On NixOS, enter the repository's `nix develop` shell first.

### From PyPI

After a release is published, install `rcswx` in an active virtual environment:

```sh
python -m pip install rcswx
python examples/distance.py
```

Or let uv create a temporary environment using the published package, rather than building this checkout:

```sh
uv run --no-project --with rcswx python examples/distance.py
```

For a uv-managed consuming project, use [`uv add rcswx`](https://docs.astral.sh/uv/guides/projects/#managing-dependencies), then run with `uv run python ...`. Do not add `rcswx` as a dependency of its own source project.

## The parents

[parents.py](parents.py) constructs these networks for a batch of shape `(4, 8)`:

| Parent | Layers | Output shape |
| --- | --- | --- |
| A | `Linear(8, 16) → ReLU` | `(4, 16)` |
| B | `Linear(8, 32) → Softmax(dim=-1)` | `(4, 32)` |

Construction uses grammar operations and `Reconstructor`, not conversion from an existing `torch.nn.Module`. `make_parents()` creates both trees; `make_builder()` provides the same grammar for offspring reconstruction. Resource limits are collected in `make_builder()`; time limits are seconds and memory limits are MiB.

## Reading the results

[`distance`](https://github.com/flxai/rcswx/blob/main/python/rcswx/api.py) measures architecture edits, not weight differences, prediction differences, or accuracy. The scripts print computed results rather than relying on hard-coded expected distances.

[`edit_path`](https://github.com/flxai/rcswx/blob/main/python/rcswx/api.py) returns an `Alignment` object, not a list: inspect `.distance`, `.operations`, and `.nontrivial_ops`. Printed operations include their costs and any enabling/disabling dependencies. They are not necessarily independent edits.

[`crossover`](https://github.com/flxai/rcswx/blob/main/python/rcswx/crossover.py) returns a tree. `crossover.py` rebuilds its metadata with `Reconstructor.re_id`, then constructs a fresh PyTorch model. Weights are not inherited, and the output width can change. For more general parents, `validated_crossover.py` demonstrates reconstruction and a test forward pass through the validation API; validation can still fail.

[`crossover_with_report`](https://github.com/flxai/rcswx/blob/main/python/rcswx/crossover.py) also reports the selected edits and distances. Its child-to-parent distances are derived from selected edit costs, not separately recomputed `distance()` calls.

The crossover scripts seed NumPy to make selection repeatable within the same software environment. This does not promise identical results across package versions or frameworks. The library may print additional diagnostic messages.

## Keep parents unchanged

The examples use fresh parents and do not need to preserve them. To keep your own parents unchanged:

```python
from copy import deepcopy
from rcswx import crossover

child = crossover(deepcopy(a), deepcopy(b))
```

The same applies to `distance` and `edit_path`. Alignment changes node IDs in the second parent; crossover can return the first parent directly when no edits are needed. These are trees, so copying them does not copy trained PyTorch models.

## Nix Flakes

The repository exports `rcswx.overlays.default`, which adds `ps.rcswx` to each supported Python package set. [nix/flake.nix](nix/flake.nix) is a consuming-project example; it does not implement the provider overlay.

Copy `nix/flake.nix` into your own project's root, then run:

```sh
nix develop
python3 -c 'import rcswx; print(rcswx.__version__)'
```

Add your other libraries beside `ps.rcswx` in `python314.withPackages`. This example targets `x86_64-linux`; other platforms need separate validation. See the [Nixpkgs Python documentation](https://nixos.org/manual/nixpkgs/unstable/#python).
