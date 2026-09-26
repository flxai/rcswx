# Examples

The examples are small CPU demonstrations. No example needs a dataset, GPU, or
an `einsearch` checkout. [portable.py](portable.py) uses only the minimal
architecture API; [reference_sampling.py](reference_sampling.py) additionally
needs `rcswx[reference]`, but not Torch. The framework examples need `rcswx[torch]`.

## Examples at a glance

| File | Requires | Demonstrates |
| --- | --- | --- |
| [portable.py](portable.py) | minimal | Data-only trees, explicit selection/application, and native seeded sampling. |
| [reference_sampling.py](reference_sampling.py) | `reference` | Explicit reference selection using NumPy's global seed, without Torch. |
| [parents.py](parents.py) | `torch` | Construct fresh parent trees, build their PyTorch models, and check shapes. |
| [distance.py](distance.py) | `torch` | Parent and self edit distances. |
| [edit_path.py](edit_path.py) | `torch` | Ordered histories and nontrivial edits. |
| [crossover.py](crossover.py) | `torch` | Tree crossover, reconstruction, and a model forward pass. |
| [crossover_report.py](crossover_report.py) | `torch` | Selection and crossover report fields. |
| [validated_crossover.py](validated_crossover.py) | `torch` | Reconstruction plus a validation forward pass. |
| [torch_module_crossover.py](torch_module_crossover.py) | `torch` | Build, capture, and module-level crossover. |
| [torch_converter.py](torch_converter.py) | `torch` | Register an exact `nn.Sequential` Linear/ReLU converter. |
| [nix/flake.nix](nix/flake.nix) | `torch` | Add the Torch-enabled Nix variant to a consuming project. |
| [web/](web/README.md) | tracked Nix/Node/WASM toolchain | Browser worker, exact edit-prefix slider, recorded recursive alignment, and seeded native sampling. |

Keep `parents.py` beside the framework examples: they import its construction
helpers. These scripts live in the repository; an installed wheel does not
install the example files.

## Run from source

Requires Python 3.12–3.14, Rust/Cargo, and [uv](https://docs.astral.sh/uv/).
On NixOS, enter `nix develop` first.

The minimal surface has no scientific-Python dependency:

```sh
uv sync --locked
uv run --locked python examples/portable.py
```

Install the PyTorch extra explicitly for the framework examples:

```sh
uv sync --locked --extra torch
uv run --locked --extra torch python examples/distance.py
```

Replace `distance.py` with any other framework example. Development contributors
can instead run `make sync`, which deliberately installs both optional extras.

## Run from a published wheel

Use the extra that matches the script:

```sh
python -m pip install rcswx
python examples/portable.py

python -m pip install 'rcswx[torch]'
python examples/crossover.py
```

For a uv-managed consuming project, use `uv add rcswx` or
`uv add 'rcswx[torch]'`, then run the scripts with `uv run`. Do not add `rcswx`
as a dependency of its own source project. The reference numerical selector is
separate: install `rcswx[reference]` and request it with `sampler="reference"`;
`reference_sampling.py` demonstrates that route. Native-default examples do not
need the reference extra.

## Reading the results

`distance` measures architecture edits, not weight differences, predictions, or
accuracy. `edit_path` returns an alignment/plan whose operations can have
enabling and disabling dependencies. `crossover` constructs a tree and the
framework examples rebuild a fresh PyTorch model; they do not transfer learned
weights. Native selection is deterministic for an RCSWX seed but does not use
or promise NumPy's global RNG stream.

## Nix flakes

The provider overlay exposes minimal and optional variants on every supported
Python set: `ps.rcswx`, `ps.rcswx-torch`, `ps.rcswx-reference`, and
`ps.rcswx-full`. The included [nix/flake.nix](nix/flake.nix) selects
`ps.rcswx-torch` for the framework scripts. It targets `x86_64-linux`; validate
other systems independently.
