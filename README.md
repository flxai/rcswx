# rcswx

RCSWX reimplements the Recursive Constrained Smith–Waterman crossover algorithm
for grammar-based neural architecture search proposed in
[Evolutionary Architecture Search through Grammar-Based Sequence Alignment](https://arxiv.org/abs/2512.04992)
(see the [original implementation](https://github.com/flxai/rcswx-paper)).
Its portable Rust-backed API aligns and applies edits to architecture data,
with optional PyTorch integration.

## Installation

RCSWX supports Python 3.12–3.14. Its minimal package has the native Python
binding and portable architecture API only: it does not install or import
Torch, NumPy, or SciPy.

```sh
uv add rcswx                         # Portable alignment/application only.
uv add 'rcswx[torch]'                # PyTorch grammar, capture, and module APIs.
uv add 'rcswx[reference]'            # NumPy/SciPy reference sampler only.
uv add 'rcswx[torch,reference]'      # Both optional surfaces.
```

Equivalent `uv pip install` or `python -m pip install` commands work in an
active virtual environment. Source builds require Rust/Cargo.

For source development, first enter the locked Nix shell on NixOS, then install
both optional surfaces deliberately:

```sh
nix develop
make sync
make develop
```

`make sync`, `make develop`, and `make test` pass `--all-extras`; a plain
minimal wheel install remains a separately tested contract.

## Portable API and sampling

Inputs are data-only grammar occurrence trees, not arbitrary TensorFlow,
Keras, or PyTorch models. A structural application does not need a scientific
Python stack:

```python
from rcswx import Architecture, apply_edits, edit_path

first = Architecture.from_tree(("sequential", ("relu",)))
second = Architecture.from_tree(("sequential", ("sigmoid",)))
explicit_plan = edit_path(first, second)

unchanged = apply_edits(explicit_plan, explicit_plan.select([]))
seeded_plan = edit_path(first, second)
seeded_child = apply_edits(seeded_plan, seeded_plan.sample(seed=42))
```

`Architecture.from_dict`/`from_json` validate schema version 1; `to_dict` and
`to_json` expose its wire representation. Child edges are occurrence indices.
Logical node IDs are canonical decimal strings, may repeat, and have no fixed
integer width. Parameters, provenance, and input assumptions remain opaque JSON;
the core preserves their numbers and keys rather than interpreting private
metadata markers.

An explicit empty selection starts from parent two. Selections belong to one
plan; they cannot be applied to another plan. Legacy trees retain historical
second-parent ID updates, and structural changes to those trees invalidate
subsequent selection or application. Portable architectures are immutable.

`edit_path(..., limits={...})` accepts `max_work`, `max_output`, and
`max_allocation_bytes` quotas. The last bounds accounted native allocations, not
process RSS. `profile=True` enables diagnostic stage timings; leave it disabled
for primary performance measurements.

Native wavefront execution is opt-in at build time and per call:

```sh
make develop-parallel  # Or: make wheel-parallel; install that exact wheel.
make parallel-check parallel-test
```

Then use `edit_path(first, second, workers=2)` (or `Alignment(..., workers=2)`).
The default `workers=1` is serial and never initializes the pool. `workers=-1`
uses the private pool's capacity; other positive integers bound this call's
simultaneous compute jobs, capped at that capacity. Concurrent calls share the
same bounded pool. Boolean/non-integer counts, zero, and values below `-1` are
rejected before legacy parent preparation. Serial-only builds reject parallel
requests; WebAssembly stays serial even with the Cargo feature enabled.

`plan.execution` reports capability, requested/resolved workers, pool capacity,
parallel/serial cells, job overlap, scratch reservation and fallback counts.
Algorithm statistics and ordered plans are unchanged. Uneconomic, aliased, or
scratch-limited waves can run serially. The coordinator handles callbacks and signals; interrupted calls
join their jobs before raising. Serial callback re-entry is supported; parallel
callback re-entry and reuse of an initialized pool after `fork` fail promptly.
Use serial execution or a fresh interpreter in a forked child. Parallel execution
helps sufficiently history-heavy waves, not every wide matrix; retain the serial
default for unmeasured workloads.
See [wavefront design and measurements](docs/wavefront.md) for the execution
contract, performance evidence, and limits.

On legacy/PyTorch trees, the `Limiter`'s `memory_crossover` RSS guard remains
host-side. Retained native operations poll it at their first checkpoint, every
1,024 checkpoints thereafter, and before returning a successful native result.
Work/output/allocation quotas still account every checkpoint in Rust; they are
not throttled with host polling. RSS checks are sampled and can overshoot a
threshold between polls, so they are not a hard process-memory ceiling.
Parallel calls additionally poll elapsed time at 10 ms intervals while waiting
for compute jobs. This does not change canonical work or allocation counters.

Native ChaCha12 sampling is the default. It is seeded through RCSWX and does
not consume NumPy's global RNG; a structural seed never seeds PyTorch weight
initialization. Select the historical numerical route explicitly with
`sampler="reference"` after installing `rcswx[reference]`.

Native seeds are integers in `[0, 2**256)` or exactly 32 bytes; integer seeds use
little-endian encoding. A stream can be shared across calls and checkpointed
without importing NumPy:

```python
from rcswx import NativeRng

stream = NativeRng(42)
checkpoint = stream.state()
selection = seeded_plan.sample(rng=stream)
replay = NativeRng.from_state(checkpoint)
assert seeded_plan.sample(rng=replay).indices == selection.indices
```

The versioned replay state names the generator and its position. Reference mode
uses NumPy's global RNG instead: seed NumPy explicitly and do not pass native
`seed`/`rng` arguments with `sampler="reference"`. The two samplers share the
discrete probability law, not seed-to-child identity.

Tree-level crossover and module-level crossover are separate APIs. A module
crossover creates a fresh child without transferring learned weights, buffer
values, or optimizer state; provenance capture is supported, but arbitrary
unannotated-model import is not. The structural core is portable, while
TensorFlow/Keras integration and a browser UI are not delivered here.

See [examples/](examples/) for the portable example, PyTorch construction,
explicit edit selection, tree crossover, capture/build, and converter examples.

## Reference compatibility

RCSWX is a corrected port, not a bug-for-bug replica of the original
`einsearch` implementation. The independent oracle is pinned to commit
`3f44ddf086bee0c213404e240ee0adf99e3e1501` and hash-checked by
[`tests/reference/original.py`](tests/reference/original.py).

The following corrections deliberately differ from that source:

- Recursive edits and wrapper boundaries use physical source occurrences rather
  than remapped trace coordinates. Swap bookkeeping distinguishes wrappers even
  when their logical IDs repeat.
- Adding a binary wrapper requires added material only in a branch that would
  otherwise be empty. The original can also require an addition in the retained
  branch, including an impossible empty enabler group.
- Zero-cost matches do not replace unchanged modules, routing functions, or
  wrapper payloads. When an accompanying binary-wrapper match requires a branch
  reorientation, application relinks its children and rebinds its anchor ID
  without copying the matched subtree from parent one.

These corrections can change edit application order, valid selections, offspring,
and failure behavior. `sampler="reference"` selects the historical NumPy/SciPy
numerical route on the **current** edit plan; it does not restore the original
operator's bugs or guarantee historical seed-to-child identity.

Reference tests retain exact comparisons for unchanged behavior. The known
branch-constraint correction is explicit in the oracle comparison; separate
regressions check wrapper boundaries, ordered topology, and payload ownership.
Passing those checks is not a claim of universal original-output parity or
tensor-level validity for every generated architecture.

## PyTorch capture and persistence

`rcswx.torch.build(architecture, input_spec=...)` creates a fresh model and
attaches data-only structural provenance. `capture(model)` recovers it and
rejects stale topology, shared modules, shared parameter/buffer objects, and
distinct tensors sharing backing storage. External modules need an explicitly
registered converter; unsupported structures are not approximated.

For retained RCSWX models, `build(capture(model))` preserves captured per-module
train/eval modes and per-tensor dtype, device, and gradient requirements; it
initializes fresh parameters and buffers.
Explicit `BuildOptions` override device, floating/complex dtype, or training
mode. `rcswx.torch.crossover` and `crossover_with_report` capture both parents,
call the portable engine, and build a fresh child without retries or a hidden
validation forward pass. Their default is compatible parent materialization
and parent one's **root** training mode; ambiguous dtype/device layouts require
an explicit build option.

Use `rcswx.torch.save`/`load` for a managed checkpoint containing a validated
manifest and state dictionary. Loading rebuilds only supported registered
grammar/converter structures and uses Torch's weights-only loader. Capture
describes architecture, not optimizer state or an arbitrary executable model.

## Wheel and WebAssembly portability gates

Build one exact wheel, then test that exact artifact in fresh minimal, Torch,
reference, and combined environments. The gate checks metadata, missing-extra
errors, root import isolation, and runnable examples; it never accepts an
editable import as proof.

```sh
make wheel
RCSWX_WHEEL=dist/rcswx-0.5.2-...whl RCSWX_PYTHON=python3.12 make wheel-test
```

For a parallel-capable artifact, use `make wheel-parallel`, then
`RCSWX_WHEEL=dist/parallel/rcswx-0.5.2-...whl make wheel-parallel-test`.
That gate verifies capability and serial/parallel plan parity in every surface.
`make wasm-parallel-check` executes the core and frozen oracle in Node with the
feature enabled, proving that native threads do not enter the WASM dependency
graph.

The Rust workspace also builds a reusable browser package and an interactive
worker-owned example. It runs the existing core, with deterministic slider
prefixes, seeded sampling, and bounded recordings of actual recursive alignment:

```sh
nix develop
make web-demo  # open the displayed /rcswx/ URL
make web-test  # exact package in Chromium, Firefox, and WebKit
```

See [WASM consumer contracts](docs/wasm.md) and the
[standalone example](examples/web/README.md). The original development-only
core gate remains separate: `make wasm-check` compiles the core—not PyO3—for
`wasm32-unknown-unknown`, and `make wasm-smoke` executes its
decode/analyze/full-history/apply, probability, seeded draw and raw RNG-vector
smoke in Node.

The workspace declares Rust 1.85 as its MSRV. The Nix shell pins Rust 1.85.0
with the WASM target and supplies the local `wasm-bindgen-test-runner` and Node;
it does not install a global target or runner. The portable core's target
libraries are serialization (`serde`/`serde_json`), arbitrary IDs
(`num-bigint`), `libm`, and `rand_chacha`/`rand_core`. Their installed Cargo
manifests declare MIT for `libm` and MIT OR Apache-2.0 for the other listed
libraries. The dev-only execution harness pins `wasm-bindgen-test` 0.3.77, whose `wasm-bindgen` 0.2.127 matches the locked Nix CLI; its manifest declares MIT OR Apache-2.0.
Native parallel builds additionally link Rayon and its MIT OR Apache-2.0
dependencies. Wheels retain the original algorithm attribution and the verbatim
numerical and optional parallel
dependency notices in [LICENSE.rust-dependencies](LICENSE.rust-dependencies).

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
