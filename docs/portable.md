# Portable architecture guide

[Quickstart](../README.md#portable-example) · [PyTorch guide](pytorch.md) ·
[Runnable examples](../examples/README.md)

Install `rcswx` for structural alignment and edit application without Torch,
NumPy, or SciPy. Portable architectures contain data, not framework objects or
learned tensors. Use the [PyTorch integration](pytorch.md) when you want to build
an executable model from a supported grammar.

## Represent an architecture

`Architecture.from_tree` accepts nested `(operation_name, *children)` tuples.
The tree records operation occurrences and their composition; it is not an
arbitrary computation graph. Grammar name/version identify the representation,
and portable parents must use matching grammar semantics.

The structural core does not execute layers or validate tensor shapes. A
structurally valid tree is not necessarily buildable by the PyTorch adapter.

## Align, select, and apply

`distance(first, second)` returns the structural edit distance. Use
`edit_path(first, second)` when you also need a retained alignment plan and edit
selection. Distance is not a measure of weight differences, prediction quality,
or accuracy.

This example has one nontrivial edit, replacing the activation:

```python
from rcswx import Architecture, apply_edits, edit_path

first = Architecture.from_tree(("computation", ("relu",)))
second = Architecture.from_tree(("computation", ("sigmoid",)))
plan = edit_path(first, second)

from_second = apply_edits(plan, plan.select([]))
with_edit = apply_edits(plan, plan.select("1"))
print("Selected child:", [node["name"] for node in with_edit.to_dict()["nodes"]])
```

An empty selection starts from **parent two**. The plan describes edits toward
parent one's structure, not a byte-for-byte JSON replacement. Metadata and IDs
have their own preservation/application semantics.

`plan.select` accepts selected-path operation indices or a bit string over the
plan's nontrivial operations. The `"1"` above selects this example's single edit.
For larger plans, edits can have enabling/disabling dependencies; selection is
validated rather than treating every subset as applicable. Use
`plan.sample(seed=42)` to draw a native selection instead of choosing one
explicitly, then pass it to `apply_edits`.

### Plan ownership and legacy trees

Selections belong to one plan and cannot be applied to another, even if it was
created from the same parents. Portable `Architecture` values are immutable;
alignment and application do not mutate the input values.

Legacy callback-bearing trees retain the historical second-parent ID updates
during preparation. Structural changes to those trees invalidate subsequent
selection or application. Keep this distinct from the portable immutability
contract and the [fresh-module API](pytorch.md#cross-two-models).

## Sampling and replay

Native ChaCha12 sampling is the default. It uses an RCSWX RNG rather than
NumPy's global stream, and a structural seed never seeds PyTorch weight
initialization.

Native seeds are integers in `[0, 2**256)` or exactly 32 bytes. Integer seeds use
little-endian encoding. Supply either a seed for an individual draw or a shared
`NativeRng` stream for a sequence of calls.

Continuing with `plan` from the alignment example, a stream can be checkpointed
and replayed without NumPy:

```python
from rcswx import NativeRng

stream = NativeRng(42)
checkpoint = stream.state()
selection = plan.sample(rng=stream)
replay = NativeRng.from_state(checkpoint)
assert plan.sample(rng=replay).indices == selection.indices
```

The versioned replay state names the generator and its position.

### Reference numerical sampling

Install `rcswx[reference]` (or `rcswx[torch,reference]` for both optional layers),
then request `sampler="reference"`. This route uses NumPy/SciPy and NumPy's
global RNG: seed NumPy explicitly, and do not pass native `seed`/`rng` arguments.
See the runnable [reference sampling example](../examples/reference_sampling.py).

The two samplers share the discrete probability law, not seed-to-child identity.
The reference sampler operates on the **current** edit plan; it does not undo
the [intentional operator corrections](compatibility.md).

## Serialize architecture data

`Architecture.from_dict` and `from_json` validate schema version 1. `to_dict`
returns an independent mapping; `to_json` exposes the validated wire
representation.

- Child edges are occurrence indices, not logical IDs.
- Logical node IDs are canonical decimal strings. They may repeat and have no
  fixed integer width; repeated IDs do not merge distinct occurrences.
- Parameters, provenance, and input assumptions remain opaque JSON. The core
  preserves their numbers and keys rather than interpreting private metadata
  markers.

Framework callbacks, model objects, and optimizer state do not cross this
serialization boundary. For model state, use [managed PyTorch checkpoints](pytorch.md#save-and-load-trained-state).

## Bound work and memory

`edit_path(..., limits={...})` accepts `max_work`, `max_output`, and
`max_allocation_bytes` quotas. The last bounds accounted native allocations,
**not process RSS**. `profile=True` enables diagnostic stage timings; leave it
disabled for primary performance measurements.

On legacy/PyTorch trees, the `Limiter`'s `memory_crossover` RSS guard remains
host-side. Retained native operations poll it at their first checkpoint, every
1,024 checkpoints thereafter, and before returning a successful native result.
Work/output/allocation quotas still account every checkpoint in Rust; they are
not throttled with host polling. RSS checks are sampled and can overshoot a
threshold between polls, so they are not a hard process-memory ceiling.
Parallel calls additionally poll elapsed time at 10 ms intervals while waiting
for compute jobs. This does not change canonical work or allocation counters.

## Optional parallel alignment

Native wavefront execution is opt-in at build time **and** per call. See
[parallel build commands](development.md#parallel-builds) before using
`edit_path(first, second, workers=2)` or `Alignment(..., workers=2)`.

- `workers=1` is the serial default and never initializes the pool.
- `workers=-1` uses the private pool's capacity.
- Other positive integers bound this call's simultaneous compute jobs, capped
  at that capacity. Concurrent calls share the same bounded pool.
- Boolean/non-integer counts, zero, and values below `-1` are rejected before
  legacy parent preparation.
- Serial-only builds reject parallel requests. WebAssembly stays serial even
  with the Cargo feature enabled.

`plan.execution` reports capability, requested/resolved workers, pool capacity,
parallel/serial cells, job overlap, scratch reservation, and fallback counts.
Algorithm statistics and ordered plans are unchanged. Uneconomic, aliased, or
scratch-limited waves can run serially.

The coordinator handles callbacks and signals; interrupted calls join their
jobs before raising. Serial callback re-entry is supported. Parallel callback
re-entry and reuse of an initialized pool after `fork` fail promptly: use serial
execution or a fresh interpreter in a forked child.

Parallel execution helps sufficiently history-heavy waves, not every wide
matrix. Retain the serial default for unmeasured workloads. See
[wavefront design and measurements](wavefront.md) for the detailed execution
contract, performance evidence, and limits.
