# Portable architecture guide

[Quickstart](../README.md#portable-example) · [PyTorch guide](pytorch.md) ·
[Runnable examples](../examples/README.md)

Install `rcswx` for structural alignment and edit application without Torch,
NumPy, or SciPy. These recipes represent two networks, inspect their edits,
choose or sample a child, and save its architecture. Run the snippets in order.

## Represent an architecture

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
```

`Architecture.from_tree` accepts `(operation_name, *children)` tuples. A leaf
such as `("relu",)` is a one-element tuple; the trailing comma matters.
The tree records operation occurrences and their composition, not an arbitrary
computation graph. This example uses the default `einspace` grammar, version
`"1"`:

- `("computation", ("linear(16)",))` is a linear layer with 16 output features.
- `("computation", ("relu",))` is a ReLU activation.
- `("sequential", first_subtree, second_subtree)` runs two subtrees in order.

Thus, `first` describes `Linear(…, 16) → ReLU`, and `second` describes
`Linear(…, 32) → Softmax`. Change the function arguments to vary the width and
activation. Input width is a separate build-time assumption; the
[PyTorch guide](pytorch.md#adapt-the-input-specification) explains that step.

Grammar name/version identify the representation; parents must use matching
grammar semantics. The structural core does not execute layers or validate
tensor shapes. A structurally valid tree is not necessarily buildable by the
PyTorch adapter, and arbitrary names are not automatically available layers.

## Align, select, and apply

Inspect the plan before choosing edits:

```python
plan = edit_path(first, second)
print("Full distance:", plan.distance)
for bit, edit in enumerate(plan.nontrivial_ops):
    print(f"Bit {bit}: {edit}")

selection = plan.select("10")
child = apply_edits(plan, selection)
print("Selected cost:", selection.cost)
print("Child operations:", [node["name"] for node in child.to_dict()["nodes"]])
```

For this pair, bit 0 replaces `softmax` with `relu` (cost `0.5`), and bit 1
replaces `linear(32)` with `linear(16)` (cost `0.25`). The full distance is `0.75`.
These `computation` mutations distinguish a different operation from a changed
parameter of the same operation. Costs are operator weights, **not counts of
changed layers**, FLOPs, weight differences, or predictive quality.

The plan describes edits **from parent two toward parent one**. Its masks give:

| Mask for this plan | Child computation | Selected cost |
| --- | --- | --- |
| `"00"` | `Linear(…, 32) → Softmax` (parent two's structure) | `0` |
| `"10"` | `Linear(…, 32) → ReLU` (mixed) | `0.5` |
| `"01"` | `Linear(…, 16) → Softmax` (mixed) | `0.25` |
| `"11"` | `Linear(…, 16) → ReLU` (parent one's structure) | `0.75` |

`plan.select([])` is another way to choose no edits. It starts from parent two's
structure, not a byte-for-byte JSON replacement; metadata and IDs have their own
preservation/application semantics.

A mask follows `plan.nontrivial_ops` order. Do not reuse this example's mask for
an unrelated plan. `plan.select` also accepts selected-path operation indices,
which are **not** the numbered bit positions printed above. Edits can have
enabling/disabling dependencies: this pair has two independent changes, but not
every subset of a larger plan is applicable. Selection is validated.

If only the distance is needed, use `rcswx.distance(first, second)` instead of
retaining an edit plan. To sample a valid selection instead of choosing one:

```python
sampled_selection = plan.sample(seed=0)
sampled_child = apply_edits(plan, sampled_selection)
print("Sampled edits:", list(sampled_selection.operations))
print("Sampled child:", [node["name"] for node in sampled_child.to_dict()["nodes"]])
```

For this pair and seed, sampling also produces `Linear(…, 32) → ReLU`.

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

For a sequence of draws, keep a `NativeRng` stream instead of reseeding every
call. Checkpoint its state to replay a later selection:

```python
from rcswx import NativeRng

stream = NativeRng(42)
checkpoint = stream.state()
selection = plan.sample(rng=stream)
replay = NativeRng.from_state(checkpoint)
assert plan.sample(rng=replay).indices == selection.indices
```

The versioned replay state names the generator and its position. Native seeds
are integers in `[0, 2**256)` or exactly 32 bytes; integer seeds use little-endian
encoding. Supply either `seed` for an individual draw or `rng` for a shared stream.

### Reference numerical sampling

Install `rcswx[reference]` (or `rcswx[torch,reference]` for both optional layers),
then request `sampler="reference"`. This route uses NumPy/SciPy and NumPy's
global RNG: seed NumPy explicitly, and do not pass native `seed`/`rng` arguments.
See the runnable [reference sampling example](../examples/reference_sampling.py).

The two samplers share the discrete probability law, not seed-to-child identity.
The reference sampler operates on the **current** edit plan; it does not undo
the [intentional operator corrections](compatibility.md).

## Serialize architecture data

Save a child for another process or a later experiment:

```python
from pathlib import Path

Path("child.json").write_text(child.to_json(), encoding="utf-8")
restored = Architecture.from_json(Path("child.json").read_text(encoding="utf-8"))
assert restored.to_dict() == child.to_dict()
```

This is structure and metadata, not a model checkpoint. Framework callbacks,
model objects, learned tensors, and optimizer state do not cross this boundary.
For trained values, use [managed PyTorch checkpoints](pytorch.md#save-and-load-trained-state).

`from_dict` and `from_json` validate schema version 1; `to_dict` returns an
independent mapping. When inspecting or exchanging the wire representation:

- Child edges are occurrence indices, not logical IDs.
- Logical node IDs are canonical decimal strings. They may repeat and have no
  fixed integer width; repeated IDs do not merge distinct occurrences.
- Parameters, provenance, and input assumptions remain opaque JSON. The core
  preserves their numbers and keys rather than interpreting private metadata
  markers.

## Bound work and memory

For larger searches, set per-alignment quotas rather than letting every parent
pair consume unbounded resources. The values here are illustrative budgets:

```python
bounded_plan = edit_path(
    first,
    second,
    limits={
        "max_work": 100_000,
        "max_output": 10_000,
        "max_allocation_bytes": 64 * 1024 * 1024,
    },
)
```

`max_allocation_bytes` bounds accounted native allocations, **not process RSS**.
On legacy/PyTorch trees, the `Limiter`'s `memory_crossover` guard samples host RSS
and can overshoot between polls. Neither is a hard process-memory ceiling.
See the [resource-check contract](wavefront.md#resource-checks-and-host-callbacks)
for accounting and polling details. `profile=True` adds diagnostic stage timings;
leave it disabled for primary performance measurements.

## Optional parallel alignment

Keep serial execution for unmeasured workloads: parallel alignment helps
sufficiently history-heavy waves, not every wide matrix. It is opt-in at build
time **and** per call. After following the
[parallel build commands](development.md#parallel-builds), request
`edit_path(first, second, workers=2)`.

`workers=1` is the serial default; `workers=-1` uses the private pool's capacity;
other positive integers bound that call's simultaneous compute jobs, capped at
capacity. Concurrent calls share the pool. Serial-only builds reject parallel
requests, and WebAssembly remains serial.

Use `plan.execution` to inspect the resolved workers and actual parallel work;
a request for multiple workers does not guarantee every wave runs in parallel.
After `fork`, do not reuse an initialized pool: use serial execution or a fresh
interpreter. The [wavefront reference](wavefront.md) covers execution diagnostics,
callback re-entry, scheduler fallbacks, and performance evidence.

## Typed ordered-tree diagnostics

`rcswx.compare_typed(first, second)` is a separate JSON-only native capability.
It accepts non-neural semantic occurrence trees rather than `Architecture`
values or einspace operation names. The minimal package includes it; no Torch,
NumPy, SciPy, network fetch, or Python scoring fallback is used.

```python
from rcswx import compare_typed


def amount(identifier, value):
    return {
        "version": "rcswx-typed-ordered-v1",
        "root": identifier,
        "nodes": [
            {
                "id": identifier,
                "kind": "literal",
                "semantic_type": "integer",
                "role": "amount",
                "value": value,
                "children": [],
            }
        ],
    }


result = compare_typed(amount("first-amount", 3), amount("second-amount", 2))
assert result["distance"] == 1
assert result["edits"] == [
    {
        "operation": "substitute",
        "first_id": "first-amount",
        "second_id": "second-amount",
        "cost": 1,
    }
]
```

### Wire and cost contract

Each tree has exactly `version`, `root`, and `nodes`. Each node has exactly
`id`, `kind`, `semantic_type`, `role`, `value`, and `children`:

- `version` is `"rcswx-typed-ordered-v1"`; `root` and every child reference are
  nonempty occurrence-ID strings, not array positions.
- `kind` is `constructor`, `list`, `tuple`, or `literal`. Semantic type and
  argument role are nonempty strings. Nullary and arbitrary-arity constructor,
  list, and tuple nodes are supported; literals have no children.
- Constructor `value` is its nonempty canonical name. Do not duplicate that
  constructor tag as an extra literal occurrence. List/tuple `value` is `null`;
  literal values may be any JSON scalar. Numbers compare as exact decimal
  values without floating-point rounding, including arbitrary-width integers;
  equivalent numeric spellings such as `1` and `1.0` compare equally within
  the same semantic type. Booleans and strings do not equal numbers.
- `children` is an ordered list of distinct occurrence references. Every node
  must be reachable exactly once. Cycles, shared occurrences, duplicate IDs,
  orphan nodes, missing references, unknown fields, and malformed values fail
  with an explicit exception. NaN/infinity are not JSON values. Number exponents
  outside the native signed 64-bit normalization range are unsupported errors,
  not rounded or fabricated scores.
- Occurrence IDs, node-table order, and JSON key order do not score. Keep
  provenance/reverse source paths outside this wire and resolve returned IDs
  back to source data in the caller.

The diagnostic policy is `"typed-unit-v1"`. Kind, semantic type, and role must
match for a substitution; an unequal constructor/literal value then costs one,
while an exact match costs zero. Incompatible nodes require deletion/insertion,
each costing one per semantic occurrence. Thus removing/adding a whole subtree
charges every occurrence, not one opaque subtree charge. There is no branch
permutation, movement operation, gameplay weighting, size normalization, or
claim of semantic equivalence/relevance.

Edits transform **second toward first**. This typed policy requires immediate
parents to correspond: insertion/deletion consumes **whole subtrees**, emitting
one charged edit for every occurrence. It does not promote removed nodes'
children or wrap retained children in inserted nodes. Matched occurrences
preserve parents and ordered sibling relationships. Equal preorder labels with
different attachments are not an identity match. This API explains full paths;
it does not expose crossover, partial-edit application, or the legacy weighted
policy. These restrictions are explicit typed-policy choices; they do not alter
the architecture API's historical wrapper behavior.

### Native path and result fields

Typed ordered mode generalizes the existing recursive RCSWX add/mutate/remove
cell recurrence and persistent operation histories. Every semantic occurrence
has a scored opening and a zero-cost owned closure; boundary bookkeeping is
not a semantic edit. The legacy grammar-specific behavior is unchanged.
Native Rust `Tree::from_json` parses the wire shape; comparison validates and
prepares each semantic tree once before scoring. The recurrence reuses two row
buffers. Closure legality is carried by the active-owner state, so zero-cost
closing transitions share the preceding semantic history rather than allocating
discarded boundary-operation records.
Because closing legality depends on earlier opening correspondences, generic
mode retains the minimum history **per active matched-owner stack**, rather than
discarding all histories above one cell-wide minimum. Only paired owners close
together, and opening pairs require their immediate parents to be the current
pair (or both occurrences to be roots). At a fixed cell, token positions and the
active stack determine future opening/closing legality, so a lower-cost history
in the same state dominates. Independent tiny mapping enumeration exercises
this pruning and cost behavior; demonstrated exactness is limited to those
tested structural cases, not a claim of general semantic optimality. This is
the native typed policy, not a flat token alignment or replacement scorer.

One deterministic minimum-cost path is returned, not all tied paths:

| Field | Value |
| --- | --- |
| `version` | `"rcswx-typed-ordered-result-v1"` |
| `tree_version` | `"rcswx-typed-ordered-v1"` |
| `policy` | `"typed-unit-v1"` |
| `backend` | `"rcswx-native-recursive-ordered"` |
| `direction` | `"second-to-first"` |
| `distance` | Native nonnegative integer sum of semantic edit costs |
| `correspondences` | Ordered `{"first_id": str, "second_id": str}` for matches/substitutions |
| `edits` | Ordered `{"operation": str, "first_id": str or null, "second_id": str or null, "cost": int}` |

`operation` is `match`, `substitute`, `insert`, or `delete`. Insertions consume
one first occurrence (`second_id=null`); deletions consume one second occurrence
(`first_id=null`). Every first/second occurrence appears exactly once in `edits`,
including zero-cost matches. Every charged operation is explained, and
`sum(edit["cost"] for edit in result["edits"]) == result["distance"]`.
Neither a successful serialization nor a low distance establishes binding-aware
equivalence or semantic approval; domain-specific validation remains the caller's
responsibility.

Typed mode is currently serial even in parallel-capable builds. Closure-state
retention can grow substantially on ambiguous repeated nested shapes; it does
not inherit the architecture API's quota/selection interfaces.
