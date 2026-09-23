# Reproduce RCSWX Table 4.2 — local-agent handoff

**Edition 2 · 22 September 2026 · exact coauthor baseline sources recovered**

## 0. Assignment and entry point

Implement and run the four ResNet–MLP-Mixer comparisons from Table 4.2 using (a) the original Python algorithm and (b) the Rust-backed `flxai/pytorch-rcswx` implementation. Use the **exact supplied architecture definitions**, preserve their derivation trees, and report distance, runtime and memory with explicit provenance and configuration.

This document **supersedes `RCSWX_BENCHMARK_HANDOFF.md`** and the earlier instructions to search for or reconstruct ResNet34/Mixer d12. Those definitions have now been supplied by the coauthor. The earlier `baselines.zip` is not the fixture authority. The latest six Python files and their pinned bytes are.

**Already provided:** source files; expanded strings; ordered static trees; source-operation streams; checksums; five passing static recovery tests; the table image; and the coauthor’s notebook setup excerpt. **Not executed here:** runtime genotype construction, PyTorch forward passes, reference/Rust alignments, or performance measurements. Do not promote static validation into runtime validation.

Use this document with its companion bundle. Run from the bundle root:

```bash
python verify_bundle.py
(cd fixtures && python -m unittest -v test_recovery)
```

Then work in the local `pytorch-rcswx` checkout, reading `AGENTS.md` and the current `IMPLEMENTATION_PLAN.md` first. Reuse the existing test/oracle and benchmark machinery. Preserve user changes; do not overwrite the reference checkout with these fixtures, push changes, change the algorithm, or add a generic `nn.Module` importer.

### Bundle map

| Path | Purpose |
|---|---|
| `HANDOFF.md` | This execution specification, including verbatim ResNet/Mixer source appendices. |
| `fixtures/source/baselines/` | All six latest coauthor Python files, byte-for-byte unchanged; upload suffixes `(1)` removed only from filenames. |
| `fixtures/definitions.py` | Importable expanded strings, with no Torch/einsearch dependency; **not runnable neural networks**. |
| `fixtures/expanded/`, `trees/`, `operations/` | Exact strings, ordered static syntax trees, and source/factory expression streams. These are not runtime genotypes. |
| `fixtures/manifest.json` | Source, string, tree and operation-stream hashes, counts and provenance. |
| `fixtures/recover.py`, `test_recovery.py` | Restricted-AST extraction and static tests. |
| `evidence/table_4_2.png`, `create_resnet.txt` | Original table image and coauthor’s notebook setup excerpt. |
| `targets.json` | Four ordered pairings, historical numbers, source symbols and unresolved historical fields. |
| `campaign.example.json` | Proposed execution policy; not a recovered historical configuration or existing runner input schema. |
| `REPORT_TEMPLATE.md` | Required four-row tables and audit sections. |
| `verify_bundle.py`, `CHECKSUMS.json`, `STATIC_VALIDATION.txt` | Local integrity check and preparation-time validation evidence. |

ViT, WideResNet and ConvNeXt sources are preserved for provenance but **excluded from this task’s primary benchmark**. Do not extend the workload to them before completing the four requested pairs.

## 1. The four fixed target rows

Source: `evidence/table_4_2.png`, supplied by the user. Keep the parent order shown below; test the reverse order only as a separately labeled diagnostic.

| Pair ID | Parent 1 symbol | Parent 2 symbol | Printed nodes | Printed compute time | Printed distance |
|---|---|---|---:|---:|---:|
| `resnet18-mixer8` | `resnet18_no_maxpool` | `mlpmixer_d8` | 267 / 264 | 39.054 seconds | 60.38 |
| `resnet18-mixer12` | `resnet18_no_maxpool` | `mlpmixer_d12` | 267 / 392 | 342.383 seconds | 95.38 |
| `resnet34-mixer8` | `resnet34_no_maxpool` | `mlpmixer_d8` | 499 / 264 | 86.633 seconds | 119.62 |
| `resnet34-mixer12` | `resnet34_no_maxpool` | `mlpmixer_d12` | 499 / 392 | **741.57 minutes** | 115.62 |

The last time literally converts to **44,494.2 seconds**, or **12.3595 hours**. Retain the printed unit and conversion separately. Whether “minutes” was a typo is **unverified**; never replace it with seconds silently or launch a twelve-hour run automatically.

The screenshot does not establish the historical hardware, input shape, source revision, pruning flag, timing boundary, precision before rounding, or node-count implementation. The recovered model definitions resolve architecture identity, not these experimental details.

## 2. Exact architecture specification

### 2.1 Authoritative symbols, counts and source ranges

Paths below are relative to `fixtures/`. The coauthor explicitly identified the new `resnet34_no_maxpool` and `mlpmixer_d12` definitions as those used, and previously identified `resnet18_no_maxpool`. Their supplied Mixer file also contains `mlpmixer_d8`. Preserve the coauthor’s attribution; do not invent a historical commit for these attachments. [U1–U3]

| Model | Source assignment | Static operation occurrences | Runtime `serialise()` target |
|---|---|---:|---:|
| ResNet18 | `source/baselines/resnet.py:74–99`, `resnet18_no_maxpool` | 267 | 267, to verify |
| ResNet34 | `source/baselines/resnet.py:127–176`, `resnet34_no_maxpool` | 499 | 499, to verify |
| Mixer d8 | `source/baselines/mlpmixer.py:58–86`, `mlpmixer_d8` | 264 | 264, to verify |
| Mixer d12 | `source/baselines/mlpmixer.py:88–128`, `mlpmixer_d12` | 392 | 392, to verify |

The static counts were regenerated from the supplied templates using the supplied loader’s operation tokenization. They match the screenshot. They are **not** PyTorch module counts, parameter counts, alignment-token counts or measured runtime tree sizes. Confirm the runtime counts and counting convention independently; matching counts alone is not a correctness gate.

Pin these full source hashes (the bundle verifier also checks every artifact):

```text
resnet.py
  017e244fbd0a82e8bd25340f2c245a46e51f0747b10cf72ff0e67f4445741111
mlpmixer.py
  b7176703826fcbb17d68ad76420180d746d680ff588bb06f177d00452c260332
__init__.py
  2be67ff3ed968ccca2eab523fa2ac09951b2af8f848403b5893032cd224cbfdf
```

### 2.2 ResNet18 and ResNet34 — the supplied einspace variants

Use the **entire new ResNet source**, including shared helpers, not just the new ResNet34 assignment grafted onto the earlier file. [U1]

The supplied stem is a 3×3, stride-1, padding-1 image/column routing operation with `linear64`, followed by `norm`, then `relu`, then another 3×3 routing operation with stride 2, padding 1 and `linear64`. It does **not** use the separate 7×7-stem symbol or a max-pooling layer.

The ordinary residual helper has two 3×3 stride-1 routed linear operations, normalization and ReLU in the exact nesting shown in Appendix A. Its shortcut is **`computation[identity]`**. This wrapper is absent in the earlier uploaded ResNet helper; omitting it changes the supplied definition. The strided helper has a stride-2 first 3×3 operation and a stride-2 1×1 projection plus normalization on its shortcut.

| Model | Block counts by width 64 / 128 / 256 / 512 | Stage transitions |
|---|---|---|
| `resnet18_no_maxpool` | 2 / 2 / 2 / 2 | The first block at 128, 256 and 512 uses `resnet_strided_block`. |
| `resnet34_no_maxpool` | 3 / 4 / 6 / 3 | The first block at 128, 256 and 512 uses `resnet_strided_block`. |

These counts describe the source; they are **not permission to rebuild a differently grouped sequential tree**. For a compact exact grouping reference, let `S(a,b)` denote one binary `sequential`, `H` the supplied stem, `R(w)` the supplied ordinary block with both linear widths `w`, and `D(w)` its strided counterpart. Each occurrence is a separate subtree, not a shared module or repetition macro:

```text
ResNet18:
S(
  S(H, S(S(R(64),R(64)), S(D(128),R(128)))),
  S(S(D(256),R(256)), S(D(512),R(512)))
)

ResNet34:
S(
  S(H,
    S(S(S(R(64),R(64)),R(64)),
      S(S(D(128),R(128)),S(R(128),R(128))))),
  S(S(S(D(256),R(256)),S(S(R(256),R(256)),S(R(256),R(256)))),
    S(S(D(512),R(512)),R(512)))
)
```

The verbatim definitions in Appendix A and the hashed expanded strings are authoritative if any summary is misread. No task head is present in these definition strings.

### 2.3 MLP-Mixer d8 and d12 — preserve the supplied block

Both use `routing[im2col4k4s0p, computation[linear512], identity]` followed by `computation[pos_enc]` as the prefix. Do not substitute a timm Mixer, change patch size, remove positional encoding, introduce GELU, or add standard Mixer residuals. [U2]

Preserve the source’s helper names and operations exactly:

```text
channel_mixer:
  routing[permute21, computation[linear_x(4)], identity]
  → computation[relu]
  → routing[identity, computation[linear_x(0.25)], permute21]

token_mixer:
  computation[linear256] → computation[relu] → computation[linear512]

mlpmixer_layer:
  branching(2)[
    clone(2),
    sequential[computation[norm], sequential[channel_mixer, token_mixer]],
    computation[identity],
    add(2)
  ]
```

The arrows above summarize execution order, not replacement syntax. Use Appendix B for the binary grouping within each helper. Keep **one residual wrapper around the combined mixing sequence**, as supplied. `linear_x(4)` and `linear_x(0.25)` are relative rules, not interchangeable with an absolute width that happens to materialize identically on one input.

Let `M` be one explicit `mlpmixer_layer`, `P` the supplied prefix, and `S` a binary `sequential`. The actual repetition grouping is:

```text
E2  = S(M,M)
E4  = S(E2,E2)
E8  = S(E4,E4)
E12 = S(E8,E4)
mlpmixer_d8  = S(P,E8)
mlpmixer_d12 = S(P,E12)
```

These equations stand for **textual copies**, not a `sequential(8)`/`sequential(12)` production, shared object, or flattened layer list.

### 2.4 Loader and representation contract

The newly supplied `source/baselines/__init__.py` **does accept `limits=`**. It strips whitespace, tokenizes operation expressions, maps aliases, instantiates operations and invokes an iterative sampler. Its initializer does not re-export ResNet34/Mixer d12. Use explicit submodule imports in a compatible full einsearch environment: [U3]

```python
from baselines import build_baseline
from baselines.resnet import resnet18_no_maxpool, resnet34_no_maxpool
from baselines.mlpmixer import mlpmixer_d8, mlpmixer_d12
```

Some supplied DSL child lists have missing or trailing commas. The supplied loader discards these delimiters. Preserve original bytes; do not “fix” punctuation and then claim to have used identical source. Static tree traversal and loader token order have been checked to agree.

Keep three identities distinct: **source expression** (for example `linear_x(4)`), **mapped factory expression**, and actual **runtime `operation.name`**. Resolve factories in the selected grammar; never treat the text of a factory expression as its runtime name.

## 3. Construction route and runtime fixture gates

### 3.1 Follow the coauthor’s two-stage replay

The reported route is `build_baseline(...)` followed by replay of the resulting operation stream through the experiment sampler. The resulting **root genotype is authoritative**; `root.build(root)` produces its PyTorch backbone. The notebook also shows a separate compilation path adding a task wrapper and memory checkpointing. Do not include that wrapper in the aligned genotype without historical evidence. [U3–U4]

Reference-side integration sketch, **requiring an explicitly selected compatible loader, sampler, limiter and input configuration**:

```python
from copy import deepcopy


def build_reported_root(definition, expected_nodes, input_params,
                        sampler, limiter, build_baseline):
    baseline_root = build_baseline(
        definition, deepcopy(input_params), limits=deepcopy(limiter.limits)
    )
    before = list(baseline_root.serialise())
    expected_names = [node.operation.name for node in before]
    replay = [node.operation for node in before]  # fresh list: sampler consumes it
    sampler.limiter.timer.start()
    root = sampler.sample(deepcopy(input_params), replay)
    after = list(root.serialise())
    if replay:
        raise RuntimeError("Unconsumed baseline operations")
    if len(before) != expected_nodes or len(after) != expected_nodes:
        raise RuntimeError("Runtime derivation-node count differs from source target")
    if [node.operation.name for node in after] != expected_names:
        raise RuntimeError("Operation replay changed the fixture")
    return root
```

This sketch is not the full gate. In setup-only diagnostics, **fail on any random PCFG fallback during baseline construction or replay**, and compare parent/child structure and relevant metadata. The inspected sampler calls `pcfg.sample` when the supplied list becomes empty; checking only that it is empty at return cannot detect an incomplete stream that was filled randomly. A temporary diagnostic guard must be restored before measurement. [R8]

### 3.2 Build both backends independently from the same source

For the paired benchmark, create fresh reference-owned roots using the hash-checked original classes and fresh native-owned roots using `rcswx`’s genotype, grammar and reconstruction classes. Reuse `tests/reference/original.py` and related serializers. Do not call the original algorithm as a native fallback or derive reference expected results from native outputs. [R1–R4]

`tests/reference/corpus.py:build_case` is useful scaffolding, **not a drop-in historical fixture builder**: the inspected helper chooses `deep_broadcast_grammar` and appends relative factors 0.5 and 2. The supplied notebook chooses `grammars[args.search_space]`, while its baseline loader initially uses `grammar`. Mixer requires the exact 4 and 0.25 factories. Audit registry selection and availability instead of silently adopting the test helper’s grammar or substituting supported widths. [R4, U3–U4]

Prefer a focused `tests/reference/table42.py` factory. Resolve the supplied alias/factory stream separately in each backend’s grammar namespace, preserving source tree grouping, child order, operation rules, IDs and metadata. Where a grammar/factory is unavailable, retain the exact failure and diagnose the revision mismatch. A benchmark-only registration/admission change, if proposed, must be documented as a separate configuration; never silently change the historical grammar or the pinned oracle.

Keep the attached source files isolated under a benchmark fixture directory. For full-package notebook replay, use a dedicated import package/worktree or equivalent isolated loader with recorded `__file__` paths. For the hash-checked oracle, reproduce the supplied loader’s operation mapping against its original grammar namespace without importing the unrelated training stack. Do not replace the original loader’s source hash checks to make an overlay appear unmodified.

### 3.3 Required runtime checks per model

Record source hashes, descriptor hashes, parser/grammar hashes, input/output metadata, and the full runtime operation/child structure. Verify **267 / 499 / 264 / 392** with `len(list(root.serialise()))` in both construction paths, unless the recovered historical counting expression proves different. Keep static count, runtime count, alignment tokens including sentinels, matrix dimensions, PyTorch module count and parameter/buffer bytes in separate fields.

Compare the resolved source stream, baseline root, replayed root, and native root structurally. Check every operation’s relevant rule/child/callback metadata, not only names and counts. Preserve `computation[identity]`, relative widths and routing endpoints. No padding, automatic head, sequential rebalancing or replacement network is allowed.

Build and forward-check each model separately on matched deterministic inputs in an isolated validation worker. Compare shapes, finite outputs and parameter/buffer metadata; use matched initialization/RNG states or an explicit weight mapping for output-value comparisons. Do not compare independently random weights and call the difference a model mismatch. Record any numerical tolerance and reason. Keep construction, forward and optional backward work outside alignment measurements; release tensor models before alignment workers start.

### 3.4 Historical configuration versus diagnostic configuration

The supplied notebook loads `configs/einspace/collimation0.yaml`, selects `cuda`, has a default seed of 42, constructs a limiter from configuration/data and creates:

```python
input_params = {
    "shape": torch.Size([args.batch_size, args.channels, *args.image_size]),
    "other_shape": None,
    "mode": "im",
    "other_mode": None,
    "branching_factor": 1,
    "last_im_shape": None,
}
```

This is a debugging-notebook recipe, **not proof of the table’s numerical configuration**. The YAML, its effective arguments, the matching surrounding source revision and the table measurement cells are not supplied. Do not treat a similarly named file at another commit as recovered historical evidence. [U4]

To proceed without waiting, use the proposed **`diagnostic_cpu32_v1`** profile: batch shape `(2,3,32,32)`, CPU float32, seed 42, one numerical/Torch thread, and the metadata keys above. These are deliberate benchmark choices, **not historical facts**. Resolve and record explicit construction/alignment limiter values from the selected environment before launch, including any required `memory_crossover` key. Do not rely on the supplied loader’s unlimited default limits.

A failed construction under this profile is a recorded result, not permission to change widths, identities, normalization or grouping. Another shape/configuration is a separately named experiment. Synthetic inputs suffice for diagnostic execution; dataset loading is not a prerequisite for alignment.

## 4. Version strata: do not conflate “original” with “historical”

The native repository was rechecked at:

```text
pytorch-rcswx: 9444dcfd861b9f8ee5ea96eac72e3c5b74268438
pinned Python reference: 3f44ddf086bee0c213404e240ee0adf99e3e1501
reference algorithm SHA-256:
bcb09ab6f115c78d14d2949eb9d3fa33dda0b2f239bcbbeadf992f93b86ad11e
```

The plan records an approved recursive branch-boundary repair in both implementations. The pinned Python reference is **not established as the historical Table 4.2 revision**. Verify every source hash in `tests/reference/original.py`, not just the one above. Record actual local commits, dirty diffs and the release extension’s hash; reconcile any later checkout with the local plan without resetting the user’s work. [R1–R3]

| Stratum | Purpose | What must match |
|---|---|---|
| `paired_current_reference` — required | Compare repaired Python reference with current Rust-backed implementation. | Exact fixtures, configuration, pruning mode, boundary, numerical environment and observable outputs. |
| `historical_table` — conditional | Reproduce the printed table using the coauthor’s historical algorithm/configuration, when recovered. | Historical source and measurement provenance; keep independent hashes and results. |
| `historical_candidate` — optional, explicit | Test an identified but unconfirmed older revision. | Clear candidate label; agreement with printed values does not prove provenance. |

Do not undo the repair, relax reference hashes, tune operation costs, or pick whichever revision/flag happens to match the screenshot. A mismatch to the printed distance can coexist with correct reference/native equivalence. Conversely, matching a printed two-decimal value does not prove equivalent implementations.

## 5. Corner collapsing: mandatory explicit experiment axis

The fixed boolean option is already implemented in original `AlignmentMatrixRecursive` and the Rust kernel, and exposed by `rcswx.api.edit_path`. Its default is `False`; `rcswx.distance(...)` does not expose it. **No optimization implementation is required for this task.** [R2, R5–R7]

Run all four pairs under **both `collapse_corners=False` and `True`**, with each flag applied identically to Python and Rust. This defines 4 pairs × 2 flags × 2 backends = **16 primary configurations per repetition**. The historical table flag remains unknown; the coauthor recalled this optimization, but no supplied table measurement cell proves which mode was used.

Equivalent alignment entry points on separately prepared parent roots and guards:

```python
from tests.reference.original import load
from rcswx.api import edit_path

reference = load()  # verifies pinned original source hashes
original_alignment = reference.algorithm.AlignmentMatrixRecursive(
    original_parent1, original_parent2,
    collapse_corners=collapse, limiter=original_guard,
)
native_alignment = edit_path(
    native_parent1, native_parent2,
    collapse_corners=collapse, limiter=native_guard,
)
# Result scalars: original_alignment.distance and native_alignment.distance.
# Each call belongs in its own worker; this is an API sketch, not a timing loop.
```

At global coordinates `I = i + start_i`, `J = j + start_j`, and full token-list lengths `n1`, `n2`, the source condition is:

```python
collapse_corners and (J - I >= n2 * 0.25 or I - J >= n1 * 0.25)
```

The source first computes a qualifying cell, then `clean(completely=False)` retains the first stored path and first entries of its directional arrays; the native `clean(false)` mirrors that rule. Preserve full-list lengths, recursion offsets, inclusive comparisons, timing of cleanup, path order and auxiliary swap-matrix behavior. This is **history pruning**, not omission of those cells or a newly implemented Itakura mask. The `0.25` is an offset fraction, not a guaranteed 25% or 75% saving in cells/time/memory. Do not add a configurable percentage or reinterpret the geometry in this benchmark. [R5–R7]

Keep three checks distinct:

1. **Python(off) versus Rust(off):** equivalence gate for the uncollapsed mode.
2. **Python(on) versus Rust(on):** equivalence gate for the collapsed mode.
3. **Off versus on within each backend:** approximation/sensitivity result, not an equality requirement. Record changes in distance, ordered operations, retained histories, failures, time and memory. Later transition legality depends on histories, so equal distance is not assumed.

Include the flag in worker commands, result/group/cache keys, semantic comparisons and report filenames. Never allow off/on rows to be pooled or a runner default to override the requested mode. Use fresh parents because alignment renumbers parent IDs. Existing reference tests cover both flags; rerun them against the built binary rather than inferring they pass. [R2, R7, R9]

## 6. Correctness and failure handling

Before claiming performance on a pair/configuration, establish runtime fixture integrity and compare full-precision distance, every exposed ordered history, selected operations, dependencies, parent state/ID effects and relevant RNG states. Reuse existing `tree_record`, `operation_record` and alignment outcome serializers. Hashing a lossy summary is not full equivalence.

A full ordered-history comparison may be expensive. Write timing/memory results before fingerprinting, compare streamed deterministic records where possible, and preserve the stage if interrupted. An unverified/incomplete history check may still support a **scalar-only observation**, but not an unrestricted alignment/edit-plan speedup claim. Do not claim crossover fidelity from this alignment-only task; selection, offspring construction and retry benchmarks are optional separate work. [R1, R10]

Run self-pairs and reversed pairs as bounded diagnostics only after primary coverage. Do not assume symmetry, monotonicity in model size or a triangle inequality. Test the helper’s zero shortcut separately from a full equal-parent alignment.

Keep full precision for backend comparisons. For the screenshot audit, report `format(distance, '.2f')`, the unrounded value and difference from the printed value, with the formatting rule stated as an assumption unless recovered. The printed numbers are not exact scalar oracles. Do not quantize to quarter units or widen backend tolerance merely to match two decimals.

Retain mismatches and all failures with their inputs/configuration and a reproducible invocation. Distinguish construction failure, algorithm-raised exception, supervisor timeout, RSS stop, unexpected worker exit and budget exclusion. A matching exception is fidelity evidence **for that failure**, not a successful distance result. Do not fix either implementation or drop the case to manufacture a speedup.

## 7. Measurement protocol

### 7.1 Primary boundary and timing

The required workload is **complete alignment-object construction**, including its normally returned histories/operations/dependencies, through the constructor/API return. Measure externally with `time.perf_counter()`. Exclude imports, fixture preparation, tensor-model construction, forward passes and post-call fingerprinting. Hold the returned result alive at the endpoint in both backends.

Record three separate durations: `align_wall_seconds`, `internal_compute_time_seconds`, and `worker_wall_seconds`. The Python and native `compute_time` fields cover different implementation internals; they are diagnostics, **not automatically like-for-like timings**. The historical “compute time” scope is unknown until its measurement code is recovered. Never claim timing reproduction by guessing that scope. [R5–R7, R10–R11]

One target call per fresh subprocess, no target warmup, one worker at a time, fresh parents and matched seeds/configuration. Alternate backend order between repetitions. Set and record Torch, OpenMP/BLAS thread controls before measurement; record CPU, RAM, affinity, OS/kernel, Python/dependency versions, source/lockfile hashes and binary path/hash. Build a **release** extension for the actual reference interpreter. The repository’s default `make develop` does not request release mode. [R10–R13]

Imports and setup are excluded from the call timer but remain within supervisor budgets. Primary timing must not include allocation profiling, Python tracing or stage hooks. Stage profiling and warmed runs, if desired later, are separate campaigns and must not be pooled with primary results.

### 7.2 Memory

Extend the existing `benchmarks/memory.py` RSS path. Use its unprofiled fresh-process measurement, with Linux high-water reset after imports/genotype preparation and before the call. Report baseline RSS, window peak RSS, peak-minus-baseline, retained RSS and after-release RSS separately. If the reset fails, mark the window peak unavailable and label any lifetime/sampled peak separately. Do not subtract two lifetime maxima and call that algorithm peak memory. [R10]

Keep inputs and returned outputs alive at the same boundary. An alignment returning all histories must not be compared with a scalar-only result under the same label. Do not retain built tensor models for only one backend. Report model parameter/buffer memory separately.

Native-inclusive Memray heap traces are optional, separate bounded diagnostic processes, never primary timings. `tracemalloc` is not a total-memory comparison for Python versus Rust. Label allocator-traced heap, RSS and any GPU memory distinctly. The existing `build` phase includes both parents’ construction, forward and backward; it is **not alignment memory**. [R10]

### 7.3 Budgets and scheduling

These are proposed operating limits, not recovered historical settings:

| Control | Proposed policy |
|---|---|
| Pilot | One attempt per primary configuration; 120 seconds per whole worker. |
| Extended worker | At most 1,800 seconds, only after pilot coverage. |
| Primary target | Three successful unprofiled repetitions per eligible configuration; keep actual attempt counts. |
| Campaign total | 7,200 measurement-worker seconds of wall budget, shared across invocations and modes. |
| RSS guard | Explicit host/job-aware limit with headroom; no invented workstation RAM size. |
| Optional diagnostics | Only with remaining budget; no full automatic 12-hour historical run. |

Preallocate result records for all 16 primary configurations. Schedule pilots across **every pair and flag before repeating easy cases or extending slow ones**. A timeout does not erase a case; one deliberate bounded retry at a larger tier may be scheduled fairly. Do not blindly rerun timed-out requests three times. Reserve/report preparation and test effort separately from the measurement budget.

Use an existing job/container hard memory limit where available, plus the harness guard. A sampled RSS guard can overshoot. Separate algorithmic limiter settings from supervisor time/RSS caps; changing either is an explicit named configuration. Record exit code, interrupted stage, limits and any already-flushed measurement. A signal alone is not proof of OOM; a fingerprinting timeout is not proof of an alignment-call timeout. No ratio from incomplete or failed results.

## 8. Implementation sequence and command contract

### Step A — verify assets, source, environment and binary

Run the bundle checks. In the local repository, read the controlling plan and inspect status. Use the established uv/Cargo/maturin/Nix workflow, not a new package manager. The existing bootstrap pins the reference numerical runtime; retain it for both backends. Add `--benchmarks` to bootstrap only when Memray is needed. [R1, R3, R12–R13]

```bash
# From the local pytorch-rcswx checkout; enter nix develop first on NixOS.
export RCSWX_REFERENCE_ROOT="$HOME/phd/ext/einsearch"
git status --short
git rev-parse HEAD
git -C "$RCSWX_REFERENCE_ROOT" rev-parse HEAD
mkdir -p benchmark-results/table42/wheels

uv run --locked python -m tests.reference.bootstrap .venv-reference \
  --python python3.12 --log benchmark-results/table42/bootstrap.log

# Same configured build backend, explicitly targeting the measured interpreter.
uv run --locked maturin build --release \
  --interpreter .venv-reference/bin/python \
  --out benchmark-results/table42/wheels

# Set RCSWX_WHEEL to the exact wheel just produced, not a stale-file wildcard.
: "${RCSWX_WHEEL:?Set the explicit newly built wheel path}"
uv pip install --python .venv-reference/bin/python --no-deps \
  --force-reinstall "$RCSWX_WHEEL"

.venv-reference/bin/python -c \
  'import hashlib, pathlib, rcswx; from rcswx import _core; p=pathlib.Path(_core.__file__); print(rcswx.__file__); print(p, hashlib.sha256(p.read_bytes()).hexdigest())'
.venv-reference/bin/python -m pytest tests/test_reference_alignment.py
.venv-reference/bin/python -m benchmarks.memory --suite smoke --list
```

Verify local CLI/help and interpreter/wheel compatibility if the checkout has advanced. Preserve bootstrap and source hashes in the output. Do not upgrade dependencies to make a historical candidate import without declaring a distinct environment.

### Step B — integrate the four fixtures and runtime audit

Implement the focused factory/manifest route from §3. Preserve source hashes and full source/factory/runtime mappings. Provide one per-model validation command that emits a runtime manifest without running alignments; document its exact invocation. Audit both construction stages and all four models, including shape/forward validation. The source-definition stage is already complete; do not spend the task searching for missing ResNet34/Mixer d12 code.

### Step C — extend existing runner, not a duplicate measurement engine

The inspected `benchmarks/memory.py` accepts `smoke`, `scaling`, `all` suites and existing `align`, `raw`, `validated`, `build` phases. It does **not** yet expose a `table42` suite or a corner-collapse CLI switch. `edit_path`/the kernel already expose the boolean. The local agent must connect that option through the runner. [R2, R10]

Required additions, keeping current suites/default behavior intact:

- `table42` suite with precisely the four ordered pairs and their asset/runtime manifests.
- Explicit `--collapse-corners off|on|both`, propagated to each worker and grouping key; no float-percentage feature.
- `--runtime-config` accepting a validated **resolved runtime configuration**, not the example policy JSON supplied here. Implement/document the schema and pin its hash; historical fields remain separate.
- Persistent cumulative budget and progress accounting, fair pilot scheduling, atomic partial results, and explicit unrun rows.
- Summary generation that gates ratios on equivalent successful completed configurations and emits one four-row table per pruning mode/stratum.

**Proposed invocation after implementing these additions; this is not a command that works on the inspected checkout yet:**

```bash
: "${RCSWX_BENCH_RSS_MIB:?Set a recorded host-aware RSS guard}"
.venv-reference/bin/python -m benchmarks.memory \
  --suite table42 --phase align --measure rss \
  --collapse-corners both \
  --runtime-config benchmark-results/table42/runtime-resolved.json \
  --repeats 1 --timeout 120 \
  --wall-budget-seconds 7200 \
  --budget-file benchmark-results/table42/campaign-budget.json \
  --rss-limit-mib "$RCSWX_BENCH_RSS_MIB" \
  --output benchmark-results/table42/pilot.json
```

Use the same campaign budget ledger for later three-repetition, 1,800-second-tier invocations. A restart must not reset the budget silently. Build historical-source support only once its identity is established, with an independent source manifest/loader and separate output; do not weaken the current oracle.

### Step D — tests, measured runs and report

Add focused tests for asset/source mapping, all four target counts, exact source grouping, complete replay with zero random fallback, native/reference ownership, flag propagation and separation, unsupported grammar failures, budget resumption, partial-write handling and speedup eligibility. Do not place expensive whole-model alignments in ordinary CI. A tiny bounded fixture should test runner mechanics.

Run configured formatting/linting and focused tests, then pilots and budget-eligible main runs. Inspect the task-owned diff, follow repository commit policy, and leave unrelated changes alone. Deliver commands, logs, manifests and findings even if some target computations cannot finish within the agreed budget.

## 9. Results and acceptance

Use `REPORT_TEMPLATE.md`. Preserve the historical four rows in their original order and units, then emit **two separate four-row primary result tables**, off and on, for the paired-current-reference stratum. Historical/candidate runs get separate tables. Do not present one mixed table choosing the faster mode for each backend.

Every worker record must identify: pair and orientation; exact model source/string/runtime hashes; source stratum and actual reference/native commits; binary/environment hashes; runtime profile; construction/call seeds and RNG fingerprints; collapse flag and fixed threshold; call boundary; algorithmic/supervisor limits; repetition; status/stage; full-precision result or exception; semantic-verification scope/status; time and memory fields. Use `null` for unavailable measurements, never zero. Encode nonfinite distances explicitly rather than silently emitting invalid JSON numbers.

Compute ratios only for **successful, semantically matched, same-boundary/same-mode/same-profile runs on the same host/environment**:

```text
runtime speedup = median Python align_wall_seconds / median Rust align_wall_seconds
memory ratio = matched Python memory metric / matched Rust memory metric
```

Use matched completed repetition sets, report exclusions and counts, and state the memory metric/aggregation. Do not compute ratios for zero or noise-dominated memory deltas, worker failures or censored timeouts. One-run results are pilots. Pruning off/on speed changes must be accompanied by outcome differences; they are not unconditional equivalent-algorithm speedups.

Never divide the screenshot’s historical times by new Rust times to claim a controlled implementation speedup. Hardware and scope are unverified. Do not infer complexity laws from four structurally different pairs.

### Definition of done

- [ ] Latest coauthor bytes are pinned; all four exact definitions and source count checks are used.
- [ ] Every model has a runtime construction/forward audit or a preserved concrete failure.
- [ ] Original/native genotypes retain exact operation rules and tree structure; no library-model substitution.
- [ ] All four pairs × both flags have explicit results, failures, or budget-unrun records for both backends.
- [ ] Correctness scope is recorded before performance claims; failed/incomplete cases remain visible.
- [ ] Historical versus repaired-reference strata, limits, timings and memory boundaries are separated.
- [ ] Literal `741.57 minutes`, unknown historical flag and unknown historical measurement setup remain explicit.
- [ ] Raw evidence, runnable commands, hashes, tests and separate four-row reports are delivered.

A completed bounded campaign is not automatically a successful historical reproduction. Conclude separately on **source recovery**, **runtime fixture fidelity**, **Python/Rust equivalence**, **printed-distance agreement** and **performance**. State exactly which historical evidence is still missing.

## 10. Evidence and source references

**Uploaded authority** (the companion bundle preserves the original file bytes):

- **U1:** `resnet(1).py` → `fixtures/source/baselines/resnet.py`; helpers and all ResNet assignments. Source reproduced verbatim in Appendix A.
- **U2:** `mlpmixer(1).py` → `fixtures/source/baselines/mlpmixer.py`; all Mixer helpers/assignments. Source reproduced verbatim in Appendix B.
- **U3:** `__init__(1).py` → `fixtures/source/baselines/__init__.py`; `build_baseline`, aliases and exports.
- **U4:** `create_resnet.txt` → `evidence/create_resnet.txt`; notebook configuration, limiter/sampler/input setup, two-stage ResNet18 recipe.
- **U5:** User’s Table 4.2 image → `evidence/table_4_2.png`; printed targets only.
- **U6:** Coauthor messages quoted in the conversation: `resnet18_no_maxpool` selection; later recovery of `mlpmixer_d12` and `resnet34_no_maxpool` as the definitions used; recollection of corner collapse. These messages do not supply a table-generating invocation or historical commit.

**Repository source** (private links require existing repository access; resolve all against the pinned commit, not floating `main`):

- **R1:** [Controlling implementation plan](https://github.com/flxai/pytorch-rcswx/blob/9444dcfd861b9f8ee5ea96eac72e3c5b74268438/IMPLEMENTATION_PLAN.md).
- **R2:** [Public edit_path / distance API](https://github.com/flxai/pytorch-rcswx/blob/9444dcfd861b9f8ee5ea96eac72e3c5b74268438/python/rcswx/api.py).
- **R3:** [Pinned source loader and hashes](https://github.com/flxai/pytorch-rcswx/blob/9444dcfd861b9f8ee5ea96eac72e3c5b74268438/tests/reference/original.py).
- **R4:** [Existing fixture construction helper](https://github.com/flxai/pytorch-rcswx/blob/9444dcfd861b9f8ee5ea96eac72e3c5b74268438/tests/reference/corpus.py).
- **R5:** [Original recursive algorithm, MatrixCell.clean and corner condition](https://github.com/flxai/einsearch/blob/3f44ddf086bee0c213404e240ee0adf99e3e1501/search_strategies/utils/recursive_constrained_smith_waterman.py).
- **R6:** [Rust recursive core and corner condition](https://github.com/flxai/pytorch-rcswx/blob/9444dcfd861b9f8ee5ea96eac72e3c5b74268438/crates/rcswx-core/src/recursive.rs).
- **R7:** [Native Python alignment wrapper / timing boundary](https://github.com/flxai/pytorch-rcswx/blob/9444dcfd861b9f8ee5ea96eac72e3c5b74268438/python/rcswx/recursive.py).
- **R8:** [Original iterative sampler / fallback behavior](https://github.com/flxai/einsearch/blob/3f44ddf086bee0c213404e240ee0adf99e3e1501/search_strategies/random_search.py).
- **R9:** [Reference alignment tests, including both collapse values](https://github.com/flxai/pytorch-rcswx/blob/9444dcfd861b9f8ee5ea96eac72e3c5b74268438/tests/test_reference_alignment.py).
- **R10:** [Fresh-process RSS/heap benchmark and CLI](https://github.com/flxai/pytorch-rcswx/blob/9444dcfd861b9f8ee5ea96eac72e3c5b74268438/benchmarks/memory.py).
- **R11:** [Timing/semantic benchmark and diagnostic stages](https://github.com/flxai/pytorch-rcswx/blob/9444dcfd861b9f8ee5ea96eac72e3c5b74268438/benchmarks/reference.py).
- **R12:** [Reference numerical environment bootstrap](https://github.com/flxai/pytorch-rcswx/blob/9444dcfd861b9f8ee5ea96eac72e3c5b74268438/tests/reference/bootstrap.py).
- **R13:** [Build commands](https://github.com/flxai/pytorch-rcswx/blob/9444dcfd861b9f8ee5ea96eac72e3c5b74268438/Makefile).
- **R14:** [Python/toolchain/dependency constraints](https://github.com/flxai/pytorch-rcswx/blob/9444dcfd861b9f8ee5ea96eac72e3c5b74268438/pyproject.toml).

The instructions and budget choices in this handoff are a new execution plan. They are not attributed to the coauthor or claimed as historical experiment settings.

## Appendix A. Verbatim supplied ResNet source

For standalone handoff use, the complete new `resnet(1).py` follows. The unused `resnet18_conv7x7_no_maxpool` definition is retained because this is a verbatim source copy; it is not a fifth table fixture.

```python
# The ResNet18 architecture, represented in einspace
# the MaxPool operation in the stem is replaced by a convolution


resnet_stem_no_maxpool = """
    sequential[
        sequential[
            sequential[
                routing[im2col3k1s1p, computation[linear64], col2im],
                computation[norm]
            ],
            computation[relu]
        ],
        routing[im2col3k2s1p, computation[linear64], col2im]
    ]"""
resnet_conv7x7_stem_no_maxpool = """
    sequential[
        sequential[
            sequential[
                routing[im2col7k2s3p, computation[linear64], col2im],
                computation[norm]
            ],
            computation[relu]
        ],
        routing[im2col3k2s1p, computation[linear64], col2im]
    ]"""
resnet_block = lambda a, b: f"""
    sequential[
        branching(2)[
            clone(2),
            sequential[
                sequential[
                    sequential[
                        routing[im2col3k1s1p, computation[{a}], col2im],
                        computation[norm]
                    ],
                    computation[relu]
                ],
                sequential[
                    routing[im2col3k1s1p, computation[{b}], col2im],
                    computation[norm]
                ]
            ],
            computation[identity],
            add(2)
        ],
        computation[relu]
    ]"""
resnet_strided_block = lambda a, b: f"""
    sequential[
        branching(2)[
            clone(2),
            sequential[
                sequential[
                    sequential[
                        routing[im2col3k2s1p, computation[{a}], col2im],
                        computation[norm]
                    ],
                    computation[relu]
                ],
                sequential[
                    routing[im2col3k1s1p, computation[{b}], col2im],
                    computation[norm]
                ]
            ],
            sequential[
                routing[im2col1k2s0p, computation[{b}], col2im],
                computation[norm]
            ]
            add(2)
        ],
        computation[relu]
    ]"""
resnet18_no_maxpool = f"""
    sequential[
        sequential[
            {resnet_stem_no_maxpool},
            sequential[
                sequential[
                    {resnet_block('linear64', 'linear64')},
                    {resnet_block('linear64', 'linear64')}
                ],
                sequential[
                    {resnet_strided_block('linear128', 'linear128')},
                    {resnet_block('linear128', 'linear128')}
                ]
            ]
        ],
        sequential[
            sequential[
                {resnet_strided_block('linear256', 'linear256')},
                {resnet_block('linear256', 'linear256')}
            ],
            sequential[
                {resnet_strided_block('linear512', 'linear512')},
                {resnet_block('linear512', 'linear512')}
            ]
        ]
    ]"""
resnet18_conv7x7_no_maxpool = f"""
    sequential[
        sequential[
            {resnet_conv7x7_stem_no_maxpool},
            sequential[
                sequential[
                    {resnet_block('linear64', 'linear64')},
                    {resnet_block('linear64', 'linear64')}
                ],
                sequential[
                    {resnet_strided_block('linear128', 'linear128')},
                    {resnet_block('linear128', 'linear128')}
                ]
            ]
        ],
        sequential[
            sequential[
                {resnet_strided_block('linear256', 'linear256')},
                {resnet_block('linear256', 'linear256')}
            ],
            sequential[
                {resnet_strided_block('linear512', 'linear512')},
                {resnet_block('linear512', 'linear512')}
            ]
        ]
    ]"""

resnet34_no_maxpool = f"""
    sequential[
        sequential[
            {resnet_stem_no_maxpool},
            sequential[
                sequential[
                    sequential[
                        {resnet_block('linear64', 'linear64')},
                        {resnet_block('linear64', 'linear64')}
                    ],
                    {resnet_block('linear64', 'linear64')}
                ],
                sequential[
                    sequential[
                        {resnet_strided_block('linear128', 'linear128')},
                        {resnet_block('linear128', 'linear128')}
                    ],
                    sequential[
                        {resnet_block('linear128', 'linear128')},
                        {resnet_block('linear128', 'linear128')}
                    ]
                ]
            ]
        ],
        sequential[
            sequential[
                sequential[
                    {resnet_strided_block('linear256', 'linear256')},
                    {resnet_block('linear256', 'linear256')}
                ],
                sequential[
                    sequential[
                        {resnet_block('linear256', 'linear256')},
                        {resnet_block('linear256', 'linear256')}
                    ],
                    sequential[
                        {resnet_block('linear256', 'linear256')},
                        {resnet_block('linear256', 'linear256')}
                    ]
                ],
            ],
            sequential[
                sequential[
                    {resnet_strided_block('linear512', 'linear512')},
                    {resnet_block('linear512', 'linear512')}
                ],
                {resnet_block('linear512', 'linear512')}
            ]
        ]
    ]"""
```

## Appendix B. Verbatim supplied MLP-Mixer source

The d2/d4 variants are retained because this is a verbatim copy; only d8/d12 are primary fixtures.

```python
channel_mixer = """
    sequential[
        sequential[
            routing[permute21, computation[linear_x(4)], identity],
            computation[relu]
        ],
        routing[identity, computation[linear_x(0.25)], permute21]
    ]"""
token_mixer = """
    sequential[
        sequential[
            computation[linear256],
            computation[relu]
        ],
        computation[linear512]
    ]"""
mlpmixer_layer = f"""
    branching(2)[
        clone(2),
        sequential[
            computation[norm],
            sequential[
                {channel_mixer},
                {token_mixer},
            ],
        ],
        computation[identity],
        add(2)
    ]"""
mlpmixer_d2 = f"""
    sequential[
        sequential[
            routing[im2col4k4s0p, computation[linear512], identity],
            computation[pos_enc]
        ],
        sequential[
            {mlpmixer_layer},
            {mlpmixer_layer}
        ]
    ]"""
mlpmixer_d4 = f"""
    sequential[
        sequential[
            routing[im2col4k4s0p, computation[linear512], identity],
            computation[pos_enc]
        ],
        sequential[
            sequential[
                {mlpmixer_layer},
                {mlpmixer_layer}
            ],
            sequential[
                {mlpmixer_layer},
                {mlpmixer_layer}
            ]
        ]
    ]"""
mlpmixer_d8 = f"""
    sequential[
        sequential[
            routing[im2col4k4s0p, computation[linear512], identity],
            computation[pos_enc]
        ],
        sequential[
            sequential[
                sequential[
                    {mlpmixer_layer},
                    {mlpmixer_layer}
                ],
                sequential[
                    {mlpmixer_layer},
                    {mlpmixer_layer}
                ]
            ],
            sequential[
                sequential[
                    {mlpmixer_layer},
                    {mlpmixer_layer}
                ],
                sequential[
                    {mlpmixer_layer},
                    {mlpmixer_layer}
                ]
            ]
        ]
    ]"""

mlpmixer_d12 = f"""
    sequential[
        sequential[
            routing[im2col4k4s0p, computation[linear512], identity],
            computation[pos_enc]
        ],
        sequential[
            sequential[
                sequential[
                    sequential[
                        {mlpmixer_layer},
                        {mlpmixer_layer}
                    ],
                    sequential[
                        {mlpmixer_layer},
                        {mlpmixer_layer}
                    ]
                ],
                sequential[
                    sequential[
                        {mlpmixer_layer},
                        {mlpmixer_layer}
                    ],
                    sequential[
                        {mlpmixer_layer},
                        {mlpmixer_layer}
                    ]
                ]
            ],
            sequential[
                sequential[
                    {mlpmixer_layer},
                    {mlpmixer_layer}
                ],
                sequential[
                    {mlpmixer_layer},
                    {mlpmixer_layer}
                ]
            ]        
        ]
    ]"""
```
