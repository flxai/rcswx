# Native wavefront execution

## Baseline and independent oracle

The original serial source is revision `2d3f9969493103de07bca78c6e5e18fc83055213`.
Before changing its kernel, the implementation recorded:

- `crates/rcswx-core/tests/fixtures/wavefront-oracle-v1.json`: 32 cases, including
  both collapse modes, recursive branch ordering, wrapper/insertion cases,
  ordered ties, repeated source identities, malformed input, and truncated
  recording. It contains complete ordered histories and dependencies, selected
  indices, explicit application prefixes and origins, seeded sampling and RNG
  states, errors, and ordered trace events. Tests consume these expectations;
  ordinary builds and browser-fixture regeneration never write them.
- `benchmarks/results/wavefront-original.json`: individual public-API latency
  samples, cold calls, peak RSS, input/path fingerprints, canonical counters,
  phase profiles, CPU affinity, and the exact baseline extension/wheel hashes.
- An isolated original wheel installation under the ignored
  `target/wavefront-baseline/` directory, for later paired comparisons.

Baseline machine: Linux x86-64, Intel i7-1355U, CPython 3.14.7. Unprofiled
five-repeat warm medians from the original wheel (collapse disabled):

| Family | Logical modules | Token matrix | Public `edit_path` median |
|---|---:|---:|---:|
| Chain | 128 | 129 × 129 | 44.1 ms |
| Chain | 256 | 257 × 257 | 179.7 ms |
| Routing wrappers | 128 | 385 × 385 | 262.9 ms |
| Routing wrappers | 256 | 769 × 769 | 1,059.5 ms |

These measurements establish scale, not a speedup claim. Clock/thermal noise
is visible between independent runs; final comparisons must run baseline and
candidate controls on the same machine. The separately profiled 769 × 769
case spent 860 ms in the kernel, 2.7 ms in preparation, and 9.2 ms in
restriction processing in its first warm sample. Browser examples alone are
not a performance corpus: their widest observed subproblem is six cells.

Reproduce a measurement with the Python belonging to the wheel being tested:

```sh
python -m benchmarks.wavefront --baseline --label original \
  --families chain wrapper --sizes 64 128 256 --repeats 5 \
  --output target/wavefront-baseline/wide.json
```

`--baseline` omits the new worker keyword. Input construction and result
fingerprinting are outside the measured public call. Preparation and retained
plan assembly are inside it. The first call and warm repetitions are separate.
The RSS high-water mark is reset between samples on Linux; unavailable reset
is represented explicitly rather than misreported as a per-call peak.

The explicit `freeze_wavefront` Cargo example requires a new output path and
an explicit source revision, and refuses to overwrite an existing oracle.
It is not part of any build/test/fixture target. Do not regenerate original
expectations with a changed kernel.

## Read/write/alias audit

The serial kernel has two different identity graphs:

1. Matrix positions own references to cells. Submatrix extraction and boundary
   exchange share cells; `set` replaces one matrix reference. Deep copies memoize
   by identity so repeated references remain repeated references in the copy.
2. Each cell owns ordered history tips. Histories own their previous prefix;
   many paths/cells may share a prefix. `mark` only changes the tip's two swap
   flags. Candidate evaluation reads kinds and node identities, never these
   flags. A deep copy must copy the flags and preserve prefix sharing within the
   copied graph without aliasing it to the source graph.

`fill_cell` currently writes candidate arrays, then the minimum, then histories
in TOP, LEFT, CORNER order; minimum reduction is TOP, CORNER, LEFT. Its ordinary
traversal alternates row direction within each anti-diagonal. After each cell,
auxiliary reference replacement and corner collapse happen before diagonal
cleanup. Cleanup can affect aliases, including previously computed cells.
Recursive branch searches, copies, boundary exchange, marking and collapse
remain coordinator-only operations.

A wave may read predecessor cells on two earlier diagonals, but coordinates
alone do not establish independence. Preparation must account for identity
aliases between every destination/cleanup target and the wave's read set.
An unsafe or uneconomic wave uses the same evaluator serially. Negative-index
predecessors, prefilled boundary candidates, and skipped computed cells keep
the original semantics; preparation of a later position must not raise an
algorithm error before an earlier position is published.

## Ownership and execution decisions

- Mutable cells remain coordinator-owned. Workers receive immutable cell views,
  not matrices containing `RefCell`, borrow guards, Python values or callbacks.
- History structure is immutable and reference-counted. Swap metadata remains
  coordinator-written. Read views stay alive through join; physical reclamation
  follows the last owning reference, separately from observable logical cleanup.
  Allocation-generation identities and weak-reference validity preserve trace
  identity without recycling exposed integer IDs or rooting dead histories.
  No unsafe `Send`/`Sync` implementation or per-cell/history mutex is needed.
- Public trace identities are allocated in observation order during publication,
  not by worker completion order. Existing recording and truncation remain the
  authority; execution diagnostics are separate.
- A private bounded native pool is distinct from each call's worker ceiling.
  Ordinary cells may run in bounded chunks; recursive orchestration stays serial.
  Serial calls do not initialize the pool, including in parallel-capable builds.
- Compute workers never acquire the Python GIL. The coordinator retains the
  existing 1024-checkpoint callback cadence and independently polls elapsed
  time while waiting. It calls Python without holding pool/storage locks.
- Cancellation joins outstanding jobs before returning. Host interruption takes
  precedence over unpublished algorithm outcomes; otherwise algorithm errors
  are selected in serial publication order. No interrupted call returns a plan.
- Serial callback re-entry remains supported. Parallel callback re-entry is
  rejected. A PID guard is checked before pool synchronization after `fork`;
  a forked child may still execute serial work, or start a fresh interpreter.

Implementation and final performance results are recorded below as their
verification gates complete.
