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

## Native scheduler and safety gates

The optional `parallel` Cargo feature is forwarded by the Python binding, not
enabled by default. Rayon is a native-target-only dependency. The private pool
uses the process's available parallelism when first initialized; `workers=1`
does not inspect or initialize it. A call's resolved ceiling is
`min(requested, capacity)`, or the full capacity for `workers=-1`.

Ordinary diagonals are visited in the original alternating order. A read-only
work estimate rejects low-density frontiers before allocating lane buffers.
Admitted rounds contain at most 1,024 positions. Their reserved payload is
bounded by 8 MiB and the remaining allocation quota; oversized rounds are
split, and an inadmissible remainder runs serially. The payload estimate includes
candidate arrays, result containers, compact predecessor proposals and the
eventual histories. It is not a process-RSS limit: allocator overhead, thread
stacks and bounded executor bookkeeping are separate.

The coordinator allocates the lane buffers. Compute jobs only fill candidates
and ordered `(direction, predecessor index)` proposals through disjoint mutable
slices. They do not create canonical histories, mutate matrices, assign public
trace IDs, call Python, or hold matrix/storage locks. The coordinator resolves
proposals and creates histories in canonical order after join. Resource failure
and algorithm error precedence therefore do not depend on completion order.

Chunks are sized by estimated candidate/history work rather than cell count.
At most eight chunks per allowed compute job enter a call-local queue; at most
the resolved worker ceiling drains that queue concurrently. A queue mutex
protects only chunk transfer and is released before evaluation. Faster workers
can take additional chunks without increasing the job ceiling or requiring
per-cell locks. This matters on heterogeneous CPUs and highly uneven histories.

`plan.execution` is separate from algorithm statistics and trace recording:
`pool_capacity=None` means this serial call did not touch the pool, even if
another call initialized it. `jobs` counts submitted compute jobs and
`peak_jobs` counts their maximum overlapping active lifetimes.
`scratch_peak_bytes` is the conservative reserved payload, not sampled RSS.
Fallback counters count failed admission attempts; scratch failures can lead to
a smaller successful round rather than a wholly serial diagonal.

The forced-scheduling Rust gate runs all 32 original frozen cases with worker
ceilings 2/4 and round shapes 2/3/17/1,024. It also compares limited-work/output
errors and their partial traces against serial execution. Additional regressions
cover destination and cleanup aliases, scratch exhaustion, concentrated-work
overlap, concurrent pool ceilings, callback cancellation and worker panic
recovery. Python exercises complete plans, selection/application, seeded
sampling, callback exception identity, re-entry, concurrent cancellation,
SIGINT, fork rejection and serial calls that create no pool threads.

```sh
make develop-parallel
make check parallel-check parallel-test
make wasm-parallel-check
make wheel-parallel
RCSWX_WHEEL=dist/parallel/rcswx-0.1.0-cp314-cp314-linux_x86_64.whl \
  make wheel-parallel-test
```

Use the exact filename produced by the build, not a wildcard or the example
Python tag above. The serial wheel has its own `make wheel` / `make wheel-test`
gate. Both gates use fresh minimal, Torch, reference and combined installations.

## Serial ownership gate

The common evaluator and deterministic publisher pass all 32 frozen cases on
native Rust and on real wasm32 execution in Node, including exact trace events
and truncation. The existing Python suite passed 611 tests, covering the pinned
reference behavior, portable API and Torch surface. Core resource/trace tests
also pass. A lifetime regression exercises iterative release of a 50,000-step
history while another owner retains a shared prefix.

Default/portable builds retain non-atomic reference counting. The immutable
history representation admits atomic ownership for native parallel-capable
builds without changing the evaluator, identity graph, or trace representation.

`benchmarks/results/wavefront-serial-refactor.json` records the serial performance
gate. Back-to-back whole-program runs showed substantial frequency/thermal
drift on this hybrid CPU, so the driver now supports `--compare-package`: it
loads the independently installed original and candidate packages/extensions,
rotates their public-call order each repetition, and checks every complete
ordered-path fingerprint. No native implementation is regenerated or substituted.

Across eight pinned-CPU chain/wrapper cases (128/256 modules, both collapse
modes), nine-repeat paired median ratios were **0.992–1.047** versus the
original. All path fingerprints and all algorithm statistics, including
accounted allocation bytes, matched. Standalone runs retain independent RSS
measurements; paired runs intentionally share an interpreter/allocator and
are latency controls, not isolated memory measurements.

The `cold` field is the first call for that case/mode, not a claim that each
row starts a fresh process or pool. Exact extension/wheel hashes identify the
executed binaries; recorded Git revisions are the checkout HEAD at measurement
time, before the corresponding semantic commit.
