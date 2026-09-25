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
- Local, ignored `benchmarks/results/wavefront-original.json`: individual public-API latency
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

The later source-topology corrections retain this immutable baseline. The
comparison still covers complete computation traces, plans, sampling, and
application results for supported selections. The old exporter also applied
raw prefixes with validation disabled, including zero-cost boundary operations
that public selection rejects. Their offspring and application traces are not
a compatibility contract: the `wrapper` case's final unsafe prefix depended on
the old insertion-anchor bug. Forced wavefront runs compare **complete** captures
against the corrected serial engine, including those unsafe prefixes, while
supported behavior remains checked against the independent frozen baseline.
Physical source coordinates now accompany histories through both serial
evaluation and coordinator-side wavefront materialization; public recursive
coordinates and computation trace ordering are unchanged.

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

The final native candidate passed the full Python suite in both builds:
643 passed / one serial-artifact-only skip in the parallel build, and
624 passed / 20 parallel-only skips in the default build. The focused parallel
suite passed 32 tests. Rust passed 40 unit tests with `trace,parallel`, plus the
resource, trace, smoke and frozen-oracle integration gates. The default Rust
build passed its 29 unit tests and applicable integrations.

Both exact release wheels passed the isolated minimal, Torch, reference and
combined installation gates, including their examples. Real wasm32 execution
passed with and without `parallel`; the binding also passed its browser-worker
test. All 30 browser tests passed across Chromium, Firefox and WebKit. Regenerated
browser recordings changed only six build-provenance fields; all algorithm
content remained identical.

## Serial ownership gate (before scheduling)

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

## Final release-wheel measurements

The final scheduler was measured without profiling on the same i7-1355U,
CPython 3.14.7, with CPU affinity `0,2,4,6` (two physical performance cores and
two efficiency cores). No build or validation ran concurrently. Each table entry
is the median of 12 warm public `edit_path` calls. Original, same-wheel serial
and four-worker calls alternated through all six orderings twice, after a
separately recorded first call.

The `nested` family is a valid, deeply nested routing/sequence architecture with
unique linear parameters and one changed middle computation. It creates
history-heavy intermediate ties; it does not replace algorithm work with sleeps
or an artificial parallel loop. Inputs and result fingerprints are outside the
timed interval; preparation and retained-plan assembly are inside it.

| Nested modules | Collapse | Token matrix | Original serial | Parallel wheel, `workers=1` | `workers=4` | Speedup vs original / same-wheel serial |
|---|---|---|---:|---:|---:|---:|
| 80 | Off | 239 × 239 | 3.516 s | 3.856 s | **2.678 s** | **1.31× / 1.44×** |
| 128 | On | 383 × 383 | 2.466 s | 2.278 s | 2.499 s | 0.99× / 0.91× |

The 80-module parallel call beat both serial controls in **all 12 paired
repetitions**. Every cold and warm result matched the independently installed
original in distance, complete ordered-path fingerprint and all canonical
statistics. Each parallel call evaluated 38,733 cells in 254 rounds, reported
four overlapping compute jobs, and reserved at most 7,773,432 bytes of scratch.
This is a measured end-to-end speedup over the untouched original, not merely
over a slowed parallel-capable serial build.

The collapse-enabled case is retained as a negative control, not excluded from
the results: only 24,316 of its 146,688 evaluated cells ran in parallel, and its
median did not improve. Pool dispatch, read-view preparation, canonical
publication, recursive orchestration and memory traffic can dominate. Width
alone does not justify parallelism. The serial default remains intentional.

Reproduce the primary comparison using separate installed release wheels:

```sh
taskset -c 0,2,4,6 target/wavefront-parallel/venv/bin/python \
  -m benchmarks.wavefront --label wavefront-final --workers 1 4 \
  --families nested --sizes 80 --collapse off --repeats 12 \
  --compare-package target/wavefront-baseline/venv/lib/python3.14/site-packages/rcswx \
  --output target/wavefront-parallel/final-80.json
```

Use available physical CPUs and the installed Python tag on another machine.
For the negative control, use `--sizes 128 --collapse on`. The comparison package
must be the independently built original revision, not the current implementation
with `workers=1`. Exact wheel and extension hashes, individual samples, cold
calls, input fingerprints, execution reports and canonical counters accompany
the local, ignored raw measurements in `benchmarks/results/wavefront-final.json`
(440 cold/warm samples across all runs).

### Serial and fallback controls

The final default wheel remains a separate, non-atomic serial build. On the
eight chain/wrapper cases (128/256 modules, both collapse modes), nine-repeat
**median paired candidate/original ratios** were 0.931–1.021. Ratios of the
separate medians were noisier, 0.954–1.129; both calculations and every sample
are retained in the data rather than treating frequency drift as a code change.

The additional nested 80-module case did expose a serial cost: the default
wheel's eight-repeat median was 3.811 s versus 3.484 s originally, **9.4% slower**
(median paired ratio 1.113). This is not a blanket zero-regression result.
The four-worker primary result still beats the untouched original, not just
this slower control.

Opting into four workers is also not free when waves fall back. The flat
chain/wrapper controls (32/256 modules, both collapse modes) were approximately
4–17% slower than the original by ratio of medians. Seven cases dispatched no
compute jobs; the remaining case parallelized only 1,155 cells. Admission
inspection and the parallel-capable ownership model still have costs. Keep
`workers=1` unless the workload benefits from parallel execution.

### Isolated memory check

Each mode below ran the nested 80-module, collapse-disabled case in a separate
process, with one first call and three warm calls. Values are the maximum
per-call RSS high-water mark across those calls, not RSS from the shared-process
latency comparison:

| Artifact / mode | Peak RSS |
|---|---:|
| Original serial | 157.36 MiB |
| Final default serial wheel | 156.92 MiB |
| Final parallel wheel, `workers=1` | 157.68 MiB |
| Final parallel wheel, `workers=4` | 154.62 MiB |

All four modes had identical inputs, complete paths, distance and canonical
statistics. The parallel run remained below the 8 MiB scratch payload ceiling;
these RSS observations do not turn that payload bound into a hard process-memory
limit or promise that parallelism always reduces RSS. The final serial modes reported
no pool access. Reproduce isolated checks by omitting `--compare-package` and
launching the Python from each wheel installation separately (`--baseline` for
the original, and `--workers 1` or `--workers 4` for the final wheel).

## Cached population comparison

The local original-versus-v0.5 CPU-time plot uses
`benchmarks/results/distance-runtime.json`. This cache and the generated plots
are local-only, ignored artifacts; they are not distributed with the repository.
Rendering does not rerun any benchmarks. Both series cover the same 1,000
frozen equal-sized pairs. The cached original runs used one worker on CPU 0;
v0.5 used a ten-worker ceiling over ten physical cores, with cold pool startup
included. This is not a serial-v0.5 population comparison.

CPU time is `process_cpu_seconds`, summed across all process threads, rather
than the calling thread's `cpu_seconds`. Limit markers use observed aggregate
CPU lower bounds. The median original/v0.5 CPU-time ratio is 13.15× over 979
matched successful nontrivial pairs; exceptions, limits and no-ops are excluded.
Both series belong to the CPU-0/ten-core campaign; the older CPU-16 original
measurements are not mixed into this comparison.

```sh
nix develop -c uv run --no-sync --with matplotlib==3.11.2 --with scipy \
  python benchmarks/distance_runtime.py \
  --input benchmarks/results/distance-runtime.json \
  --series original v0.5 --metric process_cpu_seconds \
  --output benchmarks/results/distance-runtime-original-v05-cpu
```

The renderer writes PNG, SVG and a summary with input/renderer hashes. Use
`--metric wall_seconds` for elapsed latency; omitting `--series` and `--metric`
preserves the four-version wall-time comparison.
