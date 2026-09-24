# WebAssembly consumers

`crates/rcswx-wasm` exposes the existing Rust preparation, alignment, restrictions,
application and native sampler. It does not execute tensors, validate model
shapes, train models, or reproduce the paper's experiments. The standalone
[example](../examples/web/README.md) is framework-free TypeScript and HTML/SVG.
No npm publication is implied.

## Build and run

Use the tracked environment; do not install a separate Rust toolchain or global
WASM/browser commands:

```sh
nix develop
make wasm-build          # crates/rcswx-wasm/pkg: ES module, WASM, types, licenses
make wasm-bindings-test  # native session tests and real dedicated-worker tests
make wasm-fixtures       # regenerate exact-input recordings using native Rust
make web-demo            # local example at /rcswx/
# In another shell:
make web-test            # exact release artifact in Chromium, Firefox, WebKit
make web-check           # formatter and TypeScript checks
```

`make web-build` produces static assets under `examples/web/dist`. `wasm-pack`
uses `--target web --release --mode no-install -- --locked`; optimization by an
implicitly downloaded `wasm-opt` is disabled. Nix supplies Rust 1.85.0, the WASM
target, matching wasm-bindgen CLI, wasm-pack, Node/npm, browser binaries and the
Chrome driver. The local npm lockfile pins the reference consumer's development
dependencies. Existing Python commands and `make wasm-check` / `make wasm-smoke`
remain separate; the latter still executes the core smoke in Node.

Build identity defaults to the Git revision. Release automation or source-archive
builds can supply `RCSWX_BUILD_ID`; without Git or an override the value is the
explicit label `source-archive`, not a content fingerprint.

## Package boundary

Initialize the generated module once inside a dedicated module worker:

```ts
import init, { WasmSession } from './pkg/rcswx_wasm.js';
import type { Analysis, Applied } from './pkg/browser';

await init({ module_or_path: new URL('./pkg/rcswx_wasm_bg.wasm', import.meta.url) });
const session = new WasmSession(crypto.randomUUID());
const plan: Analysis = JSON.parse(session.analyze(parent1Json, parent2Json,
  JSON.stringify({ api: 1, trace_level: 'full', collapse_corners: false })));
const child: Applied = JSON.parse(session.preview_step(plan.plan_id, 0));
// child.architecture_json is lossless JSON TEXT, not an ordinary JS object.
session.dispose(plan.plan_id);
session.free(); // after disposing all plans and ending use of this facade
```

`parent1Json` and `parent2Json` are architecture-schema-1 JSON texts supplied by
the consumer. Never parse and reserialize their opaque numeric metadata through
JavaScript. Parsing the surrounding transport DTO is safe: architecture output
remains an `architecture_json` string. Node IDs are canonical decimal strings;
occurrence/path indices are checked nonnegative wasm32 integers, not logical IDs.

All exported methods are synchronous and return JSON text; expected failures
throw `{code, stage, message, retryable, requested_step?}` objects. The example's
consumer-owned worker turns them into routed asynchronous responses. Public JSON
DTO declarations ship as `browser.d.ts` beside the generated method declarations.

| Method | Result |
|---|---|
| `version()` | Build identity, API/architecture/trace schemas, sampler and policies. |
| `analyze(parent1Json, parent2Json, optionsJson)` | Retained plan ID, selected path, parent/token display projections and trace summary. |
| `inspect(planId)` | The same bounded plan summary; no reanalysis. |
| `preview_step(planId, k)` | Exact prefix result from a fresh baseline clone. |
| `apply_selection(planId, indicesJson)` | Validated explicit raw-path indices and result. |
| `sample(planId, seedHex, skewness)` | Native sampled mask, indices, cost and applied child, or an explicit failure. |
| `trace_page(planId, offset, limit)` | Immutable analysis events and next cursor; no computation rerun. |
| `dispose(planId)` | Release the plan, recording and cached previews; repeated disposal is harmless. |

An ID is valid only within its live session. Reusing an old session identifier
across worker restarts defeats that ownership distinction: create a fresh one.
Disposal releases owned objects; it does not promise that WASM's memory high-water
mark or browser process RSS will shrink. Worker termination resets the session.

## Slider and endpoint meaning

Let `E = analysis.nontrivial`. Step `k` requests exactly `E.slice(0, k)` in the
core's application order, using values in `E` as **raw selected-path indices**.
The raw path, application order and selectable edits are distinct. Zero-cost and
bookkeeping operations remain visible but do not add stops. Costs are not a
similarity percentage. The selected history is currently index zero; changing
that field alone would not construct another consistent plan.

Empty selection starts from prepared **Parent 2**, not Parent 1. The target is
Parent 1's ordered operation-name tree. Endpoint comparisons account for ID
renumbering and output arena compaction; they are not raw input JSON equality.
Opaque parameters/provenance are inherited from actual source occurrences, and
`input_spec` is inherited from Parent 2. They are not alignment dimensions and
are not promised to converge to Parent 1.

Not every valid architecture pair admits every prefix. For example, crossing a
`computation(relu)` first parent with the article's `branching(2)` second parent
fails dependency validation at step 1 of 2. The binding returns
`invalid_selection` with `requested_step: 1`; it does not return a previous child,
repair the selection or reorder edits. Branch-swapped zero-distance pairs also
need not share ordered endpoints. The example admits only the five explicitly
listed, fully checked pairings—not their Cartesian product.

Each application clones the retained immutable plan. Interleaved previews,
explicit selections, failures and samples cannot advance its ID allocator or
reverse its dependency groups. Never apply another original-plan selection to a
previously displayed child.

## Traces and origins

The core `trace` Cargo feature is default-off. WASM enables it, while each call
chooses `none`, `summary` or `full`. None constructs no explanatory payloads;
summary is not a cell-by-cell animation. Full records actual subproblems, branch
permutations, shared cells, candidate costs, history revisions/copies, collapse
and cleanup, retained histories and the selected plan.

Records use deterministic observation-order identities, not pointer addresses.
Definitions precede their references. Cell values distinguish numbers,
`uncomputed`, `nan`, `positive_infinity` and `negative_infinity`. Full streams
retain actual history multiplicity and tie order. Target-sized allocation-byte
counters remain native `EditPlan` diagnostics; the portable recording includes
work/output counters, not platform-dependent `sizeof` accounting.

The analysis recording never changes after analysis. Every application result
has a separate recording of requested indices, actual execution and internal
materialization actions, plus a child snapshot and origins. Cache hits return
the same recording. Trace truncation sets `complete: false` and a reason, then
ordinary computation continues under its independent quota. A core quota error,
allocation failure or trap is not a successful truncated analysis.

Display keys are `(snapshot_key, occurrence_index)`. Tokens identify their owning
occurrence, including boundary tokens. Child origins are exported from actual
materialization handles and their compact output mapping. Synthesized nodes can
have multiple sources. Do not match nodes by repeated IDs, labels or unrelated
array positions. A recording header hashes the exact input JSON bytes with
SHA-256; persistent examples also retain those texts, options and build identity.

## Sampling and seeds

`seedHex` is exactly 64 lowercase hexadecimal characters, encoding 32 bytes in
byte-array order. Decimal UI seeds are strings converted with `BigInt` into a
little-endian 256-bit integer (`42` starts with `2a`, followed by 31 zero bytes).
Every request creates a fresh native ChaCha12 RNG; repeating it reproduces the
same result for the pinned engine. Sampling does not consume slider state.
An empty edit set yields an empty selection, not an operation at index zero.

Native mask multiplicity, legality and cost weighting are unchanged. An
application failure is returned, not rejection-sampled away. This is not NumPy's
stream, and seed-to-child identity is not promised across future engine versions.

## Finite browser policies

Options may lower core/trace limits, not exceed the published policy. Limits
restrict operational work, not the algorithm's mathematical domain.

| Layer | Default / upper policy |
|---|---|
| Each architecture input | 256 KiB JSON text, 256 occurrences, depth 64. |
| Core call | 200,000 work units; 20,000 output units; 64 MiB accounted allocations. |
| Recording | 50,000 events and 8 MiB serialized event bytes. |
| Session | Four plans; 32 MiB conservative retained/snapshot accounting. |
| Preview cache | 4 MiB conservative accounting; bounded eviction. |
| Response | 2 MiB JSON text. |
| Trace page | 1–256 events, at most 512 KiB serialized events. |
| Reference client | One active request; 16 queued requests, configurable up to 64. |
| Reference timeouts | 20 seconds initialization, 10 seconds subsequent requests; configurable positive integer milliseconds up to 2,147,483,647 (the browser timer limit). |
| Reference rendering | At most 256 tree nodes and 20 × 20 matrix cells per view. |

Core allocation accounting is not a hard process-memory limit. Session accounting
reserves four times serialized payload sizes for owned Rust data and includes a
working snapshot; serialization, browser copies and SVG have separate bounds.
Curated examples are much smaller; see [measurements](#verification-and-measurements).

Stable error codes include `invalid_input`, `unsupported_api`, `invalid_index`,
`invalid_seed`, `invalid_step`, `invalid_selection`, `wrong_plan`, `expired_plan`,
`input_limit`, `session_limit`, `response_limit`, `core_quota`,
`allocation_failure`, `reference_failure`, `application_failure` and
`sampling_failure`. The worker/client adds `stale_result`, `superseded`,
`queue_limit`, `cancelled`, `timeout`, `initialization_failure`, `worker_crash`
and protocol failures. Preserve their stage and requested step; do not flatten
all errors into “no crossover found.”

## Worker lifecycle and integration

The reference [`BrowserClient`](../examples/web/src/client.ts) is shared by all
figures on a page. Each request carries protocol, session, request, figure and
revision fields. Increasing a figure revision rejects obsolete queued work and
disposes its plans. Stale successful analyses are disposed too. Preview requests
coalesce while queued, with explicit rejection of superseded promises.

`releaseFigure` leaves other figures' worker session intact. `cancelAll`, a
timeout or a crash terminates the entire worker, rejects every waiter and
invalidates every plan. A queued cancel message cannot interrupt synchronous
WASM. `restart` creates a new session; rebuild from stored inputs. Call `close`
on page teardown. Cached rendering/trace playback bypass the computation queue.

The example owns no framework and imposes no website repository layout. Reuse
the DTOs and worker protocol while retaining site ownership of typography,
captions, timing, orchestration and components. Text/keyboard views accompany
SVG, focus and hover link occurrences, and reduced-motion mode disables automatic
playback. Labels are inserted as text, never `innerHTML` or executable metadata.

## Static deployment and failures

Serve `examples/web/dist` under `/rcswx/`, or change Vite's `base` before building.
Worker and WASM URLs are resolved as deployed assets. Serve `.wasm` as
`application/wasm`; use HTTPS or local development serving. Do not open the HTML
through `file://`.

The tested preview CSP is:

```text
default-src 'self'; script-src 'self' 'wasm-unsafe-eval'; worker-src 'self';
style-src 'self'; img-src 'self' data:; connect-src 'self'; object-src 'none'; base-uri 'none'
```

Apply the relevant policy to the worker script response as well as the document.
No general JavaScript `unsafe-eval`, computation backend, SharedArrayBuffer,
threading or cross-origin-isolation headers are required. Missing/blocked WASM
leaves the article and exact-pair cached slider readable; live sampling stays
disabled. Exports are explicit local downloads. There is no telemetry or upload.

The generated package and demo include original attribution and dependency
license texts collected from the locked Cargo dependency graph. Keep those
notices with redistributed assets. The build-only integration harness page is
for verification; a website needs only its chosen article entry and assets.

## Verification and measurements

Implementation report, 2026-09-24. Browser measurements used the release package
reporting engine revision `4a2be908d6e341777b7e68f1440738d5daffefcd`, served from
`/rcswx/` on localhost, with full tracing and no WASM optimizer pass. The device
was Linux/NixOS x86-64, Intel Core i7-1355U. Playwright 1.63.0 used the pinned
headless engines: Chromium 153.0.8010.12, Firefox 155.0 and WebKit 26.6.

### Executed gates

All commands ran in `nix develop`:

| Command | Observed result |
|---|---|
| `cargo test -p rcswx-core --features trace --locked` | 36 tests passed, including four trace fidelity/neutrality/origin regressions. The four trace tests were also rerun after the final core correction. |
| `make wasm-bindings-test` | Five native session tests and one dedicated-worker wasm-bindgen test passed. |
| `make web-test` | 30 tests passed: ten in each of Chromium, Firefox and WebKit; no engine skipped. |
| `make test wasm` | 28 core unit tests, three resource tests, native and Node WASM smokes, and all 611 Python tests passed. |
| `make check` | Cargo format, workspace/all-target Clippy, Ruff lint and Ruff format checks passed. |
| `make web-check` | The release package rebuilt; Prettier and TypeScript checks passed. |
| `nix fmt flake.nix` | Alejandra accepted the tracked Nix configuration. |
| `make wheel` | Built `dist/rcswx-0.1.0-cp314-cp314-linux_x86_64.whl`. |
| `RCSWX_WHEEL=dist/rcswx-0.1.0-cp314-cp314-linux_x86_64.whl RCSWX_PYTHON=python3.14 make wheel-test` | That exact wheel passed fresh minimal, Torch, reference and combined environments, including import isolation and runnable examples. |
| `npm pack --dry-run --json` in `crates/rcswx-wasm/pkg` | Eight distributable files: module, WASM, two declaration files, package metadata and three attribution/notice files. Nothing published. |

The browser gate compares complete portable native/WASM event streams and every
prefix of every admitted pair, plus three fixed samples per pair. It covers
stale-analysis disposal, coalescing/overflow, multi-figure cancellation,
initialization timeout (including signed-timer overflow rejection), actual worker failure, restart, old handles, lossless
large numbers/repeated IDs, unsafe indices, quota failures, MIME, non-root URLs,
worker CSP rejection, and missing-WASM cached playback.

A separate temporary Playwright probe exercised keyboard range input, linked
parent/child focus, local download and reduced-motion behavior in all three
engines. Chromium desktop (1280 × 900) and mobile-width (390 × 844) screenshots
were inspected; the mobile document did not overflow horizontally, while the
matrix retains its own horizontal scroll. The interactive browser bridge timed
out, so this visual check used the pinned Playwright runtime directly. Probe
source was removed after execution.

### Artifact sizes

| Built asset | Bytes |
|---|---:|
| WASM module | 825,983 |
| Worker JavaScript | 9,203 |
| Shared renderer/client/cached-record chunk | 234,909 |
| Article JavaScript | 7,255 |
| CSS | 3,905 |

Vite reported 276.90 kB gzip for WASM and 25.48 kB gzip for the shared
renderer/record chunk. The distributable's npm dry-run tarball was 298,683 bytes;
its unpacked files totaled 1,328,050 bytes, including 455,160 bytes of third-party
notices. Generated artifacts are ignored, not committed. Small native fixture
records are the documented exception.

### Local browser timings

These are one sequential localhost observation per pair, not statistical
benchmarks or guarantees. Timings include worker messages and response
serialization; analysis constructs its full trace but does not transfer all
trace pages. Application columns show the range over all first-time prefix
applications for that pair.

Cold worker initialization through the first `version` response measured
57.6 ms in Chromium, 448 ms in Firefox and 60 ms in WebKit.

| Pair | Prefixes | Events / trace bytes | Chromium analysis / application ms | Firefox analysis / application ms | WebKit analysis / application ms |
|---|---:|---:|---:|---:|---:|
| Chain | 3 | 54 / 9,223 | 32.2 / 2.0–10.2 | 6 / 3–4 | 13 / 5–7 |
| Wrapper | 2 | 46 / 8,442 | 4.7 / 1.0–2.8 | 4 / 2–3 | 6 / 2–3 |
| Recursive | 2 | 1,029 / 140,469 | 75.4 / 1.8–2.0 | 84 / 2–4 | 68 / 2 |
| Insertion | 2 | 35 / 6,425 | 2.9 / 1.3–3.9 | 4 / 1–3 | 4 / 2 |
| Identical | 1 | 25 / 4,679 | 2.0 / 1.3 | 4 / 1 | 3 / 2 |

For warm cached playback, 30 input events per pair measured synchronous DOM
updates separately from arrival at the next animation frame:

| Browser | Chain median update / next-frame ms | Recursive median update / next-frame ms |
|---|---:|---:|
| Chromium | 1.2 / 16.7 | 1.2 / 16.65 |
| Firefox | 3 / 16.5 | 3 / 16 |
| WebKit | 3 / 80.5 | 3 / 83.5 |

The largest synchronous update was 8 ms. The slower headless WebKit frame cadence
is reported rather than treated as 60-fps playback; these figures do not measure
physical display presentation.

A separate temporary native release probe on the recursive pair reported these
medians across five observations: ordinary analysis without the trace Cargo
feature, 0.198565 ms; ordinary analysis with the feature compiled in, 0.196686 ms;
the explicit traced API with level `none`, 0.246575 ms; full recording,
25.533442 ms. The ordinary-API difference is within this small probe's noise.
Full explanation is materially more expensive than ordinary computation; it is
optional and independently bounded, not presented as free instrumentation.

### Coverage limits

The three Linux headless engines were exercised, not physical mobile devices,
macOS Safari or Windows browsers. The wheel check here used CPython 3.14 on
Linux x86-64, not a new multi-platform wheel matrix. Keyboard/reduced-motion
checks and screenshots are not a formal screen-reader accessibility audit.
No tensor execution, shape validation, paper benchmarks or production-network
latency claims are implied.
