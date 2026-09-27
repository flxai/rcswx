# Development and verification

[README](../README.md) · [Runnable examples](../examples/README.md)

This page is for working on RCSWX itself. To use the installed library, start
with the [PyTorch](pytorch.md) or [portable](portable.md) guide instead.

## Source environment

From the repository root, enter the locked Nix shell on NixOS, then install the
development environment and native extension:

```sh
nix develop
make sync
make develop
```

`make sync`, `make develop`, and `make test` pass `--all-extras`, deliberately
installing both Torch and reference-sampling dependencies. A minimal wheel
installation remains a separately tested contract; an all-extras development
environment does not prove import isolation.

The workspace declares Rust 1.85 as its MSRV. The Nix shell pins Rust 1.85.0 with
the WASM target and supplies Python, uv, the local `wasm-bindgen-test-runner`,
Node, and browser tooling. It does not install a global target or runner.
For consuming Nix packages rather than developing the repository, see
[Nix flakes](../examples/README.md#nix-flakes).

## Local checks

Run the repository's configured commands inside that environment:

```sh
make check
make test
```

`make check` checks Cargo formatting, Clippy, Ruff lint, and Python formatting.
`make test` runs the Rust core and Python suites. `make fmt` applies the
configured formatters. Use the narrowest relevant test when iterating; runnable
Python examples and their required extras are listed in the
[example catalogue](../examples/README.md).

## Test the exact wheel

Build one wheel, then test that exact artifact in fresh minimal, Torch,
reference, and combined environments:

```sh
make wheel
```

Set and export `RCSWX_WHEEL` to the **exact path printed by the build** in `dist/`.
The test target requires this variable. Do not select a wheel by wildcard, which
could pick up a stale artifact. Then run:

```sh
make wheel-test
```

This uses `python3` by default. To test a particular available interpreter, use
`RCSWX_PYTHON=python3.12 make wheel-test`, for example.

The gate checks metadata, missing-extra errors, root import isolation, and
runnable examples. It never accepts an editable import as proof of the wheel's
behavior.

## Parallel builds

Build and check a parallel-capable development extension explicitly:

```sh
make develop-parallel
make parallel-check parallel-test
```

For a release artifact, use `make wheel-parallel`, set `RCSWX_WHEEL` to the exact
resulting path in `dist/parallel/`, and run `make wheel-parallel-test`. That gate
verifies capability and serial/parallel plan parity in every dependency surface.
The default build remains serial; build capability alone does not enable
parallelism on individual calls.

See [parallel alignment usage](portable.md#optional-parallel-alignment) for the
`workers` argument and [wavefront measurements](wavefront.md) for the execution
contract, fallback behavior, and performance evidence.

## WebAssembly and browser checks

The structural Rust core has no Python/LibTorch dependency. Its original
core-only gate remains separate from the consumer package:

```sh
make wasm-check
make wasm-smoke
make wasm-parallel-check
```

`wasm-check` compiles the core—not PyO3—for `wasm32-unknown-unknown`.
`wasm-smoke` executes decode/analyze/full-history/apply, probability, seeded
draw, and raw RNG-vector checks in Node. `wasm-parallel-check` executes the core
and frozen oracle with the feature enabled, checking that native threads do not
enter the WASM dependency graph.

The workspace also builds a reusable browser package and an interactive
worker-owned example. It runs the existing core with deterministic slider
prefixes, seeded sampling, and bounded recordings of recursive alignment:

```sh
make web-demo  # Open the displayed /rcswx/ URL.
make web-test  # Check the exact package in Chromium, Firefox, and WebKit.
```

Run these separately; the demo command starts a server. See
[WASM consumer contracts](wasm.md) and the
[standalone example](../examples/web/README.md) for build, deployment, and
consumer details. This is an additional structural interface, not a Python
model runtime or a reproduction of the paper experiments.

## Dependency notices

The portable core's target libraries include serialization
(`serde`/`serde_json`), arbitrary IDs (`num-bigint`), `libm`, and
`rand_chacha`/`rand_core`. Their installed Cargo manifests declare MIT for `libm`
and MIT OR Apache-2.0 for the other listed libraries.

The dev-only execution harness pins `wasm-bindgen-test` 0.3.77, whose
`wasm-bindgen` 0.2.127 matches the locked Nix CLI; its manifest declares
MIT OR Apache-2.0. Native parallel builds additionally link Rayon and its
MIT OR Apache-2.0 dependencies.

Wheels retain the original algorithm attribution in
[LICENSE.einsearch](../LICENSE.einsearch) and the verbatim numerical and optional
parallel dependency notices in
[LICENSE.rust-dependencies](../LICENSE.rust-dependencies).
