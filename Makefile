.PHONY: sync develop fmt check test wheel wheel-test wasm-check wasm-smoke wasm

# Enter `nix develop` first on NixOS; never install tools globally.
# Development commands deliberately install both optional surfaces. The wheel
# gate creates fresh environments itself so it can prove the minimal surface.
sync:
	uv sync --locked --all-extras

develop:
	uv run --locked --all-extras maturin develop

fmt:
	cargo fmt --all
	ruff format python tests benchmarks crates/rcswx-wasm/package.py
	ruff check --fix python tests benchmarks crates/rcswx-wasm/package.py

check:
	cargo fmt --all -- --check
	cargo clippy --workspace --all-targets -- -D warnings
	ruff check python tests benchmarks crates/rcswx-wasm/package.py
	ruff format --check python tests benchmarks crates/rcswx-wasm/package.py

test:
	cargo test -p rcswx-core
	uv run --locked --all-extras pytest

wheel:
	uv run --locked --all-extras maturin build --release --out dist

# Build a wheel first, then pass that exact artifact. Never discover a stale
# artifact by wildcard: this gate must prove the wheel selected by the caller.
wheel-test:
	@: "$${RCSWX_WHEEL:?Set RCSWX_WHEEL to the exact wheel produced by 'make wheel'}"
	uv run --locked --all-extras python tests/portability/wheel.py --wheel "$$RCSWX_WHEEL" --python "$${RCSWX_PYTHON:-python3}"

# The smoke is an execution gate: wasm-bindgen-test-runner (configured in
# .cargo/config.toml) runs the compiled wasm32 test in Node, not merely cargo
# compilation. The Nix development shell supplies its pinned target and runner.
wasm-check:
	cargo check -p rcswx-core --target wasm32-unknown-unknown

wasm-smoke:
	cargo test -p rcswx-core --test wasm_smoke --target wasm32-unknown-unknown

wasm: wasm-check wasm-smoke

.PHONY: wasm-build wasm-bindings-test wasm-fixtures web-deps web-build web-check web-test web-demo

wasm-build:
	wasm-pack build crates/rcswx-wasm --target web --release --mode no-install -- --locked
	python crates/rcswx-wasm/package.py

wasm-bindings-test:
	cargo test -p rcswx-wasm --locked
	cargo test -p rcswx-wasm --test worker --target wasm32-unknown-unknown --locked

wasm-fixtures:
	cargo run -p rcswx-wasm --example export_web_fixtures --locked

web-deps:
	npm --prefix examples/web ci --ignore-scripts

web-build: wasm-build wasm-fixtures web-deps
	npm --prefix examples/web run build

web-check: wasm-build web-deps
	npm --prefix examples/web run check

web-test: web-build
	npm --prefix examples/web test

web-demo: wasm-build wasm-fixtures web-deps
	npm --prefix examples/web run dev -- --host 127.0.0.1
