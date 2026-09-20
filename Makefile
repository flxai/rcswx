.PHONY: sync develop fmt check test wheel

# Enter `nix develop` first on NixOS; never install tools globally.
sync:
	uv sync --locked

develop:
	uv run --locked maturin develop

fmt:
	cargo fmt --all
	ruff format python tests benchmarks
	ruff check --fix python tests benchmarks

check:
	cargo fmt --all -- --check
	cargo clippy --workspace --all-targets -- -D warnings
	ruff check python tests benchmarks
	ruff format --check python tests benchmarks

test:
	cargo test -p rcswx-core
	uv run --locked pytest

wheel:
	uv run --locked maturin build --release --out dist
