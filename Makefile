.PHONY: sync develop fmt check test wheel

# Enter `nix develop` first on NixOS; never install tools globally.
sync:
	uv sync --locked

develop:
	uv run --locked maturin develop

fmt:
	cargo fmt --all
	ruff format python tests
	ruff check --fix python tests

check:
	cargo fmt --all -- --check
	cargo clippy --workspace --all-targets -- -D warnings
	ruff check python tests
	ruff format --check python tests

test:
	cargo test -p rcswx-core
	uv run --locked pytest

wheel:
	uv run --locked maturin build --release --out dist
