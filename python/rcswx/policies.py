"""Finite, immutable request policies and domain-separated random seeds."""

import hashlib
import math
import secrets
from dataclasses import asdict, dataclass

PROFILE_VERSION = "torch-grammar-v1"
SCHEMA_VERSION = "rcswx-ir-v1"
GRAMMAR_VERSION = "ordered-unary-v1"
REGISTRY_VERSION = "torch-cpu-f32-v1"
SAMPLING_VERSION = "gaussian-mask-v1"
REPORT_VERSION = "rcswx-report-v1"
NORMALIZATION_VERSION = "sequence-normalization-v1"
WITNESS_VERSION = "ordered-witness-v1"
RNG_VERSION = "rcswx-rng-v1"


@dataclass(frozen=True, slots=True)
class Limits:
    max_nodes: int = 2048
    max_depth: int = 64
    max_rank: int = 8
    max_states: int = 250_000
    max_predecessors: int = 500_000
    max_core_bytes: int = 256 * 1024 * 1024
    max_edits: int = 20
    max_masks: int = 1_048_576
    max_attempts: int = 128
    max_child_bytes: int = 256 * 1024 * 1024
    deadline_seconds: float = 30.0

    def __post_init__(self) -> None:
        caps = (
            ("max_nodes", 1, 1_000_000),
            ("max_depth", 1, 1024),
            ("max_rank", 1, 64),
            ("max_states", 1, 10_000_000),
            ("max_predecessors", 1, 20_000_000),
            ("max_core_bytes", 1, 4 * 1024**3),
            ("max_edits", 0, 24),
            ("max_masks", 1, 1 << 24),
            ("max_attempts", 1, 1_000_000),
            ("max_child_bytes", 1, 4 * 1024**3),
        )
        for name, minimum, maximum in caps:
            value = getattr(self, name)
            if type(value) is not int or not minimum <= value <= maximum:
                raise ValueError(f"{name} must be an integer in [{minimum}, {maximum}]")
        deadline = self.deadline_seconds
        if (
            isinstance(deadline, bool)
            or not isinstance(deadline, (int, float))
            or not math.isfinite(deadline)
            or not 0 < deadline <= 3600
        ):
            raise ValueError("deadline_seconds must be finite and in (0, 3600]")

    def as_dict(self) -> dict[str, int | float]:
        return asdict(self)

    def core_kwargs(self) -> dict[str, int | float]:
        return {
            name: getattr(self, name)
            for name in (
                "max_nodes",
                "max_states",
                "max_predecessors",
                "max_core_bytes",
                "max_edits",
                "max_masks",
                "deadline_seconds",
            )
        }


def resolve_seed(seed: int | None) -> int:
    if seed is None:
        return secrets.randbits(64)
    if type(seed) is not int or not 0 <= seed < 1 << 64:
        raise ValueError("seed must be an unsigned 64-bit integer or None")
    return seed


def seed_streams(seed: int) -> tuple[bytes, int]:
    seed = resolve_seed(seed)
    prefix = b"rcswx-rng-v1\0" + seed.to_bytes(8, "little")
    selection = hashlib.sha256(prefix + b"selection").digest()
    initialization = hashlib.sha256(prefix + b"initialization").digest()
    return selection, int.from_bytes(initialization[:8], "little")
