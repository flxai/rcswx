"""Typing declarations for the implemented PyO3 extension, not a Python fallback."""

from typing import Never

PROFILE_VERSION: str
RNG_VERSION: str

type OperationRecord = tuple[int, int, bool, float, float | None, bool]
type StepRecord = tuple[int, int | None, int | None, int, int | None]

class CoreError(RuntimeError): ...

class Limits:
    def __init__(
        self,
        *,
        max_nodes: int = 2048,
        max_states: int = 250_000,
        max_predecessors: int = 500_000,
        max_core_bytes: int = 268_435_456,
        max_edits: int = 20,
        max_masks: int = 1_048_576,
        deadline_seconds: float = 30.0,
    ) -> None: ...

class Architecture:
    def __init__(self, records: list[OperationRecord], limits: Limits) -> None: ...
    @property
    def owned_bytes(self) -> int: ...

class CancellationToken:
    def __init__(self) -> None: ...
    def cancel(self) -> None: ...
    def is_cancelled(self) -> bool: ...

class EditPath:
    def __new__(cls) -> Never: ...
    @property
    def cost_ticks(self) -> int: ...
    @property
    def edit_count(self) -> int: ...
    @property
    def steps(self) -> list[StepRecord]: ...
    @property
    def stats(self) -> dict[str, int]: ...
    @property
    def owned_bytes(self) -> int: ...
    def selected_cost(self, mask: int) -> int: ...
    def project(
        self, mask: int, limits: Limits, token: CancellationToken
    ) -> list[tuple[int, int]]: ...

class Sampler:
    def __init__(
        self,
        path: EditPath,
        seed: int,
        limits: Limits,
        token: CancellationToken,
        external_bytes: int = 0,
    ) -> None: ...
    @property
    def stats(self) -> tuple[int, int, int, float]: ...
    def draw(self) -> int: ...
    def propose(
        self, limits: Limits, token: CancellationToken
    ) -> tuple[int, list[tuple[int, int]]]: ...

def derive_seed(seed: int, domain: bytes) -> bytes: ...
def distance(
    source: Architecture, target: Architecture, limits: Limits, token: CancellationToken
) -> tuple[int, dict[str, int]]: ...
def align(
    source: Architecture, target: Architecture, limits: Limits, token: CancellationToken
) -> EditPath: ...
