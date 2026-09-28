"""Explicit grammar selection for graph import; no automatic plugin loading."""

from dataclasses import dataclass
from typing import Any

from .types import UnsupportedModuleError


@dataclass(frozen=True)
class StateBinding:
    occurrence: int
    parameters: dict[str, str]
    buffers: dict[str, str]
    reshape_weight: bool = False


@dataclass(frozen=True)
class Lowering:
    architecture: Any
    bindings: dict[str, dict]
    state_bindings: tuple[StateBinding, ...]


class EinspaceBackend:
    name = "einspace"
    version = "1"

    def lower(self, graph, input_spec) -> Lowering:
        from .einspace_import import lower

        return lower(graph, input_spec)


_BACKENDS = {("einspace", "1"): EinspaceBackend()}


def resolve_backend(grammar: str, grammar_version: str):
    if not isinstance(grammar, str) or not isinstance(grammar_version, str):
        raise UnsupportedModuleError("grammar: name and version must be strings")
    backend = _BACKENDS.get((grammar, grammar_version))
    if backend is None:
        raise UnsupportedModuleError(
            f"grammar: no graph-import backend for {grammar!r} version {grammar_version!r}"
        )
    return backend
