"""Public value types for the optional PyTorch module adapter.

The adapter deliberately keeps framework state out of its portable architecture
representation. These records only contain data that can be copied into a
managed state-dictionary checkpoint.
"""

from collections.abc import Mapping
from dataclasses import dataclass, field
from typing import Any


class TorchAdapterError(RuntimeError):
    """Base error raised by the RCSWX PyTorch adapter."""


class StaleProvenanceError(TorchAdapterError):
    """Raised when a retained architecture no longer matches its module."""


class UnsupportedModuleError(TorchAdapterError):
    """Raised when a module has neither valid provenance nor a converter."""


class UnsafeManifestError(TorchAdapterError):
    """Raised when a purported portable provenance manifest is malformed."""


@dataclass(frozen=True, slots=True)
class BuildOptions:
    """Explicit execution choices for :func:`rcswx.torch.build`.

    ``device`` and ``dtype`` are applied only after the grammar has built a
    fresh module. They therefore never become structural edit attributes.
    ``training`` is likewise an execution state, not a structural edit attribute.
    """

    device: Any | None = None
    dtype: Any | None = None
    training: bool | None = None


@dataclass(frozen=True, slots=True)
class CapturedArchitecture:
    """A portable architecture plus verified local module provenance."""

    architecture: Any
    input_spec: Mapping[str, Any]
    metadata: Mapping[str, Any]
    bindings: Mapping[str, Any]
    structure: Mapping[str, Any]

    def to_manifest(self) -> dict[str, Any]:
        """Return the data-only manifest used by managed checkpoints."""
        return {
            "format": "rcswx.torch.manifest",
            "version": 1,
            "architecture": self.architecture.to_dict(),
            "input_spec": dict(self.input_spec),
            "metadata": dict(self.metadata),
            "bindings": dict(self.bindings),
            "structure": dict(self.structure),
        }


@dataclass(frozen=True, slots=True)
class ConverterResult:
    """Data returned by a registered exact-type module converter.

    ``bindings`` records the converter's declared source-module paths for the
    architecture occurrences it imported. It is intentionally data-only: a
    saved manifest never imports converter code to rebuild a module.
    """

    architecture: Any
    input_spec: Mapping[str, Any] | None = None
    conventions: Mapping[str, Any] = field(default_factory=dict)
    bindings: Mapping[str, Any] = field(default_factory=dict)


@dataclass(frozen=True, slots=True)
class ModuleCrossoverResult:
    """The fresh module child and its structural/module crossover report."""

    child: Any
    architecture: Any
    report: Mapping[str, Any]
    parent1: CapturedArchitecture
    parent2: CapturedArchitecture
