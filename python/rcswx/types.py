"""Tensor-free, immutable snapshots. Use :func:`prepare` to capture live modules."""

import hashlib
import json
import math
from dataclasses import dataclass, field

from . import _core
from .errors import BudgetExceeded, UnsupportedConfiguration, _native_call
from .policies import (
    GRAMMAR_VERSION,
    NORMALIZATION_VERSION,
    PROFILE_VERSION,
    REGISTRY_VERSION,
    SCHEMA_VERSION,
    Limits,
)

_MAX_DIMENSION = (1 << 63) - 1


def checked_numel(shape: tuple[int, ...]) -> int:
    count = 1
    for dimension in shape:
        if count > _MAX_DIMENSION // dimension:
            raise BudgetExceeded("dimension_product", _MAX_DIMENSION, _MAX_DIMENSION + 1)
        count *= dimension
    return count


@dataclass(frozen=True, slots=True)
class TensorSpec:
    shape: tuple[int, ...]
    dtype: str = "float32"
    device: str = "cpu"
    layout: str = "contiguous"

    def __post_init__(self) -> None:
        if type(self.shape) is not tuple:
            raise TypeError("TensorSpec.shape must be an immutable tuple")
        if len(self.shape) > 64:
            raise BudgetExceeded("max_rank", 64, len(self.shape))
        for dimension in self.shape:
            if type(dimension) is not int or dimension <= 0:
                raise UnsupportedConfiguration("dimensions must be positive fixed integers")
            if dimension > _MAX_DIMENSION:
                raise BudgetExceeded("dimension", _MAX_DIMENSION, _MAX_DIMENSION + 1)
        checked_numel(self.shape)
        if any(type(value) is not str for value in (self.dtype, self.device, self.layout)):
            raise TypeError("contract metadata must contain ordinary primitive strings")
        if (self.dtype, self.device, self.layout) != ("float32", "cpu", "contiguous"):
            raise UnsupportedConfiguration(
                "only contiguous dense CPU float32 contracts are supported"
            )

    def canonical(self) -> tuple:
        return self.shape, self.dtype, self.device, self.layout


_FIELDS = {
    "identity": (),
    "relu": (),
    "linear": ("bias", "out_features"),
    "dropout": ("p",),
    "batch_norm1d": ("affine", "eps", "momentum", "track_running_stats"),
}


@dataclass(frozen=True, slots=True)
class Operation:
    kind: str
    attributes: tuple[tuple[str, bool | int | float | None], ...] = ()

    def __post_init__(self) -> None:
        if type(self.kind) is not str or self.kind not in _FIELDS:
            raise UnsupportedConfiguration("unknown logical operation kind")
        if type(self.attributes) is not tuple or any(
            type(pair) is not tuple or len(pair) != 2 or type(pair[0]) is not str
            for pair in self.attributes
        ):
            raise TypeError("operation attributes must be immutable name/value pairs")
        ordered = tuple(sorted(self.attributes, key=lambda pair: pair[0]))
        if tuple(name for name, _ in ordered) != _FIELDS[self.kind]:
            raise UnsupportedConfiguration(f"invalid attributes for {self.kind}")
        normalized = []
        for name, value in ordered:
            if name in ("bias", "affine", "track_running_stats"):
                if type(value) is not bool:
                    raise UnsupportedConfiguration(f"{name} must be boolean")
            elif name == "out_features":
                if type(value) is not int or not 0 < value <= _MAX_DIMENSION:
                    raise UnsupportedConfiguration(
                        "out_features must be a positive int64 dimension"
                    )
            elif value is None and name == "momentum":
                pass
            else:
                if type(value) not in (int, float) or not math.isfinite(value):
                    raise UnsupportedConfiguration(f"{name} must be a finite real number")
                # Signed zero has no logical effect and must not split canonical keys.
                value = 0.0 if value == 0 else float(value)
                if (
                    name == "eps"
                    and value <= 0
                    or name in ("p", "momentum")
                    and not 0 <= value <= 1
                ):
                    raise UnsupportedConfiguration(f"unsupported {name} value")
            normalized.append((name, value))
        object.__setattr__(self, "attributes", tuple(normalized))

    def get(self, name: str):
        for key, value in self.attributes:
            if key == name:
                return value
        raise KeyError(name)

    def core_record(self) -> tuple[int, int, bool, float, float | None, bool]:
        if self.kind == "linear":
            return 1, self.get("out_features"), self.get("bias"), 0.0, None, False
        if self.kind == "dropout":
            return 3, 0, False, self.get("p"), None, False
        if self.kind == "batch_norm1d":
            return (
                4,
                0,
                self.get("affine"),
                self.get("eps"),
                self.get("momentum"),
                self.get("track_running_stats"),
            )
        return (0 if self.kind == "identity" else 2), 0, False, 0.0, None, False


@dataclass(frozen=True, slots=True)
class ParameterBinding:
    name: str
    shape: tuple[int, ...]
    dtype: str

    def __post_init__(self) -> None:
        if type(self.name) is not str or self.name not in (
            "weight",
            "bias",
            "running_mean",
            "running_var",
            "num_batches_tracked",
        ):
            raise UnsupportedConfiguration("unrecognized parameter or buffer role")
        if type(self.shape) is not tuple or any(type(n) is not int or n <= 0 for n in self.shape):
            raise UnsupportedConfiguration("invalid realized tensor shape")
        checked_numel(self.shape)
        if type(self.dtype) is not str or self.dtype not in ("float32", "int64"):
            raise UnsupportedConfiguration("unsupported realized tensor dtype")


@dataclass(frozen=True, slots=True)
class Binding:
    module_path: str | None
    parameters: tuple[ParameterBinding, ...]
    buffers: tuple[ParameterBinding, ...] = ()

    def __post_init__(self) -> None:
        if self.module_path is not None:
            if type(self.module_path) is not str:
                raise TypeError("binding paths must be strings or None")
            if len(self.module_path.encode("utf-8")) > 4096:
                raise BudgetExceeded("binding_path_bytes", 4096, 4097)
        if any(
            type(group) is not tuple or any(type(item) is not ParameterBinding for item in group)
            for group in (self.parameters, self.buffers)
        ):
            raise TypeError("binding tensors must be immutable metadata tuples")


def _digest(value: tuple) -> str:
    return hashlib.sha256(
        json.dumps(value, separators=(",", ":"), ensure_ascii=True, allow_nan=False).encode()
    ).hexdigest()


@dataclass(frozen=True, slots=True, weakref_slot=True)
class Architecture:
    operations: tuple[Operation, ...]
    input_spec: TensorSpec
    output_spec: TensorSpec
    bindings: tuple[Binding, ...]
    training: bool
    _limits: Limits = field(default_factory=Limits, repr=False, compare=False)
    canonical: tuple = field(init=False, repr=False, compare=False)
    architecture_key: str = field(init=False)
    binding_key: str = field(init=False)
    native: object = field(init=False, repr=False, compare=False)

    def __post_init__(self) -> None:
        if (
            type(self.operations) is not tuple
            or not self.operations
            or any(type(op) is not Operation for op in self.operations)
        ):
            raise TypeError("architecture operations must be a nonempty immutable tuple")
        if (
            type(self.bindings) is not tuple
            or len(self.bindings) != len(self.operations)
            or any(type(binding) is not Binding for binding in self.bindings)
        ):
            raise TypeError("one immutable binding is required per operation")
        if (
            type(self.input_spec) is not TensorSpec
            or type(self.output_spec) is not TensorSpec
            or type(self.training) is not bool
            or type(self._limits) is not Limits
        ):
            raise TypeError("invalid architecture contracts, mode, or limits")
        canonical = (
            SCHEMA_VERSION,
            GRAMMAR_VERSION,
            REGISTRY_VERSION,
            PROFILE_VERSION,
            NORMALIZATION_VERSION,
            self.input_spec.canonical(),
            self.output_spec.canonical(),
            tuple((op.kind, op.attributes) for op in self.operations),
        )
        binding_data = tuple(
            (
                b.module_path,
                tuple((p.name, p.shape, p.dtype) for p in b.parameters),
                tuple((p.name, p.shape, p.dtype) for p in b.buffers),
            )
            for b in self.bindings
        )
        native_limits = _native_call(_core.Limits, **self._limits.core_kwargs())
        native = _native_call(
            _core.Architecture, [op.core_record() for op in self.operations], native_limits
        )
        object.__setattr__(self, "canonical", canonical)
        object.__setattr__(self, "architecture_key", _digest(canonical))
        object.__setattr__(self, "binding_key", _digest((canonical, binding_data)))
        object.__setattr__(self, "native", native)
