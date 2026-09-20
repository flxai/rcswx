"""Public request boundaries; native calls receive only owned primitive metadata."""

import time
from dataclasses import dataclass, replace

import torch
from torch import nn

from . import _core
from .errors import (
    BudgetExceeded,
    Cancelled,
    InputContractMismatch,
    InvalidEditSelection,
    OutputContractMismatch,
    StaleEditPath,
    _native_call,
)
from .frontend import prepare
from .policies import PROFILE_VERSION, WITNESS_VERSION, Limits
from .types import Architecture, Operation, TensorSpec

CancellationToken = _core.CancellationToken
_KINDS = ("match", "substitute", "delete", "insert")


class _Request:
    __slots__ = ("limits", "token", "start")

    def __init__(self, limits: Limits | None, cancel: CancellationToken | None):
        self.limits = Limits() if limits is None else limits
        if type(self.limits) is not Limits:
            raise TypeError("limits must be Limits")
        self.token = CancellationToken() if cancel is None else cancel
        if type(self.token) is not CancellationToken:
            raise TypeError("cancel must be a CancellationToken")
        self.start = time.monotonic()
        self.check()

    def check(self) -> None:
        if self.token.is_cancelled():
            raise Cancelled("request cancelled")
        elapsed = time.monotonic() - self.start
        if elapsed >= self.limits.deadline_seconds:
            raise BudgetExceeded("deadline_seconds", self.limits.deadline_seconds, elapsed)

    def remaining(self) -> Limits:
        self.check()
        seconds = self.limits.deadline_seconds - (time.monotonic() - self.start)
        if seconds <= 0:
            self.check()
        return replace(self.limits, deadline_seconds=seconds)

    def native_limits(self):
        return _native_call(_core.Limits, **self.remaining().core_kwargs())


def _parents(
    source,
    target,
    *,
    example_inputs,
    input_spec,
    request: _Request,
    fallback_spec=None,
    retained_path=None,
):
    prepared = type(source) is Architecture, type(target) is Architecture
    if prepared[0] != prepared[1]:
        raise TypeError("use two live modules or two prepared architectures, not mixed inputs")
    held = 0 if retained_path is None else retained_path.native.owned_bytes
    known = (
        set()
        if retained_path is None
        else {id(retained_path.source.native), id(retained_path.target.native)}
    )
    ceiling = request.limits.max_core_bytes
    if held > ceiling:
        raise BudgetExceeded("max_core_bytes", ceiling, held)
    if prepared[0]:
        if example_inputs is not None or input_spec is not None:
            raise ValueError("prepared architectures already contain their input contracts")
        pair, donors = (source, target), None
    else:
        if not isinstance(source, nn.Module) or not isinstance(target, nn.Module):
            raise TypeError("inputs must both be modules or prepared architectures")
        if example_inputs is None and input_spec is None:
            input_spec = fallback_spec
        pair, captured = [], {}
        for model in (source, target):
            # A caller must prevent concurrent mutation. Repeating the identical
            # donor in one request cannot change its capture or binding contract.
            if id(model) not in captured:
                if held >= ceiling:
                    raise BudgetExceeded("max_core_bytes", ceiling, held + 1)
                try:
                    architecture = prepare(
                        model,
                        example_inputs=example_inputs,
                        input_spec=input_spec,
                        limits=replace(request.remaining(), max_core_bytes=ceiling - held),
                    )
                except BudgetExceeded as error:
                    if error.resource == "max_core_bytes":
                        raise BudgetExceeded(
                            "max_core_bytes", ceiling, held + error.observed
                        ) from None
                    raise
                held += architecture.native.owned_bytes
                known.add(id(architecture.native))
                captured[id(model)] = architecture
            pair.append(captured[id(model)])
        donors = source, target
    for architecture in pair:
        if id(architecture.native) not in known:
            held += architecture.native.owned_bytes
            known.add(id(architecture.native))
        if len(architecture.operations) > request.limits.max_nodes:
            raise BudgetExceeded(
                "max_nodes", request.limits.max_nodes, len(architecture.operations)
            )
        rank = max(len(architecture.input_spec.shape), len(architecture.output_spec.shape))
        if rank > request.limits.max_rank:
            raise BudgetExceeded("max_rank", request.limits.max_rank, rank)
    if held > ceiling:
        raise BudgetExceeded("max_core_bytes", ceiling, held)
    request.check()
    return pair[0], pair[1], donors


def _contracts(source: Architecture, target: Architecture) -> None:
    if source.input_spec != target.input_spec:
        raise InputContractMismatch("an executable path requires equal root input contracts")
    if source.output_spec != target.output_spec:
        raise OutputContractMismatch("an executable path requires equal root output contracts")


@dataclass(frozen=True, slots=True)
class Edit:
    id: int
    cost_ticks: int
    kind: str
    source_address: int | None
    target_address: int | None


@dataclass(frozen=True, slots=True, init=False, weakref_slot=True)
class EditPath:
    source: Architecture
    target: Architecture
    native: object
    steps: tuple[tuple[int, int | None, int | None, int, int | None], ...]
    edits: tuple[Edit, ...]
    versions: tuple[str, str]

    def __init__(self) -> None:
        raise TypeError("EditPath instances are created by edit_path()")

    @classmethod
    def _create(cls, source: Architecture, target: Architecture, native) -> EditPath:
        result = object.__new__(cls)
        steps = tuple(native.steps)
        edits = tuple(
            Edit(edit_id, cost, _KINDS[kind], left, right)
            for kind, left, right, cost, edit_id in steps
            if edit_id is not None
        )
        for name, value in (
            ("source", source),
            ("target", target),
            ("native", native),
            ("steps", steps),
            ("edits", edits),
            ("versions", (PROFILE_VERSION, WITNESS_VERSION)),
        ):
            object.__setattr__(result, name, value)
        return result

    @property
    def cost_ticks(self) -> int:
        return self.native.cost_ticks

    @property
    def distance(self) -> float:
        return self.cost_ticks / 4

    def _project(
        self, mask: int, limits: Limits | None = None, token: CancellationToken | None = None
    ) -> tuple[tuple[Operation, ...], tuple[tuple[int, int], ...]]:
        if type(mask) is not int or not 0 <= mask < 1 << len(self.edits):
            raise InvalidEditSelection("selection contains an unknown edit bit")
        request = _Request(limits, token)
        origins = tuple(
            _native_call(self.native.project, mask, request.native_limits(), request.token)
        )
        operations = tuple(
            (self.source if side == 0 else self.target).operations[index] for side, index in origins
        )
        request.check()
        return operations, origins

    def _selected_cost(self, mask: int) -> int:
        if type(mask) is not int or not 0 <= mask < 1 << len(self.edits):
            raise InvalidEditSelection("selection contains an unknown edit bit")
        return _native_call(self.native.selected_cost, mask)


def _validate_path(path: EditPath, source: Architecture, target: Architecture) -> None:
    if type(path) is not EditPath:
        raise TypeError("path must be an EditPath")
    if path.versions != (PROFILE_VERSION, WITNESS_VERSION):
        raise StaleEditPath("path profile or witness version differs")
    for actual, recorded in ((source, path.source), (target, path.target)):
        if (
            actual.architecture_key != recorded.architecture_key
            or actual.canonical != recorded.canonical
        ):
            raise StaleEditPath("path does not describe these directed parent architectures")
        if actual.binding_key != recorded.binding_key or actual.bindings != recorded.bindings:
            raise StaleEditPath("path occurrence bindings differ from the supplied parents")


def distance(
    source: nn.Module | Architecture,
    target: nn.Module | Architecture,
    *,
    example_inputs: tuple[torch.Tensor, ...] | None = None,
    input_spec: TensorSpec | None = None,
    limits: Limits | None = None,
    cancel: CancellationToken | None = None,
) -> float:
    """Exact structural distance in quarter units; root contracts may differ."""
    request = _Request(limits, cancel)
    source, target, _ = _parents(
        source, target, example_inputs=example_inputs, input_spec=input_spec, request=request
    )
    ticks, _ = _native_call(
        _core.distance, source.native, target.native, request.native_limits(), request.token
    )
    request.check()
    return ticks / 4


def edit_path(
    source: nn.Module | Architecture,
    target: nn.Module | Architecture,
    *,
    example_inputs: tuple[torch.Tensor, ...] | None = None,
    input_spec: TensorSpec | None = None,
    limits: Limits | None = None,
    cancel: CancellationToken | None = None,
) -> EditPath:
    """Compute one immutable canonical complete witness for a common root task."""
    request = _Request(limits, cancel)
    source, target, _ = _parents(
        source, target, example_inputs=example_inputs, input_spec=input_spec, request=request
    )
    _contracts(source, target)
    native = _native_call(
        _core.align, source.native, target.native, request.native_limits(), request.token
    )
    result = EditPath._create(source, target, native)
    request.check()
    return result
