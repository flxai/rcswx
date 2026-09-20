"""One exact witness, IID mask proposals, then independent module construction."""

import time
from dataclasses import dataclass

import torch
from torch import nn

from . import _core
from .api import CancellationToken, EditPath, _contracts, _parents, _Request, _validate_path
from .errors import MissingDonor, SamplingExhausted, UnsupportedConfiguration, _native_call
from .materialize import materialize
from .operators import infer
from .policies import (
    PROFILE_VERSION,
    REPORT_VERSION,
    RNG_VERSION,
    SAMPLING_VERSION,
    WITNESS_VERSION,
    Limits,
    resolve_seed,
    seed_streams,
)
from .types import Architecture, TensorSpec


@dataclass(frozen=True, slots=True)
class CrossoverResult:
    model: nn.Module
    report: dict[str, object]


def crossover_with_report(
    source: nn.Module | Architecture,
    target: nn.Module | Architecture,
    *,
    path: EditPath | None = None,
    example_inputs: tuple[torch.Tensor, ...] | None = None,
    input_spec: TensorSpec | None = None,
    inherit: bool = False,
    init: str = "pytorch",
    seed: int | None = None,
    limits: Limits | None = None,
    cancel: CancellationToken | None = None,
) -> CrossoverResult:
    """Sample with replacement, rejecting shape-invalid proposals up to the finite K.

    Budgets, cancellation, stale bindings and unexpected construction failures abort
    the request, never change the proposal distribution or trigger a fallback.
    """
    request = _Request(limits, cancel)
    if type(inherit) is not bool:
        raise TypeError("inherit must be a bool")
    if type(init) is not str or init not in ("pytorch", "zeros"):
        raise ValueError("init must be 'pytorch' or 'zeros'")
    if path is not None and type(path) is not EditPath:
        raise TypeError("path must be an EditPath")
    resolved_seed = resolve_seed(seed)
    capture_start = time.monotonic()
    source, target, donors = _parents(
        source,
        target,
        example_inputs=example_inputs,
        input_spec=input_spec,
        request=request,
        fallback_spec=None if path is None else path.source.input_spec,
    )
    _contracts(source, target)
    if inherit and donors is None:
        raise MissingDonor("inherit=True requires two live donor modules")
    timings = {"capture_seconds": time.monotonic() - capture_start}
    alignment_start = time.monotonic()
    computed = path is None
    if computed:
        native = _native_call(
            _core.align, source.native, target.native, request.native_limits(), request.token
        )
        path = EditPath._create(source, target, native)
    else:
        _validate_path(path, source, target)
    timings["alignment_seconds"] = time.monotonic() - alignment_start
    sampling_start = time.monotonic()
    sampler = _native_call(
        _core.Sampler, path.native, resolved_seed, request.native_limits(), request.token
    )
    timings["distribution_seconds"] = time.monotonic() - sampling_start
    visited, valid, charged, total = sampler.stats
    report = {
        "report_version": REPORT_VERSION,
        "profile_version": PROFILE_VERSION,
        "witness_version": WITNESS_VERSION,
        "sampling_version": SAMPLING_VERSION,
        "rng_version": RNG_VERSION,
        "source_key": source.architecture_key,
        "target_key": target.architecture_key,
        "source_binding_key": source.binding_key,
        "target_binding_key": target.binding_key,
        "input_shape": source.input_spec.shape,
        "output_shape": source.output_spec.shape,
        "path_cost_ticks": path.cost_ticks,
        "path_construction_stats": dict(path.native.stats),
        "alignment_computations": int(computed),
        "seed": resolved_seed,
        "inherit": inherit,
        "init": init,
        "training": source.training,
        "limits": request.limits.as_dict(),
        "proposal": {
            "visited_masks": visited,
            "valid_masks": valid,
            "charged_bytes": charged,
            "total_weight": total,
        },
        "attempts": 0,
        "rejections": [],
        "selected_mask": None,
        "selected_edit_ids": (),
        "selected_cost_ticks": None,
        "parameter_decisions": (),
        "validation": None,
        "timings": timings,
    }
    validation_start = time.monotonic()
    for attempt in range(1, request.limits.max_attempts + 1):
        request.check()
        mask, origins = _native_call(sampler.propose, request.native_limits(), request.token)
        operations = tuple(
            (source if side == 0 else target).operations[index] for side, index in origins
        )
        report["attempts"] = attempt
        report["proposal"]["charged_bytes"] = sampler.stats[2]
        try:
            inference = infer(operations, source.input_spec, limits=request.remaining())
        except UnsupportedConfiguration:
            reason = "operator_configuration"
        else:
            reason = None if inference.output_spec == source.output_spec else "output_contract"
        request.check()
        if reason is not None:
            report["rejections"].append({"attempt": attempt, "mask": mask, "reason": reason})
            continue
        timings["proposal_validation_seconds"] = time.monotonic() - validation_start
        materialization_start = time.monotonic()
        model, decisions = materialize(
            operations,
            source.input_spec,
            inference=inference,
            training=source.training,
            init=init,
            initialization_seed=seed_streams(resolved_seed)[1],
            inherit=inherit,
            donors=donors,
            donor_architectures=(source, target) if donors is not None else None,
            provenance=tuple(origins),
            limits=request.remaining(),
        )
        request.check()
        timings["materialization_seconds"] = time.monotonic() - materialization_start
        timings["total_seconds"] = time.monotonic() - request.start
        report.update(
            selected_mask=mask,
            selected_edit_ids=tuple(edit.id for edit in path.edits if mask & (1 << edit.id)),
            selected_cost_ticks=path._selected_cost(mask),
            parameter_decisions=decisions,
            validation={
                "structural": True,
                "inferred_modes": True,
                "root_contract": True,
                "child_storage_bytes": inference.storage_bytes,
            },
        )
        return CrossoverResult(model, report)
    timings["proposal_validation_seconds"] = time.monotonic() - validation_start
    timings["materialization_seconds"] = 0.0
    timings["total_seconds"] = time.monotonic() - request.start
    raise SamplingExhausted(report)


def crossover(
    source: nn.Module | Architecture,
    target: nn.Module | Architecture,
    *,
    path: EditPath | None = None,
    example_inputs: tuple[torch.Tensor, ...] | None = None,
    input_spec: TensorSpec | None = None,
    inherit: bool = False,
    init: str = "pytorch",
    seed: int | None = None,
    limits: Limits | None = None,
    cancel: CancellationToken | None = None,
) -> nn.Module:
    """Return only the independent child; use crossover_with_report for diagnostics."""
    return crossover_with_report(
        source,
        target,
        path=path,
        example_inputs=example_inputs,
        input_spec=input_spec,
        inherit=inherit,
        init=init,
        seed=seed,
        limits=limits,
        cancel=cancel,
    ).model
