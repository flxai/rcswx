"""Fresh-module crossover composed exclusively from the portable API."""

from collections.abc import Mapping
from typing import Any

import torch

from ._data import same_data
from .build import build, resolve_build_options
from .grammar_backends import resolve_backend
from .graph_import import checked_inputs, input_signature
from .provenance import capture, captured_from_manifest
from .types import BuildOptions, CapturedArchitecture, ModuleCrossoverResult, UnsupportedModuleError
from .validation import validate_architecture, validate_runtime


def crossover(
    parent1: Any,
    parent2: Any,
    *,
    input_spec: Mapping[str, Any] | None = None,
    importer: str | None = None,
    skewness: float = 0,
    sampler: str = "native",
    seed: Any = None,
    rng: Any = None,
    build_options: Any = None,
) -> Any:
    """Return a freshly initialized module child without inherited state."""
    return crossover_with_report(
        parent1,
        parent2,
        input_spec=input_spec,
        importer=importer,
        skewness=skewness,
        sampler=sampler,
        seed=seed,
        rng=rng,
        build_options=build_options,
    ).child


def crossover_with_report(
    parent1: Any,
    parent2: Any,
    *,
    input_spec: Mapping[str, Any] | None = None,
    importer: str | None = None,
    skewness: float = 0,
    sampler: str = "native",
    seed: Any = None,
    rng: Any = None,
    build_options: Any = None,
) -> ModuleCrossoverResult:
    """Capture, cross portably, and build one fresh module child.

    This wrapper has no validation forward pass and no retry loop. It does not
    call the legacy ``validated_crossover`` API. Parent modules are inspected
    only for retained structural provenance, leaving parameters, buffers,
    gradients, training mode, and topology untouched.
    """
    captured1 = capture(parent1, input_spec=input_spec, importer=importer)
    captured2 = capture(parent2, input_spec=input_spec, importer=importer)
    effective_options = _crossover_build_options(captured1, captured2, build_options)
    _check_compatible(captured1, captured2)
    return _cross_captured(
        captured1,
        captured2,
        effective_options,
        skewness=skewness,
        sampler=sampler,
        seed=seed,
        rng=rng,
    )


def crossover_imported(
    parent1: CapturedArchitecture,
    parent2: CapturedArchitecture,
    *,
    validation_inputs: Any,
    skewness: float = 0,
    sampler: str = "native",
    seed: Any = None,
    rng: Any = None,
    build_options: Any = None,
) -> ModuleCrossoverResult:
    """Cross two verified imports once and return a validated, fresh-state child.

    Grammar and runtime validation are mandatory. An invalid sampled offspring
    raises instead of being repaired or resampled. Validation uses a disposable
    child copy; only ordinary child initialization consumes the Torch RNG.
    """
    captured1, captured2 = _verified_import(parent1), _verified_import(parent2)
    _check_compatible(captured1, captured2)
    inputs = checked_inputs(validation_inputs, context="crossover validation")
    signature = input_signature(inputs[0])
    explicit = resolve_build_options(None, build_options)
    for parent in (captured1, captured2):
        imported_signature = parent.metadata["input_signature"]
        for field, override in (
            ("shape", None),
            ("dtype", explicit.dtype),
            ("device", explicit.device),
        ):
            if override is None and imported_signature[field] != signature[field]:
                raise UnsupportedModuleError(
                    f"crossover validation: input {field} differs from imported assumptions"
                )
    effective_options = _crossover_build_options(captured1, captured2, explicit)
    if effective_options.dtype is not None and effective_options.dtype != inputs[0].dtype:
        raise UnsupportedModuleError("crossover validation: input dtype differs from child dtype")
    if effective_options.device is not None:
        device = torch.device(effective_options.device)
        if device.type != inputs[0].device.type or (
            device.index is not None and device.index != inputs[0].device.index
        ):
            raise UnsupportedModuleError(
                "crossover validation: input device differs from child device"
            )
    return _cross_captured(
        captured1,
        captured2,
        effective_options,
        skewness=skewness,
        sampler=sampler,
        seed=seed,
        rng=rng,
        validation_inputs=inputs,
    )


def _verified_import(value: CapturedArchitecture) -> CapturedArchitecture:
    if not isinstance(value, CapturedArchitecture):
        raise TypeError("crossover_imported requires CapturedArchitecture values from import_model")
    captured = captured_from_manifest(value.to_manifest())
    metadata = captured.metadata
    if metadata.get("capture_kind") != "graph_import" or metadata.get("importer") != {
        "name": "fx",
        "version": "1",
    }:
        raise UnsupportedModuleError("crossover validation: parent is not a verified graph import")
    data = captured.architecture.to_dict()
    resolve_backend(data["grammar"], data["grammar_version"])
    if any(metadata.get(field) != data[field] for field in ("grammar", "grammar_version")):
        raise UnsupportedModuleError("crossover validation: inconsistent grammar metadata")
    if not same_data(captured.input_spec, data.get("input_spec")):
        raise UnsupportedModuleError("crossover validation: inconsistent input assumptions")
    signature, validation = metadata.get("input_signature"), metadata.get("validation")
    if (
        not isinstance(signature, Mapping)
        or signature.get("shape") != captured.input_spec.get("shape")
        or not isinstance(signature.get("dtype"), str)
        or not isinstance(signature.get("device"), str)
        or not isinstance(validation, Mapping)
        or not isinstance(validation.get("checks"), list)
        or any(
            check not in validation["checks"]
            for check in (
                "grammar",
                "state_correspondence",
                "forward",
                "buffer_updates",
            )
        )
        or validation.get("training_modes") not in ([True, False], [False, True])
    ):
        raise UnsupportedModuleError(
            "crossover validation: missing or inconsistent import verification"
        )
    expected = validate_architecture(captured.architecture, captured.input_spec)
    if validation.get("output") != {
        **expected,
        "dtype": signature["dtype"],
        "device": signature["device"],
    }:
        raise UnsupportedModuleError("crossover validation: inconsistent verified output contract")
    return captured


def _cross_captured(
    captured1,
    captured2,
    effective_options,
    *,
    skewness,
    sampler,
    seed,
    rng,
    validation_inputs=None,
):

    from rcswx.crossover import crossover_with_report as portable_crossover_with_report
    from rcswx.portable import Architecture

    portable_result = portable_crossover_with_report(
        captured1.architecture,
        captured2.architecture,
        skewness=skewness,
        sampler=sampler,
        seed=seed,
        rng=rng,
    )
    if not isinstance(portable_result.child, Architecture):
        raise UnsupportedModuleError("portable crossover did not return a portable Architecture")
    expected = (
        validate_architecture(portable_result.child, captured1.input_spec)
        if validation_inputs is not None
        else None
    )
    child = build(
        portable_result.child,
        input_spec=captured1.input_spec,
        build_options=effective_options,
    )
    child_capture = capture(child)
    report = dict(portable_result.report)
    if validation_inputs is not None:
        report["validation"] = validate_runtime(child, validation_inputs, expected)
    report.update(
        {
            "module_architecture": portable_result.child.to_dict(),
            "module_capture": {
                "parent1": captured1.to_manifest(),
                "parent2": captured2.to_manifest(),
                "child": child_capture.to_manifest(),
            },
        }
    )
    return ModuleCrossoverResult(
        child=child,
        architecture=portable_result.child,
        report=report,
        parent1=captured1,
        parent2=captured2,
    )


def _crossover_build_options(
    left: CapturedArchitecture,
    right: CapturedArchitecture,
    explicit: Any,
) -> BuildOptions:
    """Use compatible materialization and parent1 mode unless overridden.

    Tensor-free parent operations contribute no device or dtype constraint.
    Training mode does not constrain portable materialization, so parent1's
    mode remains the deterministic default unless the caller explicitly sets it.
    """
    left_options = resolve_build_options(left, explicit)
    right_options = resolve_build_options(right, explicit)
    return BuildOptions(
        device=_shared_materialization_value("device", left_options.device, right_options.device),
        dtype=_shared_materialization_value("dtype", left_options.dtype, right_options.dtype),
        training=left_options.training,
    )


def _shared_materialization_value(name: str, left: Any, right: Any) -> Any:
    if left is not None and right is not None and left != right:
        raise UnsupportedModuleError(
            f"module crossover parents have incompatible {name} settings; "
            f"supply an explicit {name} build option"
        )
    return left if left is not None else right


def _check_compatible(left: CapturedArchitecture, right: CapturedArchitecture) -> None:
    left_data = left.architecture.to_dict()
    right_data = right.architecture.to_dict()
    if (left_data["grammar"], left_data["grammar_version"]) != (
        right_data["grammar"],
        right_data["grammar_version"],
    ):
        raise UnsupportedModuleError("module crossover requires matching grammar semantics")
    if not same_data(left.input_spec, right.input_spec):
        raise UnsupportedModuleError("module crossover requires matching input assumptions")
