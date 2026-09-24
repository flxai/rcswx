"""Fresh-module crossover composed exclusively from the portable API."""

from collections.abc import Mapping
from typing import Any

from ._data import same_data
from .build import build, resolve_build_options
from .provenance import capture
from .types import BuildOptions, CapturedArchitecture, ModuleCrossoverResult, UnsupportedModuleError


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
    child = build(
        portable_result.child,
        input_spec=captured1.input_spec,
        build_options=effective_options,
    )
    child_capture = capture(child)
    report = dict(portable_result.report)
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
