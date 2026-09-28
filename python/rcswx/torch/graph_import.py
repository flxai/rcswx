"""Explicit, verified import of ordinary PyTorch computation into a grammar."""

from dataclasses import replace

import torch
from torch import nn
from torch.fx import symbolic_trace

from ._isolation import isolated_model, preserve_rng
from .grammar_backends import resolve_backend
from .provenance import (
    _assert_no_sharing,
    _converter_bindings,
    _execution_assumptions,
    _structure_signature,
    _validate_bindings,
)
from .types import CapturedArchitecture, UnsupportedModuleError
from .validation import verify_import


def checked_inputs(values, *, context: str) -> tuple[torch.Tensor, ...]:
    if not isinstance(values, (tuple, list)) or len(values) != 1:
        raise UnsupportedModuleError(f"{context}: supply a tuple containing one example tensor")
    value = values[0]
    if not isinstance(value, torch.Tensor):
        raise UnsupportedModuleError(f"{context}: expected a tensor input")
    if value.dtype not in {torch.float16, torch.bfloat16, torch.float32, torch.float64}:
        raise UnsupportedModuleError(f"{context}: unsupported input dtype {value.dtype}")
    if value.layout != torch.strided or value.device.type not in {"cpu", "cuda"}:
        raise UnsupportedModuleError(f"{context}: requires a real strided CPU/CUDA tensor")
    if value.ndim not in (2, 3, 4) or any(dim <= 0 for dim in value.shape):
        raise UnsupportedModuleError(
            f"{context}: requires a positive concrete rank-2, -3, or -4 shape"
        )
    if not torch.isfinite(value).all():
        raise UnsupportedModuleError(f"{context}: example inputs must be finite")
    return (value,)


def input_signature(value: torch.Tensor) -> dict:
    return {"shape": list(value.shape), "dtype": str(value.dtype), "device": str(value.device)}


def import_model(
    model: nn.Module,
    *,
    example_inputs,
    grammar: str = "einspace",
    grammar_version: str = "1",
) -> CapturedArchitecture:
    """Import and verify a supported model, without mutating or retaining it.

    Unlike ``capture``, this explicit workflow symbolically executes an isolated
    model and numerically verifies its reconstruction. It is not a sandbox for
    external side effects in arbitrary Python forward implementations.
    """
    backend = resolve_backend(grammar, grammar_version)
    inputs = checked_inputs(example_inputs, context="import")
    if not isinstance(model, nn.Module):
        raise TypeError("import_model requires torch.nn.Module")
    _assert_no_sharing(model)
    for path, module in model.named_modules():
        if module._forward_hooks or module._forward_pre_hooks or module._backward_hooks:
            raise UnsupportedModuleError(f"capture: module hooks at {path!r} are unsupported")
    execution = _execution_assumptions(model)
    if len(set(execution["module_training"].values())) != 1:
        raise UnsupportedModuleError("capture: mixed module training modes are unsupported")
    source_structure = _structure_signature(model)
    input_spec = {
        "shape": list(inputs[0].shape),
        "mode": "im" if inputs[0].ndim == 4 else "col",
        "other_shape": None,
        "other_mode": None,
        "branching_factor": 1,
        "last_im_shape": None,
    }
    with preserve_rng((*model.parameters(), *model.buffers(), *inputs)):
        isolated = isolated_model(model, meta=True)
        before = _structure_signature(isolated)
        try:
            graph = symbolic_trace(nn.Sequential(isolated))
        except Exception as error:
            raise UnsupportedModuleError(
                f"capture: FX cannot represent the source: {error}"
            ) from error
        if _structure_signature(isolated) != before:
            raise UnsupportedModuleError("capture: forward mutated source module configuration")
        lowered = backend.lower(graph, input_spec)

        # Wrapping makes a standalone standard layer trace as a leaf too.
        # Remove that private wrapper from the source correspondence only.
        def source_path(path):
            if path == "0":
                return ""
            if path.startswith("0."):
                return path[2:]
            raise UnsupportedModuleError(f"capture: unbound source state {path!r}")

        lowered = replace(
            lowered,
            bindings={
                index: {
                    **binding,
                    "module_paths": [source_path(path) for path in binding["module_paths"]],
                }
                for index, binding in lowered.bindings.items()
            },
            state_bindings=tuple(
                replace(
                    binding,
                    parameters={
                        role: source_path(path) for role, path in binding.parameters.items()
                    },
                    buffers={role: source_path(path) for role, path in binding.buffers.items()},
                )
                for binding in lowered.state_bindings
            ),
        )
        verification = verify_import(model, lowered, inputs)
    architecture_data = lowered.architecture.to_dict()
    captured = CapturedArchitecture(
        architecture=lowered.architecture,
        input_spec=input_spec,
        metadata={
            "capture_kind": "graph_import",
            "grammar": grammar,
            "grammar_version": grammar_version,
            "importer": {"name": "fx", "version": "1"},
            "execution": execution,
            "input_signature": input_signature(inputs[0]),
            "validation": verification,
        },
        bindings={"occurrences": _converter_bindings(lowered.bindings, architecture_data)},
        structure=source_structure,
    )
    _validate_bindings(model, captured)
    return captured
