"""Build fresh PyTorch modules through the existing grammar callbacks."""

from collections.abc import Mapping
from typing import Any

import torch

from ._data import data_mapping, restore_input_spec, same_data
from .types import BuildOptions, CapturedArchitecture, UnsupportedModuleError

_DEFAULT_LIMITS = {
    "time": 60,
    "restart_time": 300,
    "max_id": 10000,
    "depth": 20,
    "memory": 8196,
    "memory_crossover": 65536,
    "individual_memory": 1024,
    "batch_pass_seconds": 0.1,
}


def build(
    architecture: Any,
    *,
    input_spec: Mapping[str, Any] | None = None,
    build_options: BuildOptions | Mapping[str, Any] | None = None,
) -> torch.nn.Module:
    """Build a fresh module from a portable or captured RCSWX architecture.

    This is deliberately grammar reconstruction, not a generic graph compiler.
    It performs no validation forward pass and never copies learned parameters,
    buffer values, gradients, or optimizer state. Captured execution settings
    are restored unless the caller explicitly overrides them.
    """
    portable, captured = _coerce_architecture(architecture)
    data = portable.to_dict()
    resolved_input = _resolve_input_spec(data, captured, input_spec)
    retained = captured is not None and captured.metadata.get("capture_kind") == "retained"
    options = resolve_build_options(captured, build_options, allow_mixed=retained)
    root = _reconstruct_tree(data, resolved_input)
    model = root.build(root)
    if options.device is not None and options.dtype is not None:
        model = model.to(device=options.device, dtype=options.dtype)
    elif options.device is not None:
        model = model.to(device=options.device)
    elif options.dtype is not None:
        model = model.to(dtype=options.dtype)
    if options.training is not None:
        model.train(options.training)
    if captured is not None:
        _restore_captured_execution(model, captured, _coerce_options(build_options))

    from .provenance import attach_built_provenance

    attach_built_provenance(model, portable, resolved_input, root)
    return model


def _coerce_architecture(value: Any) -> tuple[Any, CapturedArchitecture | None]:
    from rcswx.portable import Architecture

    if isinstance(value, CapturedArchitecture):
        return value.architecture, value
    if not isinstance(value, Architecture):
        raise TypeError("build requires rcswx.portable.Architecture or CapturedArchitecture")
    return value, None


def _resolve_input_spec(
    architecture: Mapping[str, Any],
    captured: CapturedArchitecture | None,
    explicit: Mapping[str, Any] | None,
) -> dict[str, Any]:
    stored = architecture.get("input_spec")
    captured_spec = captured.input_spec if captured is not None else None
    if captured_spec is not None and stored is not None and not same_data(captured_spec, stored):
        raise UnsupportedModuleError("captured architecture has inconsistent input assumptions")

    selected = (
        explicit
        if explicit is not None
        else (captured_spec if captured_spec is not None else stored)
    )
    if selected is None:
        raise UnsupportedModuleError(
            "build requires an explicit input_spec when the portable architecture has none"
        )
    selected_data = data_mapping(selected, context="input_spec")
    if stored is not None and explicit is not None and not same_data(stored, selected_data):
        raise UnsupportedModuleError(
            "explicit input_spec conflicts with the portable architecture assumptions"
        )
    return selected_data


def _coerce_options(value: BuildOptions | Mapping[str, Any] | None) -> BuildOptions:
    if value is None:
        return BuildOptions()
    if isinstance(value, BuildOptions):
        return value
    if not isinstance(value, Mapping):
        raise TypeError("build_options must be BuildOptions, a mapping, or None")
    allowed = {"device", "dtype", "training"}
    unknown = set(value) - allowed
    if unknown:
        raise TypeError(f"unsupported build_options: {', '.join(sorted(unknown))}")
    training = value.get("training")
    if training is not None and not isinstance(training, bool):
        raise TypeError("build_options.training must be bool or None")
    return BuildOptions(device=value.get("device"), dtype=value.get("dtype"), training=training)


def resolve_build_options(
    captured: CapturedArchitecture | None,
    value: BuildOptions | Mapping[str, Any] | None,
    *,
    allow_mixed: bool = False,
) -> BuildOptions:
    """Merge explicit build choices over compatible captured execution state."""
    explicit = _coerce_options(value)
    if captured is None:
        return explicit
    defaults = _captured_execution_options(captured, explicit, allow_mixed)
    return BuildOptions(
        device=explicit.device if explicit.device is not None else defaults.device,
        dtype=explicit.dtype if explicit.dtype is not None else defaults.dtype,
        training=explicit.training if explicit.training is not None else defaults.training,
    )


def _captured_execution_options(
    captured: CapturedArchitecture, explicit: BuildOptions, allow_mixed: bool
) -> BuildOptions:
    metadata = data_mapping(captured.metadata, context="captured metadata")
    execution = metadata.get("execution")
    if not isinstance(execution, Mapping):
        raise UnsupportedModuleError("captured architecture has no execution assumptions")
    device_name = _single_execution_value(
        execution, "devices", allow_mixed=allow_mixed or explicit.device is not None
    )
    dtype_name = _single_execution_value(
        execution, "dtypes", allow_mixed=allow_mixed or explicit.dtype is not None
    )
    training = execution.get("training")
    if not isinstance(training, bool):
        raise UnsupportedModuleError("captured execution training flag is invalid")
    try:
        device = torch.device(device_name) if device_name is not None else None
    except (RuntimeError, TypeError) as error:
        raise UnsupportedModuleError("captured execution device is invalid") from error
    dtype = _torch_dtype(dtype_name) if dtype_name is not None else None
    return BuildOptions(device=device, dtype=dtype, training=training)


def _single_execution_value(
    execution: Mapping[str, Any], field: str, *, allow_mixed: bool = False
) -> str | None:
    values = execution.get(field)
    if not isinstance(values, list) or any(not isinstance(value, str) for value in values):
        raise UnsupportedModuleError(f"captured execution {field} are invalid")
    if field == "dtypes":
        # Module.to(dtype=...) leaves integer counters alone. BatchNorm's
        # floating weights plus int64 num_batches_tracked are not ambiguous.
        values = [
            value
            for value in values
            if _torch_dtype(value).is_floating_point or _torch_dtype(value).is_complex
        ]
    if len(values) > 1:
        if allow_mixed:
            return None
        raise UnsupportedModuleError(
            f"captured execution has multiple {field}; specify an explicit build option"
        )
    return values[0] if values else None


def _torch_dtype(name: str) -> torch.dtype:
    attribute = name.removeprefix("torch.")
    dtype = getattr(torch, attribute, None)
    if not isinstance(dtype, torch.dtype):
        raise UnsupportedModuleError(f"captured execution dtype {name!r} is invalid")
    return dtype


def _restore_captured_execution(
    model: torch.nn.Module, captured: CapturedArchitecture, explicit: BuildOptions
) -> None:
    from .provenance import _named_buffers, _named_modules, _named_parameters

    metadata = data_mapping(captured.metadata, context="captured metadata")
    execution = metadata["execution"]
    modes = execution.get("module_training")
    if not isinstance(modes, Mapping) or any(
        not isinstance(path, str) or not isinstance(mode, bool) for path, mode in modes.items()
    ):
        raise UnsupportedModuleError("captured per-module training modes are invalid")
    if metadata.get("capture_kind") != "retained":
        # A converter can change the module hierarchy. Uniform execution
        # defaults transfer; mixed local modes require an explicit decision.
        if explicit.training is None and len(set(modes.values())) > 1:
            raise UnsupportedModuleError(
                "converted mixed module modes require an explicit training build option"
            )
        return

    modules = dict(_named_modules(model))
    if set(modes) != set(modules):
        raise UnsupportedModuleError("captured execution module paths no longer match the grammar")
    placements = execution.get("tensors")
    tensors = {
        path: (kind, tensor)
        for kind, entries in (
            ("parameter", _named_parameters(model)),
            ("buffer", _named_buffers(model)),
        )
        for path, tensor in entries
    }
    if not isinstance(placements, Mapping) or set(placements) != set(tensors):
        raise UnsupportedModuleError("captured execution tensor paths no longer match the grammar")
    for path, (kind, tensor) in tensors.items():
        placement = placements[path]
        if not isinstance(placement, Mapping) or placement.get("kind") != kind:
            raise UnsupportedModuleError("captured execution tensor kind is invalid")
        requires_grad = placement.get("requires_grad")
        if not isinstance(requires_grad, bool):
            raise UnsupportedModuleError("captured execution gradient flag is invalid")
        dtype = _torch_dtype(placement["dtype"])
        if explicit.dtype is not None and (dtype.is_floating_point or dtype.is_complex):
            dtype = explicit.dtype
        device = torch.device(
            explicit.device if explicit.device is not None else placement["device"]
        )
        if tensor.dtype == dtype and tensor.device == device:
            tensor.requires_grad_(requires_grad)
            continue
        owner_path, _, name = path.rpartition(".")
        owner = modules[owner_path]
        converted = tensor.detach().to(device=device, dtype=dtype)
        if kind == "parameter":
            setattr(owner, name, torch.nn.Parameter(converted, requires_grad=requires_grad))
        else:
            setattr(owner, name, converted.requires_grad_(requires_grad))
    if explicit.training is None:
        for path, module in modules.items():
            module.training = modes[path]


def _reconstruct_tree(architecture: Mapping[str, Any], input_spec: Mapping[str, Any]) -> Any:
    grammar = _grammar_module(architecture.get("grammar"), architecture.get("grammar_version"))
    nodes = architecture.get("nodes")
    root_index = architecture.get("root")
    if not isinstance(nodes, list) or not isinstance(root_index, int):
        raise UnsupportedModuleError("portable architecture has no valid occurrence tree")
    preorder = _preorder_nodes(nodes, root_index)
    _check_callback_parameters(preorder)
    operation_by_name = _operation_lookup(grammar.grammar)
    try:
        operations = [operation_by_name[node["name"]] for node in preorder]
    except (KeyError, TypeError) as error:
        raise UnsupportedModuleError(
            "architecture uses an operation unavailable from its installed grammar"
        ) from error

    from rcswx import PCFG, Limiter, Reconstructor

    limiter = Limiter(dict(_DEFAULT_LIMITS))
    limiter.timer.start()
    root = Reconstructor(PCFG(grammar.grammar, limiter), limiter, mode="iterative").sample(
        input_params=restore_input_spec(input_spec), operations=operations
    )
    rebuilt = root.serialise()
    expected = [(node["name"], len(node.get("children", []))) for node in preorder]
    actual = [(node.operation.name, len(node.children)) for node in rebuilt]
    if actual != expected or len(rebuilt) != len(nodes):
        raise UnsupportedModuleError(
            "architecture occurrence topology is not representable by its grammar callbacks"
        )
    return root


def _check_callback_parameters(nodes: list[Mapping[str, Any]]) -> None:
    for node in nodes:
        if node.get("parameters") not in (None, {}):
            raise UnsupportedModuleError(
                f"operation {node['name']!r} has parameters unsupported by the installed "
                "grammar callback reconstruction"
            )


def _grammar_module(grammar: Any, version: Any) -> Any:
    if version != "1":
        raise UnsupportedModuleError(f"unsupported grammar version {version!r}")
    if grammar == "einspace":
        from rcswx.grammars import einspace

        return einspace
    if grammar == "hnasbench201":
        from rcswx.grammars import hnasbench201

        return hnasbench201
    raise UnsupportedModuleError(f"unsupported grammar {grammar!r}")


def _operation_lookup(grammar: Mapping[str, Mapping[str, Any]]) -> dict[str, Any]:
    operations: dict[str, Any] = {}
    for rule in grammar.values():
        for operation in rule["options"]:
            previous = operations.setdefault(operation.name, operation)
            if previous != operation:
                raise UnsupportedModuleError(
                    f"installed grammar has conflicting operations named {operation.name!r}"
                )
    return operations


def _preorder_nodes(nodes: list[Any], root: int) -> list[Mapping[str, Any]]:
    if root < 0 or root >= len(nodes):
        raise UnsupportedModuleError("portable architecture root is out of range")
    seen: set[int] = set()
    active: set[int] = set()
    ordered: list[Mapping[str, Any]] = []

    def visit(index: int) -> None:
        if index in active:
            raise UnsupportedModuleError("portable architecture contains an occurrence cycle")
        if index in seen:
            raise UnsupportedModuleError(
                "portable architecture shares an occurrence between parents"
            )
        if index < 0 or index >= len(nodes) or not isinstance(nodes[index], Mapping):
            raise UnsupportedModuleError("portable architecture contains an invalid occurrence")
        node = nodes[index]
        if not isinstance(node.get("name"), str):
            raise UnsupportedModuleError("portable architecture occurrence has no operation name")
        children = node.get("children", [])
        if not isinstance(children, list) or any(not isinstance(item, int) for item in children):
            raise UnsupportedModuleError("portable architecture occurrence has invalid children")
        active.add(index)
        ordered.append(node)
        for child in children:
            visit(child)
        active.remove(index)
        seen.add(index)

    visit(root)
    if len(seen) != len(nodes):
        raise UnsupportedModuleError("portable architecture has unreachable occurrences")
    return ordered
