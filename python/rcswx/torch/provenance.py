"""Retained module provenance, structural validation, and safe checkpoints."""

from __future__ import annotations

from collections import defaultdict
from collections.abc import Mapping
from copy import deepcopy
from typing import Any

import torch
from torch import nn

from ._data import data_copy, data_mapping, same_data
from .types import (
    CapturedArchitecture,
    ConverterResult,
    StaleProvenanceError,
    UnsafeManifestError,
    UnsupportedModuleError,
)

_CARRIER_NAME = "_rcswx_provenance"
_MANIFEST_FORMAT = "rcswx.torch.manifest"
_CHECKPOINT_FORMAT = "rcswx.torch.checkpoint"


class _ProvenanceCarrier(nn.Module):
    """A registered state-dict participant containing only portable data."""

    def __init__(self, manifest: Mapping[str, Any]):
        super().__init__()
        self._manifest = _manifest_data(manifest)

    def get_extra_state(self) -> dict[str, Any]:
        return deepcopy(self._manifest)

    def set_extra_state(self, state: Any) -> None:
        self._manifest = _manifest_data(state)


def capture(
    model: nn.Module,
    *,
    input_spec: Mapping[str, Any] | None = None,
    importer: str | None = None,
) -> CapturedArchitecture:
    """Capture retained RCSWX provenance or one registered exact-type converter.

    Capture never executes ``forward``. A model bearing stale retained
    provenance raises immediately; it is never silently reinterpreted through
    a converter.
    """
    if not isinstance(model, nn.Module):
        raise TypeError("capture requires torch.nn.Module")
    carrier = _carrier_from(model)
    if carrier is not None:
        return _capture_retained(model, carrier, input_spec)
    return _capture_converted(model, input_spec, importer)


def attach_built_provenance(
    model: nn.Module,
    architecture: Any,
    input_spec: Mapping[str, Any],
    root: Any,
) -> CapturedArchitecture:
    """Attach a data-only manifest to a freshly grammar-built module."""
    if _CARRIER_NAME in model._modules or hasattr(model, _CARRIER_NAME):
        raise UnsupportedModuleError(
            f"cannot attach RCSWX provenance: {_CARRIER_NAME!r} is already reserved"
        )
    architecture_data = data_mapping(architecture.to_dict(), context="architecture")
    bindings = {"occurrences": _occurrence_bindings(root, model, architecture_data)}
    captured = CapturedArchitecture(
        architecture=architecture,
        input_spec=data_mapping(input_spec, context="input_spec"),
        metadata={
            "capture_kind": "retained",
            "grammar": architecture_data["grammar"],
            "grammar_version": architecture_data["grammar_version"],
            "execution": _execution_assumptions(model),
        },
        bindings=bindings,
        structure=_structure_signature(model),
    )
    model.add_module(_CARRIER_NAME, _ProvenanceCarrier(captured.to_manifest()))
    return captured


def save(model: nn.Module, path: Any) -> None:
    """Save a verified manifest beside a normal ``state_dict`` payload.

    The function deliberately does not save a whole module object. ``load``
    uses PyTorch's weights-only mode and the installed grammar allowlist.
    """
    captured = capture(model)
    state_dict = model.state_dict()
    state_dict[_extra_state_key()] = captured.to_manifest()
    torch.save(
        {
            "format": _CHECKPOINT_FORMAT,
            "version": 1,
            "manifest": captured.to_manifest(),
            "state_dict": state_dict,
        },
        path,
    )


def load(
    path: Any,
    *,
    map_location: Any = None,
    build_options: Any = None,
) -> nn.Module:
    """Build and restore a checkpoint without whole-model unpickling.

    All manifests, keys, tensor shapes, and dtypes are checked before the
    newly built module is mutated by ``load_state_dict``.
    """
    payload = torch.load(path, map_location=map_location, weights_only=True)
    if not isinstance(payload, Mapping):
        raise UnsafeManifestError("managed checkpoint must contain a mapping")
    if payload.get("format") != _CHECKPOINT_FORMAT or payload.get("version") != 1:
        raise UnsafeManifestError("unsupported managed checkpoint format")
    captured = captured_from_manifest(payload.get("manifest"))
    state_dict = payload.get("state_dict")
    if not isinstance(state_dict, Mapping):
        raise UnsafeManifestError("managed checkpoint has no state_dict mapping")
    _validate_checkpoint_manifest(state_dict, captured)

    from .build import build

    model = build(captured, build_options=build_options)
    _validate_state_dict(model, state_dict)
    result = model.load_state_dict(state_dict, strict=True)
    if result.missing_keys or result.unexpected_keys:
        raise UnsafeManifestError("managed checkpoint state_dict did not load exactly")
    capture(model)
    return model


def _extra_state_key() -> str:
    return f"{_CARRIER_NAME}._extra_state"


def _validate_checkpoint_manifest(
    state_dict: Mapping[str, Any], captured: CapturedArchitecture
) -> None:
    try:
        embedded = _manifest_data(state_dict[_extra_state_key()])
    except KeyError as error:
        raise UnsafeManifestError(
            "managed checkpoint has no embedded provenance manifest"
        ) from error
    if not same_data(embedded, captured.to_manifest()):
        raise UnsafeManifestError(
            "managed checkpoint outer and state-dictionary provenance manifests differ"
        )


def _validate_state_dict(model: nn.Module, state_dict: Mapping[str, Any]) -> None:
    expected = model.state_dict()
    expected_keys = set(expected)
    actual_keys = set(state_dict)
    if expected_keys != actual_keys:
        missing = sorted(expected_keys - actual_keys)
        unexpected = sorted(actual_keys - expected_keys)
        raise UnsafeManifestError(
            f"managed checkpoint keys differ (missing={missing}, unexpected={unexpected})"
        )
    for key, target in expected.items():
        source = state_dict[key]
        if isinstance(target, torch.Tensor):
            if not isinstance(source, torch.Tensor):
                raise UnsafeManifestError(f"checkpoint entry {key!r} is not a tensor")
            if tuple(source.shape) != tuple(target.shape):
                raise UnsafeManifestError(f"checkpoint tensor shape mismatch for {key!r}")
            if source.dtype != target.dtype:
                raise UnsafeManifestError(f"checkpoint tensor dtype mismatch for {key!r}")
        elif key != _extra_state_key():
            raise UnsafeManifestError(f"checkpoint entry {key!r} has unsupported non-tensor state")


def captured_from_manifest(manifest: Any) -> CapturedArchitecture:
    """Resolve a strictly data-only manifest through the installed API."""
    data = _manifest_data(manifest)
    try:
        from rcswx.portable import Architecture

        architecture = Architecture.from_dict(data["architecture"])
    except (ImportError, TypeError, ValueError) as error:
        raise UnsafeManifestError("manifest architecture is invalid or unsupported") from error
    return CapturedArchitecture(
        architecture=architecture,
        input_spec=data["input_spec"],
        metadata=data["metadata"],
        bindings=data["bindings"],
        structure=data["structure"],
    )


def _capture_retained(
    model: nn.Module,
    carrier: _ProvenanceCarrier,
    explicit_input: Mapping[str, Any] | None,
) -> CapturedArchitecture:
    captured = captured_from_manifest(carrier.get_extra_state())
    if explicit_input is not None and not same_data(explicit_input, captured.input_spec):
        raise StaleProvenanceError("explicit input_spec conflicts with retained provenance")
    _assert_no_sharing(model)
    actual_structure = _structure_signature(model)
    if not same_data(actual_structure, captured.structure):
        raise StaleProvenanceError(
            "retained RCSWX provenance is stale: module topology or configuration changed"
        )
    _validate_bindings(model, captured)
    metadata = data_mapping(captured.metadata, context="metadata")
    metadata["execution"] = _execution_assumptions(model)
    return CapturedArchitecture(
        architecture=captured.architecture,
        input_spec=data_mapping(captured.input_spec, context="input_spec"),
        metadata=metadata,
        bindings=data_mapping(captured.bindings, context="bindings"),
        structure=actual_structure,
    )


def _capture_converted(
    model: nn.Module,
    input_spec: Mapping[str, Any] | None,
    importer: str | None,
) -> CapturedArchitecture:
    _assert_no_sharing(model)
    from .registry import resolve_importer

    registration = resolve_importer(model, importer)
    result = registration.converter(model, input_spec)
    if not isinstance(result, ConverterResult):
        raise TypeError(
            f"converter {registration.name}@{registration.version} must return ConverterResult"
        )

    from rcswx.portable import Architecture

    if not isinstance(result.architecture, Architecture):
        raise TypeError("converter result architecture must be portable Architecture")
    architecture_data = data_mapping(result.architecture.to_dict(), context="architecture")
    resolved_input = _resolve_converter_input(architecture_data, result, input_spec)
    occurrences = _converter_bindings(result.bindings, architecture_data)
    captured = CapturedArchitecture(
        architecture=result.architecture,
        input_spec=resolved_input,
        metadata={
            "capture_kind": "converter",
            "grammar": architecture_data["grammar"],
            "grammar_version": architecture_data["grammar_version"],
            "importer": {"name": registration.name, "version": registration.version},
            "conventions": data_mapping(result.conventions, context="converter conventions"),
            "execution": _execution_assumptions(model),
        },
        bindings={"occurrences": occurrences},
        structure=_structure_signature(model),
    )
    _validate_bindings(model, captured)
    return captured


def _resolve_converter_input(
    architecture: Mapping[str, Any], result: ConverterResult, explicit: Mapping[str, Any] | None
) -> dict[str, Any]:
    choices = [
        item
        for item in (explicit, result.input_spec, architecture.get("input_spec"))
        if item is not None
    ]
    if not choices:
        raise UnsupportedModuleError("converter did not supply required input_spec")
    resolved = data_mapping(choices[0], context="input_spec")
    if any(not same_data(resolved, item) for item in choices[1:]):
        raise UnsupportedModuleError("converter input_spec conflicts with architecture assumptions")
    return resolved


def _converter_bindings(
    value: Mapping[str, Any], architecture: Mapping[str, Any]
) -> dict[str, Any]:
    raw = data_mapping(value, context="converter bindings")
    expected = {str(index) for index in range(len(architecture["nodes"]))}
    if set(raw) != expected:
        raise UnsupportedModuleError(
            "converter bindings must cover every architecture occurrence by its index"
        )
    result: dict[str, Any] = {}
    for index, binding in raw.items():
        if not isinstance(binding, Mapping):
            raise UnsupportedModuleError("converter occurrence binding must be a mapping")
        copied = data_mapping(binding, context="converter occurrence binding")
        module_paths = copied.get("module_paths", [])
        if not isinstance(module_paths, list) or any(
            not isinstance(path, str) for path in module_paths
        ):
            raise UnsupportedModuleError("converter module_paths must be a list of strings")
        copied["logical_id"] = architecture["nodes"][int(index)]["id"]
        result[index] = copied
    return result


def _carrier_from(model: nn.Module) -> _ProvenanceCarrier | None:
    candidate = model._modules.get(_CARRIER_NAME)
    if candidate is None:
        if hasattr(model, _CARRIER_NAME):
            raise StaleProvenanceError(
                "reserved RCSWX provenance attribute is not a manifest carrier"
            )
        return None
    if not isinstance(candidate, _ProvenanceCarrier):
        raise StaleProvenanceError("reserved RCSWX provenance module has an unsupported type")
    return candidate


def _manifest_data(value: Any) -> dict[str, Any]:
    data = data_mapping(value, context="manifest")
    if data.get("format") != _MANIFEST_FORMAT or data.get("version") != 1:
        raise UnsafeManifestError("unsupported RCSWX provenance manifest")
    required = {"architecture", "input_spec", "metadata", "bindings", "structure"}
    if not required.issubset(data):
        raise UnsafeManifestError("RCSWX provenance manifest is missing required fields")
    for key in required:
        data[key] = data_mapping(data[key], context=f"manifest {key}")
    return data


def _structure_signature(model: nn.Module) -> dict[str, Any]:
    modules = []
    for path, module in _named_modules(model):
        modules.append(
            {
                "path": path,
                "type": _type_name(module),
                "config": _module_config(module),
            }
        )
    parameters = [
        {"path": path, "shape": list(parameter.shape)}
        for path, parameter in _named_parameters(model)
    ]
    buffers = [
        {"path": path, "shape": list(buffer.shape)} for path, buffer in _named_buffers(model)
    ]
    return {"modules": modules, "parameters": parameters, "buffers": buffers}


def _module_config(module: nn.Module) -> dict[str, Any]:
    config: dict[str, Any] = {}
    for name, value in sorted(module.__dict__.items()):
        if name.startswith("_") or name == "training":
            continue
        if isinstance(value, (nn.Module, torch.Tensor)):
            continue
        if callable(value):
            config[name] = {"callable_type": _type_name(value)}
            continue
        try:
            config[name] = data_copy(value, context=f"module configuration {name}")
        except UnsafeManifestError:
            config[name] = {"python_type": _type_name(value)}
    return config


def _execution_assumptions(model: nn.Module) -> dict[str, Any]:
    parameters = _named_parameters(model)
    buffers = _named_buffers(model)
    tensors = [tensor for _, tensor in (*parameters, *buffers)]
    return {
        "devices": sorted({str(tensor.device) for tensor in tensors}),
        "dtypes": sorted({str(tensor.dtype) for tensor in tensors}),
        "training": bool(model.training),
        "module_training": {path: bool(module.training) for path, module in _named_modules(model)},
        "tensors": {
            path: {
                "kind": kind,
                "device": str(tensor.device),
                "dtype": str(tensor.dtype),
                "requires_grad": bool(tensor.requires_grad),
            }
            for kind, entries in (("parameter", parameters), ("buffer", buffers))
            for path, tensor in entries
        },
    }


def _occurrence_bindings(
    root: Any, model: nn.Module, architecture: Mapping[str, Any]
) -> dict[str, Any]:
    tree_nodes = root.serialise()
    preorder = _preorder_indices(architecture)
    if len(tree_nodes) != len(preorder):
        raise UnsupportedModuleError(
            "grammar reconstruction did not preserve occurrence cardinality"
        )
    tree_index = {id(node): occurrence for node, occurrence in zip(tree_nodes, preorder)}
    bindings: dict[str, dict[str, Any]] = {
        str(occurrence): {
            "logical_id": architecture["nodes"][occurrence]["id"],
            "module_paths": [],
            "parameter_paths": [],
            "buffer_paths": [],
        }
        for occurrence in preorder
    }

    def bind(node: Any, module: nn.Module, path: str) -> None:
        occurrence = tree_index[id(node)]
        record = bindings[str(occurrence)]
        record["module_paths"].append(path)
        record["parameter_paths"].extend(
            _join_path(path, parameter_path)
            for parameter_path, _ in module.named_parameters(recurse=True)
        )
        record["buffer_paths"].extend(
            _join_path(path, buffer_path) for buffer_path, _ in module.named_buffers(recurse=True)
        )
        for child, targets in _child_targets(node, module, path):
            for child_module, child_path in targets:
                bind(child, child_module, child_path)

    bind(root, model, "")
    for record in bindings.values():
        record["module_paths"].sort()
        record["parameter_paths"] = sorted(set(record["parameter_paths"]))
        record["buffer_paths"] = sorted(set(record["buffer_paths"]))
    return bindings


def _child_targets(
    node: Any, module: nn.Module, path: str
) -> list[tuple[Any, list[tuple[nn.Module, str]]]]:
    children = list(node.children)
    if not children:
        return []
    direct = list(_named_children(module))
    if len(direct) == len(children):
        return [
            (child, [(child_module, _join_path(path, name))])
            for child, (name, child_module) in zip(children, direct)
        ]
    if len(direct) == 1 and isinstance(direct[0][1], nn.ModuleList) and len(children) == 1:
        name, container = direct[0]
        return [
            (
                children[0],
                [
                    (child_module, _join_path(_join_path(path, name), item_name))
                    for item_name, child_module in container.named_children()
                ],
            )
        ]
    if len(direct) == 3 and isinstance(direct[1][1], nn.ModuleList):
        first_name, first = direct[0]
        list_name, container = direct[1]
        last_name, last = direct[2]
        expanded = [
            (child_module, _join_path(_join_path(path, list_name), item_name))
            for item_name, child_module in container.named_children()
        ]
        if len(children) == len(expanded) + 2:
            return [
                (children[0], [(first, _join_path(path, first_name))]),
                *[(child, [target]) for child, target in zip(children[1:-1], expanded)],
                (children[-1], [(last, _join_path(path, last_name))]),
            ]
        if len(children) == 3:
            return [
                (children[0], [(first, _join_path(path, first_name))]),
                (children[1], expanded),
                (children[2], [(last, _join_path(path, last_name))]),
            ]
    raise UnsupportedModuleError(
        f"grammar operation {node.operation.name!r} has no declarative module binding rule"
    )


def _validate_bindings(model: nn.Module, captured: CapturedArchitecture) -> None:
    bindings = data_mapping(captured.bindings, context="bindings")
    occurrences = bindings.get("occurrences")
    if not isinstance(occurrences, Mapping):
        raise StaleProvenanceError("retained provenance has no occurrence bindings")
    architecture = captured.architecture.to_dict()
    expected = {str(index) for index in range(len(architecture["nodes"]))}
    if set(occurrences) != expected:
        raise StaleProvenanceError("retained provenance occurrence bindings are incomplete")
    modules = {path for path, _ in _named_modules(model)}
    parameters = {path for path, _ in _named_parameters(model)}
    buffers = {path for path, _ in _named_buffers(model)}
    for index in expected:
        binding = occurrences[index]
        if not isinstance(binding, Mapping):
            raise StaleProvenanceError("retained provenance has malformed occurrence bindings")
        if binding.get("logical_id") != architecture["nodes"][int(index)]["id"]:
            raise StaleProvenanceError("retained provenance logical identities no longer match")
        module_paths = binding.get("module_paths", [])
        parameter_paths = binding.get("parameter_paths", [])
        buffer_paths = binding.get("buffer_paths", [])
        if any(path not in modules for path in module_paths):
            raise StaleProvenanceError("retained provenance module binding is stale")
        if any(path not in parameters for path in parameter_paths):
            raise StaleProvenanceError("retained provenance parameter binding is stale")
        if any(path not in buffers for path in buffer_paths):
            raise StaleProvenanceError("retained provenance buffer binding is stale")


def _assert_no_sharing(model: nn.Module) -> None:
    for noun, entries in (
        ("module", _named_modules(model)),
        ("parameter", _named_parameters(model)),
        ("buffer", _named_buffers(model)),
    ):
        paths: dict[int, list[str]] = defaultdict(list)
        for path, value in entries:
            paths[id(value)].append(path)
        shared = [sorted(value) for value in paths.values() if len(value) > 1]
        if shared:
            raise UnsupportedModuleError(
                f"{noun} sharing is not representable by the RCSWX occurrence tree: {shared[0]}"
            )
    storage_paths: dict[int, list[str]] = defaultdict(list)
    for path, tensor in (*_named_parameters(model), *_named_buffers(model)):
        try:
            # Storage identity, not data_ptr(): distinct empty/meta tensors
            # legitimately have the same null data pointer.
            storage = tensor.untyped_storage()._cdata
        except (RuntimeError, NotImplementedError) as error:
            raise UnsupportedModuleError(
                f"cannot verify tensor-storage ownership for {path!r}"
            ) from error
        storage_paths[storage].append(path)
    shared_storage = [paths for paths in storage_paths.values() if len(paths) > 1]
    if shared_storage:
        raise UnsupportedModuleError(
            "tensor storage sharing is not representable by the RCSWX occurrence tree: "
            f"{sorted(shared_storage[0])}"
        )


def _named_modules(model: nn.Module) -> list[tuple[str, nn.Module]]:
    return [
        (path, module)
        for path, module in model.named_modules(remove_duplicate=False)
        if not _metadata_path(path)
    ]


def _named_children(module: nn.Module) -> list[tuple[str, nn.Module]]:
    return [(name, child) for name, child in module.named_children() if name != _CARRIER_NAME]


def _named_parameters(model: nn.Module) -> list[tuple[str, torch.nn.Parameter]]:
    return [
        (path, parameter)
        for path, parameter in model.named_parameters(recurse=True, remove_duplicate=False)
        if not _metadata_path(path)
    ]


def _named_buffers(model: nn.Module) -> list[tuple[str, torch.Tensor]]:
    return [
        (path, buffer)
        for path, buffer in model.named_buffers(recurse=True, remove_duplicate=False)
        if not _metadata_path(path)
    ]


def _metadata_path(path: str) -> bool:
    return path == _CARRIER_NAME or path.startswith(f"{_CARRIER_NAME}.")


def _join_path(prefix: str, suffix: str) -> str:
    if not prefix:
        return suffix
    if not suffix:
        return prefix
    return f"{prefix}.{suffix}"


def _type_name(value: Any) -> str:
    typ = type(value)
    return f"{typ.__module__}.{typ.__qualname__}"


def _preorder_indices(architecture: Mapping[str, Any]) -> list[int]:
    nodes = architecture["nodes"]
    root = architecture["root"]
    ordered: list[int] = []

    def visit(index: int) -> None:
        ordered.append(index)
        for child in nodes[index]["children"]:
            visit(child)

    visit(root)
    return ordered
