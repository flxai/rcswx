"""Trusted, side-effect-free Python capture; not a sandbox for arbitrary forwards."""

import time

import torch
from torch import fx, nn

from .errors import (
    BudgetExceeded,
    UnsupportedArchitecture,
    UnsupportedConfiguration,
    UnsupportedEffect,
    UnsupportedOperator,
    UnsupportedSharing,
)
from .operators import _LEAVES, check_deadline, check_global_hooks, describe, infer, validate_state
from .policies import Limits
from .types import Architecture, Binding, Operation, TensorSpec

_MODULE_HOOKS = (
    "_forward_pre_hooks",
    "_forward_hooks",
    "_backward_pre_hooks",
    "_backward_hooks",
    "_state_dict_hooks",
    "_state_dict_pre_hooks",
    "_load_state_dict_pre_hooks",
    "_load_state_dict_post_hooks",
)


def _contract(example_inputs, input_spec, limits: Limits) -> TensorSpec:
    if (example_inputs is None) == (input_spec is None):
        raise ValueError("provide exactly one of example_inputs or input_spec")
    if input_spec is not None:
        if type(input_spec) is not TensorSpec:
            raise TypeError("input_spec must be a TensorSpec")
        result = input_spec
    else:
        if type(example_inputs) is not tuple or len(example_inputs) != 1:
            raise UnsupportedConfiguration("exactly one positional tensor input is supported")
        tensor = example_inputs[0]
        if type(tensor) not in (torch.Tensor, nn.Parameter):
            raise UnsupportedConfiguration("input must be an ordinary dense tensor")
        if (
            tensor.device.type != "cpu"
            or tensor.dtype is not torch.float32
            or tensor.layout is not torch.strided
            or not tensor.is_contiguous()
        ):
            raise UnsupportedConfiguration("input must be contiguous dense CPU float32")
        result = TensorSpec(tuple(tensor.shape))
    if len(result.shape) > limits.max_rank:
        raise BudgetExceeded("max_rank", limits.max_rank, len(result.shape))
    return result


def _preflight(root: nn.Module, limits: Limits, start: float) -> list[tuple[str, nn.Module]]:
    check_global_hooks()
    modules = []
    seen_modules, seen_tensors, seen_storages = set(), set(), set()
    stack = [("", root, 1)]
    module_limit = limits.max_nodes * (limits.max_depth + 1)
    while stack:
        path, module, depth = stack.pop()
        if len(modules) % 1024 == 0:
            check_deadline(start, limits)
        if depth > limits.max_depth:
            raise BudgetExceeded("max_depth", limits.max_depth, depth)
        if len(modules) >= module_limit:
            raise BudgetExceeded("capture_modules", module_limit, len(modules) + 1)
        if id(module) in seen_modules:
            raise UnsupportedSharing(
                "module aliases, repeated registration and cycles are unsupported"
            )
        seen_modules.add(id(module))
        if type(module.training) is not bool or module.training != root.training:
            raise UnsupportedConfiguration(
                "every parent submodule must have one uniform training mode"
            )
        if (
            "forward" in module.__dict__
            or getattr(module, "_compiled_call_impl", None) is not None
            or type(module).__call__ is not nn.Module.__call__
            or type(module)._call_impl is not nn.Module._call_impl
        ):
            raise UnsupportedEffect("overridden or compiled module call behavior is unsupported")
        if any(getattr(module, name, None) for name in _MODULE_HOOKS):
            raise UnsupportedEffect("module hooks are unsupported")
        if type(module) not in _LEAVES and (module._parameters or module._buffers):
            raise UnsupportedOperator("only registered leaves may own parameters or buffers")
        if any(isinstance(value, torch.Tensor) for value in module.__dict__.values()):
            raise UnsupportedEffect("unregistered tensor-valued attributes are unsupported")
        if any(
            base.__dict__.get("__slots__")
            for base in type(module).__mro__
            if base not in (nn.Module, object)
        ):
            raise UnsupportedOperator("slotted wrapper state has no verified capture route")
        for tensor in (*module._parameters.values(), *module._buffers.values()):
            if tensor is None:
                continue
            if (
                type(tensor) not in (torch.Tensor, nn.Parameter)
                or tensor.device.type != "cpu"
                or tensor.layout is not torch.strided
                or tensor.dtype not in (torch.float32, torch.int64)
                or not tensor.is_contiguous()
            ):
                raise UnsupportedConfiguration(
                    "state must be ordinary contiguous CPU tensors with registered dtypes"
                )
            if tensor.is_inference():
                raise UnsupportedConfiguration(
                    "inference-mode state cannot support normal training"
                )
            if id(tensor) in seen_tensors:
                raise UnsupportedSharing("a parameter or buffer has multiple registrations")
            seen_tensors.add(id(tensor))
            storage = tensor.untyped_storage().data_ptr()
            if tensor.numel() and storage in seen_storages:
                raise UnsupportedSharing("state tensors share storage (including disjoint views)")
            if tensor.numel():
                seen_storages.add(storage)
            if getattr(tensor, "_backward_hooks", None) or getattr(
                tensor, "_post_accumulate_grad_hooks", None
            ):
                raise UnsupportedEffect("tensor gradient hooks are unsupported")
        modules.append((path, module))
        for name, child in reversed(tuple(module._modules.items())):
            if child is not None:
                child_path = f"{path}.{name}" if path else name
                if len(child_path.encode("utf-8")) > 4096:
                    raise BudgetExceeded("binding_path_bytes", 4096, 4097)
                stack.append((child_path, child, depth + 1))
    check_deadline(start, limits)
    return modules


def _capture_copy(modules: list[tuple[str, nn.Module]], training: bool) -> nn.Module:
    # Copy only module dictionaries/registrations, NEVER tensor storage or constructors.
    # Both uniform modes are traced without changing even one live donor flag.
    copies = {}
    for _, module in reversed(modules):
        clone = object.__new__(type(module))
        clone.__dict__ = module.__dict__.copy()
        clone._parameters = module._parameters.copy()
        clone._buffers = module._buffers.copy()
        clone._modules = {
            name: copies[id(child)] if child is not None else None
            for name, child in module._modules.items()
        }
        clone.training = training
        copies[id(module)] = clone
    return copies[id(modules[0][1])]


class _Tracer(fx.Tracer):
    def __init__(self, limits: Limits, start: float):
        super().__init__()
        self.limits, self.start, self.created = limits, start, 0

    def is_leaf_module(self, module: nn.Module, qualified_name: str) -> bool:
        return type(module) in _LEAVES or super().is_leaf_module(module, qualified_name)

    def create_node(self, *args, **kwargs):
        self.created += 1
        if self.created > self.limits.max_nodes + 2:
            raise BudgetExceeded("capture_nodes", self.limits.max_nodes + 2, self.created)
        if self.created % 1024 == 1:
            check_deadline(self.start, self.limits)
        return super().create_node(*args, **kwargs)

    def path_of_module(self, module: nn.Module) -> str:
        try:
            return super().path_of_module(module)
        except NameError:
            raise UnsupportedSharing("called modules must have unique registered paths") from None


def _trace_chain(
    root: nn.Module, limits: Limits, start: float
) -> tuple[tuple[Operation, ...], tuple[str | None, ...]]:
    try:
        graph = _Tracer(limits, start).trace(root)
    except fx.proxy.TraceError as exc:
        raise UnsupportedEffect("data-dependent Python control flow is unsupported") from exc
    operations, paths, invoked = [], [], set()
    current = None
    output_seen = False
    for node in graph.nodes:
        if node.op == "placeholder":
            if current is not None or node.args or node.kwargs:
                raise UnsupportedConfiguration(
                    "one positional tensor input without defaults is required"
                )
            current = node
        elif node.op == "call_module":
            if current is None or node.args != (current,) or node.kwargs or len(current.users) != 1:
                raise UnsupportedArchitecture(
                    "only single-chain dataflow is admitted; structured regions are not flattened"
                )
            module = root.get_submodule(node.target)
            if id(module) in invoked:
                raise UnsupportedSharing("repeated calls of one registered module are unsupported")
            invoked.add(id(module))
            operations.append(describe(module))
            paths.append(node.target)
            current = node
        elif node.op == "output":
            if current is None or node.args != (current,) or len(current.users) != 1:
                raise UnsupportedArchitecture("one tensor output from a complete chain is required")
            output_seen = True
        else:
            raise UnsupportedOperator(f"unregistered FX operation: {node.op}")
    if not output_seen:
        raise UnsupportedArchitecture("captured graph has no output")
    if not operations:
        return (Operation("identity"),), (None,)
    return tuple(operations), tuple(paths)


def prepare(
    model: nn.Module,
    *,
    example_inputs: tuple[torch.Tensor, ...] | None = None,
    input_spec: TensorSpec | None = None,
    limits: Limits | None = None,
) -> Architecture:
    """Capture a trusted side-effect-free wrapper; numerical forwards are not run.

    Unsupported arbitrary Python effects cannot be rolled back. Registered leaves
    are inspected, and wrapper graphs/bindings must agree in train and eval mode.
    """
    start = time.monotonic()
    limits = Limits() if limits is None else limits
    if type(limits) is not Limits:
        raise TypeError("limits must be Limits")
    if not isinstance(model, nn.Module):
        raise TypeError("prepare expects a torch.nn.Module")
    contract = _contract(example_inputs, input_spec, limits)
    modules = _preflight(model, limits, start)
    if type(model) in _LEAVES:
        operations, paths = (describe(model),), ("",)
    else:
        operations, paths = _trace_chain(_capture_copy(modules, True), limits, start)
        other = _trace_chain(_capture_copy(modules, False), limits, start)
        if (operations, paths) != other:
            raise UnsupportedEffect("wrapper topology or donor bindings depend on training mode")
    inference = infer(operations, contract, limits=limits, check_storage=False)
    bindings = []
    for operation, path, realization in zip(operations, paths, inference.occurrences, strict=True):
        if path is not None:
            validate_state(model.get_submodule(path), operation, realization)
        bindings.append(Binding(path, realization.parameters, realization.buffers))
    result = Architecture(
        operations, contract, inference.output_spec, tuple(bindings), model.training, limits
    )
    check_deadline(start, limits)
    return result
