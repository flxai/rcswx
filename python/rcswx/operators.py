"""Verified operator registry and allocation-free dependent-shape inference."""

import time
from dataclasses import dataclass

import torch
from torch import nn

from .errors import BudgetExceeded, UnsupportedConfiguration, UnsupportedEffect, UnsupportedOperator
from .policies import Limits
from .types import Operation, ParameterBinding, TensorSpec, checked_numel

_LEAVES = (nn.Identity, nn.Linear, nn.ReLU)
_HOOKS = (
    "_global_forward_pre_hooks",
    "_global_forward_hooks",
    "_global_backward_pre_hooks",
    "_global_backward_hooks",
    "_global_module_registration_hooks",
    "_global_parameter_registration_hooks",
    "_global_buffer_registration_hooks",
)


def check_global_hooks() -> None:
    for name in _HOOKS:
        if getattr(nn.modules.module, name, None):
            raise UnsupportedEffect("global module hooks are not supported")


def check_deadline(start: float, limits: Limits) -> None:
    elapsed = time.monotonic() - start
    if elapsed > limits.deadline_seconds:
        raise BudgetExceeded("deadline_seconds", limits.deadline_seconds, elapsed)


def describe(module: nn.Module) -> Operation:
    if type(module) is nn.Identity:
        return Operation("identity")
    if type(module) is nn.Linear:
        if type(module.in_features) is not int or module.in_features <= 0:
            raise UnsupportedConfiguration("Linear.in_features must be a positive dimension")
        return Operation(
            "linear", (("out_features", module.out_features), ("bias", module.bias is not None))
        )
    if type(module) is nn.ReLU:
        if module.inplace is not False:
            raise UnsupportedEffect("in-place ReLU is not supported")
        return Operation("relu")
    raise UnsupportedOperator(
        f"unregistered module {type(module).__module__}.{type(module).__qualname__}"
    )


@dataclass(frozen=True, slots=True)
class Realization:
    input_spec: TensorSpec
    output_spec: TensorSpec
    parameters: tuple[ParameterBinding, ...]
    buffers: tuple[ParameterBinding, ...] = ()


@dataclass(frozen=True, slots=True)
class Inference:
    output_spec: TensorSpec
    occurrences: tuple[Realization, ...]
    storage_bytes: int


def infer(
    operations: tuple[Operation, ...],
    input_spec: TensorSpec,
    *,
    limits: Limits,
    check_storage: bool = True,
) -> Inference:
    start = time.monotonic()
    if not operations:
        raise UnsupportedConfiguration("an edited sequence cannot be empty")
    if len(operations) > limits.max_nodes:
        raise BudgetExceeded("max_nodes", limits.max_nodes, len(operations))
    if len(input_spec.shape) > limits.max_rank:
        raise BudgetExceeded("max_rank", limits.max_rank, len(input_spec.shape))
    current = input_spec
    occurrences = []
    storage_bytes = 0
    for index, operation in enumerate(operations):
        if index % 1024 == 0:
            check_deadline(start, limits)
        parameters = ()
        output = current
        if operation.kind == "linear":
            if not current.shape:
                raise UnsupportedConfiguration("Linear requires at least one input dimension")
            width = operation.get("out_features")
            output = TensorSpec((*current.shape[:-1], width))
            parameters = (ParameterBinding("weight", (width, current.shape[-1]), "float32"),)
            if operation.get("bias"):
                parameters += (ParameterBinding("bias", (width,), "float32"),)
        elif operation.kind not in ("identity", "relu"):
            raise UnsupportedOperator(f"{operation.kind} has not passed operator conformance")
        for parameter in parameters:
            storage_bytes += 4 * checked_numel(parameter.shape)
        if storage_bytes > (1 << 63) - 1:
            raise BudgetExceeded("tensor_bytes", (1 << 63) - 1, 1 << 63)
        if check_storage and storage_bytes > limits.max_child_bytes:
            raise BudgetExceeded("max_child_bytes", limits.max_child_bytes, storage_bytes)
        occurrences.append(Realization(current, output, parameters))
        current = output
    check_deadline(start, limits)
    return Inference(current, tuple(occurrences), storage_bytes)


def validate_state(module: nn.Module, operation: Operation, realization: Realization) -> None:
    allowed = {"weight", "bias"} if operation.kind == "linear" else set()
    if set(module._parameters) - allowed or module._buffers:
        raise UnsupportedConfiguration(
            "registered operator has unrecognized parameter/buffer roles"
        )
    actual = {name: value for name, value in module._parameters.items() if value is not None}
    if set(actual) != {parameter.name for parameter in realization.parameters}:
        raise UnsupportedConfiguration("registered parameter roles disagree with the operator")
    if operation.kind == "linear" and module.in_features != realization.input_spec.shape[-1]:
        raise UnsupportedConfiguration(
            "captured Linear.in_features disagrees with inferred input width"
        )
    for parameter in realization.parameters:
        tensor = actual[parameter.name]
        if (
            type(tensor) is not nn.Parameter
            or tuple(tensor.shape) != parameter.shape
            or tensor.dtype is not torch.float32
        ):
            raise UnsupportedConfiguration(
                f"realized {parameter.name} disagrees with inferred role/shape/dtype"
            )
