"""Verified operator registry and allocation-free dependent-shape inference."""

import time
from dataclasses import dataclass

import torch
from torch import nn

from .errors import BudgetExceeded, UnsupportedConfiguration, UnsupportedEffect, UnsupportedOperator
from .policies import Limits
from .types import Operation, ParameterBinding, TensorSpec, checked_numel

_LEAVES = (nn.Identity, nn.Linear, nn.ReLU, nn.Dropout, nn.BatchNorm1d)
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
    if type(module) is nn.Dropout:
        if module.inplace is not False:
            raise UnsupportedEffect("in-place Dropout is not supported")
        return Operation("dropout", (("p", module.p),))
    if type(module) is nn.BatchNorm1d:
        if type(module.num_features) is not int or module.num_features <= 0:
            raise UnsupportedConfiguration("BatchNorm1d.num_features must be positive")
        return Operation(
            "batch_norm1d",
            (
                ("affine", module.affine),
                ("eps", module.eps),
                ("momentum", module.momentum),
                ("track_running_stats", module.track_running_stats),
            ),
        )
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
        parameters, buffers = (), ()
        output = current
        if operation.kind == "linear":
            if not current.shape:
                raise UnsupportedConfiguration("Linear requires at least one input dimension")
            width = operation.get("out_features")
            output = TensorSpec((*current.shape[:-1], width))
            parameters = (ParameterBinding("weight", (width, current.shape[-1]), "float32"),)
            if operation.get("bias"):
                parameters += (ParameterBinding("bias", (width,), "float32"),)
        elif operation.kind == "batch_norm1d":
            if len(current.shape) not in (2, 3):
                raise UnsupportedConfiguration("BatchNorm1d requires rank two or three")
            channels = current.shape[1]
            samples = current.shape[0] * (current.shape[2] if len(current.shape) == 3 else 1)
            if samples <= 1:
                raise UnsupportedConfiguration("BatchNorm1d requires multiple samples in training")
            if operation.get("affine"):
                parameters = (
                    ParameterBinding("weight", (channels,), "float32"),
                    ParameterBinding("bias", (channels,), "float32"),
                )
            if operation.get("track_running_stats"):
                buffers = (
                    ParameterBinding("running_mean", (channels,), "float32"),
                    ParameterBinding("running_var", (channels,), "float32"),
                    ParameterBinding("num_batches_tracked", (), "int64"),
                )
        elif operation.kind not in ("identity", "relu", "dropout"):
            raise UnsupportedOperator(f"{operation.kind} has not passed operator conformance")
        for tensor in (*parameters, *buffers):
            storage_bytes += (8 if tensor.dtype == "int64" else 4) * checked_numel(tensor.shape)
        if storage_bytes > (1 << 63) - 1:
            raise BudgetExceeded("tensor_bytes", (1 << 63) - 1, 1 << 63)
        if check_storage and storage_bytes > limits.max_child_bytes:
            raise BudgetExceeded("max_child_bytes", limits.max_child_bytes, storage_bytes)
        occurrences.append(Realization(current, output, parameters, buffers))
        current = output
    check_deadline(start, limits)
    return Inference(current, tuple(occurrences), storage_bytes)


def validate_state(module: nn.Module, operation: Operation, realization: Realization) -> None:
    parameter_names = {"weight", "bias"} if operation.kind in ("linear", "batch_norm1d") else set()
    buffer_names = (
        {"running_mean", "running_var", "num_batches_tracked"}
        if operation.kind == "batch_norm1d"
        else set()
    )
    if (
        set(module._parameters) - parameter_names
        or set(module._buffers) - buffer_names
        or module._non_persistent_buffers_set
    ):
        raise UnsupportedConfiguration("registered operator has unrecognized state roles")
    if operation.kind == "linear" and module.in_features != realization.input_spec.shape[-1]:
        raise UnsupportedConfiguration(
            "captured Linear.in_features disagrees with inferred input width"
        )
    if operation.kind == "batch_norm1d" and module.num_features != realization.input_spec.shape[1]:
        raise UnsupportedConfiguration(
            "captured BatchNorm1d.num_features disagrees with inferred channels"
        )
    for registered, expected, tensor_type in (
        (module._parameters, realization.parameters, nn.Parameter),
        (module._buffers, realization.buffers, torch.Tensor),
    ):
        actual = {name: value for name, value in registered.items() if value is not None}
        if set(actual) != {tensor.name for tensor in expected}:
            raise UnsupportedConfiguration("registered state roles disagree with the operator")
        for binding in expected:
            tensor = actual[binding.name]
            if (
                type(tensor) is not tensor_type
                or tuple(tensor.shape) != binding.shape
                or tensor.dtype is not getattr(torch, binding.dtype)
                or tensor_type is torch.Tensor
                and tensor.requires_grad
            ):
                raise UnsupportedConfiguration(
                    f"realized {binding.name} disagrees with its role/shape/dtype"
                )
