"""Disposable model state for the explicit graph-import workflow."""

from collections.abc import Iterable
from contextlib import contextmanager
from copy import deepcopy

import torch
from torch import nn

from .types import UnsupportedModuleError


def isolated_model(model: nn.Module, *, meta: bool = False) -> nn.Module:
    """Clone structure, replacing reachable tensor storage before meta tracing."""
    memo = {}
    if meta:
        pending = [model]
        visited = set()
        while pending:
            value = pending.pop()
            identifier = id(value)
            if identifier in visited:
                continue
            visited.add(identifier)
            if isinstance(value, torch.Tensor):
                replacement = torch.empty_like(value, device="meta")
                if isinstance(value, nn.Parameter):
                    replacement = nn.Parameter(replacement, requires_grad=value.requires_grad)
                else:
                    replacement.requires_grad_(value.requires_grad)
                memo[identifier] = replacement
            elif isinstance(value, nn.Module):
                pending.extend(value.__dict__.values())
            elif isinstance(value, dict):
                pending.extend(value.values())
            elif isinstance(value, (list, tuple, set, frozenset)):
                pending.extend(value)
    try:
        isolated = deepcopy(model, memo)
    except Exception as error:
        raise UnsupportedModuleError(f"isolation: cannot copy source model: {error}") from error
    if isolated is model:
        raise UnsupportedModuleError("isolation: model copy returned the source instance")
    source_objects = {id(module) for module in model.modules()}
    if any(id(module) in source_objects for module in isolated.modules()):
        raise UnsupportedModuleError("isolation: copied model retains source modules")
    source_tensors = {id(tensor) for tensor in (*model.parameters(), *model.buffers())}
    if any(
        id(tensor) in source_tensors for tensor in (*isolated.parameters(), *isolated.buffers())
    ):
        raise UnsupportedModuleError("isolation: copied model retains source tensors")
    source_storage = {
        tensor.untyped_storage()._cdata for tensor in (*model.parameters(), *model.buffers())
    }
    for tensor in (*isolated.parameters(), *isolated.buffers()):
        if tensor.untyped_storage()._cdata in source_storage:
            raise UnsupportedModuleError("isolation: copied tensor retains source storage")
        if meta and tensor.device.type != "meta":
            raise UnsupportedModuleError("isolation: trace copy contains materialized state")
    return isolated


def isolated_inputs(inputs: tuple[torch.Tensor, ...]) -> tuple[torch.Tensor, ...]:
    return tuple(value.detach().clone() for value in inputs)


@contextmanager
def preserve_rng(tensors: Iterable[torch.Tensor]):
    """Preserve CPU and the CUDA generators actually used by this workflow."""
    devices = set()
    for tensor in tensors:
        if tensor.device.type == "cuda":
            devices.add(tensor.device.index)
        elif tensor.device.type not in {"cpu", "meta"}:
            raise UnsupportedModuleError(
                f"validation: RNG isolation is unsupported on {tensor.device.type!r}"
            )
    with torch.random.fork_rng(devices=sorted(devices)):
        yield
