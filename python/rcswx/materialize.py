"""Owned module construction. Statistical state is never inherited in production."""

import math
import time

import torch
from torch import nn

from .errors import BudgetExceeded, UnsupportedConfiguration, UnsupportedOperator
from .operators import Inference, check_deadline, check_global_hooks, infer
from .policies import Limits
from .types import Architecture, Operation, TensorSpec


def _leaf(operation: Operation, input_spec: TensorSpec) -> nn.Module:
    if operation.kind == "identity":
        return nn.Identity()
    if operation.kind == "relu":
        return nn.ReLU(inplace=False)
    if operation.kind == "linear":
        return nn.Linear(
            input_spec.shape[-1],
            operation.get("out_features"),
            bias=operation.get("bias"),
            device="meta",
            dtype=torch.float32,
        )
    raise UnsupportedOperator(f"{operation.kind} has no verified materialization route")


def _donor_parameter(operation, side, address, role, donors, donor_architectures):
    if donor_architectures[side].operations[address].kind != operation.kind:
        return None, "semantic_role_mismatch"
    binding = donor_architectures[side].bindings[address]
    if binding.module_path is None:
        return None, "missing_parameter"
    donor = donors[side].get_submodule(binding.module_path)
    parameter = donor._parameters.get(role)
    if parameter is None:
        return None, "missing_parameter"
    return parameter, "compatible"


def materialize(
    operations: tuple[Operation, ...],
    input_spec: TensorSpec,
    *,
    inference: Inference,
    training: bool,
    init: str,
    initialization_seed: int,
    inherit: bool,
    donors: tuple[nn.Module, nn.Module] | None,
    donor_architectures: tuple[Architecture, Architecture] | None,
    provenance: tuple[tuple[int, int], ...],
    limits: Limits,
) -> tuple[nn.Module, tuple[dict[str, object], ...]]:
    """Realize prevalidated dimensions and copy/init each actual tensor once."""
    start = time.monotonic()
    if init not in ("pytorch", "zeros"):
        raise ValueError("init must be 'pytorch' or 'zeros'")
    if inherit and (donors is None or donor_architectures is None):
        raise ValueError("inheritance requires two live donor modules")
    if inference.storage_bytes > limits.max_child_bytes:
        raise BudgetExceeded("max_child_bytes", limits.max_child_bytes, inference.storage_bytes)
    check_global_hooks()
    check_deadline(start, limits)
    root, decisions, generator = nn.Sequential(), [], None
    with torch.inference_mode(False), torch.no_grad():
        for index, (operation, realization, origin) in enumerate(
            zip(operations, inference.occurrences, provenance, strict=True)
        ):
            check_deadline(start, limits)
            leaf = _leaf(operation, realization.input_spec)
            side, address = origin
            for parameter in realization.parameters:
                chosen, reason = None, "inherit_disabled"
                if inherit:
                    chosen, reason = _donor_parameter(
                        operation, side, address, parameter.name, donors, donor_architectures
                    )
                    if chosen is not None:
                        if tuple(chosen.shape) != parameter.shape:
                            chosen, reason = None, "shape_mismatch"
                        elif (
                            chosen.dtype is not torch.float32
                            or chosen.device.type != "cpu"
                            or chosen.layout is not torch.strided
                        ):
                            chosen, reason = None, "dtype_or_layout_mismatch"
                data = torch.empty(parameter.shape, dtype=torch.float32, device="cpu")
                if chosen is not None:
                    data.copy_(chosen)
                    requires_grad, action = chosen.requires_grad, "copied"
                else:
                    requires_grad, action = True, "initialized"
                    if init == "zeros":
                        data.zero_()
                    else:
                        if generator is None:
                            generator = torch.Generator(device="cpu")
                            generator.manual_seed(initialization_seed)
                        if parameter.name == "weight":
                            nn.init.kaiming_uniform_(data, a=math.sqrt(5), generator=generator)
                        else:
                            bound = 1 / math.sqrt(realization.input_spec.shape[-1])
                            data.uniform_(-bound, bound, generator=generator)
                setattr(leaf, parameter.name, nn.Parameter(data, requires_grad=requires_grad))
                decisions.append(
                    {
                        "child_address": index,
                        "role": parameter.name,
                        "action": action,
                        "reason": reason,
                        "donor_side": side,
                        "donor_address": address,
                    }
                )
            root.add_module(str(index), leaf)
    root.train(training)
    check_deadline(start, limits)
    return root, tuple(decisions)


def copied_state(
    architecture: Architecture, donor: nn.Module, *, limits: Limits | None = None
) -> nn.Module:
    """Private numerical conformance route; not a production inheritance policy."""
    limits = Limits() if limits is None else limits
    inference = infer(architecture.operations, architecture.input_spec, limits=limits)
    model, decisions = materialize(
        architecture.operations,
        architecture.input_spec,
        inference=inference,
        training=architecture.training,
        init="zeros",
        initialization_seed=0,
        inherit=True,
        donors=(donor, donor),
        donor_architectures=(architecture, architecture),
        provenance=tuple((0, i) for i in range(len(architecture.operations))),
        limits=limits,
    )
    if any(decision["action"] != "copied" for decision in decisions):
        raise UnsupportedConfiguration(
            "copied-state conformance requires compatible complete state"
        )
    return model
