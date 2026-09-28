"""Lower an isolated FX graph to the installed einspace v1 vocabulary."""

import operator
from dataclasses import dataclass, field
from typing import NoReturn

import torch
import torch.nn.functional as F
from torch import nn
from torch.fx import Node
from torch.fx.passes.shape_prop import ShapeProp

from .grammar_backends import Lowering, StateBinding
from .types import UnsupportedModuleError


@dataclass
class _Tree:
    name: str
    children: tuple = ()
    source: str | None = None
    parameters: dict[str, str] = field(default_factory=dict)
    buffers: dict[str, str] = field(default_factory=dict)
    reshape_weight: bool = False


def _computation(name, **kwargs):
    return _Tree("computation", (_Tree(name, **kwargs),))


def _sequence(first, second):
    return second if first is None else _Tree("sequential", (first, second))


def _error(node, reason) -> NoReturn:
    raise UnsupportedModuleError(f"lowering: FX node {node.name!r} ({node.target}): {reason}")


def _argument(node, position, name, default=None):
    return node.args[position] if len(node.args) > position else node.kwargs.get(name, default)


def _shape(node):
    metadata = node.meta.get("tensor_meta")
    if metadata is None or not hasattr(metadata, "shape"):
        _error(node, "expected one tensor value")
    return tuple(metadata.shape)


def _path(owner, member):
    return f"{owner}.{member}" if owner else member


class _Lowerer:
    def __init__(self, graph, input_spec):
        from rcswx.grammars import einspace

        from .build import _operation_lookup

        self.graph = graph
        self.input_spec = dict(input_spec)
        self.operations = _operation_lookup(einspace.grammar)
        self.order = {node: index for index, node in enumerate(graph.graph.nodes)}
        self.descriptions = {}
        self.module_calls = set()
        self.parameter_uses = set()

    def state(self, node, module, path):
        parameters = {name: _path(path, name) for name, _ in module.named_parameters(recurse=False)}
        buffers = {name: _path(path, name) for name, _ in module.named_buffers(recurse=False)}
        return {"source": path, "parameters": parameters, "buffers": buffers}

    def primitive(self, node, name, **state):
        if name not in self.operations:
            _error(node, f"operation {name!r} is unavailable in einspace v1")
        return _computation(name, **state)

    def linear(self, node, width, bias, state):
        if not bias:
            _error(node, "einspace linear requires a bias parameter")
        return self.primitive(node, f"linear({width})", **state)

    def convolution(self, node, module, path):
        kernel, stride, padding = module.kernel_size, module.stride, module.padding
        if (
            isinstance(padding, str)
            or any(len(pair) != 2 or pair[0] != pair[1] for pair in (kernel, stride, padding))
            or module.dilation != (1, 1)
            or module.groups != 1
            or module.padding_mode != "zeros"
        ):
            _error(
                node, "Conv2d requires square installed routing, dilation=1, groups=1, zero padding"
            )
        route = f"im2col({kernel[0]},{stride[0]},{padding[0]})"
        if route not in self.operations:
            _error(node, f"routing {route!r} is unavailable in einspace v1")
        state = self.state(node, module, path)
        state["reshape_weight"] = True
        linear = self.linear(node, module.out_channels, module.bias is not None, state)
        return _Tree("routing", (_Tree(route), linear, _Tree("col2im")))

    def normalization(self, node, module, path, input_node):
        shape = _shape(input_node)
        if type(module) is nn.LayerNorm:
            valid = (
                len(shape) == 2
                and tuple(module.normalized_shape) == (shape[-1],)
                and module.eps == 1e-5
                and module.elementwise_affine
                and module.bias is not None
            )
        else:
            rank = 3 if type(module) is nn.BatchNorm1d else 4
            valid = (
                len(shape) == rank
                and module.num_features == shape[1]
                and module.eps == 1e-5
                and module.momentum == 0.1
                and module.affine
                and module.track_running_stats
            )
        if not valid:
            _error(node, "normalization configuration/rank does not match einspace norm")
        return self.primitive(node, "norm", **self.state(node, module, path))

    def describe(self, node) -> tuple[str, tuple[Node, ...], _Tree | str]:
        if node in self.descriptions:
            return self.descriptions[node]
        if node.op == "call_module":
            if node.target in self.module_calls:
                _error(node, "repeated module invocation is not supported")
            self.module_calls.add(node.target)
            module = self.graph.get_submodule(node.target)
            first = _argument(node, 0, "input")
            if not isinstance(first, Node) or len(node.args) > 1 or set(node.kwargs) - {"input"}:
                _error(node, "expected one tensor argument")
            state = self.state(node, module, node.target)
            if type(module) is nn.Linear:
                tree = self.linear(node, module.out_features, module.bias is not None, state)
            elif type(module) is nn.Conv2d:
                tree = self.convolution(node, module, node.target)
            elif type(module) in (nn.BatchNorm1d, nn.BatchNorm2d, nn.LayerNorm):
                tree = self.normalization(node, module, node.target, first)
            elif type(module) is nn.ReLU and not module.inplace:
                tree = self.primitive(node, "relu", **state)
            elif type(module) is nn.Identity:
                tree = self.primitive(node, "identity", **state)
            elif type(module) is nn.Softmax and module.dim in (-1, len(_shape(first)) - 1):
                tree = self.primitive(node, "softmax", **state)
            else:
                _error(node, "unsupported module type or constructor configuration")
            result = ("unary", (first,), tree)
        elif node.op in ("call_function", "call_method"):
            result = self.describe_call(node)
        else:
            _error(node, "unsupported graph value or dependency")
        self.descriptions[node] = result
        return result

    def describe_call(self, node):
        target = node.target
        first = _argument(node, 0, "input")
        if target in (operator.add, torch.add, "add"):
            second = _argument(node, 1, "other")
            if (
                not isinstance(first, Node)
                or not isinstance(second, Node)
                or _argument(node, 2, "alpha", 1) != 1
                or set(node.kwargs) - {"input", "other", "alpha"}
                or len(node.args) > 2
            ):
                _error(node, "addition requires two tensors, alpha=1, and no out/inplace option")
            if _shape(first) != _shape(second):
                _error(node, "einspace addition requires equal shapes without broadcasting")
            return ("join", (first, second), "add(2)")
        if target is torch.cat:
            operands = _argument(node, 0, "tensors")
            if (
                not isinstance(operands, (tuple, list))
                or len(operands) != 2
                or not all(isinstance(n, Node) for n in operands)
            ):
                _error(node, "concatenation requires exactly two tensor branches")
            dim = _argument(node, 1, "dim", 0)
            if isinstance(dim, bool) or not isinstance(dim, int):
                _error(node, "concatenation dimension must be a constant integer")
            dim %= len(_shape(operands[0]))
            name = f"cat(2,{dim})"
            if (
                name not in self.operations
                or len(node.args) > 2
                or set(node.kwargs) - {"tensors", "dim"}
            ):
                _error(node, "concatenation dimension/options are not in einspace v1")
            return ("join", tuple(operands), name)
        if not isinstance(first, Node):
            _error(node, "expected a tensor input")
        if target in (torch.relu, F.relu, "relu"):
            allowed = {"input", "inplace"} if target is F.relu else {"input"}
            if (
                _argument(node, 1, "inplace", False) is not False
                or len(node.args) > (2 if target is F.relu else 1)
                or set(node.kwargs) - allowed
            ):
                _error(node, "only non-inplace ReLU is supported")
            tree = self.primitive(node, "relu")
        elif target in (torch.softmax, F.softmax, "softmax"):
            dim = _argument(node, 1, "dim")
            dtype = node.kwargs.get("dtype")
            if (
                dim not in (-1, len(_shape(first)) - 1)
                or dtype is not None
                or len(node.args) > 2
                or set(node.kwargs) - {"input", "dim", "dtype", "_stacklevel"}
            ):
                _error(node, "softmax must use the last axis without dtype conversion")
            tree = self.primitive(node, "softmax")
        elif target is F.linear:
            weight = _argument(node, 1, "weight")
            bias = _argument(node, 2, "bias")
            if len(node.args) > 3 or set(node.kwargs) - {"input", "weight", "bias"}:
                _error(node, "unsupported functional linear arguments")
            parameters = {}
            for role, value in (("weight", weight), ("bias", bias)):
                if not isinstance(value, Node) or value.op != "get_attr":
                    _error(node, "functional linear requires registered weight and bias parameters")
                try:
                    parameter = self.graph.get_parameter(value.target)
                except AttributeError:
                    _error(node, f"{role} is not a registered parameter")
                parameters[role] = value.target
                if role == "weight":
                    if parameter.ndim != 2:
                        _error(node, "linear weight must have rank two")
            width = self.graph.get_parameter(parameters["weight"]).shape[0]
            tree = self.linear(node, width, True, {"parameters": parameters})
        else:
            _error(node, "unsupported functional/method operation")
        return ("unary", (first,), tree)

    def ancestors(self, node, boundary):
        result = set()
        pending = [node]
        while pending:
            current = pending.pop()
            if current in result:
                continue
            result.add(current)
            if current is boundary:
                continue
            if current.op == "placeholder":
                _error(node, "dependency crosses the enclosing fork boundary")
            pending.extend(self.describe(current)[1])
        return result

    def region(self, node, boundary):
        if node is boundary:
            return None, set()
        if node.op == "placeholder":
            _error(node, "dependency crosses the enclosing fork boundary")
        kind, inputs, operation = self.describe(node)
        if kind == "unary":
            assert isinstance(operation, _Tree)
            prefix, used = self.region(inputs[0], boundary)
            return _sequence(prefix, operation), used | {node}
        shared = self.ancestors(inputs[0], boundary) & self.ancestors(inputs[1], boundary)
        common = max(shared, key=self.order.__getitem__)
        prefix, prefix_used = self.region(common, boundary)
        left, left_used = self.region(inputs[0], common)
        right, right_used = self.region(inputs[1], common)
        if left_used & right_used or prefix_used & (left_used | right_used):
            _error(node, "branches share internal computation outside a nested fork/join")
        assert isinstance(operation, str)
        tree = _Tree(
            "branching(2)",
            (
                _Tree("clone(2)"),
                left or _computation("identity"),
                right or _computation("identity"),
                _Tree(operation),
            ),
        )
        return _sequence(prefix, tree), prefix_used | left_used | right_used | {node}

    def finish(self):
        from rcswx.portable import Architecture

        placeholders = [n for n in self.graph.graph.nodes if n.op == "placeholder"]
        outputs = [n for n in self.graph.graph.nodes if n.op == "output"]
        if len(placeholders) != 1 or len(outputs) != 1 or not isinstance(outputs[0].args[0], Node):
            raise UnsupportedModuleError(
                "lowering: requires one tensor input and one tensor output"
            )
        input_node = placeholders[0]
        tree, used = self.region(outputs[0].args[0], input_node)
        calls = {n for n in self.graph.graph.nodes if n.op.startswith("call_")}
        if calls != used:
            missing = sorted(n.name for n in calls - used)
            raise UnsupportedModuleError(f"lowering: unrepresented computation: {missing}")
        tree = tree or _computation("identity")
        # einspace's network production excludes a bare computation. A
        # sequential identity embedding preserves a singleton's exact function.
        if tree.name == "computation":
            tree = _Tree("sequential", (tree, _computation("identity")))
        records, bindings, states = [], {}, []
        pending: list[tuple[_Tree, int | None]] = [(tree, None)]
        while pending:
            current, parent = pending.pop()
            index = len(records)
            if parent is not None:
                records[parent]["children"].append(index)
            records.append(
                {
                    "id": str(index),
                    "name": current.name,
                    "children": [],
                    "parameters": None,
                    "provenance": None,
                }
            )
            bindings[str(index)] = {
                "module_paths": [] if current.source is None else [current.source],
                "virtual": current.source is None,
            }
            if current.parameters or current.buffers:
                paths = set(current.parameters.values()) | set(current.buffers.values())
                if self.parameter_uses & paths:
                    raise UnsupportedModuleError(
                        "lowering: source state is used by multiple occurrences"
                    )
                self.parameter_uses.update(paths)
                states.append(
                    StateBinding(index, current.parameters, current.buffers, current.reshape_weight)
                )
            pending.extend((child, index) for child in reversed(current.children))
        architecture = Architecture.from_dict(
            {
                "schema": 1,
                "grammar": "einspace",
                "grammar_version": "1",
                "root": 0,
                "nodes": records,
                "input_spec": self.input_spec,
            }
        )
        return Lowering(architecture, bindings, tuple(states))


def lower(graph, input_spec):
    try:
        dtype = next((p.dtype for p in graph.parameters() if p.is_floating_point()), torch.float32)
        ShapeProp(graph).propagate(torch.empty(input_spec["shape"], dtype=dtype, device="meta"))
        return _Lowerer(graph, input_spec).finish()
    except UnsupportedModuleError:
        raise
    except Exception as error:
        raise UnsupportedModuleError(f"lowering: unsupported graph or shape: {error}") from error
