# Derived from einsearch 7e713c7951397a6b12bd57638409774a74381746.
# Copyright (c) 2024 Adri Gómez Martín, Felix Möller, Linus Ericsson, Aaron Klein.
# Distributed under the MIT license; see LICENSE.einsearch.
"""Lossless derivation trees; operation callbacks remain on the Python side."""

import sys
import time
from copy import deepcopy

from tqdm import tqdm


class Operation:
    def __init__(self, name, build, infer, valid, inherit, give_back, type, child_levels=[]):
        self.name = name
        self.build = build
        self.infer = infer
        self.valid = valid
        self.inherit = inherit
        self.give_back = give_back
        self.type = type
        self.child_levels = child_levels

    def is_valid(self, node):
        return self.valid(node)

    def is_terminal(self):
        return self.type == "terminal"

    def copy(self):
        return Operation(
            name=self.name,
            build=self.build,
            infer=self.infer,
            valid=self.valid,
            inherit=self.inherit,
            give_back=self.give_back,
            type=self.type,
            child_levels=self.child_levels,
        )

    def __repr__(self):
        return f"Operation({self.name}, {self.type}, {self.child_levels})"

    def __sizeof__(self):
        return sum(map(sys.getsizeof, self.__dict__.values()))

    def __hash__(self):
        return hash(self.name)

    def __eq__(self, other):
        return self.name == other.name


class DerivationTreeNode:
    def __init__(
        self,
        id,
        level="network",
        parent=None,
        input_params={},
        output_params={},
        depth=0,
        limiter=None,
        operation=None,
    ):
        self.id = id
        self.level = level
        self.parent = parent
        self.children = []
        self.input_params = input_params
        self.output_params = output_params
        self.depth = depth
        self.limiter = limiter
        self.operation = operation
        self.available_rules = None

    def initialise(self, operation, stack=None, max_id=None, id_stack=True):
        if self.limiter:
            self.limiter.set_memory_checkpoint()
        if stack:
            self.memory = (deepcopy(stack), max_id)
        self.operation = operation
        if self.operation.is_terminal():
            self.output_params = self.operation.infer(self)
            if self.limiter and (not self.limiter.check_build_safe(self)):
                raise MemoryError(
                    f"Individual memory limit reached when initialising terminal node {self.id}"
                )
        else:
            self.children = []
            for i, child_level in enumerate(operation.child_levels):
                child = DerivationTreeNode(
                    id=max_id + i + 1,
                    level=child_level,
                    parent=self,
                    depth=self.depth + 1,
                    limiter=self.limiter,
                )
                if self.limiter and (not self.limiter.check_build_safe(child)):
                    raise MemoryError(
                        f"Individual memory limit would be exceeded by child node {child.id}"
                    )
                self.add_child(child)
        for child in reversed(self.children):
            if stack:
                if id_stack:
                    stack.append((child.id, False))
                else:
                    stack.append((child, False))
            max_id = max(child.id, max_id)
        if self.limiter and (not self.limiter.check_memory()):
            time.sleep(1)
            raise MemoryError(
                f"Memory blowup detected after initialising node {self.id}: {self.limiter.memory:.1f} MB used, {self.limiter.diff:.1f} MB added"
            )
        return (stack, max_id)

    def add_child(self, child):
        self.children.append(child)
        child.set_parent(self)

    def set_parent(self, parent):
        self.parent = parent

    def get_precursor(self):
        if self.is_root():
            self.precursor = None
        self_idx = self.parent.children.index(self)
        if self_idx == 0:
            precursor = self.parent
        else:
            precursor = self.parent.children[self_idx - 1]
            while precursor.children:
                precursor = precursor.children[-1]
        return precursor

    def inherit_input_params(self):
        child_idx = self.parent.children.index(self)
        self.parent.operation.inherit[child_idx](self)

    def give_back_output_params(self):
        if not self.is_root():
            child_idx = self.parent.children.index(self)
            self.parent.operation.give_back[child_idx](self)

    def is_root(self):
        return self.parent is None

    def is_leaf(self):
        return self.children == []

    def is_first_child(self):
        return self.parent.children[0] == self

    def get_root(self):
        if self.is_root():
            return self
        return self.parent.get_root()

    def serialise(self):
        nodes = [self]
        for child in self.children:
            nodes.extend(child.serialise())
        return nodes

    def num_params(self):
        root = self.get_root()

        def count_params(node):
            if node.is_leaf():
                output_params = node.operation.infer(node)
                if "num_params" in output_params:
                    return output_params["num_params"]
                else:
                    return 0
            else:
                return sum(count_params(child) for child in node.children)

        return count_params(root)

    def limit_options(self, operation):
        if self.available_rules:
            op_names = [op.name for op in self.available_rules["options"]]
            idx = op_names.index(operation.name)
            self.available_rules["options"].pop(idx)
            self.available_rules["probs"].pop(idx)
        else:
            pass

    def build(self, node, set_memory_checkpoint=False):
        if self.limiter and set_memory_checkpoint:
            self.limiter.set_memory_checkpoint()
        if self.limiter and (not self.limiter.check_memory()):
            raise MemoryError(
                f"Memory limit reached before building node {self.id}: {self.limiter.memory:.1f} MB"
            )
        if self.limiter and (not self.limiter.check_build_safe(node)):
            raise MemoryError(
                f"Individual memory limit would be exceeded when building node {self.id}"
            )
        network = self.operation.build(node)
        if self.limiter and (not self.limiter.check_memory()):
            raise MemoryError(
                f"Memory blowup detected after building node {self.id}: {self.limiter.memory:.1f} MB used, {self.limiter.diff:.1f} MB added"
            )
        return network

    def copy(self):
        node = DerivationTreeNode(
            id=self.id,
            level=self.level,
            input_params=deepcopy(self.input_params),
            depth=self.depth,
            limiter=self.limiter,
            operation=self.operation.copy() if self.operation else None,
        )
        node.output_params = deepcopy(self.output_params)
        node.available_rules = deepcopy(self.available_rules) if self.available_rules else None
        return node

    def replace(self, node):
        if self.is_root():
            self = node
        else:
            node.parent = self.parent
            child_idx = self.parent.children.index(self)
            self.parent.children[child_idx] = node

    def replace2(self, node):
        if self.is_root():
            return node
        node.parent = self.parent
        node.children = self.children
        for child in node.children:
            child.parent = node
        child_idx = self.parent.children.index(self)
        self.parent.children[child_idx] = node
        del self

    def __sizeof__(self):
        return sum(map(sys.getsizeof, self.__dict__.values()))

    def __repr__(self):
        return f"DerivationTreeNode(id={self.id}, level={self.level}, operation={self.operation}, input_params={self.input_params}, output_params={self.output_params}, depth={self.depth}, address={hex(id(self))})"

    def __str__(self):
        if self.operation:
            if "branching" in self.operation.name:
                brackets = "{}"
            elif "sequential" in self.operation.name:
                brackets = "()"
            elif "routing" in self.operation.name:
                brackets = "[]"
            elif "computation" in self.operation.name:
                brackets = "<>"
            elif (
                "cell" in self.operation.name
                or "residual" in self.operation.name
                or "diamond" in self.operation.name
            ):
                brackets = "()"
            else:
                brackets = None
            repr = f"{self.operation.name}"
            if brackets:
                repr += brackets[0]
            children_repr = ", ".join(str(child) for child in self.children)
            repr += children_repr
            if brackets:
                repr += brackets[1]
        else:
            repr = "None"
        return repr

    def param_string(self):
        if self.operation:
            d = {"out_feature_shape": list(self.output_params["shape"][1:])}
            return str(d)
        else:
            return ""

    def to_long_string(self):
        if self.operation:
            if "branching" in self.operation.name:
                brackets = "[]"
            elif "sequential" in self.operation.name:
                brackets = "[]"
            elif "routing" in self.operation.name:
                brackets = "[]"
            elif "computation" in self.operation.name:
                brackets = "[]"
            else:
                brackets = None
            repr = self.operation.name
            if brackets:
                repr += brackets[0]
            else:
                repr += self.param_string()
            children_repr = ", ".join(child.to_long_string() for child in self.children)
            repr += children_repr
            if brackets:
                repr += brackets[1]
        else:
            repr = "None"
        return repr

    def __hash__(self):
        return hash(self.id)

    def __eq__(self, other):
        return self.id == other.id


class Stack:
    def __init__(self, stack=[]):
        self.stack = stack

    def append(self, node):
        self.stack.append(node)

    def pop(self):
        return self.stack.pop()

    def restore(self, stack, node):
        self.stack = stack.stack
        self.stack[-1] = (self.stack[-1][0], False)
        node.limit_options(node.operation)

    def is_empty(self):
        return self.stack == []

    def is_completed(self):
        """Check if the stack only contain nodes that have been visited"""
        for node, visited in self.stack:
            if not visited:
                return False
        return True

    def copy(self):
        old_node_list = self.stack[0][0].serialise()
        new_node_list = []
        for node in tqdm(old_node_list, desc="Copying nodes"):
            new_node_list.append(node.copy())
        old_stack = self.stack
        new_stack = []
        for node, visited in tqdm(old_stack, desc="Copying stack"):
            idx = [node.id for node in new_node_list].index(node.id)
            new_stack.append((new_node_list[idx], visited))
        for i in tqdm(range(len(old_stack)), desc="Connecting parents and children"):
            if i == 0:
                new_stack[i][0].parent = None
            else:
                parent_id = old_stack[i][0].parent.id
                parent_idx = [node.id for node in new_node_list].index(parent_id)
                new_stack[i][0].parent = new_node_list[parent_idx]
            for child in old_stack[i][0].children:
                child_idx = [node.id for node in new_node_list].index(child.id)
                new_stack[i][0].add_child(new_node_list[child_idx])
        return Stack(new_stack)

    def __sizeof__(self):
        return sum(map(sys.getsizeof, self.__dict__.values()))

    def __repr__(self):
        repr = "Stack(\n"
        if self.stack:
            for node in self.stack[:-1]:
                repr += f"\t{node},\n"
            repr += f"\t{self.stack[-1]}\n"
        repr += ")"
        return repr

    def __str__(self):
        repr = "Stack(\n"
        if self.stack:
            for node in self.stack[:-1]:
                repr += f"\t{node},\n"
            repr += f"\t{self.stack[-1]}\n"
        repr += ")"
        return repr
