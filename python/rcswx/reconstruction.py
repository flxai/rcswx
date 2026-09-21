# Derived from einsearch 7e713c7951397a6b12bd57638409774a74381746.
# Copyright (c) 2024 Adri Gómez Martín, Felix Möller, Linus Ericsson, Aaron Klein.
# Distributed under the MIT license; see LICENSE.einsearch.
"""Original operation-driven reconstruction and metadata propagation."""

from .genotype import DerivationTreeNode, Stack
from .pcfg import OutOfOptionsError


class Reconstructor:
    def __init__(self, pcfg, limiter, mode, verbose=False):
        self.pcfg = pcfg
        self.limiter = limiter
        self.mode = mode
        self.verbose = verbose
        if self.mode == "iterative":
            self.__call__ = self.sample_iterative
        elif self.mode == "recursive":
            raise NotImplementedError("Recursive mode not implemented")

    def sample(self, input_params, operations=None, root=None):
        return self.__call__(input_params, operations, root)

    def re_id(self, root):
        self.limiter.timer.start()
        root = self.sample(
            input_params=root.input_params,
            root=None,
            operations=[node.operation for node in root.serialise()],
        )
        return root

    def sample_iterative(self, input_params, operations=None, root=None):
        if root is None:
            root = DerivationTreeNode(
                id=1, level="network", input_params=input_params, limiter=self.pcfg.limiter
            )
        self.nodes = {root.id: root}
        max_id = root.id
        stack = Stack([(root.id, False)])
        while not stack.is_empty():
            if self.verbose:
                print(f"Architecture so far: {root}")
            if operations is not None:
                if self.verbose:
                    print(f"Operations: {[op.name for op in operations]}")
            if self.verbose:
                print(f"Stack: {stack}")
            node_id, visited = stack.pop()
            node = self.nodes[node_id]
            if self.verbose:
                print(f"Node: {node.id}, visited: {visited}")
            if self.verbose:
                print(f"Node: {node}")
            if visited:
                node.give_back_output_params()
                if not node.is_root():
                    if self.verbose:
                        print(
                            f"Propagated output params from node {node.id} to parent {node.parent.id}"
                        )
                    if self.verbose:
                        print(
                            f"Input params for node: {node.parent.id}, {node.parent.input_params}"
                        )
                    if self.verbose:
                        print(
                            f"Output params for node: {node.parent.id}, {node.parent.output_params}"
                        )
            else:
                stack.append((node.id, True))
                if not node.is_root():
                    node.inherit_input_params()
                    if self.verbose:
                        print(
                            f"Inherited input params from parent {node.parent.id} to node {node.id}"
                        )
                    if self.verbose:
                        print(f"Input params for node: {node.id}, {node.input_params}")
                try:
                    if self.verbose:
                        print(f"Sampling node {node.id}")
                    operation = (
                        operations.pop(0)
                        if operations
                        else self.pcfg.sample(node, verbose=self.verbose)
                    )
                    if operation not in self.pcfg.get_available_options(node)[0]:
                        raise RuntimeError(
                            f"Operation {operation.name} not in available options: {[op.name for op in self.pcfg.get_available_options(node)[0]]}"
                        )
                    if self.verbose:
                        print(f"Selected operation: {operation.name}")
                    stack, max_id = node.initialise(operation, stack, max_id)
                    if self.verbose:
                        print(f"Output params for node: {node.id}, {node.output_params}")
                    for child in node.children:
                        if child.id not in self.nodes:
                            self.nodes[child.id] = child
                except OutOfOptionsError:
                    node = node.get_precursor()
                    stack, _ = node.memory
                    stack.restore(stack, node)
                    if self.verbose:
                        print(f"Backtracked to node {node.id}")
        return root
