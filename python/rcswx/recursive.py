# Derived from einsearch 7e713c7951397a6b12bd57638409774a74381746.
# Copyright (c) 2024 Adri Gómez Martín, Felix Möller, Linus Ericsson, Aaron Klein.
# Distributed under the MIT license; see LICENSE.einsearch.
"""Lossless tree boundary and reference edit application around the native kernel."""

import copy
import time

import numpy as np
from termcolor import colored

from . import _core
from .genotype import DerivationTreeNode, Operation
from .grammars import einspace
from .sampling import select_operations


class MatrixOperation:
    def __init__(
        self,
        op_id=None,
        op_type=None,
        node1_id=None,
        node2_id=None,
        i=None,
        j=None,
        ii=None,
        jj=None,
        value=0,
        disabler_ops=[],
        enabler_ops=[],
    ):
        self.id = op_id
        self.op_type = op_type
        self.i = i
        self.j = j
        self.ii = ii
        self.jj = jj
        self.node1_id = node1_id
        self.node2_id = node2_id
        self.value = value
        self.i_swapped = False
        self.j_swapped = False
        self.disabler_ops = disabler_ops
        self.enabler_ops = enabler_ops

    def __str__(self):
        string = (
            self.op_type + " (id: " + str(self.id) + ") with a cost of " + str(self.value) + ". "
        )
        if len(self.disabler_ops):
            string = string[:-2] + " (disabled by "
            for branch in self.disabler_ops:
                if type(branch) is not list:
                    branch = [branch]
                for op in branch:
                    string += str(op.id) + ", "
            if string[-3:] == "by ":
                string += "none, "
            string = string[:-2] + ")."
        if len(self.enabler_ops):
            string = (
                string[:-2]
                + " (" * (len(self.disabler_ops) == 0)
                + "; " * (len(self.disabler_ops) > 0)
                + "enabled by "
            )
            for branch in self.enabler_ops:
                if type(branch) is not list:
                    branch = [branch]
                for op in branch:
                    string += str(op.id) + ", "
            if string[-3:] == "by ":
                string += "none, "
            string = string[:-2] + ")."
        return string

    def __repr__(self):
        return str(self)

    def __eq__(self, other):
        return self.id == other.id


class DecoyOperation:
    def __init__(self, name):
        self.name = name


class DecoyNode:
    def __init__(self, parent, branch, name):
        self.parent = parent
        self.operation = DecoyOperation(name)
        self.children = []
        if parent is None:
            self.id = -1
        else:
            self.id = self.parent.id

    def is_root(self):
        return self.parent is not None

    def __eq__(self, other):
        if isinstance(other, self.__class__):
            return self.id == other.id
        else:
            return False

    def __ne__(self, other):
        return not self.__eq__(other)

    def __str__(self):
        return str(self.operation.name) + " (id " + str(self.id) + ")"

    def __repr__(self):
        return str(self)


class Alignment:
    def __init__(self, parent1, parent2, collapse_corners=False, limiter=None):
        self.collapse_corners = collapse_corners
        self.verbose = False
        self.model1 = parent1
        self.model_ops1 = [DecoyNode(None, None, "start_node")] + self.breakdown(parent1)
        self.model2 = parent2
        self.new_node_id = max(node.id for node in parent1.serialise()) + 1
        for node in parent2.serialise():
            self.update_id(node)
        self.model_ops2 = [DecoyNode(None, None, "start_node")] + self.breakdown(parent2)
        self.limiter = limiter
        self.operations = []
        self.nontrivial_ops = []
        identities = {}
        original_ids = []

        def identity(value):
            if value not in identities:
                identities[value] = len(original_ids)
                original_ids.append(value)
            return identities[value]

        def snapshot(nodes):
            return [
                (
                    identity(node.id),
                    node.operation.name,
                    [child.operation.name for child in node.children],
                    len(node.parent.children) if node.parent is not None else 0,
                )
                for node in nodes
            ]

        first = snapshot(self.model_ops1)
        second = snapshot(self.model_ops2)
        started = time.perf_counter()
        self.distance, paths, self.stats = _core.recursive_align(
            first, second, collapse_corners, lambda: self.limiter.check_memory_crossover()
        )
        self.distance = 0 if len(first) == len(second) == 1 else np.float64(self.distance)
        self.compute_time = time.perf_counter() - started
        self.paths = []
        for path in paths:
            decoded = []
            for ident, kind, node1, node2, i, j, value, i_swapped, j_swapped in path:
                operation = MatrixOperation(
                    op_id=ident,
                    op_type=kind,
                    node1_id=original_ids[node1] if node1 is not None else None,
                    node2_id=original_ids[node2] if node2 is not None else None,
                    i=i,
                    j=j,
                    value=0 if kind == "start" else np.float64(value),
                )
                operation.i_swapped = i_swapped
                operation.j_swapped = j_swapped
                decoded.append(operation)
            self.paths.append(decoded)
        self.calculate_restrictions()

    def breakdown(self, node):
        model_ops = []
        if node.parent:
            condition = "computation" not in node.parent.operation.name
        else:
            condition = True
        if condition:
            if "sequential" not in node.operation.name:
                model_ops = [node]
            if len(node.children) > 2:
                for child in range(1, len(node.children) - 1):
                    model_ops += self.breakdown(node.children[child])
                    model_ops += [
                        DecoyNode(
                            node,
                            child,
                            "wrap_"
                            + "end" * (child == len(node.children) - 2)
                            + "sep" * (child != len(node.children) - 2),
                        )
                    ]
            else:
                for child in node.children:
                    model_ops += self.breakdown(child)
        return model_ops

    def update_id(self, node):
        node.id = self.new_node_id
        self.new_node_id += 1

    def split_sequentials(self, original_node, split_id):
        if split_id not in [n.id for n in original_node.serialise()]:
            return original_node
        parent_node = original_node.parent
        if not original_node.is_root():
            child_idx = parent_node.children.index(original_node)
        nodes_list = original_node.children
        sequential_in_list = True
        while sequential_in_list:
            sequential_in_list = False
            for n, node in enumerate(nodes_list):
                if node.operation.name == "sequential":
                    sequential_in_list = True
                    nodes_list = nodes_list[:n] + node.children + nodes_list[n + 1 :]
                    break
        for n, node in enumerate(nodes_list):
            if node.id == split_id:
                break
        list1 = nodes_list[:n]
        list2 = nodes_list[n:]
        for _ in range(len(list1) - 1):
            child2 = list1.pop()
            child1 = list1.pop()
            list1 += [
                DerivationTreeNode(
                    0,
                    level=node.level,
                    parent=node.parent,
                    input_params=node.input_params,
                    depth=node.depth,
                    limiter=node.limiter,
                    operation=Operation(
                        name="sequential",
                        build=einspace.build_sequential_module,
                        infer=einspace.infer_sequential_module,
                        valid=einspace.valid_sequential_module,
                        inherit=[einspace.inherit_first_child, einspace.inherit_other_child],
                        give_back=[einspace.give_back_default, einspace.give_back_default],
                        type="nonterminal",
                        child_levels=["module", "module"],
                    ),
                )
            ]
            self.update_id(list1[-1])
            list1[-1].children = [child1, child2]
            child1.parent = list1[-1]
            child2.parent = list1[-1]
        for _ in range(len(list2) - 1):
            child2 = list2.pop()
            child1 = list2.pop()
            list2 += [
                DerivationTreeNode(
                    0,
                    level=node.level,
                    parent=node.parent,
                    input_params=node.input_params,
                    depth=node.depth,
                    limiter=node.limiter,
                    operation=Operation(
                        name="sequential",
                        build=einspace.build_sequential_module,
                        infer=einspace.infer_sequential_module,
                        valid=einspace.valid_sequential_module,
                        inherit=[einspace.inherit_first_child, einspace.inherit_other_child],
                        give_back=[einspace.give_back_default, einspace.give_back_default],
                        type="nonterminal",
                        child_levels=["module", "module"],
                    ),
                )
            ]
            self.update_id(list2[-1])
            list2[-1].children = [child1, child2]
            child1.parent = list2[-1]
            child2.parent = list2[-1]
        if len(list2):
            resequentialized_node = DerivationTreeNode(
                0,
                level=node.level,
                parent=node.parent,
                input_params=node.input_params,
                depth=node.depth,
                limiter=node.limiter,
                operation=Operation(
                    name="sequential",
                    build=einspace.build_sequential_module,
                    infer=einspace.infer_sequential_module,
                    valid=einspace.valid_sequential_module,
                    inherit=[einspace.inherit_first_child, einspace.inherit_other_child],
                    give_back=[einspace.give_back_default, einspace.give_back_default],
                    type="nonterminal",
                    child_levels=["module", "module"],
                ),
            )
            self.update_id(resequentialized_node)
            resequentialized_node.children = [list1[0], list2[0]]
            list1[0].parent = resequentialized_node
            list2[0].parent = resequentialized_node
        else:
            raise Exception("Unable to resequentialize as requested")
        if not original_node.is_root():
            parent_node.children[child_idx] = resequentialized_node
        resequentialized_node.parent = parent_node
        return resequentialized_node

    def calculate_restrictions(self, path_n=0):
        path_idxs = [path_idx for path_idx in range(len(self.paths))]
        ids1 = [node.id for node in self.model_ops1]
        ids2 = [node.id for node in self.model_ops2]
        for path_idx in path_idxs:
            operations = copy.deepcopy(self.paths[path_idx])
            for idx, op in enumerate(operations):
                if "wrap_end" in op.op_type:
                    if ("add" in op.op_type or "mut" in op.op_type) and len(
                        self.model_ops1[ids1.index(op.node1_id)].children
                    ) == 4:
                        i_ops = [
                            in_op
                            for in_op in operations
                            if in_op.op_type[:3] == op.op_type[:3] and in_op.node1_id == op.node1_id
                        ]
                        for i_op in i_ops:
                            i_op.i_swapped = op.i_swapped
                        i = [i_op.i for i_op in i_ops]
                        if op.i_swapped:
                            for inside_op in operations:
                                if inside_op.i >= i[0] and inside_op.i < i[1]:
                                    inside_op.i += i[2] - i[1]
                                elif inside_op.i >= i[1] and inside_op.i < i[2]:
                                    inside_op.i -= i[1] - i[0]
            for idx, op in enumerate(operations):
                if "wrap_end" in op.op_type:
                    if ("rem" in op.op_type or "mut" in op.op_type) and len(
                        self.model_ops2[ids2.index(op.node2_id)].children
                    ) == 4:
                        j_ops = [
                            j_op
                            for j_op in operations
                            if j_op.op_type[:3] == op.op_type[:3] and j_op.node2_id == op.node2_id
                        ]
                        for j_op in j_ops:
                            j_op.j_swapped = op.j_swapped
                        j = [j_op.j for j_op in j_ops]
                        if op.j_swapped:
                            for inside_op in operations:
                                if inside_op.j >= j[0] and inside_op.j < j[1]:
                                    inside_op.j += j[2] - j[1]
                                elif inside_op.j >= j[1] and inside_op.j < j[2]:
                                    inside_op.j -= j[1] - j[0]
            for idx, op in enumerate(operations):
                if "add" in op.op_type:
                    if len(self.model_ops1[op.i].children) == 4:
                        for sep_idx, sep_operation in enumerate(operations[idx + 1 :]):
                            if (
                                "add_wrap" in sep_operation.op_type
                                and self.model_ops1[sep_operation.i].id == self.model_ops1[op.i].id
                            ):
                                break
                        sep_idx += idx + 1
                        end_idx = 0
                        for end_idx, end_operation in enumerate(operations[sep_idx + 1 :]):
                            if (
                                end_operation.op_type == "add_wrap_end"
                                and self.model_ops1[end_operation.i].id == self.model_ops1[op.i].id
                            ):
                                break
                        end_idx += sep_idx + 1
                        disabler_ops = []
                        enabler_ops = []
                        adds = [[], []]
                        muts = [[], []]
                        rems = [[], []]
                        for inside_op in operations[idx : sep_idx + 1]:
                            if "wrap" not in inside_op.op_type:
                                if "add" in inside_op.op_type:
                                    adds[0] += [inside_op]
                                if "mut" in inside_op.op_type:
                                    muts[0] += [inside_op]
                                if "rem" in inside_op.op_type:
                                    rems[0] += [inside_op]
                        if not len(muts[0]):
                            for rem_op in rems[0]:
                                rem_op.enabler_ops = rem_op.enabler_ops + adds[0]
                                rem_op.disabler_ops = rem_op.disabler_ops + [
                                    rem_op2 for rem_op2 in rems[0] if rem_op2 != rem_op
                                ]
                            disabler_ops = disabler_ops + [rems[0]]
                            enabler_ops = enabler_ops + [adds[0] * len(rems[0])]
                        for inside_op in operations[sep_idx + 1 : end_idx]:
                            if "wrap" not in inside_op.op_type:
                                if "add" in inside_op.op_type:
                                    adds[1] += [inside_op]
                                if "mut" in inside_op.op_type:
                                    muts[1] += [inside_op]
                                if "rem" in inside_op.op_type:
                                    rems[1] += [inside_op]
                        if not len(muts[1]):
                            for rem_op in rems[1]:
                                rem_op.enabler_ops = rem_op.enabler_ops + adds[1]
                                rem_op.disabler_ops = rem_op.disabler_ops + [
                                    rem_op2 for rem_op2 in rems[1] if rem_op2 != rem_op
                                ]
                            disabler_ops = disabler_ops + [rems[1]]
                            enabler_ops = enabler_ops + [adds[1] * len(rems[1])]
                        op.disabler_ops = disabler_ops
                        op.enabler_ops = enabler_ops
                        if (
                            len(adds[0])
                            and (not len(rems[0]))
                            and (not len(muts[0]))
                            or (len(adds[1]) and (not len(rems[1])) and (not len(muts[1])))
                        ):
                            op.disabler_ops = op.disabler_ops + [[op], [op]]
                            op.enabler_ops = op.enabler_ops + adds
                    elif len(self.model_ops1[op.i].children) == 3:
                        for end_idx, end_operation in enumerate(operations[idx:]):
                            if (
                                end_operation.op_type == "add_wrap_end"
                                and self.model_ops1[end_operation.i].id == self.model_ops1[op.i].id
                            ):
                                break
                        end_idx += idx
                        disabler_ops = []
                        enabler_ops = []
                        adds = []
                        muts = []
                        rems = []
                        for inside_op in operations[idx : end_idx + 1]:
                            if "wrap" not in inside_op.op_type:
                                if "add" in inside_op.op_type:
                                    adds += [inside_op]
                                if "mut" in inside_op.op_type:
                                    muts += [inside_op]
                                if "rem" in inside_op.op_type:
                                    rems += [inside_op]
                        if not len(muts):
                            for rem_op in rems:
                                rem_op.enabler_ops = rem_op.enabler_ops + adds
                                rem_op.disabler_ops = rem_op.disabler_ops + [
                                    rem_op2 for rem_op2 in rems if rem_op2 != rem_op
                                ]
                            disabler_ops = disabler_ops + rems
                            enabler_ops = enabler_ops + adds * len(rems)
                        op.ii = end_operation.i
                        op.jj = end_operation.j
                        op.disabler_ops = disabler_ops
                        op.enabler_ops = enabler_ops
                        if len(adds) and (not len(rems)) and (not len(muts)):
                            op.disabler_ops = op.disabler_ops + [op]
                            op.enabler_ops = op.enabler_ops + adds
                elif "rem" in op.op_type:
                    if len(self.model_ops2[op.j].children) == 3:
                        for end_idx, end_operation in enumerate(operations[idx:]):
                            if (
                                end_operation.op_type == "rem_wrap_end"
                                and self.model_ops2[end_operation.j].id == self.model_ops2[op.j].id
                            ):
                                break
                        end_idx += idx
                        adds = []
                        muts = []
                        rems = []
                        for inside_op in operations[idx : end_idx + 1]:
                            if "wrap" not in inside_op.op_type:
                                if "add" in inside_op.op_type:
                                    adds += [inside_op]
                                if "mut" in inside_op.op_type:
                                    muts += [inside_op]
                                if "rem" in inside_op.op_type:
                                    rems += [inside_op]
                        if not len(muts):
                            for rem_op in rems:
                                rem_op.disabler_ops = rem_op.disabler_ops + rems
                                rem_op.enabler_ops = rem_op.enabler_ops + adds + [op]
                    elif len(self.model_ops2[op.j].children) == 4:
                        for sep_idx, sep_operation in enumerate(operations[idx + 1 :]):
                            if (
                                "rem_wrap" in sep_operation.op_type
                                and self.model_ops2[sep_operation.j].id == self.model_ops2[op.j].id
                            ):
                                break
                        sep_idx += idx + 1
                        end_idx = 0
                        for end_idx, end_operation in enumerate(operations[sep_idx + 1 :]):
                            if (
                                end_operation.op_type == "rem_wrap_end"
                                and self.model_ops2[end_operation.j].id == self.model_ops2[op.j].id
                            ):
                                break
                        end_idx += sep_idx + 1
                        adds = [[], []]
                        muts = [[], []]
                        rems = [[], []]
                        for inside_op in operations[idx : sep_idx + 1]:
                            if "wrap" not in inside_op.op_type:
                                if "add" in inside_op.op_type:
                                    adds[0] += [inside_op]
                                if "mut" in inside_op.op_type:
                                    muts[0] += [inside_op]
                                if "rem" in inside_op.op_type:
                                    rems[0] += [inside_op]
                        if not len(muts[0]):
                            for rem_op in rems[0]:
                                rem_op.disabler_ops = rem_op.disabler_ops + rems[0]
                                rem_op.enabler_ops = rem_op.enabler_ops + adds[0] + [op]
                        for inside_op in operations[sep_idx + 1 : end_idx]:
                            if "wrap" not in inside_op.op_type:
                                if "add" in inside_op.op_type:
                                    adds[1] += [inside_op]
                                if "mut" in inside_op.op_type:
                                    muts[1] += [inside_op]
                                if "rem" in inside_op.op_type:
                                    rems[1] += [inside_op]
                        if not len(muts[1]):
                            for rem_op in rems[1]:
                                rem_op.disabler_ops = rem_op.disabler_ops + rems[1]
                                rem_op.enabler_ops = rem_op.enabler_ops + adds[1] + [op]
            self.paths[path_idx] = operations
            if path_idx == path_n:
                self.operations = operations[1:]
        self.operations_unordered = self.operations.copy()
        for idx, op in enumerate(self.operations):
            if (
                "add" in op.op_type
                and (
                    len(self.model_ops1[op.i].children) == 4
                    or "sep" in self.model_ops1[op.i].operation.name
                )
                and op.i_swapped
            ):
                for sep_idx, sep_operation in enumerate(self.operations[idx + 1 :]):
                    if (
                        "add" in sep_operation.op_type
                        and self.model_ops1[sep_operation.i].id == self.model_ops1[op.i].id
                    ):
                        break
                sep_idx += idx + 1
                end_idx = 0
                for end_idx, end_operation in enumerate(self.operations[sep_idx + 1 :]):
                    if (
                        "add" in end_operation.op_type
                        and self.model_ops1[end_operation.i].id == self.model_ops1[op.i].id
                    ):
                        break
                end_idx += sep_idx + 1
                if (
                    self.model_ops1[sep_operation.i].id == self.model_ops1[op.i].id
                    and self.model_ops1[end_operation.i].id == self.model_ops1[op.i].id
                ):
                    self.operations = (
                        self.operations[: idx + 1]
                        + self.operations[sep_idx:end_idx]
                        + self.operations[idx + 1 : sep_idx]
                        + self.operations[end_idx:]
                    )
            elif (
                "rem" in op.op_type
                and (
                    len(self.model_ops2[op.j].children) == 4
                    or "sep" in self.model_ops2[op.j].operation.name
                )
                and op.j_swapped
            ):
                for sep_idx, sep_operation in enumerate(self.operations[idx + 1 :]):
                    if (
                        "rem" in sep_operation.op_type
                        and self.model_ops2[sep_operation.j].id == self.model_ops2[op.j].id
                    ):
                        break
                sep_idx += idx + 1
                end_idx = 0
                for end_idx, end_operation in enumerate(self.operations[sep_idx + 1 :]):
                    if (
                        "rem" in end_operation.op_type
                        and self.model_ops2[end_operation.j].id == self.model_ops2[op.j].id
                    ):
                        break
                end_idx += sep_idx + 1
                if (
                    self.model_ops2[sep_operation.j].id == self.model_ops2[op.j].id
                    and self.model_ops2[end_operation.j].id == self.model_ops2[op.j].id
                ):
                    self.operations = (
                        self.operations[: idx + 1]
                        + self.operations[sep_idx:end_idx]
                        + self.operations[idx + 1 : sep_idx]
                        + self.operations[end_idx:]
                    )
            elif (
                "mut" in op.op_type
                and (
                    len(self.model_ops1[op.i].children) == 4
                    or "sep" in self.model_ops1[op.i].operation.name
                )
                and (op.i_swapped or op.j_swapped)
            ):
                for sep_idx, sep_operation in enumerate(self.operations[idx + 1 :]):
                    if self.model_ops1[sep_operation.i].id == self.model_ops1[op.i].id:
                        break
                sep_idx += idx + 1
                for end_idx, end_operation in enumerate(self.operations[sep_idx + 1 :]):
                    if self.model_ops1[end_operation.i].id == self.model_ops1[op.i].id:
                        break
                end_idx += sep_idx + 1
                if (
                    self.model_ops1[sep_operation.i].id == self.model_ops1[op.i].id
                    and self.model_ops1[end_operation.i].id == self.model_ops1[op.i].id
                ):
                    self.operations = (
                        self.operations[: idx + 1]
                        + self.operations[sep_idx:end_idx]
                        + self.operations[idx + 1 : sep_idx]
                        + self.operations[end_idx:]
                    )
        self.operations.reverse()
        self.nontrivial_ops = [operation for operation in self.operations if operation.value]

    def generate_offspring(self, selected_ops=None):
        if selected_ops == None:
            selected_ops = self.nontrivial_ops
        if self.verbose:
            print(
                ">>>Parent model 1\n",
                colored(self.model2, "red"),
                "\n>>>Parent model 2\n",
                colored(self.model1, "green"),
                "\n",
            )
        self.performed_ops = []
        offspring = self.apply_all_operations(selected_ops, copy.deepcopy(self.model2))
        if self.verbose:
            print(">>>Final model\n", colored(offspring, "yellow"))
        return offspring

    def apply_all_operations(self, selected_ops, offspring):
        for op in selected_ops:
            if len(op.enabler_ops):
                if type(op.enabler_ops) is list and (op.i_swapped or op.j_swapped):
                    op.enabler_ops.reverse()
                for branch in op.enabler_ops:
                    if type(branch) is not list:
                        branch = [branch]
                    offspring = self.apply_all_operations(
                        [r_op for r_op in branch if r_op in selected_ops], offspring
                    )
            offspring = self.apply_op(op, offspring)
        return offspring

    def apply_op(self, op, offspring):
        if op.id not in self.performed_ops:
            self.performed_ops += [op.id]
            if "mut" in op.op_type:
                for node in self.model_ops1:
                    if node.id == self.model_ops1[op.i].id:
                        node1 = copy.deepcopy(node)
                        break
                for node in offspring.serialise():
                    if node.id == self.model_ops2[op.j].id or node.id == self.model_ops1[op.i].id:
                        node2 = copy.deepcopy(node)
                        break
                if not node1.id == self.model_ops1[op.i].id or (
                    not node.id == self.model_ops2[op.j].id
                    and (not node.id == self.model_ops1[op.i].id)
                ):
                    raise Exception("Nodes to mutate not found")
                node1str = str(node1)
                if "branching" in node1.operation.name:
                    node1str = node1str.split(")")[0] + ")" + node1str.split(")")[1] + ")...}"
                if "routing" in node1.operation.name:
                    node1str = node1str.split(")")[0] + ")...]"
                node2str = str(node2)
                if "branching" in node2.operation.name:
                    node2str = node2str.split(")")[0] + ")" + node2str.split(")")[1] + ")...}"
                if "routing" in node2.operation.name:
                    node2str = node2str.split(")")[0] + ")...]"
                if self.verbose:
                    print(
                        ">>>Mutating", colored(node2str, "red"), "into", colored(node1str, "green")
                    )
                node1.parent = node2.parent
                if not node2.is_root():
                    node2.parent.children[node2.parent.children.index(node2)] = node1
                if "branching(2)" in node1.operation.name:
                    if sum([op.i_swapped, op.j_swapped]) == 1:
                        node1.children[2] = node2.children[1]
                        node1.children[1] = node2.children[2]
                    else:
                        node1.children[1] = node2.children[1]
                        node1.children[2] = node2.children[2]
                    node1.children[1].parent = node1
                    node1.children[2].parent = node1
                elif "branching" in node1.operation.name or "routing" in node1.operation.name:
                    node1.children[1] = node2.children[1]
                    node1.children[1].parent = node1
                offspring = node1.get_root()
            elif "rem" in op.op_type:
                for node in offspring.serialise():
                    if node.id == self.model_ops2[op.j].id:
                        break
                if not node.id == self.model_ops2[op.j].id:
                    print(
                        ">>>Tried to remove module with id",
                        colored(self.model_ops2[op.j].id, "red"),
                        "but it was not found.",
                    )
                elif "branching(2)" in node.operation.name:
                    b1 = node.children[1 + op.j_swapped]
                    b2 = node.children[2 - op.j_swapped]
                    if self.verbose:
                        print(
                            ">>>Serializing branches",
                            colored(str(b1), "red"),
                            "and",
                            colored(str(b2), "red") + " (swapping branches)" * op.j_swapped,
                        )
                    sequential_node = DerivationTreeNode(
                        0,
                        level=node.level,
                        parent=node.parent,
                        input_params=node.input_params,
                        depth=node.depth,
                        limiter=node.limiter,
                        operation=Operation(
                            name="sequential",
                            build=einspace.build_sequential_module,
                            infer=einspace.infer_sequential_module,
                            valid=einspace.valid_sequential_module,
                            inherit=[einspace.inherit_first_child, einspace.inherit_other_child],
                            give_back=[einspace.give_back_default, einspace.give_back_default],
                            type="nonterminal",
                            child_levels=["module", "module"],
                        ),
                    )
                    self.update_id(sequential_node)
                    sequential_node.children = [b1, b2]
                    b1.parent = sequential_node
                    b2.parent = sequential_node
                    if node.parent:
                        node.parent.children[node.parent.children.index(node)] = sequential_node
                    offspring = sequential_node.get_root()
                else:
                    nodestr = str(node)
                    if "branching" in node.operation.name:
                        nodestr = nodestr.split(")")[0] + ")" + nodestr.split(")")[1] + ")...}"
                    if "routing" in node.operation.name:
                        nodestr = nodestr.split(")")[0] + ")...]"
                    if self.verbose:
                        print(">>>Removing", colored(nodestr, "red"))
                    parent_node = node.parent
                    if "branching" in node.operation.name or "routing" in node.operation.name:
                        if parent_node:
                            parent_node.children[parent_node.children.index(node)] = node.children[
                                1
                            ]
                        node.children[1].parent = node.parent
                        offspring = node.children[1].get_root()
                    else:
                        if parent_node.operation.name == "sequential":
                            sibling_node = parent_node.children[
                                parent_node.children.index(node) - 1
                            ]
                        else:
                            sibling_node = parent_node.children[
                                1 + (parent_node.children.index(node) == 1)
                            ]
                        if not parent_node.is_root():
                            sibling_node.parent = parent_node.parent
                            parent_node.parent.children[
                                parent_node.parent.children.index(parent_node)
                            ] = sibling_node
                        else:
                            sibling_node.parent = None
                        offspring = sibling_node.get_root()
            elif op.op_type == "add_wrap":
                for node in self.model_ops1:
                    if node.id == self.model_ops1[op.i].id:
                        node1 = copy.deepcopy(node)
                        break
                offspring_serialised = offspring.serialise()
                if len(node.children) == 3:
                    split_pos = [0, 0]
                    jjj = 1
                    if op.j + jjj < len(self.model_ops2):
                        while "wrap_" in self.model_ops2[
                            op.j + jjj
                        ].operation.name or self.model_ops2[op.j + jjj].id not in [
                            n.id for n in offspring.serialise()
                        ]:
                            jjj += 1
                            if op.j + jjj == len(self.model_ops2):
                                break
                    if op.j + jjj < len(self.model_ops2):
                        while (
                            offspring.serialise()[split_pos[0]].id != self.model_ops2[op.j + jjj].id
                        ):
                            split_pos[0] = split_pos[0] + 1
                            if split_pos[0] == len(offspring.serialise()):
                                break
                    else:
                        split_pos[0] = len(offspring.serialise())
                    iii = 0
                    if op.i + iii < len(self.model_ops1):
                        while "wrap_" in self.model_ops1[
                            op.i + iii
                        ].operation.name or self.model_ops1[op.i + iii].id not in [
                            n.id for n in offspring.serialise()
                        ]:
                            iii += 1
                            if op.i + iii == len(self.model_ops1):
                                break
                    if op.i + iii < len(self.model_ops1):
                        while (
                            offspring.serialise()[split_pos[1]].id != self.model_ops1[op.i + iii].id
                        ):
                            split_pos[1] += 1
                            if split_pos[1] == len(offspring.serialise()):
                                break
                    else:
                        split_pos[1] = len(offspring.serialise())
                    node = offspring.serialise()[min(split_pos)]
                    starting_node = node
                    depths = self.calculate_depth_of_path(offspring_serialised)
                    after_layer = [0, 0]
                    closing_op, closing_j = [
                        (in_idx, in_op.j)
                        for in_idx, in_op in enumerate(self.operations_unordered)
                        if in_op.node1_id == op.node1_id and "_end" in in_op.op_type
                    ][0]
                    try:
                        if (
                            self.model_ops2[closing_j] in offspring_serialised
                            and "_end" in self.model_ops2[closing_j].operation.name
                        ):
                            selected_idx0, op_distances0 = [
                                (aux_idx, abs(aux_idx - closing_op))
                                for aux_idx, aux_op in enumerate(self.operations_unordered)
                                if aux_op.j == closing_j
                            ][0]
                            after_layer[0] = True
                        else:
                            op_distances0 = []
                            for aux_idx, aux_op in enumerate(self.operations_unordered):
                                if (
                                    ("rem" in aux_op.op_type or "mut" in aux_op.op_type)
                                    and self.model_ops2[aux_op.j] in offspring_serialised
                                    and (
                                        (
                                            "_sep" not in self.model_ops2[aux_op.j].operation.name
                                            and (not aux_op.j_swapped)
                                            or (
                                                "_sep" in self.model_ops2[aux_op.j].operation.name
                                                and aux_op.j_swapped
                                            )
                                        )
                                        and "_end" not in self.model_ops2[aux_op.j].operation.name
                                    )
                                    and (
                                        aux_idx - closing_op > 0
                                        or len(self.model_ops2[aux_op.j].children) < 2
                                    )
                                ):
                                    op_distances0 += [aux_idx - closing_op]
                                else:
                                    op_distances0 += [np.inf]
                            op_distances0 = [
                                op_dist
                                if depths[op_idx] == depths[self.operations_unordered.index(op)]
                                and op_dist != 0
                                else np.inf
                                for op_idx, op_dist in enumerate(op_distances0)
                            ]
                            selected_idx0 = np.argmin([abs(op_dist0) for op_dist0 in op_distances0])
                            after_layer[0] = op_distances0[selected_idx0] < 0
                            op_distances0 = abs(op_distances0[selected_idx0])
                    except BaseException:
                        selected_idx0 = -1
                        op_distances0 = np.inf
                        after_layer[0] = False
                    try:
                        op_distances1 = []
                        for aux_idx, aux_op in enumerate(self.operations_unordered):
                            if (
                                ("add" in aux_op.op_type or "mut" in aux_op.op_type)
                                and self.model_ops1[aux_op.i] in offspring_serialised
                                and (
                                    (
                                        "_sep" not in self.model_ops1[aux_op.i].operation.name
                                        and (not aux_op.i_swapped)
                                        or (
                                            "_sep" in self.model_ops1[aux_op.i].operation.name
                                            and aux_op.i_swapped
                                        )
                                    )
                                    and "_end" not in self.model_ops1[aux_op.i].operation.name
                                )
                                and (
                                    aux_idx - closing_op > 0
                                    or len(self.model_ops1[aux_op.i].children) < 2
                                )
                            ):
                                op_distances1 += [aux_idx - closing_op]
                            else:
                                op_distances1 += [np.inf]
                        op_distances1 = [
                            op_dist
                            if depths[op_idx] == depths[self.operations_unordered.index(op)]
                            and op_dist != 0
                            else np.inf
                            for op_idx, op_dist in enumerate(op_distances1)
                        ]
                        selected_idx1 = np.argmin([abs(op_dist1) for op_dist1 in op_distances1])
                        after_layer[1] = op_distances1[selected_idx1] < 0
                        op_distances1 = abs(op_distances1[selected_idx1])
                    except BaseException:
                        selected_idx1 = -1
                        op_distances1 = np.inf
                        after_layer[1] = False
                    chosen_pos = np.argmin([op_distances0, op_distances1])
                    after_layer = after_layer[chosen_pos]
                    final_layer = [
                        self.model_ops2[self.operations_unordered[selected_idx0].j],
                        self.model_ops1[self.operations_unordered[selected_idx1].i],
                    ][chosen_pos]
                    final_layer_id = final_layer.id
                    try:
                        split_pos = offspring_serialised.index(final_layer)
                    except BaseException:
                        split_pos = len(offspring_serialised)
                    if split_pos == len(offspring_serialised):
                        end_at_id = -1
                    elif after_layer and split_pos < len(offspring_serialised):
                        jump_ids = [
                            aux_node.id for aux_node in offspring_serialised[split_pos].serialise()
                        ]
                        split_pos += 1
                        while (
                            offspring_serialised[split_pos] not in self.model_ops1 + self.model_ops2
                            or offspring_serialised[split_pos].id in jump_ids
                        ):
                            split_pos += 1
                            if split_pos == len(offspring_serialised):
                                break
                        if split_pos < len(offspring_serialised):
                            end_at_id = offspring_serialised[split_pos].id
                        else:
                            end_at_id = -1
                    else:
                        end_at_id = offspring_serialised[split_pos].id
                    found_parent_sequential = False
                    if split_pos == len(offspring_serialised):
                        while not found_parent_sequential:
                            if not node.is_root():
                                if node.parent.operation.name == "sequential" and end_at_id not in [
                                    aux_node.id for aux_node in node.serialise()
                                ]:
                                    node = node.parent
                                else:
                                    found_parent_sequential = True
                            else:
                                found_parent_sequential = True
                    else:
                        while not found_parent_sequential:
                            if not node.is_root():
                                if node.parent.operation.name == "sequential" and end_at_id not in [
                                    aux_node.id for aux_node in node.serialise()
                                ]:
                                    node = node.parent
                                else:
                                    found_parent_sequential = True
                            else:
                                found_parent_sequential = True
                    if node.operation.name == "sequential":
                        try:
                            node = self.split_sequentials(node, starting_node.id).children[1]
                        except BaseException:
                            node = node
                    else:
                        node = node
                    if node.operation.name == "sequential":
                        if end_at_id != -1 and end_at_id in [
                            aux_node.id for aux_node in node.serialise()
                        ]:
                            node2 = self.split_sequentials(node, end_at_id).children[0]
                        else:
                            node2 = node
                    else:
                        node2 = node
                    if self.verbose:
                        print(
                            ">>>Adding wrapper",
                            colored(node1.operation.name, "green"),
                            "around",
                            colored(str(node2), "red"),
                        )
                    node1.parent = node2.parent
                    if not node2.is_root():
                        node2.parent.children[node2.parent.children.index(node2)] = node1
                    node1.children[1] = node2
                    node2.parent = node1
                    offspring = node1.get_root()
                elif len(node.children) == 4:
                    split_pos = [0, 0]
                    jjj = 1
                    if op.j + jjj < len(self.model_ops2):
                        while "wrap_" in self.model_ops2[
                            op.j + jjj
                        ].operation.name or self.model_ops2[op.j + jjj].id not in [
                            n.id for n in offspring.serialise()
                        ]:
                            jjj += 1
                            if op.j + jjj == len(self.model_ops2):
                                break
                    if op.j + jjj < len(self.model_ops2):
                        while (
                            offspring.serialise()[split_pos[0]].id != self.model_ops2[op.j + jjj].id
                        ):
                            split_pos[0] = split_pos[0] + 1
                            if split_pos[0] == len(offspring.serialise()):
                                break
                    else:
                        split_pos[0] = len(offspring.serialise())
                    iii = 0
                    if op.i + iii < len(self.model_ops1):
                        while "wrap_" in self.model_ops1[
                            op.i + iii
                        ].operation.name or self.model_ops1[op.i + iii].id not in [
                            n.id for n in offspring.serialise()
                        ]:
                            iii += 1
                            if op.i + iii == len(self.model_ops1):
                                break
                    if op.i + iii < len(self.model_ops1):
                        while (
                            offspring.serialise()[split_pos[1]].id != self.model_ops1[op.i + iii].id
                        ):
                            split_pos[1] += 1
                            if split_pos[1] == len(offspring.serialise()):
                                break
                    else:
                        split_pos[1] = len(offspring.serialise())
                    node = offspring.serialise()[min(split_pos)]
                    starting_node = node
                    depths = self.calculate_depth_of_path(offspring_serialised)
                    after_layer = [0, 0]
                    closing_op, closing_j = [
                        (in_idx, in_op.j)
                        for in_idx, in_op in enumerate(self.operations_unordered)
                        if in_op.node1_id == op.node1_id and "_end" in in_op.op_type
                    ][0]
                    try:
                        if (
                            self.model_ops2[closing_j] in offspring_serialised
                            and "_end" in self.model_ops2[closing_j].operation.name
                        ):
                            selected_idx0, op_distances0 = [
                                (aux_idx, abs(aux_idx - closing_op))
                                for aux_idx, aux_op in enumerate(self.operations_unordered)
                                if aux_op.j == closing_j
                            ][0]
                            after_layer[0] = True
                        else:
                            op_distances0 = []
                            for aux_idx, aux_op in enumerate(self.operations_unordered):
                                if (
                                    ("rem" in aux_op.op_type or "mut" in aux_op.op_type)
                                    and self.model_ops2[aux_op.j] in offspring_serialised
                                    and (
                                        (
                                            "_sep" not in self.model_ops2[aux_op.j].operation.name
                                            and (not aux_op.j_swapped)
                                            or (
                                                "_sep" in self.model_ops2[aux_op.j].operation.name
                                                and aux_op.j_swapped
                                            )
                                        )
                                        and "_end" not in self.model_ops2[aux_op.j].operation.name
                                    )
                                    and (
                                        aux_idx - closing_op > 0
                                        or len(self.model_ops2[aux_op.j].children) < 2
                                    )
                                ):
                                    op_distances0 += [aux_idx - closing_op]
                                else:
                                    op_distances0 += [np.inf]
                            op_distances0 = [
                                op_dist
                                if depths[op_idx] == depths[self.operations_unordered.index(op)]
                                and op_dist != 0
                                else np.inf
                                for op_idx, op_dist in enumerate(op_distances0)
                            ]
                            selected_idx0 = np.argmin([abs(op_dist0) for op_dist0 in op_distances0])
                            after_layer[0] = op_distances0[selected_idx0] < 0
                            op_distances0 = abs(op_distances0[selected_idx0])
                    except BaseException:
                        selected_idx0 = -1
                        op_distances0 = np.inf
                        after_layer[0] = False
                    try:
                        op_distances1 = []
                        for aux_idx, aux_op in enumerate(self.operations_unordered):
                            if (
                                ("add" in aux_op.op_type or "mut" in aux_op.op_type)
                                and self.model_ops1[aux_op.i] in offspring_serialised
                                and (
                                    (
                                        "_sep" not in self.model_ops1[aux_op.i].operation.name
                                        and (not aux_op.i_swapped)
                                        or (
                                            "_sep" in self.model_ops1[aux_op.i].operation.name
                                            and aux_op.i_swapped
                                        )
                                    )
                                    and "_end" not in self.model_ops1[aux_op.i].operation.name
                                )
                                and (
                                    aux_idx - closing_op > 0
                                    or len(self.model_ops1[aux_op.i].children) < 2
                                )
                            ):
                                op_distances1 += [aux_idx - closing_op]
                            else:
                                op_distances1 += [np.inf]
                        op_distances1 = [
                            op_dist
                            if depths[op_idx] == depths[self.operations_unordered.index(op)]
                            and op_dist != 0
                            else np.inf
                            for op_idx, op_dist in enumerate(op_distances1)
                        ]
                        selected_idx1 = np.argmin([abs(op_dist1) for op_dist1 in op_distances1])
                        after_layer[1] = op_distances1[selected_idx1] < 0
                        op_distances1 = abs(op_distances1[selected_idx1])
                    except BaseException:
                        selected_idx1 = -1
                        op_distances1 = np.inf
                        after_layer[1] = False
                    chosen_pos = np.argmin([op_distances0, op_distances1])
                    after_layer = after_layer[chosen_pos]
                    final_layer = [
                        self.model_ops2[self.operations_unordered[selected_idx0].j],
                        self.model_ops1[self.operations_unordered[selected_idx1].i],
                    ][chosen_pos]
                    final_layer_id = final_layer.id
                    try:
                        split_pos = offspring_serialised.index(final_layer)
                    except BaseException:
                        split_pos = len(offspring_serialised)
                    if split_pos == len(offspring_serialised):
                        end_at_id = -1
                    elif after_layer and split_pos < len(offspring_serialised):
                        split_pos += 1
                        while (
                            offspring_serialised[split_pos] not in self.model_ops1 + self.model_ops2
                        ):
                            split_pos += 1
                            if split_pos == len(offspring_serialised):
                                break
                        if split_pos < len(offspring_serialised):
                            end_at_id = offspring_serialised[split_pos].id
                        else:
                            end_at_id = -1
                    else:
                        end_at_id = offspring_serialised[split_pos].id
                    found_parent_sequential = False
                    if split_pos == len(offspring_serialised):
                        while not found_parent_sequential:
                            if not node.is_root():
                                if node.parent.operation.name == "sequential" and end_at_id not in [
                                    aux_node.id for aux_node in node.serialise()
                                ]:
                                    node = node.parent
                                else:
                                    found_parent_sequential = True
                            else:
                                found_parent_sequential = True
                    else:
                        while not found_parent_sequential:
                            if not node.is_root():
                                if node.parent.operation.name == "sequential" and end_at_id not in [
                                    aux_node.id for aux_node in node.serialise()
                                ]:
                                    node = node.parent
                                else:
                                    found_parent_sequential = True
                            else:
                                found_parent_sequential = True
                    if node.operation.name == "sequential":
                        try:
                            node = self.split_sequentials(node, starting_node.id).children[1]
                        except BaseException:
                            node = node
                    else:
                        node = node
                    if node.operation.name == "sequential":
                        if end_at_id != -1 and end_at_id in [
                            aux_node.id for aux_node in node.serialise()
                        ]:
                            node2 = self.split_sequentials(node, end_at_id).children[0]
                        else:
                            node2 = node
                    else:
                        node2 = node
                    depths = self.calculate_depth_of_path(offspring_serialised)
                    after_layer = [0, 0]
                    closing_op = [
                        in_idx
                        for in_idx, in_op in enumerate(self.operations_unordered)
                        if in_op.node1_id == op.node1_id and "_sep" in in_op.op_type
                    ][0]
                    try:
                        op_distances0 = []
                        for aux_idx, aux_op in enumerate(self.operations_unordered):
                            if (
                                ("rem" in aux_op.op_type or "mut" in aux_op.op_type)
                                and self.model_ops2[aux_op.j] in offspring_serialised
                                and (
                                    (
                                        "_sep" not in self.model_ops2[aux_op.j].operation.name
                                        and (not aux_op.j_swapped)
                                        or (
                                            "_sep" in self.model_ops2[aux_op.j].operation.name
                                            and aux_op.j_swapped
                                        )
                                    )
                                    and "_end" not in self.model_ops2[aux_op.j].operation.name
                                )
                                and (
                                    aux_idx - closing_op > 0
                                    or len(self.model_ops2[aux_op.j].children) < 2
                                )
                            ):
                                op_distances0 += [aux_idx - closing_op]
                            else:
                                op_distances0 += [np.inf]
                        op_distances0 = [
                            op_dist
                            if depths[op_idx] == depths[self.operations_unordered.index(op)]
                            and op_dist != 0
                            else np.inf
                            for op_idx, op_dist in enumerate(op_distances0)
                        ]
                        selected_idx0 = np.argmin([abs(op_dist0) for op_dist0 in op_distances0])
                        after_layer[0] = op_distances0[selected_idx0] < 0
                        op_distances0 = abs(op_distances0[selected_idx0])
                    except BaseException:
                        selected_idx0 = -1
                        op_distances0 = np.inf
                        after_layer[0] = False
                    try:
                        op_distances1 = []
                        for aux_idx, aux_op in enumerate(self.operations_unordered):
                            if (
                                ("add" in aux_op.op_type or "mut" in aux_op.op_type)
                                and self.model_ops1[aux_op.i] in offspring_serialised
                                and (
                                    (
                                        "_sep" not in self.model_ops1[aux_op.i].operation.name
                                        and (not aux_op.i_swapped)
                                        or (
                                            "_sep" in self.model_ops1[aux_op.i].operation.name
                                            and aux_op.i_swapped
                                        )
                                    )
                                    and "_end" not in self.model_ops1[aux_op.i].operation.name
                                )
                                and (
                                    aux_idx - closing_op > 0
                                    or len(self.model_ops1[aux_op.i].children) < 2
                                )
                            ):
                                op_distances1 += [aux_idx - closing_op]
                            else:
                                op_distances1 += [np.inf]
                        op_distances1 = [
                            op_dist
                            if depths[op_idx] == depths[self.operations_unordered.index(op)]
                            and op_dist != 0
                            else np.inf
                            for op_idx, op_dist in enumerate(op_distances1)
                        ]
                        selected_idx1 = np.argmin([abs(op_dist1) for op_dist1 in op_distances1])
                        after_layer[1] = op_distances1[selected_idx1] < 0
                        op_distances1 = abs(op_distances1[selected_idx1])
                    except BaseException:
                        selected_idx1 = -1
                        op_distances1 = np.inf
                        after_layer[1] = False
                    chosen_pos = np.argmin([op_distances0, op_distances1])
                    split_pos = [selected_idx0, selected_idx1][chosen_pos]
                    after_layer = after_layer[chosen_pos]
                    final_layer_id = [
                        self.model_ops2[self.operations_unordered[selected_idx0].j].id,
                        self.model_ops1[self.operations_unordered[selected_idx1].i].id,
                    ][chosen_pos]
                    try:
                        split_pos = [
                            idx
                            for idx, node in enumerate(offspring_serialised)
                            if node.id == final_layer_id
                        ][0]
                    except BaseException:
                        split_pos = len(offspring_serialised)
                    if split_pos == len(offspring_serialised):
                        end_at_id = -1
                    elif after_layer and split_pos < len(offspring_serialised):
                        jump_ids = [
                            aux_node.id for aux_node in offspring_serialised[split_pos].serialise()
                        ]
                        split_pos += 1
                        while (
                            offspring_serialised[split_pos] not in self.model_ops1 + self.model_ops2
                            or offspring_serialised[split_pos].id in jump_ids
                        ):
                            split_pos += 1
                            if split_pos == len(offspring_serialised):
                                break
                        if split_pos < len(offspring_serialised):
                            end_at_id = offspring_serialised[split_pos].id
                        else:
                            end_at_id = -1
                    else:
                        end_at_id = offspring_serialised[split_pos].id
                    found_parent_sequential = False
                    if split_pos == len(offspring_serialised):
                        while not found_parent_sequential:
                            if not node2.is_root():
                                if (
                                    node2.parent.operation.name == "sequential"
                                    and end_at_id
                                    not in [aux_node.id for aux_node in node2.serialise()]
                                ):
                                    node2 = node2.parent
                                else:
                                    found_parent_sequential = True
                            else:
                                found_parent_sequential = True
                    else:
                        while not found_parent_sequential:
                            if not node2.is_root():
                                if (
                                    node2.parent.operation.name == "sequential"
                                    and end_at_id
                                    not in [aux_node.id for aux_node in node2.serialise()]
                                ):
                                    node2 = node2.parent
                                else:
                                    found_parent_sequential = True
                            else:
                                found_parent_sequential = True
                    if not node2.is_root():
                        while node2 not in node2.parent.children and (not node2.is_root()):
                            node2 = node2.parent
                    if node2.operation.name == "sequential" and split_pos < len(
                        offspring_serialised
                    ):
                        node2 = self.split_sequentials(node2, end_at_id)
                    else:
                        node2 = node2
                    if self.verbose:
                        print(
                            ">>>Parallelizing modules",
                            colored(str(node2.children[0]), "red"),
                            "and",
                            colored(str(node2.children[1]), "red"),
                            "using",
                            colored(node1.operation.name, "green"),
                        )
                    parent_node = node2.parent
                    if not node2.is_root():
                        parent_node.children[parent_node.children.index(node2)] = node1
                    node1.parent = parent_node
                    node1.children[1] = node2.children[0]
                    node1.children[2] = node2.children[1]
                    node2.children[0].parent = node1
                    node2.children[1].parent = node1
                    offspring = node1.get_root()
            elif "add" in op.op_type:
                for node in self.model_ops1:
                    if node.id == self.model_ops1[op.i].id:
                        add_node = copy.deepcopy(node)
                        break
                if not add_node.id == self.model_ops1[op.i].id:
                    raise Exception(
                        "Node",
                        self.model_ops1[op.i].id,
                        "not found from model 1 when attempting module addition",
                    )
                if self.model_ops2[op.j].id == -1:
                    offspring_node = offspring
                    node1 = add_node
                    node2 = offspring_node
                    after_layer = False
                else:
                    offspring_serialised = offspring.serialise()
                    depths = self.calculate_depth_of_path(offspring_serialised)
                    after_layer = [0, 0]
                    closing_op = self.operations_unordered.index(op)
                    try:
                        op_distances0 = []
                        for aux_idx, aux_op in enumerate(self.operations_unordered):
                            if (
                                "rem" in aux_op.op_type or "mut" in aux_op.op_type
                            ) and self.model_ops2[aux_op.j] in offspring_serialised:
                                op_distances0 += [aux_idx - closing_op]
                            else:
                                op_distances0 += [np.inf]
                        op_distances0 = [
                            op_dist
                            if depths[op_idx] == depths[self.operations_unordered.index(op)]
                            and op_dist != 0
                            else np.inf
                            for op_idx, op_dist in enumerate(op_distances0)
                        ]
                        selected_idx0 = np.argmin([abs(op_dist0) for op_dist0 in op_distances0])
                        op_distances0 = op_distances0[selected_idx0]
                        after_layer[0] = op_distances0 < 0
                    except BaseException:
                        selected_idx0 = -1
                        op_distances0 = np.inf
                        after_layer[0] = False
                    try:
                        op_distances1 = []
                        for aux_idx, aux_op in enumerate(self.operations_unordered):
                            if (
                                "add" in aux_op.op_type or "mut" in aux_op.op_type
                            ) and self.model_ops1[aux_op.i] in offspring_serialised:
                                op_distances1 += [aux_idx - closing_op]
                            else:
                                op_distances1 += [np.inf]
                        op_distances1 = [
                            op_dist
                            if depths[op_idx] == depths[self.operations_unordered.index(op)]
                            and op_dist != 0
                            else np.inf
                            for op_idx, op_dist in enumerate(op_distances1)
                        ]
                        selected_idx1 = np.argmin([abs(op_dist1) for op_dist1 in op_distances1])
                        op_distances1 = op_distances1[selected_idx1]
                        after_layer[1] = op_distances1 < 0
                    except BaseException:
                        selected_idx1 = -1
                        op_distances1 = np.inf
                        after_layer[1] = False
                    if np.isinf(np.min([op_distances0, op_distances1])):
                        node2 = add_node
                        offspring_node = offspring
                        node1 = offspring_node
                        after_layer = True
                    else:
                        chosen_pos = np.argmin([abs(op_distances0), abs(op_distances1)])
                        after_layer = after_layer[chosen_pos]
                        offspring_swap = [
                            self.operations_unordered[selected_idx0].j_swapped,
                            self.operations_unordered[selected_idx1].i_swapped,
                        ][chosen_pos]
                        final_layer = [
                            self.model_ops2[self.operations_unordered[selected_idx0].j],
                            self.model_ops1[self.operations_unordered[selected_idx1].i],
                        ][chosen_pos]
                        offspring_node = offspring_serialised[
                            offspring_serialised.index(final_layer)
                        ]
                        if after_layer:
                            if (
                                "_end" in final_layer.operation.name
                                or len(offspring_node.children) <= 2
                            ):
                                node1 = offspring_node
                                node2 = add_node
                                if self.verbose:
                                    print(
                                        ">>>Adding",
                                        colored(str(node2), "green"),
                                        "after",
                                        colored(str(node1), "red"),
                                    )
                            elif (
                                "_sep" in final_layer.operation.name
                                and (not offspring_swap)
                                or ("_sep" not in final_layer.operation.name and offspring_swap)
                            ):
                                node1 = add_node
                                offspring_node = offspring_node.children[2]
                                node2 = offspring_node
                                if self.verbose:
                                    print(
                                        ">>>Adding",
                                        colored(str(node1), "green"),
                                        "before",
                                        colored(str(node2), "red"),
                                    )
                            else:
                                node1 = add_node
                                offspring_node = offspring_node.children[1]
                                node2 = offspring_node
                                if self.verbose:
                                    print(
                                        ">>>Adding",
                                        colored(str(node1), "green"),
                                        "before",
                                        colored(str(node2), "red"),
                                    )
                        elif "_end" in final_layer.operation.name:
                            offspring_node = offspring_node.children[-2]
                            node1 = offspring_node
                            node2 = add_node
                            if self.verbose:
                                print(
                                    ">>>Adding",
                                    colored(str(node2), "green"),
                                    "after",
                                    colored(str(node1), "red"),
                                )
                        elif (
                            "_sep" in final_layer.operation.name
                            and (not offspring_swap)
                            or ("_sep" not in final_layer.operation.name and offspring_swap)
                        ):
                            offspring_node = offspring_node.children[1]
                            node1 = offspring_node
                            node2 = add_node
                            if self.verbose:
                                print(
                                    ">>>Adding",
                                    colored(str(node2), "green"),
                                    "after",
                                    colored(str(node1), "red"),
                                )
                        else:
                            node1 = add_node
                            node2 = offspring_node
                            if self.verbose:
                                print(
                                    ">>>Adding",
                                    colored(str(node1), "green"),
                                    "before",
                                    colored(str(node2), "red"),
                                )
                sequential_node = DerivationTreeNode(
                    0,
                    level=node1.level,
                    parent=None,
                    input_params=node1.input_params,
                    depth=node1.depth,
                    limiter=node1.limiter,
                    operation=Operation(
                        name="sequential",
                        build=einspace.build_sequential_module,
                        infer=einspace.infer_sequential_module,
                        valid=einspace.valid_sequential_module,
                        inherit=[einspace.inherit_first_child, einspace.inherit_other_child],
                        give_back=[einspace.give_back_default, einspace.give_back_default],
                        type="nonterminal",
                        child_levels=["module", "module"],
                    ),
                )
                self.update_id(sequential_node)
                sequential_node.parent = offspring_node.parent
                if not offspring_node.is_root():
                    offspring_node.parent.children[
                        offspring_node.parent.children.index(offspring_node)
                    ] = sequential_node
                sequential_node.children = [node1, node2]
                node1.parent = sequential_node
                node2.parent = sequential_node
                offspring = sequential_node.get_root()
            if self.verbose and "wrap_end" not in op.op_type and ("wrap_sep" not in op.op_type):
                print("", colored(offspring, "light_grey"), "\n")
        return offspring

    def calculate_depth_of_path(self, serialised_model, operations=None):
        node_ids = [node.id for node in serialised_model]
        node_ids1 = [node.id for node in self.model_ops1]
        node_ids2 = [node.id for node in self.model_ops2]
        if operations == None:
            operations = self.operations_unordered
        next_depth = 1
        stack = [0]
        ids = []
        for aux_idx, aux_op in enumerate(operations):
            ids.append(stack[-1])
            if (
                ("rem_wrap" in aux_op.op_type or "mut_wrap" in aux_op.op_type)
                and aux_op.node2_id in node_ids
                and (len(self.model_ops2[node_ids2.index(aux_op.node2_id)].children) == 4)
            ):
                if "end" in aux_op.op_type:
                    stack.pop()
                    stack.pop()
                else:
                    stack.append(next_depth)
                    next_depth += 1
            elif (
                ("add_wrap" in aux_op.op_type or "mut_wrap" in aux_op.op_type)
                and aux_op.node1_id in node_ids
                and (len(self.model_ops1[node_ids1.index(aux_op.node1_id)].children) == 4)
            ):
                if "end" in aux_op.op_type:
                    stack.pop()
                    stack.pop()
                else:
                    stack.append(next_depth)
                    next_depth += 1
        return ids


def raw_crossover(parent1, parent2, skewness=0, limiter=None):
    if limiter is None:
        limiter = parent1.limiter
    if parent1.serialise() == parent2.serialise():
        same = True
        for op1, op2 in zip(parent1.serialise(), parent2.serialise()):
            if op1.operation.name != op2.operation.name:
                same = False
                break
        if same:
            return (parent1, [], [], 0, 0, 0)
    matrix = Alignment(parent1, parent2, limiter=limiter)
    operations = matrix.nontrivial_ops
    if len(operations) == 0:
        return (parent1, [], [], 0, 0, 0)
    else:
        selected_ops = select_operations(operations, skewness=skewness)
        child = matrix.generate_offspring(selected_ops)
        distance_between_parents = matrix.distance
        distance_to_parent2 = sum([op.value for op in selected_ops])
        distance_to_parent1 = distance_between_parents - distance_to_parent2
        child = copy.deepcopy(child)
        selected_ops = copy.deepcopy(selected_ops)
        operations = copy.deepcopy(operations)
        del matrix
        return (
            child,
            selected_ops,
            operations,
            distance_to_parent1,
            distance_to_parent2,
            distance_between_parents,
        )
