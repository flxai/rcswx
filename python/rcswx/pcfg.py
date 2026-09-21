# Derived from einsearch 7e713c7951397a6b12bd57638409774a74381746.
# Copyright (c) 2024 Adri Gómez Martín, Felix Möller, Linus Ericsson, Aaron Klein.
# Distributed under the MIT license; see LICENSE.einsearch.
"""Reference grammar option validation used when reconstructing offspring."""

import sys
from copy import deepcopy
from random import choices

from rich import print


class OutOfOptionsError(Exception):
    pass


class PCFG:
    def __init__(self, grammar, limiter):
        self.grammar = grammar
        self.limiter = limiter

    def sample(self, node, verbose=False):
        available_options, available_probs = self.get_available_options(node, verbose)
        node.available_rules = {"options": available_options, "probs": available_probs}
        options, probs = self.filter_options(node, available_options, available_probs, verbose)
        if verbose:
            print(
                f"Filtered options for node {node.id} at level {node.level}: {[op.name for op in options]}"
            )
        if verbose:
            print(
                f"Full list of options at node {node.id}: {[op.name for op in available_options]}"
            )
        if len(options) > 0:
            operation = choices(options, weights=probs, k=1)[0]
            if verbose:
                print(
                    f"Sampled operation {operation.name} for node {node.id} at level {node.level}"
                )
        else:
            raise OutOfOptionsError(f"Out of options for node {node.id} at level {node.level}")
        return operation

    def get_available_options(self, node, verbose=False):
        if node.level not in self.grammar:
            return None
        if node.available_rules == None:
            available_options = deepcopy(self.grammar[node.level]["options"])
            available_probs = deepcopy(self.grammar[node.level]["probs"])
        else:
            available_options = node.available_rules["options"]
            available_probs = node.available_rules["probs"]
        return (available_options, available_probs)

    def filter_options(self, node, options, probs, verbose=False):
        indices = [i for i, op in enumerate(options) if op.valid(node)]
        indices_to_remove = []
        for i in indices:
            if not self.check_sequential_closure(node, options, i):
                indices_to_remove.append(i)
        indices = [i for i in indices if i not in indices_to_remove]
        success = self.limiter.check(node, verbose)
        if not success:
            if any([op.name == "computation" for op in options]):
                indices = [i for i in indices if options[i].name == "computation"]
            if verbose:
                print(
                    f"Filtered options for node {node.id} at level {node.level}: {[options[i].name for i in indices]}"
                )
        options, probs = ([options[i] for i in indices], [probs[i] for i in indices])
        probs = [p / sum(probs) for p in probs]
        return (options, probs)

    def check_sequential_closure(self, node, options, i):

        def is_seq_k_parent(node):
            if node.is_root():
                return None
            elif node.parent.operation.name.startswith("sequential("):
                return node.parent
            else:
                return is_seq_k_parent(node.parent)

        def is_final_child(parent, node):
            if node in parent.children:
                if node == parent.children[-1] and node.level != "module":
                    return True
                else:
                    return False
            elif parent.children:
                if is_final_child(parent.children[-1], node):
                    return True
            return False

        option = options[i]
        seq_k_parent = is_seq_k_parent(node)
        if seq_k_parent and is_final_child(seq_k_parent, node):
            k = int(seq_k_parent.operation.name[-2])
            return self.check_sequential_closure_inner(seq_k_parent, node, k, option)
        else:
            return True

    def check_sequential_closure_inner(self, root, current_node, k, operation):
        current_node.operation = operation
        print(f"Checking repeatable {k} times: {root}, {operation.name}")

        def infer(node):
            if not node.is_root():
                node.inherit_input_params()
            node.output_params = node.operation.infer(node)
            for child in node.children:
                infer(child)
            node.give_back_output_params()

        if not root.is_root():
            root.inherit_input_params()
        original_root_input_params = root.input_params
        original_current_node_input_params = current_node.input_params
        repeatable = True
        for _ in range(k):
            try:
                infer(root.children[0])
                root.input_params = root.output_params
                print(f"Input params: {root.input_params}")
                if 0 in root.input_params["shape"]:
                    raise ValueError(f"Invalid input params: {root.input_params}")
            except Exception as e:
                print(f"Error: {e}")
                print(f"Failed to repeat {k} times: {root}, {operation.name}")
                repeatable = False
                break
        if repeatable:
            print(f"Successfully repeated {k} times: {root}, {operation.name}")
        root.input_params = original_root_input_params
        current_node.input_params = original_current_node_input_params
        current_node.operation = None
        return repeatable

    def __repr__(self):
        result = ["Grammar:"]
        for level, rules in self.grammar.items():
            result.append(f"\t{level}:")
            for i, (rule, prob) in enumerate(zip(rules["options"], rules["probs"])):
                result.append(f"\t\t{rule.name:<20}(p={prob})")
        return "\n".join(result)

    def __str__(self):
        result = ["Grammar:"]
        for level, rules in self.grammar.items():
            result.append(f"\t{level}:")
            for i, (rule, prob) in enumerate(zip(rules["options"], rules["probs"])):
                result.append(f"\t\t{rule.name:<26}(p={prob:.3f})")
        return "\n".join(result)

    def __sizeof__(self):
        return sum(map(sys.getsizeof, self.__dict__.values()))
