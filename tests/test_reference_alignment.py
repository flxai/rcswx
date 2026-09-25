import copy
import itertools
import random

import numpy as np
import pytest
from rcswx import distance
from rcswx.genotype import DerivationTreeNode, Operation
from rcswx.recursive import Alignment, raw_crossover
from rcswx.sampling import valid_combinations

from tests.reference.original import chain, limits, load, operation_record, tree, tree_record


def routing(inner, before="identity", after="identity"):
    return ("routing", before, inner, after)


def branching(arity, inner, other=None):
    children = [inner]
    if arity == 2:
        children.append(chain(["relu"]) if other is None else other)
    return (f"branching({arity})", f"clone({arity})", *children, f"add({arity})")


CASES = [
    (chain(["identity"]), chain(["relu"])),
    (chain(["linear(16)"]), chain(["linear(x2)"])),
    (
        chain(["linear(16)", "linear(32)"] * 4 + ["linear(4)"]),
        chain(["linear(32)", "linear(16)"] * 4 + ["linear(4)"]),
    ),
    (("sequential(4)", chain(["linear(4)"])), chain(["linear(4)"])),
    (("sequential(8)", chain(["linear(x2)"])), ("sequential(4)", chain(["linear(16)"]))),
    (routing(chain(["identity"]), "perm(0,1,2)"), routing(chain(["identity"]), "perm(0,2,1)")),
    (routing(chain(["identity", "relu"])), chain(["relu", "identity"])),
    (chain(["relu", "identity"]), routing(chain(["identity", "relu"]))),
    (routing(routing(chain(["identity"]))), routing(chain(["relu"]))),
    (branching(4, chain(["identity"])), branching(8, chain(["relu"]))),
    (branching(2, chain(["identity"])), branching(2, chain(["relu"]))),
    (branching(2, chain(["identity"])), chain(["identity", "relu"])),
    (chain(["identity", "relu"]), branching(2, chain(["identity"]))),
]


BRANCH_BOUNDARY_CASES = [
    pytest.param(
        (
            branching(2, chain(["identity"])),
            branching(2, chain(["relu"]), chain(["identity"])),
        ),
        0,
        id="reversed",
    ),
    pytest.param(
        (
            branching(2, chain(["identity", "norm"])),
            branching(2, chain(["relu"]), chain(["identity", "norm"])),
        ),
        0,
        id="unequal-branches",
    ),
    pytest.param(
        (
            ("sequential", chain(["identity"]), branching(2, chain(["identity"]))),
            (
                "sequential",
                chain(["norm"]),
                branching(2, chain(["relu"]), chain(["identity"])),
            ),
        ),
        0.5,
        id="nonzero-prefix",
    ),
    pytest.param(
        (
            branching(2, branching(2, chain(["identity"])), chain(["norm"])),
            branching(2, chain(["norm"]), branching(2, chain(["relu"]), chain(["identity"]))),
        ),
        0,
        id="nested",
    ),
]
CASES.extend(case.values[0] for case in BRANCH_BOUNDARY_CASES)


def make_pair(descriptions, owned):
    guard = limits()
    kwargs = {"node_type": DerivationTreeNode, "operation_type": Operation} if owned else {}
    return tuple(tree(description, limiter=guard, **kwargs) for description in descriptions), guard


def align_outcome(descriptions, owned, collapse_corners=False):
    parents, guard = make_pair(descriptions, owned)
    factory = Alignment if owned else load().algorithm.AlignmentMatrixRecursive
    try:
        result = factory(*parents, limiter=guard, collapse_corners=collapse_corners)
    except Exception as error:
        return (type(error).__name__, tuple(tree_record(parent) for parent in parents)), None
    return (
        # Compare every ordered history, not only the selected path.
        float(result.distance),
        tuple(operation_record(op) for op in result.operations),
        tuple(operation_record(op) for op in result.nontrivial_ops),
        tuple(tree_record(parent) for parent in parents),
        tuple(
            tuple(map(operation_record, path))
            for path in (result.paths if owned else result.matrix[-1][-1].paths)
        ),
    ), result


@pytest.mark.parametrize("descriptions", CASES)
@pytest.mark.parametrize("collapse_corners", [False, True])
def test_recursive_cost_witness_dependencies_and_parent_effects(descriptions, collapse_corners):
    expected, _ = align_outcome(descriptions, False, collapse_corners)
    actual, _ = align_outcome(descriptions, True, collapse_corners)
    assert actual == expected


@pytest.mark.parametrize("descriptions,expected_distance", BRANCH_BOUNDARY_CASES)
@pytest.mark.parametrize("collapse_corners", [False, True])
@pytest.mark.parametrize("reverse_parents", [False, True])
def test_branch_boundary_costs_and_histories(
    descriptions, expected_distance, collapse_corners, reverse_parents
):
    if reverse_parents:
        descriptions = descriptions[::-1]
    expected, reference = align_outcome(descriptions, False, collapse_corners)
    actual, native = align_outcome(descriptions, True, collapse_corners)
    assert actual == expected
    assert reference.distance == native.distance == expected_distance
    for path in native.paths:
        assert sum(operation.value for operation in path) == expected_distance


def test_ordered_ties_preserve_reference_edit_domain():
    expected, reference = align_outcome(CASES[2], False)
    actual, native = align_outcome(CASES[2], True)
    assert actual == expected
    assert reference.distance == native.distance == 2
    assert [op.op_type for op in native.nontrivial_ops] == [
        op.op_type for op in reference.nontrivial_ops
    ]
    assert len(native.nontrivial_ops) == 2


def operation_tree(node):
    return node.operation.name, tuple(map(operation_tree, node.children))


def test_swap_reordering_preserves_nested_source_topology():
    descriptions = (
        (
            "branching(2)",
            "group(2,2)",
            (
                "branching(2)",
                "group(2,3)",
                chain(["linear(32)"]),
                routing(chain(["identity"]), "perm(0,1,3,2)"),
                "cat(2,3)",
            ),
            chain(["linear(64)"]),
            "cat(2,3)",
        ),
        (
            "sequential",
            chain(["linear(32)"]),
            (
                "sequential",
                (
                    "branching(2)",
                    "group(2,2)",
                    chain(["identity"]),
                    chain(["relu"]),
                    "cat(2,2)",
                ),
                chain(["linear(32)"]),
            ),
        ),
    )
    expected, _ = align_outcome(descriptions, False)
    actual, native = align_outcome(descriptions, True)
    assert actual[0] == expected[0]
    # Physical source coordinates change application order and dependencies,
    # not the retained alignment histories or their elementary edits.
    assert tuple(tuple(op[:11] for op in path) for path in actual[4]) == tuple(
        tuple(op[:11] for op in path) for path in expected[4]
    )
    child = native.generate_offspring(native.nontrivial_ops)
    assert operation_tree(child) == operation_tree(native.model1)


def test_insertion_after_a_swapped_wrapper_stays_outside_its_branches():
    wrapper = branching(2, chain(["identity"]), chain(["identity", "relu"]))
    descriptions = (("sequential", wrapper, chain(["sigmoid"])), wrapper)
    expected, _ = align_outcome(descriptions, False)
    actual, native = align_outcome(descriptions, True)
    assert actual == expected
    before = tree_record(native.model2)
    child = native.generate_offspring(native.nontrivial_ops)
    assert child.operation.name == "sequential"
    assert len(child.children) == 2
    assert tree_record(child.children[0]) == before
    assert operation_tree(child.children[1]) == ("computation", (("sigmoid", ()),))


@pytest.mark.parametrize(
    "descriptions",
    [CASES[index] for index in (0, 6, 7, 8, 9, 10, 11, 12)]
    + [case.values[0] for case in BRANCH_BOUNDARY_CASES]
    + [
        pytest.param(
            (
                (
                    "sequential",
                    branching(4, chain(["identity"])),
                    chain(["norm", "relu"]),
                ),
                (
                    "sequential",
                    routing(chain(["relu"])),
                    chain(["norm", "relu"]),
                ),
            ),
            id="failed-prefix-split-preserves-effects",
        ),
        pytest.param(
            (
                (
                    "sequential",
                    routing(chain(["linear(16)"]), "perm(0,2,1,3)", "perm(0,1,3,2)"),
                    (
                        "branching(4)",
                        "clone(4)",
                        routing(chain(["pos_enc"]), "identity", "perm(0,2,1)"),
                        "cat(4,2)",
                    ),
                ),
                routing(chain(["linear(512)"]), "im2col(1,1,0)", "perm(0,2,1)"),
            ),
            id="nearest-wrapper-anchor-compares-both-parents",
        ),
    ],
)
def test_every_valid_tiny_selection_preserves_offspring_and_application_failures(descriptions):
    expected, reference = align_outcome(descriptions, False)
    actual, native = align_outcome(descriptions, True)
    assert actual == expected
    if reference is None:
        assert native is None
        return
    masks, _ = valid_combinations(reference.nontrivial_ops)
    for mask in masks:
        # Application changes operation dependencies and consumes new node IDs.
        old = copy.deepcopy(reference)
        new = copy.deepcopy(native)
        try:
            expected_child = old.generate_offspring(
                [op for bit, op in zip(mask, old.nontrivial_ops) if bit == "1"]
            )
        except Exception as error:
            with pytest.raises(type(error)):
                new.generate_offspring(
                    [op for bit, op in zip(mask, new.nontrivial_ops) if bit == "1"]
                )
        else:
            actual_child = new.generate_offspring(
                [op for bit, op in zip(mask, new.nontrivial_ops) if bit == "1"]
            )
            assert tree_record(actual_child) == tree_record(expected_child)


@pytest.mark.parametrize(
    "descriptions",
    [
        pytest.param(
            (chain(["sigmoid", "relu"]), chain(["identity", "relu"])),
            id="parent-child-list",
        ),
        pytest.param(
            (
                routing(chain(["relu"]), "perm(0,1,2)"),
                routing(chain(["relu"]), "perm(0,2,1)"),
            ),
            id="wrapper-child-list",
        ),
    ],
)
def test_mutation_preserves_parent_child_list_aliases(descriptions):
    observations = []
    for owned in (False, True):
        parents, guard = make_pair(descriptions, owned)
        factory = Alignment if owned else load().algorithm.AlignmentMatrixRecursive
        matrix = factory(*parents, limiter=guard)
        matrix.model1.children_view = matrix.model1.children
        matrix.model2.children_view = matrix.model2.children
        child = matrix.generate_offspring(matrix.nontrivial_ops)
        observations.append(
            (
                tree_record(child),
                child.children_view is child.children,
                tuple(map(tree_record, child.children_view)),
            )
        )
    assert observations[0][1] is True
    assert observations[1] == observations[0]


def test_tiny_ordered_history_space():
    sequences = [
        chain(names)
        for size in range(1, 4)
        for names in itertools.product(("identity", "relu"), repeat=size)
    ]
    for first, second in itertools.product(sequences, repeat=2):
        expected, _ = align_outcome((first, second), False)
        actual, _ = align_outcome((first, second), True)
        assert actual == expected


def test_branch_alignment_does_not_consume_selector_rng():
    np.random.seed(47)
    before = np.random.get_state()
    expected, _ = align_outcome(CASES[10], False)
    actual, _ = align_outcome(CASES[10], True)
    assert expected == actual
    assert expected[0] == 0.5
    after = np.random.get_state()
    assert before[0] == after[0] and np.array_equal(before[1], after[1]) and before[2:] == after[2:]


@pytest.mark.parametrize("descriptions", CASES)
def test_raw_crossover_ownership_edits_parent_effects_and_rng(descriptions):
    reference = load().algorithm.recursive_constrained_smith_waterman_crossover
    for seed in range(8):
        results = []
        for owned, function in ((False, reference), (True, raw_crossover)):
            parents, guard = make_pair(descriptions, owned)
            np.random.seed(seed)
            try:
                kwargs = {"sampler": "reference"} if owned else {}
                child, selected, operations, distance1, distance2, between = function(
                    *parents, limiter=guard, **kwargs
                )
                outcome = (
                    tree_record(child),
                    child is parents[0],
                    child is parents[1],
                    tuple(map(operation_record, selected)),
                    tuple(map(operation_record, operations)),
                    distance1,
                    distance2,
                    between,
                )
            except Exception as error:
                outcome = type(error).__name__
            state = np.random.get_state()
            results.append(
                (outcome, tuple(map(tree_record, parents)), state[0], state[1].tobytes(), state[2:])
            )
        assert results[0] == results[1]


def without_empty_dependency_pairs(operation):
    empty = {
        index
        for index, pair in enumerate(zip(operation[11], operation[12], strict=False))
        if pair == ((), ())
    }
    return (
        *operation[:11],
        tuple(value for index, value in enumerate(operation[11]) if index not in empty),
        tuple(value for index, value in enumerate(operation[12]) if index not in empty),
    )


def corrected_branch_constraints(outcome, *, repair_reference=False):
    """Compare the unchanged oracle exactly, apart from the known branch-addition bug."""
    if isinstance(outcome[0], str):
        return outcome
    paths = []
    for path in outcome[4]:
        corrected = []
        for index, operation in enumerate(path):
            if repair_reference and operation[1] == "add_wrap":
                boundaries = [
                    position
                    for position in range(index + 1, len(path))
                    if path[position][2] == operation[2]
                    and path[position][1] in ("add_wrap_sep", "add_wrap_end")
                ]
                if len(boundaries) == 2:
                    separator, end = boundaries
                    branches = [
                        [op for op in section if "wrap" not in op[1]]
                        for section in (path[index : separator + 1], path[separator + 1 : end])
                    ]
                    additions = tuple(
                        tuple(op[0] for op in branch if op[1].startswith("add"))
                        for branch in branches
                    )
                    requires_addition = [
                        bool(adds) and not any(op[1].startswith(("mut", "rem")) for op in branch)
                        for branch, adds in zip(branches, additions, strict=True)
                    ]
                    if any(requires_addition):
                        # The reference appends requirements for BOTH branches
                        # when only one has additions without retained material.
                        assert operation[11][-2:] == additions
                        assert operation[12][-2:] == ((operation[0],), (operation[0],))
                        required = tuple(
                            adds
                            for adds, needed in zip(additions, requires_addition, strict=True)
                            if needed
                        )
                        operation = (
                            *operation[:11],
                            operation[11][:-2] + required,
                            operation[12][:-2] + ((operation[0],),) * len(required),
                        )
            corrected.append(without_empty_dependency_pairs(operation))
        paths.append(tuple(corrected))
    selected = {operation[0]: operation for operation in paths[0]}

    def selected_operation(operation):
        if repair_reference:
            return (*operation[:11], *selected[operation[0]][11:])
        return without_empty_dependency_pairs(operation)

    return (
        outcome[0],
        tuple(map(selected_operation, outcome[1])),
        tuple(map(selected_operation, outcome[2])),
        outcome[3],
        tuple(paths),
    )


def test_recursive_wrapper_combinations_and_failure_side_effects():
    generator = random.Random(923)

    def description(depth):
        if not depth or generator.random() < 0.35:
            return chain([generator.choice(("identity", "relu", "linear(16)", "linear(x2)"))])
        kind = generator.choice(
            (
                "sequential",
                "sequential(4)",
                "sequential(8)",
                "routing",
                "branching(2)",
                "branching(4)",
                "branching(8)",
            )
        )
        if kind == "sequential":
            return kind, description(depth - 1), description(depth - 1)
        if kind.startswith("sequential("):
            return kind, description(depth - 1)
        if kind == "routing":
            return routing(
                description(depth - 1),
                generator.choice(("identity", "perm(0,2,1)", "im2col(3,1,1)")),
                generator.choice(("identity", "col2im")),
            )
        arity = int(kind[10:-1])
        return branching(arity, description(depth - 1))

    for _ in range(96):
        pair = description(2), description(2)
        for directed in (pair, pair[::-1]):
            for collapse in (False, True):
                expected, _ = align_outcome(directed, False, collapse)
                actual, _ = align_outcome(directed, True, collapse)
                assert corrected_branch_constraints(actual) == corrected_branch_constraints(
                    expected, repair_reference=True
                ), (directed, collapse)


def test_large_and_repeated_node_ids_preserve_reference_equality():
    pair = routing(chain(["identity"])), routing(chain(["relu"]))
    results = []
    for owned in (False, True):
        parents, guard = make_pair(pair, owned)
        for parent in parents:
            for index, node in enumerate(parent.serialise()):
                node.id = 2**100 + index % 3
        factory = Alignment if owned else load().algorithm.AlignmentMatrixRecursive
        try:
            result = factory(*parents, limiter=guard)
            outcome = result.distance, tuple(map(operation_record, result.operations))
        except Exception as error:
            outcome = type(error).__name__
        results.append((outcome, tuple(map(tree_record, parents))))
    assert results[0] == results[1]


@pytest.mark.parametrize(
    "descriptions",
    [(chain(["identity"]), chain(["identity"])), CASES[0], CASES[1], CASES[3], CASES[10]],
)
def test_distance_helper_fast_path_limiter_and_parent_effects(descriptions):
    outcomes = []
    for owned in (False, True):
        parents, guard = make_pair(descriptions, owned)
        guard.limits["memory_crossover"] = 0
        function = distance if owned else load().algorithm.rcswx_distance
        try:
            result = function(*parents)
            outcome = result
        except Exception as error:
            outcome = type(error).__name__
        outcomes.append((outcome, tuple(map(tree_record, parents))))
    assert outcomes[0] == outcomes[1]
