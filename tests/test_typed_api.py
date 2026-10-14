"""Real native behavior: no Python scoring fallback or mocked native results."""

import copy
import itertools
import json

import pytest
from rcswx import _core, compare_typed


def constructor(value, *children, role="item", semantic_type="Term"):
    return {
        "kind": "constructor",
        "semantic_type": semantic_type,
        "role": role,
        "value": value,
        "children": list(children),
    }


def literal(value, *, role="value", semantic_type="number"):
    return {
        "kind": "literal",
        "semantic_type": semantic_type,
        "role": role,
        "value": value,
        "children": [],
    }


def container(*children, kind="list"):
    return {
        "kind": kind,
        "semantic_type": "Sequence",
        "role": "items",
        "value": None,
        "children": list(children),
    }


def tree(description, prefix="n"):
    nodes = []

    def append(description):
        identifier = f"{prefix}{len(nodes)}"
        node = {**description, "id": identifier, "children": []}
        nodes.append(node)
        node["children"] = [append(child) for child in description["children"]]
        return identifier

    return {"version": "rcswx-typed-ordered-v1", "root": append(description), "nodes": nodes}


def compare(first, second):
    result = compare_typed(first, second)
    assert result["version"] == "rcswx-typed-ordered-result-v1"
    assert result["tree_version"] == "rcswx-typed-ordered-v1"
    assert result["policy"] == "typed-unit-v1"
    assert result["backend"] == "rcswx-native-recursive-ordered"
    assert result["direction"] == "second-to-first"
    assert type(result["distance"]) is int
    assert sum(edit["cost"] for edit in result["edits"]) == result["distance"]
    first_ids = [edit["first_id"] for edit in result["edits"] if edit["first_id"] is not None]
    second_ids = [edit["second_id"] for edit in result["edits"] if edit["second_id"] is not None]
    assert sorted(first_ids) == sorted(node["id"] for node in first["nodes"])
    assert sorted(second_ids) == sorted(node["id"] for node in second["nodes"])
    correspondences = []
    for edit in result["edits"]:
        assert edit["cost"] == (0 if edit["operation"] == "match" else 1)
        if edit["operation"] == "insert":
            assert edit["first_id"] is not None and edit["second_id"] is None
        elif edit["operation"] == "delete":
            assert edit["first_id"] is None and edit["second_id"] is not None
        else:
            assert edit["operation"] in {"match", "substitute"}
            assert edit["first_id"] is not None and edit["second_id"] is not None
            correspondences.append({"first_id": edit["first_id"], "second_id": edit["second_id"]})
    assert result["correspondences"] == correspondences
    return result


@pytest.mark.parametrize("name", ["A", "wrap_end", "wrap_sep", "start_node", "branching(2)"])
def test_nullary_constructor_identity_and_value_mutation(name):
    first = tree(constructor(name), "first")
    assert compare(first, tree(constructor(name), "second"))["distance"] == 0
    assert compare(first, tree(constructor("different"), "second"))["distance"] == 1


@pytest.mark.parametrize("changed", range(5))
def test_all_high_arity_children_are_semantic_occurrences(changed):
    children = [literal(index) for index in range(5)]
    first = tree(constructor("A", *children))
    assert len(compare(first, first)["edits"]) == 6
    second = copy.deepcopy(first)
    second["nodes"][changed + 1]["value"] = 100
    assert compare(first, second)["distance"] == 1


@pytest.mark.parametrize("field,value", [("role", "other"), ("semantic_type", "Other")])
def test_incompatible_role_or_type_forbids_substitution(field, value):
    first = tree(constructor("A"))
    second = copy.deepcopy(first)
    second["nodes"][0][field] = value
    result = compare(first, second)
    assert result["distance"] == 2
    assert result["correspondences"] == []
    assert {edit["operation"] for edit in result["edits"]} == {"insert", "delete"}


def test_container_kind_difference_is_not_a_value_mutation():
    first = tree(container(literal(1), literal(2), kind="list"))
    second = tree(container(literal(1), literal(2), kind="tuple"))
    assert compare(first, second)["distance"] == 6


def test_order_and_duplicate_multiplicity_change_the_native_cost():
    first = tree(container(literal(1), literal(2), literal(3)))
    second = tree(container(literal(3), literal(2), literal(1)))
    assert compare(first, second)["distance"] == 2
    duplicate = tree(container(literal(1), literal(1)))
    single = tree(container(literal(1)))
    deletion = compare(single, duplicate)
    insertion = compare(duplicate, single)
    assert deletion["distance"] == insertion["distance"] == 1
    assert [edit["operation"] for edit in deletion["edits"]].count("delete") == 1
    assert [edit["operation"] for edit in insertion["edits"]].count("insert") == 1


def test_whole_subtree_removal_charges_every_occurrence():
    first = tree(constructor("A"), "first")
    second = tree(constructor("A", constructor("B", literal(1), literal(2))), "second")
    result = compare(first, second)
    assert result["distance"] == 3
    assert [edit["operation"] for edit in result["edits"]].count("delete") == 3


def test_subtree_deletion_and_insertion_cannot_promote_or_wrap_retained_children():
    first = tree(constructor("A", constructor("C", semantic_type="Leaf")), "first")
    second = tree(
        constructor(
            "A",
            constructor("B", constructor("C", semantic_type="Leaf"), semantic_type="Wrapper"),
        ),
        "second",
    )
    deleted = compare(first, second)
    inserted = compare(second, first)
    assert deleted["distance"] == inserted["distance"] == 3
    assert deleted["correspondences"] == [{"first_id": "first0", "second_id": "second0"}]
    assert {edit["second_id"] for edit in deleted["edits"] if edit["operation"] == "delete"} == {
        "second1",
        "second2",
    }
    assert {edit["first_id"] for edit in inserted["edits"] if edit["operation"] == "insert"} == {
        "second1",
        "second2",
    }


def test_equal_preorder_with_different_attachments_is_not_flat_sequence_identity():
    first = tree(constructor("A", constructor("B", constructor("C")), constructor("D")))
    second = tree(constructor("A", constructor("B"), constructor("C", constructor("D"))))
    assert [node["value"] for node in first["nodes"]] == [node["value"] for node in second["nodes"]]
    assert compare(first, second)["distance"] == 3


@pytest.mark.parametrize(
    "first,second,expected",
    [
        (2**100, 2**100 + 1, 1),
        (True, 1, 1),
        (None, None, 0),
        (1, 1.0, 0),
        (1e-320, 2e-320, 1),
        ("3", 3, 1),
    ],
)
def test_scalar_values_are_exact_and_large_integers_do_not_round(first, second, expected):
    assert compare(tree(literal(first)), tree(literal(second)))["distance"] == expected


def test_id_spelling_node_table_order_and_json_key_order_do_not_score():
    first = tree(constructor("A", constructor("B"), literal(2)), "first")
    second = tree(constructor("A", constructor("B"), literal(2)), "second")
    second["nodes"].reverse()
    second = json.loads(json.dumps(second, sort_keys=True))
    assert compare(first, second)["distance"] == 0


def invalid_trees():
    base = tree(constructor("A", constructor("B")))
    for label in [
        "version",
        "root",
        "missing_child",
        "cycle",
        "shared",
        "duplicate_id",
        "orphan",
        "unknown_field",
        "missing_field",
        "empty_id",
        "empty_role",
        "object_value",
        "constructor_null",
        "literal_children",
        "container_value",
    ]:
        invalid = copy.deepcopy(base)
        if label == "version":
            invalid["version"] = "unknown"
        elif label == "root":
            invalid["root"] = "missing"
        elif label == "missing_child":
            invalid["nodes"][0]["children"] = ["missing"]
        elif label == "cycle":
            invalid["nodes"][1]["children"] = ["n0"]
        elif label == "shared":
            invalid["nodes"][0]["children"] = ["n1", "n1"]
        elif label == "duplicate_id":
            invalid["nodes"][1]["id"] = "n0"
        elif label == "orphan":
            invalid["nodes"][0]["children"] = []
        elif label == "unknown_field":
            invalid["nodes"][0]["provenance"] = "/source"
        elif label == "missing_field":
            del invalid["nodes"][0]["value"]
        elif label == "empty_id":
            invalid["nodes"][0]["id"] = ""
        elif label == "empty_role":
            invalid["nodes"][0]["role"] = ""
        elif label == "object_value":
            invalid["nodes"][0]["value"] = {"A": 1}
        elif label == "constructor_null":
            invalid["nodes"][0]["value"] = None
        elif label == "literal_children":
            invalid["nodes"][0]["kind"] = "literal"
        elif label == "container_value":
            invalid["nodes"][0]["kind"] = "list"
        yield pytest.param(invalid, id=label)


@pytest.mark.parametrize("invalid", list(invalid_trees()))
def test_invalid_tree_never_produces_a_score(invalid):
    with pytest.raises(ValueError):
        compare_typed(invalid, tree(constructor("A")))
    with pytest.raises(ValueError):
        _core.compare_typed_json(json.dumps(invalid), json.dumps(tree(constructor("A"))))


@pytest.mark.parametrize("value", [float("nan"), float("inf"), object()])
def test_non_json_values_are_rejected_at_the_python_boundary(value):
    with pytest.raises(ValueError, match="JSON"):
        compare_typed(tree(literal(value)), tree(literal(1)))


def test_non_mapping_argument_is_an_explicit_type_error():
    with pytest.raises(TypeError, match="mappings"):
        compare_typed([], tree(constructor("A")))


def reference_distance(first, second):
    """Tiny independent mapping enumerator, not a dynamic-programming scorer.

    A legal ordered mapping preserves immediate parents and preorder; unmapped
    parents cannot have mapped descendants. Unmapped occurrences each cost one,
    and mapped compatible values match or cost one. Only tiny test trees use it.
    """

    def topology(tree):
        nodes = {node["id"]: node for node in tree["nodes"]}
        preorder = []
        parents = {}

        def visit(identifier, parent):
            preorder.append(identifier)
            parents[identifier] = parent
            for child in nodes[identifier]["children"]:
                visit(child, identifier)

        visit(tree["root"], None)
        return nodes, preorder, parents

    a, order_a, parents_a = topology(first)
    b, order_b, parents_b = topology(second)
    best = len(a) + len(b)
    for count in range(1, min(len(a), len(b)) + 1):
        for selected_a in itertools.combinations(order_a, count):
            for selected_b in itertools.combinations(order_b, count):
                mapping = list(zip(selected_a, selected_b, strict=True))
                if any(
                    any(a[x][key] != b[y][key] for key in ("kind", "semantic_type", "role"))
                    for x, y in mapping
                ):
                    continue
                pairs = dict(mapping)
                if any(
                    (parents_a[x] is None) != (parents_b[y] is None)
                    or (parents_a[x] is not None and pairs.get(parents_a[x]) != parents_b[y])
                    for x, y in mapping
                ):
                    continue
                cost = (
                    len(a)
                    + len(b)
                    - 2 * count
                    + sum(a[x]["value"] != b[y]["value"] for x, y in mapping)
                )
                best = min(best, cost)
    return best


_ORACLE_SHAPES = [
    constructor("A"),
    constructor("B"),
    constructor("A", constructor("B")),
    constructor("A", constructor("B"), constructor("C")),
    constructor("A", constructor("B", constructor("C"))),
    constructor("A", constructor("B", constructor("C")), constructor("D")),
    constructor("A", constructor("B"), constructor("C", constructor("D"))),
    constructor("A", constructor("B"), constructor("B")),
    constructor("A", constructor("B", role="other")),
]


@pytest.mark.parametrize(
    "first_index,second_index", list(itertools.product(range(len(_ORACLE_SHAPES)), repeat=2))
)
def test_native_closure_state_minimum_matches_independent_tiny_mapping_oracle(
    first_index, second_index
):
    first = tree(_ORACLE_SHAPES[first_index], "first")
    second = tree(_ORACLE_SHAPES[second_index], "second")
    result = compare(first, second)
    assert result["distance"] == reference_distance(first, second)
    assert result == compare(first, second)  # Ties remain deterministic.
