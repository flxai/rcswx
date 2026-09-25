import pytest
from rcswx import Architecture, apply_edits, edit_path


def test_portable_tree_round_trip_uses_canonical_native_schema():
    architecture = Architecture.from_tree(
        ("sequential", ("identity",), ("relu",)), grammar="einspace", input_spec={"shape": [2]}
    )

    encoded = architecture.to_dict()

    assert encoded == {
        "schema": 1,
        "grammar": "einspace",
        "grammar_version": "1",
        "root": 0,
        "nodes": [
            {
                "id": "0",
                "name": "sequential",
                "children": [1, 2],
                "parameters": None,
                "provenance": None,
            },
            {
                "id": "1",
                "name": "identity",
                "children": [],
                "parameters": None,
                "provenance": None,
            },
            {
                "id": "2",
                "name": "relu",
                "children": [],
                "parameters": None,
                "provenance": None,
            },
        ],
        "input_spec": {"shape": [2]},
    }
    assert Architecture.from_json(architecture.to_json()).to_dict() == encoded


def test_public_json_constructor_still_rejects_invalid_structure():
    cyclic = (
        '{"schema":1,"grammar":"einspace","grammar_version":"1","root":0,'
        '"nodes":[{"id":"0","name":"relu","children":[0]}]}'
    )
    with pytest.raises(ValueError):
        Architecture.from_json(cyclic)


def test_empty_plan_selection_starts_at_second_portable_parent():
    first = Architecture.from_tree(("identity",))
    second = Architecture.from_tree(("relu",))

    plan = edit_path(first, second)
    selection = plan.select([])
    child = apply_edits(plan, selection)

    assert isinstance(child, Architecture)
    assert child.to_dict()["nodes"][0]["name"] == "relu"
    assert selection.cost == 0
    assert type(selection.cost) is int


def test_selection_cannot_cross_native_plan_boundaries():
    first = Architecture.from_tree(("identity",))
    second = Architecture.from_tree(("relu",))
    selection = edit_path(first, second).select([])
    plan = edit_path(first, second)

    with pytest.raises(ValueError, match="another plan"):
        apply_edits(plan, selection)


def test_portable_distance_respects_edges_not_node_table_order():
    from rcswx import distance

    first = Architecture.from_tree(
        ("sequential", ("computation", ("relu",)), ("computation", ("sigmoid",)))
    )
    changed = first.to_dict()
    changed["nodes"][0]["children"].reverse()
    second = Architecture.from_dict(changed)

    assert edit_path(first, second).distance == 1.0
    assert distance(first, second) == 1.0


def test_opaque_metadata_numbers_survive_round_trip_and_application():
    integer = 2**100 + 1
    second_data = Architecture.from_tree(("identity",), input_spec={"counter": integer}).to_dict()
    second_data["nodes"][0]["parameters"] = {
        "positive": integer,
        "negative": -integer,
        "subnormal": 1e-320,
    }
    second_data["nodes"][0]["provenance"] = {
        "opaque": [integer],
        "$serde_json::private::Number": "opaque",
        "$serde_json::private::RawValue": "opaque",
    }
    second = Architecture.from_dict(second_data)
    assert Architecture.from_json(second.to_json()).to_dict() == second_data

    first = Architecture.from_tree(("relu",), input_spec=second_data["input_spec"])
    plan = edit_path(first, second)
    child = apply_edits(plan, plan.select([])).to_dict()
    assert child["input_spec"] == second_data["input_spec"]
    assert child["nodes"][0]["parameters"] == second_data["nodes"][0]["parameters"]
    assert child["nodes"][0]["provenance"] == second_data["nodes"][0]["provenance"]


def operation_tree(architecture):
    data = architecture.to_dict()

    def subtree(index):
        node = data["nodes"][index]
        return (node["name"], *(subtree(child) for child in node["children"]))

    return subtree(data["root"])


@pytest.mark.parametrize("reorient", [False, True])
def test_selected_mutation_preserves_unedited_module_and_wrapper_metadata(reorient):
    parents = []
    for source, activation in (("first", "relu"), ("second", "identity")):
        if reorient:
            branches = (
                ("linear(32)", "linear(16)") if source == "first" else ("linear(16)", "linear(32)")
            )
            description = (
                "sequential",
                (
                    "branching(2)",
                    ("clone(2)",),
                    *(("computation", (name,)) for name in branches),
                    ("cat(2,2)",),
                ),
                ("computation", (activation,)),
            )
        else:
            description = (
                "sequential",
                ("computation", ("linear(16)",)),
                ("routing", ("identity",), ("computation", (activation,)), ("identity",)),
            )
        data = Architecture.from_tree(description).to_dict()
        for node in data["nodes"]:
            node["parameters"] = {"source": source}
            node["provenance"] = {"source": source}
        parents.append(Architecture.from_dict(data))

    plan = edit_path(*parents)
    child = apply_edits(plan, plan.select("1" * len(plan.nontrivial_ops)))
    assert operation_tree(child) == operation_tree(parents[0])
    unchanged = {"linear(16)", "linear(32)", "routing", "branching(2)", "clone(2)", "cat(2,2)"}
    for node in child.to_dict()["nodes"]:
        if node["name"] in unchanged | {"relu"}:
            source = "second" if node["name"] in unchanged else "first"
            assert node["parameters"]["source"] == node["provenance"]["source"] == source


def test_repeated_wrapper_ids_do_not_conflate_branch_orientations():
    def branch(left, right):
        return (
            "branching(2)",
            ("clone(2)",),
            ("computation", (left,)),
            ("computation", (right,)),
            ("cat(2,2)",),
        )

    first = Architecture.from_tree(
        ("sequential", branch("linear(64)", "linear(128)"), branch("linear(32)", "relu"))
    )
    second = Architecture.from_tree(
        ("sequential", branch("linear(64)", "linear(128)"), branch("linear(16)", "linear(32)"))
    )
    original_distance = edit_path(first, second).distance
    data = first.to_dict()
    wrappers = [node for node in data["nodes"] if node["name"] == "branching(2)"]
    wrappers[1]["id"] = wrappers[0]["id"]
    first = Architecture.from_dict(data)

    plan = edit_path(first, second)
    assert plan.distance == original_distance
    child = apply_edits(plan, plan.select("1" * len(plan.nontrivial_ops)))
    assert operation_tree(child) == operation_tree(first)


def test_added_branch_requires_its_addition_but_not_an_edit_in_the_retained_branch():
    first = Architecture.from_tree(
        (
            "branching(2)",
            ("clone(2)",),
            ("computation", ("linear(16)",)),
            ("computation", ("relu",)),
            ("add(2)",),
        )
    )
    second = Architecture.from_tree(("computation", ("identity",)))
    plan = edit_path(first, second)
    wrapper_only = "".join("1" if op.op_type == "add_wrap" else "0" for op in plan.nontrivial_ops)
    with pytest.raises(ValueError):
        plan.select(wrapper_only)

    additions = "".join("1" if op.op_type.startswith("add") else "0" for op in plan.nontrivial_ops)
    child = apply_edits(plan, plan.select(additions))
    assert operation_tree(child) == (
        "branching(2)",
        ("clone(2)",),
        ("computation", ("identity",)),
        ("computation", ("relu",)),
        ("add(2)",),
    )


def test_stale_legacy_plan_rejects_selection_without_advancing_rng():
    from types import SimpleNamespace

    from rcswx import NativeRng

    def node(identifier, name):
        return SimpleNamespace(
            id=identifier,
            operation=SimpleNamespace(name=name),
            children=[],
            limiter=None,
        )

    first, second = node(0, "identity"), node(1, "relu")
    plan = edit_path(first, second)
    first.operation.name = "sigmoid"
    rng = NativeRng(0)
    before = rng.state()

    with pytest.raises(ValueError, match="parents changed"):
        plan.select([])
    with pytest.raises(ValueError, match="parents changed"):
        plan.probabilities()
    with pytest.raises(ValueError, match="parents changed"):
        plan.sample(rng=rng)
    assert rng.state() == before
