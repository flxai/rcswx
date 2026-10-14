use rcswx_core::typed::{Comparison, Tree, compare};
use serde_json::{Value, json};

fn tree(value: &str, children: &[(&str, &str)]) -> Tree {
    let mut nodes = vec![json!({
        "id": "root", "kind": "constructor", "semantic_type": "Term",
        "role": "item", "value": value,
        "children": children.iter().map(|(id, _)| *id).collect::<Vec<_>>()
    })];
    nodes.extend(children.iter().map(|(id, value)| {
        json!({
            "id": id, "kind": "constructor", "semantic_type": "Term",
            "role": "item", "value": value, "children": []
        })
    }));
    Tree::from_json(
        &json!({
            "version": "rcswx-typed-ordered-v1", "root": "root", "nodes": nodes
        })
        .to_string(),
    )
    .unwrap()
}

fn comparison(first: &Tree, second: &Tree) -> Comparison {
    let result = compare(first, second, &mut || Ok(())).unwrap();
    assert_eq!(
        result.edits.iter().map(|edit| edit.cost).sum::<usize>(),
        result.distance
    );
    assert_eq!(result.direction, "second-to-first");
    result
}

#[test]
fn nullary_constructor_values_are_native_semantics() {
    let first = tree("wrap_end", &[]);
    let same = tree("wrap_end", &[]);
    let second = tree("start_node", &[]);
    assert_eq!(comparison(&first, &same).distance, 0);
    let result = comparison(&first, &second);
    assert_eq!(result.distance, 1);
    assert_eq!(result.edits[0].operation, "substitute");
    assert_eq!(result.edits[0].first_id.as_deref(), Some("root"));
    assert_eq!(result.edits[0].second_id.as_deref(), Some("root"));
}

#[test]
fn high_arity_preserves_first_middle_and_last_children() {
    let first = tree("A", &[("a", "B"), ("b", "C"), ("c", "D"), ("d", "E")]);
    assert_eq!(comparison(&first, &first).edits.len(), 5);
    for changed in 0..4 {
        let mut second = tree("A", &[("a", "B"), ("b", "C"), ("c", "D"), ("d", "E")]);
        second.nodes[changed + 1].value =
            serde_json::value::RawValue::from_string("\"changed\"".into()).unwrap();
        assert_eq!(comparison(&first, &second).distance, 1);
    }
}

#[test]
fn incompatible_roles_and_types_require_occurrence_replacement() {
    let first = tree("A", &[]);
    let mut second = tree("A", &[]);
    second.nodes[0].role = "different".into();
    let result = comparison(&first, &second);
    assert_eq!(result.distance, 2);
    assert!(result.correspondences.is_empty());
    assert!(result.edits.iter().any(|edit| edit.operation == "insert"));
    assert!(result.edits.iter().any(|edit| edit.operation == "delete"));
    second.nodes[0].role = "item".into();
    second.nodes[0].semantic_type = "Other".into();
    assert_eq!(comparison(&first, &second).distance, 2);
}

#[test]
fn ordering_and_duplicate_multiplicity_are_not_discarded() {
    let first = tree("A", &[("a", "B"), ("b", "C"), ("c", "D")]);
    let reverse = tree("A", &[("a", "D"), ("b", "C"), ("c", "B")]);
    assert_eq!(comparison(&first, &reverse).distance, 2);
    let duplicate = tree("A", &[("a", "B"), ("b", "B")]);
    let single = tree("A", &[("a", "B")]);
    let result = comparison(&single, &duplicate);
    assert_eq!(result.distance, 1);
    assert_eq!(
        result
            .edits
            .iter()
            .filter(|edit| edit.operation == "delete")
            .count(),
        1
    );
}

fn parsed(nodes: Value) -> Tree {
    Tree::from_json(
        &json!({
            "version": "rcswx-typed-ordered-v1", "root": "a", "nodes": nodes
        })
        .to_string(),
    )
    .unwrap()
}

fn node(id: &str, value: &str, children: &[&str]) -> Value {
    json!({"id": id, "kind": "constructor", "semantic_type": "Term",
        "role": "item", "value": value, "children": children})
}

#[test]
fn deleting_incompatible_subtree_cannot_promote_its_matching_descendant() {
    let mut first = parsed(json!([node("a", "A", &["c"]), node("c", "C", &[])]));
    first.nodes[1].semantic_type = "Leaf".into();
    let mut second = parsed(json!([
        node("a", "A", &["b"]),
        node("b", "B", &["c"]),
        node("c", "C", &[])
    ]));
    second.nodes[1].semantic_type = "Wrapper".into();
    second.nodes[2].semantic_type = "Leaf".into();
    let result = comparison(&first, &second);
    assert_eq!(result.distance, 3);
    assert_eq!(result.correspondences.len(), 1);
    assert_eq!(result.correspondences[0].first_id, "a");
    assert_eq!(result.correspondences[0].second_id, "a");
    assert_eq!(
        result
            .edits
            .iter()
            .filter(|edit| edit.operation == "delete")
            .count(),
        2
    );
    assert_eq!(comparison(&second, &first).distance, 3);
}

#[test]
fn equal_preorder_different_attachments_cannot_match_for_free() {
    let first = parsed(json!([
        node("a", "A", &["b", "d"]),
        node("b", "B", &["c"]),
        node("c", "C", &[]),
        node("d", "D", &[])
    ]));
    let second = parsed(json!([
        node("a", "A", &["b", "c"]),
        node("b", "B", &[]),
        node("c", "C", &["d"]),
        node("d", "D", &[])
    ]));
    assert_eq!(comparison(&first, &second).distance, 3);
}

#[test]
fn invalid_occurrence_trees_fail_before_alignment() {
    let valid = tree("A", &[]);
    for nodes in [
        json!([node("a", "A", &["a"])]),
        json!([node("a", "A", &["b", "b"]), node("b", "B", &[])]),
        json!([node("a", "A", &["missing"])]),
        json!([node("a", "A", &[]), node("b", "B", &[])]),
        json!([node("a", "A", &[]), node("a", "B", &[])]),
    ] {
        let invalid = Tree::from_json(
            &json!({
                "version": "rcswx-typed-ordered-v1", "root": "a", "nodes": nodes
            })
            .to_string(),
        )
        .unwrap();
        assert!(compare(&invalid, &valid, &mut || Ok(())).is_err());
    }
}

#[test]
fn cancellation_is_not_a_score() {
    let first = tree("A", &[]);
    assert!(
        compare(&first, &first, &mut || Err(
            rcswx_core::error::Error::Cancelled
        ))
        .is_err()
    );
}
