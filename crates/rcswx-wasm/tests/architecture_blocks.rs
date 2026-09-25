//! Source-derived structural blocks: these tests do not execute neural tensors.
#![cfg(not(target_arch = "wasm32"))]

use rcswx_core::architecture::Architecture;
use rcswx_wasm::session::Session;
use serde_json::Value;

#[derive(Debug, Eq, PartialEq, Ord, PartialOrd)]
struct Topology {
    name: String,
    children: Vec<Topology>,
}

fn topology(architecture: &Architecture, index: usize) -> Topology {
    let node = &architecture.nodes[index];
    let mut children = Vec::new();
    for &child in &node.children {
        let shape = topology(architecture, child);
        if node.name == "sequential" && shape.name == "sequential" {
            children.extend(shape.children);
        } else {
            children.push(shape);
        }
    }
    // Association of sequential composition and order of additive branches do
    // not change the network. All other ordering, especially Q/K, is retained.
    if node.name == "branching(2)"
        && children.len() == 4
        && children[3].name == "add(2)"
        && children[3].children.is_empty()
        && children[1] > children[2]
    {
        children.swap(1, 2);
    }
    Topology {
        name: node.name.clone(),
        children,
    }
}

fn check_route(from: &str, to: &str) {
    let blocks: Value = serde_json::from_str(include_str!("architecture_blocks.json")).unwrap();
    let first = blocks[to].to_string();
    let second = blocks[from].to_string();
    let target = Architecture::from_json(&first).unwrap();
    let baseline = Architecture::from_json(&second).unwrap();
    let mut session = Session::new("block-regression").unwrap();
    let analysis = session
        .analyze(&first, &second, r#"{"api":1,"trace_level":"none"}"#)
        .unwrap();
    assert_eq!(analysis.prefix_steps.first(), Some(&0));
    assert_eq!(
        analysis.prefix_steps.last(),
        Some(&analysis.nontrivial.len())
    );
    for &k in &analysis.prefix_steps {
        let frame = session
            .preview_step(&analysis.plan_id, k)
            .unwrap_or_else(|error| panic!("{from} → {to}, prefix {k}: {error:?}"));
        let child = Architecture::from_json(&frame.architecture_json).unwrap();
        if k == 0 {
            assert_eq!(
                topology(&child, child.root),
                topology(&baseline, baseline.root)
            );
        }
        if k == analysis.nontrivial.len() {
            assert_eq!(topology(&child, child.root), topology(&target, target.root));
        }
    }
}

#[test]
fn adding_a_residual_skip_does_not_disable_its_own_wrapper() {
    check_route("plain-cnn", "residual-cnn");
}

#[test]
fn nested_permuted_attention_wrappers_preserve_source_occurrences() {
    check_route("plain-cnn", "transformer");
}

#[test]
fn removing_a_swapped_wrapper_preserves_edits_in_both_branches() {
    check_route("mixer", "plain-cnn");
}

#[test]
fn insertion_after_a_residual_stays_outside_its_skip_branch() {
    check_route("mixer", "residual-cnn");
}

#[test]
fn insertion_before_attention_uses_the_retained_branch_orientation() {
    check_route("residual-cnn", "transformer");
}
