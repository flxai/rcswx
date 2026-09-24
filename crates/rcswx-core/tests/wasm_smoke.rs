//! Development-only execution proof for the portable core. This deliberately
//! stays an integration test: no PyO3 crate, browser product, or host entropy
//! is linked for wasm32-unknown-unknown.

use rcswx_core::apply::apply;
use rcswx_core::architecture::{Architecture, Budget, Limits};
use rcswx_core::edit_plan::{EditPlan, analyze};
use rcswx_core::error::Result;
use rcswx_core::sampling::{NativeRng, draw, probabilities};
use rcswx_core::tokens::prepare;
#[cfg(target_arch = "wasm32")]
use wasm_bindgen_test::wasm_bindgen_test;

type EditSignature<'a> = (
    usize,
    &'a str,
    Option<&'a str>,
    Option<&'a str>,
    usize,
    usize,
    Option<usize>,
    Option<usize>,
    u64,
    (bool, bool),
    usize,
    usize,
);

fn decoded_architecture(root_name: &str, leaf: &str) -> Architecture {
    Architecture::from_json(&format!(
        r#"{{"schema":1,"grammar":"wasm-smoke","grammar_version":"1","root":0,"nodes":[{{"id":"0","name":"{root_name}","children":[1]}},{{"id":"1","name":"{leaf}","children":[]}}]}}"#
    ))
    .expect("the frozen smoke architecture is valid JSON and an occurrence tree")
}

fn unlimited_budget(check: &mut dyn FnMut() -> Result<()>) -> Budget<'_> {
    Budget {
        limits: Limits::default(),
        work: 0,
        output: 0,
        allocation_bytes: 0,
        check,
        wait_check: None,
    }
}

fn histories(plan: &EditPlan) -> Vec<Vec<EditSignature<'_>>> {
    plan.paths
        .iter()
        .map(|history| {
            history
                .iter()
                .map(|edit| {
                    (
                        edit.id,
                        edit.op_type.as_str(),
                        edit.node1_id.as_deref(),
                        edit.node2_id.as_deref(),
                        edit.i,
                        edit.j,
                        edit.ii,
                        edit.jj,
                        edit.value.to_bits(),
                        (edit.i_swapped, edit.j_swapped),
                        edit.enabler_ops.len(),
                        edit.disabler_ops.len(),
                    )
                })
                .collect()
        })
        .collect()
}

#[cfg_attr(target_arch = "wasm32", wasm_bindgen_test)]
#[cfg_attr(not(target_arch = "wasm32"), test)]
fn decode_analyze_apply_and_sample_without_host_services() {
    #[cfg(target_arch = "wasm32")]
    {
        use rcswx_core::execution::{Execution, Workers};
        assert!(!Execution::serial().report().parallel_capable);
        assert!(Workers::new(1).is_ok());
        assert!(matches!(
            Workers::new(2),
            Err(rcswx_core::error::Error::Execution(_))
        ));
    }
    // This fixture is frozen from the pinned Python 3.12 reference run in
    // /home/flx/tmp/rcswx-consolidation-20260923/wasm-oracle-fixture.json.
    let first = decoded_architecture("sequential", "relu");
    let second = decoded_architecture("sequential", "sigmoid");
    let prepared = prepare(first, second).expect("prepare portable architectures");
    let mut check = || Ok(());
    let mut budget = unlimited_budget(&mut check);
    let mut plan = analyze(prepared, false, "wasm-smoke".into(), &mut budget)
        .expect("analyze every retained history on wasm");

    assert_eq!(plan.distance.to_bits(), 2.0_f64.to_bits());
    assert_eq!(
        histories(&plan),
        vec![
            vec![
                (
                    0,
                    "start",
                    Some("-1"),
                    Some("-1"),
                    0,
                    0,
                    None,
                    None,
                    0.0_f64.to_bits(),
                    (false, false),
                    0,
                    0
                ),
                (
                    1,
                    "rem",
                    None,
                    Some("3"),
                    0,
                    1,
                    None,
                    None,
                    1.0_f64.to_bits(),
                    (false, false),
                    0,
                    0
                ),
                (
                    2,
                    "add_module",
                    Some("1"),
                    Some("3"),
                    1,
                    1,
                    None,
                    None,
                    1.0_f64.to_bits(),
                    (false, false),
                    0,
                    0
                ),
            ],
            vec![
                (
                    0,
                    "start",
                    Some("-1"),
                    Some("-1"),
                    0,
                    0,
                    None,
                    None,
                    0.0_f64.to_bits(),
                    (false, false),
                    0,
                    0
                ),
                (
                    1,
                    "add_module",
                    Some("1"),
                    Some("-1"),
                    1,
                    0,
                    None,
                    None,
                    1.0_f64.to_bits(),
                    (false, false),
                    0,
                    0
                ),
                (
                    2,
                    "rem",
                    None,
                    Some("3"),
                    1,
                    1,
                    None,
                    None,
                    1.0_f64.to_bits(),
                    (false, false),
                    0,
                    0
                ),
            ],
        ]
    );
    assert_eq!(plan.operations, vec![2, 1]);
    assert_eq!(plan.nontrivial, vec![2, 1]);

    let expected_child = plan.prepared.second.clone();
    let applied = apply(&mut plan, &[], true, &mut budget)
        .expect("an explicit empty selection materializes parent two");
    let encoded = applied
        .architecture
        .to_json()
        .expect("serialize applied architecture");
    assert_eq!(encoded, expected_child.to_json().unwrap());
    assert_eq!(
        Architecture::from_json(&encoded)
            .unwrap()
            .to_json()
            .unwrap(),
        encoded
    );

    let prepared = prepare(
        decoded_architecture("computation", "identity"),
        decoded_architecture("computation", "relu"),
    )
    .unwrap();
    let mut plan = analyze(prepared, false, "wasm-edit".into(), &mut budget).unwrap();
    let selected = plan.nontrivial.clone();
    let edited = apply(&mut plan, &selected, true, &mut budget).unwrap();
    assert_eq!(
        edited
            .architecture
            .nodes
            .iter()
            .map(|node| node.name.as_str())
            .collect::<Vec<_>>(),
        ["computation", "identity"]
    );

    // Two independent 0.25-cost edits yield masks with costs [0, .25, .25,
    // .5]. At skewness zero, their reflected grid positions have equal weight.
    let weights =
        probabilities(&[0.0, 0.25, 0.25, 0.5], 0.0).expect("construct native probabilities");
    assert_eq!(weights.len(), 4);
    for weight in weights {
        assert!(
            (weight - 0.25).abs() < 1e-12,
            "unexpected probability: {weight}"
        );
    }

    // This public rand_chacha 0.9 ChaCha12 vector pins the raw stream before
    // any floating-point conversion. It is intentionally separate from draw.
    let mut raw = NativeRng::from_seed([0; 32]);
    let expected_raw = [0x53f9_5507_6a9a_f49b_u64];
    let actual_raw = expected_raw.map(|_| raw.next_u64());
    assert_eq!(actual_raw, expected_raw);

    let mut seeded = NativeRng::from_seed([0; 32]);
    assert_eq!(draw(&[0.25; 4], &mut seeded).unwrap(), 1);
}
