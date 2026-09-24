#![cfg(feature = "trace")]
use rcswx_core::{
    apply::apply_traced,
    architecture::{Architecture, Budget, Limits, Node},
    edit_plan::{analyze, analyze_traced},
    error::Error,
    tokens::prepare,
    trace::{Recorder, TraceLevel, TraceLimits, number},
};
use serde_json::{Value, json};
use std::collections::HashMap;
fn tree(value: Value) -> Architecture {
    fn add(v: &Value, nodes: &mut Vec<Node>) -> usize {
        let index = nodes.len();
        nodes.push(Node {
            id: index.to_string(),
            name: v[0].as_str().unwrap().into(),
            children: vec![],
            parameters: None,
            provenance: None,
        });
        nodes[index].children = v.as_array().unwrap()[1..]
            .iter()
            .map(|v| add(v, nodes))
            .collect();
        index
    }
    let mut nodes = vec![];
    add(&value, &mut nodes);
    Architecture {
        schema: 1,
        grammar: "trace".into(),
        grammar_version: "1".into(),
        root: 0,
        nodes,
        input_spec: None,
    }
}
fn with_budget<T>(f: impl FnOnce(&mut Budget<'_>) -> T) -> T {
    let mut check = || Ok(());
    f(&mut Budget {
        limits: Limits::default(),
        work: 0,
        output: 0,
        allocation_bytes: 0,
        check: &mut check,
    })
}
fn pair() -> rcswx_core::tokens::PreparedPair {
    prepare(
        tree(json!([
            "branching(2)",
            ["clone(2)"],
            ["computation", ["relu"]],
            ["computation", ["sigmoid"]],
            ["add(2)"]
        ])),
        tree(json!([
            "branching(2)",
            ["clone(2)"],
            ["computation", ["identity"]],
            ["computation", ["sigmoid"]],
            ["add(2)"]
        ])),
    )
    .unwrap()
}
#[test]
fn replay_preserves_recursive_aliases_candidates_and_retained_histories() {
    let baseline = with_budget(|b| analyze(pair(), false, "p".into(), b)).unwrap();
    let mut recorder = Recorder::new(TraceLevel::Full, TraceLimits::default());
    let traced =
        with_budget(|b| analyze_traced(pair(), false, "p".into(), b, &mut recorder)).unwrap();
    assert_eq!(
        serde_json::to_value(&baseline.paths).unwrap(),
        serde_json::to_value(&traced.paths).unwrap()
    );
    assert_eq!(baseline.stats, traced.stats);
    assert_eq!(baseline.nontrivial, traced.nontrivial);
    let recording = recorder.finish();
    assert!(recording.complete);
    let mut histories = HashMap::new();
    let mut cells = HashMap::new();
    let mut cell_uses = HashMap::<u64, usize>::new();
    let mut problems = 0;
    let mut swapped = false;
    let mut fractional = false;
    let mut uncomputed = false;
    let mut infinite = false;
    let mut retained = vec![];
    for (seq, event) in recording.events.iter().enumerate() {
        assert_eq!(seq, event.seq);
        let d = &event.data;
        match event.kind.as_str() {
            "history" => {
                let v = &d["value"];
                if let Some(parent) = v["previous"].as_u64() {
                    assert!(histories.contains_key(&parent));
                }
                histories.insert(d["id"].as_u64().unwrap(), v.clone());
            }
            "cell" => {
                let v = &d["value"];
                for id in v["histories"].as_array().unwrap() {
                    assert!(histories.contains_key(&id.as_u64().unwrap()));
                }
                uncomputed |= v["value"]["state"] == "uncomputed";
                fractional |= v["value"].as_f64().is_some_and(|n| n.fract() != 0.0);
                infinite |= v["top"]
                    .as_array()
                    .unwrap()
                    .iter()
                    .chain(v["left"].as_array().unwrap())
                    .chain(v["corner"].as_array().unwrap())
                    .any(|n| n["state"] == "positive_infinity");
                cells.insert(d["id"].as_u64().unwrap(), v.clone());
            }
            "matrix" => {
                for row in d["cells"].as_array().unwrap() {
                    for id in row.as_array().unwrap() {
                        let id = id.as_u64().unwrap();
                        assert!(cells.contains_key(&id));
                        *cell_uses.entry(id).or_default() += 1;
                    }
                }
            }
            "computed" => {
                let v = &cells[&d["cell"].as_u64().unwrap()];
                let minimum = v["top"]
                    .as_array()
                    .unwrap()
                    .iter()
                    .chain(v["left"].as_array().unwrap())
                    .chain(v["corner"].as_array().unwrap())
                    .filter_map(Value::as_f64)
                    .reduce(f64::min);
                if let Some(expected) = minimum {
                    assert_eq!(v["value"].as_f64(), Some(expected));
                }
                for id in d["predecessors"].as_array().unwrap() {
                    if let Some(id) = id.as_u64() {
                        assert!(cells.contains_key(&id));
                    }
                }
            }
            "subproblem" => {
                problems += 1;
                for side in ["first_indices", "second_indices"] {
                    swapped |= d[side]
                        .as_array()
                        .unwrap()
                        .windows(2)
                        .any(|w| w[0].as_u64() > w[1].as_u64());
                }
            }
            "retained_histories" => retained = d["histories"].as_array().unwrap().clone(),
            _ => {}
        }
    }
    assert!(problems > 1 && swapped && fractional && uncomputed && infinite);
    assert!(cell_uses.values().any(|&uses| uses > 1));
    assert_eq!(retained.len(), traced.paths.len());
    for (tail, path) in retained.iter().zip(&traced.paths) {
        let mut id = Some(tail.as_u64().unwrap());
        let mut replayed = vec![];
        while let Some(current) = id {
            let h = &histories[&current];
            replayed.push(h["step"].clone());
            id = h["previous"].as_u64();
        }
        replayed.reverse();
        assert_eq!(replayed.len(), path.len());
        for (step, edit) in replayed.iter().zip(path) {
            assert_eq!(step["kind"], edit.op_type);
            assert_eq!(step["value"], number(edit.value, false));
        }
    }
}
#[test]
fn truncation_does_not_change_computation_or_hide_core_failures() {
    let baseline = with_budget(|b| analyze(pair(), false, "p".into(), b)).unwrap();
    for level in [TraceLevel::None, TraceLevel::Summary, TraceLevel::Full] {
        let mut recorder = Recorder::new(
            level,
            TraceLimits {
                max_events: 2,
                max_bytes: 4096,
            },
        );
        let plan =
            with_budget(|b| analyze_traced(pair(), false, "p".into(), b, &mut recorder)).unwrap();
        assert_eq!(
            serde_json::to_value(plan.paths).unwrap(),
            serde_json::to_value(&baseline.paths).unwrap()
        );
        assert_eq!(plan.stats, baseline.stats);
        let record = recorder.finish();
        assert!(record.events.len() <= 2);
        assert_eq!(record.complete, level == TraceLevel::None);
    }
    let mut recorder = Recorder::new(
        TraceLevel::Full,
        TraceLimits {
            max_events: 0,
            max_bytes: 0,
        },
    );
    let error = with_budget(|b| {
        b.limits.max_work = Some(0);
        analyze_traced(pair(), false, "p".into(), b, &mut recorder)
    })
    .unwrap_err();
    assert!(matches!(error, Error::Limit(_)));
}
#[test]
fn synthesized_origins_follow_handles_and_baselines_are_repeatable() {
    let prepared = prepare(
        tree(json!([
            "sequential",
            ["computation", ["relu"]],
            ["computation", ["sigmoid"]]
        ])),
        tree(json!(["computation", ["sigmoid"]])),
    )
    .unwrap();
    let plan = with_budget(|b| analyze(prepared, false, "p".into(), b)).unwrap();
    let run = |selected: &[usize]| {
        let mut recorder = Recorder::new(TraceLevel::Full, TraceLimits::default());
        let child =
            with_budget(|b| apply_traced(&mut plan.clone(), selected, true, b, &mut recorder))
                .unwrap();
        (child, recorder.finish())
    };
    let (a, ar) = run(&plan.nontrivial);
    let _ = run(&[]);
    let (b, br) = run(&plan.nontrivial);
    assert_eq!(
        serde_json::to_value(&a).unwrap(),
        serde_json::to_value(&b).unwrap()
    );
    assert_eq!(
        serde_json::to_value(ar).unwrap(),
        serde_json::to_value(br).unwrap()
    );
    assert!(a.origins[0].synthesized);
    assert!(a.origins[0].sources.iter().any(|s| s.parent == 1));
    assert!(a.origins[0].sources.iter().any(|s| s.parent == 2));
}
