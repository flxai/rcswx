// Native contract tests for the same session used by the WASM facade.
#![cfg(not(target_arch = "wasm32"))]
use rcswx_wasm::{
    dto::{MAX_INPUT_BYTES, MAX_PLANS},
    session::{Session, checked_index},
};
use serde_json::{Value, json};
fn inputs() -> Vec<Value> {
    serde_json::from_str::<Value>(include_str!("../../../examples/web/fixtures/inputs.json"))
        .unwrap()["pairs"]
        .as_array()
        .unwrap()
        .clone()
}
fn parents(pair: &Value) -> (&str, &str) {
    (
        pair["parent1_json"].as_str().unwrap(),
        pair["parent2_json"].as_str().unwrap(),
    )
}
#[test]
fn every_admitted_prefix_repeats_after_failures_and_seeded_samples() {
    for pair in inputs() {
        let (a, b) = parents(&pair);
        let mut s = Session::new("test").unwrap();
        let p = s
            .analyze(a, b, r#"{"api":1,"trace_level":"full"}"#)
            .unwrap();
        let id = &p.plan_id;
        let frames = (0..=p.nontrivial.len())
            .map(|k| serde_json::to_value(s.preview_step(id, k).unwrap()).unwrap())
            .collect::<Vec<_>>();
        let before = serde_json::to_value(s.recording(id).unwrap()).unwrap();
        assert!(s.apply_selection(id, "[99999]").is_err());
        for k in [
            0,
            p.nontrivial.len(),
            usize::from(!p.nontrivial.is_empty()),
            p.nontrivial.len(),
            0,
        ] {
            let _ = s.sample(id, &"00".repeat(32), 0.0).unwrap();
            assert_eq!(
                serde_json::to_value(s.preview_step(id, k).unwrap()).unwrap(),
                frames[k]
            );
        }
        assert_eq!(
            serde_json::to_value(s.recording(id).unwrap()).unwrap(),
            before
        );
        let mut offset = 0;
        let mut events = vec![];
        loop {
            let page = s.trace_page(id, offset, 7).unwrap();
            events.extend(page.events);
            if let Some(next) = page.next_offset {
                assert!(next > offset);
                offset = next;
            } else {
                break;
            }
        }
        assert_eq!(serde_json::to_value(events).unwrap(), before["events"]);
        assert!(s.dispose(id));
        assert!(!s.dispose(id));
        assert_eq!(s.inspect(id).unwrap_err().code, "expired_plan");
    }
}
#[test]
fn unsupported_prefix_is_an_error_not_the_previous_child() {
    let pairs = inputs();
    let mut s = Session::new("invalid-prefix").unwrap();
    let (first, _) = parents(&pairs[4]);
    let (second, _) = parents(&pairs[2]);
    let p = s.analyze(first, second, r#"{"api":1}"#).unwrap();
    let initial = s.preview_step(&p.plan_id, 0).unwrap().architecture_json;
    let error = s.preview_step(&p.plan_id, 1).unwrap_err();
    assert_eq!(error.code, "invalid_selection");
    assert_eq!(error.requested_step, Some(1));
    assert_eq!(
        s.preview_step(&p.plan_id, 0).unwrap().architecture_json,
        initial
    );
}
#[test]
fn json_payloads_and_repeated_logical_ids_remain_lossless() {
    let text = r#"{"schema":1,"grammar":"x","grammar_version":"1","root":0,"input_spec":{"n":999999999999999999999999999999999999},"nodes":[{"id":"9007199254740993","name":"computation","children":[1],"parameters":{"n":9007199254740993123456789}},{"id":"9007199254740993","name":"relu","children":[],"provenance":{"label":"<script>alert(1)</script>"}}]}"#;
    let mut s = Session::new("json").unwrap();
    let p = s.analyze(text, text, r#"{"api":1}"#).unwrap();
    let child = s.preview_step(&p.plan_id, 0).unwrap();
    assert!(
        child
            .architecture_json
            .contains("9007199254740993123456789")
    );
    assert!(
        child
            .architecture_json
            .contains("999999999999999999999999999999999999")
    );
    assert_eq!(child.origins.len(), 2);
    assert_ne!(
        child.origins[0].sources[0].occurrence_index,
        child.origins[1].sources[0].occurrence_index
    );
}
#[test]
fn resource_and_boundary_errors_are_typed() {
    let pairs = inputs();
    let (a, b) = parents(&pairs[0]);
    let mut s = Session::new("bounds").unwrap();
    assert_eq!(
        s.analyze(a, b, r#"{"api":99}"#).unwrap_err().code,
        "unsupported_api"
    );
    assert_eq!(
        s.analyze(&" ".repeat(MAX_INPUT_BYTES + 1), b, "{}")
            .unwrap_err()
            .code,
        "input_limit"
    );
    assert_eq!(
        s.analyze(
            a,
            b,
            r#"{"limits":{"max_work":0},"trace_limits":{"max_events":0,"max_bytes":0}}"#
        )
        .unwrap_err()
        .code,
        "core_quota"
    );
    let p = s
        .analyze(
            a,
            b,
            r#"{"trace_level":"full","trace_limits":{"max_events":1,"max_bytes":4096}}"#,
        )
        .unwrap();
    assert!(!p.trace.complete);
    assert_eq!(s.inspect("foreign:1").unwrap_err().code, "wrong_plan");
    for v in [-1.0, 0.5, f64::NAN, f64::INFINITY, 4294967296.0] {
        assert!(checked_index(v, "test").is_err());
    }
    assert!(s.apply_selection(&p.plan_id, "[0.5]").is_err());
    assert_eq!(
        s.sample(&p.plan_id, &"AA".repeat(32), 0.0)
            .unwrap_err()
            .code,
        "invalid_seed"
    );
    for _ in 1..MAX_PLANS {
        s.analyze(a, b, "{}").unwrap();
    }
    assert_eq!(s.analyze(a, b, "{}").unwrap_err().code, "session_limit");
    s.dispose(&p.plan_id);
    assert!(s.analyze(a, b, "{}").is_ok());
}
#[test]
fn seeded_masks_use_nontrivial_positions_and_empty_edits_stay_empty() {
    let pairs = inputs();
    let mut s = Session::new("sample").unwrap();
    for pair in [&pairs[0], &pairs[4]] {
        let (a, b) = parents(pair);
        let p = s.analyze(a, b, "{}").unwrap();
        let sample = s.sample(&p.plan_id, &"00".repeat(32), 0.0).unwrap();
        let selected = sample
            .mask
            .as_ref()
            .unwrap()
            .bytes()
            .zip(&p.nontrivial)
            .filter_map(|(bit, &index)| (bit == b'1').then_some(index))
            .collect::<Vec<_>>();
        assert_eq!(sample.selected_indices, selected);
        let explicit = s
            .apply_selection(&p.plan_id, &json!(selected).to_string())
            .unwrap();
        assert_eq!(sample.architecture_json, explicit.architecture_json);
        assert_eq!(
            serde_json::to_value(s.sample(&p.plan_id, &"00".repeat(32), 0.0).unwrap()).unwrap(),
            serde_json::to_value(sample).unwrap()
        );
    }
}
