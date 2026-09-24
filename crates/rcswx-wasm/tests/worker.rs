// This browser-only suite does not change the core's Node smoke configuration.
#![cfg(target_arch = "wasm32")]
use rcswx_wasm::WasmSession;
use serde_json::Value;
use wasm_bindgen_test::*;
wasm_bindgen_test_configure!(run_in_dedicated_worker);
#[wasm_bindgen_test]
fn worker_exports_preserve_prefixes_and_structured_errors() {
    let pairs: Value =
        serde_json::from_str(include_str!("../../../examples/web/fixtures/inputs.json")).unwrap();
    let mut session = WasmSession::new("dedicated-worker").unwrap();
    for pair in pairs["pairs"].as_array().unwrap() {
        let analysis: Value = serde_json::from_str(
            &session
                .analyze(
                    pair["parent1_json"].as_str().unwrap(),
                    pair["parent2_json"].as_str().unwrap(),
                    r#"{"api":1,"trace_level":"full"}"#,
                )
                .unwrap(),
        )
        .unwrap();
        let id = analysis["plan_id"].as_str().unwrap();
        let n = analysis["nontrivial"].as_array().unwrap().len();
        let first = session.preview_step(id, 0.0).unwrap();
        for k in 0..=n {
            let frame: Value =
                serde_json::from_str(&session.preview_step(id, k as f64).unwrap()).unwrap();
            assert_eq!(
                frame["selected_indices"],
                Value::Array(analysis["nontrivial"].as_array().unwrap()[..k].to_vec())
            );
        }
        assert_eq!(session.preview_step(id, 0.0).unwrap(), first);
        let error = session.preview_step(id, 0.5).unwrap_err();
        assert_eq!(
            js_sys::Reflect::get(&error, &"code".into())
                .unwrap()
                .as_string()
                .as_deref(),
            Some("invalid_index")
        );
        session.dispose(id).unwrap();
        assert!(session.inspect(id).is_err());
    }
}
