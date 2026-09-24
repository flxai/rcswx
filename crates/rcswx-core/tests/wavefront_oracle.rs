#![cfg(feature = "trace")]
#[path = "support/wavefront_oracle.rs"]
mod oracle;

#[cfg(target_arch = "wasm32")]
use wasm_bindgen_test::wasm_bindgen_test;

#[cfg_attr(target_arch = "wasm32", wasm_bindgen_test)]
#[cfg_attr(not(target_arch = "wasm32"), test)]
fn original_serial_content_and_ordered_trace_are_preserved() {
    let frozen: serde_json::Value =
        serde_json::from_str(include_str!("fixtures/wavefront-oracle-v1.json")).unwrap();
    for case in frozen["cases"].as_array().unwrap() {
        let actual = oracle::capture(
            case["first"].as_str().unwrap(),
            case["second"].as_str().unwrap(),
            case["collapse_corners"].as_bool().unwrap(),
            case["max_events"].as_u64().unwrap() as usize,
        );
        assert_eq!(
            actual, case["expected"],
            "{} collapse={} max_events={}",
            case["name"], case["collapse_corners"], case["max_events"]
        );
    }
}
