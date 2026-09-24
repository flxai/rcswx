//! Synchronous JSON-text boundary; a consumer-owned worker supplies asynchrony.
#![forbid(unsafe_code)]
pub mod dto;
pub mod session;
use dto::{BrowserError, MAX_RESPONSE_BYTES};
use session::{Session, checked_index, encode};
use wasm_bindgen::prelude::*;
fn javascript_error(error: BrowserError) -> JsValue {
    let text = serde_json::to_string(&error)
        .expect("browser errors contain only JSON-safe strings and integers");
    js_sys::JSON::parse(&text).unwrap_or_else(|_| JsValue::from_str(&text))
}
fn response<T: serde::Serialize>(result: dto::Result<T>) -> std::result::Result<String, JsValue> {
    result
        .and_then(|value| encode(&value, MAX_RESPONSE_BYTES))
        .map_err(javascript_error)
}
#[wasm_bindgen]
pub struct WasmSession {
    inner: Session,
}
#[wasm_bindgen]
impl WasmSession {
    #[wasm_bindgen(constructor)]
    pub fn new(session_id: &str) -> std::result::Result<WasmSession, JsValue> {
        Session::new(session_id)
            .map(|inner| Self { inner })
            .map_err(javascript_error)
    }
    pub fn version(&self) -> std::result::Result<String, JsValue> {
        response(Ok(self.inner.version()))
    }
    pub fn analyze(
        &mut self,
        parent1_json: &str,
        parent2_json: &str,
        options_json: &str,
    ) -> std::result::Result<String, JsValue> {
        response(self.inner.analyze(parent1_json, parent2_json, options_json))
    }
    pub fn inspect(&self, plan_id: &str) -> std::result::Result<String, JsValue> {
        response(self.inner.inspect(plan_id))
    }
    pub fn preview_step(&mut self, plan_id: &str, k: f64) -> std::result::Result<String, JsValue> {
        response(checked_index(k, "selection").and_then(|k| self.inner.preview_step(plan_id, k)))
    }
    pub fn apply_selection(
        &self,
        plan_id: &str,
        path_indices_json: &str,
    ) -> std::result::Result<String, JsValue> {
        response(self.inner.apply_selection(plan_id, path_indices_json))
    }
    pub fn sample(
        &self,
        plan_id: &str,
        seed_hex: &str,
        skewness: f64,
    ) -> std::result::Result<String, JsValue> {
        response(self.inner.sample(plan_id, seed_hex, skewness))
    }
    pub fn trace_page(
        &self,
        plan_id: &str,
        offset: f64,
        limit: f64,
    ) -> std::result::Result<String, JsValue> {
        response(checked_index(offset, "trace").and_then(|offset| {
            checked_index(limit, "trace")
                .and_then(|limit| self.inner.trace_page(plan_id, offset, limit))
        }))
    }
    pub fn dispose(&mut self, plan_id: &str) -> std::result::Result<String, JsValue> {
        response(Ok(
            serde_json::json!({"disposed":self.inner.dispose(plan_id)}),
        ))
    }
}
