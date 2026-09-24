use rcswx_core::{
    architecture::{Architecture, Limits},
    edit_plan::Edit,
    trace::{OccurrenceOrigin, Recording, TraceEvent, TraceLevel, TraceLimits},
};
use serde::{Deserialize, Serialize};

pub const API_VERSION: u32 = 1;
pub const MAX_INPUT_BYTES: usize = 256 * 1024;
pub const MAX_NODES: usize = 256;
pub const MAX_DEPTH: usize = 64;
pub const MAX_PLANS: usize = 4;
pub const MAX_SESSION_BYTES: usize = 32 * 1024 * 1024;
pub const MAX_RESPONSE_BYTES: usize = 2 * 1024 * 1024;
pub const MAX_CACHE_BYTES: usize = 4 * 1024 * 1024;
pub const MAX_TRACE_PAGE: usize = 256;
pub const MAX_TRACE_PAGE_BYTES: usize = 512 * 1024;
#[derive(Clone, Debug, Serialize, Deserialize)]
#[serde(default, deny_unknown_fields)]
pub struct CoreLimits {
    pub max_work: usize,
    pub max_output: usize,
    pub max_allocation_bytes: usize,
}
impl Default for CoreLimits {
    fn default() -> Self {
        Self {
            max_work: 200_000,
            max_output: 20_000,
            max_allocation_bytes: 64 * 1024 * 1024,
        }
    }
}
impl CoreLimits {
    pub fn core(&self) -> Limits {
        Limits {
            max_work: Some(self.max_work),
            max_output: Some(self.max_output),
            max_allocation_bytes: Some(self.max_allocation_bytes),
        }
    }
}
#[derive(Clone, Debug, Serialize, Deserialize)]
#[serde(default, deny_unknown_fields)]
pub struct AnalyzeOptions {
    pub api: u32,
    pub collapse_corners: bool,
    pub trace_level: TraceLevel,
    pub limits: CoreLimits,
    pub trace_limits: TraceLimits,
}
impl Default for AnalyzeOptions {
    fn default() -> Self {
        Self {
            api: API_VERSION,
            collapse_corners: false,
            trace_level: TraceLevel::Summary,
            limits: CoreLimits::default(),
            trace_limits: TraceLimits::default(),
        }
    }
}
#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct BrowserError {
    pub code: String,
    pub stage: String,
    pub message: String,
    pub retryable: bool,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub requested_step: Option<usize>,
}
impl BrowserError {
    pub fn new(code: &str, stage: &str, message: impl ToString) -> Self {
        Self {
            code: code.into(),
            stage: stage.into(),
            message: message.to_string(),
            retryable: false,
            requested_step: None,
        }
    }
}
impl std::fmt::Display for BrowserError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(f, "{} ({}): {}", self.code, self.stage, self.message)
    }
}
impl std::error::Error for BrowserError {}
pub type Result<T, E = BrowserError> = std::result::Result<T, E>;
#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct DisplayNode {
    pub occurrence_index: usize,
    pub id: String,
    pub name: String,
    pub children: Vec<usize>,
}
#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct DisplaySnapshot {
    pub snapshot_key: String,
    pub root: usize,
    pub nodes: Vec<DisplayNode>,
}
impl DisplaySnapshot {
    pub fn new(key: &str, architecture: &Architecture) -> Self {
        Self {
            snapshot_key: key.into(),
            root: architecture.root,
            nodes: architecture
                .nodes
                .iter()
                .enumerate()
                .map(|(occurrence_index, n)| DisplayNode {
                    occurrence_index,
                    id: n.id.clone(),
                    name: n.name.clone(),
                    children: n.children.clone(),
                })
                .collect(),
        }
    }
}
#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct Parents<T> {
    pub first: T,
    pub second: T,
}
#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct TokenView {
    pub index: usize,
    pub id: String,
    pub name: String,
    pub children: Vec<String>,
    pub parent_arity: usize,
    pub occurrence: Option<usize>,
}
#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct TraceSummary {
    pub level: TraceLevel,
    pub complete: bool,
    pub event_count: usize,
    pub bytes: usize,
    pub truncation_reason: Option<String>,
}
impl From<&Recording> for TraceSummary {
    fn from(r: &Recording) -> Self {
        Self {
            level: r.level,
            complete: r.complete,
            event_count: r.events.len(),
            bytes: r.bytes,
            truncation_reason: r.truncation_reason.clone(),
        }
    }
}
#[derive(Clone, Debug, Serialize)]
pub struct Analysis {
    pub api: u32,
    pub plan_id: String,
    pub distance: f64,
    pub path_index: usize,
    pub history_count: usize,
    pub path: Vec<Edit>,
    pub operations: Vec<usize>,
    pub operations_unordered: Vec<usize>,
    pub nontrivial: Vec<usize>,
    pub parents: Parents<DisplaySnapshot>,
    pub tokens: Parents<Vec<TokenView>>,
    pub trace: TraceSummary,
}
#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct Applied {
    pub api: u32,
    pub plan_id: String,
    pub selected_indices: Vec<usize>,
    pub cost: f64,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub step: Option<usize>,
    pub architecture_json: String,
    pub display: DisplaySnapshot,
    pub origins: Vec<OccurrenceOrigin>,
    pub recording: Recording,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub seed_hex: Option<String>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub skewness: Option<f64>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub mask: Option<String>,
}
#[derive(Clone, Debug, Serialize)]
pub struct TracePage {
    pub api: u32,
    pub plan_id: String,
    pub offset: usize,
    pub next_offset: Option<usize>,
    pub total: usize,
    pub complete: bool,
    pub bytes: usize,
    pub truncation_reason: Option<String>,
    pub events: Vec<TraceEvent>,
}
