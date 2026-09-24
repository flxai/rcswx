//! Portable, bounded observations of real engine work, independent of its quotas.
use serde::{Deserialize, Serialize};
use serde_json::Value;
use std::io::{self, Write};

pub const SCHEMA_VERSION: u32 = 1;
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum TraceLevel {
    #[default]
    None,
    Summary,
    Full,
}
#[derive(Clone, Copy, Debug, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct TraceLimits {
    pub max_events: usize,
    pub max_bytes: usize,
}
impl Default for TraceLimits {
    fn default() -> Self {
        Self {
            max_events: 50_000,
            max_bytes: 8 * 1024 * 1024,
        }
    }
}
#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct TraceEvent {
    pub seq: usize,
    pub kind: String,
    pub data: Value,
}
#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct Recording {
    pub schema: u32,
    pub level: TraceLevel,
    pub events: Vec<TraceEvent>,
    pub complete: bool,
    pub bytes: usize,
    pub truncation_reason: Option<String>,
    pub last_complete_event: Option<usize>,
}
/// A serializer sink that counts bytes without allocating an unbounded buffer.
struct Counter {
    size: usize,
    limit: usize,
}
impl Write for Counter {
    fn write(&mut self, bytes: &[u8]) -> io::Result<usize> {
        self.size = self
            .size
            .checked_add(bytes.len())
            .ok_or_else(|| io::Error::other("size overflow"))?;
        if self.size > self.limit {
            return Err(io::Error::other("recording limit"));
        }
        Ok(bytes.len())
    }
    fn flush(&mut self) -> io::Result<()> {
        Ok(())
    }
}
pub struct Recorder {
    recording: Recording,
    limits: TraceLimits,
}
impl Recorder {
    pub fn new(level: TraceLevel, limits: TraceLimits) -> Self {
        Self {
            recording: Recording {
                schema: SCHEMA_VERSION,
                level,
                events: Vec::new(),
                complete: true,
                bytes: 0,
                truncation_reason: None,
                last_complete_event: None,
            },
            limits,
        }
    }
    pub fn enabled(&self) -> bool {
        self.recording.level != TraceLevel::None && self.recording.complete
    }
    pub fn full(&self) -> bool {
        self.recording.level == TraceLevel::Full && self.recording.complete
    }
    pub fn level(&self) -> TraceLevel {
        self.recording.level
    }
    /// The closure is never evaluated when observation is disabled or truncated.
    pub fn emit(&mut self, kind: &str, payload: impl FnOnce() -> Value) {
        if !self.enabled() {
            return;
        }
        if self.recording.events.len() >= self.limits.max_events {
            self.truncate("event_limit");
            return;
        }
        let event = TraceEvent {
            seq: self.recording.events.len(),
            kind: kind.into(),
            data: payload(),
        };
        let mut counter = Counter {
            size: 0,
            limit: self.limits.max_bytes.saturating_sub(self.recording.bytes),
        };
        if serde_json::to_writer(&mut counter, &event).is_err() {
            self.truncate("byte_limit");
            return;
        }
        self.recording.bytes += counter.size;
        self.recording.last_complete_event = Some(event.seq);
        self.recording.events.push(event);
    }
    fn truncate(&mut self, reason: &str) {
        self.recording.complete = false;
        self.recording.truncation_reason = Some(reason.into());
    }
    pub fn finish(self) -> Recording {
        self.recording
    }
}
/// Do not conflate an uncomputed cell with a computed NaN or either infinity.
pub fn number(value: f64, uncomputed: bool) -> Value {
    if uncomputed {
        serde_json::json!({"state":"uncomputed"})
    } else if value.is_nan() {
        serde_json::json!({"state":"nan"})
    } else if value == f64::INFINITY {
        serde_json::json!({"state":"positive_infinity"})
    } else if value == f64::NEG_INFINITY {
        serde_json::json!({"state":"negative_infinity"})
    } else {
        serde_json::json!(value)
    }
}
#[derive(Clone, Debug, PartialEq, Eq, PartialOrd, Ord, Serialize, Deserialize)]
pub struct SourceOccurrence {
    pub parent: u8,
    pub occurrence_index: usize,
}
#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct OccurrenceOrigin {
    pub occurrence_index: usize,
    pub sources: Vec<SourceOccurrence>,
    pub synthesized: bool,
}
#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct ObservedApplication {
    pub result: crate::apply::ApplicationResult,
    pub origins: Vec<OccurrenceOrigin>,
}
