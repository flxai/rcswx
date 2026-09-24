use crate::dto::*;
use rcswx_core::{
    apply::apply_traced,
    architecture::{Architecture, Budget},
    edit_plan::{EditPlan, analyze_traced},
    error::Error,
    sampling::{NativeRng, sample_with_budget},
    tokens::prepare,
    trace::{Recorder, Recording},
};
use serde::Serialize;
use serde_json::{Value, json};
use sha2::{Digest, Sha256};
use std::{
    collections::{BTreeMap, VecDeque},
    io::{self, Write},
};

struct BoundedWriter {
    bytes: Vec<u8>,
    limit: usize,
}
impl Write for BoundedWriter {
    fn write(&mut self, data: &[u8]) -> io::Result<usize> {
        if data.len() > self.limit.saturating_sub(self.bytes.len()) {
            return Err(io::Error::other("response byte limit"));
        }
        self.bytes.extend_from_slice(data);
        Ok(data.len())
    }
    fn flush(&mut self) -> io::Result<()> {
        Ok(())
    }
}
pub fn encode<T: Serialize>(value: &T, limit: usize) -> Result<String> {
    let mut out = BoundedWriter {
        bytes: Vec::new(),
        limit,
    };
    serde_json::to_writer(&mut out, value)
        .map_err(|e| BrowserError::new("response_limit", "serialize", e))?;
    String::from_utf8(out.bytes)
        .map_err(|e| BrowserError::new("serialization_failure", "serialize", e))
}
fn core_error(error: Error, stage: &str) -> BrowserError {
    let code = match &error {
        Error::InvalidInput(_) => {
            if stage == "selection" {
                "invalid_selection"
            } else {
                "invalid_input"
            }
        }
        Error::Limit(_) => "core_quota",
        Error::Memory => "allocation_failure",
        Error::Cancelled => "cancelled",
        Error::PlanMismatch => "wrong_plan",
        Error::Numerical(_) => "sampling_failure",
        _ => {
            if stage == "application" {
                "application_failure"
            } else {
                "reference_failure"
            }
        }
    };
    BrowserError::new(code, stage, error)
}
fn budgeted<T>(
    limits: &CoreLimits,
    stage: &str,
    f: impl FnOnce(&mut Budget<'_>) -> rcswx_core::error::Result<T>,
) -> Result<T> {
    let mut check = || Ok(());
    let mut budget = Budget {
        limits: limits.core(),
        work: 0,
        output: 0,
        allocation_bytes: 0,
        check: &mut check,
        wait_check: None,
    };
    f(&mut budget).map_err(|e| core_error(e, stage))
}
pub fn checked_index(value: f64, stage: &str) -> Result<usize> {
    if !value.is_finite() || value.fract() != 0.0 || value < 0.0 || value > u32::MAX as f64 {
        return Err(BrowserError::new(
            "invalid_index",
            stage,
            "expected a nonnegative wasm32 integer",
        ));
    }
    Ok(value as usize)
}
fn decode_architecture(text: &str) -> Result<Architecture> {
    if text.len() > MAX_INPUT_BYTES {
        return Err(BrowserError::new(
            "input_limit",
            "input",
            "architecture JSON exceeds byte limit",
        ));
    }
    let architecture: Architecture =
        serde_json::from_str(text).map_err(|e| BrowserError::new("invalid_input", "input", e))?;
    if architecture.nodes.len() > MAX_NODES {
        return Err(BrowserError::new(
            "input_limit",
            "input",
            "occurrence count exceeds browser limit",
        ));
    }
    architecture
        .validate()
        .map_err(|e| core_error(e, "input"))?;
    let mut pending = vec![(architecture.root, 0)];
    while let Some((index, depth)) = pending.pop() {
        if depth > MAX_DEPTH {
            return Err(BrowserError::new(
                "input_limit",
                "input",
                "architecture depth exceeds browser limit",
            ));
        }
        pending.extend(
            architecture.nodes[index]
                .children
                .iter()
                .map(|&i| (i, depth + 1)),
        );
    }
    Ok(architecture)
}
fn options(text: &str) -> Result<AnalyzeOptions> {
    if text.len() > 4096 {
        return Err(BrowserError::new(
            "input_limit",
            "options",
            "options JSON exceeds byte limit",
        ));
    }
    let options: AnalyzeOptions =
        serde_json::from_str(text).map_err(|e| BrowserError::new("invalid_input", "options", e))?;
    if options.api != API_VERSION {
        return Err(BrowserError::new(
            "unsupported_api",
            "options",
            "unsupported API version",
        ));
    }
    let defaults = AnalyzeOptions::default();
    if options.limits.max_work > defaults.limits.max_work
        || options.limits.max_output > defaults.limits.max_output
        || options.limits.max_allocation_bytes > defaults.limits.max_allocation_bytes
        || options.trace_limits.max_events > defaults.trace_limits.max_events
        || options.trace_limits.max_bytes > defaults.trace_limits.max_bytes
    {
        return Err(BrowserError::new(
            "input_limit",
            "options",
            "per-call limits may lower, not exceed, browser policy",
        ));
    }
    Ok(options)
}
fn token_views(
    tokens: &[rcswx_core::tokens::PreparedToken],
    identities: &[String],
) -> Vec<TokenView> {
    tokens
        .iter()
        .enumerate()
        .map(|(index, t)| TokenView {
            index,
            id: identities[t.token.id as usize].clone(),
            name: t.token.name.clone(),
            children: t.token.children.clone(),
            parent_arity: t.token.parent_arity,
            occurrence: t.occurrence,
        })
        .collect()
}
struct Retained {
    plan: EditPlan,
    options: AnalyzeOptions,
    recording: Recording,
    summary: Analysis,
    bytes: usize,
    snapshot_bytes: usize,
    cache: VecDeque<(usize, Applied, usize)>,
}
pub struct Session {
    id: String,
    next: u64,
    plans: BTreeMap<String, Retained>,
    retained_bytes: usize,
    cache_bytes: usize,
}
impl Session {
    pub fn new(id: &str) -> Result<Self> {
        if id.is_empty() || id.len() > 128 {
            return Err(BrowserError::new(
                "invalid_input",
                "session",
                "session identifier must contain 1..128 UTF-8 bytes",
            ));
        }
        Ok(Self {
            id: id.into(),
            next: 0,
            plans: BTreeMap::new(),
            retained_bytes: 0,
            cache_bytes: 0,
        })
    }
    pub fn version(&self) -> Value {
        json!({"package":"rcswx-wasm","version":env!("CARGO_PKG_VERSION"),"build":option_env!("RCSWX_BUILD_ID").unwrap_or("development"),"api":API_VERSION,"architecture_schema":1,"trace_schema":1,"sampler":"ChaCha12/rand_chacha-0.9","limits":{"input_bytes":MAX_INPUT_BYTES,"nodes":MAX_NODES,"depth":MAX_DEPTH,"plans":MAX_PLANS,"session_bytes":MAX_SESSION_BYTES,"cache_bytes":MAX_CACHE_BYTES,"response_bytes":MAX_RESPONSE_BYTES,"trace_page_events":MAX_TRACE_PAGE,"trace_page_bytes":MAX_TRACE_PAGE_BYTES,"defaults":AnalyzeOptions::default()}})
    }
    fn retained(&self, id: &str) -> Result<&Retained> {
        self.plans.get(id).ok_or_else(|| {
            BrowserError::new(
                if id.starts_with(&format!("{}:", self.id)) {
                    "expired_plan"
                } else {
                    "wrong_plan"
                },
                "plan",
                "plan does not belong to this live session",
            )
        })
    }
    pub fn analyze(&mut self, first: &str, second: &str, options_json: &str) -> Result<Analysis> {
        if self.plans.len() >= MAX_PLANS {
            return Err(BrowserError::new(
                "session_limit",
                "analysis",
                "dispose a retained plan before analyzing another pair",
            ));
        }
        let input_texts = (first, second);
        let options = options(options_json)?;
        let first = decode_architecture(first)?;
        let second = decode_architecture(second)?;
        let prepared = prepare(first, second).map_err(|e| core_error(e, "preparation"))?;
        self.next = self.next.checked_add(1).ok_or_else(|| {
            BrowserError::new(
                "session_limit",
                "session",
                "plan identifier space exhausted",
            )
        })?;
        let id = format!("{}:{}", self.id, self.next);
        let mut recorder = Recorder::new(options.trace_level, options.trace_limits);
        recorder.emit("header",||json!({"api":API_VERSION,"architecture_schema":1,"engine_version":env!("CARGO_PKG_VERSION"),"build":option_env!("RCSWX_BUILD_ID").unwrap_or("development"),"options":options,"direction":"parent2_to_parent1","input_hashes":{"first":format!("{:x}",Sha256::digest(input_texts.0.as_bytes())),"second":format!("{:x}",Sha256::digest(input_texts.1.as_bytes()))}}));
        let plan = budgeted(&options.limits, "analysis", |budget| {
            analyze_traced(
                prepared,
                options.collapse_corners,
                id.clone(),
                budget,
                &mut recorder,
            )
        })?;
        if !plan.distance.is_finite() {
            return Err(BrowserError::new(
                "reference_failure",
                "analysis",
                "alignment did not produce a finite distance",
            ));
        }
        let recording = recorder.finish();
        let summary = Analysis {
            api: API_VERSION,
            plan_id: id.clone(),
            distance: plan.distance,
            path_index: plan.path_index,
            history_count: plan.paths.len(),
            path: plan.paths[plan.path_index].clone(),
            operations: plan.operations.clone(),
            operations_unordered: plan.operations_unordered.clone(),
            nontrivial: plan.nontrivial.clone(),
            parents: Parents {
                first: DisplaySnapshot::new("parent1", &plan.prepared.first),
                second: DisplaySnapshot::new("parent2", &plan.prepared.second),
            },
            tokens: Parents {
                first: token_views(&plan.prepared.first_tokens, &plan.prepared.identities),
                second: token_views(&plan.prepared.second_tokens, &plan.prepared.identities),
            },
            trace: TraceSummary::from(&recording),
        };
        let summary_bytes = encode(&summary, MAX_RESPONSE_BYTES)?.len();
        // Conservative accounting for owned Rust payloads and working snapshots;
        // this is not a process-memory or WASM linear-memory cap.
        let snapshot_bytes = (encode(&plan.paths, MAX_SESSION_BYTES)?.len()
            + encode(&plan.prepared.first, MAX_RESPONSE_BYTES)?.len()
            + encode(&plan.prepared.second, MAX_RESPONSE_BYTES)?.len()
            + summary_bytes)
            .saturating_mul(4);
        let bytes = snapshot_bytes
            .saturating_add(recording.bytes.saturating_mul(4))
            .saturating_add(summary_bytes);
        if bytes
            .saturating_add(snapshot_bytes)
            .saturating_add(self.retained_bytes)
            .saturating_add(self.cache_bytes)
            > MAX_SESSION_BYTES
        {
            return Err(BrowserError::new(
                "session_limit",
                "analysis",
                "retained plan and isolated snapshot exceed session byte policy",
            ));
        }
        self.retained_bytes += bytes;
        self.plans.insert(
            id,
            Retained {
                plan,
                options,
                recording,
                summary: summary.clone(),
                bytes,
                snapshot_bytes,
                cache: VecDeque::new(),
            },
        );
        Ok(summary)
    }
    pub fn inspect(&self, id: &str) -> Result<Analysis> {
        Ok(self.retained(id)?.summary.clone())
    }
    fn apply(&self, id: &str, selected: &[usize], step: Option<usize>) -> Result<Applied> {
        let retained = self.retained(id)?;
        retained
            .plan
            .validate_selection(selected)
            .map_err(|e| core_error(e, "selection"))?;
        if self
            .retained_bytes
            .saturating_add(self.cache_bytes)
            .saturating_add(retained.snapshot_bytes)
            > MAX_SESSION_BYTES
        {
            return Err(BrowserError::new(
                "session_limit",
                "application",
                "working snapshot exceeds session policy",
            ));
        }
        let mut plan = retained.plan.clone();
        let mut recorder =
            Recorder::new(retained.options.trace_level, retained.options.trace_limits);
        let applied = budgeted(&retained.options.limits, "application", |budget| {
            apply_traced(&mut plan, selected, true, budget, &mut recorder)
        })?;
        let cost = selected
            .iter()
            .map(|&index| retained.plan.paths[retained.plan.path_index][index].value)
            .sum();
        let result = Applied {
            api: API_VERSION,
            plan_id: id.into(),
            selected_indices: selected.to_vec(),
            cost,
            step,
            architecture_json: applied
                .result
                .architecture
                .to_json()
                .map_err(|e| core_error(e, "application"))?,
            display: DisplaySnapshot::new("child", &applied.result.architecture),
            origins: applied.origins,
            recording: recorder.finish(),
            seed_hex: None,
            skewness: None,
            mask: None,
        };
        encode(&result, MAX_RESPONSE_BYTES)?;
        Ok(result)
    }
    pub fn preview_step(&mut self, id: &str, step: usize) -> Result<Applied> {
        let result = (|| {
            let retained = self.retained(id)?;
            if step > retained.plan.nontrivial.len() {
                return Err(BrowserError::new(
                    "invalid_step",
                    "selection",
                    "step is outside this plan's nontrivial edit list",
                ));
            }
            if let Some((_, result, _)) = retained.cache.iter().find(|(k, _, _)| *k == step) {
                return Ok(result.clone());
            }
            let selected = retained.plan.nontrivial[..step].to_vec();
            let result = self.apply(id, &selected, Some(step))?;
            let bytes = encode(&result, MAX_RESPONSE_BYTES)?.len().saturating_mul(4);
            if bytes <= MAX_CACHE_BYTES
                && self
                    .retained_bytes
                    .saturating_add(bytes)
                    .saturating_add(self.retained(id)?.snapshot_bytes)
                    <= MAX_SESSION_BYTES
            {
                while self.cache_bytes + bytes > MAX_CACHE_BYTES
                    || self.retained_bytes
                        + self.cache_bytes
                        + bytes
                        + self.retained(id)?.snapshot_bytes
                        > MAX_SESSION_BYTES
                {
                    let mut removed = false;
                    for retained in self.plans.values_mut() {
                        if let Some((_, _, size)) = retained.cache.pop_front() {
                            self.cache_bytes -= size;
                            removed = true;
                            break;
                        }
                    }
                    if !removed {
                        break;
                    }
                }
                self.plans
                    .get_mut(id)
                    .unwrap()
                    .cache
                    .push_back((step, result.clone(), bytes));
                self.cache_bytes += bytes;
            }
            Ok(result)
        })();
        result.map_err(|mut e: BrowserError| {
            e.requested_step = Some(step);
            e
        })
    }
    pub fn apply_selection(&self, id: &str, indices_json: &str) -> Result<Applied> {
        if indices_json.len() > MAX_INPUT_BYTES {
            return Err(BrowserError::new(
                "input_limit",
                "selection",
                "selection JSON exceeds byte limit",
            ));
        }
        let raw: Vec<f64> = serde_json::from_str(indices_json)
            .map_err(|e| BrowserError::new("invalid_selection", "selection", e))?;
        let selected = raw
            .into_iter()
            .map(|n| checked_index(n, "selection"))
            .collect::<Result<Vec<_>>>()?;
        self.apply(id, &selected, None)
    }
    pub fn sample(&self, id: &str, seed_hex: &str, skewness: f64) -> Result<Applied> {
        if seed_hex.len() != 64
            || !seed_hex
                .bytes()
                .all(|b| b.is_ascii_digit() || (b'a'..=b'f').contains(&b))
        {
            return Err(BrowserError::new(
                "invalid_seed",
                "sampling",
                "seed must be exactly 64 lowercase hexadecimal characters",
            ));
        }
        if !skewness.is_finite() {
            return Err(BrowserError::new(
                "invalid_input",
                "sampling",
                "skewness must be finite",
            ));
        }
        let mut seed = [0u8; 32];
        for (i, byte) in seed.iter_mut().enumerate() {
            *byte = u8::from_str_radix(&seed_hex[i * 2..i * 2 + 2], 16)
                .map_err(|e| BrowserError::new("invalid_seed", "sampling", e))?;
        }
        let retained = self.retained(id)?;
        let (mask, cost, selected) = if retained.plan.nontrivial.is_empty() {
            (String::new(), 0.0, Vec::new())
        } else {
            let operations = retained
                .plan
                .selection_operations()
                .map_err(|e| core_error(e, "selection"))?;
            let (mask, cost) = budgeted(&retained.options.limits, "sampling", |budget| {
                sample_with_budget(
                    &operations,
                    skewness,
                    &mut NativeRng::from_seed(seed),
                    budget,
                )
            })?;
            let selected = mask
                .bytes()
                .zip(&retained.plan.nontrivial)
                .filter_map(|(bit, &index)| (bit == b'1').then_some(index))
                .collect();
            (mask, cost, selected)
        };
        let mut result = self.apply(id, &selected, None)?;
        result.cost = cost;
        result.seed_hex = Some(seed_hex.into());
        result.skewness = Some(skewness);
        result.mask = Some(mask);
        encode(&result, MAX_RESPONSE_BYTES)?;
        Ok(result)
    }
    pub fn trace_page(&self, id: &str, offset: usize, limit: usize) -> Result<TracePage> {
        let recording = &self.retained(id)?.recording;
        if offset > recording.events.len() || limit == 0 || limit > MAX_TRACE_PAGE {
            return Err(BrowserError::new(
                "invalid_index",
                "trace",
                "invalid trace page offset or limit",
            ));
        }
        let mut events = Vec::new();
        let mut bytes = 0;
        for event in recording.events.iter().skip(offset).take(limit) {
            let size = encode(event, MAX_TRACE_PAGE_BYTES)?.len();
            if bytes + size > MAX_TRACE_PAGE_BYTES {
                break;
            }
            bytes += size;
            events.push(event.clone());
        }
        let next = offset + events.len();
        Ok(TracePage {
            api: API_VERSION,
            plan_id: id.into(),
            offset,
            next_offset: (next < recording.events.len()).then_some(next),
            total: recording.events.len(),
            complete: recording.complete,
            bytes: recording.bytes,
            truncation_reason: recording.truncation_reason.clone(),
            events,
        })
    }
    pub fn recording(&self, id: &str) -> Result<&Recording> {
        Ok(&self.retained(id)?.recording)
    }
    pub fn dispose(&mut self, id: &str) -> bool {
        if let Some(retained) = self.plans.remove(id) {
            self.retained_bytes -= retained.bytes;
            self.cache_bytes -= retained
                .cache
                .iter()
                .map(|(_, _, bytes)| bytes)
                .sum::<usize>();
            true
        } else {
            false
        }
    }
}
