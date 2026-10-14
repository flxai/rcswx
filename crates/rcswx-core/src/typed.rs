//! Lossless semantic occurrences for the ordered, typed recursive RCSWX policy.
//!
//! This surface is deliberately separate from the einspace architecture grammar.
//! IDs identify occurrences; only kind, semantic type, role and scalar value score.

use crate::error::{Error, Result};
use crate::recursive::Kind;
use crate::recursive::ordered::{self, Event};
use serde::{Deserialize, Serialize};
use serde_json::value::RawValue;
use std::collections::HashMap;

pub const TREE_VERSION: &str = "rcswx-typed-ordered-v1";
pub const RESULT_VERSION: &str = "rcswx-typed-ordered-result-v1";
pub const POLICY: &str = "typed-unit-v1";
pub const BACKEND: &str = "rcswx-native-recursive-ordered";

#[derive(Clone, Copy, Debug, Deserialize, PartialEq, Eq)]
#[serde(rename_all = "lowercase")]
pub enum NodeKind {
    Constructor,
    List,
    Tuple,
    Literal,
}

#[derive(Debug, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct Node {
    pub id: String,
    pub kind: NodeKind,
    pub semantic_type: String,
    pub role: String,
    pub value: Box<RawValue>,
    pub children: Vec<String>,
}

#[derive(Debug, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct Tree {
    pub version: String,
    pub root: String,
    pub nodes: Vec<Node>,
}

#[derive(Debug, PartialEq, Eq)]
enum Scalar {
    Null,
    Bool(bool),
    String(String),
    Number(String),
}

fn invalid(message: impl Into<String>) -> Error {
    Error::InvalidInput(message.into())
}

// Compare JSON numbers as exact decimal values, never rounded f64s. Trailing
// zeroes and JSON exponent spelling do not become artificial value mutations.
fn number(text: &str) -> Result<String> {
    let (mantissa, exponent) = text
        .split_once(['e', 'E'])
        .map_or((text, "0"), |parts| parts);
    let exponent: i64 = exponent
        .parse()
        .map_err(|_| invalid("typed number exponent is out of range"))?;
    let negative = mantissa.starts_with('-');
    let mantissa = mantissa.strip_prefix('-').unwrap_or(mantissa);
    let fractional = mantissa
        .split_once('.')
        .map_or(0, |(_, fraction)| fraction.len());
    let mut digits: String = mantissa.chars().filter(|&c| c != '.').collect();
    let trailing = digits.len() - digits.trim_end_matches('0').len();
    digits.truncate(digits.len() - trailing);
    let digits = digits.trim_start_matches('0');
    if digits.is_empty() {
        return Ok("0".into());
    }
    let exponent = exponent
        .checked_sub(i64::try_from(fractional).map_err(|_| invalid("typed number is too long"))?)
        .and_then(|value| value.checked_add(i64::try_from(trailing).ok()?))
        .ok_or_else(|| invalid("typed number exponent is out of range"))?;
    Ok(format!(
        "{}{digits}e{exponent}",
        if negative { "-" } else { "" }
    ))
}

fn scalar(value: &RawValue) -> Result<Scalar> {
    let text = value.get().trim();
    match text {
        "null" => Ok(Scalar::Null),
        "true" => Ok(Scalar::Bool(true)),
        "false" => Ok(Scalar::Bool(false)),
        _ if text.starts_with('"') => serde_json::from_str(text)
            .map(Scalar::String)
            .map_err(|error| invalid(error.to_string())),
        _ if text.starts_with('-') || text.starts_with(|c: char| c.is_ascii_digit()) => {
            number(text).map(Scalar::Number)
        }
        _ => Err(invalid("typed node values must be JSON scalars")),
    }
}

/// An internally validated occurrence tree. The event stream brackets every
/// semantic node, including literals and nullary constructors, exactly once.
pub(crate) struct Prepared<'a> {
    pub nodes: &'a [Node],
    values: Vec<Scalar>,
    pub parents: Vec<Option<usize>>,
    pub events: Vec<Event>,
}

impl Tree {
    /// Parse the wire shape and JSON syntax. Semantic occurrence validation is
    /// performed once, by `compare`, before the native recurrence can run.
    pub fn from_json(text: &str) -> Result<Self> {
        serde_json::from_str(text).map_err(|error| invalid(error.to_string()))
    }

    fn prepare(&self) -> Result<Prepared<'_>> {
        if self.version != TREE_VERSION {
            return Err(invalid("unsupported typed tree version"));
        }
        let mut ids = HashMap::with_capacity(self.nodes.len());
        let mut values = Vec::with_capacity(self.nodes.len());
        for (index, node) in self.nodes.iter().enumerate() {
            if node.id.is_empty() || ids.insert(node.id.as_str(), index).is_some() {
                return Err(invalid("typed occurrence IDs must be nonempty and unique"));
            }
            if node.semantic_type.is_empty() || node.role.is_empty() {
                return Err(invalid(
                    "typed semantic types and argument roles must be nonempty",
                ));
            }
            let value = scalar(&node.value)?;
            match node.kind {
                NodeKind::Constructor => {
                    if !matches!(&value, Scalar::String(name) if !name.is_empty()) {
                        return Err(invalid("typed constructors require a nonempty name value"));
                    }
                }
                NodeKind::List | NodeKind::Tuple if value != Scalar::Null => {
                    return Err(invalid("typed containers require a null value"));
                }
                NodeKind::Literal if !node.children.is_empty() => {
                    return Err(invalid("typed literals cannot have children"));
                }
                _ => {}
            }
            values.push(value);
        }
        let root = *ids
            .get(self.root.as_str())
            .ok_or_else(|| invalid("typed root occurrence is missing"))?;
        let mut seen = vec![false; self.nodes.len()];
        let mut parents = vec![None; self.nodes.len()];
        let event_count = self.nodes.len().checked_mul(2).ok_or(Error::Memory)?;
        let mut pending = Vec::with_capacity(event_count);
        pending.push(Event::Open(root));
        let mut events = Vec::with_capacity(event_count.checked_add(1).ok_or(Error::Memory)?);
        events.push(Event::Start);
        while let Some(event) = pending.pop() {
            if let Event::Open(index) = event {
                if std::mem::replace(&mut seen[index], true) {
                    return Err(invalid("shared or cyclic typed tree occurrence"));
                }
                pending.push(Event::Close(index));
                for child in self.nodes[index].children.iter().rev() {
                    let child = *ids
                        .get(child.as_str())
                        .ok_or_else(|| invalid("typed child occurrence is missing"))?;
                    parents[child] = Some(index);
                    pending.push(Event::Open(child));
                }
            }
            events.push(event);
        }
        if seen.iter().any(|seen| !seen) {
            return Err(invalid("unreachable typed tree occurrence"));
        }
        Ok(Prepared {
            nodes: &self.nodes,
            values,
            events,
            parents,
        })
    }
}

impl Prepared<'_> {
    pub(crate) fn mutation_cost(&self, first: usize, other: &Self, second: usize) -> Option<usize> {
        let one = &self.nodes[first];
        let two = &other.nodes[second];
        (one.kind == two.kind && one.semantic_type == two.semantic_type && one.role == two.role)
            .then(|| usize::from(self.values[first] != other.values[second]))
    }
}

#[derive(Debug, Serialize)]
pub struct Correspondence {
    pub first_id: String,
    pub second_id: String,
}

#[derive(Debug, Serialize)]
pub struct Edit {
    pub operation: &'static str,
    pub first_id: Option<String>,
    pub second_id: Option<String>,
    pub cost: usize,
}

#[derive(Debug, Serialize)]
pub struct Comparison {
    pub version: &'static str,
    pub tree_version: &'static str,
    pub policy: &'static str,
    pub backend: &'static str,
    pub direction: &'static str,
    pub distance: usize,
    pub correspondences: Vec<Correspondence>,
    pub edits: Vec<Edit>,
}

/// Transform the second tree toward the first under typed-unit-v1. The native
/// recurrence preserves immediate parents and retains one path per closure state.
/// Independent tiny mapping tests exercise its cost and pruning behavior.
pub fn compare(
    first: &Tree,
    second: &Tree,
    check: &mut dyn FnMut() -> Result<()>,
) -> Result<Comparison> {
    let first = first.prepare()?;
    let second = second.prepare()?;
    let alignment = ordered::align(&first, &second, check)?;
    let mut correspondences = Vec::with_capacity(first.nodes.len().min(second.nodes.len()));
    let mut edits = Vec::with_capacity(alignment.path.len());
    for step in alignment.path {
        let operation = match step.kind {
            Kind::AddWrap => "insert",
            Kind::RemWrap => "delete",
            Kind::MutWrap if step.value == 0.0 => "match",
            Kind::MutWrap => "substitute",
            _ => continue, // Start/close bookkeeping does not score occurrences.
        };
        let first_id = step
            .node1_id
            .map(|index| first.nodes[index as usize].id.clone());
        let second_id = step
            .node2_id
            .map(|index| second.nodes[index as usize].id.clone());
        if let (Some(first_id), Some(second_id)) = (&first_id, &second_id) {
            correspondences.push(Correspondence {
                first_id: first_id.clone(),
                second_id: second_id.clone(),
            });
        }
        edits.push(Edit {
            operation,
            first_id,
            second_id,
            cost: step.value as usize,
        });
    }
    if edits.iter().map(|edit| edit.cost).sum::<usize>() != alignment.distance {
        return Err(Error::Execution(
            "typed edit cost accounting invariant failed".into(),
        ));
    }
    Ok(Comparison {
        version: RESULT_VERSION,
        tree_version: TREE_VERSION,
        policy: POLICY,
        backend: BACKEND,
        direction: "second-to-first",
        distance: alignment.distance,
        correspondences,
        edits,
    })
}
