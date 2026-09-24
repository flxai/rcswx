//! Versioned, tensor-free architecture data. External logical IDs are decimal
//! integers, not arena indices: legacy callers may use repeated arbitrary-width IDs.
use crate::error::{Error, Result};
use num_bigint::BigInt;
use serde::{Deserialize, Serialize};
use serde_json::value::RawValue;
use std::collections::HashSet;
use std::str::FromStr;

pub const SCHEMA_VERSION: u32 = 1;

#[derive(Clone, Debug, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct Node {
    pub id: String,
    pub name: String,
    #[serde(default)]
    pub children: Vec<usize>,
    /// Uninterpreted JSON: raw values preserve large numbers and arbitrary keys.
    #[serde(default)]
    pub parameters: Option<Box<RawValue>>,
    #[serde(default)]
    pub provenance: Option<Box<RawValue>>,
}

#[derive(Clone, Debug, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct Architecture {
    pub schema: u32,
    pub grammar: String,
    pub grammar_version: String,
    pub root: usize,
    pub nodes: Vec<Node>,
    #[serde(default)]
    pub input_spec: Option<Box<RawValue>>,
}

impl Architecture {
    pub fn from_json(text: &str) -> Result<Self> {
        let architecture: Self =
            serde_json::from_str(text).map_err(|error| Error::InvalidInput(error.to_string()))?;
        architecture.validate()?;
        Ok(architecture)
    }

    pub fn to_json(&self) -> Result<String> {
        self.validate()?;
        serde_json::to_string(self).map_err(|error| Error::InvalidInput(error.to_string()))
    }

    pub fn validate(&self) -> Result<()> {
        if self.schema != SCHEMA_VERSION {
            return Err(Error::InvalidInput(
                "unsupported architecture schema".into(),
            ));
        }
        if self.grammar.is_empty() || self.grammar_version.is_empty() {
            return Err(Error::InvalidInput(
                "grammar identity and version are required".into(),
            ));
        }
        if self.root >= self.nodes.len() {
            return Err(Error::InvalidInput(
                "architecture root is out of bounds".into(),
            ));
        }
        let mut seen = HashSet::with_capacity(self.nodes.len());
        let mut pending = vec![self.root];
        while let Some(index) = pending.pop() {
            let node = self
                .nodes
                .get(index)
                .ok_or_else(|| Error::InvalidInput("child reference is out of bounds".into()))?;
            if !seen.insert(index) {
                return Err(Error::InvalidInput(
                    "shared or cyclic tree occurrence".into(),
                ));
            }
            let id = BigInt::from_str(&node.id)
                .map_err(|_| Error::InvalidInput("logical IDs must be decimal integers".into()))?;
            if id.to_string() != node.id {
                return Err(Error::InvalidInput(
                    "logical IDs must use canonical decimal encoding".into(),
                ));
            }
            if node.name.is_empty() {
                return Err(Error::InvalidInput("operation identity is empty".into()));
            }
            pending.extend(node.children.iter().rev().copied());
        }
        if seen.len() != self.nodes.len() {
            return Err(Error::InvalidInput(
                "unreachable architecture occurrences".into(),
            ));
        }
        Ok(())
    }

    pub fn preorder(&self) -> Vec<usize> {
        let mut result = Vec::with_capacity(self.nodes.len());
        let mut pending = vec![self.root];
        while let Some(index) = pending.pop() {
            result.push(index);
            pending.extend(self.nodes[index].children.iter().rev().copied());
        }
        result
    }

    pub fn parents(&self) -> Vec<Option<usize>> {
        let mut parents = vec![None; self.nodes.len()];
        for (index, node) in self.nodes.iter().enumerate() {
            for &child in &node.children {
                parents[child] = Some(index);
            }
        }
        parents
    }
}

/// Limits are operational, never a declaration that a larger tree is invalid.
#[derive(Clone, Debug, Default, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct Limits {
    pub max_work: Option<usize>,
    pub max_output: Option<usize>,
    pub max_allocation_bytes: Option<usize>,
}

pub struct Budget<'a> {
    pub limits: Limits,
    pub work: usize,
    pub output: usize,
    pub allocation_bytes: usize,
    pub check: &'a mut dyn FnMut() -> Result<()>,
}

impl Budget<'_> {
    pub fn account(&mut self, work: usize, output: usize, allocation: usize) -> Result<()> {
        self.work = self.work.checked_add(work).ok_or(Error::Limit("work"))?;
        self.output = self
            .output
            .checked_add(output)
            .ok_or(Error::Limit("output"))?;
        self.allocation_bytes = self
            .allocation_bytes
            .checked_add(allocation)
            .ok_or(Error::Limit("allocation"))?;
        if self.limits.max_work.is_some_and(|limit| self.work > limit) {
            return Err(Error::Limit("work"));
        }
        if self
            .limits
            .max_output
            .is_some_and(|limit| self.output > limit)
        {
            return Err(Error::Limit("output"));
        }
        if self
            .limits
            .max_allocation_bytes
            .is_some_and(|limit| self.allocation_bytes > limit)
        {
            return Err(Error::Limit("allocation"));
        }
        Ok(())
    }

    pub fn checkpoint(&mut self, work: usize, output: usize, allocation: usize) -> Result<()> {
        self.account(work, output, allocation)?;
        (self.check)()
    }
}
