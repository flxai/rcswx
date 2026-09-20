use std::fmt;
use std::mem::size_of;
use std::sync::{
    Arc,
    atomic::{AtomicBool, Ordering},
};
use std::time::Instant;

#[derive(Debug, Clone)]
pub enum Error {
    BudgetExceeded {
        resource: &'static str,
        limit: u64,
        observed: u64,
    },
    Deadline {
        limit: f64,
        observed: f64,
    },
    Cancelled,
    InvalidInput(String),
    InvalidSelection(String),
    NoAdmissibleAlignment,
    InternalInvariant(String),
}
impl fmt::Display for Error {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "{self:?}")
    }
}
impl std::error::Error for Error {}
pub type Result<T, E = Error> = std::result::Result<T, E>;

pub fn check_limit(resource: &'static str, limit: usize, observed: usize) -> Result<()> {
    if observed > limit {
        Err(Error::BudgetExceeded {
            resource,
            limit: limit as u64,
            observed: observed as u64,
        })
    } else {
        Ok(())
    }
}

#[derive(Clone, Debug)]
pub struct Limits {
    pub max_nodes: usize,
    pub max_states: usize,
    pub max_predecessors: usize,
    pub max_core_bytes: usize,
    pub max_edits: usize,
    pub max_masks: usize,
    pub deadline_seconds: f64,
}
impl Default for Limits {
    fn default() -> Self {
        Self {
            max_nodes: 2048,
            max_states: 250_000,
            max_predecessors: 500_000,
            max_core_bytes: 256 * 1024 * 1024,
            max_edits: 20,
            max_masks: 1_048_576,
            deadline_seconds: 30.0,
        }
    }
}
impl Limits {
    pub fn validate(&self) -> Result<()> {
        for (name, value, min, max) in [
            ("max_nodes", self.max_nodes, 1, 1_000_000),
            ("max_states", self.max_states, 1, 10_000_000),
            ("max_predecessors", self.max_predecessors, 1, 20_000_000),
            (
                "max_core_bytes",
                self.max_core_bytes,
                1,
                4 * 1024_usize.pow(3),
            ),
            ("max_edits", self.max_edits, 0, 24),
            ("max_masks", self.max_masks, 1, 1 << 24),
        ] {
            if value < min || value > max {
                return Err(Error::InvalidInput(format!(
                    "{name} outside [{min}, {max}]"
                )));
            }
        }
        if !self.deadline_seconds.is_finite()
            || self.deadline_seconds <= 0.0
            || self.deadline_seconds > 3600.0
        {
            return Err(Error::InvalidInput("invalid deadline_seconds".into()));
        }
        Ok(())
    }
}

#[derive(Clone, Default)]
pub struct CancellationToken(Arc<AtomicBool>);
impl CancellationToken {
    pub fn cancel(&self) {
        self.0.store(true, Ordering::Relaxed);
    }
    pub fn is_cancelled(&self) -> bool {
        self.0.load(Ordering::Relaxed)
    }
}

/// Immutable scalar descriptors. Input-dependent dimensions are deliberately absent.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct Op {
    pub kind: u8,
    pub width: u64,
    pub bias: bool,
    pub param: f64,
    pub momentum: Option<f64>,
    pub track: bool,
}
impl Op {
    pub fn validate(&self) -> Result<()> {
        let unused = self.width == 0
            && !self.bias
            && self.param == 0.0
            && self.momentum.is_none()
            && !self.track;
        let valid = match self.kind {
            0 | 2 => unused,
            1 => {
                self.width > 0
                    && self.width <= i64::MAX as u64
                    && self.param == 0.0
                    && self.momentum.is_none()
                    && !self.track
            }
            3 => {
                self.width == 0
                    && !self.bias
                    && self.param.is_finite()
                    && (0.0..=1.0).contains(&self.param)
                    && self.momentum.is_none()
                    && !self.track
            }
            4 => {
                self.width == 0
                    && self.param.is_finite()
                    && self.param > 0.0
                    && self
                        .momentum
                        .is_none_or(|m| m.is_finite() && (0.0..=1.0).contains(&m))
            }
            _ => false,
        };
        if valid {
            Ok(())
        } else {
            Err(Error::InvalidInput(
                "noncanonical or unsupported operation descriptor".into(),
            ))
        }
    }
}

#[derive(Debug)]
pub struct Architecture {
    pub(crate) ops: Vec<Op>,
}
impl Architecture {
    pub fn new(ops: Vec<Op>, limits: &Limits) -> Result<Self> {
        let value = Self { ops };
        value.validate(limits)?;
        Ok(value)
    }
    pub fn operations(&self) -> &[Op] {
        &self.ops
    }
    pub fn validate(&self, limits: &Limits) -> Result<()> {
        let mut guard = Guard::new(limits, &CancellationToken::default())?;
        if self.ops.is_empty() {
            return Err(Error::InvalidInput(
                "empty architecture; normalize root Identity first".into(),
            ));
        }
        check_limit("max_nodes", limits.max_nodes, self.ops.len())?;
        guard.bytes(self.owned_bytes())?;
        for op in &self.ops {
            op.validate()?;
            guard.tick()?;
        }
        guard.check()
    }
    pub fn owned_bytes(&self) -> usize {
        size_of::<Self>() + self.ops.capacity() * size_of::<Op>()
    }
}

#[derive(Clone, Copy, Debug, Default)]
pub struct Stats {
    pub states: usize,
    pub predecessors: usize,
    pub charged_bytes: usize,
}

/// Per-request accounting. Charged bytes include input capacities and reserved scratch;
/// allocator overhead and the caller's Python metadata are not claimed as RSS.
pub(crate) struct Guard {
    pub limits: Limits,
    token: CancellationToken,
    start: Instant,
    steps: usize,
}
impl Guard {
    pub fn new(limits: &Limits, token: &CancellationToken) -> Result<Self> {
        limits.validate()?;
        let guard = Self {
            limits: limits.clone(),
            token: token.clone(),
            start: Instant::now(),
            steps: 0,
        };
        guard.check()?;
        Ok(guard)
    }
    pub fn check(&self) -> Result<()> {
        if self.token.is_cancelled() {
            return Err(Error::Cancelled);
        }
        let elapsed = self.start.elapsed().as_secs_f64();
        if elapsed > self.limits.deadline_seconds {
            return Err(Error::Deadline {
                limit: self.limits.deadline_seconds,
                observed: elapsed,
            });
        }
        Ok(())
    }
    pub fn tick(&mut self) -> Result<()> {
        self.steps = (self.steps + 1) & 1023;
        if self.steps == 0 {
            self.check()?;
        }
        Ok(())
    }
    pub fn bytes(&self, bytes: usize) -> Result<()> {
        self.check()?;
        check_limit("max_core_bytes", self.limits.max_core_bytes, bytes)
    }
}

pub(crate) fn allocation<T>(count: usize, base_bytes: usize, guard: &Guard) -> Result<Vec<T>> {
    let bytes = count
        .checked_mul(size_of::<T>())
        .and_then(|n| n.checked_add(base_bytes))
        .ok_or(Error::BudgetExceeded {
            resource: "max_core_bytes",
            limit: guard.limits.max_core_bytes as u64,
            observed: u64::MAX,
        })?;
    guard.bytes(bytes)?;
    let mut result = Vec::new();
    result
        .try_reserve_exact(count)
        .map_err(|_| Error::BudgetExceeded {
            resource: "allocation",
            limit: guard.limits.max_core_bytes as u64,
            observed: bytes as u64,
        })?;
    Ok(result)
}
