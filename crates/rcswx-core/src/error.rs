//! Explicit errors at the portable/legacy boundary; no algorithmic panic recovery.
use std::fmt;

#[derive(Clone, Debug, PartialEq, Eq)]
pub enum Error {
    InvalidInput(String),
    Index,
    EmptyMinimum,
    MissingMutationTarget,
    Unbound(&'static str),
    Reference(String),
    Numerical(String),
    Execution(String),
    Memory,
    Limit(&'static str),
    Cancelled,
    Callback,
    PlanMismatch,
}

pub type Result<T> = std::result::Result<T, Error>;

impl From<crate::recursive::Failure> for Error {
    fn from(error: crate::recursive::Failure) -> Self {
        match error {
            crate::recursive::Failure::Index => Self::Index,
            crate::recursive::Failure::EmptyMinimum => Self::EmptyMinimum,
            crate::recursive::Failure::Unbound(name) => Self::Unbound(name),
            crate::recursive::Failure::Memory => Self::Memory,
            crate::recursive::Failure::Callback => Self::Callback,
            crate::recursive::Failure::Operational(error) => error,
        }
    }
}

impl fmt::Display for Error {
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::InvalidInput(message)
            | Self::Reference(message)
            | Self::Numerical(message)
            | Self::Execution(message) => formatter.write_str(message),
            Self::Index => formatter.write_str("list index out of range"),
            Self::EmptyMinimum => formatter.write_str("min() arg is an empty sequence"),
            Self::MissingMutationTarget => formatter.write_str("Nodes to mutate not found"),
            Self::Unbound(name) => write!(
                formatter,
                "local variable '{name}' referenced before assignment"
            ),
            Self::Memory => formatter.write_str("native allocation failed"),
            Self::Limit(kind) => write!(formatter, "native {kind} limit exceeded"),
            Self::Cancelled => formatter.write_str("native work was cancelled"),
            Self::Callback => formatter.write_str("host checkpoint failed"),
            Self::PlanMismatch => formatter.write_str("selection belongs to another plan"),
        }
    }
}

impl std::error::Error for Error {}
