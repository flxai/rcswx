use pyo3::{create_exception, exceptions::PyRuntimeError, prelude::*, types::PyList};
use rcswx_core as core;
use std::mem::size_of;
use std::panic::{AssertUnwindSafe, catch_unwind};
use std::sync::Arc;

create_exception!(_core, CoreError, PyRuntimeError);

pub fn native_error(error: core::Error) -> PyErr {
    match error {
        core::Error::BudgetExceeded {
            resource,
            limit,
            observed,
        } => CoreError::new_err(("budget", resource, limit, observed)),
        core::Error::Deadline { limit, observed } => {
            CoreError::new_err(("budget", "deadline_seconds", limit, observed))
        }
        core::Error::Cancelled => CoreError::new_err(("cancelled",)),
        core::Error::InvalidInput(message) => CoreError::new_err(("invalid_input", message)),
        core::Error::InvalidSelection(message) => {
            CoreError::new_err(("invalid_selection", message))
        }
        core::Error::NoAdmissibleAlignment => CoreError::new_err(("no_alignment",)),
        core::Error::InternalInvariant(message) => CoreError::new_err(("internal", message)),
    }
}

pub fn guarded<T>(operation: impl FnOnce() -> core::Result<T>) -> PyResult<T> {
    match catch_unwind(AssertUnwindSafe(operation)) {
        Ok(result) => result.map_err(native_error),
        Err(_) => Err(CoreError::new_err(("internal", "native invariant panic"))),
    }
}

#[pyclass(name = "Limits", frozen)]
pub struct PyLimits {
    pub inner: core::Limits,
}
#[pymethods]
impl PyLimits {
    #[new]
    #[pyo3(signature = (*, max_nodes=2048, max_states=250_000, max_predecessors=500_000, max_core_bytes=268_435_456, max_edits=20, max_masks=1_048_576, deadline_seconds=30.0))]
    fn new(
        max_nodes: usize,
        max_states: usize,
        max_predecessors: usize,
        max_core_bytes: usize,
        max_edits: usize,
        max_masks: usize,
        deadline_seconds: f64,
    ) -> PyResult<Self> {
        let inner = core::Limits {
            max_nodes,
            max_states,
            max_predecessors,
            max_core_bytes,
            max_edits,
            max_masks,
            deadline_seconds,
        };
        inner.validate().map_err(native_error)?;
        Ok(Self { inner })
    }
}

#[pyclass(name = "Architecture", frozen)]
pub struct PyArchitecture {
    pub inner: Arc<core::Architecture>,
}
#[pymethods]
impl PyArchitecture {
    #[new]
    fn new(py: Python<'_>, records: &Bound<'_, PyList>, limits: &PyLimits) -> PyResult<Self> {
        let limits = limits.inner.clone();
        limits.validate().map_err(native_error)?;
        let count = records.len();
        core::check_limit("max_nodes", limits.max_nodes, count).map_err(native_error)?;
        let bytes = count
            .checked_mul(size_of::<core::Op>())
            .and_then(|n| n.checked_add(size_of::<core::Architecture>()))
            .ok_or_else(|| {
                CoreError::new_err(("budget", "max_core_bytes", limits.max_core_bytes, u64::MAX))
            })?;
        core::check_limit("max_core_bytes", limits.max_core_bytes, bytes).map_err(native_error)?;
        let mut ops = Vec::new();
        ops.try_reserve_exact(count).map_err(|_| {
            CoreError::new_err(("budget", "allocation", limits.max_core_bytes, bytes))
        })?;
        for record in records.iter() {
            let (kind, width, bias, param, momentum, track) =
                record.extract::<(u8, u64, bool, f64, Option<f64>, bool)>()?;
            ops.push(core::Op {
                kind,
                width,
                bias,
                param,
                momentum,
                track,
            });
        }
        let inner = py.detach(move || guarded(|| core::Architecture::new(ops, &limits)))?;
        Ok(Self {
            inner: Arc::new(inner),
        })
    }
    #[getter]
    fn owned_bytes(&self) -> usize {
        self.inner.owned_bytes()
    }
}

#[pyclass(name = "CancellationToken", frozen)]
#[derive(Default)]
pub struct PyCancellationToken {
    pub inner: core::CancellationToken,
}
#[pymethods]
impl PyCancellationToken {
    #[new]
    fn new() -> Self {
        Self::default()
    }
    fn cancel(&self) {
        self.inner.cancel();
    }
    fn is_cancelled(&self) -> bool {
        self.inner.is_cancelled()
    }
}

pub fn register(module: &Bound<'_, PyModule>) -> PyResult<()> {
    module.add("CoreError", module.py().get_type::<CoreError>())?;
    module.add_class::<PyLimits>()?;
    module.add_class::<PyArchitecture>()?;
    module.add_class::<PyCancellationToken>()?;
    Ok(())
}
