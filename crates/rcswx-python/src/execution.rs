//! One worker parser for the public Python preflight and every native entry point.
use pyo3::exceptions::{PyTypeError, PyValueError};
use pyo3::prelude::*;
use pyo3::types::{PyBool, PyInt};
use rcswx_core::execution::{PARALLEL_CAPABLE, Workers};

#[derive(Clone, Copy)]
pub(crate) struct WorkerCount(pub Workers);
impl WorkerCount {
    pub const SERIAL: Self = Self(Workers::SERIAL);
}
impl<'a, 'py> FromPyObject<'a, 'py> for WorkerCount {
    type Error = PyErr;
    fn extract(value: Borrowed<'a, 'py, PyAny>) -> PyResult<Self> {
        if !value.is_instance_of::<PyInt>() || value.is_instance_of::<PyBool>() {
            return Err(PyTypeError::new_err(
                "workers must be an integer, not a boolean",
            ));
        }
        // Check invalid negative integers before narrowing arbitrary-precision
        // Python values. Positive isize overflow remains an OverflowError.
        if value.lt(-1)? || value.eq(0)? {
            return Err(PyValueError::new_err(
                "workers must be -1 or a positive integer",
            ));
        }
        let requested = value.extract::<isize>()?;
        Workers::new(requested)
            .map(Self)
            .map_err(|error| crate::engine::python_error(error, None))
    }
}

#[pyfunction]
fn validate_workers(workers: WorkerCount) -> isize {
    workers.0.requested()
}

pub fn register(module: &Bound<'_, PyModule>) -> PyResult<()> {
    module.add("PARALLEL_CAPABLE", PARALLEL_CAPABLE)?;
    module.add_function(wrap_pyfunction!(validate_workers, module)?)?;
    Ok(())
}
