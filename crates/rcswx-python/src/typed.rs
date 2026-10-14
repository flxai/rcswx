//! JSON-only Python ownership boundary for the typed recursive policy.

use pyo3::exceptions::PyRuntimeError;
use pyo3::prelude::*;
use rcswx_core::error::Error;
use rcswx_core::typed::{self, Tree};
use std::panic::{AssertUnwindSafe, catch_unwind};

#[pyfunction]
fn compare_typed_json(py: Python<'_>, first: &str, second: &str) -> PyResult<String> {
    let first = Tree::from_json(first).map_err(|error| crate::engine::python_error(error, None))?;
    let second =
        Tree::from_json(second).map_err(|error| crate::engine::python_error(error, None))?;
    let (result, callback_error) = py.detach(move || {
        let mut callback_error = None;
        let result = catch_unwind(AssertUnwindSafe(|| {
            let mut check = || match Python::attach(|py| py.check_signals()) {
                Ok(()) => Ok(()),
                Err(error) => {
                    callback_error = Some(error);
                    Err(Error::Callback)
                }
            };
            typed::compare(&first, &second, &mut check)
        }));
        (result, callback_error)
    });
    let comparison = match result {
        Ok(Ok(comparison)) => comparison,
        Ok(Err(error)) => return Err(crate::engine::python_error(error, callback_error)),
        Err(_) => {
            return Err(PyRuntimeError::new_err(
                "native typed recursive alignment invariant failed",
            ));
        }
    };
    serde_json::to_string(&comparison).map_err(|error| PyRuntimeError::new_err(error.to_string()))
}

pub fn register(module: &Bound<'_, PyModule>) -> PyResult<()> {
    module.add_function(wrap_pyfunction!(compare_typed_json, module)?)?;
    Ok(())
}
