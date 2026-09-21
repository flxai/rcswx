use pyo3::exceptions::{PyIndexError, PyMemoryError};
use pyo3::prelude::*;
use rcswx_core::selection::{self as core, Failure};

type Record = (i64, f64, Vec<Vec<i64>>, Vec<Vec<i64>>);

#[pyfunction]
fn valid_combinations(py: Python<'_>, records: Vec<Record>) -> PyResult<(Vec<String>, Vec<f64>)> {
    let operations: Vec<_> = records
        .into_iter()
        .map(|(id, value, enablers, disablers)| core::Operation {
            id,
            value,
            enablers,
            disablers,
        })
        .collect();
    py.detach(move || core::combinations(&operations))
        .map_err(|error| match error {
            Failure::Index => PyIndexError::new_err("list index out of range"),
            Failure::Memory => {
                PyMemoryError::new_err("unable to allocate reference selection distribution")
            }
        })
}

pub fn register(module: &Bound<'_, PyModule>) -> PyResult<()> {
    module.add_function(wrap_pyfunction!(valid_combinations, module)?)?;
    Ok(())
}
