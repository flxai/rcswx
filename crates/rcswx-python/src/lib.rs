use pyo3::prelude::*;

mod recursive;
mod selection;

#[pymodule]
fn _core(module: &Bound<'_, PyModule>) -> PyResult<()> {
    recursive::register(module)?;
    selection::register(module)?;
    Ok(())
}
