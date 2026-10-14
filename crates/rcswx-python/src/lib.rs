use pyo3::prelude::*;

pub(crate) mod engine;
mod execution;
mod recursive;
pub(crate) mod sampling;
mod selection;
mod typed;

#[pymodule]
fn _core(module: &Bound<'_, PyModule>) -> PyResult<()> {
    recursive::register(module)?;
    selection::register(module)?;
    sampling::register(module)?;
    engine::register(module)?;
    execution::register(module)?;
    typed::register(module)?;
    Ok(())
}
