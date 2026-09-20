use pyo3::prelude::*;

#[pyfunction]
fn derive_seed(seed: u64, domain: &[u8]) -> Vec<u8> {
    rcswx_core::derive_seed(seed, domain).to_vec()
}

#[pymodule]
fn _core(module: &Bound<'_, PyModule>) -> PyResult<()> {
    module.add("PROFILE_VERSION", rcswx_core::PROFILE_VERSION)?;
    module.add("RNG_VERSION", rcswx_core::RNG_VERSION)?;
    module.add_function(wrap_pyfunction!(derive_seed, module)?)?;
    Ok(())
}
