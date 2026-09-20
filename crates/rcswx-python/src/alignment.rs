use crate::metadata::{PyArchitecture, PyCancellationToken, PyLimits, guarded};
use pyo3::prelude::*;
use rcswx_core as core;
use std::collections::BTreeMap;
use std::sync::Arc;

pub fn stats(value: core::Stats) -> BTreeMap<&'static str, usize> {
    BTreeMap::from([
        ("states", value.states),
        ("predecessors", value.predecessors),
        ("charged_bytes", value.charged_bytes),
    ])
}
type StepRecord = (u8, Option<usize>, Option<usize>, u64, Option<usize>);

#[pyclass(name = "EditPath", frozen)]
pub struct PyEditPath {
    pub inner: Arc<core::Path>,
}
#[pymethods]
impl PyEditPath {
    #[getter]
    fn cost_ticks(&self) -> u64 {
        self.inner.cost_ticks()
    }
    #[getter]
    fn edit_count(&self) -> usize {
        self.inner.edit_count()
    }
    #[getter]
    fn steps(&self) -> Vec<StepRecord> {
        self.inner
            .steps()
            .iter()
            .map(|step| {
                (
                    step.kind,
                    step.source,
                    step.target,
                    step.cost_ticks,
                    step.edit_id,
                )
            })
            .collect()
    }
    #[getter]
    fn stats(&self) -> BTreeMap<&'static str, usize> {
        stats(self.inner.stats())
    }
    #[getter]
    fn owned_bytes(&self) -> usize {
        self.inner.owned_bytes()
    }
    fn selected_cost(&self, mask: u64) -> PyResult<u64> {
        guarded(|| self.inner.selected_cost(mask))
    }
    fn project(
        &self,
        py: Python<'_>,
        mask: u64,
        limits: &PyLimits,
        token: &PyCancellationToken,
    ) -> PyResult<Vec<(u8, usize)>> {
        let path = self.inner.clone();
        let limits = limits.inner.clone();
        let token = token.inner.clone();
        py.detach(move || guarded(|| path.project(mask, &limits, &token)))
    }
}

#[pyfunction]
fn distance(
    py: Python<'_>,
    source: &PyArchitecture,
    target: &PyArchitecture,
    limits: &PyLimits,
    token: &PyCancellationToken,
) -> PyResult<(u64, BTreeMap<&'static str, usize>)> {
    let source = source.inner.clone();
    let target = target.inner.clone();
    let limits = limits.inner.clone();
    let token = token.inner.clone();
    let result =
        py.detach(move || guarded(|| core::distance(&source, &target, &limits, &token)))?;
    Ok((result.cost_ticks, stats(result.stats)))
}

#[pyfunction]
fn align(
    py: Python<'_>,
    source: &PyArchitecture,
    target: &PyArchitecture,
    limits: &PyLimits,
    token: &PyCancellationToken,
) -> PyResult<PyEditPath> {
    let source = source.inner.clone();
    let target = target.inner.clone();
    let limits = limits.inner.clone();
    let token = token.inner.clone();
    let path = py.detach(move || guarded(|| core::align(source, target, &limits, &token)))?;
    Ok(PyEditPath {
        inner: Arc::new(path),
    })
}

#[pyclass(name = "Sampler")]
struct PySampler {
    inner: core::Sampler,
}
#[pymethods]
impl PySampler {
    #[new]
    #[pyo3(signature = (path, seed, limits, token, external_bytes=0))]
    fn new(
        py: Python<'_>,
        path: &PyEditPath,
        seed: u64,
        limits: &PyLimits,
        token: &PyCancellationToken,
        external_bytes: usize,
    ) -> PyResult<Self> {
        let path = path.inner.clone();
        let limits = limits.inner.clone();
        let token = token.inner.clone();
        let inner = py.detach(move || {
            guarded(|| core::Sampler::new(path, seed, &limits, &token, external_bytes))
        })?;
        Ok(Self { inner })
    }
    fn draw(&mut self) -> PyResult<u64> {
        guarded(|| self.inner.draw())
    }
    fn propose(
        &mut self,
        py: Python<'_>,
        limits: &PyLimits,
        token: &PyCancellationToken,
    ) -> PyResult<(u64, Vec<(u8, usize)>)> {
        let limits = limits.inner.clone();
        let token = token.inner.clone();
        py.detach(|| guarded(|| self.inner.propose(&limits, &token)))
    }
    #[getter]
    fn stats(&self) -> (usize, usize, usize, f64) {
        let stats = self.inner.stats;
        (
            stats.visited_masks,
            stats.valid_masks,
            stats.charged_bytes,
            stats.total_weight,
        )
    }
}

pub fn register(module: &Bound<'_, PyModule>) -> PyResult<()> {
    module.add_class::<PyEditPath>()?;
    module.add_class::<PySampler>()?;
    module.add_function(wrap_pyfunction!(distance, module)?)?;
    module.add_function(wrap_pyfunction!(align, module)?)?;
    Ok(())
}
