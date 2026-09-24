use pyo3::exceptions::{PyIndexError, PyMemoryError, PyRuntimeError, PyTypeError, PyValueError};
use pyo3::prelude::*;
use pyo3::types::{PyBytes, PyDict, PyInt, PyList, PyTuple, PyType};
use rcswx_core::error::Error;
use rcswx_core::sampling::{self as core, NativeRngState};
use rcswx_core::selection::Operation;

#[pyclass(name = "NativeRng", module = "rcswx._core")]
pub struct NativeRng {
    /// Engine bindings borrow this directly so plan sampling never round-trips
    /// masks, costs, or generator state through Python containers.
    pub inner: core::NativeRng,
}

fn native_error(error: Error) -> PyErr {
    let message = error.to_string();
    match error {
        Error::Index => PyIndexError::new_err(message),
        Error::Memory => PyMemoryError::new_err(message),
        Error::InvalidInput(_) | Error::Numerical(_) | Error::EmptyMinimum => {
            PyValueError::new_err(message)
        }
        Error::Unbound(_)
        | Error::Reference(_)
        | Error::MissingMutationTarget
        | Error::Limit(_)
        | Error::Cancelled
        | Error::Execution(_)
        | Error::Callback
        | Error::PlanMismatch => PyRuntimeError::new_err(message),
    }
}

fn seed_bytes(seed: &Bound<'_, PyAny>) -> PyResult<[u8; 32]> {
    if let Ok(bytes) = seed.cast::<PyBytes>() {
        let bytes = bytes.as_bytes();
        return bytes.try_into().map_err(|_| {
            PyValueError::new_err("native RNG byte seeds must contain exactly 32 bytes")
        });
    }
    if seed.is_instance_of::<PyInt>() {
        // Python performs the exact arbitrary-width range check: negative values
        // and values >= 2**256 raise OverflowError rather than silently wrapping.
        let bytes = seed.call_method1("to_bytes", (32, "little"))?;
        let bytes = bytes.cast::<PyBytes>()?.as_bytes();
        return bytes
            .try_into()
            .map_err(|_| PyRuntimeError::new_err("Python integer seed encoding was not 32 bytes"));
    }
    Err(PyTypeError::new_err(
        "native RNG seed must be an integer in [0, 2**256) or exactly 32 bytes",
    ))
}

fn state_value(state: &Bound<'_, PyAny>) -> PyResult<NativeRngState> {
    let state = state.cast::<PyDict>()?;
    let required = |name| {
        state
            .get_item(name)?
            .ok_or_else(|| PyValueError::new_err(format!("native RNG state is missing {name:?}")))
    };
    Ok(NativeRngState {
        version: required("version")?.extract()?,
        algorithm: required("algorithm")?.extract()?,
        seed: required("seed")?.extract()?,
        stream: required("stream")?.extract()?,
        word_position: required("word_position")?.extract()?,
    })
}

#[pymethods]
impl NativeRng {
    #[new]
    fn new(seed: &Bound<'_, PyAny>) -> PyResult<Self> {
        Ok(Self {
            inner: core::NativeRng::from_seed(seed_bytes(seed)?),
        })
    }

    #[classmethod]
    fn from_state(_class: &Bound<'_, PyType>, state: &Bound<'_, PyAny>) -> PyResult<Self> {
        Ok(Self {
            inner: core::NativeRng::from_state(state_value(state)?).map_err(native_error)?,
        })
    }

    fn state<'py>(&self, py: Python<'py>) -> PyResult<Bound<'py, PyDict>> {
        let state = self.inner.state();
        let output = PyDict::new(py);
        output.set_item("version", state.version)?;
        output.set_item("algorithm", state.algorithm)?;
        output.set_item("seed", state.seed)?;
        output.set_item("stream", state.stream)?;
        output.set_item("word_position", state.word_position)?;
        Ok(output)
    }

    fn next_u64(&mut self) -> u64 {
        self.inner.next_u64()
    }

    fn uniform(&mut self) -> f64 {
        self.inner.uniform()
    }
}

fn groups(value: &Bound<'_, PyAny>) -> PyResult<Vec<Vec<i64>>> {
    value
        .try_iter()?
        .map(|dependency| {
            let dependency = dependency?;
            if let Ok(group) = dependency.cast::<PyList>() {
                group.extract::<Vec<i64>>()
            } else {
                Ok(vec![dependency.extract::<i64>()?])
            }
        })
        .collect()
}

fn operations(records: &Bound<'_, PyAny>) -> PyResult<Vec<Operation>> {
    records
        .try_iter()?
        .map(|record| {
            let record = record?;
            let record = record.cast::<PyTuple>()?;
            if record.len() != 4 {
                return Err(PyValueError::new_err(
                    "native selection records must be (id, value, enablers, disablers)",
                ));
            }
            Ok(Operation {
                id: record.get_item(0)?.extract()?,
                value: record.get_item(1)?.extract()?,
                enablers: groups(&record.get_item(2)?)?,
                disablers: groups(&record.get_item(3)?)?,
            })
        })
        .collect()
}

#[pyfunction]
fn native_probabilities(py: Python<'_>, values: Vec<f64>, skewness: f64) -> PyResult<Vec<f64>> {
    py.detach(move || core::probabilities(&values, skewness))
        .map_err(native_error)
}

#[pyfunction]
fn native_select(
    py: Python<'_>,
    records: &Bound<'_, PyAny>,
    skewness: f64,
    mut rng: PyRefMut<'_, NativeRng>,
) -> PyResult<(String, f64)> {
    let operations = operations(records)?;
    let inner = &mut rng.inner;
    py.detach(move || core::sample(&operations, skewness, inner))
        .map_err(native_error)
}

#[pyfunction]
fn native_draw(
    py: Python<'_>,
    probabilities: Vec<f64>,
    mut rng: PyRefMut<'_, NativeRng>,
) -> PyResult<usize> {
    let inner = &mut rng.inner;
    py.detach(move || core::draw(&probabilities, inner))
        .map_err(native_error)
}

pub fn register(module: &Bound<'_, PyModule>) -> PyResult<()> {
    module.add_class::<NativeRng>()?;
    module.add_function(wrap_pyfunction!(native_probabilities, module)?)?;
    module.add_function(wrap_pyfunction!(native_select, module)?)?;
    module.add_function(wrap_pyfunction!(native_draw, module)?)?;
    Ok(())
}
