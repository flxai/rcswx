use crate::execution::WorkerCount;
use pyo3::exceptions::{
    PyIndexError, PyMemoryError, PyRuntimeError, PyUnboundLocalError, PyValueError,
};
use pyo3::prelude::*;
use rcswx_core::recursive::{self as core, Failure};
use std::collections::BTreeMap;
use std::panic::{AssertUnwindSafe, catch_unwind};

type TokenRecord = (i64, String, Vec<String>, usize);
type StepRecord = (
    usize,
    String,
    Option<i64>,
    Option<i64>,
    usize,
    usize,
    f64,
    bool,
    bool,
);
type AlignmentRecord = (f64, Vec<Vec<StepRecord>>, BTreeMap<&'static str, usize>);

fn failure(error: Failure, callback: Option<PyErr>) -> PyErr {
    match error {
        Failure::Index => PyIndexError::new_err("list index out of range"),
        Failure::EmptyMinimum => PyValueError::new_err(
            "zero-size array to reduction operation fmin which has no identity",
        ),
        Failure::Unbound(name) => PyUnboundLocalError::new_err(format!(
            "cannot access local variable '{name}' where it is not associated with a value"
        )),
        Failure::Memory => PyMemoryError::new_err("Memory limit exceeded for crossover"),
        Failure::Callback => {
            callback.unwrap_or_else(|| PyRuntimeError::new_err("missing memory-check exception"))
        }
        Failure::Operational(error) => crate::engine::python_error(error, callback),
    }
}

#[pyfunction]
#[pyo3(signature = (tokens1, tokens2, collapse_corners=false, memory_check=None, *, workers=WorkerCount::SERIAL))]
fn recursive_align(
    py: Python<'_>,
    tokens1: Vec<TokenRecord>,
    tokens2: Vec<TokenRecord>,
    collapse_corners: bool,
    memory_check: Option<Py<PyAny>>,
    workers: WorkerCount,
) -> PyResult<AlignmentRecord> {
    let convert = |rows: Vec<TokenRecord>| {
        rows.into_iter()
            .map(|(id, name, children, parent_arity)| core::Token {
                id,
                name,
                children,
                parent_arity,
            })
            .collect::<Vec<_>>()
    };
    let first = convert(tokens1);
    let second = convert(tokens2);
    let (result, callback_error) = py.detach(move || {
        let mut callback_error = None;
        let result = catch_unwind(AssertUnwindSafe(|| {
            let mut execution =
                rcswx_core::execution::Execution::new(workers.0).map_err(Failure::Operational)?;
            let mut check = || {
                rcswx_core::execution::callback(|| {
                    match Python::attach(|py| {
                        py.check_signals()?;
                        match &memory_check {
                            Some(callback) => callback.call0(py)?.is_truthy(py),
                            None => Ok(true),
                        }
                    }) {
                        Ok(true) => Ok(()),
                        Ok(false) => Err(Failure::Memory),
                        Err(error) => {
                            callback_error = Some(error);
                            Err(Failure::Callback)
                        }
                    }
                })
            };
            core::align_with_execution(
                &first,
                &second,
                collapse_corners,
                &mut check,
                &mut execution,
            )
        }));
        (result, callback_error)
    });
    let alignment = match result {
        Ok(Ok(value)) => value,
        Ok(Err(error)) => return Err(failure(error, callback_error)),
        Err(_) => {
            return Err(PyRuntimeError::new_err(
                "native recursive alignment invariant failed",
            ));
        }
    };
    let paths = alignment
        .paths
        .into_iter()
        .map(|path| {
            path.into_iter()
                .map(|step| {
                    (
                        step.id,
                        step.kind.name().to_owned(),
                        step.node1_id,
                        step.node2_id,
                        step.i,
                        step.j,
                        step.value,
                        step.i_swapped,
                        step.j_swapped,
                    )
                })
                .collect()
        })
        .collect();
    Ok((
        alignment.distance,
        paths,
        BTreeMap::from([
            ("cells_created", alignment.stats.cells_created),
            ("histories_created", alignment.stats.histories_created),
            ("memory_checkpoints", alignment.stats.checkpoints),
        ]),
    ))
}

pub fn register(module: &Bound<'_, PyModule>) -> PyResult<()> {
    module.add_function(wrap_pyfunction!(recursive_align, module)?)?;
    Ok(())
}
