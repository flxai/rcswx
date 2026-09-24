//! PyO3 ownership boundary for prepared snapshots and retained native edit plans.

use crate::execution::WorkerCount;
use crate::sampling::NativeRng;
use pyo3::exceptions::{
    PyException, PyIndexError, PyMemoryError, PyRuntimeError, PyUnboundLocalError, PyValueError,
};
use pyo3::prelude::*;
use pyo3::types::PyModule;
use rcswx_core::apply;
use rcswx_core::architecture::{Architecture, Budget, Limits};
use rcswx_core::edit_plan::{self, EditPlan};
use rcswx_core::error::Error;
use rcswx_core::execution::{Execution, Report};
use rcswx_core::sampling;
use rcswx_core::selection;
use rcswx_core::tokens::{self, PreparedPair};
use std::collections::BTreeMap;
use std::time::Instant;

type PreparedTokenRecord = (String, String, Vec<String>, usize, Option<usize>);

pub(crate) fn python_error(error: Error, callback: Option<PyErr>) -> PyErr {
    match error {
        Error::InvalidInput(message) | Error::Reference(message) | Error::Numerical(message) => {
            PyValueError::new_err(message)
        }
        Error::Index => PyIndexError::new_err("list index out of range"),
        Error::MissingMutationTarget => PyException::new_err("Nodes to mutate not found"),
        Error::EmptyMinimum => PyValueError::new_err(
            "zero-size array to reduction operation fmin which has no identity",
        ),
        Error::Unbound(name) => PyUnboundLocalError::new_err(format!(
            "cannot access local variable '{name}' where it is not associated with a value"
        )),
        Error::Memory => PyMemoryError::new_err("Memory limit exceeded for crossover"),
        Error::Limit(kind) => PyMemoryError::new_err(format!("native {kind} limit exceeded")),
        Error::Cancelled => PyRuntimeError::new_err("native work was cancelled"),
        Error::Execution(message) => PyRuntimeError::new_err(message),
        Error::Callback => {
            callback.unwrap_or_else(|| PyRuntimeError::new_err("host checkpoint failed"))
        }
        Error::PlanMismatch => PyValueError::new_err("selection belongs to another plan"),
    }
}

fn json_error(error: serde_json::Error) -> PyErr {
    PyValueError::new_err(error.to_string())
}

fn limits_from_python(py: Python<'_>, limits: Option<&Bound<'_, PyAny>>) -> PyResult<Limits> {
    let Some(limits) = limits else {
        return Ok(Limits::default());
    };
    let json = PyModule::import(py, "json")?;
    let encoded = json.call_method1("dumps", (limits,))?.extract::<String>()?;
    serde_json::from_str(&encoded).map_err(json_error)
}

fn detached_with_budget<T, F>(
    py: Python<'_>,
    limits: Limits,
    memory_check: Option<Py<PyAny>>,
    operation: F,
) -> (std::result::Result<T, Error>, Option<PyErr>)
where
    T: Send,
    F: FnOnce(&mut Budget<'_>) -> std::result::Result<T, Error> + Send,
{
    py.detach(move || {
        let mut callback_error = None;
        let mut poll = || {
            rcswx_core::execution::callback(|| {
                match Python::attach(|py| {
                    py.check_signals()?;
                    match &memory_check {
                        Some(callback) => callback.call0(py)?.is_truthy(py),
                        None => Ok(true),
                    }
                }) {
                    Ok(true) => Ok(()),
                    Ok(false) => Err(Error::Memory),
                    Err(error) => {
                        callback_error = Some(error);
                        Err(Error::Callback)
                    }
                }
            })
        };
        let result = {
            // Core checkpoints account every unit of work/allocation. Host RSS
            // polling is much more expensive: check initially, periodically,
            // and before publishing success, not for every enumeration visit.
            let poll = std::cell::RefCell::new(&mut poll);
            let mut until_poll = 0;
            let mut check = || {
                if memory_check.is_none() {
                    return Ok(());
                }
                if until_poll == 0 {
                    until_poll = 1023;
                    (*poll.borrow_mut())()
                } else {
                    until_poll -= 1;
                    Ok(())
                }
            };
            let mut wait_check = || (*poll.borrow_mut())();
            let mut budget = Budget {
                limits,
                work: 0,
                output: 0,
                allocation_bytes: 0,
                check: &mut check,
                wait_check: Some(&mut wait_check),
            };
            operation(&mut budget)
        };
        let result = result.and_then(|value| poll().map(|()| value));
        (result, callback_error)
    })
}
/// Opt-in profiling spans are binding-owned, independent of executor polling.
#[derive(Clone, Default)]
struct Profile {
    enabled: bool,
    values: BTreeMap<String, f64>,
}

impl Profile {
    fn requested(enabled: bool) -> Self {
        Self {
            enabled,
            values: BTreeMap::new(),
        }
    }

    fn record(&mut self, name: &str, elapsed: std::time::Duration) {
        if self.enabled {
            *self.values.entry(name.to_owned()).or_default() += elapsed.as_secs_f64();
        }
    }

    fn json(&self) -> PyResult<String> {
        serde_json::to_string(&self.values).map_err(json_error)
    }
}

fn token_views(
    tokens: &[rcswx_core::tokens::PreparedToken],
    identities: &[String],
) -> Vec<PreparedTokenRecord> {
    tokens
        .iter()
        .map(|prepared| {
            let id = usize::try_from(prepared.token.id)
                .ok()
                .and_then(|index| identities.get(index))
                .cloned()
                .unwrap_or_else(|| prepared.token.id.to_string());
            (
                id,
                prepared.token.name.clone(),
                prepared.token.children.clone(),
                prepared.token.parent_arity,
                prepared.occurrence,
            )
        })
        .collect()
}
#[pyclass(module = "rcswx._core")]
#[derive(Clone)]
pub struct NativePrepared {
    prepared: PreparedPair,
    limits: Limits,
    profile: Profile,
}

#[pymethods]
impl NativePrepared {
    fn __deepcopy__(&self, _memo: &Bound<'_, PyAny>) -> Self {
        self.clone()
    }

    #[getter]
    fn parent2_ids(&self) -> Vec<String> {
        self.prepared.parent2_ids.clone()
    }

    fn first_tokens(&self) -> Vec<PreparedTokenRecord> {
        token_views(&self.prepared.first_tokens, &self.prepared.identities)
    }

    fn second_tokens(&self) -> Vec<PreparedTokenRecord> {
        token_views(&self.prepared.second_tokens, &self.prepared.identities)
    }

    fn profile_json(&self) -> PyResult<String> {
        self.profile.json()
    }
    #[pyo3(signature = (plan_id, collapse_corners=false, memory_check=None, *, workers=WorkerCount::SERIAL))]
    fn analyze(
        &self,
        py: Python<'_>,
        plan_id: String,
        collapse_corners: bool,
        memory_check: Option<Py<PyAny>>,
        workers: WorkerCount,
    ) -> PyResult<NativePlan> {
        let prepared = self.prepared.clone();
        let limits = self.limits.clone();
        let inherited_profile = self.profile.clone();
        let (outcome, callback_error) =
            detached_with_budget(py, limits.clone(), memory_check, move |budget| {
                let mut profile = inherited_profile;
                let mut execution = Execution::new(workers.0)?;
                let result = if profile.enabled {
                    let mut active = BTreeMap::<&'static str, Instant>::new();
                    let mut stage = |name: &'static str, started: bool| {
                        if started {
                            active.insert(name, Instant::now());
                        } else if let Some(start) = active.remove(name) {
                            profile.record(name, start.elapsed());
                        }
                    };
                    edit_plan::analyze_with_execution(
                        prepared,
                        collapse_corners,
                        plan_id,
                        budget,
                        &mut execution,
                        Some(&mut stage),
                    )
                } else {
                    edit_plan::analyze_with_execution(
                        prepared,
                        collapse_corners,
                        plan_id,
                        budget,
                        &mut execution,
                        None,
                    )
                };
                result.map(|plan| (plan, profile, execution.into_report()))
            });
        let (plan, profile, execution) =
            outcome.map_err(|error| python_error(error, callback_error))?;
        Ok(NativePlan {
            plan,
            limits,
            profile,
            execution,
        })
    }
}

#[pyclass(module = "rcswx._core")]
#[derive(Clone)]
pub struct NativePlan {
    plan: EditPlan,
    limits: Limits,
    profile: Profile,
    execution: Report,
}

#[pymethods]
impl NativePlan {
    fn __deepcopy__(&self, _memo: &Bound<'_, PyAny>) -> Self {
        self.clone()
    }

    #[getter]
    fn id(&self) -> String {
        self.plan.id.clone()
    }

    #[getter]
    fn distance(&self) -> f64 {
        self.plan.distance
    }

    #[getter]
    fn path_index(&self) -> usize {
        self.plan.path_index
    }

    #[getter]
    fn operations(&self) -> Vec<usize> {
        self.plan.operations.clone()
    }

    #[getter]
    fn operations_unordered(&self) -> Vec<usize> {
        self.plan.operations_unordered.clone()
    }

    #[getter]
    fn nontrivial(&self) -> Vec<usize> {
        self.plan.nontrivial.clone()
    }

    fn stats_json(&self) -> PyResult<String> {
        serde_json::to_string(&self.plan.stats).map_err(json_error)
    }

    fn execution_json(&self) -> PyResult<String> {
        serde_json::to_string(&self.execution).map_err(json_error)
    }

    fn selected_path_json(&self) -> PyResult<String> {
        let path = self
            .plan
            .paths
            .get(self.plan.path_index)
            .ok_or_else(|| PyIndexError::new_err("list index out of range"))?;
        serde_json::to_string(path).map_err(json_error)
    }

    fn paths_json(&self) -> PyResult<String> {
        serde_json::to_string(&self.plan.paths).map_err(json_error)
    }

    fn profile_json(&self) -> PyResult<String> {
        self.profile.json()
    }

    fn validate_selection(&self, selected: Vec<usize>) -> PyResult<()> {
        self.plan
            .validate_selection(&selected)
            .map_err(|error| python_error(error, None))
    }
    #[pyo3(signature = (memory_check=None))]
    fn combinations(
        &mut self,
        py: Python<'_>,
        memory_check: Option<Py<PyAny>>,
    ) -> PyResult<(Vec<String>, Vec<f64>)> {
        let operations = self
            .plan
            .selection_operations()
            .map_err(|error| python_error(error, None))?;
        let started = self.profile.enabled.then(Instant::now);
        let limits = self.limits.clone();
        let (outcome, callback_error) =
            detached_with_budget(py, limits, memory_check, move |budget| {
                selection::combinations_with_budget(&operations, budget)
            });
        let result = outcome.map_err(|error| python_error(error, callback_error))?;
        if let Some(start) = started {
            self.profile.record("enumeration", start.elapsed());
        }
        Ok(result)
    }

    #[pyo3(signature = (skewness, memory_check=None))]
    fn probabilities(
        &mut self,
        py: Python<'_>,
        skewness: f64,
        memory_check: Option<Py<PyAny>>,
    ) -> PyResult<(Vec<String>, Vec<f64>, Vec<f64>)> {
        let operations = self
            .plan
            .selection_operations()
            .map_err(|error| python_error(error, None))?;
        let profiled = self.profile.enabled;
        let limits = self.limits.clone();
        let (outcome, callback_error) =
            detached_with_budget(py, limits, memory_check, move |budget| {
                let enumeration_start = profiled.then(Instant::now);
                let (masks, costs) = selection::combinations_with_budget(&operations, budget)?;
                let enumeration = enumeration_start.map(|start| start.elapsed());
                let weights_start = profiled.then(Instant::now);
                let probabilities = sampling::probabilities_with_budget(&costs, skewness, budget)?;
                Ok((
                    (masks, costs, probabilities),
                    enumeration,
                    weights_start.map(|start| start.elapsed()),
                ))
            });
        let ((masks, costs, probabilities), enumeration, weights) =
            outcome.map_err(|error| python_error(error, callback_error))?;
        if let Some(elapsed) = enumeration {
            self.profile.record("enumeration", elapsed);
        }
        if let Some(elapsed) = weights {
            self.profile.record("weights", elapsed);
        }
        Ok((masks, costs, probabilities))
    }

    #[pyo3(signature = (skewness, rng, memory_check=None))]
    fn sample(
        &mut self,
        py: Python<'_>,
        skewness: f64,
        rng: &mut NativeRng,
        memory_check: Option<Py<PyAny>>,
    ) -> PyResult<(String, f64)> {
        let operations = self
            .plan
            .selection_operations()
            .map_err(|error| python_error(error, None))?;
        let profiled = self.profile.enabled;
        let limits = self.limits.clone();
        let (outcome, callback_error) =
            detached_with_budget(py, limits, memory_check, move |budget| {
                let mut spans = Profile::requested(profiled);
                let result = if profiled {
                    let mut active = BTreeMap::<&'static str, Instant>::new();
                    let mut stage = |name: &'static str, started: bool| {
                        if started {
                            active.insert(name, Instant::now());
                        } else if let Some(start) = active.remove(name) {
                            spans.record(name, start.elapsed());
                        }
                    };
                    sampling::sample_with_budget_profiled(
                        &operations,
                        skewness,
                        &mut rng.inner,
                        budget,
                        &mut stage,
                    )
                } else {
                    sampling::sample_with_budget(&operations, skewness, &mut rng.inner, budget)
                };
                result.map(|value| (value, spans.values))
            });
        let ((mask, cost), spans) = outcome.map_err(|error| python_error(error, callback_error))?;
        for (name, elapsed) in spans {
            *self.profile.values.entry(name).or_default() += elapsed;
        }
        Ok((mask, cost))
    }

    #[pyo3(signature = (selected, validate=true, memory_check=None))]
    fn apply(
        &mut self,
        py: Python<'_>,
        selected: Vec<usize>,
        validate: bool,
        memory_check: Option<Py<PyAny>>,
    ) -> PyResult<(String, String, usize)> {
        let started = self.profile.enabled.then(Instant::now);
        let limits = self.limits.clone();
        let (outcome, callback_error) = detached_with_budget(py, limits, memory_check, |budget| {
            apply::apply(&mut self.plan, &selected, validate, budget)
        });
        let result = outcome.map_err(|error| python_error(error, callback_error))?;
        if let Some(start) = started {
            self.profile.record("apply", start.elapsed());
        }
        let architecture = result
            .architecture
            .to_json()
            .map_err(|error| python_error(error, None))?;
        let recipe = serde_json::to_string(&result.recipe).map_err(json_error)?;
        Ok((architecture, recipe, result.root))
    }

    #[pyo3(signature = (selected, validate=true, memory_check=None))]
    fn apply_legacy(
        &mut self,
        py: Python<'_>,
        selected: Vec<usize>,
        validate: bool,
        memory_check: Option<Py<PyAny>>,
    ) -> PyResult<(String, usize)> {
        let started = self.profile.enabled.then(Instant::now);
        let limits = self.limits.clone();
        let (outcome, callback_error) = detached_with_budget(py, limits, memory_check, |budget| {
            apply::apply_materialization(&mut self.plan, &selected, validate, budget)
        });
        let (recipe, root) = outcome.map_err(|error| python_error(error, callback_error))?;
        if let Some(start) = started {
            self.profile.record("apply", start.elapsed());
        }
        Ok((serde_json::to_string(&recipe).map_err(json_error)?, root))
    }
}

#[pyfunction]
fn validate_architecture(text: &str) -> PyResult<String> {
    Architecture::from_json(text)
        .and_then(|architecture| architecture.to_json())
        .map_err(|error| python_error(error, None))
}

#[pyfunction]
fn architecture_tokens(text: &str) -> PyResult<Vec<PreparedTokenRecord>> {
    let architecture = Architecture::from_json(text).map_err(|error| python_error(error, None))?;
    let (tokens, identities) =
        tokens::tokenize_architecture(&architecture).map_err(|error| python_error(error, None))?;
    Ok(token_views(&tokens, &identities))
}

#[pyfunction]
#[pyo3(signature = (first, second, profile=false, limits=None, *, legacy_origins=None))]
fn prepare_architectures(
    py: Python<'_>,
    first: &str,
    second: &str,
    profile: bool,
    limits: Option<&Bound<'_, PyAny>>,
    legacy_origins: Option<(Vec<usize>, Vec<usize>)>,
) -> PyResult<NativePrepared> {
    let limits = limits_from_python(py, limits)?;
    let first = Architecture::from_json(first).map_err(|error| python_error(error, None))?;
    let second = Architecture::from_json(second).map_err(|error| python_error(error, None))?;
    let mut timings = Profile::requested(profile);
    let started = profile.then(Instant::now);
    let prepared = match legacy_origins {
        Some((first_origins, second_origins)) => {
            tokens::prepare_with_origins(first, second, &first_origins, &second_origins)
        }
        None => tokens::prepare(first, second),
    }
    .map_err(|error| python_error(error, None))?;
    if let Some(start) = started {
        timings.record("prepare", start.elapsed());
    }
    Ok(NativePrepared {
        prepared,
        limits,
        profile: timings,
    })
}

pub fn register(module: &Bound<'_, PyModule>) -> PyResult<()> {
    module.add_class::<NativePrepared>()?;
    module.add_class::<NativePlan>()?;
    module.add_function(wrap_pyfunction!(validate_architecture, module)?)?;
    module.add_function(wrap_pyfunction!(architecture_tokens, module)?)?;
    module.add_function(wrap_pyfunction!(prepare_architectures, module)?)?;
    Ok(())
}
