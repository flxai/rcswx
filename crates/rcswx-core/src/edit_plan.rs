//! Reference-compatible restriction processing and immutable edit plans.
//!
//! This module ports the Python `Alignment.calculate_restrictions` list
//! mutations verbatim at the value level. Dependencies point at positions in a
//! retained path so grouped dependencies, duplicate members, and source order
//! survive the native/Python boundary.

use crate::architecture::Budget;
use crate::error::{Error, Result};
use crate::recursive::{self, Failure};
use crate::selection;
use crate::tokens::PreparedPair;
use serde::Serialize;
use std::collections::{BTreeMap, HashSet};

#[derive(Clone, Debug, PartialEq, Eq, Serialize)]
#[serde(untagged)]
pub enum Dependency {
    One(usize),
    Group(Vec<usize>),
}

#[derive(Clone, Debug, Serialize)]
pub struct Edit {
    pub id: usize,
    pub op_type: String,
    pub node1_id: Option<String>,
    pub node2_id: Option<String>,
    pub i: usize,
    pub j: usize,
    /// Original token occurrences; i/j may be remapped for branch-order routing.
    #[serde(skip)]
    pub(crate) source_i: usize,
    #[serde(skip)]
    pub(crate) source_j: usize,
    #[serde(skip)]
    pub(crate) source_i_swapped: bool,
    #[serde(skip)]
    pub(crate) source_j_swapped: bool,
    pub ii: Option<usize>,
    pub jj: Option<usize>,
    pub value: f64,
    pub i_swapped: bool,
    pub j_swapped: bool,
    pub enabler_ops: Vec<Dependency>,
    pub disabler_ops: Vec<Dependency>,
}

#[derive(Clone, Debug)]
pub struct EditPlan {
    pub id: String,
    pub prepared: PreparedPair,
    pub distance: f64,
    pub paths: Vec<Vec<Edit>>,
    pub path_index: usize,
    pub operations: Vec<usize>,
    pub operations_unordered: Vec<usize>,
    pub nontrivial: Vec<usize>,
    pub stats: BTreeMap<String, usize>,
}

fn external_id(prepared: &PreparedPair, dense: i64) -> Result<String> {
    let index = usize::try_from(dense).map_err(|_| Error::Index)?;
    prepared.identities.get(index).cloned().ok_or(Error::Index)
}

fn edit_from_step(prepared: &PreparedPair, step: recursive::Step) -> Result<Edit> {
    Ok(Edit {
        id: step.id,
        op_type: step.kind.name().into(),
        node1_id: step
            .node1_id
            .map(|identity| external_id(prepared, identity))
            .transpose()?,
        node2_id: step
            .node2_id
            .map(|identity| external_id(prepared, identity))
            .transpose()?,
        i: step.i,
        j: step.j,
        source_i: step.source_i,
        source_j: step.source_j,
        source_i_swapped: false,
        source_j_swapped: false,
        ii: None,
        jj: None,
        value: step.value,
        i_swapped: step.i_swapped,
        j_swapped: step.j_swapped,
        enabler_ops: Vec::new(),
        disabler_ops: Vec::new(),
    })
}

fn token_id(prepared: &PreparedPair, first: bool, index: usize) -> Result<String> {
    let token = if first {
        prepared.first_tokens.get(index)
    } else {
        prepared.second_tokens.get(index)
    }
    .ok_or(Error::Index)?;
    external_id(prepared, token.token.id)
}

fn token_name(prepared: &PreparedPair, first: bool, index: usize) -> Result<&str> {
    let token = if first {
        prepared.first_tokens.get(index)
    } else {
        prepared.second_tokens.get(index)
    }
    .ok_or(Error::Index)?;
    Ok(&token.token.name)
}

fn token_arity(prepared: &PreparedPair, first: bool, index: usize) -> Result<usize> {
    let token = if first {
        prepared.first_tokens.get(index)
    } else {
        prepared.second_tokens.get(index)
    }
    .ok_or(Error::Index)?;
    Ok(token.token.children.len())
}

fn contains(operation: &Edit, text: &str) -> bool {
    operation.op_type.contains(text)
}

fn kind_prefix(operation: &Edit) -> &str {
    // Kernel operation names are ASCII and always at least three bytes long.
    &operation.op_type[..3]
}

fn source_section(
    operations: &[Edit],
    range: std::ops::Range<usize>,
) -> (Vec<usize>, Vec<usize>, Vec<usize>) {
    let mut adds = Vec::new();
    let mut mutations = Vec::new();
    let mut removals = Vec::new();
    for index in range {
        let Some(operation) = operations.get(index) else {
            break;
        };
        if contains(operation, "wrap") {
            continue;
        }
        if contains(operation, "add") {
            adds.push(index);
        }
        if contains(operation, "mut") {
            mutations.push(index);
        }
        if contains(operation, "rem") {
            removals.push(index);
        }
    }
    (adds, mutations, removals)
}

fn append_ones(target: &mut Vec<Dependency>, values: &[usize]) {
    target.extend(values.iter().copied().map(Dependency::One));
}

fn append_group(target: &mut Vec<Dependency>, values: Vec<usize>) {
    target.push(Dependency::Group(values));
}

fn different_ids(operations: &[Edit], candidates: &[usize], current: usize) -> Result<Vec<usize>> {
    let current_id = operations.get(current).ok_or(Error::Index)?.id;
    let mut result = Vec::new();
    result
        .try_reserve(candidates.len())
        .map_err(|_| Error::Memory)?;
    for &candidate in candidates {
        if operations.get(candidate).ok_or(Error::Index)?.id != current_id {
            result.push(candidate);
        }
    }
    Ok(result)
}

/// Python loop locals retain both their index and operation coordinates across
/// empty slices, later loops, and later histories in the same restriction pass.
fn find_loop(
    operations: &[Edit],
    start: usize,
    initialized: Option<usize>,
    index: &mut Option<usize>,
    operation: &mut Option<(usize, usize)>,
    name: &'static str,
    mut predicate: impl FnMut(&Edit) -> bool,
) -> Result<usize> {
    if let Some(initialized) = initialized {
        *index = Some(initialized);
    }
    for (relative, other) in operations[start.min(operations.len())..].iter().enumerate() {
        *index = Some(relative);
        *operation = Some((other.source_i, other.source_j));
        if predicate(other) {
            break;
        }
    }
    advance_loop_index(index, start, name)
}

fn process_swaps(prepared: &PreparedPair, operations: &mut [Edit], first: bool) -> Result<()> {
    let tokens = if first {
        &prepared.first_tokens
    } else {
        &prepared.second_tokens
    };
    let coordinate = |edit: &Edit| if first { edit.source_i } else { edit.source_j };
    for index in 0..operations.len() {
        let operation = operations.get(index).ok_or(Error::Index)?.clone();
        if !contains(&operation, "wrap_end")
            || !(contains(&operation, if first { "add" } else { "rem" })
                || contains(&operation, "mut"))
        {
            continue;
        }
        // Logical IDs may repeat; a boundary triple belongs to one physical
        // occurrence, not every wrapper carrying the same ID.
        let owner = tokens
            .get(coordinate(&operation))
            .and_then(|token| token.occurrence)
            .ok_or(Error::Index)?;
        let prefix = kind_prefix(&operation);
        let mut matching = Vec::new();
        for (candidate, other) in operations.iter().enumerate() {
            if kind_prefix(other) == prefix
                && tokens
                    .get(coordinate(other))
                    .is_some_and(|token| token.occurrence == Some(owner))
            {
                matching.push(candidate);
            }
        }
        let opening = *matching.first().ok_or(Error::Index)?;
        if token_arity(prepared, first, coordinate(&operations[opening]))? != 4 {
            continue;
        }
        // Routing flags describe recursive alternatives. Materialization needs
        // the order encoded by the retained path's physical source occurrences.
        let separator = *matching.get(1).ok_or(Error::Index)?;
        let source_open = coordinate(&operations[opening]);
        let source_sep = coordinate(&operations[separator]);
        let source_end = coordinate(&operation);
        let source_swapped = operations[opening + 1..separator]
            .iter()
            .filter(|edit| {
                contains(edit, if first { "add" } else { "rem" }) || contains(edit, "mut")
            })
            .map(coordinate)
            .find(|&position| {
                position > source_open && position < source_end && position != source_sep
            })
            .is_some_and(|position| position > source_sep);
        for &candidate in &matching {
            if first {
                operations.get_mut(candidate).ok_or(Error::Index)?.i_swapped = operation.i_swapped;
                operations
                    .get_mut(candidate)
                    .ok_or(Error::Index)?
                    .source_i_swapped = source_swapped;
            } else {
                operations.get_mut(candidate).ok_or(Error::Index)?.j_swapped = operation.j_swapped;
                operations
                    .get_mut(candidate)
                    .ok_or(Error::Index)?
                    .source_j_swapped = source_swapped;
            }
        }
        let positions: Vec<usize> = matching
            .iter()
            .map(|&candidate| {
                let operation = operations.get(candidate).ok_or(Error::Index)?;
                Ok(if first { operation.i } else { operation.j })
            })
            .collect::<Result<_>>()?;
        let (first_position, middle_position, last_position) = (
            *positions.first().ok_or(Error::Index)?,
            *positions.get(1).ok_or(Error::Index)?,
            *positions.get(2).ok_or(Error::Index)?,
        );
        if if first {
            operation.i_swapped
        } else {
            operation.j_swapped
        } {
            for other in operations.iter_mut() {
                let position = if first { &mut other.i } else { &mut other.j };
                if *position >= first_position && *position < middle_position {
                    *position += last_position - middle_position;
                } else if *position >= middle_position && *position < last_position {
                    *position -= middle_position - first_position;
                }
            }
        }
    }
    Ok(())
}

fn add_restrictions(
    prepared: &PreparedPair,
    operations: &mut [Edit],
    index: usize,
    locals: &mut ReorderLocals,
) -> Result<()> {
    let operation = operations.get(index).ok_or(Error::Index)?.clone();
    match token_arity(prepared, true, operation.source_i)? {
        4 => {
            let owner = token_id(prepared, true, operation.source_i)?;
            let separator = find_loop(
                operations,
                index + 1,
                None,
                &mut locals.sep_index,
                &mut locals.sep_operation,
                "sep_idx",
                |other| {
                    contains(other, "add_wrap")
                        && token_id(prepared, true, other.source_i).is_ok_and(|id| id == owner)
                },
            )?;
            let end = find_loop(
                operations,
                separator + 1,
                Some(0),
                &mut locals.end_index,
                &mut locals.end_operation,
                "end_idx",
                |other| {
                    other.op_type == "add_wrap_end"
                        && token_id(prepared, true, other.source_i).is_ok_and(|id| id == owner)
                },
            )?;
            let (adds0, mutations0, removals0) = source_section(operations, index..separator + 1);
            if mutations0.is_empty() {
                for &removal in &removals0 {
                    let others = different_ids(operations, &removals0, removal)?;
                    let target = operations.get_mut(removal).ok_or(Error::Index)?;
                    append_ones(&mut target.enabler_ops, &adds0);
                    append_ones(&mut target.disabler_ops, &others);
                }
                let target = operations.get_mut(index).ok_or(Error::Index)?;
                append_group(&mut target.disabler_ops, removals0.clone());
                let repeated = adds0
                    .iter()
                    .copied()
                    .cycle()
                    .take(adds0.len().saturating_mul(removals0.len()))
                    .collect();
                append_group(&mut target.enabler_ops, repeated);
            }
            let (adds1, mutations1, removals1) = source_section(operations, separator + 1..end);
            if mutations1.is_empty() {
                for &removal in &removals1 {
                    let others = different_ids(operations, &removals1, removal)?;
                    let target = operations.get_mut(removal).ok_or(Error::Index)?;
                    append_ones(&mut target.enabler_ops, &adds1);
                    append_ones(&mut target.disabler_ops, &others);
                }
                let target = operations.get_mut(index).ok_or(Error::Index)?;
                append_group(&mut target.disabler_ops, removals1.clone());
                let repeated = adds1
                    .iter()
                    .copied()
                    .cycle()
                    .take(adds1.len().saturating_mul(removals1.len()))
                    .collect();
                append_group(&mut target.enabler_ops, repeated);
            }
            for (adds, mutations, removals) in [
                (adds0, mutations0, removals0),
                (adds1, mutations1, removals1),
            ] {
                if !adds.is_empty() && removals.is_empty() && mutations.is_empty() {
                    let target = operations.get_mut(index).ok_or(Error::Index)?;
                    append_group(&mut target.disabler_ops, vec![index]);
                    append_group(&mut target.enabler_ops, adds);
                }
            }
        }
        3 => {
            let owner = token_id(prepared, true, operation.source_i)?;
            let end = find_loop(
                operations,
                index,
                None,
                &mut locals.end_index,
                &mut locals.end_operation,
                "end_idx",
                |other| {
                    other.op_type == "add_wrap_end"
                        && token_id(prepared, true, other.source_i).is_ok_and(|id| id == owner)
                },
            )?;
            let (adds, mutations, removals) = source_section(operations, index..end + 1);
            let mut disablers = Vec::new();
            let mut enablers = Vec::new();
            if mutations.is_empty() {
                for &removal in &removals {
                    let others = different_ids(operations, &removals, removal)?;
                    let target = operations.get_mut(removal).ok_or(Error::Index)?;
                    append_ones(&mut target.enabler_ops, &adds);
                    append_ones(&mut target.disabler_ops, &others);
                }
                append_ones(&mut disablers, &removals);
                for _ in &removals {
                    append_ones(&mut enablers, &adds);
                }
            }
            let closing = operations.get(end).ok_or(Error::Index)?;
            let (ii, jj) = (closing.i, closing.j);
            let target = operations.get_mut(index).ok_or(Error::Index)?;
            target.ii = Some(ii);
            target.jj = Some(jj);
            target.disabler_ops = disablers;
            target.enabler_ops = enablers;
            if !adds.is_empty() && removals.is_empty() && mutations.is_empty() {
                target.disabler_ops.push(Dependency::One(index));
                append_ones(&mut target.enabler_ops, &adds);
            }
        }
        _ => {}
    }
    Ok(())
}

fn rem_restrictions(
    prepared: &PreparedPair,
    operations: &mut [Edit],
    index: usize,
    locals: &mut ReorderLocals,
) -> Result<()> {
    let operation = operations.get(index).ok_or(Error::Index)?.clone();
    match token_arity(prepared, false, operation.source_j)? {
        3 => {
            let owner = token_id(prepared, false, operation.source_j)?;
            let end = find_loop(
                operations,
                index,
                None,
                &mut locals.end_index,
                &mut locals.end_operation,
                "end_idx",
                |other| {
                    other.op_type == "rem_wrap_end"
                        && token_id(prepared, false, other.source_j).is_ok_and(|id| id == owner)
                },
            )?;
            let (adds, mutations, removals) = source_section(operations, index..end + 1);
            if mutations.is_empty() {
                for &removal in &removals {
                    let target = operations.get_mut(removal).ok_or(Error::Index)?;
                    append_ones(&mut target.disabler_ops, &removals);
                    append_ones(&mut target.enabler_ops, &adds);
                    target.enabler_ops.push(Dependency::One(index));
                }
            }
        }
        4 => {
            let owner = token_id(prepared, false, operation.source_j)?;
            let separator = find_loop(
                operations,
                index + 1,
                None,
                &mut locals.sep_index,
                &mut locals.sep_operation,
                "sep_idx",
                |other| {
                    contains(other, "rem_wrap")
                        && token_id(prepared, false, other.source_j).is_ok_and(|id| id == owner)
                },
            )?;
            let end = find_loop(
                operations,
                separator + 1,
                Some(0),
                &mut locals.end_index,
                &mut locals.end_operation,
                "end_idx",
                |other| {
                    other.op_type == "rem_wrap_end"
                        && token_id(prepared, false, other.source_j).is_ok_and(|id| id == owner)
                },
            )?;
            let (adds0, mutations0, removals0) = source_section(operations, index..separator + 1);
            if mutations0.is_empty() {
                for &removal in &removals0 {
                    let target = operations.get_mut(removal).ok_or(Error::Index)?;
                    append_ones(&mut target.disabler_ops, &removals0);
                    append_ones(&mut target.enabler_ops, &adds0);
                    target.enabler_ops.push(Dependency::One(index));
                }
            }
            let (adds1, mutations1, removals1) = source_section(operations, separator + 1..end);
            if mutations1.is_empty() {
                for &removal in &removals1 {
                    let target = operations.get_mut(removal).ok_or(Error::Index)?;
                    append_ones(&mut target.disabler_ops, &removals1);
                    append_ones(&mut target.enabler_ops, &adds1);
                    target.enabler_ops.push(Dependency::One(index));
                }
            }
        }
        _ => {}
    }
    Ok(())
}

fn calculate_path_restrictions(
    prepared: &PreparedPair,
    operations: &mut [Edit],
    locals: &mut ReorderLocals,
) -> Result<()> {
    process_swaps(prepared, operations, true)?;
    process_swaps(prepared, operations, false)?;
    for index in 0..operations.len() {
        let operation = operations.get(index).ok_or(Error::Index)?.clone();
        if contains(&operation, "add") {
            add_restrictions(prepared, operations, index, locals)?;
        } else if contains(&operation, "rem") {
            rem_restrictions(prepared, operations, index, locals)?;
        }
    }
    Ok(())
}

fn reordered_operations(
    prepared: &PreparedPair,
    source: &[Edit],
    mut locals: ReorderLocals,
) -> Result<(Vec<usize>, Vec<usize>)> {
    let mut operations = Vec::new();
    operations
        .try_reserve(source.len().saturating_sub(1))
        .map_err(|_| Error::Memory)?;
    operations.extend(1..source.len());
    let unordered = operations.clone();
    // The same Python function locals survive restriction processing for every
    // retained history and the subsequent reordering of the selected history.
    let enumerated = operations.clone();
    for (index, operation_index) in enumerated.into_iter().enumerate() {
        let operation = source.get(operation_index).ok_or(Error::Index)?;
        if contains(operation, "add")
            && (token_arity(prepared, true, operation.source_i)? == 4
                || token_name(prepared, true, operation.source_i)?.contains("sep"))
            && operation.i_swapped
        {
            let owner = token_id(prepared, true, operation.source_i)?;
            python_search(
                source,
                &operations,
                index + 1,
                &mut locals.sep_index,
                &mut locals.sep_operation,
                |other| {
                    contains(other, "add")
                        && token_id(prepared, true, other.source_i).is_ok_and(|id| id == owner)
                },
            )?;
            let separator = advance_loop_index(&mut locals.sep_index, index + 1, "sep_idx")?;
            locals.end_index = Some(0);
            python_search(
                source,
                &operations,
                separator + 1,
                &mut locals.end_index,
                &mut locals.end_operation,
                |other| {
                    contains(other, "add")
                        && token_id(prepared, true, other.source_i).is_ok_and(|id| id == owner)
                },
            )?;
            let end = advance_loop_index(&mut locals.end_index, separator + 1, "end_idx")?;
            if token_id(
                prepared,
                true,
                local_operation(locals.sep_operation, "sep_operation")?.0,
            )? == owner
                && token_id(
                    prepared,
                    true,
                    local_operation(locals.end_operation, "end_operation")?.0,
                )? == owner
            {
                rebind_order(&mut operations, index, separator, end)?;
            }
        } else if contains(operation, "rem")
            && (token_arity(prepared, false, operation.source_j)? == 4
                || token_name(prepared, false, operation.source_j)?.contains("sep"))
            && operation.j_swapped
        {
            let owner = token_id(prepared, false, operation.source_j)?;
            python_search(
                source,
                &operations,
                index + 1,
                &mut locals.sep_index,
                &mut locals.sep_operation,
                |other| {
                    contains(other, "rem")
                        && token_id(prepared, false, other.source_j).is_ok_and(|id| id == owner)
                },
            )?;
            let separator = advance_loop_index(&mut locals.sep_index, index + 1, "sep_idx")?;
            locals.end_index = Some(0);
            python_search(
                source,
                &operations,
                separator + 1,
                &mut locals.end_index,
                &mut locals.end_operation,
                |other| {
                    contains(other, "rem")
                        && token_id(prepared, false, other.source_j).is_ok_and(|id| id == owner)
                },
            )?;
            let end = advance_loop_index(&mut locals.end_index, separator + 1, "end_idx")?;
            if token_id(
                prepared,
                false,
                local_operation(locals.sep_operation, "sep_operation")?.1,
            )? == owner
                && token_id(
                    prepared,
                    false,
                    local_operation(locals.end_operation, "end_operation")?.1,
                )? == owner
            {
                rebind_order(&mut operations, index, separator, end)?;
            }
        } else if contains(operation, "mut")
            && (token_arity(prepared, true, operation.source_i)? == 4
                || token_name(prepared, true, operation.source_i)?.contains("sep"))
            && (operation.i_swapped || operation.j_swapped)
        {
            let owner = token_id(prepared, true, operation.source_i)?;
            python_search(
                source,
                &operations,
                index + 1,
                &mut locals.sep_index,
                &mut locals.sep_operation,
                |other| token_id(prepared, true, other.source_i).is_ok_and(|id| id == owner),
            )?;
            let separator = advance_loop_index(&mut locals.sep_index, index + 1, "sep_idx")?;
            python_search(
                source,
                &operations,
                separator + 1,
                &mut locals.end_index,
                &mut locals.end_operation,
                |other| token_id(prepared, true, other.source_i).is_ok_and(|id| id == owner),
            )?;
            let end = advance_loop_index(&mut locals.end_index, separator + 1, "end_idx")?;
            if token_id(
                prepared,
                true,
                local_operation(locals.sep_operation, "sep_operation")?.0,
            )? == owner
                && token_id(
                    prepared,
                    true,
                    local_operation(locals.end_operation, "end_operation")?.0,
                )? == owner
            {
                rebind_order(&mut operations, index, separator, end)?;
            }
        }
    }
    operations.reverse();
    Ok((operations, unordered))
}

#[derive(Default)]
struct ReorderLocals {
    sep_index: Option<usize>,
    end_index: Option<usize>,
    sep_operation: Option<(usize, usize)>,
    end_operation: Option<(usize, usize)>,
}

fn python_search(
    source: &[Edit],
    operations: &[usize],
    start: usize,
    index: &mut Option<usize>,
    operation: &mut Option<(usize, usize)>,
    mut predicate: impl FnMut(&Edit) -> bool,
) -> Result<()> {
    for (relative, &source_index) in operations[start.min(operations.len())..].iter().enumerate() {
        let other = source.get(source_index).ok_or(Error::Index)?;
        *index = Some(relative);
        *operation = Some((other.source_i, other.source_j));
        if predicate(other) {
            break;
        }
    }
    Ok(())
}

fn advance_loop_index(
    index: &mut Option<usize>,
    offset: usize,
    name: &'static str,
) -> Result<usize> {
    let index = index.as_mut().ok_or(Error::Unbound(name))?;
    *index = index.checked_add(offset).ok_or(Error::Index)?;
    Ok(*index)
}

fn local_operation(
    operation: Option<(usize, usize)>,
    name: &'static str,
) -> Result<(usize, usize)> {
    operation.ok_or(Error::Unbound(name))
}

fn rebind_order(
    operations: &mut Vec<usize>,
    index: usize,
    separator: usize,
    end: usize,
) -> Result<()> {
    let length = operations.len();
    let index = index.min(length);
    let separator = separator.min(length);
    let end = end.min(length);
    let prefix = (index + 1).min(length);
    let middle = (index + 1).min(length);
    let separated = end.saturating_sub(separator);
    let intervening = separator.saturating_sub(middle);
    let suffix = length.saturating_sub(end);
    let required = prefix
        .checked_add(separated)
        .and_then(|size| size.checked_add(intervening))
        .and_then(|size| size.checked_add(suffix))
        .ok_or(Error::Memory)?;
    let mut rebound = Vec::new();
    rebound.try_reserve(required).map_err(|_| Error::Memory)?;
    rebound.extend_from_slice(&operations[..prefix]);
    if separator < end {
        rebound.extend_from_slice(&operations[separator..end]);
    }
    if middle < separator {
        rebound.extend_from_slice(&operations[middle..separator]);
    }
    if end < length {
        rebound.extend_from_slice(&operations[end..]);
    }
    *operations = rebound;
    Ok(())
}

fn dependency_groups(path: &[Edit], dependencies: &[Dependency]) -> Result<Vec<Vec<usize>>> {
    let mut groups = Vec::new();
    let mut scalar = Vec::new();
    for dependency in dependencies {
        match dependency {
            Dependency::One(index) => {
                scalar.push(path.get(*index).ok_or(Error::Index)?.id);
            }
            Dependency::Group(indices) => {
                let mut group = Vec::new();
                group
                    .try_reserve(indices.len())
                    .map_err(|_| Error::Memory)?;
                for &index in indices {
                    group.push(path.get(index).ok_or(Error::Index)?.id);
                }
                groups.push(group);
            }
        }
    }
    groups.push(scalar);
    Ok(groups)
}

impl EditPlan {
    fn selected_path(&self) -> Result<&[Edit]> {
        self.paths
            .get(self.path_index)
            .map(Vec::as_slice)
            .ok_or(Error::Index)
    }

    /// Convert the selected nontrivial path to the exact grouped dependency
    /// records consumed by native subset enumeration. Python's historical
    /// adapter keeps list groups and appends all scalar dependencies as one
    /// final group; that unusual shape is intentional.
    pub fn selection_operations(&self) -> Result<Vec<selection::Operation>> {
        let path = self.selected_path()?;
        let groups = |dependencies: &[Dependency]| {
            dependency_groups(path, dependencies)?
                .into_iter()
                .map(|group| {
                    group
                        .into_iter()
                        .map(|id| {
                            i64::try_from(id).map_err(|_| {
                                Error::InvalidInput("operation ID exceeds native range".into())
                            })
                        })
                        .collect::<Result<Vec<_>>>()
                })
                .collect::<Result<Vec<_>>>()
        };
        self.nontrivial
            .iter()
            .map(|&index| {
                let edit = path.get(index).ok_or(Error::Index)?;
                let id = i64::try_from(edit.id)
                    .map_err(|_| Error::InvalidInput("operation ID exceeds native range".into()))?;
                Ok(selection::Operation {
                    id,
                    value: edit.value,
                    enablers: groups(&edit.enabler_ops)?,
                    disablers: groups(&edit.disabler_ops)?,
                })
            })
            .collect()
    }

    /// Validate one explicit selection without enumerating alternative masks.
    /// Membership follows legacy `MatrixOperation.__eq__`: equal operation IDs
    /// are equivalent even when their retained path occurrences differ.
    pub fn validate_selection(&self, indices: &[usize]) -> Result<()> {
        let path = self.selected_path()?;
        let nontrivial: HashSet<usize> = self.nontrivial.iter().copied().collect();
        let mut selected_ids = HashSet::new();
        for &index in indices {
            let edit = path.get(index).ok_or(Error::Index)?;
            if !nontrivial.contains(&index) {
                return Err(Error::InvalidInput(
                    "selection contains a trivial operation".into(),
                ));
            }
            selected_ids.insert(edit.id);
        }
        for &index in indices {
            let edit = path.get(index).ok_or(Error::Index)?;
            let enablers = dependency_groups(path, &edit.enabler_ops)?;
            let disablers = dependency_groups(path, &edit.disabler_ops)?;
            for (branch, enabler) in enablers.iter().enumerate() {
                let disabler = disablers.get(branch).ok_or(Error::Index)?;
                if !disabler.is_empty()
                    && disabler.iter().all(|id| selected_ids.contains(id))
                    && !enabler.iter().any(|id| selected_ids.contains(id))
                {
                    return Err(Error::InvalidInput(
                        "selection violates edit dependencies".into(),
                    ));
                }
            }
        }
        Ok(())
    }
}

fn kernel_tokens(tokens: &[crate::tokens::PreparedToken]) -> Result<Vec<crate::recursive::Token>> {
    let mut result = Vec::new();
    result
        .try_reserve(tokens.len())
        .map_err(|_| Error::Memory)?;
    result.extend(tokens.iter().map(|prepared| prepared.token.clone()));
    Ok(result)
}

/// Analyze a prepared pair, retain every ordered history, and derive the
/// selected path's source-order application indices and dependencies.
pub fn analyze(
    prepared: PreparedPair,
    collapse_corners: bool,
    id: String,
    budget: &mut Budget<'_>,
) -> Result<EditPlan> {
    analyze_with_execution(
        prepared,
        collapse_corners,
        id,
        budget,
        &mut crate::execution::Execution::serial(),
        None,
    )
}

/// Analyze under a call-local execution policy; diagnostics remain on that policy.
pub fn analyze_with_execution(
    prepared: PreparedPair,
    collapse_corners: bool,
    id: String,
    budget: &mut Budget<'_>,
    execution: &mut crate::execution::Execution,
    stage: Option<&mut dyn FnMut(&'static str, bool)>,
) -> Result<EditPlan> {
    analyze_internal(
        prepared,
        collapse_corners,
        id,
        budget,
        stage,
        execution,
        #[cfg(feature = "trace")]
        None,
    )
}

/// Analyze a prepared pair while allowing a host to time native stages.
/// Profiling provides boundaries only; stage timing remains host-owned.
pub fn analyze_profiled(
    prepared: PreparedPair,
    collapse_corners: bool,
    id: String,
    budget: &mut Budget<'_>,
    stage: &mut dyn FnMut(&'static str, bool),
) -> Result<EditPlan> {
    analyze_with_execution(
        prepared,
        collapse_corners,
        id,
        budget,
        &mut crate::execution::Execution::serial(),
        Some(stage),
    )
}

/// Record computation once; callers retain this recording independently of applications.
#[cfg(feature = "trace")]
pub fn analyze_traced(
    prepared: PreparedPair,
    collapse_corners: bool,
    id: String,
    budget: &mut Budget<'_>,
    recorder: &mut crate::trace::Recorder,
) -> Result<EditPlan> {
    analyze_traced_with_execution(
        prepared,
        collapse_corners,
        id,
        budget,
        &mut crate::execution::Execution::serial(),
        recorder,
    )
}

#[cfg(feature = "trace")]
pub fn analyze_traced_with_execution(
    prepared: PreparedPair,
    collapse_corners: bool,
    id: String,
    budget: &mut Budget<'_>,
    execution: &mut crate::execution::Execution,
    recorder: &mut crate::trace::Recorder,
) -> Result<EditPlan> {
    recorder.emit("preparation", || {
        let tokens = |tokens: &[crate::tokens::PreparedToken]| tokens.iter().enumerate().map(|(index,t)| serde_json::json!({
            "index":index,"id":prepared.identities[t.token.id as usize],"name":t.token.name,
            "children":t.token.children,"parent_arity":t.token.parent_arity,"occurrence":t.occurrence
        })).collect::<Vec<_>>();
        serde_json::json!({"collapse_corners":collapse_corners,"direction":"parent2_to_parent1",
            "identities":prepared.identities,"first_tokens":tokens(&prepared.first_tokens),"second_tokens":tokens(&prepared.second_tokens)})
    });
    let result = analyze_internal(
        prepared,
        collapse_corners,
        id,
        budget,
        None,
        execution,
        Some(recorder),
    );
    if let Ok(plan) = &result {
        recorder.emit("plan", || {
            serde_json::json!({
                "path_index":plan.path_index,"paths":plan.paths.iter().map(|path|
                    path.iter().map(|edit| {
                        let mut value = serde_json::to_value(edit).expect("edit serialization is infallible");
                        value["value"] = crate::trace::number(edit.value, false);
                        value
                    }).collect::<Vec<_>>()).collect::<Vec<_>>(),"operations":plan.operations,
                "operations_unordered":plan.operations_unordered,"nontrivial":plan.nontrivial,
                "distance":crate::trace::number(plan.distance,false),
                // Native allocation-byte accounting depends on pointer width;
                // keep it on EditPlan, not in portable computation recordings.
                "stats": {
                    "cells_created":plan.stats["cells_created"],"histories_created":plan.stats["histories_created"],
                    "checkpoints":plan.stats["checkpoints"],"output_steps":plan.stats["output_steps"]
                }
            })
        });
    }
    recorder.emit("outcome", || serde_json::json!({"ok":result.is_ok(),"error":result.as_ref().err().map(ToString::to_string)}));
    result
}

fn analyze_internal(
    prepared: PreparedPair,
    collapse_corners: bool,
    id: String,
    budget: &mut Budget<'_>,
    mut stage: Option<&mut dyn FnMut(&'static str, bool)>,
    execution: &mut crate::execution::Execution,
    #[cfg(feature = "trace")] recorder: Option<&mut crate::trace::Recorder>,
) -> Result<EditPlan> {
    execution.allocation_limit(
        budget
            .limits
            .max_allocation_bytes
            .map(|limit| limit.saturating_sub(budget.allocation_bytes)),
    );
    let first_tokens = kernel_tokens(&prepared.first_tokens)?;
    let second_tokens = kernel_tokens(&prepared.second_tokens)?;
    if let Some(callback) = stage.as_deref_mut() {
        callback("kernel", true);
    }
    let mut previous_cells = 0;
    let mut previous_histories = 0;
    let mut previous_checkpoints = 0;
    let mut previous_output = 0;
    let mut previous_allocation = 0;
    let mut budget_error = None;
    let alignment = {
        let mut observe = |stats: &recursive::Stats, wait_poll: bool| {
            let Some(cells) = stats.cells_created.checked_sub(previous_cells) else {
                budget_error = Some(Error::Limit("work"));
                return Err(Failure::Callback);
            };
            let Some(histories) = stats.histories_created.checked_sub(previous_histories) else {
                budget_error = Some(Error::Limit("work"));
                return Err(Failure::Callback);
            };
            let Some(checkpoints) = stats.checkpoints.checked_sub(previous_checkpoints) else {
                budget_error = Some(Error::Limit("work"));
                return Err(Failure::Callback);
            };
            let Some(work) = cells
                .checked_add(histories)
                .and_then(|total| total.checked_add(checkpoints))
            else {
                budget_error = Some(Error::Limit("work"));
                return Err(Failure::Callback);
            };
            let Some(output) = stats.output_steps.checked_sub(previous_output) else {
                budget_error = Some(Error::Limit("output"));
                return Err(Failure::Callback);
            };
            let Some(allocation) = stats.allocation_bytes.checked_sub(previous_allocation) else {
                budget_error = Some(Error::Limit("allocation"));
                return Err(Failure::Callback);
            };
            if let Err(error) = budget.account(work, output, allocation) {
                budget_error = Some(error);
                return Err(Failure::Callback);
            }
            if checkpoints > 0 {
                if let Err(error) = (budget.check)() {
                    budget_error = Some(error);
                    return Err(Failure::Callback);
                }
            }
            if wait_poll {
                if let Err(error) = budget.poll_wait() {
                    budget_error = Some(error);
                    return Err(Failure::Callback);
                }
            }
            previous_cells = stats.cells_created;
            previous_histories = stats.histories_created;
            previous_checkpoints = stats.checkpoints;
            previous_output = stats.output_steps;
            previous_allocation = stats.allocation_bytes;
            Ok(())
        };
        #[cfg(feature = "trace")]
        if let Some(recorder) = recorder {
            recursive::align_traced_with_execution(
                &first_tokens,
                &second_tokens,
                collapse_corners,
                &mut observe,
                execution,
                recorder,
            )
        } else {
            recursive::align_observed_with_execution(
                &first_tokens,
                &second_tokens,
                collapse_corners,
                &mut observe,
                execution,
            )
        }
        #[cfg(not(feature = "trace"))]
        recursive::align_observed_with_execution(
            &first_tokens,
            &second_tokens,
            collapse_corners,
            &mut observe,
            execution,
        )
    };
    if let Some(callback) = stage.as_deref_mut() {
        callback("kernel", false);
    }
    if let Some(error) = budget_error {
        return Err(error);
    }
    let alignment = alignment.map_err(Error::from)?;
    let recursive::Alignment {
        distance,
        paths: kernel_paths,
        stats: kernel_stats,
    } = alignment;

    if let Some(callback) = stage.as_deref_mut() {
        callback("restrictions", true);
    }
    let planned: Result<_> = (|| {
        let mut paths = Vec::new();
        paths
            .try_reserve(kernel_paths.len())
            .map_err(|_| Error::Memory)?;
        let mut locals = ReorderLocals::default();
        for path in kernel_paths {
            let mut edits = Vec::new();
            edits.try_reserve(path.len()).map_err(|_| Error::Memory)?;
            for step in path {
                edits.push(edit_from_step(&prepared, step)?);
            }
            calculate_path_restrictions(&prepared, &mut edits, &mut locals)?;
            paths.push(edits);
        }
        let selected = paths.first().ok_or(Error::Index)?;
        let (operations, operations_unordered) = reordered_operations(&prepared, selected, locals)?;
        let mut nontrivial = Vec::new();
        nontrivial
            .try_reserve(operations.len())
            .map_err(|_| Error::Memory)?;
        for &index in &operations {
            if selected.get(index).ok_or(Error::Index)?.value != 0.0 {
                nontrivial.push(index);
            }
        }
        Ok((paths, operations, operations_unordered, nontrivial))
    })();
    if let Some(callback) = stage {
        callback("restrictions", false);
    }
    let (paths, operations, operations_unordered, nontrivial) = planned?;

    let mut stats = BTreeMap::new();
    stats.insert("allocation_bytes".into(), kernel_stats.allocation_bytes);
    stats.insert("cells_created".into(), kernel_stats.cells_created);
    stats.insert("checkpoints".into(), kernel_stats.checkpoints);
    stats.insert("histories_created".into(), kernel_stats.histories_created);
    stats.insert("output_steps".into(), kernel_stats.output_steps);
    Ok(EditPlan {
        id,
        prepared,
        distance,
        paths,
        path_index: 0,
        operations,
        operations_unordered,
        nontrivial,
        stats,
    })
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::architecture::{Architecture, Limits, Node, SCHEMA_VERSION};
    use crate::tokens::prepare;

    fn chain(id: &str, operation: &str) -> Architecture {
        Architecture {
            schema: SCHEMA_VERSION,
            grammar: "einspace".into(),
            grammar_version: "1".into(),
            root: 0,
            nodes: vec![
                Node {
                    id: id.into(),
                    name: "computation".into(),
                    children: vec![1],
                    parameters: None,
                    provenance: None,
                },
                Node {
                    id: "0".into(),
                    name: operation.into(),
                    children: vec![],
                    parameters: None,
                    provenance: None,
                },
            ],
            input_spec: None,
        }
    }

    fn edit(id: usize, op_type: &str, i: usize, j: usize, value: f64) -> Edit {
        Edit {
            id,
            op_type: op_type.into(),
            node1_id: Some("first".into()),
            node2_id: Some("second".into()),
            i,
            j,
            source_i: i,
            source_j: j,
            source_i_swapped: false,
            source_j_swapped: false,
            ii: None,
            jj: None,
            value,
            i_swapped: false,
            j_swapped: false,
            enabler_ops: vec![],
            disabler_ops: vec![],
        }
    }

    #[test]
    fn retains_native_history_and_rejects_trivial_selection() {
        let prepared = prepare(chain("5", "identity"), chain("9", "relu")).unwrap();
        let mut check = || Ok(());
        let mut budget = Budget {
            limits: Limits::default(),
            work: 0,
            output: 0,
            allocation_bytes: 0,
            check: &mut check,
            wait_check: None,
        };
        let plan = analyze(prepared, false, "plan".into(), &mut budget).unwrap();
        assert_eq!(plan.distance, 0.5);
        assert!(
            plan.paths
                .iter()
                .all(|path| path.first().is_some_and(|edit| edit.op_type == "start"))
        );
        assert_eq!(plan.nontrivial.len(), 1);
        assert!(plan.validate_selection(&plan.nontrivial).is_ok());
        assert_eq!(
            plan.validate_selection(&[0]),
            Err(Error::InvalidInput(
                "selection contains a trivial operation".into()
            ))
        );
        assert_eq!(plan.validate_selection(&[usize::MAX]), Err(Error::Index));
    }

    #[test]
    fn preserves_scalar_dependency_grouping_and_wrapper_restrictions() {
        let mut prepared = prepare(chain("1", "identity"), chain("2", "relu")).unwrap();
        prepared.identities = vec!["-1".into(), "first".into(), "second".into()];
        prepared.first_tokens[1].token.children = vec!["a".into(), "b".into(), "c".into()];
        prepared.first_tokens.push(crate::tokens::PreparedToken {
            token: crate::recursive::Token {
                id: 1,
                name: "wrap_end".into(),
                children: vec![],
                parent_arity: 3,
            },
            occurrence: Some(0),
        });
        prepared.second_tokens[1].token.children = vec!["a".into(), "b".into(), "c".into()];
        prepared.second_tokens.push(crate::tokens::PreparedToken {
            token: crate::recursive::Token {
                id: 2,
                name: "wrap_end".into(),
                children: vec![],
                parent_arity: 3,
            },
            occurrence: Some(0),
        });
        let mut path = vec![
            edit(0, "start", 0, 0, 0.0),
            edit(1, "add_wrap", 1, 1, 1.0),
            edit(2, "rem", 1, 1, 1.0),
            edit(3, "add_wrap_end", 2, 2, 0.0),
        ];
        calculate_path_restrictions(&prepared, &mut path, &mut ReorderLocals::default()).unwrap();
        assert_eq!(path[1].ii, Some(2));
        assert_eq!(path[1].jj, Some(2));
        assert_eq!(path[1].disabler_ops, vec![Dependency::One(2)]);
        let plan = EditPlan {
            id: "plan".into(),
            prepared,
            distance: 2.0,
            paths: vec![path],
            path_index: 0,
            operations: vec![1, 2, 3],
            operations_unordered: vec![1, 2, 3],
            nontrivial: vec![1, 2],
            stats: BTreeMap::new(),
        };
        let records = plan.selection_operations().unwrap();
        assert_eq!(records.len(), 2);
        assert_eq!(records[0].disablers, vec![vec![2]]);
    }

    fn branched_prepared() -> crate::tokens::PreparedPair {
        let mut prepared = prepare(chain("1", "identity"), chain("2", "relu")).unwrap();
        prepared.identities = vec!["-1".into(), "root".into(), "left".into(), "right".into()];
        prepared.first_tokens[1] = crate::tokens::PreparedToken {
            token: crate::recursive::Token {
                id: 1,
                name: "branching".into(),
                children: vec!["clone".into(), "left".into(), "right".into(), "add".into()],
                parent_arity: 4,
            },
            occurrence: None,
        };
        for (id, name, occurrence) in [
            (2, "identity", Some(1)),
            (1, "wrap_sep", Some(0)),
            (3, "relu", Some(2)),
            (1, "wrap_end", Some(0)),
        ] {
            prepared.first_tokens.push(crate::tokens::PreparedToken {
                token: crate::recursive::Token {
                    id,
                    name: name.into(),
                    children: vec![],
                    parent_arity: 4,
                },
                occurrence,
            });
        }
        prepared
    }

    #[test]
    fn omits_branch_wrapper_groups_when_both_sides_mutate() {
        let prepared = branched_prepared();
        let mut path = vec![
            edit(0, "start", 0, 0, 0.0),
            edit(1, "add_wrap", 1, 0, 1.0),
            edit(2, "mut", 2, 0, 0.0),
            edit(3, "add_wrap_sep", 3, 0, 0.0),
            edit(4, "mut", 4, 0, 0.0),
            edit(5, "add_wrap_end", 5, 0, 0.0),
        ];
        add_restrictions(&prepared, &mut path, 1, &mut ReorderLocals::default()).unwrap();
        assert!(path[1].enabler_ops.is_empty());
        assert!(path[1].disabler_ops.is_empty());
    }

    #[test]
    fn omits_ternary_wrapper_dependencies_when_the_interval_mutates() {
        let mut prepared = prepare(chain("1", "identity"), chain("2", "relu")).unwrap();
        prepared.identities = vec!["-1".into(), "root".into(), "child".into()];
        prepared.first_tokens[1] = crate::tokens::PreparedToken {
            token: crate::recursive::Token {
                id: 1,
                name: "branching".into(),
                children: vec!["clone".into(), "child".into(), "add".into()],
                parent_arity: 3,
            },
            occurrence: None,
        };
        for (id, name, occurrence) in [(2, "identity", Some(1)), (1, "wrap_end", Some(0))] {
            prepared.first_tokens.push(crate::tokens::PreparedToken {
                token: crate::recursive::Token {
                    id,
                    name: name.into(),
                    children: vec![],
                    parent_arity: 3,
                },
                occurrence,
            });
        }
        let mut path = vec![
            edit(0, "start", 0, 0, 0.0),
            edit(1, "add_wrap", 1, 0, 1.0),
            edit(2, "mut", 2, 0, 0.0),
            edit(3, "rem", 2, 0, 1.0),
            edit(4, "add_wrap_end", 3, 0, 0.0),
        ];

        add_restrictions(&prepared, &mut path, 1, &mut ReorderLocals::default()).unwrap();

        assert_eq!(path[1].ii, Some(3));
        assert_eq!(path[1].jj, Some(0));
        assert!(path[1].enabler_ops.is_empty());
        assert!(path[1].disabler_ops.is_empty());
    }

    #[test]
    fn preserves_python_rebind_locals_across_empty_mutation_loop() {
        let prepared = branched_prepared();
        let mut source = vec![
            edit(0, "start", 0, 0, 0.0),
            edit(1, "mut_wrap", 3, 0, 0.0),
            edit(2, "mut", 4, 0, 0.0),
            edit(3, "mut_wrap_sep", 1, 0, 0.0),
            edit(4, "mut", 2, 0, 0.0),
            edit(5, "mut_wrap_end", 5, 0, 0.0),
        ];
        source[1].i_swapped = true;
        source[3].i_swapped = true;
        let (ordered, unordered) =
            reordered_operations(&prepared, &source, ReorderLocals::default()).unwrap();
        assert_eq!(unordered, vec![1, 2, 3, 4, 5]);
        assert_eq!(ordered, vec![2, 5, 4, 3, 1]);
    }
}
