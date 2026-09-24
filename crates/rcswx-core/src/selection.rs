//! Ordered reference subset enumeration without a machine-word edit ceiling.
//! Prefixes are pruned only after a dependency violation is irrevocable. This
//! removes no valid mask and preserves lexicographic enumeration and multiplicity.
use crate::architecture::Budget;
use crate::error::{Error, Result as CoreResult};
use std::collections::HashMap;

#[derive(Clone, Debug)]
pub struct Operation {
    pub id: i64,
    pub value: f64,
    pub enablers: Vec<Vec<i64>>,
    pub disablers: Vec<Vec<i64>>,
}
#[derive(Debug)]
pub enum Failure {
    Index,
    Memory,
}

fn checkpoint(
    budget: &mut Option<&mut Budget<'_>>,
    work: usize,
    output: usize,
    allocation: usize,
) -> CoreResult<()> {
    if let Some(budget) = budget.as_deref_mut() {
        budget.checkpoint(work, output, allocation)?;
    }
    Ok(())
}

fn membership(
    id: i64,
    positions: &HashMap<i64, Vec<usize>>,
    bits: &[bool],
    depth: usize,
) -> Option<bool> {
    let Some(indices) = positions.get(&id) else {
        return Some(false);
    };
    if indices.iter().any(|&index| index < depth && bits[index]) {
        return Some(true);
    }
    if indices.iter().any(|&index| index >= depth) {
        None
    } else {
        Some(false)
    }
}

fn combinations_inner(
    operations: &[Operation],
    mut budget: Option<&mut Budget<'_>>,
) -> CoreResult<(Vec<String>, Vec<f64>)> {
    // Charge the input-sized live bit set before allocating it. Hash-map bucket
    // layout is implementation-owned; the operation count is its portable bound.
    checkpoint(&mut budget, operations.len(), 0, operations.len())?;
    let mut positions = HashMap::<i64, Vec<usize>>::new();
    for (index, operation) in operations.iter().enumerate() {
        positions.entry(operation.id).or_default().push(index);
    }
    // Malformed groups may raise even in an already-invalid mask. Never prune
    // those inputs: execute the source's checks in source order instead.
    let prune = operations
        .iter()
        .all(|op| op.enablers.len() <= op.disablers.len());
    let constrained = operations
        .iter()
        .any(|op| op.disablers.iter().any(|group| !group.is_empty()));
    let mut bits = vec![false; operations.len()];
    let mut depth = 0;
    let mut masks = Vec::new();
    let mut values = Vec::new();
    loop {
        checkpoint(&mut budget, 1, 0, 0)?;
        let mut invalid_prefix = false;
        if prune && constrained {
            for (index, operation) in operations.iter().enumerate().take(depth) {
                if !bits[index] {
                    continue;
                }
                for (enablers, disablers) in operation.enablers.iter().zip(&operation.disablers) {
                    if !disablers.is_empty()
                        && disablers
                            .iter()
                            .all(|&id| membership(id, &positions, &bits, depth) == Some(true))
                        && enablers
                            .iter()
                            .all(|&id| membership(id, &positions, &bits, depth) == Some(false))
                    {
                        invalid_prefix = true;
                        break;
                    }
                }
                if invalid_prefix {
                    break;
                }
            }
        }
        if !invalid_prefix && depth < bits.len() {
            bits[depth] = false;
            depth += 1;
            continue;
        }
        if !invalid_prefix && depth == bits.len() {
            let mut value = 0.0;
            for (selected, operation) in bits.iter().zip(operations) {
                if *selected {
                    value += operation.value;
                }
            }
            for (selected, operation) in bits.iter().zip(operations) {
                if !selected {
                    continue;
                }
                for (branch, enablers) in operation.enablers.iter().enumerate() {
                    let disablers = operation.disablers.get(branch).ok_or(Error::Index)?;
                    if !disablers.is_empty()
                        && disablers
                            .iter()
                            .all(|&id| membership(id, &positions, &bits, depth) == Some(true))
                        && !enablers
                            .iter()
                            .any(|&id| membership(id, &positions, &bits, depth) == Some(true))
                    {
                        value = f64::NAN;
                    }
                }
            }
            if !value.is_nan() {
                // Check output and its input-dependent storage before allocation.
                let allocation = bits
                    .len()
                    .max(1)
                    .checked_add(std::mem::size_of::<String>())
                    .and_then(|size| size.checked_add(std::mem::size_of::<f64>()))
                    .ok_or(Error::Limit("allocation"))?;
                checkpoint(&mut budget, 0, 1, allocation)?;
                let mut mask = String::new();
                mask.try_reserve_exact(bits.len().max(1))
                    .map_err(|_| Error::Memory)?;
                if bits.is_empty() {
                    mask.push('0');
                } else {
                    for selected in &bits {
                        mask.push(if *selected { '1' } else { '0' });
                    }
                }
                masks.try_reserve(1).map_err(|_| Error::Memory)?;
                values.try_reserve(1).map_err(|_| Error::Memory)?;
                masks.push(mask);
                values.push(value);
            }
        }
        while depth > 0 && bits[depth - 1] {
            depth -= 1;
        }
        if depth == 0 {
            break;
        }
        bits[depth - 1] = true;
    }
    Ok((masks, values))
}

/// Enumerate masks using the historical subset law without operational limits.
pub fn combinations(operations: &[Operation]) -> Result<(Vec<String>, Vec<f64>), Failure> {
    combinations_inner(operations, None).map_err(|error| match error {
        Error::Index => Failure::Index,
        Error::Memory => Failure::Memory,
        _ => unreachable!("unbudgeted selection cannot produce a core error"),
    })
}

/// Enumerate masks while charging selection search and output to `budget`.
pub fn combinations_with_budget(
    operations: &[Operation],
    budget: &mut Budget<'_>,
) -> CoreResult<(Vec<String>, Vec<f64>)> {
    combinations_inner(operations, Some(budget))
}

/// Test a selected set directly without enumerating every legal mask.
///
/// Membership remains operation-ID based, matching the reference's overloaded
/// `MatrixOperation.__eq__`. Repeated selected indices are idempotent because a
/// selected mask has one bit per operation occurrence.
pub fn is_legal(operations: &[Operation], selected: &[usize]) -> Result<bool, Failure> {
    let mut bits = vec![false; operations.len()];
    for &index in selected {
        *bits.get_mut(index).ok_or(Failure::Index)? = true;
    }
    let mut positions = HashMap::<i64, Vec<usize>>::new();
    for (index, operation) in operations.iter().enumerate() {
        positions.entry(operation.id).or_default().push(index);
    }
    for (index, operation) in operations.iter().enumerate() {
        if !bits[index] {
            continue;
        }
        for (branch, enablers) in operation.enablers.iter().enumerate() {
            let disablers = operation.disablers.get(branch).ok_or(Failure::Index)?;
            if !disablers.is_empty()
                && disablers
                    .iter()
                    .all(|&id| membership(id, &positions, &bits, operations.len()) == Some(true))
                && !enablers
                    .iter()
                    .any(|&id| membership(id, &positions, &bits, operations.len()) == Some(true))
            {
                return Ok(false);
            }
        }
    }
    Ok(true)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn direct_legality_matches_operation_id_membership() {
        let operations = [
            Operation {
                id: 10,
                value: 1.0,
                enablers: vec![vec![]],
                disablers: vec![vec![20]],
            },
            Operation {
                id: 20,
                value: 1.0,
                enablers: vec![],
                disablers: vec![],
            },
            // A repeated logical ID is one membership identity in the reference.
            Operation {
                id: 20,
                value: 1.0,
                enablers: vec![],
                disablers: vec![],
            },
        ];
        assert!(is_legal(&operations, &[0]).unwrap());
        assert!(!is_legal(&operations, &[0, 2]).unwrap());
    }

    #[test]
    fn direct_legality_preserves_malformed_branch_index_error() {
        let operations = [Operation {
            id: 0,
            value: 1.0,
            enablers: vec![vec![]],
            disablers: vec![],
        }];
        assert!(matches!(is_legal(&operations, &[0]), Err(Failure::Index)));
    }
}
