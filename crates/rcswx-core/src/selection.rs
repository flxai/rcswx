//! Ordered reference subset enumeration without a machine-word edit ceiling.
//! Prefixes are pruned only after a dependency violation is irrevocable. This
//! removes no valid mask and preserves lexicographic enumeration and multiplicity.
use std::collections::HashMap;

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

pub fn combinations(operations: &[Operation]) -> Result<(Vec<String>, Vec<f64>), Failure> {
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
                    let disablers = operation.disablers.get(branch).ok_or(Failure::Index)?;
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
                let mut mask = String::new();
                mask.try_reserve_exact(bits.len().max(1))
                    .map_err(|_| Failure::Memory)?;
                if bits.is_empty() {
                    mask.push('0');
                } else {
                    for selected in &bits {
                        mask.push(if *selected { '1' } else { '0' });
                    }
                }
                masks.try_reserve(1).map_err(|_| Failure::Memory)?;
                values.try_reserve(1).map_err(|_| Failure::Memory)?;
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
