use std::mem::size_of;
use std::sync::Arc;

use crate::metadata::{Guard, allocation};
use crate::{Architecture, CancellationToken, Error, Limits, Op, Result, Stats, check_limit};

pub(crate) fn substitution(left: &Op, right: &Op) -> u64 {
    if left == right {
        0
    } else if left.kind == right.kind {
        1
    } else {
        2
    }
}
fn add(left: u64, right: u64) -> Result<u64> {
    left.checked_add(right)
        .ok_or_else(|| Error::InternalInvariant("cost overflow".into()))
}
fn input_bytes(source: &Architecture, target: &Architecture) -> usize {
    source.owned_bytes()
        + if std::ptr::eq(source, target) {
            0
        } else {
            target.owned_bytes()
        }
}
fn dimensions(
    source: &Architecture,
    target: &Architecture,
    guard: &Guard,
) -> Result<(usize, usize, usize)> {
    check_limit("max_nodes", guard.limits.max_nodes, source.ops.len())?;
    check_limit("max_nodes", guard.limits.max_nodes, target.ops.len())?;
    let (n, m) = (source.ops.len(), target.ops.len());
    let states = (n + 1).checked_mul(m + 1).ok_or(Error::BudgetExceeded {
        resource: "max_states",
        limit: guard.limits.max_states as u64,
        observed: u64::MAX,
    })?;
    check_limit("max_states", guard.limits.max_states, states)?;
    Ok((n, m, states))
}

pub struct Distance {
    pub cost_ticks: u64,
    pub stats: Stats,
}

/// Empty structural-obligation state: continuations depend only on suffix positions.
/// This rolling-row compression is valid only for the registered ordered-unary grammar.
pub fn distance(
    source: &Architecture,
    target: &Architecture,
    limits: &Limits,
    cancel: &CancellationToken,
) -> Result<Distance> {
    let mut guard = Guard::new(limits, cancel)?;
    let (_, _, states) = dimensions(source, target, &guard)?;
    let base = input_bytes(source, target);
    let (left, right) = if source.ops.len() >= target.ops.len() {
        (&source.ops, &target.ops)
    } else {
        (&target.ops, &source.ops)
    };
    let columns = right.len() + 1;
    let mut rows = allocation::<u64>(2 * columns, base, &guard)?;
    rows.resize(2 * columns, 0);
    let (mut previous, mut current) = rows.split_at_mut(columns);
    for (j, value) in previous.iter_mut().enumerate() {
        *value = 4 * j as u64;
        guard.tick()?;
    }
    for (i, left_op) in left.iter().enumerate() {
        current[0] = 4 * (i as u64 + 1);
        guard.tick()?;
        for (j, right_op) in right.iter().enumerate() {
            current[j + 1] = add(previous[j], substitution(left_op, right_op))?
                .min(add(previous[j + 1], 4)?)
                .min(add(current[j], 4)?);
            guard.tick()?;
        }
        std::mem::swap(&mut previous, &mut current);
    }
    guard.check()?;
    Ok(Distance {
        cost_ticks: previous[columns - 1],
        stats: Stats {
            states,
            predecessors: 0,
            charged_bytes: base + 2 * columns * size_of::<u64>(),
        },
    })
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct Step {
    pub kind: u8,
    pub source: Option<usize>,
    pub target: Option<usize>,
    pub cost_ticks: u64,
    pub edit_id: Option<usize>,
}
#[derive(Clone, Copy)]
pub(crate) struct Effect {
    pub cost: u64,
    pub length_delta: i8,
}

pub struct Path {
    source: Arc<Architecture>,
    target: Arc<Architecture>,
    cost_ticks: u64,
    steps: Vec<Step>,
    stats: Stats,
    pub(crate) effects: Vec<Effect>,
}
impl Path {
    pub fn cost_ticks(&self) -> u64 {
        self.cost_ticks
    }
    pub fn steps(&self) -> &[Step] {
        &self.steps
    }
    pub fn stats(&self) -> Stats {
        self.stats
    }
    pub fn edit_count(&self) -> usize {
        self.effects.len()
    }
    pub fn source_len(&self) -> usize {
        self.source.ops.len()
    }
    pub fn owned_bytes(&self) -> usize {
        input_bytes(&self.source, &self.target)
            + size_of::<Self>()
            + self.steps.capacity() * size_of::<Step>()
            + self.effects.capacity() * size_of::<Effect>()
    }
    fn check_mask(&self, mask: u64) -> Result<()> {
        if mask >> self.effects.len() != 0 {
            Err(Error::InvalidSelection(
                "mask contains an unknown edit bit".into(),
            ))
        } else {
            Ok(())
        }
    }
    pub fn selected_cost(&self, mask: u64) -> Result<u64> {
        self.check_mask(mask)?;
        let mut result = 0;
        for (bit, effect) in self.effects.iter().enumerate() {
            if mask & (1 << bit) != 0 {
                result = add(result, effect.cost)?;
            }
        }
        Ok(result)
    }
    pub fn project(
        &self,
        mask: u64,
        limits: &Limits,
        cancel: &CancellationToken,
    ) -> Result<Vec<(u8, usize)>> {
        let mut guard = Guard::new(limits, cancel)?;
        self.check_mask(mask)?;
        check_limit("max_edits", limits.max_edits, self.edit_count())?;
        let mut count = self.source.ops.len() as i64;
        for (bit, effect) in self.effects.iter().enumerate() {
            if mask & (1 << bit) != 0 {
                count += effect.length_delta as i64;
            }
        }
        if count <= 0 {
            return Err(Error::InvalidSelection(
                "selection removes the entire required sequence".into(),
            ));
        }
        check_limit("max_nodes", limits.max_nodes, count as usize)?;
        let mut result = allocation(count as usize, self.owned_bytes(), &guard)?;
        for step in &self.steps {
            let selected = step.edit_id.is_some_and(|bit| mask & (1 << bit) != 0);
            let origin = match step.kind {
                0 => Some((0, step.source)),
                1 if selected => Some((1, step.target)),
                1 | 2 if !selected => Some((0, step.source)),
                3 if selected => Some((1, step.target)),
                2 | 3 => None,
                _ => return Err(Error::InternalInvariant("invalid event tag".into())),
            };
            if let Some((side, address)) = origin {
                result.push((
                    side,
                    address.ok_or_else(|| {
                        Error::InternalInvariant("missing occurrence address".into())
                    })?,
                ));
            }
            guard.tick()?;
        }
        if result.len() != count as usize {
            return Err(Error::InternalInvariant(
                "projection length mismatch".into(),
            ));
        }
        guard.check()?;
        Ok(result)
    }
}

/// Suffix optimal costs plus first optimal forward edge yields the globally
/// lexicographically smallest complete witness, not a predecessor-local heuristic.
/// At each position there is at most one edge per tag: match, substitute, delete, insert.
pub fn align(
    source: Arc<Architecture>,
    target: Arc<Architecture>,
    limits: &Limits,
    cancel: &CancellationToken,
) -> Result<Path> {
    let mut guard = Guard::new(limits, cancel)?;
    let (n, m, states) = dimensions(&source, &target, &guard)?;
    check_limit("max_edits", limits.max_edits, n.abs_diff(m))?;
    let columns = m + 1;
    let inputs = input_bytes(&source, &target);
    let mut suffix = allocation::<u64>(states, inputs, &guard)?;
    suffix.resize(states, 0);
    for i in (0..=n).rev() {
        for j in (0..=m).rev() {
            suffix[i * columns + j] = match (i == n, j == m) {
                (true, true) => 0,
                (true, false) => add(4, suffix[i * columns + j + 1])?,
                (false, true) => add(4, suffix[(i + 1) * columns + j])?,
                (false, false) => add(
                    substitution(&source.ops[i], &target.ops[j]),
                    suffix[(i + 1) * columns + j + 1],
                )?
                .min(add(4, suffix[(i + 1) * columns + j])?)
                .min(add(4, suffix[i * columns + j + 1])?),
            };
            guard.tick()?;
        }
    }
    let cost_ticks = suffix[0];
    let max_steps = (n + m)
        .min(n.max(m) + limits.max_edits / 2)
        .min(limits.max_predecessors);
    let effect_capacity = limits.max_edits.min(cost_ticks as usize);
    let matrix_bytes = inputs + states * size_of::<u64>();
    let mut steps = allocation::<Step>(max_steps, matrix_bytes + size_of::<Path>(), &guard)?;
    let mut effects = allocation::<Effect>(
        effect_capacity,
        matrix_bytes + size_of::<Path>() + max_steps * size_of::<Step>(),
        &guard,
    )?;
    let (mut i, mut j) = (0, 0);
    while i < n || j < m {
        let optimum = suffix[i * columns + j];
        let mut step = None;
        if i < n && j < m {
            let cost = substitution(&source.ops[i], &target.ops[j]);
            if add(cost, suffix[(i + 1) * columns + j + 1])? == optimum {
                step = Some(Step {
                    kind: if cost == 0 { 0 } else { 1 },
                    source: Some(i),
                    target: Some(j),
                    cost_ticks: cost,
                    edit_id: None,
                });
            }
        }
        if step.is_none() && i < n && add(4, suffix[(i + 1) * columns + j])? == optimum {
            step = Some(Step {
                kind: 2,
                source: Some(i),
                target: None,
                cost_ticks: 4,
                edit_id: None,
            });
        }
        if step.is_none() && j < m && add(4, suffix[i * columns + j + 1])? == optimum {
            step = Some(Step {
                kind: 3,
                source: None,
                target: Some(j),
                cost_ticks: 4,
                edit_id: None,
            });
        }
        let mut step =
            step.ok_or_else(|| Error::InternalInvariant("no optimal continuation".into()))?;
        check_limit("max_predecessors", limits.max_predecessors, steps.len() + 1)?;
        if step.cost_ticks > 0 {
            check_limit("max_edits", limits.max_edits, effects.len() + 1)?;
            step.edit_id = Some(effects.len());
            effects.push(Effect {
                cost: step.cost_ticks,
                length_delta: match step.kind {
                    2 => -1,
                    3 => 1,
                    _ => 0,
                },
            });
        }
        if step.source.is_some() {
            i += 1;
        }
        if step.target.is_some() {
            j += 1;
        }
        steps.push(step);
        guard.tick()?;
    }
    guard.check()?;
    let stats = Stats {
        states,
        predecessors: steps.len(),
        charged_bytes: matrix_bytes
            + size_of::<Path>()
            + steps.capacity() * size_of::<Step>()
            + effects.capacity() * size_of::<Effect>(),
    };
    Ok(Path {
        source,
        target,
        cost_ticks,
        steps,
        stats,
        effects,
    })
}

#[cfg(test)]
mod tests {
    use super::*;

    fn identity() -> Op {
        Op {
            kind: 0,
            width: 0,
            bias: false,
            param: 0.0,
            momentum: None,
            track: false,
        }
    }
    fn linear(width: u64) -> Op {
        Op {
            kind: 1,
            width,
            bias: true,
            ..identity()
        }
    }
    fn architecture(ops: Vec<Op>) -> Arc<Architecture> {
        Arc::new(Architecture::new(ops, &Limits::default()).unwrap())
    }

    #[test]
    fn complete_forward_tie_prefers_substitution_before_insertion() {
        let source = architecture(vec![identity()]);
        let target = architecture(vec![linear(8), linear(4)]);
        let path = align(
            source,
            target,
            &Limits::default(),
            &CancellationToken::default(),
        )
        .unwrap();
        assert_eq!(path.cost_ticks(), 6);
        assert_eq!(
            path.steps()
                .iter()
                .map(|s| (s.kind, s.source, s.target))
                .collect::<Vec<_>>(),
            vec![(1, Some(0), Some(0)), (3, None, Some(1))]
        );
    }

    #[test]
    fn mixed_cost_events_reconstruct_both_complete_endpoints() {
        let relu = Op {
            kind: 2,
            ..identity()
        };
        let source = architecture(vec![identity(), linear(8), relu, linear(4)]);
        let target = architecture(vec![linear(16), relu, linear(4)]);
        let limits = Limits::default();
        let token = CancellationToken::default();
        let path = align(source.clone(), target.clone(), &limits, &token).unwrap();
        assert_eq!(
            distance(&source, &target, &limits, &token)
                .unwrap()
                .cost_ticks,
            5
        );
        assert_eq!(path.selected_cost(1).unwrap(), 4);
        assert_eq!(path.selected_cost(2).unwrap(), 1);
        for (mask, expected) in [(0, &source.ops), (3, &target.ops)] {
            let projected = path
                .project(mask, &limits, &token)
                .unwrap()
                .into_iter()
                .map(|(side, index)| {
                    if side == 0 {
                        source.ops[index]
                    } else {
                        target.ops[index]
                    }
                })
                .collect::<Vec<_>>();
            assert_eq!(&projected, expected);
        }
        assert!(matches!(
            path.project(4, &limits, &token),
            Err(Error::InvalidSelection(_))
        ));
    }

    #[test]
    fn empty_projection_is_not_repaired_with_an_identity() {
        let source = architecture(vec![identity()]);
        let target = architecture(vec![Op {
            kind: 2,
            ..identity()
        }]);
        // A legal, deliberately nonoptimal history tests the translator separately.
        let path = Path {
            source,
            target,
            cost_ticks: 8,
            stats: Stats::default(),
            steps: vec![
                Step {
                    kind: 2,
                    source: Some(0),
                    target: None,
                    cost_ticks: 4,
                    edit_id: Some(0),
                },
                Step {
                    kind: 3,
                    source: None,
                    target: Some(0),
                    cost_ticks: 4,
                    edit_id: Some(1),
                },
            ],
            effects: vec![
                Effect {
                    cost: 4,
                    length_delta: -1,
                },
                Effect {
                    cost: 4,
                    length_delta: 1,
                },
            ],
        };
        assert!(matches!(
            path.project(1, &Limits::default(), &CancellationToken::default()),
            Err(Error::InvalidSelection(_))
        ));
    }

    #[test]
    fn zero_path_retains_source_correspondence_and_has_no_bits() {
        let source = architecture(vec![identity(), linear(4)]);
        let path = align(
            source.clone(),
            source,
            &Limits {
                max_edits: 0,
                ..Limits::default()
            },
            &CancellationToken::default(),
        )
        .unwrap();
        assert_eq!(path.cost_ticks(), 0);
        assert_eq!(
            path.project(0, &Limits::default(), &CancellationToken::default())
                .unwrap(),
            vec![(0, 0), (0, 1)]
        );
        assert!(matches!(
            path.selected_cost(1),
            Err(Error::InvalidSelection(_))
        ));
    }

    #[test]
    fn resource_stops_and_cancellation_are_not_negative_alignment_proofs() {
        let source = architecture(vec![identity(), linear(4)]);
        let target = architecture(vec![linear(4)]);
        let token = CancellationToken::default();
        assert!(matches!(
            distance(
                &source,
                &target,
                &Limits {
                    max_states: 5,
                    ..Limits::default()
                },
                &token
            ),
            Err(Error::BudgetExceeded {
                resource: "max_states",
                ..
            })
        ));
        assert!(matches!(
            align(
                source.clone(),
                target.clone(),
                &Limits {
                    max_predecessors: 1,
                    ..Limits::default()
                },
                &token
            ),
            Err(Error::BudgetExceeded {
                resource: "max_predecessors",
                ..
            })
        ));
        assert!(matches!(
            distance(
                &source,
                &target,
                &Limits {
                    max_core_bytes: 1,
                    ..Limits::default()
                },
                &token
            ),
            Err(Error::BudgetExceeded {
                resource: "max_core_bytes",
                ..
            })
        ));
        token.cancel();
        assert!(matches!(
            distance(&source, &target, &Limits::default(), &token),
            Err(Error::Cancelled)
        ));
    }
}
