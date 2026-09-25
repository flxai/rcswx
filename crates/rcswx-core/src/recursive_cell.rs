//! A cell's read phase owns no mutable matrix, recorder, or host callback.
use super::{Cell, Failure, Kernel, Kind, Path, Result, Step, Token, mutation_cost};
use std::sync::atomic::{AtomicBool, Ordering};

#[inline]
pub(super) fn check_cancel(cancelled: Option<&AtomicBool>) -> Result<()> {
    if cancelled.is_some_and(|flag| flag.load(Ordering::Relaxed)) {
        Err(Failure::Operational(crate::error::Error::Cancelled))
    } else {
        Ok(())
    }
}

/// Borrow guards and their owning cells stay on the coordinator. These views
/// contain immutable values and cannot outlive the guards kept by the caller.
pub(super) struct Input<'a> {
    pub destination: &'a Cell,
    pub predecessors: [&'a Cell; 3],
    pub first: &'a Token,
    pub second: &'a Token,
    pub position: (usize, usize),
}

impl Input<'_> {
    pub fn step(&self, direction: usize, parent: &Path, value: f64) -> Step {
        let one = self.first;
        let two = self.second;
        let source = self.predecessors[direction];
        let kind = match direction {
            0 => {
                if one.name == "wrap_end" {
                    Kind::AddEnd
                } else if one.name == "wrap_sep" {
                    Kind::AddSep
                } else if one.children.len() >= 3 {
                    Kind::AddWrap
                } else {
                    Kind::AddModule
                }
            }
            1 => {
                if two.name == "wrap_end" {
                    Kind::RemEnd
                } else if two.name == "wrap_sep" {
                    Kind::RemSep
                } else if two.children.len() >= 3 {
                    Kind::RemWrap
                } else {
                    Kind::Rem
                }
            }
            _ => {
                if one.name == "wrap_end" {
                    Kind::MutEnd
                } else if one.name == "wrap_sep" {
                    Kind::MutSep
                } else if two.children.len() >= 3 {
                    Kind::MutWrap
                } else {
                    Kind::Mut
                }
            }
        };
        let boundary = matches!(
            kind,
            Kind::AddEnd | Kind::AddSep | Kind::RemEnd | Kind::RemSep | Kind::MutEnd | Kind::MutSep
        );
        let charge = if boundary {
            0.0
        } else if value.is_infinite() {
            value
        } else {
            value - source.value
        };
        Step {
            id: parent.len,
            kind,
            node1_id: if kind == Kind::Rem {
                None
            } else {
                Some(one.id)
            },
            node2_id: Some(two.id),
            i: self.position.0,
            j: self.position.1,
            value: charge,
            i_swapped: false,
            j_swapped: false,
        }
    }
}

/// Partial results are intentional: an error follows previously created
/// histories in serial order, including their accounting/publication effects.
pub(super) struct Draft<P = Path> {
    pub candidates: [Option<Vec<f64>>; 3],
    pub value: Option<f64>,
    pub paths: Vec<P>,
    pub failure: Option<Failure>,
}

impl<P> Draft<P> {
    pub fn evaluate(
        input: &Input<'_>,
        mut reserve_history: impl FnMut() -> Result<()>,
        mut history: impl FnMut(usize, usize, f64, &Path) -> P,
    ) -> Self {
        let mut draft = Self::empty();
        draft.failure = draft
            .compute::<false>(input, &mut reserve_history, &mut history, None)
            .err();
        draft
    }

    fn empty() -> Self {
        Self {
            candidates: [None, None, None],
            value: None,
            paths: Vec::new(),
            failure: None,
        }
    }

    /// Allocate lane buffers on the coordinator, before borrowing them to jobs.
    #[cfg(all(feature = "parallel", not(target_arch = "wasm32")))]
    pub fn prepared(input: &Input<'_>) -> Result<Self> {
        let mut draft = Self::empty();
        let presets = [
            &input.destination.top,
            &input.destination.left,
            &input.destination.corner,
        ];
        let mut histories = 0_usize;
        for (direction, preset) in presets.iter().enumerate() {
            let count = if preset.is_empty() {
                let count = input.predecessors[direction].paths.len();
                let mut values = Vec::new();
                values
                    .try_reserve_exact(count)
                    .map_err(|_| Failure::Memory)?;
                draft.candidates[direction] = Some(values);
                count
            } else {
                preset.len()
            };
            histories = histories.checked_add(count).ok_or(Failure::Memory)?;
        }
        draft
            .paths
            .try_reserve_exact(histories)
            .map_err(|_| Failure::Memory)?;
        Ok(draft)
    }

    pub fn compute<const CANCELLABLE: bool>(
        &mut self,
        input: &Input<'_>,
        reserve_history: &mut impl FnMut() -> Result<()>,
        history: &mut impl FnMut(usize, usize, f64, &Path) -> P,
        cancelled: Option<&AtomicBool>,
    ) -> Result<()> {
        let presets = [
            &input.destination.top,
            &input.destination.left,
            &input.destination.corner,
        ];
        let one = input.first;
        let two = input.second;
        for (direction, preset) in presets.iter().enumerate() {
            if CANCELLABLE {
                check_cancel(cancelled)?;
            }
            if !preset.is_empty() {
                continue;
            }
            // Keep mutation's possible Index error after TOP and LEFT, before
            // reduction or any proposed history, exactly as in the serial core.
            let mutation = if direction == 2 {
                mutation_cost(one, two)?
            } else {
                0.0
            };
            let source = input.predecessors[direction];
            let mut values = self.candidates[direction].take().unwrap_or_default();
            values
                .try_reserve_exact(source.paths.len())
                .map_err(|_| Failure::Memory)?;
            for (index, path) in source.paths.iter().enumerate() {
                if CANCELLABLE && index % 64 == 0 {
                    check_cancel(cancelled)?;
                }
                let cost = match direction {
                    0 if one.boundary() => Kernel::closing_cost::<CANCELLABLE>(
                        path,
                        one,
                        true,
                        source.value,
                        cancelled,
                    )?,
                    1 if two.boundary() => Kernel::closing_cost::<CANCELLABLE>(
                        path,
                        two,
                        false,
                        source.value,
                        cancelled,
                    )?,
                    0 | 1 => source.value + 1.0,
                    _ if one.boundary() && one.name == two.name => {
                        Kernel::closing_mutation::<CANCELLABLE>(
                            path,
                            one,
                            two,
                            source.value,
                            cancelled,
                        )?
                    }
                    _ => source.value + mutation,
                };
                values.push(cost);
            }
            self.candidates[direction] = Some(values);
        }
        let costs = [
            self.candidates[0].as_deref().unwrap_or(presets[0]),
            self.candidates[1].as_deref().unwrap_or(presets[1]),
            self.candidates[2].as_deref().unwrap_or(presets[2]),
        ];
        let value = costs[0]
            .iter()
            .chain(costs[2])
            .chain(costs[1])
            .copied()
            .reduce(f64::min)
            .ok_or(Failure::EmptyMinimum)?;
        self.value = Some(value);
        // Candidate reduction is TOP, CORNER, LEFT; history order is different.
        for (direction, candidates) in costs.iter().enumerate() {
            let source = input.predecessors[direction];
            for (index, &cost) in candidates.iter().enumerate() {
                if CANCELLABLE && index % 64 == 0 {
                    check_cancel(cancelled)?;
                }
                if cost != value {
                    continue;
                }
                let parent = source.paths.get(index).ok_or(Failure::Index)?;
                reserve_history()?;
                self.paths.try_reserve(1).map_err(|_| Failure::Memory)?;
                self.paths.push(history(direction, index, value, parent));
            }
        }
        Ok(())
    }
}

#[cfg(test)]
mod tests {
    use super::super::{History, HistoryRef};
    use super::*;

    #[test]
    fn long_history_release_does_not_reclaim_a_shared_prefix() {
        let step = Step {
            id: 0,
            kind: Kind::Start,
            node1_id: None,
            node2_id: None,
            i: 0,
            j: 0,
            value: 0.0,
            i_swapped: false,
            j_swapped: false,
        };
        let root = History::new(step.clone(), None, 1);
        let mut path = root.clone();
        let mut shared = None;
        for index in 1..50_000 {
            let mut next = step.clone();
            next.id = index;
            next.kind = Kind::Mut;
            path = History::new(next, Some(path), index + 1);
            if index == 25_000 {
                shared = Some(path.clone());
            }
        }
        let terminal = HistoryRef::downgrade(&path);
        let prefix = HistoryRef::downgrade(shared.as_ref().unwrap());
        drop(path);
        assert!(terminal.upgrade().is_none());
        assert!(prefix.upgrade().is_some());
        drop(shared);
        assert!(prefix.upgrade().is_none());
    }
}
