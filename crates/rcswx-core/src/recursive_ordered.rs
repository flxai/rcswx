//! Typed ordered policy for the retained-history RCSWX recurrence.
//!
//! Unlike the legacy grammar, every semantic occurrence is an owned wrapper.
//! Closing legality depends on active opening correspondences, so pruning is
//! local to a closure state, not a cell-wide value. Histories still use the same
//! persistent native operation paths as recursive einspace RCSWX.
//! Open correspondences additionally require directly corresponding parents;
//! unmatched owners therefore consume their entire subtrees without promotion.

use super::{History, Kind, Path, Step, cell_eval};
use crate::error::{Error, Result};
use crate::typed::Prepared;
use std::collections::{BTreeMap, HashMap};

#[derive(Clone, Copy, Debug)]
pub(crate) enum Event {
    Start,
    Open(usize),
    Close(usize),
}

pub(super) struct Lane {
    pub cost: usize,
    pub history: Path,
}

pub(super) type OrderedCell = BTreeMap<usize, Lane>;
type Transition = (usize, usize, Kind, Option<usize>, Option<usize>);

#[derive(Clone, Copy)]
pub(super) struct Mutation {
    cost: usize,
    parents: (Option<usize>, Option<usize>),
}

#[derive(Clone, Copy)]
struct Pair {
    previous: usize,
    first: usize,
    second: usize,
}

/// Intern active matched-wrapper stacks. Matches require their immediate parents
/// to be the current pair, and each pair closes with its exact opposite owner.
/// At a fixed cell, consumed token positions plus this stack determine all future
/// legality; a cheaper history in the same state dominates. Tiny mapping-oracle
/// tests independently exercise this retention rule, not global semantic claims.
pub(super) struct States {
    pairs: Vec<Option<Pair>>,
    lookup: HashMap<(usize, usize, usize), usize>,
}

impl States {
    fn new() -> Self {
        Self {
            pairs: vec![None],
            lookup: HashMap::new(),
        }
    }

    fn push(&mut self, previous: usize, first: usize, second: usize) -> Result<usize> {
        let key = (previous, first, second);
        if let Some(&state) = self.lookup.get(&key) {
            return Ok(state);
        }
        self.pairs.try_reserve(1).map_err(|_| Error::Memory)?;
        self.lookup.try_reserve(1).map_err(|_| Error::Memory)?;
        let state = self.pairs.len();
        self.pairs.push(Some(Pair {
            previous,
            first,
            second,
        }));
        self.lookup.insert(key, state);
        Ok(state)
    }

    pub fn transition(
        &mut self,
        state: usize,
        direction: usize,
        first: Event,
        second: Event,
        mutation: Option<Mutation>,
    ) -> Result<Option<Transition>> {
        let pair = self.pairs[state];
        let parents = pair.map_or((None, None), |pair| (Some(pair.first), Some(pair.second)));
        let transition = match (direction, first, second) {
            (0, Event::Open(one), _) => Some((state, 1, Kind::AddWrap, Some(one), None)),
            (1, _, Event::Open(two)) => Some((state, 1, Kind::RemWrap, None, Some(two))),
            (0, Event::Close(one), _) if pair.is_none_or(|pair| pair.first != one) => {
                Some((state, 0, Kind::AddEnd, Some(one), None))
            }
            (1, _, Event::Close(two)) if pair.is_none_or(|pair| pair.second != two) => {
                Some((state, 0, Kind::RemEnd, None, Some(two)))
            }
            (2, Event::Open(one), Event::Open(two)) => match mutation {
                Some(mutation) if mutation.parents == parents => Some((
                    self.push(state, one, two)?,
                    mutation.cost,
                    Kind::MutWrap,
                    Some(one),
                    Some(two),
                )),
                _ => None,
            },
            (2, Event::Close(one), Event::Close(two)) => pair
                .filter(|pair| pair.first == one && pair.second == two)
                .map(|pair| (pair.previous, 0, Kind::MutEnd, Some(one), Some(two))),
            _ => None,
        };
        Ok(transition)
    }
}

pub(crate) struct OrderedAlignment {
    pub distance: usize,
    pub path: Vec<Step>,
}

pub(crate) fn align(
    first: &Prepared<'_>,
    second: &Prepared<'_>,
    check: &mut dyn FnMut() -> Result<()>,
) -> Result<OrderedAlignment> {
    check()?;
    let mut states = States::new();
    let mut previous = Vec::new();
    previous
        .try_reserve_exact(second.events.len())
        .map_err(|_| Error::Memory)?;
    for _ in &second.events {
        previous.push(OrderedCell::new());
    }
    let mut current = Vec::new();
    current
        .try_reserve_exact(second.events.len())
        .map_err(|_| Error::Memory)?;
    for (i, &one) in first.events.iter().enumerate() {
        current.clear();
        for (j, &two) in second.events.iter().enumerate() {
            // Match the recursive kernel's TOP, CORNER, LEFT tie preference.
            // A row checkpoint also provides Python signal cancellation.
            if j == 0 {
                check()?;
            }
            if i == 0 && j == 0 {
                current.push(BTreeMap::from([(
                    0,
                    Lane {
                        cost: 0,
                        history: History::new(
                            Step {
                                id: 0,
                                kind: Kind::Start,
                                node1_id: None,
                                node2_id: None,
                                i: 0,
                                j: 0,
                                source_i: 0,
                                source_j: 0,
                                value: 0.0,
                                i_swapped: false,
                                j_swapped: false,
                            },
                            None,
                            1,
                        ),
                    },
                )]));
                continue;
            }
            let top = (i > 0).then(|| &previous[j]);
            let left = (j > 0).then(|| &current[j - 1]);
            let corner = (i > 0 && j > 0).then(|| &previous[j - 1]);
            let mutation = match (one, two) {
                (Event::Open(one), Event::Open(two)) => {
                    first.mutation_cost(one, second, two).map(|cost| Mutation {
                        cost,
                        parents: (first.parents[one], second.parents[two]),
                    })
                }
                _ => None,
            };
            current.push(cell_eval::ordered_cell(
                [top, left, corner],
                &mut states,
                (one, two),
                (i, j),
                mutation,
            )?);
        }
        std::mem::swap(&mut previous, &mut current);
    }
    let lane = previous
        .last_mut()
        .and_then(|cell| cell.remove(&0))
        .ok_or_else(|| Error::Execution("typed recursive alignment has no closed path".into()))?;
    let mut path = Vec::new();
    path.try_reserve_exact(lane.history.len - 1)
        .map_err(|_| Error::Memory)?;
    let mut current = Some(lane.history.as_ref());
    while let Some(history) = current {
        if matches!(
            history.step.kind,
            Kind::AddWrap | Kind::RemWrap | Kind::MutWrap
        ) {
            path.push(history.snapshot());
        }
        current = history.previous.as_deref();
    }
    path.reverse();
    check()?;
    Ok(OrderedAlignment {
        distance: lane.cost,
        path,
    })
}
