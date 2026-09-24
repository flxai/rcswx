//! Source-compatible structural edit application.
//!
//! The reference mutates a deepcopy of parent two while recursively applying
//! dependency-enabling edits.  This module performs those topology decisions in
//! Rust and emits the complete object recipe for the Python adapter to replay.

use crate::architecture::{Architecture, Budget, Node};
use crate::edit_plan::{Dependency, Edit, EditPlan};
use crate::error::{Error, Result};
use serde::{Deserialize, Serialize};
use std::collections::{HashMap, HashSet};

/// Replayable Python object-graph changes.  Handles below `n1 + n2` name source
/// parent occurrences; every mapping target and `NewSequence` handle is fresh.
#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
#[serde(tag = "kind", rename_all = "snake_case")]
pub enum Materialization {
    DeepCopy {
        source: usize,
        mapping: Vec<(usize, usize)>,
    },
    NewSequence {
        node: usize,
        template: usize,
        parent: Option<usize>,
        id: String,
    },
    SetChildren {
        node: usize,
        children: Vec<usize>,
    },
    ReplaceChild {
        node: usize,
        index: usize,
        child: usize,
    },
    SetParent {
        node: usize,
        parent: Option<usize>,
    },
    SetId {
        node: usize,
        id: String,
    },
}

#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct ApplicationResult {
    pub architecture: Architecture,
    pub recipe: Vec<Materialization>,
    /// Recipe-local root handle. `architecture` uses an independent compact arena.
    pub root: usize,
}

#[derive(Clone, Copy, Debug)]
enum Parent {
    First,
    Second,
}

#[derive(Clone, Debug)]
struct RuntimeNode {
    node: Node,
    parent: Option<usize>,
}

struct Application<'a> {
    plan: &'a mut EditPlan,
    nodes: Vec<Option<RuntimeNode>>,
    root: usize,
    recipe: Vec<Materialization>,
    next_id: num_bigint::BigInt,
    first_len: usize,
    #[cfg(feature = "trace")]
    recorder: Option<&'a mut crate::trace::Recorder>,
}

impl<'a> Application<'a> {
    fn new(plan: &'a mut EditPlan, budget: &mut Budget<'_>) -> Result<Self> {
        let first_len = plan.prepared.first.nodes.len();
        let second_len = plan.prepared.second.nodes.len();
        if first_len == 0 || second_len == 0 {
            return Err(Error::InvalidInput("empty architecture".into()));
        }
        let next_id = plan
            .prepared
            .next_id
            .parse::<num_bigint::BigInt>()
            .map_err(|_| {
                Error::InvalidInput("next application identity is not a decimal integer".into())
            })?;
        let slots = first_len.checked_add(second_len).ok_or(Error::Memory)?;
        let mut nodes = Vec::new();
        nodes.try_reserve_exact(slots).map_err(|_| Error::Memory)?;
        nodes.resize(slots, None);
        let mut this = Self {
            plan,
            nodes,
            root: 0,
            recipe: Vec::new(),
            next_id,
            first_len,
            #[cfg(feature = "trace")]
            recorder: None,
        };
        let second_root = this.plan.prepared.second.root;
        let copied = this.copy_parent(Parent::Second, second_root, budget)?;
        this.root = copied[second_root];
        Ok(this)
    }

    fn source(&self, parent: Parent) -> &Architecture {
        match parent {
            Parent::First => &self.plan.prepared.first,
            Parent::Second => &self.plan.prepared.second,
        }
    }
    fn source_base(&self, parent: Parent) -> usize {
        match parent {
            Parent::First => 0,
            Parent::Second => self.first_len,
        }
    }
    fn tokens(&self, parent: Parent) -> &[crate::tokens::PreparedToken] {
        match parent {
            Parent::First => &self.plan.prepared.first_tokens,
            Parent::Second => &self.plan.prepared.second_tokens,
        }
    }
    fn occurrence(&self, parent: Parent, index: usize) -> Result<usize> {
        self.tokens(parent)
            .get(index)
            .ok_or(Error::Index)?
            .occurrence
            .ok_or(Error::Index)
    }
    fn source_id(&self, parent: Parent, index: usize) -> Result<&str> {
        let token = self.tokens(parent).get(index).ok_or(Error::Index)?;
        if let Some(occurrence) = token.occurrence {
            return self
                .source(parent)
                .nodes
                .get(occurrence)
                .map(|node| node.id.as_str())
                .ok_or(Error::Index);
        }
        let identity = usize::try_from(token.token.id).map_err(|_| Error::Index)?;
        self.plan
            .prepared
            .identities
            .get(identity)
            .map(String::as_str)
            .ok_or(Error::Index)
    }
    fn source_name(&self, parent: Parent, index: usize) -> Result<&str> {
        Ok(&self
            .tokens(parent)
            .get(index)
            .ok_or(Error::Index)?
            .token
            .name)
    }
    fn source_arity(&self, parent: Parent, index: usize) -> Result<usize> {
        let token = self.tokens(parent).get(index).ok_or(Error::Index)?;
        match token.occurrence {
            Some(occurrence) => self
                .source(parent)
                .nodes
                .get(occurrence)
                .map(|node| node.children.len())
                .ok_or(Error::Index),
            None => Ok(token.token.children.len()),
        }
    }
    fn depth(&self) -> Result<Vec<usize>> {
        let mut stack = vec![0usize];
        let mut next = 1usize;
        let mut depths = Vec::with_capacity(self.plan.operations_unordered.len());
        for &index in &self.plan.operations_unordered {
            let op = self.op(index)?;
            depths.push(*stack.last().ok_or(Error::Index)?);
            let rem_present = match op.node2_id.as_deref() {
                Some(id) => self.present(id)?,
                None => false,
            };
            let add_present = match op.node1_id.as_deref() {
                Some(id) => self.present(id)?,
                None => false,
            };
            let rem = (op.op_type.contains("rem_wrap") || op.op_type.contains("mut_wrap"))
                && rem_present
                && self.depth_arity(Parent::Second, op.node2_id.as_deref().ok_or(Error::Index)?)?
                    == 4;
            let add = !rem
                && (op.op_type.contains("add_wrap") || op.op_type.contains("mut_wrap"))
                && add_present
                && self.depth_arity(Parent::First, op.node1_id.as_deref().ok_or(Error::Index)?)?
                    == 4;
            if rem || add {
                if op.op_type.contains("end") {
                    stack.pop().ok_or(Error::Index)?;
                    stack.pop().ok_or(Error::Index)?;
                } else {
                    stack.push(next);
                    next += 1;
                }
            }
        }
        Ok(depths)
    }
    fn same_depth(&self, operation: &Edit, position: usize, depths: &[usize]) -> Result<bool> {
        let current = self.unordered_position(operation)?;
        Ok(position != current && depths.get(position) == depths.get(current))
    }
    fn active_token(&self, parent: Parent, token: usize) -> Result<bool> {
        self.present(self.source_id(parent, token)?)
    }

    fn nearest_add_anchor(&self, operation: &Edit, depths: &[usize]) -> Result<Option<Anchor>> {
        let current = self.unordered_position(operation)? as isize;
        // The reference takes a separate argmin per parent; parent two wins ties.
        let mut best = None;
        for (position, &index) in self.plan.operations_unordered.iter().enumerate() {
            let candidate = self.op(index)?;
            let candidates = [
                (
                    Parent::Second,
                    candidate.j,
                    candidate.op_type.contains("rem") || candidate.op_type.contains("mut"),
                    candidate.j_swapped,
                ),
                (
                    Parent::First,
                    candidate.i,
                    candidate.op_type.contains("add") || candidate.op_type.contains("mut"),
                    candidate.i_swapped,
                ),
            ];
            for (parent, token, relevant, swapped) in candidates {
                if !relevant
                    || !self.active_token(parent, token)?
                    || !self.same_depth(operation, position, depths)?
                {
                    continue;
                }
                let item = Anchor {
                    parent,
                    token,
                    distance: position as isize - current,
                    after: position as isize - current < 0,
                    swapped,
                };
                if best.as_ref().is_none_or(|best: &Anchor| {
                    (
                        item.distance.unsigned_abs(),
                        matches!(item.parent, Parent::First),
                    ) < (
                        best.distance.unsigned_abs(),
                        matches!(best.parent, Parent::First),
                    )
                }) {
                    best = Some(item);
                }
            }
        }
        Ok(best)
    }

    /// Reference wrapper anchors additionally ignore endpoints/separators on the
    /// wrong orientation and disallow backwards jumps through an interior branch.
    fn nearest_wrapper_anchor(
        &self,
        operation: &Edit,
        closing: usize,
        closing_second: Option<usize>,
        depths: &[usize],
    ) -> Result<Option<Anchor>> {
        let mut best = None;
        // A present closer is parent two's candidate, not a global winner.
        if let Some(second) = closing_second {
            if self.active_token(Parent::Second, second)?
                && self.source_name(Parent::Second, second)?.contains("_end")
            {
                for (position, &index) in self.plan.operations_unordered.iter().enumerate() {
                    let candidate = self.op(index)?;
                    if candidate.j == second {
                        best = Some(Anchor {
                            parent: Parent::Second,
                            token: second,
                            distance: (position as isize - closing as isize).unsigned_abs()
                                as isize,
                            after: true,
                            swapped: candidate.j_swapped,
                        });
                        break;
                    }
                }
            }
        }
        let special_second = best.is_some();
        for (position, &index) in self.plan.operations_unordered.iter().enumerate() {
            let candidate = self.op(index)?;
            for (parent, token, relevant, swapped) in [
                (
                    Parent::Second,
                    candidate.j,
                    candidate.op_type.contains("rem") || candidate.op_type.contains("mut"),
                    candidate.j_swapped,
                ),
                (
                    Parent::First,
                    candidate.i,
                    candidate.op_type.contains("add") || candidate.op_type.contains("mut"),
                    candidate.i_swapped,
                ),
            ] {
                if special_second && matches!(parent, Parent::Second) {
                    continue;
                }
                if !relevant
                    || !self.active_token(parent, token)?
                    || !self.same_depth(operation, position, depths)?
                {
                    continue;
                }
                let name = self.source_name(parent, token)?;
                if name.contains("_end") || name.contains("_sep") != swapped {
                    continue;
                }
                let distance = position as isize - closing as isize;
                if distance == 0 || (distance < 0 && self.source_arity(parent, token)? >= 2) {
                    continue;
                }
                let item = Anchor {
                    parent,
                    token,
                    distance,
                    after: distance < 0,
                    swapped,
                };
                if best.as_ref().is_none_or(|best: &Anchor| {
                    (
                        item.distance.unsigned_abs(),
                        matches!(item.parent, Parent::First),
                    ) < (
                        best.distance.unsigned_abs(),
                        matches!(best.parent, Parent::First),
                    )
                }) {
                    best = Some(item);
                }
            }
        }
        Ok(best)
    }
    fn anchor_node(&self, anchor: Anchor) -> Result<usize> {
        self.first(self.source_id(anchor.parent, anchor.token)?)?
            .ok_or(Error::Index)
    }
    fn token_member(&self, node: usize) -> Result<bool> {
        let id = self.id(node)?;
        Ok([Parent::First, Parent::Second].into_iter().any(|parent| {
            self.tokens(parent)
                .iter()
                .filter_map(|token| token.occurrence)
                .any(|occurrence| {
                    self.source(parent)
                        .nodes
                        .get(occurrence)
                        .is_some_and(|source| source.id == id)
                })
        }))
    }

    fn next_after(&self, node: usize, skip_subtree: bool) -> Result<Option<usize>> {
        let preorder = self.preorder()?;
        let position = preorder
            .iter()
            .position(|&item| item == node)
            .ok_or(Error::Index)?;
        let skipped = if skip_subtree {
            self.subtree_handles(node)?
        } else {
            HashSet::new()
        };
        for candidate in preorder[position + 1..].iter().copied() {
            if !skipped.contains(&candidate) && self.token_member(candidate)? {
                return Ok(Some(candidate));
            }
        }
        Ok(None)
    }

    fn sequence_boundary(&self, mut node: usize, end: Option<&str>) -> Result<usize> {
        while let Some(parent) = self.parent(node)? {
            let contains_end = match end {
                Some(id) => self.subtree_contains(node, id)?,
                None => false,
            };
            if self.name(parent)? != "sequential" || contains_end {
                break;
            }
            node = parent;
        }
        Ok(node)
    }

    /// Reconstruct the same serial segment chosen by `add_wrap`. The initial end
    /// anchor pass selects three-child wrappers. Four-child branch wrappers repeat
    /// selection at their separator before replacing two serialized branches.
    fn wrapper_target(
        &mut self,
        operation: &Edit,
        arity: usize,
        budget: &mut Budget<'_>,
    ) -> Result<usize> {
        let mut starts = Vec::new();
        for (parent, begin) in [
            (Parent::Second, operation.j.saturating_add(1)),
            (Parent::First, operation.i),
        ] {
            for token in begin..self.tokens(parent).len() {
                if self.source_name(parent, token)?.contains("wrap_")
                    || !self.active_token(parent, token)?
                {
                    continue;
                }
                if let Some(node) = self.first(self.source_id(parent, token)?)? {
                    starts.push((
                        self.preorder()?
                            .iter()
                            .position(|&item| item == node)
                            .ok_or(Error::Index)?,
                        node,
                    ));
                }
                break;
            }
        }
        let (_, start) = starts
            .into_iter()
            .min_by_key(|(position, _)| *position)
            .ok_or(Error::Index)?;
        let depths = self.depth()?;
        let (end_position, end_second) = self
            .plan
            .operations_unordered
            .iter()
            .enumerate()
            .find_map(|(position, &index)| {
                let candidate = self.op(index).ok()?;
                (candidate.node1_id == operation.node1_id && candidate.op_type.contains("_end"))
                    .then_some((position, candidate.j))
            })
            .ok_or(Error::Index)?;
        let endpoint =
            self.nearest_wrapper_anchor(operation, end_position, Some(end_second), &depths)?;
        let mut end = match endpoint {
            None => None,
            Some(anchor) if anchor.after => self
                .next_after(self.anchor_node(anchor)?, arity == 3)?
                .map(|node| self.id(node).map(str::to_owned))
                .transpose()?,
            Some(anchor) => Some(self.id(self.anchor_node(anchor)?)?.to_owned()),
        };
        let mut target = self.sequence_boundary(start, end.as_deref())?;
        if self.name(target)? == "sequential" {
            let split_id = self.id(start)?.to_owned();
            match self.split_sequences(target, &split_id, budget) {
                Ok(split) => target = *self.children(split)?.get(1).ok_or(Error::Index)?,
                Err(Error::Index) | Err(Error::Reference(_)) | Err(Error::Unbound(_)) => {}
                Err(error) => return Err(error),
            }
        }
        if self.name(target)? == "sequential" {
            if let Some(id) = end.as_deref() {
                if self.subtree_contains(target, id)? {
                    let split = self.split_sequences(target, id, budget)?;
                    target = *self.children(split)?.first().ok_or(Error::Index)?;
                }
            }
        }
        if arity == 4 {
            let depths = self.depth()?;
            let separator = self
                .plan
                .operations_unordered
                .iter()
                .enumerate()
                .find_map(|(position, &index)| {
                    let candidate = self.op(index).ok()?;
                    (candidate.node1_id == operation.node1_id && candidate.op_type.contains("_sep"))
                        .then_some(position)
                })
                .ok_or(Error::Index)?;
            end = match self.nearest_wrapper_anchor(operation, separator, None, &depths)? {
                None => None,
                Some(anchor) if anchor.after => self
                    .next_after(self.anchor_node(anchor)?, true)?
                    .map(|node| self.id(node).map(str::to_owned))
                    .transpose()?,
                Some(anchor) => Some(self.id(self.anchor_node(anchor)?)?.to_owned()),
            };
            target = self.sequence_boundary(target, end.as_deref())?;
            while let Some(parent) = self.parent(target)? {
                let id = self.id(target)?;
                if self
                    .children(parent)?
                    .iter()
                    .any(|&child| self.id(child).is_ok_and(|child_id| child_id == id))
                {
                    break;
                }
                target = parent;
            }
        }
        if arity == 4 && self.name(target)? == "sequential" {
            if let Some(id) = end.as_deref() {
                target = self.split_sequences(target, id, budget)?;
            }
        }
        Ok(target)
    }

    fn mutation(&mut self, operation: &Edit, budget: &mut Budget<'_>) -> Result<()> {
        let occurrence = self.occurrence(Parent::First, operation.i)?;
        let copied = self.copy_parent(Parent::First, occurrence, budget)?;
        let replacement = copied[occurrence];
        let second_id = self.source_id(Parent::Second, operation.j)?.to_owned();
        let first_id = self.source_id(Parent::First, operation.i)?.to_owned();
        let target = self
            .preorder()?
            .into_iter()
            .find(|&node| {
                self.id(node)
                    .is_ok_and(|id| id == second_id || id == first_id)
            })
            .ok_or(Error::MissingMutationTarget)?;
        let target = self.copy_runtime(target, budget)?;
        let old_children = self.children(target)?;
        self.replace(target, replacement)?;
        let name = self.name(replacement)?.to_owned();
        if name.contains("branching(2)") {
            if old_children.len() < 3 {
                return Err(Error::Index);
            }
            let (one, two) = if operation.i_swapped ^ operation.j_swapped {
                (old_children[2], old_children[1])
            } else {
                (old_children[1], old_children[2])
            };
            if operation.i_swapped ^ operation.j_swapped {
                self.replace_child(replacement, 2, two)?;
                self.replace_child(replacement, 1, one)?;
            } else {
                self.replace_child(replacement, 1, one)?;
                self.replace_child(replacement, 2, two)?;
            }
            self.set_parent(one, Some(replacement))?;
            self.set_parent(two, Some(replacement))?;
        } else if name.contains("branching") || name.contains("routing") {
            let child = *old_children.get(1).ok_or(Error::Index)?;
            self.replace_child(replacement, 1, child)?;
            self.set_parent(child, Some(replacement))?;
        }
        self.root = self.root_of(replacement)?;
        Ok(())
    }

    fn removal(&mut self, operation: &Edit, budget: &mut Budget<'_>) -> Result<()> {
        let id = self.source_id(Parent::Second, operation.j)?.to_owned();
        let Some(target) = self.first(&id)? else {
            return Ok(());
        }; // reference prints and continues
        let name = self.name(target)?.to_owned();
        let children = self.children(target)?;
        if name.contains("branching(2)") {
            if children.len() < 3 {
                return Err(Error::Index);
            }
            let sequence = self.fresh_sequence(target, self.parent(target)?, budget)?;
            self.set_children(
                sequence,
                vec![
                    children[1 + usize::from(operation.j_swapped)],
                    children[2 - usize::from(operation.j_swapped)],
                ],
            )?;
            self.replace(target, sequence)?;
            self.root = self.root_of(sequence)?;
        } else if name.contains("branching") || name.contains("routing") {
            let retained = *children.get(1).ok_or(Error::Index)?;
            self.replace(target, retained)?;
            self.root = self.root_of(retained)?;
        } else {
            let parent = self.parent(target)?.ok_or(Error::Index)?;
            let children = self.children(parent)?;
            let position = children
                .iter()
                .position(|&child| self.id(child).is_ok_and(|child_id| child_id == id))
                .ok_or(Error::Index)?;
            let sibling = if self.name(parent)? == "sequential" {
                *children
                    .get(if position == 0 {
                        children.len() - 1
                    } else {
                        position - 1
                    })
                    .ok_or(Error::Index)?
            } else {
                *children
                    .get(1 + usize::from(position == 1))
                    .ok_or(Error::Index)?
            };
            self.replace(parent, sibling)?;
            self.root = self.root_of(sibling)?;
        }
        Ok(())
    }

    fn add_wrapper(&mut self, operation: &Edit, budget: &mut Budget<'_>) -> Result<()> {
        let arity = self.source_arity(Parent::First, operation.i)?;
        if arity != 3 && arity != 4 {
            return Ok(());
        }
        let occurrence = self.occurrence(Parent::First, operation.i)?;
        let copied = self.copy_parent(Parent::First, occurrence, budget)?;
        let wrapper = copied[occurrence];
        let target = self.wrapper_target(operation, arity, budget)?;
        if arity == 3 {
            self.replace(target, wrapper)?;
            self.replace_child(wrapper, 1, target)?;
            self.set_parent(target, Some(wrapper))?;
        } else {
            let target_children = self.children(target)?;
            if target_children.len() < 2 {
                return Err(Error::Index);
            }
            self.replace(target, wrapper)?;
            self.replace_child(wrapper, 1, target_children[0])?;
            self.replace_child(wrapper, 2, target_children[1])?;
            self.set_parent(target_children[0], Some(wrapper))?;
            self.set_parent(target_children[1], Some(wrapper))?;
        }
        self.root = self.root_of(wrapper)?;
        Ok(())
    }

    fn addition(&mut self, operation: &Edit, budget: &mut Budget<'_>) -> Result<()> {
        let occurrence = self.occurrence(Parent::First, operation.i)?;
        let copied = self.copy_parent(Parent::First, occurrence, budget)?;
        let added = copied[occurrence];
        let second_id = self.source_id(Parent::Second, operation.j)?.to_owned();
        let (before, target) = if second_id == "-1" {
            (true, self.root)
        } else {
            match self.nearest_add_anchor(operation, &self.depth()?)? {
                None => (false, self.root),
                Some(anchor) => {
                    let node = self.anchor_node(anchor)?;
                    let name = self.source_name(anchor.parent, anchor.token)?;
                    if anchor.distance < 0 {
                        if name.contains("_end") || self.children(node)?.len() <= 2 {
                            (false, node)
                        } else if (name.contains("_sep") && !anchor.swapped)
                            || (!name.contains("_sep") && anchor.swapped)
                        {
                            (true, *self.children(node)?.get(2).ok_or(Error::Index)?)
                        } else {
                            (true, *self.children(node)?.get(1).ok_or(Error::Index)?)
                        }
                    } else if name.contains("_end") {
                        (
                            false,
                            *self
                                .children(node)?
                                .iter()
                                .rev()
                                .nth(1)
                                .ok_or(Error::Index)?,
                        )
                    } else if (name.contains("_sep") && !anchor.swapped)
                        || (!name.contains("_sep") && anchor.swapped)
                    {
                        (false, *self.children(node)?.get(1).ok_or(Error::Index)?)
                    } else {
                        (true, node)
                    }
                }
            }
        };
        let sequence = self.fresh_sequence(added, self.parent(target)?, budget)?;
        self.replace(target, sequence)?;
        self.set_children(
            sequence,
            if before {
                vec![added, target]
            } else {
                vec![target, added]
            },
        )?;
        self.root = self.root_of(sequence)?;
        Ok(())
    }

    fn apply_op(
        &mut self,
        operation: &Edit,
        performed: &mut HashSet<usize>,
        budget: &mut Budget<'_>,
    ) -> Result<()> {
        if !performed.insert(operation.id) {
            return Ok(());
        }
        if operation.op_type.contains("mut") {
            self.mutation(operation, budget)
        } else if operation.op_type.contains("rem") {
            self.removal(operation, budget)
        } else if operation.op_type == "add_wrap" {
            self.add_wrapper(operation, budget)
        } else if operation.op_type.contains("add") {
            self.addition(operation, budget)
        } else {
            Ok(())
        }
    }

    fn apply_all(
        &mut self,
        selected: &[usize],
        selected_ids: &HashSet<usize>,
        performed: &mut HashSet<usize>,
        budget: &mut Budget<'_>,
    ) -> Result<()> {
        for &index in selected {
            let reverse = {
                let op = self.op(index)?;
                !op.enabler_ops.is_empty() && (op.i_swapped || op.j_swapped)
            };
            if reverse {
                self.plan
                    .paths
                    .get_mut(self.plan.path_index)
                    .and_then(|path| path.get_mut(index))
                    .ok_or(Error::Index)?
                    .enabler_ops
                    .reverse();
            }
            let operation = self.op(index)?.clone();
            for dependency in &operation.enabler_ops {
                let branch = match dependency {
                    Dependency::One(index) => vec![*index],
                    Dependency::Group(indices) => indices.clone(),
                };
                let enabled = branch
                    .into_iter()
                    .filter(|&item| {
                        self.op(item)
                            .is_ok_and(|candidate| selected_ids.contains(&candidate.id))
                    })
                    .collect::<Vec<_>>();
                let enabled_ids = enabled
                    .iter()
                    .map(|&item| self.op(item).map(|candidate| candidate.id))
                    .collect::<Result<HashSet<_>>>()?;
                self.apply_all(&enabled, &enabled_ids, performed, budget)?;
            }
            #[cfg(feature = "trace")]
            let recipe_start = self.recipe.len();
            #[cfg(feature = "trace")]
            if let Some(recorder) = &mut self.recorder {
                recorder.emit("execute", || serde_json::json!({"path_index":index,"operation_id":operation.id,
                    "operation":operation.op_type,"already_performed":performed.contains(&operation.id)}));
            }
            let outcome = self.apply_op(&operation, performed, budget);
            #[cfg(feature = "trace")]
            if let Some(recorder) = &mut self.recorder {
                recorder.emit("actions", || serde_json::json!({"path_index":index,"recipe_offset":recipe_start,
                    "actions":self.recipe[recipe_start..],"ok":outcome.is_ok(),"error":outcome.as_ref().err().map(ToString::to_string)}));
            }
            outcome?;
        }
        Ok(())
    }

    fn depth_arity(&self, parent: Parent, id: &str) -> Result<usize> {
        // Swapped histories remap matrix coordinates, not the wrapper identity.
        for candidate in 0..self.tokens(parent).len() {
            if self.source_id(parent, candidate)? == id {
                return self.source_arity(parent, candidate);
            }
        }
        Err(Error::Index)
    }

    fn copy_parent(
        &mut self,
        parent: Parent,
        occurrence: usize,
        budget: &mut Budget<'_>,
    ) -> Result<Vec<usize>> {
        let source = self.source(parent).clone();
        if occurrence >= source.nodes.len() {
            return Err(Error::Index);
        }
        let base = self.source_base(parent);
        let first = self.nodes.len();
        let end = first.checked_add(source.nodes.len()).ok_or(Error::Memory)?;
        let allocation = source
            .nodes
            .len()
            .checked_mul(std::mem::size_of::<RuntimeNode>())
            .ok_or(Error::Memory)?;
        let mut mapping = Vec::new();
        mapping
            .try_reserve(source.nodes.len())
            .map_err(|_| Error::Memory)?;
        for index in 0..source.nodes.len() {
            mapping.push((
                base.checked_add(index).ok_or(Error::Memory)?,
                first.checked_add(index).ok_or(Error::Memory)?,
            ));
        }
        budget.checkpoint(source.nodes.len(), source.nodes.len(), allocation)?;
        self.recipe.push(Materialization::DeepCopy {
            source: base.checked_add(occurrence).ok_or(Error::Memory)?,
            mapping,
        });
        let parents = source.parents();
        self.nodes
            .try_reserve(source.nodes.len())
            .map_err(|_| Error::Memory)?;
        for (index, source_node) in source.nodes.iter().enumerate() {
            let mut node = source_node.clone();
            node.children = source_node
                .children
                .iter()
                .map(|child| first.checked_add(*child).ok_or(Error::Memory))
                .collect::<Result<Vec<_>>>()?;
            let parent = parents[index]
                .map(|parent| first.checked_add(parent).ok_or(Error::Memory))
                .transpose()?;
            self.nodes.push(Some(RuntimeNode { node, parent }));
        }
        let mut copied = Vec::new();
        copied
            .try_reserve(source.nodes.len())
            .map_err(|_| Error::Memory)?;
        copied.extend(first..end);
        Ok(copied)
    }
    fn copy_runtime(&mut self, source: usize, budget: &mut Budget<'_>) -> Result<usize> {
        let first = self.nodes.len();
        let mut mapping = HashMap::new();
        let mut originals = Vec::new();
        let mut pending = vec![source];
        while let Some(old) = pending.pop() {
            if mapping.contains_key(&old) {
                continue;
            }
            let new = first.checked_add(originals.len()).ok_or(Error::Memory)?;
            mapping.try_reserve(1).map_err(|_| Error::Memory)?;
            originals.try_reserve(1).map_err(|_| Error::Memory)?;
            mapping.insert(old, new);
            originals.push(old);
            let node = self.get(old)?;
            pending
                .try_reserve(node.node.children.len().saturating_add(1))
                .map_err(|_| Error::Memory)?;
            pending.extend(node.node.children.iter().copied());
            pending.extend(node.parent);
        }
        let count = originals.len();
        budget.checkpoint(
            count,
            count,
            count
                .checked_mul(std::mem::size_of::<RuntimeNode>())
                .ok_or(Error::Memory)?,
        )?;
        self.nodes.try_reserve(count).map_err(|_| Error::Memory)?;
        for &old in &originals {
            let mut copied = self.get(old)?.clone();
            for child in &mut copied.node.children {
                *child = *mapping.get(child).ok_or(Error::Index)?;
            }
            copied.parent = copied
                .parent
                .map(|parent| mapping.get(&parent).copied().ok_or(Error::Index))
                .transpose()?;
            self.nodes.push(Some(copied));
        }
        let copied = *mapping.get(&source).ok_or(Error::Index)?;
        self.recipe.push(Materialization::DeepCopy {
            source,
            mapping: originals
                .into_iter()
                .map(|old| Ok((old, *mapping.get(&old).ok_or(Error::Index)?)))
                .collect::<Result<_>>()?,
        });
        Ok(copied)
    }

    fn root_of(&self, mut node: usize) -> Result<usize> {
        while let Some(parent) = self.parent(node)? {
            node = parent;
        }
        Ok(node)
    }

    fn get(&self, node: usize) -> Result<&RuntimeNode> {
        self.nodes
            .get(node)
            .and_then(Option::as_ref)
            .ok_or(Error::Index)
    }
    fn get_mut(&mut self, node: usize) -> Result<&mut RuntimeNode> {
        self.nodes
            .get_mut(node)
            .and_then(Option::as_mut)
            .ok_or(Error::Index)
    }
    fn id(&self, node: usize) -> Result<&str> {
        Ok(&self.get(node)?.node.id)
    }
    fn name(&self, node: usize) -> Result<&str> {
        Ok(&self.get(node)?.node.name)
    }
    fn parent(&self, node: usize) -> Result<Option<usize>> {
        Ok(self.get(node)?.parent)
    }
    fn children(&self, node: usize) -> Result<Vec<usize>> {
        Ok(self.get(node)?.node.children.clone())
    }
    fn set_parent(&mut self, node: usize, parent: Option<usize>) -> Result<()> {
        self.get_mut(node)?.parent = parent;
        self.recipe
            .push(Materialization::SetParent { node, parent });
        Ok(())
    }
    fn set_children(&mut self, node: usize, children: Vec<usize>) -> Result<()> {
        for &child in &children {
            self.get(child)?;
        }
        self.get_mut(node)?.node.children = children.clone();
        self.recipe.push(Materialization::SetChildren {
            node,
            children: children.clone(),
        });
        for child in children {
            self.set_parent(child, Some(node))?;
        }
        Ok(())
    }
    fn child_position(&self, parent: usize, child: usize) -> Result<usize> {
        let id = self.id(child)?;
        self.get(parent)?
            .node
            .children
            .iter()
            .position(|&child| self.id(child).is_ok_and(|child_id| child_id == id))
            .ok_or_else(|| Error::Reference("node is not in its parent's children".into()))
    }

    fn replace_child(&mut self, parent: usize, index: usize, child: usize) -> Result<()> {
        self.get(child)?;
        // Preserve child-list aliases and untouched siblings' parent links.
        *self
            .get_mut(parent)?
            .node
            .children
            .get_mut(index)
            .ok_or(Error::Index)? = child;
        self.recipe.push(Materialization::ReplaceChild {
            node: parent,
            index,
            child,
        });
        Ok(())
    }
    fn replace(&mut self, old: usize, new: usize) -> Result<()> {
        let parent = self.parent(old)?;
        if let Some(parent) = parent {
            let index = self.child_position(parent, old)?;
            self.replace_child(parent, index, new)?;
        } else {
            self.root = new;
        }
        self.set_parent(new, parent)
    }
    fn preorder(&self) -> Result<Vec<usize>> {
        let mut result = Vec::new();
        let mut pending = vec![self.root];
        while let Some(node) = pending.pop() {
            let runtime = self.get(node)?;
            result.push(node);
            pending.extend(runtime.node.children.iter().rev().copied());
        }
        Ok(result)
    }
    fn present(&self, id: &str) -> Result<bool> {
        Ok(self
            .preorder()?
            .into_iter()
            .any(|node| self.id(node) == Ok(id)))
    }
    fn first(&self, id: &str) -> Result<Option<usize>> {
        for node in self.preorder()? {
            if self.id(node)? == id {
                return Ok(Some(node));
            }
        }
        Ok(None)
    }
    fn subtree_contains(&self, root: usize, id: &str) -> Result<bool> {
        let mut pending = vec![root];
        while let Some(node) = pending.pop() {
            if self.id(node)? == id {
                return Ok(true);
            }
            pending.extend(self.children(node)?);
        }
        Ok(false)
    }
    fn subtree_handles(&self, root: usize) -> Result<HashSet<usize>> {
        let mut found = HashSet::new();
        let mut pending = vec![root];
        while let Some(node) = pending.pop() {
            if found.insert(node) {
                pending.extend(self.children(node)?);
            }
        }
        Ok(found)
    }
    fn fresh_sequence(
        &mut self,
        template: usize,
        parent: Option<usize>,
        budget: &mut Budget<'_>,
    ) -> Result<usize> {
        let mut node = self.get(template)?.node.clone();
        let id = self.next_id.to_string();
        self.next_id += 1;
        self.plan.prepared.next_id = self.next_id.to_string();
        node.id = id.clone();
        node.name = "sequential".into();
        node.children.clear();
        let handle = self.nodes.len();
        budget.checkpoint(1, 1, std::mem::size_of::<RuntimeNode>())?;
        self.nodes.push(Some(RuntimeNode { node, parent }));
        self.recipe.push(Materialization::NewSequence {
            node: handle,
            template,
            parent,
            id,
        });
        Ok(handle)
    }
    fn flatten_sequences(&self, node: usize, values: &mut Vec<usize>) -> Result<()> {
        if self.name(node)? == "sequential" {
            for child in self.children(node)? {
                self.flatten_sequences(child, values)?;
            }
        } else {
            values.push(node);
        }
        Ok(())
    }
    fn resequence(
        &mut self,
        values: &mut Vec<usize>,
        template: usize,
        budget: &mut Budget<'_>,
    ) -> Result<()> {
        while values.len() > 1 {
            let right = values.pop().ok_or(Error::Index)?;
            let left = values.pop().ok_or(Error::Index)?;
            let sequence = self.fresh_sequence(template, self.parent(template)?, budget)?;
            self.set_children(sequence, vec![left, right])?;
            values.push(sequence);
        }
        Ok(())
    }
    fn split_sequences(
        &mut self,
        original: usize,
        split_id: &str,
        budget: &mut Budget<'_>,
    ) -> Result<usize> {
        if !self.subtree_contains(original, split_id)? {
            return Ok(original);
        }
        let attachment = self
            .parent(original)?
            .map(|parent| {
                self.child_position(parent, original)
                    .map(|index| (parent, index))
            })
            .transpose()?;
        let mut flattened = Vec::new();
        for child in self.children(original)? {
            self.flatten_sequences(child, &mut flattened)?;
        }
        let split = flattened
            .iter()
            .position(|&node| self.id(node) == Ok(split_id))
            .or_else(|| flattened.len().checked_sub(1))
            .ok_or(Error::Index)?;
        let mut left = flattened[..split].to_vec();
        let mut right = flattened[split..].to_vec();
        let template = *right.first().or_else(|| left.first()).ok_or(Error::Index)?;
        self.resequence(&mut left, template, budget)?;
        self.resequence(&mut right, template, budget)?;
        let sequence = self.fresh_sequence(template, self.parent(template)?, budget)?;
        // The reference consumes IDs and reparents the right suffix before an
        // empty prefix raises. Callers may catch that failure and keep editing.
        let left_root = *left.first().ok_or(Error::Index)?;
        let right_root = *right.first().ok_or(Error::Index)?;
        self.set_children(sequence, vec![left_root, right_root])?;
        if let Some((parent, index)) = attachment {
            self.replace_child(parent, index, sequence)?;
        }
        self.set_parent(sequence, attachment.map(|(parent, _)| parent))?;
        Ok(sequence)
    }
    fn op(&self, index: usize) -> Result<&Edit> {
        self.plan
            .paths
            .get(self.plan.path_index)
            .and_then(|path| path.get(index))
            .ok_or(Error::Index)
    }
    fn unordered_position(&self, operation: &Edit) -> Result<usize> {
        self.plan
            .operations_unordered
            .iter()
            .position(|&index| {
                self.op(index)
                    .is_ok_and(|candidate| candidate.id == operation.id)
            })
            .ok_or(Error::Index)
    }

    fn result(self) -> Result<ApplicationResult> {
        self.plan.prepared.next_id = self.next_id.to_string();
        let preorder = self.preorder()?;
        let compact = preorder
            .iter()
            .enumerate()
            .map(|(index, &handle)| (handle, index))
            .collect::<HashMap<_, _>>();
        let nodes = preorder
            .iter()
            .map(|&handle| {
                let mut node = self.get(handle)?.node.clone();
                node.children = node
                    .children
                    .iter()
                    .map(|child| compact.get(child).copied().ok_or(Error::Index))
                    .collect::<Result<Vec<_>>>()?;
                Ok(node)
            })
            .collect::<Result<Vec<_>>>()?;
        let parent = &self.plan.prepared.second;
        Ok(ApplicationResult {
            architecture: Architecture {
                schema: parent.schema,
                grammar: parent.grammar.clone(),
                grammar_version: parent.grammar_version.clone(),
                root: 0,
                nodes,
                input_spec: parent.input_spec.clone(),
            },
            recipe: self.recipe,
            root: self.root,
        })
    }

    fn materialization(self) -> Result<(Vec<Materialization>, usize)> {
        self.plan.prepared.next_id = self.next_id.to_string();
        let parent = &self.plan.prepared.second;
        crate::architecture::validate_tree(
            parent.schema,
            &parent.grammar,
            &parent.grammar_version,
            self.root,
            None,
            |index| Ok(&self.get(index)?.node),
        )?;
        Ok((self.recipe, self.root))
    }
}

#[derive(Clone, Copy, Debug)]
struct Anchor {
    parent: Parent,
    token: usize,
    distance: isize,
    after: bool,
    swapped: bool,
}

/// Apply a source-order, plan-bound explicit selection.  A validated empty
/// selection is intentionally a deepcopy of parent two, not the raw-crossover
/// zero-operation shortcut that aliases parent one.
pub fn apply(
    plan: &mut EditPlan,
    selected: &[usize],
    validate: bool,
    budget: &mut Budget<'_>,
) -> Result<ApplicationResult> {
    apply_selection(
        plan,
        selected,
        validate,
        budget,
        #[cfg(feature = "trace")]
        None,
    )?
    .result()
}

/// Apply edits for a legacy host without constructing a portable output arena.
/// The reachable output is validated; returned handles belong to the recipe's
/// object graph rather than a compact architecture node table.
pub fn apply_materialization(
    plan: &mut EditPlan,
    selected: &[usize],
    validate: bool,
    budget: &mut Budget<'_>,
) -> Result<(Vec<Materialization>, usize)> {
    apply_selection(
        plan,
        selected,
        validate,
        budget,
        #[cfg(feature = "trace")]
        None,
    )?
    .materialization()
}

fn apply_selection<'a>(
    plan: &'a mut EditPlan,
    selected: &[usize],
    validate: bool,
    budget: &mut Budget<'_>,
    #[cfg(feature = "trace")] recorder: Option<&'a mut crate::trace::Recorder>,
) -> Result<Application<'a>> {
    if validate {
        plan.validate_selection(selected)?;
    }
    let ids = selected
        .iter()
        .map(|&index| {
            plan.paths
                .get(plan.path_index)
                .and_then(|path| path.get(index))
                .map(|op| op.id)
                .ok_or(Error::Index)
        })
        .collect::<Result<HashSet<_>>>()?;
    let mut application = Application::new(plan, budget)?;
    #[cfg(feature = "trace")]
    {
        application.recorder = recorder;
        if let Some(recorder) = &mut application.recorder {
            recorder.emit(
                "requested",
                || serde_json::json!({"selected_indices":selected}),
            );
            recorder.emit("actions", || serde_json::json!({"path_index":null,"recipe_offset":0,"actions":application.recipe,"ok":true}));
        }
    }
    let mut performed = HashSet::new();
    application.apply_all(selected, &ids, &mut performed, budget)?;
    Ok(application)
}

/// Observe actual materialization handles rather than guessing origins from IDs.
#[cfg(feature = "trace")]
pub fn apply_traced(
    plan: &mut EditPlan,
    selected: &[usize],
    validate: bool,
    budget: &mut Budget<'_>,
    recorder: &mut crate::trace::Recorder,
) -> Result<crate::trace::ObservedApplication> {
    use crate::trace::{ObservedApplication, OccurrenceOrigin, SourceOccurrence};
    use std::collections::BTreeSet;
    let outcome = (|| {
        let application = apply_selection(plan, selected, validate, budget, Some(recorder))?;
        let handles = application.preorder()?;
        let first_len = application.plan.prepared.first.nodes.len();
        let second_len = application.plan.prepared.second.nodes.len();
        let mut origins = (0..first_len + second_len)
            .map(|handle| {
                (
                    handle,
                    (
                        BTreeSet::from([SourceOccurrence {
                            parent: if handle < first_len { 1 } else { 2 },
                            occurrence_index: if handle < first_len {
                                handle
                            } else {
                                handle - first_len
                            },
                        }]),
                        false,
                    ),
                )
            })
            .collect::<HashMap<_, _>>();
        for action in &application.recipe {
            match action {
                Materialization::DeepCopy { mapping, .. } => {
                    for &(source, target) in mapping {
                        let origin = origins.get(&source).ok_or(Error::Index)?.clone();
                        origins.insert(target, origin);
                    }
                }
                Materialization::NewSequence { node, template, .. } => {
                    let sources = origins.get(template).ok_or(Error::Index)?.0.clone();
                    origins.insert(*node, (sources, true));
                }
                _ => {}
            }
        }
        for &handle in handles.iter().rev() {
            if origins
                .get(&handle)
                .is_some_and(|(_, synthetic)| *synthetic)
            {
                let children = application
                    .children(handle)?
                    .iter()
                    .map(|child| {
                        origins
                            .get(child)
                            .map(|(sources, _)| sources.clone())
                            .ok_or(Error::Index)
                    })
                    .collect::<Result<Vec<_>>>()?;
                let origin = origins.get_mut(&handle).ok_or(Error::Index)?;
                for sources in children {
                    origin.0.extend(sources);
                }
            }
        }
        let origins = handles
            .iter()
            .enumerate()
            .map(|(occurrence_index, handle)| {
                let (sources, synthesized) = origins.get(handle).ok_or(Error::Index)?;
                Ok(OccurrenceOrigin {
                    occurrence_index,
                    sources: sources.iter().cloned().collect(),
                    synthesized: *synthesized,
                })
            })
            .collect::<Result<Vec<_>>>()?;
        let result = application.result()?;
        Ok((ObservedApplication { result, origins }, handles))
    })();
    if let Ok((result, handles)) = &outcome {
        recorder.emit("materialized", || {
            serde_json::json!({"handles":handles,"origins":result.origins,
            "architecture_json":result.result.architecture.to_json().ok()})
        });
    }
    recorder.emit("outcome", || serde_json::json!({"ok":outcome.is_ok(),"error":outcome.as_ref().err().map(ToString::to_string)}));
    outcome.map(|(result, _)| result)
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::architecture::{Limits, SCHEMA_VERSION};
    use crate::tokens::prepare;
    use std::collections::BTreeMap;

    fn node(id: &str, name: &str, children: &[usize]) -> Node {
        Node {
            id: id.into(),
            name: name.into(),
            children: children.to_vec(),
            parameters: None,
            provenance: None,
        }
    }
    fn architecture(nodes: Vec<Node>) -> Architecture {
        Architecture {
            schema: SCHEMA_VERSION,
            grammar: "einspace".into(),
            grammar_version: "1".into(),
            root: 0,
            nodes,
            input_spec: None,
        }
    }
    fn edit(
        id: usize,
        kind: &str,
        node1_id: Option<String>,
        node2_id: Option<String>,
        i: usize,
        j: usize,
    ) -> Edit {
        Edit {
            id,
            op_type: kind.into(),
            node1_id,
            node2_id,
            i,
            j,
            ii: None,
            jj: None,
            value: 1.0,
            i_swapped: false,
            j_swapped: false,
            enabler_ops: Vec::new(),
            disabler_ops: Vec::new(),
        }
    }
    fn plan(prepared: crate::tokens::PreparedPair, edits: Vec<Edit>) -> EditPlan {
        let indexes = (0..edits.len()).collect::<Vec<_>>();
        EditPlan {
            id: "application-fixture".into(),
            prepared,
            distance: 1.0,
            paths: vec![edits],
            path_index: 0,
            operations: indexes.clone(),
            operations_unordered: indexes.clone(),
            nontrivial: indexes,
            stats: BTreeMap::new(),
        }
    }
    fn apply_fixture(plan: &mut EditPlan, selected: &[usize]) -> ApplicationResult {
        let mut check = || Ok(());
        let mut budget = Budget {
            limits: Limits::default(),
            work: 0,
            output: 0,
            allocation_bytes: 0,
            check: &mut check,
        };
        apply(plan, selected, false, &mut budget).unwrap()
    }

    #[test]
    fn empty_selection_is_a_fresh_second_parent_copy() {
        let first = architecture(vec![node("1", "identity", &[])]);
        let second = architecture(vec![
            node("9", "sequential", &[1, 2]),
            node("10", "identity", &[]),
            node("11", "relu", &[]),
        ]);
        let mut plan = plan(prepare(first, second).unwrap(), Vec::new());
        let child = apply_fixture(&mut plan, &[]);
        assert_eq!(
            child
                .architecture
                .nodes
                .iter()
                .map(|node| node.name.as_str())
                .collect::<Vec<_>>(),
            ["sequential", "identity", "relu"]
        );
    }

    #[test]
    fn materialization_rejects_shared_output_occurrences() {
        let first = architecture(vec![node("1", "identity", &[])]);
        let second = architecture(vec![
            node("9", "sequential", &[1, 2]),
            node("10", "identity", &[]),
            node("11", "relu", &[]),
        ]);
        let mut plan = plan(prepare(first, second).unwrap(), Vec::new());
        let mut check = || Ok(());
        let mut budget = Budget {
            limits: Limits::default(),
            work: 0,
            output: 0,
            allocation_bytes: 0,
            check: &mut check,
        };
        let mut application = apply_selection(
            &mut plan,
            &[],
            true,
            &mut budget,
            #[cfg(feature = "trace")]
            None,
        )
        .unwrap();
        let children = &mut application.nodes[application.root]
            .as_mut()
            .unwrap()
            .node
            .children;
        children.push(children[0]);
        assert!(matches!(
            application.materialization(),
            Err(Error::InvalidInput(_))
        ));
    }

    #[test]
    fn mutation_preserves_branch_swap_orientation() {
        let first = architecture(vec![
            node("1", "branching(2)", &[1, 2, 3, 4]),
            node("2", "clone(2)", &[]),
            node("3", "first", &[]),
            node("4", "second", &[]),
            node("5", "add(2)", &[]),
        ]);
        let second = architecture(vec![
            node("8", "branching(2)", &[1, 2, 3, 4]),
            node("9", "clone(2)", &[]),
            node("10", "left", &[]),
            node("11", "right", &[]),
            node("12", "add(2)", &[]),
        ]);
        let prepared = prepare(first, second).unwrap();
        let mut mutation = edit(
            1,
            "mut",
            Some(prepared.first.nodes[0].id.clone()),
            Some(prepared.second.nodes[0].id.clone()),
            1,
            1,
        );
        mutation.i_swapped = true;
        let mut plan = plan(prepared, vec![mutation]);
        let child = apply_fixture(&mut plan, &[0]);
        assert_eq!(child.architecture.nodes[0].name, "branching(2)");
        let names = &child.architecture.nodes[0].children;
        assert_eq!(child.architecture.nodes[names[1]].name, "right");
        assert_eq!(child.architecture.nodes[names[2]].name, "left");
    }

    #[test]
    fn removal_promotes_the_serial_sibling() {
        let first = architecture(vec![node("1", "identity", &[])]);
        let second = architecture(vec![
            node("2", "sequential", &[1, 2]),
            node("3", "identity", &[]),
            node("4", "relu", &[]),
        ]);
        let prepared = prepare(first, second).unwrap();
        let removal = edit(
            1,
            "rem",
            Some(prepared.first.nodes[0].id.clone()),
            Some(prepared.second.nodes[2].id.clone()),
            1,
            2,
        );
        let mut plan = plan(prepared, vec![removal]);
        let child = apply_fixture(&mut plan, &[0]);
        assert_eq!(child.architecture.nodes.len(), 1);
        assert_eq!(child.architecture.nodes[0].name, "identity");
    }

    #[test]
    fn removing_first_serial_child_promotes_last_sibling() {
        let first = architecture(vec![node("1", "identity", &[])]);
        let second = architecture(vec![
            node("2", "sequential", &[1, 2]),
            node("3", "identity", &[]),
            node("4", "relu", &[]),
        ]);
        let prepared = prepare(first, second).unwrap();
        let removal = edit(
            1,
            "rem",
            Some(prepared.first.nodes[0].id.clone()),
            Some(prepared.second.nodes[1].id.clone()),
            1,
            1,
        );
        let mut plan = plan(prepared, vec![removal]);
        let child = apply_fixture(&mut plan, &[0]);
        assert_eq!(child.architecture.nodes.len(), 1);
        assert_eq!(child.architecture.nodes[0].name, "relu");
    }

    #[test]
    fn addition_rebuilds_a_binary_sequence() {
        let first = architecture(vec![
            node("1", "sequential", &[1, 2]),
            node("2", "identity", &[]),
            node("3", "relu", &[]),
        ]);
        let second = architecture(vec![node("8", "identity", &[])]);
        let prepared = prepare(first, second).unwrap();
        let addition = edit(
            1,
            "add",
            Some(prepared.first.nodes[2].id.clone()),
            Some(prepared.second.nodes[0].id.clone()),
            2,
            1,
        );
        let mut plan = plan(prepared, vec![addition]);
        let child = apply_fixture(&mut plan, &[0]);
        assert_eq!(child.architecture.nodes[0].name, "sequential");
        assert_eq!(
            child.architecture.nodes[child.architecture.nodes[0].children[0]].name,
            "identity"
        );
        assert_eq!(
            child.architecture.nodes[child.architecture.nodes[0].children[1]].name,
            "relu"
        );
    }

    #[test]
    fn remapped_nonwrapper_does_not_consume_insertion_identities() {
        let first = architecture(vec![
            node("1", "routing", &[1, 2, 4]),
            node("2", "identity", &[]),
            node("3", "computation", &[3]),
            node("4", "relu", &[]),
            node("5", "identity", &[]),
        ]);
        let second = architecture(vec![
            node("8", "sequential", &[1, 2]),
            node("9", "identity", &[]),
            node("10", "sequential", &[3, 4]),
            node("11", "relu", &[]),
            node("12", "norm", &[]),
        ]);
        let prepared = prepare(first, second).unwrap();
        let next_id = prepared.next_id.clone();
        let wrapper_id = prepared.first.nodes[0].id.clone();
        // Historical reordering can remap a wrapper edit to a computation.
        // Its no-op must not resequence parent two before a later insertion.
        let wrapper = edit(
            1,
            "add_wrap",
            Some(wrapper_id.clone()),
            Some("-1".into()),
            2,
            0,
        );
        let end = edit(2, "add_wrap_end", Some(wrapper_id), Some("-1".into()), 3, 0);
        let insertion = edit(
            3,
            "add_module",
            Some(prepared.first.nodes[2].id.clone()),
            Some("-1".into()),
            2,
            0,
        );
        let mut plan = plan(prepared, vec![wrapper, end, insertion]);
        let child = apply_fixture(&mut plan, &[0, 2]);
        assert_eq!(
            child.architecture.nodes[child.architecture.root].id,
            next_id
        );
    }

    #[test]
    fn missing_branch_separator_preserves_reference_index_error() {
        let first = architecture(vec![
            node("1", "branching(2)", &[1, 2, 3, 4]),
            node("2", "clone(2)", &[]),
            node("3", "first", &[]),
            node("4", "second", &[]),
            node("5", "add(2)", &[]),
        ]);
        let second = architecture(vec![
            node("9", "sequential", &[1, 2]),
            node("10", "identity", &[]),
            node("11", "relu", &[]),
        ]);
        let prepared = prepare(first, second).unwrap();
        let root = prepared.first.nodes[0].id.clone();
        let target = prepared.second.nodes[1].id.clone();
        let wrapper = edit(
            1,
            "add_wrap",
            Some(root.clone()),
            Some(target.clone()),
            1,
            1,
        );
        let end = edit(2, "add_wrap_end", Some(root), Some(target), 5, 1);
        let mut plan = plan(prepared, vec![wrapper, end]);
        // The frozen Python application also fails: an arity-four wrapper
        // requires a separator operation, which this plan deliberately omits.
        let mut check = || Ok(());
        let mut budget = Budget {
            limits: Limits::default(),
            work: 0,
            output: 0,
            allocation_bytes: 0,
            check: &mut check,
        };
        assert!(matches!(
            apply(&mut plan, &[0], false, &mut budget),
            Err(Error::Index)
        ));
    }
}
