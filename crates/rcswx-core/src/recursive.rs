//! Source-compatible recursive RCSWX, including reference failure paths.
//!
//! Histories are persistent lists, not merged dynamic-programming states. Appending
//! shares immutable prefixes and the original mutable operation identities, just
//! as Python's `path + [operation]` shares the preceding operation objects.
//! Derived from einsearch 3f44ddf086bee0c213404e240ee0adf99e3e1501.
//! See LICENSE.einsearch for the original MIT attribution.

use std::cell::RefCell;
use std::collections::HashMap;
use std::rc::Rc;

#[derive(Debug)]
pub enum Failure {
    Index,
    EmptyMinimum,
    Unbound(&'static str),
    Memory,
    Callback,
}
pub type Result<T> = std::result::Result<T, Failure>;

#[derive(Clone, Debug)]
pub struct Token {
    pub id: i64,
    pub name: String,
    pub children: Vec<String>,
    pub parent_arity: usize,
}
impl Token {
    fn boundary(&self) -> bool {
        matches!(self.name.as_str(), "wrap_end" | "wrap_sep")
    }
    fn decoy(&self) -> bool {
        self.boundary() || self.name == "start_node"
    }
    fn source_eq(&self, other: &Self) -> bool {
        self.id == other.id && (!self.decoy() || other.decoy())
    }
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Kind {
    Start,
    AddModule,
    AddWrap,
    AddEnd,
    AddSep,
    Rem,
    RemWrap,
    RemEnd,
    RemSep,
    Mut,
    MutWrap,
    MutEnd,
    MutSep,
}
impl Kind {
    pub fn name(self) -> &'static str {
        match self {
            Self::Start => "start",
            Self::AddModule => "add_module",
            Self::AddWrap => "add_wrap",
            Self::AddEnd => "add_wrap_end",
            Self::AddSep => "add_wrap_sep",
            Self::Rem => "rem",
            Self::RemWrap => "rem_wrap",
            Self::RemEnd => "rem_wrap_end",
            Self::RemSep => "rem_wrap_sep",
            Self::Mut => "mut",
            Self::MutWrap => "mut_wrap",
            Self::MutEnd => "mut_wrap_end",
            Self::MutSep => "mut_wrap_sep",
        }
    }
    fn add(self) -> bool {
        matches!(
            self,
            Self::AddModule | Self::AddWrap | Self::AddEnd | Self::AddSep
        )
    }
    fn rem(self) -> bool {
        matches!(
            self,
            Self::Rem | Self::RemWrap | Self::RemEnd | Self::RemSep
        )
    }
    fn mutation(self) -> bool {
        matches!(
            self,
            Self::Mut | Self::MutWrap | Self::MutEnd | Self::MutSep
        )
    }
}

#[derive(Clone, Debug)]
pub struct Step {
    pub id: usize,
    pub kind: Kind,
    pub node1_id: Option<i64>,
    pub node2_id: Option<i64>,
    pub i: usize,
    pub j: usize,
    pub value: f64,
    pub i_swapped: bool,
    pub j_swapped: bool,
}
struct History {
    step: RefCell<Step>,
    previous: Option<Rc<History>>,
    len: usize,
}
type Path = Rc<History>;

#[derive(Clone)]
struct Cell {
    top: Vec<f64>,
    left: Vec<f64>,
    corner: Vec<f64>,
    paths: Vec<Path>,
    value: f64,
}
impl Cell {
    fn empty() -> Self {
        Self {
            top: vec![],
            left: vec![],
            corner: vec![],
            paths: vec![],
            value: f64::NAN,
        }
    }
    fn clean(&mut self, completely: bool) -> Result<()> {
        if completely {
            self.top.clear();
            self.left.clear();
            self.corner.clear();
            self.paths.clear();
        } else {
            // The reference indexes before assigning, rather than accepting an empty list.
            let top = *self.top.first().ok_or(Failure::Index)?;
            let left = *self.left.first().ok_or(Failure::Index)?;
            let corner = *self.corner.first().ok_or(Failure::Index)?;
            let path = self.paths.first().ok_or(Failure::Index)?.clone();
            self.top = vec![top];
            self.left = vec![left];
            self.corner = vec![corner];
            self.paths = vec![path];
        }
        Ok(())
    }
}
type CellRef = Rc<RefCell<Cell>>;
type Matrix = Vec<Vec<CellRef>>;

#[derive(Default, Debug)]
pub struct Stats {
    pub cells_created: usize,
    pub histories_created: usize,
    pub checkpoints: usize,
}
pub struct Alignment {
    pub distance: f64,
    pub paths: Vec<Vec<Step>>,
    pub stats: Stats,
}

fn at<T>(values: &[T], index: isize) -> Result<&T> {
    let index = if index < 0 {
        values.len() as isize + index
    } else {
        index
    };
    values
        .get(usize::try_from(index).map_err(|_| Failure::Index)?)
        .ok_or(Failure::Index)
}
fn cell(matrix: &Matrix, i: isize, j: isize) -> Result<CellRef> {
    Ok(at(at(matrix, i)?, j)?.clone())
}
fn set(matrix: &mut Matrix, i: usize, j: usize, value: CellRef) -> Result<()> {
    *matrix
        .get_mut(i)
        .and_then(|row| row.get_mut(j))
        .ok_or(Failure::Index)? = value;
    Ok(())
}
fn row_slice(matrix: &Matrix, i: usize, first: usize, end: usize) -> Result<Vec<CellRef>> {
    let row = matrix.get(i).ok_or(Failure::Index)?;
    let first = first.min(row.len());
    Ok(row[first..end.min(row.len()).max(first)].to_vec())
}
fn submatrix(matrix: &Matrix, i: usize, end_i: usize, j: usize, end_j: usize) -> Result<Matrix> {
    (i..end_i.min(matrix.len()))
        .map(|row| {
            // Unlike a Python slice, the reference's column list comprehension indexes each column.
            (j..end_j)
                .map(|col| cell(matrix, row as isize, col as isize))
                .collect()
        })
        .collect()
}
fn put_submatrix(
    matrix: &mut Matrix,
    sub: &Matrix,
    i: usize,
    end_i: usize,
    j: usize,
    end_j: usize,
) -> Result<()> {
    for row in i..end_i {
        for col in j..end_j {
            set(
                matrix,
                row,
                col,
                cell(sub, (row - i) as isize, (col - j) as isize)?,
            )?;
        }
    }
    Ok(())
}
fn mark(value: &CellRef, first: bool, swapped: bool) {
    for path in &value.borrow().paths {
        if first {
            path.step.borrow_mut().i_swapped = swapped;
        } else {
            path.step.borrow_mut().j_swapped = swapped;
        }
    }
}
fn collapse(a: &mut Matrix, b: &mut Matrix, i: usize, j: usize, first: bool) -> Result<()> {
    let left = cell(a, i as isize, j as isize)?;
    let right = cell(b, i as isize, j as isize)?;
    let lv = left.borrow().value;
    let rv = right.borrow().value;
    if lv < rv {
        set(b, i, j, left.clone())?;
        mark(&left, first, false);
    } else if lv > rv {
        set(a, i, j, right.clone())?;
        mark(&right, first, true);
    }
    Ok(())
}

fn parameter_cost(left: &str, right: &str) -> f64 {
    if left.contains("wrap_") || right.contains("wrap_") {
        return 1.0;
    }
    if left.split('(').next() != right.split('(').next() {
        return 1.0;
    }
    if left == right {
        0.0
    } else {
        0.5
    }
}
pub fn mutation_cost(left: &Token, right: &Token) -> Result<f64> {
    if left.name.contains("wrap_") || right.name.contains("wrap_") {
        return Ok(f64::INFINITY);
    }
    if left.name.split('(').next() != right.name.split('(').next() {
        return Ok(f64::INFINITY);
    }
    if (left.name == "branching(2)") != (right.name == "branching(2)") {
        return Ok(f64::INFINITY);
    }
    let first = parameter_cost(
        left.children.first().ok_or(Failure::Index)?,
        right.children.first().ok_or(Failure::Index)?,
    );
    if left.children.len() > 2 {
        let last = parameter_cost(
            left.children.last().ok_or(Failure::Index)?,
            right.children.last().ok_or(Failure::Index)?,
        );
        Ok((first + last) / 4.0)
    } else {
        Ok(first / 2.0)
    }
}

struct Kernel<'a> {
    original1: &'a [Token],
    original2: &'a [Token],
    collapse_corners: bool,
    check: &'a mut dyn FnMut() -> Result<()>,
    stats: Stats,
}
impl Kernel<'_> {
    fn initialize(&mut self, first: &[Token], second: &[Token], complete: bool) -> Result<Matrix> {
        let mut matrix = Vec::new();
        matrix
            .try_reserve_exact(first.len())
            .map_err(|_| Failure::Memory)?;
        for _ in first {
            let mut row = Vec::new();
            row.try_reserve_exact(second.len())
                .map_err(|_| Failure::Memory)?;
            for _ in second {
                row.push(Rc::new(RefCell::new(Cell::empty())));
                self.stats.cells_created += 1;
            }
            matrix.push(row);
        }
        if complete {
            for i in 0..first.len() {
                let item = cell(&matrix, i as isize, 0)?;
                item.borrow_mut().left = vec![f64::INFINITY];
                item.borrow_mut().corner = vec![f64::INFINITY];
            }
            for j in 0..second.len() {
                let item = cell(&matrix, 0, j as isize)?;
                item.borrow_mut().top = vec![f64::INFINITY];
                item.borrow_mut().corner = vec![f64::INFINITY];
            }
            let start = cell(&matrix, 0, 0)?;
            start.borrow_mut().value = 0.0;
            start.borrow_mut().paths = vec![Rc::new(History {
                step: RefCell::new(Step {
                    id: 0,
                    kind: Kind::Start,
                    node1_id: Some(at(first, 0)?.id),
                    node2_id: Some(at(second, 0)?.id),
                    i: 0,
                    j: 0,
                    value: 0.0,
                    i_swapped: false,
                    j_swapped: false,
                }),
                previous: None,
                len: 1,
            })];
            self.stats.histories_created += 1;
        }
        Ok(matrix)
    }

    fn deep_copy(&mut self, matrix: &Matrix) -> Matrix {
        fn history(path: &Path, memo: &mut HashMap<usize, Path>, count: &mut usize) -> Path {
            let key = Rc::as_ptr(path) as usize;
            if let Some(existing) = memo.get(&key) {
                return existing.clone();
            }
            // Iterative prefix cloning avoids a Rust recursion limit for long traces.
            let mut pending = vec![];
            let mut next = Some(path.clone());
            while let Some(current) = next {
                if memo.contains_key(&(Rc::as_ptr(&current) as usize)) {
                    break;
                }
                next = current.previous.clone();
                pending.push(current);
            }
            for original in pending.into_iter().rev() {
                let previous = original
                    .previous
                    .as_ref()
                    .map(|p| memo[&(Rc::as_ptr(p) as usize)].clone());
                let cloned = Rc::new(History {
                    step: RefCell::new(original.step.borrow().clone()),
                    previous,
                    len: original.len,
                });
                memo.insert(Rc::as_ptr(&original) as usize, cloned);
                *count += 1;
            }
            memo[&key].clone()
        }
        let mut cells = HashMap::<usize, CellRef>::new();
        let mut paths = HashMap::<usize, Path>::new();
        matrix
            .iter()
            .map(|row| {
                row.iter()
                    .map(|original| {
                        let key = Rc::as_ptr(original) as usize;
                        if let Some(existing) = cells.get(&key) {
                            return existing.clone();
                        }
                        let source = original.borrow();
                        let cloned = Rc::new(RefCell::new(Cell {
                            top: source.top.clone(),
                            left: source.left.clone(),
                            corner: source.corner.clone(),
                            value: source.value,
                            paths: source
                                .paths
                                .iter()
                                .map(|path| {
                                    history(path, &mut paths, &mut self.stats.histories_created)
                                })
                                .collect(),
                        }));
                        self.stats.cells_created += 1;
                        cells.insert(key, cloned.clone());
                        cloned
                    })
                    .collect()
            })
            .collect()
    }

    fn closing_cost(path: &Path, token: &Token, first: bool, previous_value: f64) -> f64 {
        let mut level = 0i64;
        let mut closed = 0i64;
        let mut current = Some(path.clone());
        while let Some(history) = current {
            let op = history.step.borrow();
            let own_id = if first { op.node1_id } else { op.node2_id };
            let own_kind = if first { op.kind.add() } else { op.kind.rem() };
            if own_id == Some(token.id) && own_kind {
                closed += 1;
            }
            let opens = if first {
                matches!(op.kind, Kind::RemWrap | Kind::MutWrap)
            } else {
                matches!(op.kind, Kind::AddWrap | Kind::MutWrap)
            };
            let closes = if first {
                matches!(op.kind, Kind::RemEnd | Kind::MutEnd)
            } else {
                matches!(op.kind, Kind::AddEnd | Kind::MutEnd)
            };
            if opens {
                level -= 1;
            } else if closes {
                level += 1;
            }
            if level == -1 || op.kind == Kind::Start {
                break;
            }
            let opening = if first { Kind::AddWrap } else { Kind::RemWrap };
            if own_id == Some(token.id) && op.kind == opening {
                break;
            }
            current = history.previous.clone();
        }
        let required = token.parent_arity as i64 - 2 - i64::from(token.name == "wrap_sep");
        if level == 0 && required == closed {
            previous_value
        } else {
            f64::INFINITY
        }
    }
    fn closing_mutation(path: &Path, first: &Token, second: &Token, previous_value: f64) -> f64 {
        let mut closed1 = 0i64;
        let mut closed2 = 0i64;
        let mut current = Some(path.clone());
        while let Some(history) = current {
            let op = history.step.borrow();
            if op.node1_id == Some(first.id) && op.kind.mutation() {
                closed1 += 1;
            }
            if op.node2_id == Some(second.id) && op.kind.mutation() {
                closed2 += 1;
            }
            if op.kind == Kind::Start {
                break;
            }
            if op.node1_id == Some(first.id)
                && op.node2_id == Some(second.id)
                && op.kind == Kind::MutWrap
            {
                break;
            }
            current = history.previous.clone();
        }
        let required1 = first.parent_arity as i64 - 2 - i64::from(first.name == "wrap_sep");
        let required2 = second.parent_arity as i64 - 2 - i64::from(second.name == "wrap_sep");
        if required1 == closed1 && required2 == closed2 {
            previous_value
        } else {
            f64::INFINITY
        }
    }

    fn fill_cell(
        &mut self,
        matrix: &Matrix,
        first: &[Token],
        second: &[Token],
        (i, j): (usize, usize),
        (start_i, start_j): (usize, usize),
    ) -> Result<bool> {
        let destination = cell(matrix, i as isize, j as isize)?;
        if !destination.borrow().value.is_nan() {
            return Ok(false);
        }
        let one = at(first, i as isize)?;
        let two = at(second, j as isize)?;
        if destination.borrow().top.is_empty() {
            let previous = cell(matrix, i as isize - 1, j as isize)?;
            let source = previous.borrow();
            destination.borrow_mut().top = source
                .paths
                .iter()
                .map(|p| {
                    if one.boundary() {
                        Self::closing_cost(p, one, true, source.value)
                    } else {
                        source.value + 1.0
                    }
                })
                .collect();
        }
        if destination.borrow().left.is_empty() {
            let previous = cell(matrix, i as isize, j as isize - 1)?;
            let source = previous.borrow();
            destination.borrow_mut().left = source
                .paths
                .iter()
                .map(|p| {
                    if two.boundary() {
                        Self::closing_cost(p, two, false, source.value)
                    } else {
                        source.value + 1.0
                    }
                })
                .collect();
        }
        if destination.borrow().corner.is_empty() {
            let cost = mutation_cost(one, two)?;
            let previous = cell(matrix, i as isize - 1, j as isize - 1)?;
            let source = previous.borrow();
            destination.borrow_mut().corner = source
                .paths
                .iter()
                .map(|p| {
                    if one.boundary() && one.name == two.name {
                        Self::closing_mutation(p, one, two, source.value)
                    } else {
                        source.value + cost
                    }
                })
                .collect();
        }
        let value = {
            let data = destination.borrow();
            data.top
                .iter()
                .chain(data.corner.iter())
                .chain(data.left.iter())
                .copied()
                .reduce(f64::min)
                .ok_or(Failure::EmptyMinimum)?
        };
        destination.borrow_mut().value = value;
        // History order is TOP, LEFT, CORNER, despite TOP,CORNER,LEFT in nanmin.
        for direction in 0..3 {
            let costs = {
                let mut data = destination.borrow_mut();
                std::mem::take(match direction {
                    0 => &mut data.top,
                    1 => &mut data.left,
                    _ => &mut data.corner,
                })
            };
            for (index, cost) in costs.iter().enumerate() {
                if *cost != value {
                    continue;
                }
                let previous = match direction {
                    0 => cell(matrix, i as isize - 1, j as isize)?,
                    1 => cell(matrix, i as isize, j as isize - 1)?,
                    _ => cell(matrix, i as isize - 1, j as isize - 1)?,
                };
                let parent = previous
                    .borrow()
                    .paths
                    .get(index)
                    .ok_or(Failure::Index)?
                    .clone();
                let charge = if value.is_infinite() {
                    value
                } else {
                    value - previous.borrow().value
                };
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
                    Kind::AddEnd
                        | Kind::AddSep
                        | Kind::RemEnd
                        | Kind::RemSep
                        | Kind::MutEnd
                        | Kind::MutSep
                );
                let history = Rc::new(History {
                    step: RefCell::new(Step {
                        id: parent.len,
                        kind,
                        node1_id: if kind == Kind::Rem {
                            None
                        } else {
                            Some(one.id)
                        },
                        node2_id: Some(two.id),
                        i: i + start_i,
                        j: j + start_j,
                        value: if boundary { 0.0 } else { charge },
                        i_swapped: false,
                        j_swapped: false,
                    }),
                    len: parent.len + 1,
                    previous: Some(parent),
                });
                self.stats.histories_created += 1;
                destination.borrow_mut().paths.push(history);
            }
            let mut data = destination.borrow_mut();
            *match direction {
                0 => &mut data.top,
                1 => &mut data.left,
                _ => &mut data.corner,
            } = costs;
        }
        Ok(true)
    }

    fn calculate(
        &mut self,
        mut matrix: Matrix,
        first: &[Token],
        second: &[Token],
        start_i: usize,
        start_j: usize,
    ) -> Result<Matrix> {
        let mut iswap: Option<Matrix> = None;
        let mut jswap: Option<Matrix> = None;
        let mut ijswap: Option<Matrix> = None;
        let (mut prev_i, mut prev_j) = (0, 0);
        let (mut max_i_saved, mut max_j_saved) = (None, None);
        while cell(&matrix, -1, -1)?.borrow().value.is_nan() {
            self.stats.checkpoints += 1;
            (self.check)()?;
            let compute_i = at(first, prev_i as isize)?.name == "branching(2)" && prev_i > 0;
            let compute_j = at(second, prev_j as isize)?.name == "branching(2)" && prev_j > 0;
            let mut mid_i = None;
            let mut mid_j = None;
            if compute_i {
                for index in prev_i + 1..first.len() {
                    mid_i = Some(index);
                    if first[prev_i].id == first[index].id && first[index].name.contains("sep") {
                        break;
                    }
                }
                for index in mid_i.ok_or(Failure::Unbound("mid_i"))? + 1..first.len() {
                    max_i_saved = Some(index);
                    if first[prev_i].id == first[index].id && first[index].name.contains("end") {
                        break;
                    }
                }
            } else {
                for (index, token) in first.iter().enumerate().skip(prev_i + 1) {
                    max_i_saved = Some(index);
                    if token.name == "branching(2)" {
                        break;
                    }
                }
            }
            let max_i = max_i_saved.ok_or(Failure::Unbound("max_i"))? + 1;
            max_i_saved = Some(max_i);
            if compute_j {
                for index in prev_j + 1..second.len() {
                    mid_j = Some(index);
                    if second[prev_j].id == second[index].id && second[index].name.contains("sep") {
                        break;
                    }
                }
                for index in mid_j.ok_or(Failure::Unbound("mid_j"))? + 1..second.len() {
                    max_j_saved = Some(index);
                    if second[prev_j].id == second[index].id && second[index].name.contains("end") {
                        break;
                    }
                }
            } else {
                for (index, token) in second.iter().enumerate().skip(prev_j + 1) {
                    max_j_saved = Some(index);
                    if token.name == "branching(2)" {
                        break;
                    }
                }
            }
            let max_j = max_j_saved.ok_or(Failure::Unbound("max_j"))? + 1;
            max_j_saved = Some(max_j);
            if (compute_i || compute_j) && (prev_i, prev_j) != (0, 0) {
                if iswap.is_none() {
                    iswap = Some(self.deep_copy(&matrix));
                }
                if jswap.is_none() {
                    jswap = Some(self.deep_copy(&matrix));
                }
                if ijswap.is_none() {
                    ijswap = Some(self.deep_copy(&matrix));
                }
                let im = iswap.as_mut().ok_or(Failure::Index)?;
                let jm = jswap.as_mut().ok_or(Failure::Index)?;
                let ijm = ijswap.as_mut().ok_or(Failure::Index)?;
                let a = &first[prev_i..max_i.min(first.len())];
                let b = &second[prev_j..max_j.min(second.len())];
                let swapped = |tokens: &[Token],
                               prev: usize,
                               mid: Option<usize>,
                               end: usize|
                 -> Result<Vec<Token>> {
                    if let Some(mid) = mid {
                        let mut result = vec![at(tokens, prev as isize)?.clone()];
                        result.extend_from_slice(
                            &tokens[(mid + 1).min(tokens.len())..(end - 1).min(tokens.len())],
                        );
                        result.push(at(tokens, mid as isize)?.clone());
                        result.extend_from_slice(
                            &tokens[(prev + 1).min(tokens.len())..mid.min(tokens.len())],
                        );
                        result.push(at(tokens, end as isize - 1)?.clone());
                        Ok(result)
                    } else {
                        Ok(tokens[prev..end.min(tokens.len())].to_vec())
                    }
                };
                let a_swap = swapped(first, prev_i, mid_i, max_i)?;
                let b_swap = swapped(second, prev_j, mid_j, max_j)?;
                let aux = self.calculate(
                    submatrix(&matrix, prev_i, max_i, prev_j, max_j)?,
                    a,
                    b,
                    start_i + prev_i,
                    start_j + prev_j,
                )?;
                put_submatrix(&mut matrix, &aux, prev_i, max_i, prev_j, max_j)?;
                let mut aux_i = None;
                let mut aux_j = None;
                if compute_i {
                    let mut aux = self.initialize(&a_swap, b, false)?;
                    for i in 0..aux.len() {
                        set(
                            &mut aux,
                            i,
                            0,
                            cell(im, (prev_i + i) as isize, prev_j as isize)?,
                        )?;
                    }
                    if cell(&aux, 0, -1)?.borrow().value.is_nan() {
                        aux[0] = row_slice(&matrix, prev_i, prev_j, max_j)?;
                    }
                    let aux =
                        self.calculate(aux, &a_swap, b, start_i + prev_i, start_j + prev_j)?;
                    for j in prev_j..max_j {
                        mark(&cell(im, max_i as isize - 1, j as isize)?, true, true);
                    }
                    if !cell(ijm, prev_i as isize, prev_j as isize)?
                        .borrow()
                        .value
                        .is_nan()
                        && !compute_j
                    {
                        let mut both = self.initialize(&a_swap, &b_swap, false)?;
                        both[0] = row_slice(ijm, prev_i, prev_j, max_j)?;
                        for i in 0..both.len() {
                            set(
                                &mut both,
                                i,
                                0,
                                cell(ijm, (prev_i + i) as isize, prev_j as isize)?,
                            )?;
                        }
                        let both = self.calculate(
                            both,
                            &a_swap,
                            &b_swap,
                            start_i + prev_i,
                            start_j + prev_j,
                        )?;
                        put_submatrix(ijm, &both, prev_i, max_i, prev_j, max_j)?;
                        for j in prev_j..max_j {
                            collapse(im, ijm, max_i - 1, j, false)?;
                        }
                    }
                    put_submatrix(im, &aux, prev_i, max_i, prev_j, max_j)?;
                    for j in prev_j..max_j {
                        collapse(&mut matrix, im, max_i - 1, j, true)?;
                    }
                    aux_i = Some(aux);
                }
                if compute_j {
                    let mut aux = self.initialize(a, &b_swap, false)?;
                    aux[0] = row_slice(jm, prev_i, prev_j, max_j)?;
                    if cell(&aux, -1, 0)?.borrow().value.is_nan() {
                        for i in 0..aux.len() {
                            set(
                                &mut aux,
                                i,
                                0,
                                cell(&matrix, (prev_i + i) as isize, prev_j as isize)?,
                            )?;
                        }
                    }
                    let aux =
                        self.calculate(aux, a, &b_swap, start_i + prev_i, start_j + prev_j)?;
                    for i in prev_i..max_i {
                        mark(&cell(jm, i as isize, max_j as isize - 1)?, false, true);
                    }
                    if !cell(ijm, prev_i as isize, prev_j as isize)?
                        .borrow()
                        .value
                        .is_nan()
                        && !compute_i
                    {
                        let mut both = self.initialize(&a_swap, &b_swap, false)?;
                        both[0] = row_slice(ijm, prev_i, prev_j, max_j)?;
                        for i in 0..both.len() {
                            set(
                                &mut both,
                                i,
                                0,
                                cell(ijm, (prev_i + i) as isize, prev_j as isize)?,
                            )?;
                        }
                        let both = self.calculate(
                            both,
                            &a_swap,
                            &b_swap,
                            start_i + prev_i,
                            start_j + prev_j,
                        )?;
                        put_submatrix(ijm, &both, prev_i, max_i, prev_j, max_j)?;
                        // The source sets j_swapped in this collapse too.
                        for i in prev_i..max_i {
                            collapse(jm, ijm, i, max_j - 1, false)?;
                        }
                    }
                    put_submatrix(jm, &aux, prev_i, max_i, prev_j, max_j)?;
                    for i in prev_i..max_i {
                        collapse(&mut matrix, jm, i, max_j - 1, false)?;
                    }
                    aux_j = Some(aux);
                }
                if compute_i && compute_j {
                    let mut both = self.initialize(&a_swap, &b_swap, false)?;
                    // Offsets address the enclosing matrix, not the local auxiliary block.
                    both[0] = row_slice(ijm, prev_i, prev_j, max_j)?;
                    for i in 0..both.len() {
                        set(
                            &mut both,
                            i,
                            0,
                            cell(ijm, (prev_i + i) as isize, prev_j as isize)?,
                        )?;
                    }
                    if cell(&both, 0, -1)?.borrow().value.is_nan() {
                        both[0] = row_slice(jm, prev_i, prev_j, max_j)?;
                    }
                    if cell(&both, -1, 0)?.borrow().value.is_nan() {
                        for i in 0..both.len() {
                            set(
                                &mut both,
                                i,
                                0,
                                cell(im, (prev_i + i) as isize, prev_j as isize)?,
                            )?;
                        }
                    }
                    let mut both =
                        self.calculate(both, &a_swap, &b_swap, start_i + prev_i, start_j + prev_j)?;
                    let ai = aux_i.as_ref().ok_or(Failure::Unbound("aux_matrix_iswap"))?;
                    let aj = aux_j.as_ref().ok_or(Failure::Unbound("aux_matrix_jswap"))?;
                    for i in 0..both.len() {
                        let current = cell(&both, i as isize, -1)?;
                        mark(&current, true, true);
                        let other = cell(aj, i as isize, -1)?;
                        if other.borrow().value < current.borrow().value {
                            let last = both[i].len().checked_sub(1).ok_or(Failure::Index)?;
                            set(&mut both, i, last, other)?;
                        }
                    }
                    let width = at(&both, 0)?.len();
                    for j in 0..width {
                        let current = cell(&both, -1, j as isize)?;
                        mark(&current, false, true);
                        let other = cell(ai, -1, j as isize)?;
                        if other.borrow().value < current.borrow().value {
                            let last = both.len().checked_sub(1).ok_or(Failure::Index)?;
                            set(&mut both, last, j, other)?;
                        }
                    }
                    put_submatrix(ijm, &both, prev_i, max_i, prev_j, max_j)?;
                    let current = cell(&matrix, max_i as isize - 1, max_j as isize - 1)?;
                    let other = cell(ijm, max_i as isize - 1, max_j as isize - 1)?;
                    if current.borrow().value > other.borrow().value {
                        set(&mut matrix, max_i - 1, max_j - 1, other)?;
                    }
                }
            } else {
                let rows = max_i.checked_sub(prev_i).ok_or(Failure::Index)?;
                let cols = max_j.checked_sub(prev_j).ok_or(Failure::Index)?;
                for diagonal in 0..rows + cols - 1 {
                    let low = diagonal.saturating_sub(cols - 1);
                    let high = diagonal.min(rows - 1);
                    for offset in 0..=high - low {
                        let row = if diagonal % 2 == 0 {
                            low + offset
                        } else {
                            high - offset
                        };
                        let i = prev_i + row;
                        let j = prev_j + diagonal - row;
                        if self.fill_cell(&matrix, first, second, (i, j), (start_i, start_j))? {
                            for auxiliary in
                                [&mut iswap, &mut jswap, &mut ijswap].into_iter().flatten()
                            {
                                set(auxiliary, i, j, cell(&matrix, i as isize, j as isize)?)?;
                            }
                            if self.collapse_corners
                                && (((j + start_j) as f64 - (i + start_i) as f64
                                    >= self.original2.len() as f64 * 0.25)
                                    || ((i + start_i) as f64 - (j + start_j) as f64
                                        >= self.original1.len() as f64 * 0.25))
                            {
                                cell(&matrix, i as isize, j as isize)?
                                    .borrow_mut()
                                    .clean(false)?;
                                for auxiliary in [&iswap, &jswap, &ijswap].into_iter().flatten() {
                                    cell(auxiliary, i as isize, j as isize)?
                                        .borrow_mut()
                                        .clean(false)?;
                                }
                            }
                        }
                        if i > 1 && j > 1 {
                            cell(&matrix, i as isize - 1, j as isize - 1)?
                                .borrow_mut()
                                .clean(true)?;
                            for auxiliary in [&iswap, &jswap, &ijswap].into_iter().flatten() {
                                cell(auxiliary, i as isize - 1, j as isize - 1)?
                                    .borrow_mut()
                                    .clean(true)?;
                            }
                        }
                    }
                }
            }
            let full_first = first.len() == self.original1.len()
                && first
                    .iter()
                    .zip(self.original1)
                    .all(|(a, b)| a.source_eq(b));
            let full_second = second.len() == self.original2.len()
                && second
                    .iter()
                    .zip(self.original2)
                    .all(|(a, b)| a.source_eq(b));
            if full_first && full_second {
                for i in prev_i..max_i - usize::from(max_i < first.len()) {
                    for j in prev_j..max_j - usize::from(max_j < second.len()) {
                        if i < self.original1.len() - 1 && j < self.original2.len() - 1 {
                            cell(&matrix, i as isize, j as isize)?
                                .borrow_mut()
                                .clean(true)?;
                            for auxiliary in [&iswap, &jswap, &ijswap].into_iter().flatten() {
                                cell(auxiliary, i as isize, j as isize)?
                                    .borrow_mut()
                                    .clean(true)?;
                            }
                        }
                    }
                }
            }
            if max_j >= second.len() {
                prev_j = 0;
                prev_i = max_i - 1;
            } else {
                prev_j = max_j - 1;
            }
        }
        Ok(matrix)
    }
}

pub fn align(
    first: &[Token],
    second: &[Token],
    collapse_corners: bool,
    check: &mut dyn FnMut() -> Result<()>,
) -> Result<Alignment> {
    let mut kernel = Kernel {
        original1: first,
        original2: second,
        collapse_corners,
        check,
        stats: Stats::default(),
    };
    let initial = kernel.initialize(first, second, true)?;
    let matrix = kernel.calculate(initial, first, second, 0, 0)?;
    let end = cell(&matrix, -1, -1)?;
    let end = end.borrow();
    let paths = end
        .paths
        .iter()
        .map(|path| {
            let mut steps = Vec::with_capacity(path.len);
            let mut current = Some(path.clone());
            while let Some(history) = current {
                steps.push(history.step.borrow().clone());
                current = history.previous.clone();
            }
            steps.reverse();
            steps
        })
        .collect();
    Ok(Alignment {
        distance: end.value,
        paths,
        stats: kernel.stats,
    })
}
