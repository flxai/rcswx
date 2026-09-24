//! Source-compatible recursive RCSWX, including reference failure paths.
//!
//! Histories are persistent lists, not merged dynamic-programming states. Appending
//! shares immutable prefixes and the original mutable operation identities, just
//! as Python's `path + [operation]` shares the preceding operation objects.
//! Derived from einsearch 3f44ddf086bee0c213404e240ee0adf99e3e1501.
//! See LICENSE.einsearch for the original MIT attribution.

use crate::execution::{self, Execution};
use std::cell::RefCell;
use std::collections::HashMap;
use std::rc::Rc;
#[cfg(not(all(feature = "parallel", not(target_arch = "wasm32"))))]
use std::rc::Rc as HistoryRef;
#[cfg(all(
    feature = "trace",
    not(all(feature = "parallel", not(target_arch = "wasm32")))
))]
use std::rc::Weak as HistoryWeak;
#[cfg(all(feature = "parallel", not(target_arch = "wasm32")))]
use std::sync::Arc as HistoryRef;
#[cfg(all(feature = "trace", feature = "parallel", not(target_arch = "wasm32")))]
use std::sync::Weak as HistoryWeak;
use std::sync::atomic::{AtomicU8, Ordering};

#[path = "recursive_cell.rs"]
mod cell_eval;

#[cfg(all(feature = "parallel", not(target_arch = "wasm32")))]
#[path = "recursive_wave.rs"]
mod wave;

#[cfg(feature = "trace")]
#[path = "recursive_trace.rs"]
mod recording;

#[derive(Debug)]
pub enum Failure {
    Index,
    EmptyMinimum,
    Unbound(&'static str),
    Memory,
    Callback,
    Operational(crate::error::Error),
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
/// Prefix structure is immutable. Only the coordinator writes swap annotations,
/// after all readers have joined; evaluation never consults those annotations.
struct History {
    step: Step,
    swapped: AtomicU8,
    previous: Option<Path>,
    len: usize,
}
impl History {
    fn new(step: Step, previous: Option<Path>, len: usize) -> Path {
        let swapped = u8::from(step.i_swapped) | (u8::from(step.j_swapped) << 1);
        HistoryRef::new(Self {
            step,
            swapped: AtomicU8::new(swapped),
            previous,
            len,
        })
    }
    fn snapshot(&self) -> Step {
        let swapped = self.swapped.load(Ordering::Relaxed);
        let mut step = self.step.clone();
        step.i_swapped = swapped & 1 != 0;
        step.j_swapped = swapped & 2 != 0;
        step
    }
    fn mark(&self, first: bool, swapped: bool) {
        let mask = if first { 1 } else { 2 };
        let flags = self.swapped.load(Ordering::Relaxed);
        self.swapped.store(
            if swapped { flags | mask } else { flags & !mask },
            Ordering::Relaxed,
        );
    }
}
impl Drop for History {
    fn drop(&mut self) {
        // Release unshared suffixes iteratively. Shared prefixes stay alive for
        // other cells, auxiliary matrices, or a joined wave's read views.
        let mut previous = self.previous.take();
        while let Some(mut history) = previous.and_then(HistoryRef::into_inner) {
            previous = history.previous.take();
        }
    }
}
type Path = HistoryRef<History>;

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

#[derive(Clone, Copy)]
struct Diagonal {
    sum: usize,
    low: usize,
    high: usize,
    origin: (usize, usize),
}
impl Diagonal {
    fn len(self) -> usize {
        self.high - self.low + 1
    }
    fn position(self, offset: usize) -> (usize, usize) {
        let row = if self.sum % 2 == 0 {
            self.low + offset
        } else {
            self.high - offset
        };
        (self.origin.0 + row, self.origin.1 + self.sum - row)
    }
}

#[derive(Default, Debug)]
pub struct Stats {
    pub cells_created: usize,
    pub histories_created: usize,
    pub checkpoints: usize,
    /// Cumulative tracked payload requests, not live heap usage or process RSS.
    pub allocation_bytes: usize,
    pub output_steps: usize,
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
        path.mark(first, swapped);
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
    if left == right { 0.0 } else { 0.5 }
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
    check: &'a mut dyn FnMut(&Stats, bool) -> Result<()>,
    execution: &'a mut Execution,
    observe_allocations: bool,
    stats: Stats,
    #[cfg(feature = "trace")]
    trace: Option<recording::Observer<'a>>,
}
impl Kernel<'_> {
    fn account(
        &mut self,
        cells: usize,
        histories: usize,
        bytes: usize,
        output: usize,
    ) -> Result<()> {
        self.stats.cells_created = self
            .stats
            .cells_created
            .checked_add(cells)
            .ok_or(Failure::Memory)?;
        self.stats.histories_created = self
            .stats
            .histories_created
            .checked_add(histories)
            .ok_or(Failure::Memory)?;
        self.stats.allocation_bytes = self
            .stats
            .allocation_bytes
            .checked_add(bytes)
            .ok_or(Failure::Memory)?;
        self.stats.output_steps = self
            .stats
            .output_steps
            .checked_add(output)
            .ok_or(Failure::Memory)?;
        if self.observe_allocations {
            (self.check)(&self.stats, false)?;
        }
        Ok(())
    }

    fn mark(&mut self, value: &CellRef, first: bool, swapped: bool) {
        mark(value, first, swapped);
        #[cfg(feature = "trace")]
        if let Some(trace) = &mut self.trace {
            trace.cell(value);
        }
    }

    fn collapse(
        &mut self,
        a: &mut Matrix,
        b: &mut Matrix,
        i: usize,
        j: usize,
        first: bool,
    ) -> Result<()> {
        #[cfg(feature = "trace")]
        if let Some(trace) = &mut self.trace {
            let left = trace.cell(&cell(a, i as isize, j as isize)?);
            let right = trace.cell(&cell(b, i as isize, j as isize)?);
            trace.recorder.emit(
                "collapse",
                || serde_json::json!({"left":left,"right":right,"i":i,"j":j,"first":first}),
            );
        }
        collapse(a, b, i, j, first)?;
        #[cfg(feature = "trace")]
        if let Some(trace) = &mut self.trace {
            trace.cell(&cell(a, i as isize, j as isize)?);
            trace.cell(&cell(b, i as isize, j as isize)?);
            trace.matrix(a, "collapse_first");
            trace.matrix(b, "collapse_second");
        }
        Ok(())
    }

    fn clean(&mut self, value: &CellRef, completely: bool) -> Result<()> {
        #[cfg(feature = "trace")]
        if let Some(trace) = &mut self.trace {
            trace.cell(value);
        }
        value.borrow_mut().clean(completely)?;
        #[cfg(feature = "trace")]
        if let Some(trace) = &mut self.trace {
            let id = trace.cell(value);
            trace.recorder.emit(
                "clean",
                || serde_json::json!({"cell":id,"completely":completely}),
            );
        }
        Ok(())
    }

    fn initialize(&mut self, first: &[Token], second: &[Token], complete: bool) -> Result<Matrix> {
        let cells = first
            .len()
            .checked_mul(second.len())
            .ok_or(Failure::Memory)?;
        let cell_bytes = std::mem::size_of::<Cell>()
            + 2 * std::mem::size_of::<usize>()
            + std::mem::size_of::<CellRef>();
        let bytes = cells
            .checked_mul(cell_bytes)
            .and_then(|bytes| {
                first
                    .len()
                    .checked_mul(std::mem::size_of::<Vec<CellRef>>())
                    .and_then(|rows| bytes.checked_add(rows))
            })
            .ok_or(Failure::Memory)?;
        self.account(cells, 0, bytes, 0)?;
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
            self.account(
                0,
                1,
                std::mem::size_of::<History>() + 2 * std::mem::size_of::<usize>(),
                0,
            )?;
            start.borrow_mut().paths = vec![History::new(
                Step {
                    id: 0,
                    kind: Kind::Start,
                    node1_id: Some(at(first, 0)?.id),
                    node2_id: Some(at(second, 0)?.id),
                    i: 0,
                    j: 0,
                    value: 0.0,
                    i_swapped: false,
                    j_swapped: false,
                },
                None,
                1,
            )];
        }
        Ok(matrix)
    }

    fn deep_copy(&mut self, matrix: &Matrix) -> Result<Matrix> {
        fn history(
            kernel: &mut Kernel<'_>,
            path: &Path,
            memo: &mut HashMap<usize, Path>,
        ) -> Result<Path> {
            let key = HistoryRef::as_ptr(path) as usize;
            if let Some(existing) = memo.get(&key) {
                return Ok(existing.clone());
            }
            // Iterative prefix cloning preserves the shared history graph.
            let mut pending = vec![];
            let mut next = Some(path.clone());
            while let Some(current) = next {
                if memo.contains_key(&(HistoryRef::as_ptr(&current) as usize)) {
                    break;
                }
                next = current.previous.clone();
                pending.push(current);
            }
            for original in pending.into_iter().rev() {
                let previous = original
                    .previous
                    .as_ref()
                    .map(|p| memo[&(HistoryRef::as_ptr(p) as usize)].clone());
                kernel.account(
                    0,
                    1,
                    std::mem::size_of::<History>() + 2 * std::mem::size_of::<usize>(),
                    0,
                )?;
                let cloned = History::new(original.snapshot(), previous, original.len);
                #[cfg(feature = "trace")]
                if let Some(trace) = &mut kernel.trace {
                    trace.history_clone(&original, &cloned);
                }
                memo.insert(HistoryRef::as_ptr(&original) as usize, cloned);
            }
            Ok(memo[&key].clone())
        }
        let mut cells = HashMap::<usize, CellRef>::new();
        let mut paths = HashMap::<usize, Path>::new();
        let mut result = Vec::new();
        self.account(
            0,
            0,
            matrix
                .len()
                .checked_mul(std::mem::size_of::<Vec<CellRef>>())
                .ok_or(Failure::Memory)?,
            0,
        )?;
        result
            .try_reserve_exact(matrix.len())
            .map_err(|_| Failure::Memory)?;
        for row in matrix {
            let mut cloned_row = Vec::new();
            self.account(
                0,
                0,
                row.len()
                    .checked_mul(std::mem::size_of::<CellRef>())
                    .ok_or(Failure::Memory)?,
                0,
            )?;
            cloned_row
                .try_reserve_exact(row.len())
                .map_err(|_| Failure::Memory)?;
            for original in row {
                let key = Rc::as_ptr(original) as usize;
                if let Some(existing) = cells.get(&key) {
                    cloned_row.push(existing.clone());
                    continue;
                }
                let source = original.borrow();
                let mut cloned_paths = Vec::new();
                self.account(
                    0,
                    0,
                    source
                        .paths
                        .len()
                        .checked_mul(std::mem::size_of::<Path>())
                        .ok_or(Failure::Memory)?,
                    0,
                )?;
                cloned_paths
                    .try_reserve_exact(source.paths.len())
                    .map_err(|_| Failure::Memory)?;
                for path in &source.paths {
                    cloned_paths.push(history(self, path, &mut paths)?);
                }
                let values = source
                    .top
                    .len()
                    .checked_add(source.left.len())
                    .and_then(|n| n.checked_add(source.corner.len()))
                    .ok_or(Failure::Memory)?;
                let bytes = values
                    .checked_mul(std::mem::size_of::<f64>())
                    .and_then(|n| {
                        n.checked_add(
                            std::mem::size_of::<Cell>() + 2 * std::mem::size_of::<usize>(),
                        )
                    })
                    .ok_or(Failure::Memory)?;
                self.account(1, 0, bytes, 0)?;
                let cloned = Rc::new(RefCell::new(Cell {
                    top: source.top.clone(),
                    left: source.left.clone(),
                    corner: source.corner.clone(),
                    value: source.value,
                    paths: cloned_paths,
                }));
                #[cfg(feature = "trace")]
                if let Some(trace) = &mut self.trace {
                    trace.cell_clone(original, &cloned);
                }
                cells.insert(key, cloned.clone());
                cloned_row.push(cloned);
            }
            result.push(cloned_row);
        }
        Ok(result)
    }

    fn closing_cost(
        path: &Path,
        token: &Token,
        first: bool,
        previous_value: f64,
        cancelled: Option<&std::sync::atomic::AtomicBool>,
    ) -> Result<f64> {
        let mut level = 0i64;
        let mut closed = 0i64;
        let mut current = Some(path.as_ref());
        let mut walked = 0_usize;
        while let Some(history) = current {
            if walked % 64 == 0 {
                cell_eval::check_cancel(cancelled)?;
            }
            walked += 1;
            let op = &history.step;
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
            current = history.previous.as_deref();
        }
        let required = token.parent_arity as i64 - 2 - i64::from(token.name == "wrap_sep");
        if level == 0 && required == closed {
            Ok(previous_value)
        } else {
            Ok(f64::INFINITY)
        }
    }
    fn closing_mutation(
        path: &Path,
        first: &Token,
        second: &Token,
        previous_value: f64,
        cancelled: Option<&std::sync::atomic::AtomicBool>,
    ) -> Result<f64> {
        let mut closed1 = 0i64;
        let mut closed2 = 0i64;
        let mut current = Some(path.as_ref());
        let mut walked = 0_usize;
        while let Some(history) = current {
            if walked % 64 == 0 {
                cell_eval::check_cancel(cancelled)?;
            }
            walked += 1;
            let op = &history.step;
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
            current = history.previous.as_deref();
        }
        let required1 = first.parent_arity as i64 - 2 - i64::from(first.name == "wrap_sep");
        let required2 = second.parent_arity as i64 - 2 - i64::from(second.name == "wrap_sep");
        if required1 == closed1 && required2 == closed2 {
            Ok(previous_value)
        } else {
            Ok(f64::INFINITY)
        }
    }

    fn account_history(&mut self) -> Result<()> {
        self.account(
            0,
            1,
            std::mem::size_of::<History>()
                + 2 * std::mem::size_of::<usize>()
                + std::mem::size_of::<Path>(),
            0,
        )
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
        self.execution.report.serial_cells += 1;
        let one = at(first, i as isize)?;
        let two = at(second, j as isize)?;
        let top = cell(matrix, i as isize - 1, j as isize)?;
        let left = cell(matrix, i as isize, j as isize - 1)?;
        let corner = cell(matrix, i as isize - 1, j as isize - 1)?;
        let draft = {
            let destination = destination.borrow();
            let top = top.borrow();
            let left = left.borrow();
            let corner = corner.borrow();
            let input = cell_eval::Input {
                destination: &destination,
                predecessors: [&top, &left, &corner],
                first: one,
                second: two,
                position: (i + start_i, j + start_j),
            };
            cell_eval::Draft::evaluate(&input, || self.account_history(), None)
        };
        self.publish_cell(matrix, i, j, draft, false)?;
        Ok(true)
    }

    fn publish_cell(
        &mut self,
        matrix: &Matrix,
        i: usize,
        j: usize,
        mut draft: cell_eval::Draft,
        account_paths: bool,
    ) -> Result<()> {
        let destination = cell(matrix, i as isize, j as isize)?;
        if account_paths {
            for _ in 0..draft.history_requests {
                self.account_history()?;
            }
        }
        {
            let mut data = destination.borrow_mut();
            if let Some(top) = draft.candidates[0].take() {
                data.top = top;
            }
            if let Some(left) = draft.candidates[1].take() {
                data.left = left;
            }
            if let Some(corner) = draft.candidates[2].take() {
                data.corner = corner;
            }
            if let Some(value) = draft.value {
                data.value = value;
            }
            if data.paths.is_empty() {
                data.paths = draft.paths;
            } else {
                data.paths
                    .try_reserve(draft.paths.len())
                    .map_err(|_| Failure::Memory)?;
                data.paths.append(&mut draft.paths);
            }
        }
        if let Some(error) = draft.failure {
            return Err(error);
        }
        #[cfg(feature = "trace")]
        if let Some(trace) = &mut self.trace {
            trace.computed(matrix, i, j);
        }
        Ok(())
    }

    fn calculate(
        &mut self,
        mut matrix: Matrix,
        first: &[Token],
        second: &[Token],
        start_i: usize,
        start_j: usize,
        #[cfg(feature = "trace")] positions: Option<(Vec<usize>, Vec<usize>)>,
    ) -> Result<Matrix> {
        #[cfg(all(feature = "parallel", not(target_arch = "wasm32")))]
        let fresh_root =
            std::ptr::eq(first, self.original1) && std::ptr::eq(second, self.original2);
        #[cfg(all(feature = "parallel", not(target_arch = "wasm32")))]
        let parallel = self.execution.report.worker_limit > 1;
        #[cfg(feature = "trace")]
        if let (Some(trace), Some(positions)) = (&mut self.trace, &positions) {
            trace.begin(&matrix, positions, start_i, start_j);
        }
        let mut iswap: Option<Matrix> = None;
        let mut jswap: Option<Matrix> = None;
        let mut ijswap: Option<Matrix> = None;
        let (mut prev_i, mut prev_j) = (0, 0);
        let (mut max_i_saved, mut max_j_saved) = (None, None);
        while cell(&matrix, -1, -1)?.borrow().value.is_nan() {
            self.stats.checkpoints += 1;
            execution::callback(|| (self.check)(&self.stats, false))?;
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
                    iswap = Some(self.deep_copy(&matrix)?);
                }
                if jswap.is_none() {
                    jswap = Some(self.deep_copy(&matrix)?);
                }
                if ijswap.is_none() {
                    ijswap = Some(self.deep_copy(&matrix)?);
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
                #[cfg(feature = "trace")]
                let coordinates = |first_swapped, second_swapped| {
                    positions.as_ref().map(|(first, second)| {
                        (
                            recording::positions(first, prev_i, max_i, mid_i, first_swapped),
                            recording::positions(second, prev_j, max_j, mid_j, second_swapped),
                        )
                    })
                };
                let a_swap = swapped(first, prev_i, mid_i, max_i)?;
                let b_swap = swapped(second, prev_j, mid_j, max_j)?;
                let aux = self.calculate(
                    submatrix(&matrix, prev_i, max_i, prev_j, max_j)?,
                    a,
                    b,
                    start_i + prev_i,
                    start_j + prev_j,
                    #[cfg(feature = "trace")]
                    coordinates(false, false),
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
                    let aux = self.calculate(
                        aux,
                        &a_swap,
                        b,
                        start_i + prev_i,
                        start_j + prev_j,
                        #[cfg(feature = "trace")]
                        coordinates(true, false),
                    )?;
                    for j in prev_j..max_j {
                        self.mark(&cell(im, max_i as isize - 1, j as isize)?, true, true);
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
                            #[cfg(feature = "trace")]
                            coordinates(true, true),
                        )?;
                        put_submatrix(ijm, &both, prev_i, max_i, prev_j, max_j)?;
                        for j in prev_j..max_j {
                            self.collapse(im, ijm, max_i - 1, j, false)?;
                        }
                    }
                    put_submatrix(im, &aux, prev_i, max_i, prev_j, max_j)?;
                    for j in prev_j..max_j {
                        self.collapse(&mut matrix, im, max_i - 1, j, true)?;
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
                    let aux = self.calculate(
                        aux,
                        a,
                        &b_swap,
                        start_i + prev_i,
                        start_j + prev_j,
                        #[cfg(feature = "trace")]
                        coordinates(false, true),
                    )?;
                    for i in prev_i..max_i {
                        self.mark(&cell(jm, i as isize, max_j as isize - 1)?, false, true);
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
                            #[cfg(feature = "trace")]
                            coordinates(true, true),
                        )?;
                        put_submatrix(ijm, &both, prev_i, max_i, prev_j, max_j)?;
                        // The source sets j_swapped in this collapse too.
                        for i in prev_i..max_i {
                            self.collapse(jm, ijm, i, max_j - 1, false)?;
                        }
                    }
                    put_submatrix(jm, &aux, prev_i, max_i, prev_j, max_j)?;
                    for i in prev_i..max_i {
                        self.collapse(&mut matrix, jm, i, max_j - 1, false)?;
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
                    let mut both = self.calculate(
                        both,
                        &a_swap,
                        &b_swap,
                        start_i + prev_i,
                        start_j + prev_j,
                        #[cfg(feature = "trace")]
                        coordinates(true, true),
                    )?;
                    let ai = aux_i.as_ref().ok_or(Failure::Unbound("aux_matrix_iswap"))?;
                    let aj = aux_j.as_ref().ok_or(Failure::Unbound("aux_matrix_jswap"))?;
                    for i in 0..both.len() {
                        let current = cell(&both, i as isize, -1)?;
                        self.mark(&current, true, true);
                        let other = cell(aj, i as isize, -1)?;
                        if other.borrow().value < current.borrow().value {
                            let last = both[i].len().checked_sub(1).ok_or(Failure::Index)?;
                            set(&mut both, i, last, other)?;
                        }
                    }
                    let width = at(&both, 0)?.len();
                    for j in 0..width {
                        let current = cell(&both, -1, j as isize)?;
                        self.mark(&current, false, true);
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
                #[cfg(all(feature = "parallel", not(target_arch = "wasm32")))]
                let round_cells = wave::ROUND_CELLS;
                #[cfg(all(test, feature = "parallel", not(target_arch = "wasm32")))]
                let round_cells = round_cells.min(self.execution.round_cells.max(1));
                for diagonal in 0..rows + cols - 1 {
                    let low = diagonal.saturating_sub(cols - 1);
                    let high = diagonal.min(rows - 1);
                    let geometry = Diagonal {
                        sum: diagonal,
                        low,
                        high,
                        origin: (prev_i, prev_j),
                    };
                    let mut offset = 0;
                    while offset < geometry.len() {
                        let count = geometry.len() - offset;
                        #[cfg(all(feature = "parallel", not(target_arch = "wasm32")))]
                        let count = if parallel {
                            count.min(round_cells)
                        } else {
                            count
                        };
                        #[cfg(all(feature = "parallel", not(target_arch = "wasm32")))]
                        let mut batch = if parallel {
                            wave::evaluate(
                                self,
                                wave::Round {
                                    matrix: &matrix,
                                    tokens: (first, second),
                                    diagonal: geometry,
                                    offset,
                                    count,
                                    start: (start_i, start_j),
                                    auxiliaries: [iswap.as_ref(), jswap.as_ref(), ijswap.as_ref()],
                                    fresh: fresh_root
                                        && iswap.is_none()
                                        && jswap.is_none()
                                        && ijswap.is_none(),
                                },
                            )?
                        } else {
                            None
                        };
                        #[cfg(all(feature = "parallel", not(target_arch = "wasm32")))]
                        let mut next_draft = 0;
                        for rank in 0..count {
                            let (i, j) = geometry.position(offset + rank);
                            #[cfg(all(feature = "parallel", not(target_arch = "wasm32")))]
                            let computed = if let Some(batch) = &mut batch {
                                if batch.ranks.get(next_draft) == Some(&rank) {
                                    let draft =
                                        batch.drafts[next_draft].take().ok_or_else(|| {
                                            Failure::Operational(crate::error::Error::Execution(
                                                "joined native round is missing a cell".into(),
                                            ))
                                        })?;
                                    self.publish_cell(&matrix, i, j, draft, true)?;
                                    next_draft += 1;
                                    true
                                } else {
                                    false
                                }
                            } else {
                                self.fill_cell(&matrix, first, second, (i, j), (start_i, start_j))?
                            };
                            #[cfg(not(all(feature = "parallel", not(target_arch = "wasm32"))))]
                            let computed =
                                self.fill_cell(&matrix, first, second, (i, j), (start_i, start_j))?;
                            if computed {
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
                                    self.clean(&cell(&matrix, i as isize, j as isize)?, false)?;
                                    for auxiliary in [&iswap, &jswap, &ijswap].into_iter().flatten()
                                    {
                                        self.clean(
                                            &cell(auxiliary, i as isize, j as isize)?,
                                            false,
                                        )?;
                                    }
                                }
                            }
                            if i > 1 && j > 1 {
                                self.clean(&cell(&matrix, i as isize - 1, j as isize - 1)?, true)?;
                                for auxiliary in [&iswap, &jswap, &ijswap].into_iter().flatten() {
                                    self.clean(
                                        &cell(auxiliary, i as isize - 1, j as isize - 1)?,
                                        true,
                                    )?;
                                }
                            }
                        }
                        offset += count;
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
                            self.clean(&cell(&matrix, i as isize, j as isize)?, true)?;
                            for auxiliary in [&iswap, &jswap, &ijswap].into_iter().flatten() {
                                self.clean(&cell(auxiliary, i as isize, j as isize)?, true)?;
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
        #[cfg(feature = "trace")]
        if let Some(trace) = &mut self.trace {
            trace.end(&matrix);
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
    align_with_execution(
        first,
        second,
        collapse_corners,
        check,
        &mut Execution::serial(),
    )
}

pub fn align_with_execution(
    first: &[Token],
    second: &[Token],
    collapse_corners: bool,
    check: &mut dyn FnMut() -> Result<()>,
    execution: &mut Execution,
) -> Result<Alignment> {
    align_internal(
        first,
        second,
        collapse_corners,
        &mut |_, _| check(),
        false,
        execution,
        #[cfg(feature = "trace")]
        None,
    )
}

/// Observe resource counters before retained matrix/history/output allocation.
/// `checkpoints` advances only at the original host-callback boundary.
pub fn align_observed(
    first: &[Token],
    second: &[Token],
    collapse_corners: bool,
    observe: &mut dyn FnMut(&Stats) -> Result<()>,
) -> Result<Alignment> {
    align_observed_with_execution(
        first,
        second,
        collapse_corners,
        &mut |stats, _| observe(stats),
        &mut Execution::serial(),
    )
}

/// A true observer flag requests an elapsed-time host poll without advancing
/// canonical checkpoints. Resource and execution diagnostics stay independent.
pub(crate) fn align_observed_with_execution(
    first: &[Token],
    second: &[Token],
    collapse_corners: bool,
    observe: &mut dyn FnMut(&Stats, bool) -> Result<()>,
    execution: &mut Execution,
) -> Result<Alignment> {
    align_internal(
        first,
        second,
        collapse_corners,
        observe,
        true,
        execution,
        #[cfg(feature = "trace")]
        None,
    )
}

#[cfg(feature = "trace")]
pub fn align_traced(
    first: &[Token],
    second: &[Token],
    collapse_corners: bool,
    observe: &mut dyn FnMut(&Stats) -> Result<()>,
    recorder: &mut crate::trace::Recorder,
) -> Result<Alignment> {
    align_traced_with_execution(
        first,
        second,
        collapse_corners,
        &mut |stats, _| observe(stats),
        &mut Execution::serial(),
        recorder,
    )
}

#[cfg(feature = "trace")]
pub(crate) fn align_traced_with_execution(
    first: &[Token],
    second: &[Token],
    collapse_corners: bool,
    observe: &mut dyn FnMut(&Stats, bool) -> Result<()>,
    execution: &mut Execution,
    recorder: &mut crate::trace::Recorder,
) -> Result<Alignment> {
    align_internal(
        first,
        second,
        collapse_corners,
        observe,
        true,
        execution,
        Some(recorder),
    )
}

fn align_internal(
    first: &[Token],
    second: &[Token],
    collapse_corners: bool,
    check: &mut dyn FnMut(&Stats, bool) -> Result<()>,
    observe_allocations: bool,
    execution: &mut Execution,
    #[cfg(feature = "trace")] recorder: Option<&mut crate::trace::Recorder>,
) -> Result<Alignment> {
    execution::Workers::new(execution.report().requested_workers).map_err(Failure::Operational)?;
    let mut kernel = Kernel {
        original1: first,
        original2: second,
        collapse_corners,
        check,
        execution,
        observe_allocations,
        stats: Stats::default(),
        #[cfg(feature = "trace")]
        trace: recorder.filter(|r| r.full()).map(recording::Observer::new),
    };
    let initial = kernel.initialize(first, second, true)?;
    let matrix = kernel.calculate(
        initial,
        first,
        second,
        0,
        0,
        #[cfg(feature = "trace")]
        kernel
            .trace
            .as_ref()
            .map(|_| ((0..first.len()).collect(), (0..second.len()).collect())),
    )?;
    let end = cell(&matrix, -1, -1)?;
    let end = end.borrow();
    #[cfg(feature = "trace")]
    if let Some(trace) = &mut kernel.trace {
        trace.retained(&end.paths);
    }
    let mut paths = Vec::new();
    kernel.account(
        0,
        0,
        end.paths
            .len()
            .checked_mul(std::mem::size_of::<Vec<Step>>())
            .ok_or(Failure::Memory)?,
        0,
    )?;
    paths
        .try_reserve_exact(end.paths.len())
        .map_err(|_| Failure::Memory)?;
    for path in &end.paths {
        let bytes = path
            .len
            .checked_mul(std::mem::size_of::<Step>())
            .ok_or(Failure::Memory)?;
        kernel.account(0, 0, bytes, path.len)?;
        let mut steps = Vec::new();
        steps
            .try_reserve_exact(path.len)
            .map_err(|_| Failure::Memory)?;
        let mut current = Some(path.as_ref());
        while let Some(history) = current {
            steps.push(history.snapshot());
            current = history.previous.as_deref();
        }
        steps.reverse();
        paths.push(steps);
    }
    Ok(Alignment {
        distance: end.value,
        paths,
        stats: kernel.stats,
    })
}
