//! Private identities follow observation order; addresses never leave this module.
use super::*;
use crate::trace::{Recorder, number};
use serde_json::{Value, json};
use std::rc::Weak;

struct Identity<W> {
    weak: W,
    id: usize,
    revision: usize,
    value: Value,
    computed: bool,
}
pub(super) struct Observer<'a> {
    pub recorder: &'a mut Recorder,
    cells: HashMap<usize, Identity<Weak<RefCell<Cell>>>>,
    histories: HashMap<usize, Identity<HistoryWeak<History>>>,
    next_cell: usize,
    next_history: usize,
    next_problem: usize,
    stack: Vec<usize>,
}
impl<'a> Observer<'a> {
    pub fn new(recorder: &'a mut Recorder) -> Self {
        Self {
            recorder,
            cells: HashMap::new(),
            histories: HashMap::new(),
            next_cell: 0,
            next_history: 0,
            next_problem: 0,
            stack: Vec::new(),
        }
    }
    pub fn active(&self) -> bool {
        self.recorder.full()
    }
    fn history(&mut self, path: &Path) -> usize {
        let mut pending = Vec::new();
        let mut next = Some(path.clone());
        while let Some(current) = next {
            next = current.previous.clone();
            pending.push(current);
        }
        let mut previous = None;
        let mut last = 0;
        for current in pending.into_iter().rev() {
            if !self.active() {
                return last;
            }
            let key = HistoryRef::as_ptr(&current) as usize;
            let step = current.snapshot();
            let value = json!({"previous":previous,"len":current.len,"step":{"id":step.id,"kind":step.kind.name(),"node1_id":step.node1_id.map(|x|x.to_string()),"node2_id":step.node2_id.map(|x|x.to_string()),"i":step.i,"j":step.j,"value":number(step.value,false),"i_swapped":step.i_swapped,"j_swapped":step.j_swapped}});
            let valid = self.histories.get(&key).is_some_and(|entry| {
                entry
                    .weak
                    .upgrade()
                    .is_some_and(|p| HistoryRef::ptr_eq(&p, &current))
            });
            if !valid {
                let id = self.next_history;
                self.next_history += 1;
                self.histories.insert(
                    key,
                    Identity {
                        weak: HistoryRef::downgrade(&current),
                        id,
                        revision: 0,
                        value: Value::Null,
                        computed: false,
                    },
                );
            }
            let entry = self.histories.get_mut(&key).unwrap();
            if entry.value != value {
                let revision = entry.revision;
                entry.revision += 1;
                entry.value = value.clone();
                self.recorder.emit(
                    "history",
                    || json!({"id":entry.id,"revision":revision,"value":value}),
                );
            }
            last = entry.id;
            previous = Some(last);
        }
        last
    }
    pub fn cell(&mut self, cell: &CellRef) -> usize {
        self.observe_cell(cell, false)
    }
    fn observe_cell(&mut self, cell: &CellRef, computed: bool) -> usize {
        if !self.active() {
            return 0;
        }
        let data = cell.borrow();
        let histories = data
            .paths
            .iter()
            .map(|p| self.history(p))
            .collect::<Vec<_>>();
        if !self.active() {
            return 0;
        }
        let key = Rc::as_ptr(cell) as usize;
        let valid = self
            .cells
            .get(&key)
            .is_some_and(|entry| entry.weak.upgrade().is_some_and(|p| Rc::ptr_eq(&p, cell)));
        if !valid {
            let id = self.next_cell;
            self.next_cell += 1;
            self.cells.insert(
                key,
                Identity {
                    weak: Rc::downgrade(cell),
                    id,
                    revision: 0,
                    value: Value::Null,
                    computed: false,
                },
            );
        }
        let entry = self.cells.get_mut(&key).unwrap();
        entry.computed |= computed || !data.value.is_nan();
        let values = |v: &[f64]| v.iter().map(|&n| number(n, false)).collect::<Vec<_>>();
        let value = json!({"value":number(data.value,!entry.computed),"top":values(&data.top),
            "left":values(&data.left),"corner":values(&data.corner),"histories":histories});
        if entry.value != value {
            let revision = entry.revision;
            entry.revision += 1;
            entry.value = value.clone();
            self.recorder.emit(
                "cell",
                || json!({"id":entry.id,"revision":revision,"value":value}),
            );
        }
        entry.id
    }
    pub fn matrix(&mut self, matrix: &Matrix, phase: &str) {
        if !self.active() {
            return;
        }
        let mut rows = Vec::new();
        for row in matrix {
            let mut ids = Vec::new();
            for cell in row {
                if !self.active() {
                    return;
                }
                ids.push(self.cell(cell));
            }
            rows.push(ids);
        }
        self.recorder.emit(
            "matrix",
            || json!({"subproblem":self.stack.last(),"phase":phase,"cells":rows}),
        );
    }
    pub fn begin(
        &mut self,
        matrix: &Matrix,
        positions: &(Vec<usize>, Vec<usize>),
        start_i: usize,
        start_j: usize,
    ) {
        if !self.active() {
            return;
        }
        let id = self.next_problem;
        self.next_problem += 1;
        self.recorder.emit("subproblem",||json!({"id":id,"parent":self.stack.last(),"rows":matrix.len(),"columns":matrix.first().map_or(0,Vec::len),"start_i":start_i,"start_j":start_j,"first_indices":positions.0,"second_indices":positions.1}));
        self.stack.push(id);
        self.matrix(matrix, "enter");
    }
    pub fn end(&mut self, matrix: &Matrix) {
        if self.active() {
            self.matrix(matrix, "complete");
            self.recorder
                .emit("subproblem_end", || json!({"id":self.stack.last()}));
        }
        self.stack.pop();
    }
    pub fn computed(&mut self, matrix: &Matrix, i: usize, j: usize) {
        if !self.active() {
            return;
        }
        let predecessor = [
            (i as isize - 1, j as isize),
            (i as isize, j as isize - 1),
            (i as isize - 1, j as isize - 1),
        ]
        .into_iter()
        .map(|(i, j)| cell(matrix, i, j).ok().map(|c| self.cell(&c)))
        .collect::<Vec<_>>();
        let destination = self.observe_cell(&matrix[i][j], true);
        self.recorder.emit("computed",||json!({"subproblem":self.stack.last(),"i":i,"j":j,"cell":destination,"predecessors":predecessor}));
    }
    pub fn retained(&mut self, paths: &[Path]) {
        if !self.active() {
            return;
        }
        let ids = paths
            .iter()
            .map(|path| self.history(path))
            .collect::<Vec<_>>();
        self.recorder
            .emit("retained_histories", || json!({"histories":ids}));
    }
    pub fn history_clone(&mut self, source: &Path, target: &Path) {
        if !self.active() {
            return;
        }
        let source = self.history(source);
        let target = self.history(target);
        self.recorder
            .emit("history_clone", || json!({"source":source,"target":target}));
    }
    pub fn cell_clone(&mut self, source: &CellRef, target: &CellRef) {
        if !self.active() {
            return;
        }
        let source = self.cell(source);
        let target = self.cell(target);
        self.recorder
            .emit("cell_clone", || json!({"source":source,"target":target}));
    }
}
/// Apply the exact same slice permutation as the token branch-order search.
pub(super) fn positions(
    source: &[usize],
    start: usize,
    end: usize,
    mid: Option<usize>,
    swapped: bool,
) -> Vec<usize> {
    if swapped {
        if let Some(mid) = mid {
            let mut out = vec![source[start]];
            out.extend_from_slice(
                &source[(mid + 1).min(source.len())..(end - 1).min(source.len())],
            );
            out.push(source[mid]);
            out.extend_from_slice(&source[(start + 1).min(source.len())..mid.min(source.len())]);
            out.push(source[end - 1]);
            return out;
        }
    }
    source[start..end.min(source.len())].to_vec()
}
