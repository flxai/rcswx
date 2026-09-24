//! Coordinator-only preparation. Workers receive immutable views and own drafts.
use super::*;
use std::cell::Ref;
use std::mem::size_of;

pub(super) const ROUND_CELLS: usize = 1024;
const MIN_CELLS: usize = 32;
const MIN_WORK: usize = 512;

type Access = (usize, usize, bool);

pub(super) struct Round<'a> {
    pub matrix: &'a Matrix,
    pub tokens: (&'a [Token], &'a [Token]),
    pub diagonal: Diagonal,
    pub offset: usize,
    pub count: usize,
    pub start: (usize, usize),
    pub auxiliaries: [Option<&'a Matrix>; 3],
    pub fresh: bool,
}

pub(super) struct Evaluated {
    pub ranks: Vec<usize>,
    pub drafts: Vec<Option<cell_eval::Draft>>,
}

struct Prepared<'a> {
    ranks: Vec<usize>,
    guards: Vec<Ref<'a, Cell>>,
    tokens: Vec<(&'a Token, &'a Token)>,
    scratch: usize,
}

enum Fallback {
    Narrow,
    Alias,
    Scratch,
}
type Preparation<T> = std::result::Result<T, Fallback>;

fn buffer<T>(count: usize) -> Preparation<Vec<T>> {
    let mut values = Vec::new();
    values
        .try_reserve_exact(count)
        .map_err(|_| Fallback::Scratch)?;
    Ok(values)
}
fn cell_ref(matrix: &Matrix, i: isize, j: isize) -> Preparation<&CellRef> {
    at(at(matrix, i).map_err(|_| Fallback::Alias)?, j).map_err(|_| Fallback::Alias)
}
fn reserve(total: &mut usize, count: usize, bytes: usize, limit: usize) -> Preparation<()> {
    *total = count
        .checked_mul(bytes)
        .and_then(|extra| total.checked_add(extra))
        .ok_or(Fallback::Scratch)?;
    if *total > limit {
        return Err(Fallback::Scratch);
    }
    Ok(())
}

fn prepare<'a>(round: &Round<'a>, limit: usize, forced: bool) -> Preparation<Prepared<'a>> {
    if round.count < 2 || (!forced && round.count < MIN_CELLS) {
        return Err(Fallback::Narrow);
    }
    // Reserve all owned payload containers, even those allocated after join.
    // Backend thread stacks/bookkeeping are bounded separately by pool capacity.
    let mut scratch = 0;
    reserve(
        &mut scratch,
        round.count,
        size_of::<usize>()
            + 4 * size_of::<Ref<'_, Cell>>()
            + size_of::<(&Token, &Token)>()
            + size_of::<cell_eval::Input<'_>>()
            + size_of::<Option<cell_eval::Draft>>(),
        limit,
    )?;
    if !round.fresh {
        reserve(&mut scratch, round.count, 9 * size_of::<Access>(), limit)?;
    }
    let mut ranks = buffer(round.count)?;
    let mut guards = buffer(round.count * 4)?;
    let mut tokens = buffer(round.count)?;
    let mut accesses = if round.fresh {
        Vec::new()
    } else {
        buffer(round.count * 9)?
    };
    let mut work = 0_usize;
    for rank in 0..round.count {
        let (i, j) = round.diagonal.position(round.offset + rank);
        let destination = cell_ref(round.matrix, i as isize, j as isize)?;
        let data = destination.borrow();
        if !round.fresh {
            accesses.push((Rc::as_ptr(destination) as usize, rank, false));
            if i > 1 && j > 1 {
                let clean = cell_ref(round.matrix, i as isize - 1, j as isize - 1)?;
                accesses.push((Rc::as_ptr(clean) as usize, rank, true));
                for auxiliary in round.auxiliaries.into_iter().flatten() {
                    let clean = cell_ref(auxiliary, i as isize - 1, j as isize - 1)?;
                    accesses.push((Rc::as_ptr(clean) as usize, rank, true));
                }
            }
        }
        if !data.value.is_nan() {
            continue;
        }
        let one = round.tokens.0.get(i).ok_or(Fallback::Alias)?;
        let two = round.tokens.1.get(j).ok_or(Fallback::Alias)?;
        let previous = [
            cell_ref(round.matrix, i as isize - 1, j as isize)?,
            cell_ref(round.matrix, i as isize, j as isize - 1)?,
            cell_ref(round.matrix, i as isize - 1, j as isize - 1)?,
        ];
        if !round.fresh {
            accesses.push((Rc::as_ptr(destination) as usize, rank, true));
            for source in previous {
                if Rc::ptr_eq(source, destination) {
                    return Err(Fallback::Alias);
                }
                accesses.push((Rc::as_ptr(source) as usize, rank, false));
            }
        }
        let base = guards.len();
        guards.push(data);
        for source in previous {
            guards.push(source.borrow());
        }
        let destination = &guards[base];
        let presets = [&destination.top, &destination.left, &destination.corner];
        let mut histories = 0_usize;
        for (direction, preset) in presets.iter().enumerate() {
            let source = &guards[base + direction + 1];
            let candidates = if preset.is_empty() {
                source.paths.len()
            } else {
                preset.len()
            };
            if preset.is_empty() {
                reserve(&mut scratch, candidates, size_of::<f64>(), limit)?;
            }
            histories = histories.checked_add(candidates).ok_or(Fallback::Scratch)?;
            let closing = match direction {
                0 => one.boundary(),
                1 => two.boundary(),
                _ => one.boundary() && one.name == two.name,
            };
            let depth = if closing {
                source
                    .paths
                    .first()
                    .map_or(0, |path| path.len.min(128) / 16)
            } else {
                0
            };
            work = work.saturating_add(candidates.saturating_mul(1 + depth));
        }
        reserve(
            &mut scratch,
            histories,
            size_of::<History>() + 2 * size_of::<usize>(),
            limit,
        )?;
        let capacity = if histories == 0 {
            0
        } else {
            histories
                .max(4)
                .checked_next_power_of_two()
                .ok_or(Fallback::Scratch)?
        };
        reserve(&mut scratch, capacity, size_of::<Path>(), limit)?;
        ranks.push(rank);
        tokens.push((one, two));
    }
    if ranks.len() < 2 || (!forced && work < MIN_WORK) {
        return Err(Fallback::Narrow);
    }
    // Ordinary root blocks before any branch swap are freshly initialized,
    // unique-position matrices. Recursive/reused blocks require this identity
    // audit: coordinates alone cannot prove cleanup/destination independence.
    accesses.sort_unstable_by_key(|access| access.0);
    for group in accesses.chunk_by(|left, right| left.0 == right.0) {
        if group.iter().any(|entry| entry.2) && group.iter().any(|entry| entry.1 != group[0].1) {
            return Err(Fallback::Alias);
        }
    }
    Ok(Prepared {
        ranks,
        guards,
        tokens,
        scratch,
    })
}

pub(super) fn evaluate(kernel: &mut Kernel<'_>, round: Round<'_>) -> Result<Option<Evaluated>> {
    #[cfg(test)]
    let forced = kernel.execution.force_parallel;
    #[cfg(not(test))]
    let forced = false;
    let limit = kernel
        .execution
        .scratch_limit(kernel.stats.allocation_bytes);
    let prepared = match prepare(&round, limit, forced) {
        Ok(prepared) => prepared,
        Err(reason) => {
            match reason {
                Fallback::Narrow => kernel.execution.report.narrow_fallbacks += 1,
                Fallback::Alias => kernel.execution.report.alias_fallbacks += 1,
                Fallback::Scratch => kernel.execution.report.scratch_fallbacks += 1,
            }
            return Ok(None);
        }
    };
    let mut inputs = match buffer(prepared.ranks.len()) {
        Ok(inputs) => inputs,
        Err(_) => {
            kernel.execution.report.scratch_fallbacks += 1;
            return Ok(None);
        }
    };
    for (index, (&rank, &(one, two))) in prepared.ranks.iter().zip(&prepared.tokens).enumerate() {
        let cells = &prepared.guards[index * 4..index * 4 + 4];
        let (i, j) = round.diagonal.position(round.offset + rank);
        inputs.push(cell_eval::Input {
            destination: &cells[0],
            predecessors: [&cells[1], &cells[2], &cells[3]],
            first: one,
            second: two,
            position: (i + round.start.0, j + round.start.1),
        });
    }
    kernel.execution.report.scratch_peak_bytes = kernel
        .execution
        .report
        .scratch_peak_bytes
        .max(prepared.scratch);
    let check = &mut kernel.check;
    let stats = &kernel.stats;
    let drafts = kernel
        .execution
        .run(
            inputs.len(),
            &mut || check(stats, true).map_err(crate::error::Error::from),
            |index, cancelled| {
                cell_eval::Draft::evaluate(&inputs[index], || Ok(()), Some(cancelled))
            },
        )
        .map_err(Failure::Operational)?;
    kernel.execution.report.parallel_cells += prepared.ranks.len();
    // Guards remain coordinator-owned, then end before any logical publication.
    drop(inputs);
    drop(prepared.guards);
    Ok(Some(Evaluated {
        ranks: prepared.ranks,
        drafts,
    }))
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::execution::Workers;

    #[cfg(feature = "trace")]
    mod oracle {
        use crate as rcswx_core;
        include!("../tests/support/wavefront_oracle.rs");
    }

    fn tokens(size: usize) -> Vec<Token> {
        let mut tokens = vec![Token {
            id: 0,
            name: "start_node".into(),
            children: vec![],
            parent_arity: 0,
        }];
        tokens.extend((1..size).map(|index| Token {
            id: index as i64,
            name: "computation".into(),
            children: vec![format!("linear({index})")],
            parent_arity: 0,
        }));
        tokens
    }

    fn initialized(tokens: &[Token], until: usize) -> Matrix {
        let mut execution = Execution::serial();
        let mut check = |_: &Stats, _: bool| Ok(());
        let mut kernel = Kernel {
            original1: tokens,
            original2: tokens,
            collapse_corners: false,
            check: &mut check,
            execution: &mut execution,
            observe_allocations: false,
            stats: Stats::default(),
            #[cfg(feature = "trace")]
            trace: None,
        };
        let matrix = kernel.initialize(tokens, tokens, true).unwrap();
        for diagonal in 0..until {
            for i in 0..tokens.len() {
                if let Some(j) = diagonal.checked_sub(i).filter(|&j| j < tokens.len()) {
                    kernel
                        .fill_cell(&matrix, tokens, tokens, (i, j), (0, 0))
                        .unwrap();
                }
            }
        }
        matrix
    }

    fn round<'a>(matrix: &'a Matrix, tokens: &'a [Token], sum: usize, offset: usize) -> Round<'a> {
        Round {
            matrix,
            tokens: (tokens, tokens),
            diagonal: Diagonal {
                sum,
                low: sum.saturating_sub(tokens.len() - 1),
                high: sum.min(tokens.len() - 1),
                origin: (0, 0),
            },
            offset,
            count: 2,
            start: (0, 0),
            auxiliaries: [None, None, None],
            fresh: false,
        }
    }

    #[test]
    fn shared_destination_identity_requires_serial_publication() {
        let tokens = tokens(4);
        let mut matrix = initialized(&tokens, 3);
        assert!(prepare(&round(&matrix, &tokens, 3, 1), usize::MAX, true).is_ok());
        matrix[1][2] = matrix[2][1].clone();
        assert!(matches!(
            prepare(&round(&matrix, &tokens, 3, 1), usize::MAX, true),
            Err(Fallback::Alias)
        ));
    }

    #[test]
    fn cleanup_aliases_in_main_or_swap_storage_require_serial_publication() {
        let tokens = tokens(5);
        for swapped in [false, true] {
            let mut matrix = initialized(&tokens, 5);
            assert!(prepare(&round(&matrix, &tokens, 5, 1), usize::MAX, true).is_ok());
            let mut auxiliary = initialized(&tokens, 5);
            if swapped {
                auxiliary[2][1] = matrix[1][3].clone();
            } else {
                matrix[2][1] = matrix[1][3].clone();
            }
            let mut round = round(&matrix, &tokens, 5, 1);
            round.auxiliaries[0] = swapped.then_some(&auxiliary);
            assert!(matches!(
                prepare(&round, usize::MAX, true),
                Err(Fallback::Alias)
            ));
        }
    }

    #[test]
    fn exhausted_scratch_falls_back_without_truncating_paths() {
        let tokens = tokens(8);
        let expected = align(&tokens, &tokens, false, &mut || Ok(())).unwrap();
        let mut execution = Execution::new(Workers::new(2).unwrap()).unwrap();
        if execution.report.worker_limit < 2 {
            return;
        }
        execution.force_parallel = true;
        execution.allocation_limit(Some(0));
        let actual =
            align_with_execution(&tokens, &tokens, false, &mut || Ok(()), &mut execution).unwrap();
        let histories = |alignment: &Alignment| {
            alignment
                .paths
                .iter()
                .map(|path| {
                    path.iter()
                        .map(|step| {
                            (
                                step.id,
                                step.kind.name(),
                                step.node1_id,
                                step.node2_id,
                                step.i,
                                step.j,
                                step.value.to_bits(),
                                step.i_swapped,
                                step.j_swapped,
                            )
                        })
                        .collect::<Vec<_>>()
                })
                .collect::<Vec<_>>()
        };
        assert_eq!(actual.distance.to_bits(), expected.distance.to_bits());
        assert_eq!(histories(&actual), histories(&expected));
        assert_eq!(
            actual.stats.histories_created,
            expected.stats.histories_created
        );
        assert_eq!(
            actual.stats.allocation_bytes,
            expected.stats.allocation_bytes
        );
        assert_eq!(execution.report.parallel_rounds, 0);
        assert!(execution.report.scratch_fallbacks > 0);
    }

    #[cfg(feature = "trace")]
    #[test]
    fn forced_round_shapes_preserve_the_independent_frozen_oracle() {
        use crate::architecture::Limits;
        let probe = Execution::new(Workers::new(2).unwrap()).unwrap();
        if probe.report.worker_limit < 2 {
            return;
        }
        let frozen: serde_json::Value =
            serde_json::from_str(include_str!("../tests/fixtures/wavefront-oracle-v1.json"))
                .unwrap();
        let mut parallel_cells = 0;
        for case in frozen["cases"].as_array().unwrap() {
            let first = case["first"].as_str().unwrap();
            let second = case["second"].as_str().unwrap();
            let collapse = case["collapse_corners"].as_bool().unwrap();
            let max_events = case["max_events"].as_u64().unwrap() as usize;
            assert_eq!(
                oracle::capture(first, second, collapse, max_events),
                case["expected"]
            );
            for workers in [2, 4] {
                for size in [2, 3, 17, ROUND_CELLS] {
                    let mut execution = Execution::new(Workers::new(workers).unwrap()).unwrap();
                    execution.force_parallel = true;
                    execution.round_cells = size;
                    let actual = oracle::capture_with_execution(
                        first,
                        second,
                        collapse,
                        max_events,
                        &mut execution,
                        Limits::default(),
                    );
                    assert_eq!(
                        actual, case["expected"],
                        "{} workers={workers} round={size} collapse={collapse}",
                        case["name"]
                    );
                    parallel_cells += execution.report.parallel_cells;
                }
            }
        }
        assert!(
            parallel_cells > 0,
            "the oracle must exercise workers, not only serial fallback"
        );
    }

    #[cfg(feature = "trace")]
    #[test]
    fn resource_failures_keep_serial_partial_trace_and_error_precedence() {
        use crate::architecture::Limits;
        let probe = Execution::new(Workers::new(2).unwrap()).unwrap();
        if probe.report.worker_limit < 2 {
            return;
        }
        let frozen: serde_json::Value =
            serde_json::from_str(include_str!("../tests/fixtures/wavefront-oracle-v1.json"))
                .unwrap();
        let mut rounds = 0;
        for case in frozen["cases"].as_array().unwrap() {
            let stats = &case["expected"]["analysis"]["stats"];
            let Some(histories) = stats["histories_created"]
                .as_u64()
                .filter(|&count| count > 20)
            else {
                continue;
            };
            if case["max_events"].as_u64().unwrap() < 1000 {
                continue;
            }
            let cells = stats["cells_created"].as_u64().unwrap();
            for limits in [
                Limits {
                    max_work: Some((cells + histories / 2) as usize),
                    ..Limits::default()
                },
                Limits {
                    max_output: Some(1),
                    ..Limits::default()
                },
            ] {
                let first = case["first"].as_str().unwrap();
                let second = case["second"].as_str().unwrap();
                let collapse = case["collapse_corners"].as_bool().unwrap();
                let expected = oracle::capture_with_execution(
                    first,
                    second,
                    collapse,
                    100_000,
                    &mut Execution::serial(),
                    limits.clone(),
                );
                let mut execution = Execution::new(Workers::new(2).unwrap()).unwrap();
                execution.force_parallel = true;
                execution.round_cells = 3;
                let actual = oracle::capture_with_execution(
                    first,
                    second,
                    collapse,
                    100_000,
                    &mut execution,
                    limits,
                );
                assert!(
                    actual["analysis"].get("error").is_some(),
                    "the budget must reject the plan"
                );
                assert_eq!(actual, expected, "{}", case["name"]);
                rounds += execution.report.parallel_rounds;
            }
        }
        assert!(rounds > 0, "resource failures must cover dispatched work");
    }
}
