//! Native execution policy. Portable/default builds do not link a thread backend.
use crate::error::{Error, Result};
use serde::Serialize;

pub const PARALLEL_CAPABLE: bool = cfg!(all(feature = "parallel", not(target_arch = "wasm32")));

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct Workers(isize);
impl Workers {
    pub const SERIAL: Self = Self(1);

    pub fn new(value: isize) -> Result<Self> {
        if value == 0 || value < -1 {
            return Err(Error::InvalidInput(
                "workers must be -1 or a positive integer".into(),
            ));
        }
        let workers = Self(value);
        workers.validate()?;
        Ok(workers)
    }

    pub const fn requested(self) -> isize {
        self.0
    }

    /// Does not initialize a pool. Hosts call this before input preparation.
    pub fn validate(self) -> Result<()> {
        if self == Self::SERIAL {
            return Ok(());
        }
        if !PARALLEL_CAPABLE {
            return Err(Error::Execution(
                "native parallel execution is unavailable in this build".into(),
            ));
        }
        #[cfg(all(feature = "parallel", not(target_arch = "wasm32")))]
        native::validate()?;
        Ok(())
    }
}

/// Diagnostics are deliberately separate from portable algorithm statistics.
#[derive(Clone, Debug, Serialize)]
pub struct Report {
    pub parallel_capable: bool,
    pub requested_workers: isize,
    pub worker_limit: usize,
    pub pool_capacity: Option<usize>,
    pub parallel_rounds: usize,
    pub jobs: usize,
    pub peak_jobs: usize,
    pub parallel_cells: usize,
    pub serial_cells: usize,
    pub scratch_peak_bytes: usize,
    pub host_polls: usize,
    pub narrow_fallbacks: usize,
    pub alias_fallbacks: usize,
    pub scratch_fallbacks: usize,
}

/// Call-local policy. Non-Send host polling is supplied only by the coordinator.
pub struct Execution {
    pub(crate) report: Report,
    #[cfg(all(feature = "parallel", not(target_arch = "wasm32")))]
    native: Option<native::State>,
    #[cfg(all(test, feature = "parallel", not(target_arch = "wasm32")))]
    pub(crate) force_parallel: bool,
    #[cfg(all(test, feature = "parallel", not(target_arch = "wasm32")))]
    pub(crate) round_cells: usize,
}

impl Execution {
    pub fn serial() -> Self {
        Self {
            report: Report {
                parallel_capable: PARALLEL_CAPABLE,
                requested_workers: 1,
                worker_limit: 1,
                pool_capacity: None,
                parallel_rounds: 0,
                jobs: 0,
                peak_jobs: 0,
                parallel_cells: 0,
                serial_cells: 0,
                scratch_peak_bytes: 0,
                host_polls: 0,
                narrow_fallbacks: 0,
                alias_fallbacks: 0,
                scratch_fallbacks: 0,
            },
            #[cfg(all(feature = "parallel", not(target_arch = "wasm32")))]
            native: None,
            #[cfg(all(test, feature = "parallel", not(target_arch = "wasm32")))]
            force_parallel: false,
            #[cfg(all(test, feature = "parallel", not(target_arch = "wasm32")))]
            round_cells: 1024,
        }
    }

    pub fn new(workers: Workers) -> Result<Self> {
        workers.validate()?;
        #[cfg(all(feature = "parallel", not(target_arch = "wasm32")))]
        {
            let mut execution = Self::serial();
            if workers != Workers::SERIAL {
                let pool = native::pool()?;
                let capacity = pool.current_num_threads();
                execution.report.requested_workers = workers.requested();
                execution.report.pool_capacity = Some(capacity);
                execution.report.worker_limit = if workers.requested() == -1 {
                    capacity
                } else {
                    (workers.requested() as usize).min(capacity)
                };
                execution.native = Some(native::State {
                    pool,
                    last_poll: None,
                    allocation_limit: None,
                });
            }
            Ok(execution)
        }
        #[cfg(not(all(feature = "parallel", not(target_arch = "wasm32"))))]
        Ok(Self::serial())
    }

    pub fn report(&self) -> &Report {
        &self.report
    }
    pub fn into_report(self) -> Report {
        self.report
    }

    /// The quota is relative to kernel entry; canonical allocation accounting
    /// remains on the coordinator. Scratch is an additional bounded live reserve.
    pub(crate) fn allocation_limit(&mut self, _remaining: Option<usize>) {
        #[cfg(all(feature = "parallel", not(target_arch = "wasm32")))]
        if let Some(native) = &mut self.native {
            native.allocation_limit = _remaining;
        }
    }

    #[cfg(all(feature = "parallel", not(target_arch = "wasm32")))]
    pub(crate) fn scratch_limit(&self, canonical_bytes: usize) -> usize {
        const MAX_SCRATCH: usize = 8 * 1024 * 1024;
        self.native
            .as_ref()
            .and_then(|state| state.allocation_limit)
            .map_or(MAX_SCRATCH, |limit| {
                limit.saturating_sub(canonical_bytes).min(MAX_SCRATCH)
            })
    }

    #[cfg(all(feature = "parallel", not(target_arch = "wasm32")))]
    pub(crate) fn run<T: Send>(
        &mut self,
        slots: &mut [T],
        weights: &[usize],
        poll: &mut dyn FnMut() -> Result<()>,
        evaluate: impl Fn(usize, &mut T, &std::sync::atomic::AtomicBool) + Sync,
    ) -> Result<()> {
        let state = self
            .native
            .as_mut()
            .ok_or_else(|| Error::Execution("parallel round has no native executor".into()))?;
        state.run(
            slots,
            weights,
            self.report.worker_limit,
            poll,
            evaluate,
            &mut self.report,
        )
    }
}

/// Hosts use this around actual callbacks, including ordinary serial checkpoints.
/// Serial re-entry is permitted; a nested request for parallel work is rejected.
pub fn callback<T>(operation: impl FnOnce() -> T) -> T {
    #[cfg(all(feature = "parallel", not(target_arch = "wasm32")))]
    {
        native::callback(operation)
    }
    #[cfg(not(all(feature = "parallel", not(target_arch = "wasm32"))))]
    {
        operation()
    }
}

#[cfg(all(feature = "parallel", not(target_arch = "wasm32")))]
mod native {
    use super::*;
    use std::cell::Cell;
    use std::collections::VecDeque;
    use std::panic::{AssertUnwindSafe, catch_unwind};
    use std::sync::atomic::{AtomicBool, AtomicU32, AtomicUsize, Ordering};
    use std::sync::{LazyLock, Mutex, mpsc};
    use std::time::{Duration, Instant};

    static OWNER_PID: AtomicU32 = AtomicU32::new(0);
    static POOL: LazyLock<std::result::Result<rayon::ThreadPool, String>> = LazyLock::new(|| {
        let capacity = std::thread::available_parallelism()
            .map_or(1, |value| value.get())
            .min(rayon::max_num_threads());
        rayon::ThreadPoolBuilder::new()
            .num_threads(capacity)
            .thread_name(|index| format!("rcswx-{index}"))
            .build()
            .map_err(|error| error.to_string())
    });
    const POLL_INTERVAL: Duration = Duration::from_millis(10);
    thread_local! { static IN_CALLBACK: Cell<usize> = const { Cell::new(0) }; }

    pub(super) fn validate() -> Result<()> {
        // This load must precede every pool/lazy-init operation.
        let owner = OWNER_PID.load(Ordering::Acquire);
        if owner != 0 && owner != std::process::id() {
            return Err(Error::Execution("parallel execution after fork requires a fresh interpreter; workers=1 remains available".into()));
        }
        if IN_CALLBACK.get() != 0 {
            return Err(Error::Execution(
                "parallel callback re-entry is unsupported; workers=1 remains available".into(),
            ));
        }
        Ok(())
    }

    pub(super) fn callback<T>(operation: impl FnOnce() -> T) -> T {
        struct Guard;
        impl Drop for Guard {
            fn drop(&mut self) {
                IN_CALLBACK.set(IN_CALLBACK.get() - 1);
            }
        }
        IN_CALLBACK.set(IN_CALLBACK.get() + 1);
        let _guard = Guard;
        operation()
    }

    pub(super) fn pool() -> Result<&'static rayon::ThreadPool> {
        validate()?;
        let pid = std::process::id();
        if let Err(owner) = OWNER_PID.compare_exchange(0, pid, Ordering::AcqRel, Ordering::Acquire)
        {
            if owner != pid {
                return Err(Error::Execution(
                    "native executor belongs to another process".into(),
                ));
            }
        }
        LazyLock::force(&POOL).as_ref().map_err(|message| {
            Error::Execution(format!("unable to create native executor: {message}"))
        })
    }

    pub(super) struct State {
        pub pool: &'static rayon::ThreadPool,
        pub last_poll: Option<Instant>,
        pub allocation_limit: Option<usize>,
    }
    impl State {
        fn poll(
            &mut self,
            poll: &mut dyn FnMut() -> Result<()>,
            report: &mut Report,
        ) -> Result<()> {
            report.host_polls += 1;
            let result = catch_unwind(AssertUnwindSafe(|| callback(poll)));
            self.last_poll = Some(Instant::now());
            result.unwrap_or_else(|_| Err(Error::Execution("native host callback panicked".into())))
        }

        pub(super) fn run<T: Send>(
            &mut self,
            slots: &mut [T],
            weights: &[usize],
            limit: usize,
            poll: &mut dyn FnMut() -> Result<()>,
            evaluate: impl Fn(usize, &mut T, &AtomicBool) + Sync,
            report: &mut Report,
        ) -> Result<()> {
            validate()?;
            let count = slots.len();
            if count == 0 || limit < 2 || weights.len() != count {
                return Err(Error::Execution("invalid parallel round".into()));
            }
            if self
                .last_poll
                .is_none_or(|last| last.elapsed() >= POLL_INTERVAL)
            {
                self.poll(poll, report)?;
            }
            let jobs = count.min(limit);
            let cancelled = AtomicBool::new(false);
            let panicked = AtomicBool::new(false);
            let active = AtomicUsize::new(0);
            let peak = AtomicUsize::new(0);
            // More chunks than compute jobs lets faster workers take more work,
            // while the job count still enforces the caller's hard ceiling.
            let chunks = count.min(jobs.saturating_mul(8));
            let mut queue = VecDeque::new();
            queue.try_reserve_exact(chunks).map_err(|_| Error::Memory)?;
            let mut remaining = slots;
            let mut first = 0;
            let mut remaining_weight = weights
                .iter()
                .fold(0_usize, |sum, &weight| sum.saturating_add(weight));
            for chunk in 0..chunks {
                let left = chunks - chunk;
                let (length, weight) = if left == 1 {
                    (remaining.len(), remaining_weight)
                } else {
                    let target = remaining_weight.div_ceil(left);
                    let max_length = remaining.len() - (left - 1);
                    let mut length = 1;
                    let mut weight = weights[first];
                    while length < max_length {
                        let next = weight.saturating_add(weights[first + length]);
                        if weight.abs_diff(target) <= next.abs_diff(target) {
                            break;
                        }
                        weight = next;
                        length += 1;
                    }
                    (length, weight)
                };
                remaining_weight = remaining_weight.saturating_sub(weight);
                let (chunk, rest) = remaining.split_at_mut(length);
                remaining = rest;
                queue.push_back((first, chunk));
                first += length;
            }
            let queue = Mutex::new(queue);
            let (notify, completed) = mpsc::sync_channel(jobs);
            let pool = self.pool;
            let mut interrupted = None;
            pool.in_place_scope(|scope| {
                for _ in 0..jobs {
                    let notify = notify.clone();
                    let (cancelled, panicked, active, peak, evaluate, queue) =
                        (&cancelled, &panicked, &active, &peak, &evaluate, &queue);
                    scope.spawn(move |_| {
                        struct Active<'a>(&'a AtomicUsize);
                        impl Drop for Active<'_> {
                            fn drop(&mut self) {
                                self.0.fetch_sub(1, Ordering::Relaxed);
                            }
                        }
                        let outcome = catch_unwind(AssertUnwindSafe(|| {
                            let mut executing = None;
                            loop {
                                if cancelled.load(Ordering::Relaxed) {
                                    break;
                                }
                                // Release the queue guard before evaluating any cell.
                                // No cell/storage lock is acquired by a compute job.
                                let next = queue
                                    .lock()
                                    .expect("native chunk queue poisoned")
                                    .pop_front();
                                let Some((first, slots)) = next else {
                                    break;
                                };
                                if executing.is_none() {
                                    let now = active.fetch_add(1, Ordering::Relaxed) + 1;
                                    executing = Some(Active(active));
                                    peak.fetch_max(now, Ordering::Relaxed);
                                }
                                for (offset, slot) in slots.iter_mut().enumerate() {
                                    if cancelled.load(Ordering::Relaxed) {
                                        break;
                                    }
                                    evaluate(first + offset, slot, cancelled);
                                }
                            }
                        }));
                        if outcome.is_err() {
                            panicked.store(true, Ordering::Release);
                            cancelled.store(true, Ordering::Relaxed);
                        }
                        // Capacity equals the submitted job count, so completion
                        // cannot block a worker while the coordinator is in Python.
                        let _ = notify.send(());
                    });
                }
                drop(notify);
                let mut joined = 0;
                while joined < jobs {
                    let timeout = self.last_poll.map_or(POLL_INTERVAL, |last| {
                        POLL_INTERVAL.saturating_sub(last.elapsed())
                    });
                    match completed.recv_timeout(if interrupted.is_some() {
                        POLL_INTERVAL
                    } else {
                        timeout
                    }) {
                        Ok(()) => joined += 1,
                        Err(mpsc::RecvTimeoutError::Timeout) => {}
                        Err(mpsc::RecvTimeoutError::Disconnected) => {
                            cancelled.store(true, Ordering::Relaxed);
                            interrupted.get_or_insert_with(|| {
                                Error::Execution("native completion channel disconnected".into())
                            });
                            break;
                        }
                    }
                    if interrupted.is_none()
                        && self
                            .last_poll
                            .is_none_or(|last| last.elapsed() >= POLL_INTERVAL)
                    {
                        if let Err(error) = self.poll(poll, report) {
                            interrupted = Some(error);
                            cancelled.store(true, Ordering::Relaxed);
                        }
                    }
                }
            });
            // in_place_scope joins before any borrowed input/result is released,
            // including callback failure, cancellation, and caught worker panic.
            report.parallel_rounds += 1;
            report.jobs += jobs;
            report.peak_jobs = report.peak_jobs.max(peak.load(Ordering::Relaxed));
            if let Some(error) = interrupted {
                return Err(error);
            }
            if panicked.load(Ordering::Acquire) {
                return Err(Error::Execution("native compute worker panicked".into()));
            }
            Ok(())
        }
    }
}

#[cfg(all(test, feature = "parallel", not(target_arch = "wasm32")))]
mod tests {
    use super::*;
    use std::cell::Cell;
    use std::rc::Rc;
    use std::sync::Barrier;
    use std::sync::atomic::{AtomicBool, AtomicUsize, Ordering};
    use std::time::{Duration, Instant};

    #[test]
    fn jobs_overlap_but_polling_and_error_order_stay_on_the_coordinator() {
        let coordinator = std::thread::current().id();
        let calls = Rc::new(Cell::new(0));
        let mut poll = || {
            assert_eq!(std::thread::current().id(), coordinator);
            calls.set(calls.get() + 1);
            Ok(())
        };
        let mut execution = Execution::new(Workers::new(2).unwrap()).unwrap();
        if std::thread::available_parallelism().unwrap().get() < 2 {
            return;
        }
        assert_eq!(execution.report.worker_limit, 2);
        let barrier = Barrier::new(2);
        let mut results = vec![None; 2];
        execution
            .run(&mut results, &[1; 2], &mut poll, |index, slot, _| {
                assert_ne!(std::thread::current().id(), coordinator);
                barrier.wait();
                *slot = Some(if index == 0 {
                    std::thread::sleep(Duration::from_millis(35));
                    Err::<usize, _>(Error::Index)
                } else {
                    Err(Error::EmptyMinimum)
                });
            })
            .unwrap();
        assert_eq!(execution.report.peak_jobs, 2);
        assert!(calls.get() >= 2);
        assert_eq!(
            results,
            vec![Some(Err(Error::Index)), Some(Err(Error::EmptyMinimum))]
        );
    }

    #[test]
    fn waiting_callback_cancellation_joins_and_releases_all_jobs() {
        let polls = Cell::new(0);
        let mut poll = || {
            polls.set(polls.get() + 1);
            if polls.get() == 1 {
                Ok(())
            } else {
                Err(Error::Cancelled)
            }
        };
        let mut execution = Execution::new(Workers::new(2).unwrap()).unwrap();
        if execution.report.worker_limit < 2 {
            return;
        }
        let active = AtomicUsize::new(0);
        let timed_out = AtomicBool::new(false);
        let outcome = execution.run(&mut [(); 2], &[1; 2], &mut poll, |_, _, cancelled| {
            active.fetch_add(1, Ordering::SeqCst);
            let deadline = Instant::now() + Duration::from_secs(2);
            while !cancelled.load(Ordering::Relaxed) {
                if Instant::now() >= deadline {
                    timed_out.store(true, Ordering::Relaxed);
                    break;
                }
                std::thread::sleep(Duration::from_millis(1));
            }
            active.fetch_sub(1, Ordering::SeqCst);
        });
        assert_eq!(outcome, Err(Error::Cancelled));
        assert!(!timed_out.load(Ordering::Relaxed));
        assert_eq!(active.load(Ordering::SeqCst), 0);
    }

    #[test]
    fn a_panicking_worker_does_not_poison_later_rounds() {
        let mut poll = || Ok(());
        let mut execution = Execution::new(Workers::new(2).unwrap()).unwrap();
        if execution.report.worker_limit < 2 {
            return;
        }
        let dropped = AtomicUsize::new(0);
        let started = AtomicUsize::new(0);
        struct Guard<'a>(&'a AtomicUsize);
        impl Drop for Guard<'_> {
            fn drop(&mut self) {
                self.0.fetch_add(1, Ordering::SeqCst);
            }
        }
        let failed = execution.run(&mut [0; 31], &[1; 31], &mut poll, |index, slot, _| {
            started.fetch_add(1, Ordering::SeqCst);
            let _guard = Guard(&dropped);
            if index == 0 {
                panic!("intentional worker failure");
            }
            *slot = (0..=index).sum::<usize>();
        });
        assert!(matches!(failed, Err(Error::Execution(_))));
        assert_eq!(
            started.load(Ordering::SeqCst),
            dropped.load(Ordering::SeqCst)
        );
        let mut values = vec![None; 31];
        execution
            .run(&mut values, &[1; 31], &mut poll, |index, slot, _| {
                *slot = Some((0..=index).sum::<usize>())
            })
            .unwrap();
        assert_eq!(
            values,
            (0..31)
                .map(|index| Some(index * (index + 1) / 2))
                .collect::<Vec<_>>()
        );
    }

    #[test]
    fn concurrent_calls_share_the_pool_ceiling_and_keep_results_isolated() {
        let probe = Execution::new(Workers::new(-1).unwrap()).unwrap();
        let capacity = probe.report.worker_limit;
        if capacity < 2 {
            return;
        }
        let active = AtomicUsize::new(0);
        let peak = AtomicUsize::new(0);
        let start = Barrier::new(2);
        std::thread::scope(|scope| {
            let run = |offset| {
                let mut poll = || Ok(());
                let mut execution = Execution::new(Workers::new(isize::MAX).unwrap()).unwrap();
                start.wait();
                let mut values = vec![None; 97];
                execution
                    .run(&mut values, &[1; 97], &mut poll, |index, slot, _| {
                        let now = active.fetch_add(1, Ordering::SeqCst) + 1;
                        peak.fetch_max(now, Ordering::SeqCst);
                        std::thread::sleep(Duration::from_millis(2));
                        let value = (0..=index + offset).sum::<usize>();
                        active.fetch_sub(1, Ordering::SeqCst);
                        *slot = Some(value);
                    })
                    .unwrap();
                assert!(execution.report.peak_jobs <= capacity);
                assert_eq!(
                    values,
                    (0..97)
                        .map(|index| {
                            let index = index + offset;
                            Some(index * (index + 1) / 2)
                        })
                        .collect::<Vec<_>>()
                );
            };
            let first = scope.spawn(move || run(0));
            let second = scope.spawn(move || run(100));
            first.join().unwrap();
            second.join().unwrap();
        });
        assert!(peak.load(Ordering::SeqCst) <= capacity);
        assert_eq!(active.load(Ordering::SeqCst), 0);
    }

    #[test]
    fn concentrated_work_overlaps_without_reordering_results() {
        let mut execution = Execution::new(Workers::new(2).unwrap()).unwrap();
        if execution.report.worker_limit < 2 {
            return;
        }
        let entered = AtomicUsize::new(0);
        let mut slots = [0; 8];
        execution
            .run(
                &mut slots,
                &[1000, 1000, 1, 1, 1, 1, 1, 1],
                &mut || Ok(()),
                |index, slot, _| {
                    if index < 2 {
                        entered.fetch_add(1, Ordering::SeqCst);
                        let deadline = Instant::now() + Duration::from_secs(2);
                        while entered.load(Ordering::SeqCst) < 2 {
                            assert!(
                                Instant::now() < deadline,
                                "heavy cells were serialized in one job"
                            );
                            std::thread::sleep(Duration::from_millis(1));
                        }
                    }
                    *slot = index * index;
                },
            )
            .unwrap();
        assert_eq!(slots, [0, 1, 4, 9, 16, 25, 36, 49]);
        assert_eq!(execution.report.peak_jobs, 2);
    }

    #[test]
    fn callback_guard_allows_serial_and_recovers_after_unwind() {
        let result = std::panic::catch_unwind(|| {
            callback(|| {
                assert!(Workers::new(1).is_ok());
                assert!(matches!(Workers::new(2), Err(Error::Execution(_))));
                panic!("intentional callback failure");
            })
        });
        assert!(result.is_err());
        assert!(Workers::new(2).is_ok());
    }
}
