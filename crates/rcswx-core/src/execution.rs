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
        count: usize,
        poll: &mut dyn FnMut() -> Result<()>,
        evaluate: impl Fn(usize, &std::sync::atomic::AtomicBool) -> T + Sync,
    ) -> Result<Vec<Option<T>>> {
        let state = self
            .native
            .as_mut()
            .ok_or_else(|| Error::Execution("parallel round has no native executor".into()))?;
        state.run(
            count,
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
    use std::panic::{AssertUnwindSafe, catch_unwind};
    use std::sync::atomic::{AtomicBool, AtomicU32, AtomicUsize, Ordering};
    use std::sync::{LazyLock, mpsc};
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
            count: usize,
            limit: usize,
            poll: &mut dyn FnMut() -> Result<()>,
            evaluate: impl Fn(usize, &AtomicBool) -> T + Sync,
            report: &mut Report,
        ) -> Result<Vec<Option<T>>> {
            validate()?;
            if count == 0 || limit < 2 {
                return Err(Error::Execution("invalid parallel round".into()));
            }
            if self
                .last_poll
                .is_none_or(|last| last.elapsed() >= POLL_INTERVAL)
            {
                self.poll(poll, report)?;
            }
            let jobs = count.min(limit);
            let mut results = Vec::new();
            results
                .try_reserve_exact(count)
                .map_err(|_| Error::Memory)?;
            results.resize_with(count, || None);
            let cancelled = AtomicBool::new(false);
            let panicked = AtomicBool::new(false);
            let active = AtomicUsize::new(0);
            let peak = AtomicUsize::new(0);
            let (notify, completed) = mpsc::sync_channel(jobs);
            let pool = self.pool;
            let mut interrupted = None;
            pool.in_place_scope(|scope| {
                let mut remaining = results.as_mut_slice();
                let mut first = 0;
                for job in 0..jobs {
                    let length = count / jobs + usize::from(job < count % jobs);
                    let (slots, rest) = remaining.split_at_mut(length);
                    remaining = rest;
                    let notify = notify.clone();
                    let (cancelled, panicked, active, peak, evaluate) =
                        (&cancelled, &panicked, &active, &peak, &evaluate);
                    scope.spawn(move |_| {
                        struct Active<'a>(&'a AtomicUsize);
                        impl Drop for Active<'_> {
                            fn drop(&mut self) {
                                self.0.fetch_sub(1, Ordering::Relaxed);
                            }
                        }
                        let outcome = catch_unwind(AssertUnwindSafe(|| {
                            let now = active.fetch_add(1, Ordering::Relaxed) + 1;
                            let _active = Active(active);
                            peak.fetch_max(now, Ordering::Relaxed);
                            for (offset, slot) in slots.iter_mut().enumerate() {
                                if cancelled.load(Ordering::Relaxed) {
                                    break;
                                }
                                *slot = Some(evaluate(first + offset, cancelled));
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
                    first += length;
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
            if results.iter().any(Option::is_none) {
                return Err(Error::Execution(
                    "native round returned incomplete output".into(),
                ));
            }
            Ok(results)
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
        let results = execution
            .run(2, &mut poll, |index, _| {
                assert_ne!(std::thread::current().id(), coordinator);
                barrier.wait();
                if index == 0 {
                    std::thread::sleep(Duration::from_millis(35));
                    Err::<usize, _>(Error::Index)
                } else {
                    Err(Error::EmptyMinimum)
                }
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
        let outcome = execution.run(2, &mut poll, |_, cancelled| {
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
        let failed = execution.run(31, &mut poll, |index, _| {
            started.fetch_add(1, Ordering::SeqCst);
            let _guard = Guard(&dropped);
            if index == 0 {
                panic!("intentional worker failure");
            }
            (0..=index).sum::<usize>()
        });
        assert!(matches!(failed, Err(Error::Execution(_))));
        assert_eq!(
            started.load(Ordering::SeqCst),
            dropped.load(Ordering::SeqCst)
        );
        let values = execution
            .run(31, &mut poll, |index, _| (0..=index).sum::<usize>())
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
                let values = execution
                    .run(97, &mut poll, |index, _| {
                        let now = active.fetch_add(1, Ordering::SeqCst) + 1;
                        peak.fetch_max(now, Ordering::SeqCst);
                        std::thread::sleep(Duration::from_millis(2));
                        let value = (0..=index + offset).sum::<usize>();
                        active.fetch_sub(1, Ordering::SeqCst);
                        value
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
