//! Bounded streaming calibration and explicit freeze control.

use std::error::Error;
use std::fmt;
use std::sync::atomic::Ordering;

use crate::bucket::{
    CalibrationEstimate, CalibrationTerminal, calibration_bounds, estimate_quantile,
};
use crate::init::{StageId, Tracegrams};
use crate::recorder::{
    CALIBRATION_COLLECTING, CALIBRATION_FREEZING, CALIBRATION_FROZEN, CalibrationPopulation,
    Segment,
};

/// Availability of a threshold in a freeze report.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
#[non_exhaustive]
pub enum CalibrationThresholdAvailability {
    /// A finite threshold was derived and published.
    Available,
    /// The stage had no predecessor-bearing calibration population.
    AbsentNoPredecessor,
    /// The nearest-rank estimate selected a terminal bucket.
    RangeInsufficient,
}

/// Freeze information for one calibration population.
#[derive(Clone, Copy, Debug, PartialEq)]
#[non_exhaustive]
pub struct CalibrationPopulationReport {
    /// Represented population.
    pub population: CalibrationPopulation,
    /// Population size observed by the freeze scan.
    pub samples: u128,
    /// Nearest-rank estimate, including terminal estimates; `None` when the
    /// previous-cumulative population is absent.
    pub estimate: Option<CalibrationEstimate>,
}

impl CalibrationPopulationReport {
    /// Returns threshold availability derived from the estimate.
    pub const fn availability(self) -> CalibrationThresholdAvailability {
        match self.estimate {
            None => CalibrationThresholdAvailability::AbsentNoPredecessor,
            Some(CalibrationEstimate {
                terminal: CalibrationTerminal::Finite,
                ..
            }) => CalibrationThresholdAvailability::Available,
            Some(_) => CalibrationThresholdAvailability::RangeInsufficient,
        }
    }
}

/// Calibration information for one stage.
#[derive(Clone, Copy, Debug, PartialEq)]
#[non_exhaustive]
pub struct StageCalibrationReport {
    /// Reported stage.
    pub stage: StageId,
    /// Local-latency threshold report.
    pub local: CalibrationPopulationReport,
    /// Cumulative-after threshold report.
    pub cumulative_after: CalibrationPopulationReport,
    /// Previous-cumulative threshold report.
    pub previous_cumulative: CalibrationPopulationReport,
}

/// Owned report from one relaxed freeze attempt.
#[derive(Clone, Debug, PartialEq)]
#[non_exhaustive]
pub struct FreezeReport {
    /// Stage reports in registration order.
    pub stages: Box<[StageCalibrationReport]>,
}

impl FreezeReport {
    /// Returns the report for `stage` when it belongs to this recorder.
    pub fn stage(&self, stage: StageId) -> Option<StageCalibrationReport> {
        self.stages
            .iter()
            .copied()
            .find(|entry| entry.stage == stage)
    }
}

/// A typed failure to freeze calibration.
#[derive(Clone, Debug, PartialEq)]
#[non_exhaustive]
pub enum FreezeError {
    /// A required or observed population has fewer samples than requested.
    ///
    /// Local and cumulative-after populations are required and must be
    /// nonempty. An empty previous-cumulative population is permitted; a
    /// nonempty one must reach the minimum.
    NotReady {
        /// First stage, in registration order, with an unready population.
        stage: StageId,
        /// Unready population.
        population: CalibrationPopulation,
        /// Samples observed by the relaxed scan.
        samples: u128,
        /// Requested minimum sample count.
        minimum: u64,
    },
    /// Another caller currently owns the freeze transition.
    AlreadyFreezing,
    /// Calibration has already frozen successfully.
    AlreadyFrozen,
    /// A quantile rank landed in a terminal bucket.
    CalibrationRangeInsufficient {
        /// Stage whose population selected a terminal bucket.
        stage: StageId,
        /// Population whose range was insufficient.
        population: CalibrationPopulation,
        /// Selected lower or upper terminal.
        terminal: CalibrationTerminal,
        /// Full retained relaxed report for the failed attempt.
        report: FreezeReport,
    },
}

impl fmt::Display for FreezeError {
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::NotReady {
                stage,
                population,
                samples,
                minimum,
            } => write!(
                formatter,
                "{stage:?} {population:?} has {samples} calibration samples; {minimum} required"
            ),
            Self::AlreadyFreezing => formatter.write_str("calibration is already freezing"),
            Self::AlreadyFrozen => formatter.write_str("calibration is already frozen"),
            Self::CalibrationRangeInsufficient {
                stage,
                population,
                terminal,
                ..
            } => write!(
                formatter,
                "calibration range is insufficient for {stage:?} {population:?}: {terminal:?}"
            ),
        }
    }
}

impl Error for FreezeError {}

#[derive(Clone, Copy, Debug, PartialEq)]
pub(crate) struct FrozenStageCalibration {
    report: StageCalibrationReport,
}

impl FrozenStageCalibration {
    pub(crate) const fn report(self) -> StageCalibrationReport {
        self.report
    }

    pub(crate) const fn local_threshold(self) -> u64 {
        self.report.local.estimate.unwrap().value.unwrap()
    }

    pub(crate) const fn cumulative_after_threshold(self) -> u64 {
        self.report
            .cumulative_after
            .estimate
            .unwrap()
            .value
            .unwrap()
    }

    pub(crate) const fn previous_cumulative_threshold(self) -> Option<u64> {
        match self.report.previous_cumulative.estimate {
            Some(estimate) => estimate.value,
            None => None,
        }
    }
}

struct PopulationScan {
    population: CalibrationPopulation,
    counts: Box<[u64]>,
    samples: u128,
}

struct StageScan {
    stage: StageId,
    local: PopulationScan,
    cumulative_after: PopulationScan,
    previous_cumulative: PopulationScan,
}

impl Tracegrams {
    /// Attempts the single explicit collecting-to-frozen transition for every
    /// registered stage.
    ///
    /// `minimum_samples` applies to each population. A zero minimum is
    /// permitted, but an empty required population still returns
    /// [`FreezeError::NotReady`] before this call claims the freeze
    /// transition.
    pub fn try_freeze_calibration(
        &self,
        minimum_samples: u64,
    ) -> Result<FreezeReport, FreezeError> {
        match self.inner.calibration_state.load(Ordering::Acquire) {
            CALIBRATION_COLLECTING => {}
            CALIBRATION_FREEZING => return Err(FreezeError::AlreadyFreezing),
            CALIBRATION_FROZEN => return Err(FreezeError::AlreadyFrozen),
            _ => unreachable!("calibration state has a fixed internal representation"),
        }

        if let Some(error) = first_unready(&self.scan_calibration(), minimum_samples) {
            return Err(error);
        }

        #[cfg(test)]
        test_hooks::run(&test_hooks::BEFORE_CLAIM);

        if let Err(observed) = self.inner.calibration_state.compare_exchange(
            CALIBRATION_COLLECTING,
            CALIBRATION_FREEZING,
            Ordering::AcqRel,
            Ordering::Acquire,
        ) {
            return Err(match observed {
                CALIBRATION_FREEZING => FreezeError::AlreadyFreezing,
                CALIBRATION_FROZEN => FreezeError::AlreadyFrozen,
                _ => unreachable!("calibration state has a fixed internal representation"),
            });
        }

        self.freeze_after_claim(minimum_samples)
    }

    fn freeze_after_claim(&self, minimum_samples: u64) -> Result<FreezeReport, FreezeError> {
        // Re-scan after claiming the transition; any readiness observed
        // before the claim may be stale. Writers that observed Collecting can
        // interleave their two or three increments with this relaxed scan
        // (first marks skip the previous-cumulative cell), so cells
        // may represent different effective request sets. The freeze contract
        // promises per-cell atomicity only: thresholds are statistical
        // estimates over approximately coincident populations, not a
        // cross-cell-consistent snapshot.
        let scans = self.scan_calibration();
        if let Some(error) = first_unready(&scans, minimum_samples) {
            self.restore_collecting();
            return Err(error);
        }

        let stage_reports = self.derive_stage_reports(&scans);
        if let Some((stage, population, terminal)) = first_terminal(&stage_reports) {
            self.restore_collecting();
            return Err(FreezeError::CalibrationRangeInsufficient {
                stage,
                population,
                terminal,
                report: FreezeReport {
                    stages: stage_reports.into_boxed_slice(),
                },
            });
        }

        // Online cells are written only in the frozen state, so they are still
        // zero here. Publish thresholds before the release store below exposes
        // the frozen state.
        #[cfg(test)]
        test_hooks::run(&test_hooks::BEFORE_PUBLISH);

        for report in &stage_reports {
            let published = FrozenStageCalibration { report: *report };
            self.inner.frozen_calibration[report.stage.index()]
                .set(published)
                .expect("only the single successful freeze publishes thresholds");
        }

        self.inner
            .calibration_state
            .store(CALIBRATION_FROZEN, Ordering::Release);
        Ok(FreezeReport {
            stages: stage_reports.into_boxed_slice(),
        })
    }

    pub(crate) fn is_frozen(&self) -> bool {
        self.inner.calibration_state.load(Ordering::Acquire) == CALIBRATION_FROZEN
    }

    pub(crate) fn frozen_calibration_report(&self) -> Option<FreezeReport> {
        if !self.is_frozen() {
            return None;
        }
        let stages = self
            .inner
            .frozen_calibration
            .iter()
            .filter_map(|published| published.get().copied())
            .map(FrozenStageCalibration::report)
            .collect::<Vec<_>>()
            .into_boxed_slice();
        Some(FreezeReport { stages })
    }

    fn scan_calibration(&self) -> Vec<StageScan> {
        self.stage_ids()
            .map(|stage| StageScan {
                stage,
                local: self.scan_population(stage, CalibrationPopulation::Local),
                cumulative_after: self
                    .scan_population(stage, CalibrationPopulation::CumulativeAfter),
                previous_cumulative: self
                    .scan_population(stage, CalibrationPopulation::PreviousCumulative),
            })
            .collect()
    }

    fn scan_population(&self, stage: StageId, population: CalibrationPopulation) -> PopulationScan {
        let range = self
            .inner
            .layout
            .segment(stage.index(), Segment::Calibration(population))
            .expect("registered stages have calibration storage");
        let counts = self.inner.counters[range]
            .iter()
            .map(|counter| counter.load(Ordering::Relaxed))
            .collect::<Vec<_>>()
            .into_boxed_slice();
        let samples = counts.iter().map(|count| u128::from(*count)).sum();
        PopulationScan {
            population,
            counts,
            samples,
        }
    }

    fn derive_stage_reports(&self, scans: &[StageScan]) -> Vec<StageCalibrationReport> {
        scans
            .iter()
            .map(|scan| StageCalibrationReport {
                stage: scan.stage,
                local: self.derive_population_report(&scan.local),
                cumulative_after: self.derive_population_report(&scan.cumulative_after),
                previous_cumulative: self.derive_population_report(&scan.previous_cumulative),
            })
            .collect()
    }

    /// Readiness has already rejected empty required populations, so only an
    /// absent previous-cumulative population yields no estimate.
    fn derive_population_report(&self, scan: &PopulationScan) -> CalibrationPopulationReport {
        CalibrationPopulationReport {
            population: scan.population,
            samples: scan.samples,
            estimate: estimate_quantile(
                &scan.counts,
                calibration_bounds(),
                self.inner.tail_quantile,
            ),
        }
    }

    fn restore_collecting(&self) {
        self.inner
            .calibration_state
            .store(CALIBRATION_COLLECTING, Ordering::Release);
    }
}

/// Returns the first population, in registration and population order, that
/// does not permit freezing.
fn first_unready(scans: &[StageScan], minimum: u64) -> Option<FreezeError> {
    let ready = |population: &PopulationScan, may_be_absent: bool| {
        if population.samples == 0 {
            may_be_absent
        } else {
            population.samples >= u128::from(minimum)
        }
    };
    scans.iter().find_map(|scan| {
        [
            (&scan.local, false),
            (&scan.cumulative_after, false),
            (&scan.previous_cumulative, true),
        ]
        .into_iter()
        .find(|(population, may_be_absent)| !ready(population, *may_be_absent))
        .map(|(population, _)| FreezeError::NotReady {
            stage: scan.stage,
            population: population.population,
            samples: population.samples,
            minimum,
        })
    })
}

fn first_terminal(
    reports: &[StageCalibrationReport],
) -> Option<(StageId, CalibrationPopulation, CalibrationTerminal)> {
    reports.iter().find_map(|stage| {
        [
            stage.local,
            stage.cumulative_after,
            stage.previous_cumulative,
        ]
        .into_iter()
        .find_map(|population| {
            let estimate = population.estimate?;
            (estimate.terminal != CalibrationTerminal::Finite).then_some((
                stage.stage,
                population.population,
                estimate.terminal,
            ))
        })
    })
}

// One-shot, thread-local hooks cannot suspend an unrelated parallel test or
// survive their worker thread. No hook storage or calls exist outside unit tests.
#[cfg(test)]
pub(crate) mod test_hooks {
    use std::cell::RefCell;
    use std::thread::LocalKey;

    pub(crate) type Hook = RefCell<Option<Box<dyn FnOnce()>>>;

    thread_local! {
        pub(crate) static BEFORE_CLAIM: Hook = RefCell::new(None);
        pub(crate) static BEFORE_PUBLISH: Hook = RefCell::new(None);
        pub(crate) static COLLECTING_MARK: Hook = RefCell::new(None);
    }

    pub(crate) fn run(hook: &'static LocalKey<Hook>) {
        let callback = hook.take();
        if let Some(callback) = callback {
            callback();
        }
    }
}

#[cfg(test)]
mod tests {
    use std::sync::mpsc::{self, Receiver, Sender};
    use std::thread::{self, JoinHandle, LocalKey};
    use std::time::Duration;

    use super::*;
    use crate::{CalibrationState, Outcome};

    // Channels (rather than sleeps) select the interleaving. Disconnection or
    // a timeout turns a broken branch into a failure instead of a hung suite.
    const WAIT: Duration = Duration::from_secs(10);

    struct Pause {
        reached: Receiver<()>,
        resume: Sender<()>,
    }

    fn paused_at<T: Send + 'static>(
        hook: &'static LocalKey<test_hooks::Hook>,
        action: impl FnOnce() -> T + Send + 'static,
    ) -> (Pause, JoinHandle<T>) {
        let (reached_tx, reached) = mpsc::channel();
        let (resume, resume_rx) = mpsc::channel();
        let worker = thread::spawn(move || {
            hook.set(Some(Box::new(move || {
                reached_tx.send(()).unwrap();
                resume_rx.recv_timeout(WAIT).unwrap();
            })));
            action()
        });
        (Pause { reached, resume }, worker)
    }

    fn warmed_pair() -> (Tracegrams, StageId, StageId) {
        let mut builder = Tracegrams::builder();
        let first = builder.stage("first").unwrap();
        let second = builder.stage("second").unwrap();
        let tracegrams = builder.build().unwrap();
        record_pair(&tracegrams, first, second);
        (tracegrams, first, second)
    }

    fn record_pair(tracegrams: &Tracegrams, first: StageId, second: StageId) {
        let mut context = tracegrams.start_manual();
        tracegrams.record_elapsed(&mut context, first, Duration::from_nanos(100));
        tracegrams.finish_manual(context, second, Duration::from_nanos(200), Outcome::Success);
    }

    #[test]
    fn losing_freezer_observes_frozen_at_compare_exchange() {
        let (tracegrams, _, _) = warmed_pair();
        let loser_recorder = tracegrams.clone();
        let (pause, loser) = paused_at(&test_hooks::BEFORE_CLAIM, move || {
            loser_recorder.try_freeze_calibration(1)
        });
        pause.reached.recv_timeout(WAIT).unwrap();
        let report = tracegrams.try_freeze_calibration(1).unwrap();
        pause.resume.send(()).unwrap();
        assert_eq!(loser.join().unwrap(), Err(FreezeError::AlreadyFrozen));
        assert_eq!(
            tracegrams.snapshot_relaxed().calibration_report(),
            Some(&report)
        );
    }

    #[test]
    fn competing_freezers_observe_freezing_at_load_and_compare_exchange() {
        let (tracegrams, _, _) = warmed_pair();
        let loser_recorder = tracegrams.clone();
        let (loser_pause, loser) = paused_at(&test_hooks::BEFORE_CLAIM, move || {
            loser_recorder.try_freeze_calibration(1)
        });
        loser_pause.reached.recv_timeout(WAIT).unwrap();

        let winner_recorder = tracegrams.clone();
        let (winner_pause, winner) = paused_at(&test_hooks::BEFORE_PUBLISH, move || {
            winner_recorder.try_freeze_calibration(1)
        });
        winner_pause.reached.recv_timeout(WAIT).unwrap();
        assert_eq!(
            tracegrams.try_freeze_calibration(1),
            Err(FreezeError::AlreadyFreezing)
        );
        loser_pause.resume.send(()).unwrap();
        assert_eq!(loser.join().unwrap(), Err(FreezeError::AlreadyFreezing));

        let snapshot = tracegrams.snapshot_relaxed();
        assert_eq!(snapshot.calibration_state(), CalibrationState::Freezing);
        assert_eq!(snapshot.calibration_report(), None);
        winner_pause.resume.send(()).unwrap();
        let report = winner.join().unwrap().unwrap();
        assert_eq!(
            tracegrams.snapshot_relaxed().calibration_report(),
            Some(&report)
        );
    }

    #[test]
    fn collecting_writer_can_increment_after_the_freeze_scan_and_publication() {
        let (tracegrams, first, second) = warmed_pair();
        let mut context = tracegrams.start_manual();
        tracegrams.record_elapsed(&mut context, first, Duration::from_nanos(100));
        let writer_recorder = tracegrams.clone();
        let (pause, writer) = paused_at(&test_hooks::COLLECTING_MARK, move || {
            writer_recorder.finish_manual(
                context,
                second,
                Duration::from_nanos(200),
                Outcome::Error,
            );
        });
        pause.reached.recv_timeout(WAIT).unwrap();
        let report = tracegrams.try_freeze_calibration(1).unwrap();
        let scanned = report.stage(second).unwrap();
        assert_eq!(scanned.local.samples, 1);
        assert_eq!(scanned.cumulative_after.samples, 1);
        assert_eq!(scanned.previous_cumulative.samples, 1);

        pause.resume.send(()).unwrap();
        writer.join().unwrap();
        let snapshot = tracegrams.snapshot_relaxed();
        let counts = snapshot.sample_counts(second).unwrap();
        assert_eq!(counts.local, 2);
        assert_eq!(counts.cumulative_after, 2);
        assert_eq!(counts.cause, 2);
        assert_eq!(counts.incoming, 2);
        assert_eq!(counts.calibration_local, 2);
        assert_eq!(counts.calibration_cumulative_after, 2);
        assert_eq!(counts.calibration_previous_cumulative, 2);
        assert_eq!(counts.online, 0);
        assert_eq!(snapshot.completion_counts(second).unwrap().success, 1);
        assert_eq!(snapshot.completion_counts(second).unwrap().error, 1);
        assert_eq!(snapshot.diagnostics().total(), 0);
        // Freeze reports retain their relaxed scan, not subsequent increments.
        assert_eq!(snapshot.calibration_report(), Some(&report));
    }

    #[test]
    fn writers_during_publication_keep_matrices_but_not_calibration_or_online() {
        let (tracegrams, first, second) = warmed_pair();
        let freezer_recorder = tracegrams.clone();
        let (pause, freezer) = paused_at(&test_hooks::BEFORE_PUBLISH, move || {
            freezer_recorder.try_freeze_calibration(1)
        });
        pause.reached.recv_timeout(WAIT).unwrap();
        let before = tracegrams.snapshot_relaxed();
        assert_eq!(before.calibration_state(), CalibrationState::Freezing);
        assert_eq!(before.calibration_report(), None);

        // Both kinds of context start during Freezing. Their predecessor mark
        // runs before publication and their finish runs after publication.
        tracegrams.inner.clock_readings.lock().unwrap().extend([
            (0, false),
            (100, false),
            (300, false),
            (400, false),
            (500, false),
            (700, false),
        ]);
        let mut manual = tracegrams.start_manual();
        let mut clocked = tracegrams.start();
        tracegrams.record_elapsed(&mut manual, first, Duration::from_nanos(100));
        tracegrams.mark(&mut clocked, first);
        record_pair(&tracegrams, first, second);
        let during = tracegrams.snapshot_relaxed();
        assert_eq!(during.calibration_state(), CalibrationState::Freezing);
        assert_eq!(during.calibration_report(), None);
        assert_eq!(
            during
                .diagnostics()
                .calibration_samples_skipped_while_freezing,
            4
        );
        assert_eq!(during.completion_counts(second).unwrap().success, 2);
        for stage in [first, second] {
            let counts = during.sample_counts(stage).unwrap();
            let old = before.sample_counts(stage).unwrap();
            assert_eq!(counts.calibration_local, old.calibration_local);
            assert_eq!(
                counts.calibration_cumulative_after,
                old.calibration_cumulative_after
            );
            assert_eq!(
                counts.calibration_previous_cumulative,
                old.calibration_previous_cumulative
            );
            assert_eq!(counts.online, 0);
        }
        assert_eq!(during.sample_counts(first).unwrap().local, 4);
        assert_eq!(during.sample_counts(second).unwrap().cause, 2);
        assert_eq!(during.sample_counts(second).unwrap().incoming, 2);

        pause.resume.send(()).unwrap();
        let report = freezer.join().unwrap().unwrap();
        tracegrams.finish_manual(manual, second, Duration::from_nanos(200), Outcome::Error);
        tracegrams.finish(clocked, second, Outcome::Error);
        let crossed = tracegrams.snapshot_relaxed();
        for stage in [first, second] {
            assert_eq!(crossed.sample_counts(stage).unwrap().online, 0);
        }

        // Fresh manual and clocked contexts see every published threshold.
        record_pair(&tracegrams, first, second);
        let mut clocked = tracegrams.start();
        tracegrams.mark(&mut clocked, first);
        tracegrams.finish(clocked, second, Outcome::Success);
        let after = tracegrams.snapshot_relaxed();
        assert_eq!(after.calibration_state(), CalibrationState::Frozen);
        assert_eq!(after.calibration_report(), Some(&report));
        for stage in [first, second] {
            let counts = after.sample_counts(stage).unwrap();
            assert_eq!(counts.local, 6);
            assert_eq!(counts.cumulative_after, 6);
            assert_eq!(counts.calibration_local, 1);
            assert_eq!(counts.calibration_cumulative_after, 1);
            assert_eq!(counts.online, 2);
        }
        let counts = after.sample_counts(second).unwrap();
        assert_eq!(counts.cause, 6);
        assert_eq!(counts.incoming, 6);
        assert_eq!(counts.calibration_previous_cumulative, 1);
        assert_eq!(after.completion_counts(second).unwrap().success, 4);
        assert_eq!(after.completion_counts(second).unwrap().error, 2);
        assert_eq!(
            after
                .diagnostics()
                .calibration_samples_skipped_while_freezing,
            4
        );
        assert_eq!(
            after
                .diagnostics()
                .online_samples_skipped_missing_previous_threshold,
            0
        );
        assert_eq!(after.diagnostics().total(), 4);
        assert!(tracegrams.inner.clock_readings.lock().unwrap().is_empty());
    }

    #[test]
    fn revalidation_rejects_an_absent_to_insufficient_previous_population_race() {
        let mut builder = Tracegrams::builder();
        let first = builder.stage("first").unwrap();
        let destination = builder.stage("destination").unwrap();
        let tracegrams = builder.build().unwrap();
        for stage in [first, first, destination, destination] {
            tracegrams.record_elapsed(
                &mut tracegrams.start_manual(),
                stage,
                Duration::from_nanos(200),
            );
        }
        assert_eq!(first_unready(&tracegrams.scan_calibration(), 2), None);

        let freezer_recorder = tracegrams.clone();
        let (pause, freezer) = paused_at(&test_hooks::BEFORE_CLAIM, move || {
            freezer_recorder.try_freeze_calibration(2)
        });
        pause.reached.recv_timeout(WAIT).unwrap();
        record_pair(&tracegrams, first, destination);
        pause.resume.send(()).unwrap();

        assert_eq!(
            freezer.join().unwrap(),
            Err(FreezeError::NotReady {
                stage: destination,
                population: CalibrationPopulation::PreviousCumulative,
                samples: 1,
                minimum: 2,
            })
        );
        assert_eq!(
            tracegrams.inner.calibration_state.load(Ordering::Acquire),
            CALIBRATION_COLLECTING
        );
        assert!(
            tracegrams
                .inner
                .frozen_calibration
                .iter()
                .all(|published| published.get().is_none())
        );
        record_pair(&tracegrams, first, destination);
        let retry = tracegrams.try_freeze_calibration(2).unwrap();
        assert_eq!(
            retry
                .stage(destination)
                .unwrap()
                .previous_cumulative
                .samples,
            2
        );
        assert_eq!(
            tracegrams.snapshot_relaxed().calibration_report(),
            Some(&retry)
        );
    }
}
