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
        let counts = (0..=calibration_bounds().len())
            .map(|bucket| {
                let index = self
                    .inner
                    .layout
                    .calibration(stage.index(), population, bucket)
                    .expect("registered stages and pinned buckets have storage");
                self.inner.counters[index].load(Ordering::Relaxed)
            })
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

#[cfg(test)]
mod tests {
    use std::time::Duration;

    use super::*;
    use crate::bucket::bucketize;

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

        tracegrams
            .inner
            .calibration_state
            .store(CALIBRATION_FREEZING, Ordering::Release);
        let previous_bucket = bucketize(100, calibration_bounds());
        let previous = tracegrams
            .inner
            .layout
            .calibration(
                destination.index(),
                CalibrationPopulation::PreviousCumulative,
                previous_bucket,
            )
            .unwrap();
        tracegrams.inner.increment(previous);

        assert_eq!(
            tracegrams.freeze_after_claim(2),
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
    }
}
