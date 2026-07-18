//! Bounded streaming calibration readiness and explicit freeze control.

use std::error::Error;
use std::fmt;
use std::sync::atomic::Ordering;

use crate::bucket::{QuantileEstimate, TerminalStatus, estimate_quantile};
use crate::init::{StageId, Tracegrams};
use crate::recorder::{
    CALIBRATION_COLLECTING, CALIBRATION_FREEZING, CALIBRATION_FROZEN, CalibrationDistribution,
};

/// Consistency guarantee attached to calibration scans and reports.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
#[non_exhaustive]
pub enum CalibrationConsistency {
    /// Cells were loaded atomically, but the scan has no single instant.
    Relaxed,
}

/// Readiness of one calibration population.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
#[non_exhaustive]
pub enum PopulationReadiness {
    /// The population has enough samples to derive a threshold.
    Ready {
        /// Samples observed by the relaxed scan.
        samples: u128,
        /// Required minimum sample count.
        minimum: u64,
    },
    /// The population has not reached its required minimum.
    Insufficient {
        /// Samples observed by the relaxed scan.
        samples: u128,
        /// Required minimum sample count.
        minimum: u64,
    },
    /// No predecessor-bearing marks have reached this stage.
    AbsentNoPredecessor,
}

impl PopulationReadiness {
    /// Returns the observed sample count.
    pub const fn samples(self) -> u128 {
        match self {
            Self::Ready { samples, .. } | Self::Insufficient { samples, .. } => samples,
            Self::AbsentNoPredecessor => 0,
        }
    }

    /// Returns whether this population permits freezing.
    pub const fn permits_freeze(self) -> bool {
        matches!(self, Self::Ready { .. } | Self::AbsentNoPredecessor)
    }
}

/// Per-population readiness for one stage.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub struct StageCalibrationReadiness {
    stage: StageId,
    local: PopulationReadiness,
    cumulative_after: PopulationReadiness,
    previous_cumulative: PopulationReadiness,
}

impl StageCalibrationReadiness {
    /// Returns the stage described by this entry.
    pub const fn stage(self) -> StageId {
        self.stage
    }

    /// Returns local-latency readiness.
    pub const fn local(self) -> PopulationReadiness {
        self.local
    }

    /// Returns cumulative-after readiness.
    pub const fn cumulative_after(self) -> PopulationReadiness {
        self.cumulative_after
    }

    /// Returns previous-cumulative readiness.
    pub const fn previous_cumulative(self) -> PopulationReadiness {
        self.previous_cumulative
    }

    /// Returns whether every required or observed population permits freezing.
    pub const fn is_ready(self) -> bool {
        self.local.permits_freeze()
            && self.cumulative_after.permits_freeze()
            && self.previous_cumulative.permits_freeze()
    }
}

/// Owned relaxed readiness scan for selected stages.
#[derive(Clone, Debug, Eq, PartialEq)]
pub struct CalibrationReadiness {
    minimum_samples: u64,
    consistency: CalibrationConsistency,
    stages: Box<[StageCalibrationReadiness]>,
}

impl CalibrationReadiness {
    /// Returns the requested per-population minimum.
    pub const fn minimum_samples(&self) -> u64 {
        self.minimum_samples
    }

    /// Returns the scan consistency.
    pub const fn consistency(&self) -> CalibrationConsistency {
        self.consistency
    }

    /// Returns stage entries in registration order.
    pub fn stages(&self) -> &[StageCalibrationReadiness] {
        &self.stages
    }

    /// Returns the entry for `stage` when it belongs to this scan.
    pub fn stage(&self, stage: StageId) -> Option<StageCalibrationReadiness> {
        self.stages
            .iter()
            .copied()
            .find(|entry| entry.stage == stage)
    }

    /// Returns whether every selected stage permits freezing.
    pub fn is_ready(&self) -> bool {
        self.stages.iter().all(|stage| stage.is_ready())
    }
}

/// Stages and minimum population size requested for an explicit freeze.
#[derive(Clone, Debug, Eq, PartialEq)]
pub struct FreezeCriteria {
    minimum_samples: u64,
    selection: FreezeSelection,
}

#[derive(Clone, Debug, Eq, PartialEq)]
enum FreezeSelection {
    All,
    Selected(Box<[StageId]>),
}

impl FreezeCriteria {
    /// Selects every registered stage.
    pub const fn all_stages(minimum_samples: u64) -> Self {
        Self {
            minimum_samples,
            selection: FreezeSelection::All,
        }
    }

    /// Selects only the supplied stages; excluded stages remain uncalibrated.
    pub fn for_stages(stages: &[StageId], minimum_samples: u64) -> Self {
        Self {
            minimum_samples,
            selection: FreezeSelection::Selected(stages.into()),
        }
    }

    /// Returns the requested per-population minimum.
    pub const fn minimum_samples(&self) -> u64 {
        self.minimum_samples
    }
}

/// Terminal status of a calibration quantile estimate.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
#[non_exhaustive]
pub enum CalibrationTerminal {
    /// The selected rank is below the finite calibration range.
    Lower,
    /// The selected rank has finite lower and upper bounds.
    Finite,
    /// The selected rank is above the finite calibration range.
    Upper,
}

/// One nearest-rank estimate from the pinned calibration grid.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct CalibrationEstimate {
    rank: u128,
    samples: u128,
    lower_bound: Option<u64>,
    upper_bound: Option<u64>,
    value: Option<u64>,
    max_relative_error: Option<f64>,
    terminal: CalibrationTerminal,
}

impl CalibrationEstimate {
    /// Returns the selected nearest rank.
    pub const fn rank(self) -> u128 {
        self.rank
    }

    /// Returns the scanned population size.
    pub const fn samples(self) -> u128 {
        self.samples
    }

    /// Returns the inclusive finite lower bound, when known.
    pub const fn lower_bound(self) -> Option<u64> {
        self.lower_bound
    }

    /// Returns the exclusive finite upper bound, when known.
    pub const fn upper_bound(self) -> Option<u64> {
        self.upper_bound
    }

    /// Returns the finite minimax threshold representative.
    pub const fn value(self) -> Option<u64> {
        self.value
    }

    /// Returns the finite bucket's maximum relative value error.
    pub const fn max_relative_error(self) -> Option<f64> {
        self.max_relative_error
    }

    /// Returns whether the selected rank is finite or terminal.
    pub const fn terminal(self) -> CalibrationTerminal {
        self.terminal
    }
}

impl From<QuantileEstimate> for CalibrationEstimate {
    fn from(estimate: QuantileEstimate) -> Self {
        Self {
            rank: estimate.selection.rank,
            samples: estimate.selection.samples,
            lower_bound: estimate.lower_bound,
            upper_bound: estimate.upper_bound,
            value: estimate.value,
            max_relative_error: estimate.max_relative_error,
            terminal: match estimate.terminal {
                TerminalStatus::Lower => CalibrationTerminal::Lower,
                TerminalStatus::Finite => CalibrationTerminal::Finite,
                TerminalStatus::Upper => CalibrationTerminal::Upper,
            },
        }
    }
}

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
pub struct CalibrationPopulationReport {
    population: crate::snapshot::CalibrationPopulation,
    samples: u128,
    availability: CalibrationThresholdAvailability,
    estimate: Option<CalibrationEstimate>,
}

impl CalibrationPopulationReport {
    /// Returns the represented population.
    pub const fn population(self) -> crate::snapshot::CalibrationPopulation {
        self.population
    }

    /// Returns the population size observed by the freeze scan.
    pub const fn samples(self) -> u128 {
        self.samples
    }

    /// Returns threshold availability.
    pub const fn availability(self) -> CalibrationThresholdAvailability {
        self.availability
    }

    /// Returns the nearest-rank estimate, including terminal estimates.
    pub const fn estimate(self) -> Option<CalibrationEstimate> {
        self.estimate
    }
}

/// Frozen calibration information for one selected stage.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct StageCalibrationReport {
    stage: StageId,
    local: CalibrationPopulationReport,
    cumulative_after: CalibrationPopulationReport,
    previous_cumulative: CalibrationPopulationReport,
}

impl StageCalibrationReport {
    /// Returns the selected stage.
    pub const fn stage(self) -> StageId {
        self.stage
    }

    /// Returns the local-latency threshold report.
    pub const fn local(self) -> CalibrationPopulationReport {
        self.local
    }

    /// Returns the cumulative-after threshold report.
    pub const fn cumulative_after(self) -> CalibrationPopulationReport {
        self.cumulative_after
    }

    /// Returns the previous-cumulative threshold report.
    pub const fn previous_cumulative(self) -> CalibrationPopulationReport {
        self.previous_cumulative
    }
}

/// Owned report from one relaxed freeze attempt.
#[derive(Clone, Debug, PartialEq)]
pub struct FreezeReport {
    epoch: Option<u16>,
    consistency: CalibrationConsistency,
    stages: Box<[StageCalibrationReport]>,
}

impl FreezeReport {
    /// Returns the published epoch, or `None` for a failed range check.
    pub const fn epoch(&self) -> Option<u16> {
        self.epoch
    }

    /// Returns the explicitly relaxed freeze consistency.
    pub const fn consistency(&self) -> CalibrationConsistency {
        self.consistency
    }

    /// Returns selected stage reports in registration order.
    pub fn stages(&self) -> &[StageCalibrationReport] {
        &self.stages
    }

    /// Returns the report for `stage` when it was selected.
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
    /// A selected-stage criterion contained no stages.
    NoStagesSelected,
    /// A selected stage does not belong to this recorder.
    InvalidStageSelection {
        /// Rejected stage identifier.
        stage: StageId,
    },
    /// A selected stage occurred more than once.
    DuplicateStageSelection {
        /// Repeated stage identifier.
        stage: StageId,
    },
    /// At least one required or observed population is not ready.
    NotReady {
        /// Relaxed readiness scan that prevented the freeze.
        readiness: CalibrationReadiness,
    },
    /// Another caller currently owns the freeze transition.
    AlreadyFreezing,
    /// Calibration has already frozen successfully.
    AlreadyFrozen {
        /// Existing frozen epoch.
        epoch: u16,
    },
    /// A selected quantile rank landed in a terminal bucket.
    CalibrationRangeInsufficient {
        /// Stage whose population selected a terminal bucket.
        stage: StageId,
        /// Population whose range was insufficient.
        population: crate::snapshot::CalibrationPopulation,
        /// Selected lower or upper terminal.
        terminal: CalibrationTerminal,
        /// Full retained relaxed report for the failed attempt.
        report: FreezeReport,
    },
    /// Calibration counters exceeded the supported nearest-rank total.
    CalibrationArithmeticOverflow {
        /// Stage whose population overflowed.
        stage: StageId,
        /// Population whose counter total overflowed.
        population: crate::snapshot::CalibrationPopulation,
    },
}

impl fmt::Display for FreezeError {
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::NoStagesSelected => formatter.write_str("freeze criteria selected no stages"),
            Self::InvalidStageSelection { stage } => {
                write!(formatter, "freeze criteria contains foreign {stage:?}")
            }
            Self::DuplicateStageSelection { stage } => {
                write!(formatter, "freeze criteria repeats {stage:?}")
            }
            Self::NotReady { .. } => {
                formatter.write_str("selected calibration populations are not ready")
            }
            Self::AlreadyFreezing => formatter.write_str("calibration is already freezing"),
            Self::AlreadyFrozen { epoch } => {
                write!(formatter, "calibration is already frozen at epoch {epoch}")
            }
            Self::CalibrationRangeInsufficient {
                stage,
                population,
                terminal,
                ..
            } => write!(
                formatter,
                "calibration range is insufficient for {stage:?} {population:?}: {terminal:?}"
            ),
            Self::CalibrationArithmeticOverflow { stage, population } => write!(
                formatter,
                "calibration counter total overflowed for {stage:?} {population:?}"
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
    population: crate::snapshot::CalibrationPopulation,
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
    /// Reports calibration readiness for every registered stage.
    pub fn calibration_readiness(&self, minimum_samples: u64) -> CalibrationReadiness {
        let stages = (0..self.inner.layout.stage_count())
            .map(|index| {
                // Registry construction bounds stage indices to 0..=63.
                #[allow(clippy::cast_possible_truncation)]
                let index = index as u8;
                StageId::new(self.inner.cookie, index)
            })
            .collect::<Vec<_>>();
        readiness_from_scans(&self.scan_calibration(&stages), minimum_samples)
    }

    /// Attempts the single explicit collecting-to-frozen transition.
    pub fn try_freeze_calibration(
        &self,
        criteria: FreezeCriteria,
    ) -> Result<FreezeReport, FreezeError> {
        let minimum_samples = criteria.minimum_samples;
        let stages = self.validate_selection(criteria)?;
        match self.inner.calibration_state.load(Ordering::Acquire) {
            CALIBRATION_COLLECTING => {}
            CALIBRATION_FREEZING => return Err(FreezeError::AlreadyFreezing),
            CALIBRATION_FROZEN => return Err(self.already_frozen()),
            _ => unreachable!("calibration state has a fixed internal representation"),
        }

        let readiness = readiness_from_scans(&self.scan_calibration(&stages), minimum_samples);
        if !readiness.is_ready() {
            return Err(FreezeError::NotReady { readiness });
        }

        if let Err(observed) = self.inner.calibration_state.compare_exchange(
            CALIBRATION_COLLECTING,
            CALIBRATION_FREEZING,
            Ordering::AcqRel,
            Ordering::Acquire,
        ) {
            return Err(match observed {
                CALIBRATION_FREEZING => FreezeError::AlreadyFreezing,
                CALIBRATION_FROZEN => self.already_frozen(),
                _ => unreachable!("calibration state has a fixed internal representation"),
            });
        }

        self.freeze_after_claim(&stages, minimum_samples)
    }

    fn freeze_after_claim(
        &self,
        stages: &[StageId],
        minimum_samples: u64,
    ) -> Result<FreezeReport, FreezeError> {
        let scans = self.scan_calibration(stages);
        let readiness = readiness_from_scans(&scans, minimum_samples);
        if !readiness.is_ready() {
            self.restore_collecting();
            return Err(FreezeError::NotReady { readiness });
        }

        let stage_reports = match self.derive_stage_reports(&scans) {
            Ok(reports) => reports,
            Err(error) => {
                self.restore_collecting();
                return Err(error);
            }
        };
        let failed_report = FreezeReport {
            epoch: None,
            consistency: CalibrationConsistency::Relaxed,
            stages: stage_reports.clone().into_boxed_slice(),
        };
        if let Some((stage, population, terminal)) = first_terminal(&stage_reports) {
            self.restore_collecting();
            return Err(FreezeError::CalibrationRangeInsufficient {
                stage,
                population,
                terminal,
                report: failed_report,
            });
        }

        for index in self.inner.layout.online_start()
            ..self.inner.layout.online_start() + self.inner.layout.online_counter_count()
        {
            self.inner.counters[index].store(0, Ordering::Relaxed);
        }
        for report in &stage_reports {
            let published = FrozenStageCalibration { report: *report };
            self.inner.frozen_calibration[report.stage.index()]
                .set(published)
                .expect("only the single successful freeze publishes thresholds");
        }

        let epoch = 1;
        self.inner.calibration_epoch.store(epoch, Ordering::Relaxed);
        self.inner
            .calibration_state
            .store(CALIBRATION_FROZEN, Ordering::Release);
        Ok(FreezeReport {
            epoch: Some(epoch),
            consistency: CalibrationConsistency::Relaxed,
            stages: stage_reports.into_boxed_slice(),
        })
    }

    pub(crate) fn current_calibration_epoch(&self) -> u16 {
        if self.inner.calibration_state.load(Ordering::Acquire) == CALIBRATION_FROZEN {
            self.inner.calibration_epoch.load(Ordering::Relaxed)
        } else {
            0
        }
    }

    pub(crate) fn frozen_calibration_report(&self) -> Option<FreezeReport> {
        if self.inner.calibration_state.load(Ordering::Acquire) != CALIBRATION_FROZEN {
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
        Some(FreezeReport {
            epoch: Some(self.inner.calibration_epoch.load(Ordering::Relaxed)),
            consistency: CalibrationConsistency::Relaxed,
            stages,
        })
    }

    fn validate_selection(&self, criteria: FreezeCriteria) -> Result<Vec<StageId>, FreezeError> {
        let mut stages = match criteria.selection {
            FreezeSelection::All => (0..self.inner.layout.stage_count())
                .map(|index| {
                    // Registry construction bounds stage indices to 0..=63.
                    #[allow(clippy::cast_possible_truncation)]
                    let index = index as u8;
                    StageId::new(self.inner.cookie, index)
                })
                .collect(),
            FreezeSelection::Selected(stages) => stages.into_vec(),
        };
        if stages.is_empty() {
            return Err(FreezeError::NoStagesSelected);
        }
        for (offset, stage) in stages.iter().copied().enumerate() {
            if stage.cookie() != self.inner.cookie
                || stage.index() >= self.inner.layout.stage_count()
            {
                return Err(FreezeError::InvalidStageSelection { stage });
            }
            if stages[..offset].contains(&stage) {
                return Err(FreezeError::DuplicateStageSelection { stage });
            }
        }
        stages.sort_unstable_by_key(|stage| stage.index());
        Ok(stages)
    }

    fn scan_calibration(&self, stages: &[StageId]) -> Vec<StageScan> {
        stages
            .iter()
            .copied()
            .map(|stage| StageScan {
                stage,
                local: self.scan_population(
                    stage,
                    crate::snapshot::CalibrationPopulation::Local,
                    CalibrationDistribution::Local,
                ),
                cumulative_after: self.scan_population(
                    stage,
                    crate::snapshot::CalibrationPopulation::CumulativeAfter,
                    CalibrationDistribution::CumulativeAfter,
                ),
                previous_cumulative: self.scan_population(
                    stage,
                    crate::snapshot::CalibrationPopulation::PreviousCumulative,
                    CalibrationDistribution::PreviousCumulative,
                ),
            })
            .collect()
    }

    fn scan_population(
        &self,
        stage: StageId,
        population: crate::snapshot::CalibrationPopulation,
        distribution: CalibrationDistribution,
    ) -> PopulationScan {
        let counts = (0..=self.inner.calibration_bounds.len())
            .map(|bucket| {
                let index = self
                    .inner
                    .layout
                    .calibration(stage.index(), distribution, bucket)
                    .expect("validated stages and pinned buckets have storage");
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

    fn derive_stage_reports(
        &self,
        scans: &[StageScan],
    ) -> Result<Vec<StageCalibrationReport>, FreezeError> {
        scans
            .iter()
            .map(|scan| {
                Ok(StageCalibrationReport {
                    stage: scan.stage,
                    local: self.derive_population_report(scan.stage, &scan.local, false)?,
                    cumulative_after: self.derive_population_report(
                        scan.stage,
                        &scan.cumulative_after,
                        false,
                    )?,
                    previous_cumulative: self.derive_population_report(
                        scan.stage,
                        &scan.previous_cumulative,
                        true,
                    )?,
                })
            })
            .collect()
    }

    fn derive_population_report(
        &self,
        stage: StageId,
        scan: &PopulationScan,
        absent_allowed: bool,
    ) -> Result<CalibrationPopulationReport, FreezeError> {
        if absent_allowed && scan.samples == 0 {
            return Ok(CalibrationPopulationReport {
                population: scan.population,
                samples: 0,
                availability: CalibrationThresholdAvailability::AbsentNoPredecessor,
                estimate: None,
            });
        }
        let estimate = estimate_quantile(
            &scan.counts,
            &self.inner.calibration_bounds,
            self.inner.tail_quantile,
        )
        .map_err(|_| FreezeError::CalibrationArithmeticOverflow {
            stage,
            population: scan.population,
        })?
        .ok_or(FreezeError::CalibrationArithmeticOverflow {
            stage,
            population: scan.population,
        })?;
        let estimate = CalibrationEstimate::from(estimate);
        let availability = if estimate.terminal == CalibrationTerminal::Finite {
            CalibrationThresholdAvailability::Available
        } else {
            CalibrationThresholdAvailability::RangeInsufficient
        };
        Ok(CalibrationPopulationReport {
            population: scan.population,
            samples: scan.samples,
            availability,
            estimate: Some(estimate),
        })
    }

    fn restore_collecting(&self) {
        self.inner
            .calibration_state
            .store(CALIBRATION_COLLECTING, Ordering::Release);
    }

    fn already_frozen(&self) -> FreezeError {
        FreezeError::AlreadyFrozen {
            epoch: self.inner.calibration_epoch.load(Ordering::Relaxed),
        }
    }
}

fn readiness_from_scans(scans: &[StageScan], minimum: u64) -> CalibrationReadiness {
    let stages = scans
        .iter()
        .map(|scan| StageCalibrationReadiness {
            stage: scan.stage,
            local: required_readiness(scan.local.samples, minimum),
            cumulative_after: required_readiness(scan.cumulative_after.samples, minimum),
            previous_cumulative: previous_readiness(scan.previous_cumulative.samples, minimum),
        })
        .collect::<Vec<_>>()
        .into_boxed_slice();
    CalibrationReadiness {
        minimum_samples: minimum,
        consistency: CalibrationConsistency::Relaxed,
        stages,
    }
}

fn required_readiness(samples: u128, minimum: u64) -> PopulationReadiness {
    if samples > 0 && samples >= u128::from(minimum) {
        PopulationReadiness::Ready { samples, minimum }
    } else {
        PopulationReadiness::Insufficient { samples, minimum }
    }
}

fn previous_readiness(samples: u128, minimum: u64) -> PopulationReadiness {
    if samples == 0 {
        PopulationReadiness::AbsentNoPredecessor
    } else {
        required_readiness(samples, minimum)
    }
}

fn first_terminal(
    reports: &[StageCalibrationReport],
) -> Option<(
    StageId,
    crate::snapshot::CalibrationPopulation,
    CalibrationTerminal,
)> {
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
        let _first = builder.stage("first").unwrap();
        let destination = builder.stage("destination").unwrap();
        let tracegrams = builder.build().unwrap();
        for _ in 0..2 {
            tracegrams.record_elapsed(
                &mut tracegrams.start_manual(),
                destination,
                Duration::from_nanos(200),
            );
        }
        let stages = [destination];
        let initial = readiness_from_scans(&tracegrams.scan_calibration(&stages), 2);
        assert!(initial.is_ready());
        assert_eq!(
            initial.stage(destination).unwrap().previous_cumulative(),
            PopulationReadiness::AbsentNoPredecessor
        );

        tracegrams
            .inner
            .calibration_state
            .store(CALIBRATION_FREEZING, Ordering::Release);
        let previous_bucket = bucketize(100, &tracegrams.inner.calibration_bounds);
        let previous = tracegrams
            .inner
            .layout
            .calibration(
                destination.index(),
                CalibrationDistribution::PreviousCumulative,
                previous_bucket,
            )
            .unwrap();
        tracegrams.inner.increment(previous);

        assert!(matches!(
            tracegrams.freeze_after_claim(&stages, 2),
            Err(FreezeError::NotReady { readiness })
                if readiness.stage(destination).unwrap().previous_cumulative()
                    == PopulationReadiness::Insufficient { samples: 1, minimum: 2 }
        ));
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

    #[test]
    fn successful_freeze_resets_every_online_counter_before_publication() {
        let mut builder = Tracegrams::builder();
        let stage = builder.stage("stage").unwrap();
        let tracegrams = builder.build().unwrap();
        tracegrams.record_elapsed(
            &mut tracegrams.start_manual(),
            stage,
            Duration::from_nanos(100),
        );
        for counter in 0..crate::recorder::ONLINE_COUNTERS_PER_STAGE {
            let index = tracegrams
                .inner
                .layout
                .online(stage.index(), counter)
                .unwrap();
            tracegrams.inner.counters[index]
                .store(u64::try_from(counter).unwrap() + 1, Ordering::Relaxed);
        }
        assert!(
            tracegrams
                .snapshot_relaxed()
                .sample_counts(stage)
                .unwrap()
                .online()
                > 0
        );

        tracegrams
            .try_freeze_calibration(FreezeCriteria::all_stages(1))
            .unwrap();

        assert_eq!(
            tracegrams
                .snapshot_relaxed()
                .sample_counts(stage)
                .unwrap()
                .online(),
            0
        );
    }
}
