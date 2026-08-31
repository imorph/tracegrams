//! Owned relaxed scans of recorder counters.

use std::error::Error;
use std::fmt;
use std::sync::atomic::Ordering;

use crate::bucket::BUCKETS;
use crate::calibration::FreezeReport;
use crate::init::{MemoryEstimate, RegistryCookie, StageId, Tracegrams};
use crate::recorder::{
    CalibrationDistribution, DiagnosticCounter, FixedStorageLayout, ONLINE_COUNTERS_PER_STAGE,
};

const MATRIX_CELLS: usize = BUCKETS * BUCKETS;

/// Consistency guarantee attached to read-plane data.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
#[non_exhaustive]
pub enum Consistency {
    /// Cells were loaded atomically, but the scan has no single instant.
    Relaxed,
}

/// Whether calibrated-online counters can be subtracted across a window.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
#[non_exhaustive]
pub enum OnlineDeltaAvailability {
    /// Both endpoints belong to the same online calibration epoch.
    SameEpoch {
        /// The shared epoch.
        epoch: u16,
    },
    /// The endpoints belong to different online calibration epochs.
    EpochMismatch {
        /// Epoch observed in the earlier snapshot.
        earlier: u16,
        /// Epoch observed in the later snapshot.
        later: u16,
    },
}

/// A typed failure to subtract two snapshots.
#[derive(Clone, Debug, Eq, PartialEq)]
#[non_exhaustive]
pub enum DeltaError {
    /// The snapshots came from different recorder registries.
    RegistryMismatch,
    /// A supposedly later monotonic counter was smaller than its earlier value.
    CounterUnderflow {
        /// Value observed in the earlier snapshot.
        earlier: u64,
        /// Value observed in the later snapshot.
        later: u64,
    },
}

impl fmt::Display for DeltaError {
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::RegistryMismatch => {
                formatter.write_str("snapshots belong to different registries")
            }
            Self::CounterUnderflow { earlier, later } => write!(
                formatter,
                "later counter value {later} is below earlier value {earlier}"
            ),
        }
    }
}

impl Error for DeltaError {}

/// Recorder calibration lifecycle state observed by a snapshot.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
#[non_exhaustive]
pub enum CalibrationState {
    /// Calibration distributions are collecting samples.
    Collecting,
    /// One control-plane caller is deriving calibration thresholds.
    Freezing,
    /// Calibration thresholds have been published.
    Frozen,
}

impl CalibrationState {
    fn from_raw(raw: u8) -> Self {
        match raw {
            0 => Self::Collecting,
            1 => Self::Freezing,
            2 => Self::Frozen,
            _ => unreachable!("calibration state is only written by the recorder lifecycle"),
        }
    }
}

/// A calibration distribution stored for each stage.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
#[non_exhaustive]
pub enum CalibrationPopulation {
    /// Local latency at the stage.
    Local,
    /// Cumulative latency after the stage.
    CumulativeAfter,
    /// Cumulative latency before predecessor-bearing marks.
    PreviousCumulative,
}

impl CalibrationPopulation {
    const fn distribution(self) -> CalibrationDistribution {
        match self {
            Self::Local => CalibrationDistribution::Local,
            Self::CumulativeAfter => CalibrationDistribution::CumulativeAfter,
            Self::PreviousCumulative => CalibrationDistribution::PreviousCumulative,
        }
    }
}

/// Per-population sample totals observed for one stage.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub struct SampleCounts {
    local: u128,
    cumulative_after: u128,
    cause: u128,
    incoming: u128,
    calibration_local: u128,
    calibration_cumulative_after: u128,
    calibration_previous_cumulative: u128,
    online: u128,
}

impl SampleCounts {
    /// Returns ordinary local-distribution samples.
    pub const fn local(self) -> u128 {
        self.local
    }

    /// Returns ordinary cumulative-after-distribution samples.
    pub const fn cumulative_after(self) -> u128 {
        self.cumulative_after
    }

    /// Returns predecessor-bearing cause-matrix samples.
    pub const fn cause(self) -> u128 {
        self.cause
    }

    /// Returns predecessor-bearing incoming-matrix samples.
    pub const fn incoming(self) -> u128 {
        self.incoming
    }

    /// Returns local calibration samples.
    pub const fn calibration_local(self) -> u128 {
        self.calibration_local
    }

    /// Returns cumulative-after calibration samples.
    pub const fn calibration_cumulative_after(self) -> u128 {
        self.calibration_cumulative_after
    }

    /// Returns previous-cumulative calibration samples.
    pub const fn calibration_previous_cumulative(self) -> u128 {
        self.calibration_previous_cumulative
    }

    /// Returns calibrated-online truth-table samples.
    pub const fn online(self) -> u128 {
        self.online
    }
}

/// Coarse availability of each score path's raw inputs.
///
/// Individual scores may still be unavailable because their denominator is
/// zero or because a relaxed scan observed only part of a checkpoint.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub struct ScoreAvailability {
    matrix_derived: bool,
    calibrated_online: bool,
}

impl ScoreAvailability {
    /// Returns whether both destination matrices contain predecessor samples.
    pub const fn matrix_derived(self) -> bool {
        self.matrix_derived
    }

    /// Returns whether frozen online truth-table samples are present.
    pub const fn calibrated_online(self) -> bool {
        self.calibrated_online
    }
}

/// Snapshot-visible hot-path anomaly and rejection counters.
#[derive(Clone, Copy, Debug, Default, Eq, PartialEq)]
pub struct Diagnostics {
    invalid_stage_marks: u64,
    invalid_context_marks: u64,
    non_monotonic_marks: u64,
    clock_regressions: u64,
    latency_overflows: u64,
    cumulative_overflows: u64,
    calibration_samples_skipped_while_freezing: u64,
    online_samples_skipped_missing_previous_threshold: u64,
}

impl Diagnostics {
    /// Returns marks carrying a stage from another registry.
    pub const fn invalid_stage_marks(self) -> u64 {
        self.invalid_stage_marks
    }

    /// Returns marks carrying a context from another registry.
    pub const fn invalid_context_marks(self) -> u64 {
        self.invalid_context_marks
    }

    /// Returns repeated or decreasing stage marks.
    pub const fn non_monotonic_marks(self) -> u64 {
        self.non_monotonic_marks
    }

    /// Returns rejected clock readings earlier than the context timestamp.
    pub const fn clock_regressions(self) -> u64 {
        self.clock_regressions
    }

    /// Returns local durations outside the exact `u64` nanosecond range.
    pub const fn latency_overflows(self) -> u64 {
        self.latency_overflows
    }

    /// Returns cumulative-duration additions that overflowed `u64`.
    pub const fn cumulative_overflows(self) -> u64 {
        self.cumulative_overflows
    }

    /// Returns calibration samples skipped while a freeze was in progress.
    pub const fn calibration_samples_skipped_while_freezing(self) -> u64 {
        self.calibration_samples_skipped_while_freezing
    }

    /// Returns online samples skipped because no previous threshold exists.
    pub const fn online_samples_skipped_missing_previous_threshold(self) -> u64 {
        self.online_samples_skipped_missing_previous_threshold
    }

    /// Returns the sum of all diagnostic counters without `u64` overflow.
    pub const fn total(self) -> u128 {
        self.invalid_stage_marks as u128
            + self.invalid_context_marks as u128
            + self.non_monotonic_marks as u128
            + self.clock_regressions as u128
            + self.latency_overflows as u128
            + self.cumulative_overflows as u128
            + self.calibration_samples_skipped_while_freezing as u128
            + self.online_samples_skipped_missing_previous_threshold as u128
    }
}

/// Owned metadata for one registered stage.
#[derive(Clone, Debug, Eq, PartialEq)]
pub struct StageMetadata {
    id: StageId,
    name: Box<str>,
}

impl StageMetadata {
    /// Returns the registry-branded stage identifier.
    pub const fn id(&self) -> StageId {
        self.id
    }

    /// Returns the registered stage name.
    pub fn name(&self) -> &str {
        &self.name
    }
}

/// Success and error completions observed for one stage.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub struct CompletionCounts {
    success: u64,
    error: u64,
}

impl CompletionCounts {
    /// Returns successful completions.
    pub const fn success(self) -> u64 {
        self.success
    }

    /// Returns error completions.
    pub const fn error(self) -> u64 {
        self.error
    }
}

/// An owned scan of one recorder's atomic counters.
///
/// Each cell was loaded atomically, but concurrent checkpoints may be only
/// partly represented. The snapshot remains valid after all recorder handles
/// are dropped.
#[derive(Clone, Debug)]
pub struct Snapshot {
    cookie: RegistryCookie,
    stages: Box<[StageMetadata]>,
    layout: FixedStorageLayout,
    counters: Box<[u64]>,
    bucket_bounds: Box<[u64]>,
    calibration_bucket_bounds: Box<[u64]>,
    tail_quantile: f64,
    calibration_state: CalibrationState,
    calibration_epoch: u16,
    calibration_report: Option<FreezeReport>,
    memory_budget_bytes: usize,
    memory_estimate: MemoryEstimate,
    consistency: Consistency,
}

impl Snapshot {
    /// Returns the explicitly weak consistency guarantee for this scan.
    pub const fn consistency(&self) -> Consistency {
        self.consistency
    }

    /// Returns owned metadata for all stages in registration order.
    pub fn stages(&self) -> &[StageMetadata] {
        &self.stages
    }

    /// Returns the registered name when `stage` belongs to this snapshot.
    pub fn stage_name(&self, stage: StageId) -> Option<&str> {
        let stage = self.stage_index(stage)?;
        Some(self.stages[stage].name())
    }

    /// Returns the pinned ordinary bucket bounds; each bound starts the next bucket.
    pub fn bucket_bounds(&self) -> &[u64] {
        &self.bucket_bounds
    }

    /// Returns the pinned calibration-grid bounds, using the same convention.
    pub fn calibration_bucket_bounds(&self) -> &[u64] {
        &self.calibration_bucket_bounds
    }

    /// Returns the configured calibration tail quantile.
    pub const fn tail_quantile(&self) -> f64 {
        self.tail_quantile
    }

    /// Returns the observed calibration lifecycle state.
    pub const fn calibration_state(&self) -> CalibrationState {
        self.calibration_state
    }

    /// Returns the observed calibration epoch.
    pub const fn calibration_epoch(&self) -> u16 {
        self.calibration_epoch
    }

    /// Returns the immutable successful freeze report after publication.
    pub const fn calibration_report(&self) -> Option<&FreezeReport> {
        self.calibration_report.as_ref()
    }

    /// Returns the recorder's configured memory budget.
    pub const fn memory_budget_bytes(&self) -> usize {
        self.memory_budget_bytes
    }

    /// Returns the recorder's itemized owned-memory estimate.
    pub const fn memory_estimate(&self) -> MemoryEstimate {
        self.memory_estimate
    }

    /// Returns the local-latency distribution for `stage`.
    pub fn local_counts(&self, stage: StageId) -> Option<&[u64]> {
        self.distribution(stage, |layout, stage| layout.local(stage, 0))
    }

    /// Returns the cumulative-latency-after distribution for `stage`.
    pub fn cumulative_counts(&self, stage: StageId) -> Option<&[u64]> {
        self.distribution(stage, |layout, stage| layout.cumulative(stage, 0))
    }

    /// Returns the destination-indexed cause matrix in row-major order.
    ///
    /// Rows are previous-cumulative buckets and columns are local buckets.
    pub fn cause_counts(&self, destination: StageId) -> Option<&[u64]> {
        self.matrix(destination, |layout, stage| layout.cause(stage, 0, 0))
    }

    /// Returns the destination-indexed incoming matrix in row-major order.
    ///
    /// Rows are previous-cumulative buckets and columns are cumulative-after
    /// buckets. The first registered stage has no incoming matrix.
    pub fn incoming_counts(&self, destination: StageId) -> Option<&[u64]> {
        self.matrix(destination, |layout, stage| layout.incoming(stage, 0, 0))
    }

    /// Returns a calibration distribution for `stage`.
    pub fn calibration_counts(
        &self,
        stage: StageId,
        population: CalibrationPopulation,
    ) -> Option<&[u64]> {
        let stage = self.stage_index(stage)?;
        let start = self
            .layout
            .calibration(stage, population.distribution(), 0)?;
        self.counters
            .get(start..start + self.calibration_bucket_bounds.len() + 1)
    }

    pub(crate) fn online_counts(&self, stage: StageId) -> Option<&[u64]> {
        let stage = self.stage_index(stage)?;
        let start = self.layout.online(stage, 0)?;
        self.counters.get(start..start + ONLINE_COUNTERS_PER_STAGE)
    }

    /// Returns totals for every stored sample population at `stage`.
    pub fn sample_counts(&self, stage: StageId) -> Option<SampleCounts> {
        self.stage_index(stage)?;
        let local = sum(self.local_counts(stage)?);
        let cumulative_after = sum(self.cumulative_counts(stage)?);
        let cause = sum(self.cause_counts(stage)?);
        let incoming = self.incoming_counts(stage).map_or(0, sum);
        let calibration_local = sum(self.calibration_counts(stage, CalibrationPopulation::Local)?);
        let calibration_cumulative_after =
            sum(self.calibration_counts(stage, CalibrationPopulation::CumulativeAfter)?);
        let calibration_previous_cumulative =
            sum(self.calibration_counts(stage, CalibrationPopulation::PreviousCumulative)?);
        let online = sum(self.online_counts(stage)?);

        Some(SampleCounts {
            local,
            cumulative_after,
            cause,
            incoming,
            calibration_local,
            calibration_cumulative_after,
            calibration_previous_cumulative,
            online,
        })
    }

    /// Returns coarse score-path input availability for `stage`.
    pub fn score_availability(&self, stage: StageId) -> Option<ScoreAvailability> {
        let samples = self.sample_counts(stage)?;
        Some(ScoreAvailability {
            matrix_derived: samples.cause > 0 && samples.incoming > 0,
            calibrated_online: self.calibration_state == CalibrationState::Frozen
                && samples.online > 0,
        })
    }

    /// Returns terminal completion counts for `stage`.
    pub fn completion_counts(&self, stage: StageId) -> Option<CompletionCounts> {
        let stage = self.stage_index(stage)?;
        let success = self.layout.completion(stage, false)?;
        let error = self.layout.completion(stage, true)?;
        Some(CompletionCounts {
            success: self.counters[success],
            error: self.counters[error],
        })
    }

    /// Returns all snapshot-visible anomaly and rejection counters.
    pub fn diagnostics(&self) -> Diagnostics {
        let count = |diagnostic| self.counters[self.layout.diagnostic(diagnostic)];
        Diagnostics {
            invalid_stage_marks: count(DiagnosticCounter::InvalidStageMarks),
            invalid_context_marks: count(DiagnosticCounter::InvalidContextMarks),
            non_monotonic_marks: count(DiagnosticCounter::NonMonotonicMarks),
            clock_regressions: count(DiagnosticCounter::ClockRegressions),
            latency_overflows: count(DiagnosticCounter::LatencyOverflows),
            cumulative_overflows: count(DiagnosticCounter::CumulativeOverflows),
            calibration_samples_skipped_while_freezing: count(
                DiagnosticCounter::CalibrationSamplesSkippedWhileFreezing,
            ),
            online_samples_skipped_missing_previous_threshold: count(
                DiagnosticCounter::OnlineSamplesSkippedMissingPreviousThreshold,
            ),
        }
    }

    /// Purely subtracts `earlier` from this later snapshot.
    ///
    /// When the endpoints belong to different calibration epochs, online
    /// counters are zeroed instead of subtracted and the window reports
    /// [`OnlineDeltaAvailability::EpochMismatch`].
    pub fn delta(&self, earlier: &Self) -> Result<DeltaSnapshot, DeltaError> {
        if self.cookie != earlier.cookie {
            return Err(DeltaError::RegistryMismatch);
        }
        let same_epoch = self.calibration_epoch == earlier.calibration_epoch;
        let online_start = self.layout.online_start();
        let online_end = online_start + self.layout.online_counter_count();
        let counters = self
            .counters
            .iter()
            .zip(&earlier.counters)
            .enumerate()
            .map(|(index, (later, earlier))| {
                if !same_epoch && (online_start..online_end).contains(&index) {
                    return Ok(0);
                }
                later
                    .checked_sub(*earlier)
                    .ok_or(DeltaError::CounterUnderflow {
                        earlier: *earlier,
                        later: *later,
                    })
            })
            .collect::<Result<Vec<_>, _>>()?
            .into_boxed_slice();
        let online_delta_availability = if same_epoch {
            OnlineDeltaAvailability::SameEpoch {
                epoch: self.calibration_epoch,
            }
        } else {
            OnlineDeltaAvailability::EpochMismatch {
                earlier: earlier.calibration_epoch,
                later: self.calibration_epoch,
            }
        };
        let snapshot = Snapshot {
            cookie: self.cookie,
            stages: self.stages.clone(),
            layout: self.layout,
            counters,
            bucket_bounds: self.bucket_bounds.clone(),
            calibration_bucket_bounds: self.calibration_bucket_bounds.clone(),
            tail_quantile: self.tail_quantile,
            calibration_state: self.calibration_state,
            calibration_epoch: self.calibration_epoch,
            calibration_report: self.calibration_report.clone(),
            memory_budget_bytes: self.memory_budget_bytes,
            memory_estimate: self.memory_estimate,
            consistency: Consistency::Relaxed,
        };

        Ok(DeltaSnapshot {
            snapshot,
            online_delta_availability,
        })
    }

    fn stage_index(&self, stage: StageId) -> Option<usize> {
        (stage.cookie() == self.cookie && stage.index() < self.stages.len()).then(|| stage.index())
    }

    fn distribution(
        &self,
        stage: StageId,
        start: impl FnOnce(FixedStorageLayout, usize) -> Option<usize>,
    ) -> Option<&[u64]> {
        let stage = self.stage_index(stage)?;
        let start = start(self.layout, stage)?;
        self.counters.get(start..start + BUCKETS)
    }

    fn matrix(
        &self,
        stage: StageId,
        start: impl FnOnce(FixedStorageLayout, usize) -> Option<usize>,
    ) -> Option<&[u64]> {
        let stage = self.stage_index(stage)?;
        let start = start(self.layout, stage)?;
        self.counters.get(start..start + MATRIX_CELLS)
    }
}

/// Owned counter differences for a window between two relaxed snapshots.
///
/// Cross-epoch online counters are zeroed, not subtracted.
#[derive(Clone, Debug)]
pub struct DeltaSnapshot {
    snapshot: Snapshot,
    online_delta_availability: OnlineDeltaAvailability,
}

impl DeltaSnapshot {
    /// Returns the explicitly weak consistency guarantee for this window.
    pub const fn consistency(&self) -> Consistency {
        self.snapshot.consistency()
    }

    /// Returns online-counter epoch comparability for this window.
    pub const fn online_delta_availability(&self) -> OnlineDeltaAvailability {
        self.online_delta_availability
    }

    /// Returns owned metadata for all stages in registration order.
    pub fn stages(&self) -> &[StageMetadata] {
        self.snapshot.stages()
    }

    /// Returns the registered name when `stage` belongs to this window.
    pub fn stage_name(&self, stage: StageId) -> Option<&str> {
        self.snapshot.stage_name(stage)
    }

    /// Returns the pinned ordinary bucket bounds; each bound starts the next bucket.
    pub fn bucket_bounds(&self) -> &[u64] {
        self.snapshot.bucket_bounds()
    }

    /// Returns the pinned calibration-grid bounds, using the same convention.
    pub fn calibration_bucket_bounds(&self) -> &[u64] {
        self.snapshot.calibration_bucket_bounds()
    }

    /// Returns the configured calibration tail quantile.
    pub const fn tail_quantile(&self) -> f64 {
        self.snapshot.tail_quantile()
    }

    /// Returns the calibration state observed at the later endpoint.
    pub const fn calibration_state(&self) -> CalibrationState {
        self.snapshot.calibration_state()
    }

    /// Returns the calibration epoch observed at the later endpoint.
    pub const fn calibration_epoch(&self) -> u16 {
        self.snapshot.calibration_epoch()
    }

    /// Returns the successful freeze report observed at the later endpoint.
    pub const fn calibration_report(&self) -> Option<&FreezeReport> {
        self.snapshot.calibration_report()
    }

    /// Returns the recorder's configured memory budget.
    pub const fn memory_budget_bytes(&self) -> usize {
        self.snapshot.memory_budget_bytes()
    }

    /// Returns the recorder's itemized owned-memory estimate.
    pub const fn memory_estimate(&self) -> MemoryEstimate {
        self.snapshot.memory_estimate()
    }

    /// Returns the local-latency distribution delta for `stage`.
    pub fn local_counts(&self, stage: StageId) -> Option<&[u64]> {
        self.snapshot.local_counts(stage)
    }

    /// Returns the cumulative-after distribution delta for `stage`.
    pub fn cumulative_counts(&self, stage: StageId) -> Option<&[u64]> {
        self.snapshot.cumulative_counts(stage)
    }

    /// Returns the destination-indexed cause-matrix delta in row-major order.
    pub fn cause_counts(&self, destination: StageId) -> Option<&[u64]> {
        self.snapshot.cause_counts(destination)
    }

    /// Returns the destination-indexed incoming-matrix delta in row-major order.
    pub fn incoming_counts(&self, destination: StageId) -> Option<&[u64]> {
        self.snapshot.incoming_counts(destination)
    }

    /// Returns a calibration-distribution delta for `stage`.
    pub fn calibration_counts(
        &self,
        stage: StageId,
        population: CalibrationPopulation,
    ) -> Option<&[u64]> {
        self.snapshot.calibration_counts(stage, population)
    }

    pub(crate) fn online_counts(&self, stage: StageId) -> Option<&[u64]> {
        self.snapshot.online_counts(stage)
    }

    /// Returns totals for every stored sample population in this window.
    pub fn sample_counts(&self, stage: StageId) -> Option<SampleCounts> {
        self.snapshot.sample_counts(stage)
    }

    /// Returns coarse score-path input availability for this window.
    pub fn score_availability(&self, stage: StageId) -> Option<ScoreAvailability> {
        let mut availability = self.snapshot.score_availability(stage)?;
        if matches!(
            self.online_delta_availability,
            OnlineDeltaAvailability::EpochMismatch { .. }
        ) {
            availability.calibrated_online = false;
        }
        Some(availability)
    }

    /// Returns terminal completion deltas for `stage`.
    pub fn completion_counts(&self, stage: StageId) -> Option<CompletionCounts> {
        self.snapshot.completion_counts(stage)
    }

    /// Returns all anomaly and rejection counter deltas.
    pub fn diagnostics(&self) -> Diagnostics {
        self.snapshot.diagnostics()
    }
}

fn sum(counts: &[u64]) -> u128 {
    counts.iter().map(|count| u128::from(*count)).sum()
}

impl Tracegrams {
    /// Loads each counter atomically into an owned relaxed snapshot.
    ///
    /// The scan does not synchronize concurrent checkpoints into a
    /// single-instant cut; a racing checkpoint may be only partly visible.
    pub fn snapshot_relaxed(&self) -> Snapshot {
        let counters = self
            .inner
            .counters
            .iter()
            .map(|counter| counter.load(Ordering::Relaxed))
            .collect::<Vec<_>>()
            .into_boxed_slice();
        let stages = self
            .inner
            .stage_names
            .iter()
            .enumerate()
            .map(|(index, name)| StageMetadata {
                // Registry construction bounds indices to 0..=63.
                #[allow(clippy::cast_possible_truncation)]
                id: StageId::new(self.inner.cookie, index as u8),
                name: name.clone(),
            })
            .collect::<Vec<_>>()
            .into_boxed_slice();

        let calibration_state =
            CalibrationState::from_raw(self.inner.calibration_state.load(Ordering::Acquire));
        let calibration_report = (calibration_state == CalibrationState::Frozen)
            .then(|| self.frozen_calibration_report())
            .flatten();

        Snapshot {
            cookie: self.inner.cookie,
            stages,
            layout: self.inner.layout,
            counters,
            bucket_bounds: self.inner.default_bounds.clone(),
            calibration_bucket_bounds: self.inner.calibration_bounds.clone(),
            tail_quantile: self.inner.tail_quantile,
            calibration_state,
            calibration_epoch: if calibration_state == CalibrationState::Frozen {
                self.inner.calibration_epoch.load(Ordering::Relaxed)
            } else {
                0
            },
            calibration_report,
            memory_budget_bytes: self.inner.memory_budget_bytes,
            memory_estimate: self.inner.memory_estimate,
            consistency: Consistency::Relaxed,
        }
    }
}

#[cfg(test)]
mod tests {
    use std::time::Duration;

    use super::*;
    use crate::context::Outcome;

    fn two_stage_recorder() -> (Tracegrams, StageId, StageId) {
        let mut builder = Tracegrams::builder();
        let first = builder.stage("first").unwrap();
        let second = builder.stage("second").unwrap();
        (builder.build().unwrap(), first, second)
    }

    #[test]
    fn cross_epoch_delta_keeps_matrices_and_discards_online_counters() {
        let (tracegrams, first, second) = two_stage_recorder();
        let online = tracegrams.inner.layout.online(second.index(), 0).unwrap();
        tracegrams.inner.counters[online].store(5, Ordering::Relaxed);
        let earlier = tracegrams.snapshot_relaxed();

        tracegrams.inner.counters[online].store(0, Ordering::Relaxed);
        tracegrams
            .inner
            .calibration_epoch
            .store(1, Ordering::Relaxed);
        tracegrams
            .inner
            .calibration_state
            .store(crate::recorder::CALIBRATION_FROZEN, Ordering::Release);
        let mut context = tracegrams.start_manual();
        tracegrams.record_elapsed(&mut context, first, Duration::from_nanos(50));
        tracegrams.finish_manual(context, second, Duration::from_nanos(75), Outcome::Success);
        let later = tracegrams.snapshot_relaxed();

        let delta = later.delta(&earlier).unwrap();
        assert_eq!(
            delta.online_delta_availability(),
            OnlineDeltaAvailability::EpochMismatch {
                earlier: 0,
                later: 1,
            }
        );
        assert_eq!(sum(delta.cause_counts(second).unwrap()), 1);
        assert_eq!(delta.sample_counts(second).unwrap().online(), 0);
        assert!(
            !delta
                .score_availability(second)
                .unwrap()
                .calibrated_online()
        );
    }

    #[test]
    fn every_diagnostic_counter_is_snapshot_visible() {
        let (tracegrams, _, _) = two_stage_recorder();
        let diagnostics = [
            DiagnosticCounter::InvalidStageMarks,
            DiagnosticCounter::InvalidContextMarks,
            DiagnosticCounter::NonMonotonicMarks,
            DiagnosticCounter::ClockRegressions,
            DiagnosticCounter::LatencyOverflows,
            DiagnosticCounter::CumulativeOverflows,
            DiagnosticCounter::CalibrationSamplesSkippedWhileFreezing,
            DiagnosticCounter::OnlineSamplesSkippedMissingPreviousThreshold,
        ];
        for (offset, diagnostic) in diagnostics.into_iter().enumerate() {
            tracegrams.inner.counters[tracegrams.inner.layout.diagnostic(diagnostic)]
                .store(u64::try_from(offset).unwrap() + 1, Ordering::Relaxed);
        }

        let observed = tracegrams.snapshot_relaxed().diagnostics();
        assert_eq!(observed.invalid_stage_marks(), 1);
        assert_eq!(observed.invalid_context_marks(), 2);
        assert_eq!(observed.non_monotonic_marks(), 3);
        assert_eq!(observed.clock_regressions(), 4);
        assert_eq!(observed.latency_overflows(), 5);
        assert_eq!(observed.cumulative_overflows(), 6);
        assert_eq!(observed.calibration_samples_skipped_while_freezing(), 7);
        assert_eq!(
            observed.online_samples_skipped_missing_previous_threshold(),
            8
        );
        assert_eq!(observed.total(), 36);
    }
}
