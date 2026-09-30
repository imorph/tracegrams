//! Owned relaxed scans of recorder counters and windows between them.

use std::error::Error;
use std::fmt;
use std::sync::atomic::Ordering;

use crate::bucket::{calibration_bounds, default_bounds};
use crate::calibration::FreezeReport;
use crate::init::{MemoryEstimate, RegistryCookie, StageId, Tracegrams};
use crate::recorder::{CalibrationPopulation, DiagnosticCounter, Segment, StorageLayout};

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

/// Per-population sample totals observed for one stage.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
#[non_exhaustive]
pub struct SampleCounts {
    /// Ordinary local-distribution samples.
    pub local: u128,
    /// Ordinary cumulative-after-distribution samples.
    pub cumulative_after: u128,
    /// Predecessor-bearing cause-matrix samples.
    pub cause: u128,
    /// Predecessor-bearing incoming-matrix samples.
    pub incoming: u128,
    /// Local calibration samples.
    pub calibration_local: u128,
    /// Cumulative-after calibration samples.
    pub calibration_cumulative_after: u128,
    /// Previous-cumulative calibration samples.
    pub calibration_previous_cumulative: u128,
    /// Calibrated-online truth-table samples.
    pub online: u128,
}

/// Snapshot-visible hot-path anomaly and rejection counters.
#[derive(Clone, Copy, Debug, Default, Eq, PartialEq)]
#[non_exhaustive]
pub struct Diagnostics {
    /// Marks carrying a stage from another registry.
    pub invalid_stage_marks: u64,
    /// Marks carrying a context from another registry.
    pub invalid_context_marks: u64,
    /// Repeated or decreasing stage marks.
    pub non_monotonic_marks: u64,
    /// Rejected clock readings earlier than the context timestamp.
    pub clock_regressions: u64,
    /// Local durations outside the exact `u64` nanosecond range.
    pub latency_overflows: u64,
    /// Cumulative-duration additions that overflowed `u64`.
    pub cumulative_overflows: u64,
    /// Calibration samples skipped while a freeze was in progress.
    pub calibration_samples_skipped_while_freezing: u64,
    /// Online samples skipped because no previous threshold exists.
    pub online_samples_skipped_missing_previous_threshold: u64,
}

impl Diagnostics {
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
#[non_exhaustive]
pub struct StageMetadata {
    /// Registry-branded stage identifier.
    pub id: StageId,
    /// Registered stage name.
    pub name: Box<str>,
}

/// Success and error completions observed for one stage.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
#[non_exhaustive]
pub struct CompletionCounts {
    /// Successful completions.
    pub success: u64,
    /// Error completions.
    pub error: u64,
}

/// An owned relaxed scan of one recorder's counters, or the window between
/// two such scans.
///
/// Each cell was loaded atomically, but concurrent checkpoints may be only
/// partly represented. The snapshot remains valid after all recorder handles
/// are dropped.
///
/// [`Snapshot::delta`] returns a window with the same type: every accessor
/// reports counter differences, and calibration state and report are those of
/// the later endpoint. The type does not distinguish a window from an absolute
/// scan; callers that need absolute counts must keep track of which one they
/// hold.
#[derive(Clone, Debug)]
pub struct Snapshot {
    cookie: RegistryCookie,
    stages: Box<[StageMetadata]>,
    layout: StorageLayout,
    counters: Box<[u64]>,
    tail_quantile: f64,
    calibration_state: CalibrationState,
    calibration_report: Option<FreezeReport>,
    memory_estimate: MemoryEstimate,
    spans_freeze: bool,
}

impl Snapshot {
    /// Returns owned metadata for all stages in registration order.
    pub fn stages(&self) -> &[StageMetadata] {
        &self.stages
    }

    /// Returns the pinned ordinary bucket bounds; each bound starts the next bucket.
    pub fn bucket_bounds(&self) -> &'static [u64] {
        default_bounds()
    }

    /// Returns the pinned calibration-grid bounds, using the same convention.
    pub fn calibration_bucket_bounds(&self) -> &'static [u64] {
        calibration_bounds()
    }

    /// Returns the configured calibration tail quantile.
    pub const fn tail_quantile(&self) -> f64 {
        self.tail_quantile
    }

    /// Returns the observed calibration lifecycle state.
    pub const fn calibration_state(&self) -> CalibrationState {
        self.calibration_state
    }

    /// Returns the immutable successful freeze report after publication.
    pub const fn calibration_report(&self) -> Option<&FreezeReport> {
        self.calibration_report.as_ref()
    }

    /// Returns the recorder's itemized owned-memory estimate.
    pub const fn memory_estimate(&self) -> MemoryEstimate {
        self.memory_estimate
    }

    /// Returns whether this window spans the calibration freeze.
    ///
    /// Such a window's online counters cover only contexts started after the
    /// freeze, so online diagnosis rejects it. Always `false` for a scan
    /// returned by [`Tracegrams::snapshot_relaxed`].
    pub const fn spans_freeze(&self) -> bool {
        self.spans_freeze
    }

    /// Returns the local-latency distribution for `stage`.
    pub fn local_counts(&self, stage: StageId) -> Option<&[u64]> {
        self.counts(stage, Segment::Local)
    }

    /// Returns the cumulative-latency-after distribution for `stage`.
    pub fn cumulative_counts(&self, stage: StageId) -> Option<&[u64]> {
        self.counts(stage, Segment::Cumulative)
    }

    /// Returns the destination-indexed cause matrix in row-major order.
    ///
    /// Rows are previous-cumulative buckets and columns are local buckets.
    pub fn cause_counts(&self, destination: StageId) -> Option<&[u64]> {
        self.counts(destination, Segment::Cause)
    }

    /// Returns the destination-indexed incoming matrix in row-major order.
    ///
    /// Rows are previous-cumulative buckets and columns are cumulative-after
    /// buckets. The first registered stage has no incoming matrix.
    pub fn incoming_counts(&self, destination: StageId) -> Option<&[u64]> {
        self.counts(destination, Segment::Incoming)
    }

    /// Returns a calibration distribution for `stage`.
    pub fn calibration_counts(
        &self,
        stage: StageId,
        population: CalibrationPopulation,
    ) -> Option<&[u64]> {
        self.counts(stage, Segment::Calibration(population))
    }

    pub(crate) fn online_counts(&self, stage: StageId) -> Option<&[u64]> {
        self.counts(stage, Segment::Online)
    }

    /// Returns totals for every stored sample population at `stage`.
    pub fn sample_counts(&self, stage: StageId) -> Option<SampleCounts> {
        let calibration = |population| self.calibration_counts(stage, population).map(sum);
        Some(SampleCounts {
            local: sum(self.local_counts(stage)?),
            cumulative_after: sum(self.cumulative_counts(stage)?),
            cause: sum(self.cause_counts(stage)?),
            incoming: self.incoming_counts(stage).map_or(0, sum),
            calibration_local: calibration(CalibrationPopulation::Local)?,
            calibration_cumulative_after: calibration(CalibrationPopulation::CumulativeAfter)?,
            calibration_previous_cumulative: calibration(
                CalibrationPopulation::PreviousCumulative,
            )?,
            online: sum(self.online_counts(stage)?),
        })
    }

    /// Returns terminal completion counts for `stage`.
    pub fn completion_counts(&self, stage: StageId) -> Option<CompletionCounts> {
        let &[success, error] = self.counts(stage, Segment::Completion)? else {
            unreachable!("a completion segment holds two counters");
        };
        Some(CompletionCounts { success, error })
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

    /// Purely subtracts `earlier` from this later snapshot and returns the
    /// window between them.
    ///
    /// When exactly one endpoint observed the frozen state, the window spans
    /// the calibration freeze; see [`Snapshot::spans_freeze`].
    pub fn delta(&self, earlier: &Self) -> Result<Self, DeltaError> {
        if self.cookie != earlier.cookie {
            return Err(DeltaError::RegistryMismatch);
        }
        let counters = self
            .counters
            .iter()
            .zip(&earlier.counters)
            .map(|(later, earlier)| {
                later
                    .checked_sub(*earlier)
                    .ok_or(DeltaError::CounterUnderflow {
                        earlier: *earlier,
                        later: *later,
                    })
            })
            .collect::<Result<Vec<_>, _>>()?
            .into_boxed_slice();
        let frozen = |snapshot: &Self| snapshot.calibration_state == CalibrationState::Frozen;
        Ok(Self {
            cookie: self.cookie,
            stages: self.stages.clone(),
            layout: self.layout,
            counters,
            tail_quantile: self.tail_quantile,
            calibration_state: self.calibration_state,
            calibration_report: self.calibration_report.clone(),
            memory_estimate: self.memory_estimate,
            spans_freeze: frozen(self) != frozen(earlier),
        })
    }

    pub(crate) fn stage_index(&self, stage: StageId) -> Option<usize> {
        (stage.cookie() == self.cookie && stage.index() < self.stages.len()).then(|| stage.index())
    }

    fn counts(&self, stage: StageId, segment: Segment) -> Option<&[u64]> {
        let range = self.layout.segment(self.stage_index(stage)?, segment)?;
        Some(&self.counters[range])
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
            .stage_ids()
            .zip(self.inner.stage_names.iter())
            .map(|(id, name)| StageMetadata {
                id,
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
            tail_quantile: self.inner.tail_quantile,
            calibration_state,
            calibration_report,
            memory_estimate: self.inner.memory_estimate,
            spans_freeze: false,
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn every_diagnostic_counter_is_snapshot_visible() {
        let mut builder = Tracegrams::builder();
        builder.stage("first").unwrap();
        let tracegrams = builder.build().unwrap();
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
        assert_eq!(observed.total(), 36);
        assert_eq!(
            observed,
            Diagnostics {
                invalid_stage_marks: 1,
                invalid_context_marks: 2,
                non_monotonic_marks: 3,
                clock_regressions: 4,
                latency_overflows: 5,
                cumulative_overflows: 6,
                calibration_samples_skipped_while_freezing: 7,
                online_samples_skipped_missing_previous_threshold: 8,
            }
        );
    }
}
