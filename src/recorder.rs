//! Fixed shared recorder storage and checkpoint counter primitives.

#[cfg(test)]
use std::collections::VecDeque;
use std::ops::Range;
#[cfg(test)]
use std::sync::Mutex;
use std::sync::OnceLock;
use std::sync::atomic::{AtomicU8, AtomicU64, Ordering};
use std::time::Instant;

use crate::bucket::{BUCKETS, Bucket, CALIBRATION_BUCKETS};
use crate::calibration::FrozenStageCalibration;
use crate::context::{IncomingStageIndex, ValidatedStageIndex};
use crate::init::{MemoryEstimate, RegistryCookie};

const MATRIX_CELLS: usize = BUCKETS * BUCKETS;
const CALIBRATION_POPULATIONS_PER_STAGE: usize = 3;
pub(crate) const CALIBRATION_COLLECTING: u8 = 0;
pub(crate) const CALIBRATION_FREEZING: u8 = 1;
pub(crate) const CALIBRATION_FROZEN: u8 = 2;
// First marks classify two tail predicates (4 cells); predecessor-bearing
// marks classify three (8 cells). The 12 cells per stage are disjoint.
pub(crate) const ONLINE_FIRST_COUNTERS: usize = 4;
pub(crate) const ONLINE_COUNTERS_PER_STAGE: usize = 12;
const COMPLETION_COUNTERS_PER_STAGE: usize = 2;
const DIAGNOSTIC_COUNTERS: usize = 8;

/// Returns the online cell below [`ONLINE_FIRST_COUNTERS`] for a first mark.
#[inline]
pub(crate) fn first_online_counter(local_tail: bool, cumulative_after_tail: bool) -> usize {
    usize::from(local_tail) * 2 + usize::from(cumulative_after_tail)
}

/// Returns the online cell at or above [`ONLINE_FIRST_COUNTERS`] for a
/// predecessor-bearing mark.
#[inline]
pub(crate) fn predecessor_online_counter(
    previous_tail: bool,
    local_tail: bool,
    cumulative_after_tail: bool,
) -> usize {
    ONLINE_FIRST_COUNTERS
        + usize::from(previous_tail) * 4
        + usize::from(local_tail) * 2
        + usize::from(cumulative_after_tail)
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
    const fn offset(self) -> usize {
        match self {
            Self::Local => 0,
            Self::CumulativeAfter => 1,
            Self::PreviousCumulative => 2,
        }
    }
}

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub(crate) enum DiagnosticCounter {
    InvalidStageMarks,
    InvalidContextMarks,
    NonMonotonicMarks,
    ClockRegressions,
    LatencyOverflows,
    CumulativeOverflows,
    CalibrationSamplesSkippedWhileFreezing,
    OnlineSamplesSkippedMissingPreviousThreshold,
}

impl DiagnosticCounter {
    const fn offset(self) -> usize {
        match self {
            Self::InvalidStageMarks => 0,
            Self::InvalidContextMarks => 1,
            Self::NonMonotonicMarks => 2,
            Self::ClockRegressions => 3,
            Self::LatencyOverflows => 4,
            Self::CumulativeOverflows => 5,
            Self::CalibrationSamplesSkippedWhileFreezing => 6,
            Self::OnlineSamplesSkippedMissingPreviousThreshold => 7,
        }
    }
}

/// One per-stage run of counters in the fixed storage.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub(crate) enum Segment {
    Local,
    Cumulative,
    /// Row-major previous-cumulative by local buckets.
    Cause,
    /// Row-major previous-cumulative by cumulative-after buckets; the first
    /// registered stage has no predecessor and therefore no incoming matrix.
    Incoming,
    Calibration(CalibrationPopulation),
    Online,
    /// Success, then error.
    Completion,
}

/// Offsets of every counter segment in one flat array.
///
/// Cold readers take a checked [`StorageLayout::segment`] range. The hot
/// `*_index` methods consume bounded values produced at the recorder boundary
/// and share the same geometry; debug assertions and the final slice bounds
/// check are defense in depth.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub(crate) struct StorageLayout {
    stage_count: usize,
    cumulative_start: usize,
    cause_start: usize,
    incoming_start: usize,
    calibration_start: usize,
    online_start: usize,
    completion_start: usize,
    diagnostic_start: usize,
    counter_count: usize,
}

impl StorageLayout {
    pub(crate) fn new(stage_count: usize) -> Option<Self> {
        if stage_count == 0 {
            return None;
        }
        let distribution_cells = stage_count.checked_mul(BUCKETS)?;
        let cumulative_start = distribution_cells;
        let cause_start = cumulative_start.checked_add(distribution_cells)?;
        let incoming_start = cause_start.checked_add(stage_count.checked_mul(MATRIX_CELLS)?)?;
        let calibration_start =
            incoming_start.checked_add((stage_count - 1).checked_mul(MATRIX_CELLS)?)?;
        let online_start = calibration_start.checked_add(
            stage_count
                .checked_mul(CALIBRATION_POPULATIONS_PER_STAGE)?
                .checked_mul(CALIBRATION_BUCKETS)?,
        )?;
        let completion_start =
            online_start.checked_add(stage_count.checked_mul(ONLINE_COUNTERS_PER_STAGE)?)?;
        let diagnostic_start = completion_start
            .checked_add(stage_count.checked_mul(COMPLETION_COUNTERS_PER_STAGE)?)?;
        let counter_count = diagnostic_start.checked_add(DIAGNOSTIC_COUNTERS)?;

        Some(Self {
            stage_count,
            cumulative_start,
            cause_start,
            incoming_start,
            calibration_start,
            online_start,
            completion_start,
            diagnostic_start,
            counter_count,
        })
    }

    pub(crate) const fn stage_count(self) -> usize {
        self.stage_count
    }

    pub(crate) const fn counter_count(self) -> usize {
        self.counter_count
    }

    /// Counters in ordinary distributions and transition matrices.
    pub(crate) const fn matrix_counter_count(self) -> usize {
        self.calibration_start
    }

    /// Counters in calibration distributions.
    pub(crate) const fn calibration_counter_count(self) -> usize {
        self.online_start - self.calibration_start
    }

    /// Returns the counters of `segment` for `stage`, or `None` for an
    /// out-of-range stage and for the first stage's incoming matrix.
    pub(crate) fn segment(self, stage: usize, segment: Segment) -> Option<Range<usize>> {
        if stage >= self.stage_count || (stage == 0 && segment == Segment::Incoming) {
            return None;
        }
        let (start, _, length) = self.geometry(segment, stage);
        Some(start..start + length)
    }

    /// Returns `(start, stage_stride, length)` of `segment` at `stage`.
    ///
    /// Incoming matrices are stored for stages `1..stage_count` only, so their
    /// stride starts one matrix earlier; callers exclude stage 0.
    #[inline]
    const fn geometry(self, segment: Segment, stage: usize) -> (usize, usize, usize) {
        let (base, stride, length) = match segment {
            Segment::Local => (0, BUCKETS, BUCKETS),
            Segment::Cumulative => (self.cumulative_start, BUCKETS, BUCKETS),
            Segment::Cause => (self.cause_start, MATRIX_CELLS, MATRIX_CELLS),
            Segment::Incoming => (
                self.incoming_start - MATRIX_CELLS,
                MATRIX_CELLS,
                MATRIX_CELLS,
            ),
            Segment::Calibration(population) => (
                self.calibration_start + population.offset() * CALIBRATION_BUCKETS,
                CALIBRATION_POPULATIONS_PER_STAGE * CALIBRATION_BUCKETS,
                CALIBRATION_BUCKETS,
            ),
            Segment::Online => (
                self.online_start,
                ONLINE_COUNTERS_PER_STAGE,
                ONLINE_COUNTERS_PER_STAGE,
            ),
            Segment::Completion => (
                self.completion_start,
                COMPLETION_COUNTERS_PER_STAGE,
                COMPLETION_COUNTERS_PER_STAGE,
            ),
        };
        (base + stage * stride, stride, length)
    }

    #[inline]
    fn cell(self, segment: Segment, stage: usize, offset: usize) -> usize {
        let (start, _, length) = self.geometry(segment, stage);
        debug_assert!(stage < self.stage_count && offset < length);
        start + offset
    }

    #[inline]
    pub(crate) fn local_index(self, stage: ValidatedStageIndex, bucket: Bucket) -> usize {
        self.cell(Segment::Local, stage.index(), bucket.index())
    }

    #[inline]
    pub(crate) fn cumulative_index(self, stage: ValidatedStageIndex, bucket: Bucket) -> usize {
        self.cell(Segment::Cumulative, stage.index(), bucket.index())
    }

    #[inline]
    pub(crate) fn cause_index(
        self,
        destination: ValidatedStageIndex,
        previous_cumulative: Bucket,
        local: Bucket,
    ) -> usize {
        self.cell(
            Segment::Cause,
            destination.index(),
            previous_cumulative.index() * BUCKETS + local.index(),
        )
    }

    /// An `IncomingStageIndex` exists only for a destination after a
    /// predecessor, so it is never the first stage.
    #[inline]
    pub(crate) fn incoming_index(
        self,
        destination: IncomingStageIndex,
        previous_cumulative: Bucket,
        cumulative_after: Bucket,
    ) -> usize {
        self.cell(
            Segment::Incoming,
            destination.destination(),
            previous_cumulative.index() * BUCKETS + cumulative_after.index(),
        )
    }

    #[inline]
    pub(crate) fn calibration_index(
        self,
        stage: ValidatedStageIndex,
        population: CalibrationPopulation,
        bucket: usize,
    ) -> usize {
        // The calibration bucket is raw; this release assertion was measured
        // free and keeps an invalid bucket from reaching another segment.
        assert!(bucket < CALIBRATION_BUCKETS);
        self.cell(Segment::Calibration(population), stage.index(), bucket)
    }

    /// `counter` comes from one of the two online truth-table functions above.
    #[inline]
    pub(crate) fn online_index(self, stage: ValidatedStageIndex, counter: usize) -> usize {
        self.cell(Segment::Online, stage.index(), counter)
    }

    #[inline]
    pub(crate) fn completion_index(self, stage: ValidatedStageIndex, error: bool) -> usize {
        self.cell(Segment::Completion, stage.index(), usize::from(error))
    }

    pub(crate) fn diagnostic(self, diagnostic: DiagnosticCounter) -> usize {
        self.diagnostic_start + diagnostic.offset()
    }

    /// Checked cell lookup for tests: `offset` within `segment` at `stage`.
    #[cfg(test)]
    pub(crate) fn cell_at(self, stage: usize, segment: Segment, offset: usize) -> usize {
        let range = self
            .segment(stage, segment)
            .expect("the test names a valid segment");
        assert!(offset < range.len());
        range.start + offset
    }
}

pub(crate) struct Inner {
    pub(crate) cookie: RegistryCookie,
    pub(crate) stage_names: Box<[Box<str>]>,
    pub(crate) tail_quantile: f64,
    pub(crate) memory_estimate: MemoryEstimate,
    pub(crate) layout: StorageLayout,
    pub(crate) counters: Box<[AtomicU64]>,
    pub(crate) calibration_state: AtomicU8,
    pub(crate) frozen_calibration: Box<[OnceLock<FrozenStageCalibration>]>,
    pub(crate) clock_epoch: Instant,
    #[cfg(test)]
    pub(crate) clock_readings: Mutex<VecDeque<(u64, bool)>>,
    #[cfg(test)]
    pub(crate) clock_reads: AtomicU64,
}

impl Inner {
    pub(crate) fn new(
        cookie: RegistryCookie,
        stage_names: Box<[Box<str>]>,
        tail_quantile: f64,
        memory_estimate: MemoryEstimate,
        layout: StorageLayout,
    ) -> Self {
        let mut counters = Vec::with_capacity(layout.counter_count());
        counters.resize_with(layout.counter_count(), || AtomicU64::new(0));

        Self {
            cookie,
            stage_names,
            tail_quantile,
            memory_estimate,
            layout,
            counters: counters.into_boxed_slice(),
            calibration_state: AtomicU8::new(CALIBRATION_COLLECTING),
            frozen_calibration: (0..layout.stage_count())
                .map(|_| OnceLock::new())
                .collect::<Vec<_>>()
                .into_boxed_slice(),
            clock_epoch: Instant::now(),
            #[cfg(test)]
            clock_readings: Mutex::new(VecDeque::new()),
            #[cfg(test)]
            clock_reads: AtomicU64::new(0),
        }
    }

    pub(crate) fn increment(&self, index: usize) {
        increment_counter(&self.counters[index]);
    }

    pub(crate) fn increment_diagnostic(&self, diagnostic: DiagnosticCounter) {
        self.increment(self.layout.diagnostic(diagnostic));
    }
}

/// Increments `counter`; wrap is out of scope because one increment per
/// nanosecond takes approximately 584 years to exhaust `u64`.
pub(crate) fn increment_counter(counter: &AtomicU64) {
    counter.fetch_add(1, Ordering::Relaxed);
}

#[cfg(test)]
mod tests {
    use super::*;

    fn all_segments() -> [Segment; 9] {
        [
            Segment::Local,
            Segment::Cumulative,
            Segment::Cause,
            Segment::Incoming,
            Segment::Calibration(CalibrationPopulation::Local),
            Segment::Calibration(CalibrationPopulation::CumulativeAfter),
            Segment::Calibration(CalibrationPopulation::PreviousCumulative),
            Segment::Online,
            Segment::Completion,
        ]
    }

    #[test]
    fn segments_and_diagnostics_cover_the_storage_exactly_once() {
        let stages = 3;
        let layout = StorageLayout::new(stages).unwrap();
        let mut visits = vec![0_u8; layout.counter_count()];

        for stage in 0..stages {
            for segment in all_segments() {
                if let Some(range) = layout.segment(stage, segment) {
                    for index in range {
                        visits[index] += 1;
                    }
                }
            }
        }
        for diagnostic in [
            DiagnosticCounter::InvalidStageMarks,
            DiagnosticCounter::InvalidContextMarks,
            DiagnosticCounter::NonMonotonicMarks,
            DiagnosticCounter::ClockRegressions,
            DiagnosticCounter::LatencyOverflows,
            DiagnosticCounter::CumulativeOverflows,
            DiagnosticCounter::CalibrationSamplesSkippedWhileFreezing,
            DiagnosticCounter::OnlineSamplesSkippedMissingPreviousThreshold,
        ] {
            visits[layout.diagnostic(diagnostic)] += 1;
        }

        assert!(visits.iter().all(|visits| *visits == 1));
    }

    #[test]
    fn segments_have_their_documented_lengths_and_order() {
        let layout = StorageLayout::new(3).unwrap();
        let length = |segment| layout.segment(1, segment).unwrap().len();

        assert_eq!(length(Segment::Local), BUCKETS);
        assert_eq!(length(Segment::Cause), BUCKETS * BUCKETS);
        assert_eq!(length(Segment::Incoming), BUCKETS * BUCKETS);
        assert_eq!(
            length(Segment::Calibration(CalibrationPopulation::Local)),
            CALIBRATION_BUCKETS
        );
        assert_eq!(length(Segment::Online), ONLINE_COUNTERS_PER_STAGE);
        assert_eq!(length(Segment::Completion), 2);
        let start = |stage, segment| layout.segment(stage, segment).unwrap().start;
        assert_eq!(
            start(2, Segment::Cause) - start(1, Segment::Cause),
            BUCKETS * BUCKETS
        );
        assert_eq!(start(1, Segment::Incoming), layout.incoming_start);
        assert_eq!(
            start(2, Segment::Incoming) - start(1, Segment::Incoming),
            BUCKETS * BUCKETS
        );
    }

    #[test]
    fn out_of_range_stages_and_the_first_incoming_matrix_have_no_segment() {
        let layout = StorageLayout::new(2).unwrap();

        assert_eq!(layout.segment(0, Segment::Incoming), None);
        for segment in all_segments() {
            assert_eq!(layout.segment(2, segment), None);
        }
    }

    #[test]
    fn checked_layout_arithmetic_covers_the_maximum_stage_count() {
        let six = StorageLayout::new(6).unwrap();
        assert_eq!(six.matrix_counter_count(), 45_824);
        assert_eq!(six.calibration_counter_count(), 6 * 3 * 250);

        let max = StorageLayout::new(64).unwrap();
        assert_eq!(max.matrix_counter_count(), 528_384);
        assert_eq!(max.calibration_counter_count(), 48_000);
        assert_eq!(max.counter_count(), 577_288);
        assert_eq!(StorageLayout::new(0), None);
        assert_eq!(StorageLayout::new(usize::MAX), None);
    }

    #[test]
    fn online_truth_table_cells_are_disjoint_and_complete() {
        // (local_tail, cumulative_after_tail) -> cell
        let first = [
            ((false, false), 0),
            ((false, true), 1),
            ((true, false), 2),
            ((true, true), 3),
        ];
        // (previous_tail, local_tail, cumulative_after_tail) -> cell
        let predecessor = [
            ((false, false, false), 4),
            ((false, false, true), 5),
            ((false, true, false), 6),
            ((false, true, true), 7),
            ((true, false, false), 8),
            ((true, false, true), 9),
            ((true, true, false), 10),
            ((true, true, true), 11),
        ];

        let mut cells = Vec::new();
        for ((local_tail, cumulative_after_tail), expected) in first {
            let cell = first_online_counter(local_tail, cumulative_after_tail);
            assert_eq!(cell, expected);
            cells.push(cell);
        }
        for ((previous_tail, local_tail, cumulative_after_tail), expected) in predecessor {
            let cell = predecessor_online_counter(previous_tail, local_tail, cumulative_after_tail);
            assert_eq!(cell, expected);
            cells.push(cell);
        }
        cells.sort_unstable();

        assert_eq!(cells, (0..ONLINE_COUNTERS_PER_STAGE).collect::<Vec<_>>());
    }
}
