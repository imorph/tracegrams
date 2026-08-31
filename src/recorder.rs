//! Fixed shared recorder storage and checkpoint counter primitives.

#[cfg(test)]
use std::collections::VecDeque;
#[cfg(test)]
use std::sync::Mutex;
use std::sync::OnceLock;
use std::sync::atomic::{AtomicU8, AtomicU16, AtomicU64, Ordering};
use std::time::Instant;

use crate::bucket::{CALIBRATION_BUCKETS, calibration_bounds, default_bounds};
use crate::calibration::FrozenStageCalibration;
use crate::init::{MemoryEstimate, RegistryCookie};
use crate::matrix::StorageLayout;

const CALIBRATION_DISTRIBUTIONS_PER_STAGE: usize = 3;
pub(crate) const CALIBRATION_COLLECTING: u8 = 0;
pub(crate) const CALIBRATION_FREEZING: u8 = 1;
pub(crate) const CALIBRATION_FROZEN: u8 = 2;
// First marks classify two tail predicates (4 cells); predecessor-bearing
// marks classify three (8 cells). The 12 cells per stage are disjoint.
pub(crate) const ONLINE_FIRST_COUNTERS: usize = 4;
pub(crate) const ONLINE_COUNTERS_PER_STAGE: usize = 12;

#[inline]
pub(crate) fn first_online_counter(local_tail: bool, cumulative_after_tail: bool) -> usize {
    usize::from(local_tail) * 2 + usize::from(cumulative_after_tail)
}

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
const COMPLETION_COUNTERS_PER_STAGE: usize = 2;
const DIAGNOSTIC_COUNTERS: usize = 10;

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub(crate) enum CalibrationDistribution {
    Local,
    CumulativeAfter,
    PreviousCumulative,
}

impl CalibrationDistribution {
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
    /// Reserved and currently never emitted: the owned-context API makes a
    /// mark after `finish` unrepresentable. Re-evaluate this slot before 1.0.
    MarksAfterFinish,
    ClockRegressions,
    LatencyOverflows,
    CumulativeOverflows,
    CounterOverflows,
    CalibrationSamplesSkippedWhileFreezing,
    OnlineSamplesSkippedMissingPreviousThreshold,
}

impl DiagnosticCounter {
    const fn offset(self) -> usize {
        match self {
            Self::InvalidStageMarks => 0,
            Self::InvalidContextMarks => 1,
            Self::NonMonotonicMarks => 2,
            Self::MarksAfterFinish => 3,
            Self::ClockRegressions => 4,
            Self::LatencyOverflows => 5,
            Self::CumulativeOverflows => 6,
            Self::CounterOverflows => 7,
            Self::CalibrationSamplesSkippedWhileFreezing => 8,
            Self::OnlineSamplesSkippedMissingPreviousThreshold => 9,
        }
    }
}

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub(crate) struct FixedStorageLayout {
    stage_count: usize,
    matrix: StorageLayout,
    calibration_start: usize,
    calibration_count: usize,
    online_start: usize,
    online_count: usize,
    completion_start: usize,
    completion_count: usize,
    diagnostic_start: usize,
    counter_count: usize,
}

impl FixedStorageLayout {
    pub(crate) fn new(stage_count: usize) -> Option<Self> {
        let matrix = StorageLayout::new(stage_count)?;
        let calibration_count = stage_count
            .checked_mul(CALIBRATION_DISTRIBUTIONS_PER_STAGE)?
            .checked_mul(CALIBRATION_BUCKETS)?;
        let calibration_start = matrix.counter_count();
        let online_count = stage_count.checked_mul(ONLINE_COUNTERS_PER_STAGE)?;
        let online_start = calibration_start.checked_add(calibration_count)?;
        let completion_count = stage_count.checked_mul(COMPLETION_COUNTERS_PER_STAGE)?;
        let completion_start = online_start.checked_add(online_count)?;
        let diagnostic_start = completion_start.checked_add(completion_count)?;
        let counter_count = diagnostic_start.checked_add(DIAGNOSTIC_COUNTERS)?;

        Some(Self {
            stage_count,
            matrix,
            calibration_start,
            calibration_count,
            online_start,
            online_count,
            completion_start,
            completion_count,
            diagnostic_start,
            counter_count,
        })
    }

    pub(crate) const fn stage_count(self) -> usize {
        self.stage_count
    }

    pub(crate) fn matrix_counter_bytes(self) -> Option<usize> {
        self.matrix.counter_bytes()
    }

    pub(crate) const fn calibration_counter_count(self) -> usize {
        self.calibration_count
    }

    pub(crate) const fn online_counter_count(self) -> usize {
        self.online_count
    }

    pub(crate) const fn online_start(self) -> usize {
        self.online_start
    }

    pub(crate) const fn completion_counter_count(self) -> usize {
        self.completion_count
    }

    pub(crate) const fn diagnostic_counter_count() -> usize {
        DIAGNOSTIC_COUNTERS
    }

    pub(crate) const fn counter_count(self) -> usize {
        self.counter_count
    }

    pub(crate) fn local(self, stage: usize, bucket: usize) -> Option<usize> {
        self.matrix.local(stage, bucket)
    }

    pub(crate) fn cumulative(self, stage: usize, bucket: usize) -> Option<usize> {
        self.matrix.cumulative(stage, bucket)
    }

    pub(crate) fn cause(
        self,
        destination: usize,
        previous_cumulative: usize,
        local: usize,
    ) -> Option<usize> {
        self.matrix.cause(destination, previous_cumulative, local)
    }

    pub(crate) fn incoming(
        self,
        destination: usize,
        previous_cumulative: usize,
        cumulative_after: usize,
    ) -> Option<usize> {
        self.matrix
            .incoming(destination, previous_cumulative, cumulative_after)
    }

    pub(crate) fn calibration(
        self,
        stage: usize,
        distribution: CalibrationDistribution,
        bucket: usize,
    ) -> Option<usize> {
        if stage >= self.stage_count || bucket >= CALIBRATION_BUCKETS {
            return None;
        }
        self.calibration_start
            .checked_add(
                stage
                    .checked_mul(CALIBRATION_DISTRIBUTIONS_PER_STAGE)?
                    .checked_add(distribution.offset())?
                    .checked_mul(CALIBRATION_BUCKETS)?,
            )?
            .checked_add(bucket)
    }

    pub(crate) fn online(self, stage: usize, counter: usize) -> Option<usize> {
        if stage >= self.stage_count || counter >= ONLINE_COUNTERS_PER_STAGE {
            return None;
        }
        self.online_start
            .checked_add(stage.checked_mul(ONLINE_COUNTERS_PER_STAGE)?)?
            .checked_add(counter)
    }

    pub(crate) fn completion(self, stage: usize, error: bool) -> Option<usize> {
        if stage >= self.stage_count {
            return None;
        }
        self.completion_start
            .checked_add(stage.checked_mul(COMPLETION_COUNTERS_PER_STAGE)?)?
            .checked_add(usize::from(error))
    }

    pub(crate) fn diagnostic(self, diagnostic: DiagnosticCounter) -> usize {
        self.diagnostic_start + diagnostic.offset()
    }
}

pub(crate) struct Inner {
    pub(crate) cookie: RegistryCookie,
    pub(crate) stage_names: Box<[Box<str>]>,
    pub(crate) tail_quantile: f64,
    pub(crate) memory_budget_bytes: usize,
    pub(crate) memory_estimate: MemoryEstimate,
    pub(crate) layout: FixedStorageLayout,
    pub(crate) counters: Box<[AtomicU64]>,
    pub(crate) default_bounds: Box<[u64]>,
    pub(crate) calibration_bounds: Box<[u64]>,
    pub(crate) calibration_state: AtomicU8,
    pub(crate) calibration_epoch: AtomicU16,
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
        memory_budget_bytes: usize,
        memory_estimate: MemoryEstimate,
        layout: FixedStorageLayout,
    ) -> Self {
        let mut counters = Vec::with_capacity(layout.counter_count());
        counters.resize_with(layout.counter_count(), || AtomicU64::new(0));

        Self {
            cookie,
            stage_names,
            tail_quantile,
            memory_budget_bytes,
            memory_estimate,
            layout,
            counters: counters.into_boxed_slice(),
            default_bounds: default_bounds().to_vec().into_boxed_slice(),
            calibration_bounds: calibration_bounds().to_vec().into_boxed_slice(),
            calibration_state: AtomicU8::new(CALIBRATION_COLLECTING),
            calibration_epoch: AtomicU16::new(0),
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
        let overflow = self.layout.diagnostic(DiagnosticCounter::CounterOverflows);
        increment_counter(&self.counters[index], &self.counters[overflow]);
    }

    pub(crate) fn increment_diagnostic(&self, diagnostic: DiagnosticCounter) {
        self.increment(self.layout.diagnostic(diagnostic));
    }
}

/// Increments `counter`, diverting the increment into `overflows` once the
/// counter saturates so no total is silently lost.
pub fn increment_counter(counter: &AtomicU64, overflows: &AtomicU64) {
    if !increment_saturating(counter) {
        increment_saturating(overflows);
    }
}

fn increment_saturating(counter: &AtomicU64) -> bool {
    let mut current = counter.load(Ordering::Relaxed);
    loop {
        let Some(next) = current.checked_add(1) else {
            return false;
        };
        match counter.compare_exchange_weak(current, next, Ordering::Relaxed, Ordering::Relaxed) {
            Ok(_) => return true,
            Err(observed) => current = observed,
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn fixed_layout_segments_are_contiguous() {
        let layout = FixedStorageLayout::new(6).unwrap();

        assert_eq!(layout.matrix_counter_bytes(), Some(366_592));
        assert_eq!(layout.calibration_counter_count(), 6 * 3 * 250);
        assert_eq!(layout.online_counter_count(), 6 * 12);
        assert_eq!(layout.completion_start, layout.online_start + 6 * 12);
        assert_eq!(layout.completion_counter_count(), 6 * 2);
        assert_eq!(layout.diagnostic_start, layout.completion_start + 6 * 2);
        assert_eq!(layout.counter_count(), layout.diagnostic_start + 10);
    }

    #[test]
    fn checked_fixed_layout_covers_the_maximum_stage_count() {
        let layout = FixedStorageLayout::new(64).unwrap();

        assert_eq!(layout.matrix_counter_bytes(), Some(4_227_072));
        assert_eq!(layout.calibration_counter_count(), 48_000);
        assert_eq!(layout.online_counter_count(), 768);
        assert_eq!(layout.completion_counter_count(), 128);
        assert_eq!(layout.counter_count(), 577_290);
        assert_eq!(FixedStorageLayout::new(0), None);
        assert_eq!(FixedStorageLayout::new(usize::MAX), None);
    }

    #[test]
    fn counter_saturates_and_accounts_for_every_overflow() {
        let counter = AtomicU64::new(u64::MAX - 1);
        let overflows = AtomicU64::new(0);

        increment_counter(&counter, &overflows);
        assert_eq!(counter.load(Ordering::Relaxed), u64::MAX);
        assert_eq!(overflows.load(Ordering::Relaxed), 0);

        increment_counter(&counter, &overflows);
        increment_counter(&counter, &overflows);
        assert_eq!(counter.load(Ordering::Relaxed), u64::MAX);
        assert_eq!(overflows.load(Ordering::Relaxed), 2);
    }

    #[test]
    fn overflow_accounting_counter_cannot_wrap() {
        let counter = AtomicU64::new(u64::MAX);
        let overflows = AtomicU64::new(u64::MAX);

        increment_counter(&counter, &overflows);
        assert_eq!(counter.load(Ordering::Relaxed), u64::MAX);
        assert_eq!(overflows.load(Ordering::Relaxed), u64::MAX);
    }

    #[test]
    fn online_truth_table_cells_are_disjoint_and_complete() {
        let mut cells = Vec::new();
        for local_tail in [false, true] {
            for cumulative_after_tail in [false, true] {
                cells.push(first_online_counter(local_tail, cumulative_after_tail));
            }
        }
        for previous_tail in [false, true] {
            for local_tail in [false, true] {
                for cumulative_after_tail in [false, true] {
                    cells.push(predecessor_online_counter(
                        previous_tail,
                        local_tail,
                        cumulative_after_tail,
                    ));
                }
            }
        }
        cells.sort_unstable();

        assert_eq!(cells, (0..ONLINE_COUNTERS_PER_STAGE).collect::<Vec<_>>());
    }
}
