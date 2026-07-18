//! Fixed shared recorder storage and checkpoint counter primitives.

use std::sync::atomic::{AtomicU8, AtomicU16, AtomicU64, Ordering};

use crate::bucket::{CALIBRATION_BUCKETS, calibration_bounds, default_bounds};
use crate::init::{MemoryEstimate, RegistryCookie};
use crate::matrix::StorageLayout;

const CALIBRATION_DISTRIBUTIONS_PER_STAGE: usize = 3;
// F×(Tl,Ta) = 4 + P×(Tp,Tl,Ta) = 8 disjoint truth-table cells (§5).
const ONLINE_COUNTERS_PER_STAGE: usize = 12;
const COMPLETION_COUNTERS_PER_STAGE: usize = 2;
const DIAGNOSTIC_COUNTERS: usize = 10;

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

    pub(crate) const fn completion_counter_count(self) -> usize {
        self.completion_count
    }

    pub(crate) const fn diagnostic_counter_count() -> usize {
        DIAGNOSTIC_COUNTERS
    }

    pub(crate) const fn counter_count(self) -> usize {
        self.counter_count
    }
}

#[allow(dead_code)] // Storage fields are connected to checkpoints in Stage 3.
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
            calibration_state: AtomicU8::new(0),
            calibration_epoch: AtomicU16::new(0),
        }
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
}
