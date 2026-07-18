//! Per-request latency contexts and manual checkpoint recording.

use std::fmt;
use std::sync::atomic::Ordering;
use std::time::{Duration, Instant};

use crate::bucket::bucketize;
use crate::init::{RegistryCookie, StageId, Tracegrams};
use crate::recorder::{
    CALIBRATION_COLLECTING, CALIBRATION_FREEZING, CALIBRATION_FROZEN, CalibrationDistribution,
    DiagnosticCounter,
};

/// The terminal outcome of a recorded request.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
#[non_exhaustive]
pub enum Outcome {
    /// The request completed successfully.
    Success,
    /// The request completed with an error.
    Error,
}

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
struct PreviousMark {
    stage: u8,
    cumulative_bucket: u8,
}

const COOKIE_MASK: u64 = 0xffff_ffff;
const CALIBRATION_EPOCH_SHIFT: u32 = 32;
const PREVIOUS_STAGE_SHIFT: u32 = 48;
const PREVIOUS_BUCKET_SHIFT: u32 = 54;
const PREVIOUS_FIELD_MASK: u64 = 0x3f;
const HAS_PREVIOUS_BIT: u64 = 1 << 60;
const IDENTITY_MASK: u64 = (1 << PREVIOUS_STAGE_SHIFT) - 1;

/// Clock-reading state for one linear request path.
///
/// A context is tied to the recorder that created it. It is neither cloneable
/// nor convertible to [`ManualCtx`]. Moving it transfers its single logical
/// ownership, including across threads.
///
/// `Ctx` is intentionally not [`Clone`]:
///
/// ```compile_fail
/// use tracegrams::Tracegrams;
///
/// let mut builder = Tracegrams::builder();
/// builder.stage("only").unwrap();
/// let tracegrams = builder.build().unwrap();
/// let context = tracegrams.start();
/// let _duplicate = context.clone();
/// ```
///
/// Nor is it [`Copy`]:
///
/// ```compile_fail
/// use tracegrams::Tracegrams;
///
/// let mut builder = Tracegrams::builder();
/// builder.stage("only").unwrap();
/// let tracegrams = builder.build().unwrap();
/// let context = tracegrams.start();
/// let _moved = context;
/// let _used_again = context;
/// ```
pub struct Ctx {
    last_timestamp_ns: u64,
    cumulative_ns: u64,
    metadata: u64,
}

impl Ctx {
    fn new(last_timestamp_ns: u64, cookie: RegistryCookie, calibration_epoch: u16) -> Self {
        Self {
            last_timestamp_ns,
            cumulative_ns: 0,
            metadata: u64::from(cookie.raw())
                | (u64::from(calibration_epoch) << CALIBRATION_EPOCH_SHIFT),
        }
    }

    const fn registry_cookie(&self) -> u64 {
        self.metadata & COOKIE_MASK
    }

    const fn calibration_epoch(&self) -> u16 {
        #[allow(clippy::cast_possible_truncation)]
        let epoch = (self.metadata >> CALIBRATION_EPOCH_SHIFT) as u16;
        epoch
    }

    const fn previous(&self) -> Option<PreviousMark> {
        if self.metadata & HAS_PREVIOUS_BIT == 0 {
            return None;
        }
        #[allow(clippy::cast_possible_truncation)]
        let stage = ((self.metadata >> PREVIOUS_STAGE_SHIFT) & PREVIOUS_FIELD_MASK) as u8;
        #[allow(clippy::cast_possible_truncation)]
        let cumulative_bucket =
            ((self.metadata >> PREVIOUS_BUCKET_SHIFT) & PREVIOUS_FIELD_MASK) as u8;
        Some(PreviousMark {
            stage,
            cumulative_bucket,
        })
    }

    fn set_previous(&mut self, previous: PreviousMark) {
        self.metadata = (self.metadata & IDENTITY_MASK)
            | (u64::from(previous.stage) << PREVIOUS_STAGE_SHIFT)
            | (u64::from(previous.cumulative_bucket) << PREVIOUS_BUCKET_SHIFT)
            | HAS_PREVIOUS_BIT;
    }
}

impl fmt::Debug for Ctx {
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        formatter
            .debug_struct("Ctx")
            .field("last_timestamp_ns", &self.last_timestamp_ns)
            .field("cumulative_ns", &self.cumulative_ns)
            .field("calibration_epoch", &self.calibration_epoch())
            .field(
                "previous_stage",
                &self.previous().map(|previous| previous.stage),
            )
            .finish_non_exhaustive()
    }
}

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
struct ClockReading {
    timestamp_ns: u64,
    overflowed: bool,
}

impl ClockReading {
    fn from_elapsed(elapsed: Duration) -> Self {
        match u64::try_from(elapsed.as_nanos()) {
            Ok(timestamp_ns) => Self {
                timestamp_ns,
                overflowed: false,
            },
            Err(_) => Self {
                timestamp_ns: u64::MAX,
                overflowed: true,
            },
        }
    }
}

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
struct MarkAdvance {
    cumulative_ns: u64,
    previous: PreviousMark,
}

/// Caller-timed state for one linear request path.
///
/// A manual context is tied to the recorder that created it. It is neither
/// cloneable nor convertible to the clock-reading context.
pub struct ManualCtx {
    cumulative_ns: u64,
    cookie: RegistryCookie,
    calibration_epoch: u16,
    previous: Option<PreviousMark>,
}

impl fmt::Debug for ManualCtx {
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        formatter
            .debug_struct("ManualCtx")
            .field("cumulative_ns", &self.cumulative_ns)
            .field("calibration_epoch", &self.calibration_epoch)
            .field(
                "previous_stage",
                &self.previous.map(|previous| previous.stage),
            )
            .finish_non_exhaustive()
    }
}

impl Tracegrams {
    /// Starts a wall-clock-timed request with one clock read.
    pub fn start(&self) -> Ctx {
        let reading = self.read_clock();
        Ctx::new(
            reading.timestamp_ns,
            self.inner.cookie,
            self.current_calibration_epoch(),
        )
    }

    /// Records one wall-clock interval ending at `stage`.
    pub fn mark(&self, context: &mut Ctx, stage: StageId) {
        let reading = self.read_clock();
        self.record_clocked_mark(context, stage, reading, None);
    }

    /// Records the final wall-clock interval and outcome, consuming the context.
    pub fn finish(&self, mut context: Ctx, stage: StageId, outcome: Outcome) {
        let reading = self.read_clock();
        self.record_clocked_mark(&mut context, stage, reading, Some(outcome));
    }

    /// Starts a caller-timed request without reading the clock.
    pub fn start_manual(&self) -> ManualCtx {
        ManualCtx {
            cumulative_ns: 0,
            cookie: self.inner.cookie,
            calibration_epoch: self.current_calibration_epoch(),
            previous: None,
        }
    }

    /// Records one caller-supplied elapsed interval at `stage`.
    pub fn record_elapsed(&self, context: &mut ManualCtx, stage: StageId, elapsed: Duration) {
        self.record_manual_mark(context, stage, elapsed, None);
    }

    /// Records the final interval and completion outcome, consuming the context.
    pub fn finish_manual(
        &self,
        mut context: ManualCtx,
        stage: StageId,
        elapsed: Duration,
        outcome: Outcome,
    ) {
        self.record_manual_mark(&mut context, stage, elapsed, Some(outcome));
    }

    fn read_clock(&self) -> ClockReading {
        #[cfg(test)]
        {
            self.inner.clock_reads.fetch_add(1, Ordering::Relaxed);
            if let Some((timestamp_ns, overflowed)) = self
                .inner
                .clock_readings
                .lock()
                .expect("test clock mutex must not be poisoned")
                .pop_front()
            {
                return ClockReading {
                    timestamp_ns,
                    overflowed,
                };
            }
        }

        let now = Instant::now();
        let elapsed = now
            .checked_duration_since(self.inner.clock_epoch)
            .unwrap_or(Duration::ZERO);
        ClockReading::from_elapsed(elapsed)
    }

    fn record_clocked_mark(
        &self,
        context: &mut Ctx,
        stage: StageId,
        reading: ClockReading,
        outcome: Option<Outcome>,
    ) {
        let previous = context.previous();
        if !self.validate_mark(context.registry_cookie(), previous, stage) {
            return;
        }
        let Some(measured_ns) = reading.timestamp_ns.checked_sub(context.last_timestamp_ns) else {
            self.inner
                .increment_diagnostic(DiagnosticCounter::ClockRegressions);
            return;
        };
        let (local_ns, latency_overflowed) = if reading.overflowed {
            (u64::MAX, true)
        } else {
            (measured_ns, false)
        };
        let advance = self.record_valid_mark(
            context.cumulative_ns,
            previous,
            stage,
            local_ns,
            latency_overflowed,
            outcome,
        );

        // Every shared update above precedes advancement of request-local state.
        context.last_timestamp_ns = reading.timestamp_ns;
        context.cumulative_ns = advance.cumulative_ns;
        context.set_previous(advance.previous);
    }

    fn record_manual_mark(
        &self,
        context: &mut ManualCtx,
        stage: StageId,
        elapsed: Duration,
        outcome: Option<Outcome>,
    ) {
        if !self.validate_mark(u64::from(context.cookie.raw()), context.previous, stage) {
            return;
        }
        let (local_ns, latency_overflowed) = match u64::try_from(elapsed.as_nanos()) {
            Ok(value) => (value, false),
            Err(_) => (u64::MAX, true),
        };
        let advance = self.record_valid_mark(
            context.cumulative_ns,
            context.previous,
            stage,
            local_ns,
            latency_overflowed,
            outcome,
        );

        // Every shared update above precedes advancement of request-local state.
        context.cumulative_ns = advance.cumulative_ns;
        context.previous = Some(advance.previous);
    }

    fn validate_mark(
        &self,
        context_cookie: u64,
        previous: Option<PreviousMark>,
        stage: StageId,
    ) -> bool {
        if context_cookie != u64::from(self.inner.cookie.raw()) {
            self.inner
                .increment_diagnostic(DiagnosticCounter::InvalidContextMarks);
            return false;
        }
        if stage.cookie() != self.inner.cookie {
            self.inner
                .increment_diagnostic(DiagnosticCounter::InvalidStageMarks);
            return false;
        }
        if previous.is_some_and(|previous| stage.index() <= usize::from(previous.stage)) {
            self.inner
                .increment_diagnostic(DiagnosticCounter::NonMonotonicMarks);
            return false;
        }
        true
    }

    // Keep the ordered atomic update sequence together: context advancement
    // must remain visibly outside and last in each caller.
    #[allow(clippy::too_many_lines)]
    fn record_valid_mark(
        &self,
        previous_cumulative_ns: u64,
        previous: Option<PreviousMark>,
        stage: StageId,
        local_ns: u64,
        latency_overflowed: bool,
        outcome: Option<Outcome>,
    ) -> MarkAdvance {
        if latency_overflowed {
            self.inner
                .increment_diagnostic(DiagnosticCounter::LatencyOverflows);
        }
        let cumulative_ns = if let Some(value) = previous_cumulative_ns.checked_add(local_ns) {
            value
        } else {
            self.inner
                .increment_diagnostic(DiagnosticCounter::CumulativeOverflows);
            u64::MAX
        };
        let local_bucket = bucketize(local_ns, &self.inner.default_bounds);
        let cumulative_bucket = bucketize(cumulative_ns, &self.inner.default_bounds);

        if let Some(index) = self.inner.layout.local(stage.index(), local_bucket) {
            self.inner.increment(index);
        }
        if let Some(index) = self
            .inner
            .layout
            .cumulative(stage.index(), cumulative_bucket)
        {
            self.inner.increment(index);
        }

        if let Some(previous) = previous {
            let previous_bucket = usize::from(previous.cumulative_bucket);
            if let Some(index) =
                self.inner
                    .layout
                    .cause(stage.index(), previous_bucket, local_bucket)
            {
                self.inner.increment(index);
            }
            if let Some(index) =
                self.inner
                    .layout
                    .incoming(stage.index(), previous_bucket, cumulative_bucket)
            {
                self.inner.increment(index);
            }
        }

        match self.inner.calibration_state.load(Ordering::Acquire) {
            CALIBRATION_COLLECTING => {
                let calibration_local = bucketize(local_ns, &self.inner.calibration_bounds);
                let calibration_after = bucketize(cumulative_ns, &self.inner.calibration_bounds);
                if let Some(index) = self.inner.layout.calibration(
                    stage.index(),
                    CalibrationDistribution::Local,
                    calibration_local,
                ) {
                    self.inner.increment(index);
                }
                if let Some(index) = self.inner.layout.calibration(
                    stage.index(),
                    CalibrationDistribution::CumulativeAfter,
                    calibration_after,
                ) {
                    self.inner.increment(index);
                }
                if previous.is_some() {
                    let calibration_previous =
                        bucketize(previous_cumulative_ns, &self.inner.calibration_bounds);
                    if let Some(index) = self.inner.layout.calibration(
                        stage.index(),
                        CalibrationDistribution::PreviousCumulative,
                        calibration_previous,
                    ) {
                        self.inner.increment(index);
                    }
                }
            }
            CALIBRATION_FREEZING => self
                .inner
                .increment_diagnostic(DiagnosticCounter::CalibrationSamplesSkippedWhileFreezing),
            CALIBRATION_FROZEN => {}
            _ => unreachable!("calibration state has a fixed internal representation"),
        }

        if let Some(outcome) = outcome {
            let error = match outcome {
                Outcome::Success => false,
                Outcome::Error => true,
            };
            if let Some(index) = self.inner.layout.completion(stage.index(), error) {
                self.inner.increment(index);
            }
        }

        MarkAdvance {
            cumulative_ns,
            previous: PreviousMark {
                // Registry construction bounds stage indices to 0..=63.
                #[allow(clippy::cast_possible_truncation)]
                stage: stage.index() as u8,
                #[allow(clippy::cast_possible_truncation)]
                cumulative_bucket: cumulative_bucket as u8,
            },
        }
    }
}

#[cfg(test)]
mod tests {
    use std::mem::size_of;
    use std::sync::atomic::Ordering;

    use super::*;
    use crate::bucket::{BUCKETS, CALIBRATION_BUCKETS};
    use crate::recorder::FixedStorageLayout;

    fn two_stage_recorder() -> (Tracegrams, StageId, StageId) {
        let mut builder = Tracegrams::builder();
        let first = builder.stage("first").unwrap();
        let second = builder.stage("second").unwrap();
        (builder.build().unwrap(), first, second)
    }

    fn count(tracegrams: &Tracegrams, index: usize) -> u64 {
        tracegrams.inner.counters[index].load(Ordering::Relaxed)
    }

    fn total(tracegrams: &Tracegrams, indices: impl Iterator<Item = usize>) -> u64 {
        indices.map(|index| count(tracegrams, index)).sum()
    }

    fn raw_counters(tracegrams: &Tracegrams) -> Vec<u64> {
        tracegrams
            .inner
            .counters
            .iter()
            .map(|counter| counter.load(Ordering::Relaxed))
            .collect()
    }

    fn context_state(context: &ManualCtx) -> (u64, RegistryCookie, u16, Option<PreviousMark>) {
        (
            context.cumulative_ns,
            context.cookie,
            context.calibration_epoch,
            context.previous,
        )
    }

    fn clocked_context_state(context: &Ctx) -> (u64, u64, u64) {
        (
            context.last_timestamp_ns,
            context.cumulative_ns,
            context.metadata,
        )
    }

    fn install_test_clock(tracegrams: &Tracegrams, readings: &[(u64, bool)]) {
        tracegrams.inner.clock_reads.store(0, Ordering::Relaxed);
        let mut queued = tracegrams
            .inner
            .clock_readings
            .lock()
            .expect("test clock mutex must not be poisoned");
        queued.clear();
        queued.extend(readings.iter().copied());
    }

    fn assert_test_clock_consumed(tracegrams: &Tracegrams, expected_reads: u64) {
        assert_eq!(
            tracegrams.inner.clock_reads.load(Ordering::Relaxed),
            expected_reads
        );
        assert!(
            tracegrams
                .inner
                .clock_readings
                .lock()
                .expect("test clock mutex must not be poisoned")
                .is_empty()
        );
    }

    fn assert_only_diagnostic_changed(
        tracegrams: &Tracegrams,
        before: &[u64],
        diagnostic: DiagnosticCounter,
    ) {
        let mut expected = before.to_vec();
        expected[tracegrams.inner.layout.diagnostic(diagnostic)] += 1;
        assert_eq!(raw_counters(tracegrams), expected);
    }

    #[derive(Default)]
    struct ReferenceContext {
        cumulative_ns: u64,
        previous_cumulative_bucket: Option<usize>,
    }

    fn record_reference(
        counters: &mut [u64],
        layout: FixedStorageLayout,
        context: &mut ReferenceContext,
        stage: usize,
        local_ns: u64,
        outcome: Option<Outcome>,
    ) {
        let cumulative_ns = context.cumulative_ns.checked_add(local_ns).unwrap();
        let local_bucket = bucketize(local_ns, crate::bucket::default_bounds());
        let cumulative_bucket = bucketize(cumulative_ns, crate::bucket::default_bounds());
        counters[layout.local(stage, local_bucket).unwrap()] += 1;
        counters[layout.cumulative(stage, cumulative_bucket).unwrap()] += 1;

        if let Some(previous_bucket) = context.previous_cumulative_bucket {
            counters[layout.cause(stage, previous_bucket, local_bucket).unwrap()] += 1;
            counters[layout
                .incoming(stage, previous_bucket, cumulative_bucket)
                .unwrap()] += 1;
        }

        let calibration_local = bucketize(local_ns, crate::bucket::calibration_bounds());
        let calibration_after = bucketize(cumulative_ns, crate::bucket::calibration_bounds());
        counters[layout
            .calibration(stage, CalibrationDistribution::Local, calibration_local)
            .unwrap()] += 1;
        counters[layout
            .calibration(
                stage,
                CalibrationDistribution::CumulativeAfter,
                calibration_after,
            )
            .unwrap()] += 1;
        if context.previous_cumulative_bucket.is_some() {
            let calibration_previous =
                bucketize(context.cumulative_ns, crate::bucket::calibration_bounds());
            counters[layout
                .calibration(
                    stage,
                    CalibrationDistribution::PreviousCumulative,
                    calibration_previous,
                )
                .unwrap()] += 1;
        }
        if let Some(outcome) = outcome {
            counters[layout
                .completion(stage, matches!(outcome, Outcome::Error))
                .unwrap()] += 1;
        }

        context.cumulative_ns = cumulative_ns;
        context.previous_cumulative_bucket = Some(cumulative_bucket);
    }

    #[test]
    fn packed_clocked_metadata_preserves_every_field() {
        let (tracegrams, _, _) = two_stage_recorder();
        tracegrams
            .inner
            .calibration_epoch
            .store(u16::MAX, Ordering::Relaxed);
        tracegrams
            .inner
            .calibration_state
            .store(CALIBRATION_FROZEN, Ordering::Release);
        install_test_clock(&tracegrams, &[(u64::MAX, false)]);
        let mut context = tracegrams.start();

        context.set_previous(PreviousMark {
            stage: 63,
            cumulative_bucket: 63,
        });

        assert_eq!(size_of::<Ctx>(), 3 * size_of::<u64>());
        assert_eq!(
            context.registry_cookie(),
            u64::from(tracegrams.inner.cookie.raw())
        );
        assert_eq!(context.calibration_epoch(), u16::MAX);
        assert_eq!(
            context.previous(),
            Some(PreviousMark {
                stage: 63,
                cumulative_bucket: 63,
            })
        );
        assert_eq!(context.last_timestamp_ns, u64::MAX);
        assert_test_clock_consumed(&tracegrams, 1);
    }

    #[test]
    fn deterministic_clocked_recording_matches_manual_after_moving_threads() {
        let (clocked, clocked_first, clocked_second) = two_stage_recorder();
        install_test_clock(&clocked, &[(1_000, false), (1_100, false), (1_300, false)]);

        let context = clocked.start();
        let worker_tracegrams = clocked.clone();
        std::thread::spawn(move || {
            let mut context = context;
            worker_tracegrams.mark(&mut context, clocked_first);
            worker_tracegrams.finish(context, clocked_second, Outcome::Success);
        })
        .join()
        .unwrap();

        let (manual, manual_first, manual_second) = two_stage_recorder();
        let mut context = manual.start_manual();
        manual.record_elapsed(&mut context, manual_first, Duration::from_nanos(100));
        manual.finish_manual(
            context,
            manual_second,
            Duration::from_nanos(200),
            Outcome::Success,
        );

        assert_eq!(raw_counters(&clocked), raw_counters(&manual));
        assert_test_clock_consumed(&clocked, 3);
    }

    #[test]
    fn clocked_marks_reuse_context_stage_and_order_validation() {
        let (tracegrams, first, second) = two_stage_recorder();
        let (foreign, foreign_first, _) = two_stage_recorder();
        install_test_clock(
            &tracegrams,
            &[
                (100, false),
                (200, false),
                (300, false),
                (400, false),
                (500, false),
            ],
        );
        install_test_clock(&foreign, &[(50, false)]);
        let mut context = tracegrams.start();

        let before_context = clocked_context_state(&context);
        let before_counters = raw_counters(&tracegrams);
        tracegrams.mark(&mut context, foreign_first);
        assert_eq!(clocked_context_state(&context), before_context);
        assert_only_diagnostic_changed(
            &tracegrams,
            &before_counters,
            DiagnosticCounter::InvalidStageMarks,
        );

        tracegrams.mark(&mut context, second);
        let before_context = clocked_context_state(&context);
        let before_counters = raw_counters(&tracegrams);
        tracegrams.mark(&mut context, first);
        assert_eq!(clocked_context_state(&context), before_context);
        assert_only_diagnostic_changed(
            &tracegrams,
            &before_counters,
            DiagnosticCounter::NonMonotonicMarks,
        );

        let mut foreign_context = foreign.start();
        let before_context = clocked_context_state(&foreign_context);
        let before_counters = raw_counters(&tracegrams);
        tracegrams.mark(&mut foreign_context, first);
        assert_eq!(clocked_context_state(&foreign_context), before_context);
        assert_only_diagnostic_changed(
            &tracegrams,
            &before_counters,
            DiagnosticCounter::InvalidContextMarks,
        );
        assert_test_clock_consumed(&tracegrams, 5);
        assert_test_clock_consumed(&foreign, 1);
    }

    #[test]
    fn clock_regression_changes_only_its_diagnostic_counter() {
        let (tracegrams, first, second) = two_stage_recorder();
        install_test_clock(&tracegrams, &[(100, false), (200, false), (150, false)]);
        let mut context = tracegrams.start();
        tracegrams.mark(&mut context, first);
        let before_context = clocked_context_state(&context);
        let before_counters = raw_counters(&tracegrams);

        tracegrams.mark(&mut context, second);

        assert_eq!(clocked_context_state(&context), before_context);
        assert_only_diagnostic_changed(
            &tracegrams,
            &before_counters,
            DiagnosticCounter::ClockRegressions,
        );
        assert_test_clock_consumed(&tracegrams, 3);
    }

    #[test]
    fn clock_timestamp_and_cumulative_overflow_are_total() {
        let (tracegrams, first, second) = two_stage_recorder();
        install_test_clock(
            &tracegrams,
            &[(0, false), (u64::MAX, false), (u64::MAX, true)],
        );
        let mut context = tracegrams.start();
        tracegrams.mark(&mut context, first);
        tracegrams.finish(context, second, Outcome::Error);

        assert_eq!(
            count(
                &tracegrams,
                tracegrams
                    .inner
                    .layout
                    .local(second.index(), BUCKETS - 1)
                    .unwrap(),
            ),
            1
        );
        assert_eq!(
            count(
                &tracegrams,
                tracegrams
                    .inner
                    .layout
                    .diagnostic(DiagnosticCounter::LatencyOverflows),
            ),
            1
        );
        assert_eq!(
            count(
                &tracegrams,
                tracegrams
                    .inner
                    .layout
                    .diagnostic(DiagnosticCounter::CumulativeOverflows),
            ),
            1
        );
        assert_test_clock_consumed(&tracegrams, 3);
    }

    #[test]
    fn clock_normalization_checks_the_full_duration_range() {
        assert_eq!(
            ClockReading::from_elapsed(Duration::from_nanos(u64::MAX)),
            ClockReading {
                timestamp_ns: u64::MAX,
                overflowed: false,
            }
        );
        assert_eq!(
            ClockReading::from_elapsed(Duration::new(u64::MAX, 999_999_999)),
            ClockReading {
                timestamp_ns: u64::MAX,
                overflowed: true,
            }
        );
    }

    #[test]
    fn concurrent_independent_contexts_sum_into_shared_counters() {
        const WORKERS: u64 = 8;
        const REQUESTS_PER_WORKER: u64 = 200;

        let (tracegrams, first, second) = two_stage_recorder();
        let mut workers = Vec::new();
        for _ in 0..WORKERS {
            let tracegrams = tracegrams.clone();
            workers.push(std::thread::spawn(move || {
                for _ in 0..REQUESTS_PER_WORKER {
                    let mut context = tracegrams.start_manual();
                    tracegrams.record_elapsed(&mut context, first, Duration::from_nanos(100));
                    tracegrams.finish_manual(
                        context,
                        second,
                        Duration::from_nanos(200),
                        Outcome::Success,
                    );
                }
            }));
        }
        for worker in workers {
            worker.join().unwrap();
        }

        let mut reference = vec![0; tracegrams.inner.layout.counter_count()];
        let mut context = ReferenceContext::default();
        record_reference(
            &mut reference,
            tracegrams.inner.layout,
            &mut context,
            first.index(),
            100,
            None,
        );
        record_reference(
            &mut reference,
            tracegrams.inner.layout,
            &mut context,
            second.index(),
            200,
            Some(Outcome::Success),
        );
        let expected_requests = WORKERS * REQUESTS_PER_WORKER;
        for counter in &mut reference {
            *counter *= expected_requests;
        }
        assert_eq!(raw_counters(&tracegrams), reference);
    }

    #[test]
    fn sequential_atomic_recording_matches_the_reference_recorder() {
        let mut builder = Tracegrams::builder();
        let stages = [
            builder.stage("parse").unwrap(),
            builder.stage("cache").unwrap(),
            builder.stage("db").unwrap(),
        ];
        let tracegrams = builder.build().unwrap();
        let layout = tracegrams.inner.layout;
        let mut expected = vec![0; layout.counter_count()];

        let mut actual = tracegrams.start_manual();
        let mut reference = ReferenceContext::default();
        for (stage, elapsed) in [(0, 100), (1, 200)] {
            tracegrams.record_elapsed(&mut actual, stages[stage], Duration::from_nanos(elapsed));
            record_reference(&mut expected, layout, &mut reference, stage, elapsed, None);
        }
        tracegrams.finish_manual(
            actual,
            stages[2],
            Duration::from_nanos(300),
            Outcome::Success,
        );
        record_reference(
            &mut expected,
            layout,
            &mut reference,
            2,
            300,
            Some(Outcome::Success),
        );

        tracegrams.finish_manual(
            tracegrams.start_manual(),
            stages[0],
            Duration::from_secs(10),
            Outcome::Error,
        );
        let mut reference = ReferenceContext::default();
        record_reference(
            &mut expected,
            layout,
            &mut reference,
            0,
            10_000_000_000,
            Some(Outcome::Error),
        );

        let mut actual = tracegrams.start_manual();
        tracegrams.record_elapsed(&mut actual, stages[0], Duration::from_nanos(50));
        let mut reference = ReferenceContext::default();
        record_reference(&mut expected, layout, &mut reference, 0, 50, None);
        tracegrams.finish_manual(
            actual,
            stages[2],
            Duration::from_nanos(75),
            Outcome::Success,
        );
        record_reference(
            &mut expected,
            layout,
            &mut reference,
            2,
            75,
            Some(Outcome::Success),
        );

        let mut actual = tracegrams.start_manual();
        tracegrams.record_elapsed(&mut actual, stages[1], Duration::from_nanos(135));
        let mut reference = ReferenceContext::default();
        record_reference(&mut expected, layout, &mut reference, 1, 135, None);

        assert_eq!(raw_counters(&tracegrams), expected);
    }

    #[test]
    fn finish_only_outcome_changes_only_completion_counters() {
        let (successful, success_stage, _) = two_stage_recorder();
        successful.finish_manual(
            successful.start_manual(),
            success_stage,
            Duration::from_nanos(100),
            Outcome::Success,
        );
        let (failed, error_stage, _) = two_stage_recorder();
        failed.finish_manual(
            failed.start_manual(),
            error_stage,
            Duration::from_nanos(100),
            Outcome::Error,
        );

        let success_completion = successful
            .inner
            .layout
            .completion(success_stage.index(), false)
            .unwrap();
        let error_completion = successful
            .inner
            .layout
            .completion(success_stage.index(), true)
            .unwrap();
        let successful_counters = raw_counters(&successful);
        let failed_counters = raw_counters(&failed);
        for (index, (success, error)) in
            successful_counters.iter().zip(&failed_counters).enumerate()
        {
            let expected = if index == success_completion {
                (1, 0)
            } else if index == error_completion {
                (0, 1)
            } else {
                (*success, *success)
            };
            assert_eq!((*success, *error), expected, "counter {index}");
        }
    }

    #[test]
    fn cumulative_overflow_saturates_into_the_last_bucket() {
        let (tracegrams, first, second) = two_stage_recorder();
        let mut context = tracegrams.start_manual();

        tracegrams.record_elapsed(&mut context, first, Duration::from_nanos(u64::MAX));
        tracegrams.record_elapsed(&mut context, second, Duration::from_nanos(1));

        assert_eq!(context.cumulative_ns, u64::MAX);
        assert_eq!(
            count(
                &tracegrams,
                tracegrams
                    .inner
                    .layout
                    .cumulative(second.index(), BUCKETS - 1)
                    .unwrap(),
            ),
            1
        );
        assert_eq!(
            count(
                &tracegrams,
                tracegrams
                    .inner
                    .layout
                    .incoming(second.index(), BUCKETS - 1, BUCKETS - 1)
                    .unwrap(),
            ),
            1
        );
        assert_eq!(
            count(
                &tracegrams,
                tracegrams
                    .inner
                    .layout
                    .diagnostic(DiagnosticCounter::CumulativeOverflows),
            ),
            1
        );
        assert_eq!(
            count(
                &tracegrams,
                tracegrams
                    .inner
                    .layout
                    .diagnostic(DiagnosticCounter::LatencyOverflows),
            ),
            0
        );
    }

    #[test]
    fn duration_and_bucket_overflow_are_total() {
        let (tracegrams, first, _) = two_stage_recorder();
        let mut context = tracegrams.start_manual();

        tracegrams.record_elapsed(&mut context, first, Duration::new(u64::MAX, 999_999_999));

        assert_eq!(context.cumulative_ns, u64::MAX);
        assert_eq!(
            count(
                &tracegrams,
                tracegrams
                    .inner
                    .layout
                    .local(first.index(), BUCKETS - 1)
                    .unwrap(),
            ),
            1
        );
        assert_eq!(
            count(
                &tracegrams,
                tracegrams
                    .inner
                    .layout
                    .calibration(
                        first.index(),
                        CalibrationDistribution::Local,
                        CALIBRATION_BUCKETS - 1,
                    )
                    .unwrap(),
            ),
            1
        );
        assert_eq!(
            count(
                &tracegrams,
                tracegrams
                    .inner
                    .layout
                    .diagnostic(DiagnosticCounter::LatencyOverflows),
            ),
            1
        );
        assert_eq!(
            count(
                &tracegrams,
                tracegrams
                    .inner
                    .layout
                    .diagnostic(DiagnosticCounter::CumulativeOverflows),
            ),
            0
        );

        let (bucket_only, first, _) = two_stage_recorder();
        let mut context = bucket_only.start_manual();
        bucket_only.record_elapsed(&mut context, first, Duration::from_secs(10));
        assert_eq!(
            count(
                &bucket_only,
                bucket_only
                    .inner
                    .layout
                    .local(first.index(), BUCKETS - 1)
                    .unwrap(),
            ),
            1
        );
        assert_eq!(
            count(
                &bucket_only,
                bucket_only
                    .inner
                    .layout
                    .diagnostic(DiagnosticCounter::LatencyOverflows),
            ),
            0
        );
    }

    #[test]
    fn saturated_hot_cell_increments_counter_overflow_diagnostic() {
        let (tracegrams, first, _) = two_stage_recorder();
        let bucket = bucketize(100, &tracegrams.inner.default_bounds);
        let local = tracegrams
            .inner
            .layout
            .local(first.index(), bucket)
            .unwrap();
        tracegrams.inner.counters[local].store(u64::MAX, Ordering::Relaxed);
        let mut context = tracegrams.start_manual();

        tracegrams.record_elapsed(&mut context, first, Duration::from_nanos(100));

        assert_eq!(count(&tracegrams, local), u64::MAX);
        assert_eq!(
            count(
                &tracegrams,
                tracegrams
                    .inner
                    .layout
                    .diagnostic(DiagnosticCounter::CounterOverflows),
            ),
            1
        );
    }

    #[test]
    fn rejected_finish_does_not_record_completion() {
        let (tracegrams, _, _) = two_stage_recorder();
        let (_, foreign, _) = two_stage_recorder();
        let before = raw_counters(&tracegrams);

        tracegrams.finish_manual(
            tracegrams.start_manual(),
            foreign,
            Duration::from_nanos(100),
            Outcome::Success,
        );

        assert_only_diagnostic_changed(&tracegrams, &before, DiagnosticCounter::InvalidStageMarks);
    }

    #[test]
    fn rejected_marks_change_only_their_diagnostic_counter() {
        let (first_recorder, first, second) = two_stage_recorder();
        let (foreign_recorder, foreign_first, _) = two_stage_recorder();

        let mut foreign_context = foreign_recorder.start_manual();
        let before_context = context_state(&foreign_context);
        let before_counters = raw_counters(&first_recorder);
        first_recorder.record_elapsed(&mut foreign_context, first, Duration::from_nanos(100));
        assert_eq!(context_state(&foreign_context), before_context);
        assert_only_diagnostic_changed(
            &first_recorder,
            &before_counters,
            DiagnosticCounter::InvalidContextMarks,
        );

        let mut context = first_recorder.start_manual();
        let before_context = context_state(&context);
        let before_counters = raw_counters(&first_recorder);
        first_recorder.record_elapsed(&mut context, foreign_first, Duration::from_nanos(100));
        assert_eq!(context_state(&context), before_context);
        assert_only_diagnostic_changed(
            &first_recorder,
            &before_counters,
            DiagnosticCounter::InvalidStageMarks,
        );

        first_recorder.record_elapsed(&mut context, second, Duration::from_nanos(100));
        let before_context = context_state(&context);
        let before_counters = raw_counters(&first_recorder);
        first_recorder.record_elapsed(&mut context, second, Duration::from_nanos(100));
        assert_eq!(context_state(&context), before_context);
        assert_only_diagnostic_changed(
            &first_recorder,
            &before_counters,
            DiagnosticCounter::NonMonotonicMarks,
        );

        let before_counters = raw_counters(&first_recorder);
        first_recorder.record_elapsed(&mut context, first, Duration::from_nanos(100));
        assert_eq!(context_state(&context), before_context);
        assert_only_diagnostic_changed(
            &first_recorder,
            &before_counters,
            DiagnosticCounter::NonMonotonicMarks,
        );
    }

    #[test]
    fn skipped_stages_update_the_destination_matrices() {
        let mut builder = Tracegrams::builder();
        let parse = builder.stage("parse").unwrap();
        let skipped = builder.stage("skipped").unwrap();
        let db = builder.stage("db").unwrap();
        let tracegrams = builder.build().unwrap();
        let mut context = tracegrams.start_manual();

        tracegrams.record_elapsed(&mut context, parse, Duration::from_nanos(100));
        tracegrams.finish_manual(context, db, Duration::from_nanos(200), Outcome::Success);

        let previous_bucket = bucketize(100, &tracegrams.inner.default_bounds);
        let local_bucket = bucketize(200, &tracegrams.inner.default_bounds);
        let after_bucket = bucketize(300, &tracegrams.inner.default_bounds);
        assert_eq!(
            count(
                &tracegrams,
                tracegrams
                    .inner
                    .layout
                    .cause(db.index(), previous_bucket, local_bucket)
                    .unwrap(),
            ),
            1
        );
        assert_eq!(
            count(
                &tracegrams,
                tracegrams
                    .inner
                    .layout
                    .incoming(db.index(), previous_bucket, after_bucket)
                    .unwrap(),
            ),
            1
        );
        assert_eq!(
            count(
                &tracegrams,
                tracegrams
                    .inner
                    .layout
                    .cause(skipped.index(), previous_bucket, local_bucket)
                    .unwrap(),
            ),
            0
        );
        assert_eq!(
            count(
                &tracegrams,
                tracegrams
                    .inner
                    .layout
                    .incoming(skipped.index(), previous_bucket, after_bucket)
                    .unwrap(),
            ),
            0
        );
        assert_eq!(
            tracegrams
                .inner
                .counters
                .iter()
                .map(|counter| counter.load(Ordering::Relaxed))
                .sum::<u64>(),
            12
        );
    }

    #[test]
    fn freezing_mark_skips_the_entire_calibration_sample() {
        let (tracegrams, first, _) = two_stage_recorder();
        tracegrams
            .inner
            .calibration_state
            .store(CALIBRATION_FREEZING, Ordering::Release);

        tracegrams.record_elapsed(
            &mut tracegrams.start_manual(),
            first,
            Duration::from_nanos(1_000),
        );

        let ordinary_bucket = bucketize(1_000, &tracegrams.inner.default_bounds);
        assert_eq!(
            count(
                &tracegrams,
                tracegrams
                    .inner
                    .layout
                    .local(first.index(), ordinary_bucket)
                    .unwrap(),
            ),
            1
        );
        for distribution in [
            CalibrationDistribution::Local,
            CalibrationDistribution::CumulativeAfter,
            CalibrationDistribution::PreviousCumulative,
        ] {
            assert_eq!(
                total(
                    &tracegrams,
                    (0..CALIBRATION_BUCKETS).map(|bucket| {
                        tracegrams
                            .inner
                            .layout
                            .calibration(first.index(), distribution, bucket)
                            .unwrap()
                    }),
                ),
                0
            );
        }
        assert_eq!(
            count(
                &tracegrams,
                tracegrams
                    .inner
                    .layout
                    .diagnostic(DiagnosticCounter::CalibrationSamplesSkippedWhileFreezing,),
            ),
            1
        );
    }

    #[test]
    fn first_mark_uses_a_clean_sentinel() {
        let (tracegrams, first, _) = two_stage_recorder();
        let mut context = tracegrams.start_manual();
        tracegrams.record_elapsed(&mut context, first, Duration::from_nanos(1_000));

        let ordinary_bucket = bucketize(1_000, &tracegrams.inner.default_bounds);
        let calibration_bucket = bucketize(1_000, &tracegrams.inner.calibration_bounds);
        assert_eq!(
            count(
                &tracegrams,
                tracegrams
                    .inner
                    .layout
                    .local(first.index(), ordinary_bucket)
                    .unwrap(),
            ),
            1
        );
        assert_eq!(
            count(
                &tracegrams,
                tracegrams
                    .inner
                    .layout
                    .cumulative(first.index(), ordinary_bucket)
                    .unwrap(),
            ),
            1
        );
        assert_eq!(
            total(
                &tracegrams,
                (0..BUCKETS).map(|bucket| {
                    tracegrams
                        .inner
                        .layout
                        .cause(first.index(), ordinary_bucket, bucket)
                        .unwrap()
                }),
            ),
            0
        );
        assert_eq!(
            total(
                &tracegrams,
                (0..CALIBRATION_BUCKETS).map(|bucket| {
                    tracegrams
                        .inner
                        .layout
                        .calibration(
                            first.index(),
                            CalibrationDistribution::PreviousCumulative,
                            bucket,
                        )
                        .unwrap()
                }),
            ),
            0
        );
        assert_eq!(
            count(
                &tracegrams,
                tracegrams
                    .inner
                    .layout
                    .calibration(
                        first.index(),
                        CalibrationDistribution::Local,
                        calibration_bucket,
                    )
                    .unwrap(),
            ),
            1
        );
        assert_eq!(
            tracegrams
                .inner
                .counters
                .iter()
                .map(|counter| counter.load(Ordering::Relaxed))
                .sum::<u64>(),
            4
        );
    }
}
