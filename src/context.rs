//! Per-request latency contexts and manual checkpoint recording.

use std::fmt;
use std::sync::atomic::Ordering;
use std::time::Duration;

use crate::bucket::bucketize;
use crate::init::{RegistryCookie, StageId, Tracegrams};
use crate::recorder::{CalibrationDistribution, DiagnosticCounter};

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
    /// Starts a caller-timed request without reading the clock.
    pub fn start_manual(&self) -> ManualCtx {
        ManualCtx {
            cumulative_ns: 0,
            cookie: self.inner.cookie,
            calibration_epoch: self.inner.calibration_epoch.load(Ordering::Acquire),
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

    // Keep the ordered atomic update sequence together: context advancement
    // must remain visibly last.
    #[allow(clippy::too_many_lines)]
    fn record_manual_mark(
        &self,
        context: &mut ManualCtx,
        stage: StageId,
        elapsed: Duration,
        outcome: Option<Outcome>,
    ) {
        if context.cookie != self.inner.cookie {
            self.inner
                .increment_diagnostic(DiagnosticCounter::InvalidContextMarks);
            return;
        }
        if stage.cookie() != self.inner.cookie {
            self.inner
                .increment_diagnostic(DiagnosticCounter::InvalidStageMarks);
            return;
        }
        if context
            .previous
            .is_some_and(|previous| stage.index() <= usize::from(previous.stage))
        {
            self.inner
                .increment_diagnostic(DiagnosticCounter::NonMonotonicMarks);
            return;
        }

        let local_ns = if let Ok(value) = u64::try_from(elapsed.as_nanos()) {
            value
        } else {
            self.inner
                .increment_diagnostic(DiagnosticCounter::LatencyOverflows);
            u64::MAX
        };
        let cumulative_ns = if let Some(value) = context.cumulative_ns.checked_add(local_ns) {
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

        if let Some(previous) = context.previous {
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
        if context.previous.is_some() {
            let calibration_previous =
                bucketize(context.cumulative_ns, &self.inner.calibration_bounds);
            if let Some(index) = self.inner.layout.calibration(
                stage.index(),
                CalibrationDistribution::PreviousCumulative,
                calibration_previous,
            ) {
                self.inner.increment(index);
            }
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

        // Every shared update above precedes advancement of request-local state.
        context.cumulative_ns = cumulative_ns;
        context.previous = Some(PreviousMark {
            // Registry construction bounds stage indices to 0..=63.
            #[allow(clippy::cast_possible_truncation)]
            stage: stage.index() as u8,
            #[allow(clippy::cast_possible_truncation)]
            cumulative_bucket: cumulative_bucket as u8,
        });
    }
}

#[cfg(test)]
mod tests {
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
