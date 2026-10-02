//! Calibration collection, readiness, and explicit-freeze contract tests.

use std::sync::atomic::{AtomicBool, AtomicUsize, Ordering};
use std::sync::{Arc, Barrier};
use std::time::Duration;

use tracegrams::{
    CalibrationPopulation, CalibrationState, CalibrationTerminal, CalibrationThresholdAvailability,
    FreezeError, Outcome, Tracegrams,
};

#[test]
fn freeze_errors_display_exact_messages() {
    let mut builder = Tracegrams::builder();
    let first = builder.stage("first").unwrap();
    let second = builder.stage("second").unwrap();
    let mut cases = vec![
        (
            FreezeError::NotReady {
                stage: second,
                population: CalibrationPopulation::PreviousCumulative,
                samples: 17,
                minimum: 23,
            },
            "StageId(1) PreviousCumulative has 17 calibration samples; 23 required",
        ),
        (
            FreezeError::NotReady {
                stage: first,
                population: CalibrationPopulation::CumulativeAfter,
                samples: 0,
                minimum: 1,
            },
            "StageId(0) CumulativeAfter has 0 calibration samples; 1 required",
        ),
        (
            FreezeError::AlreadyFreezing,
            "calibration is already freezing",
        ),
        (FreezeError::AlreadyFrozen, "calibration is already frozen"),
    ];
    for (duration, expected) in [
        (
            Duration::from_nanos(50),
            "calibration range is insufficient for StageId(1) Local: Lower",
        ),
        (
            Duration::from_secs(10),
            "calibration range is insufficient for StageId(1) Local: Upper",
        ),
    ] {
        let mut builder = Tracegrams::builder();
        let first = builder.stage("first").unwrap();
        let second = builder.stage("second").unwrap();
        let tracegrams = builder.build().unwrap();
        tracegrams.record_elapsed(
            &mut tracegrams.start_manual(),
            first,
            Duration::from_nanos(100),
        );
        tracegrams.record_elapsed(&mut tracegrams.start_manual(), second, duration);
        cases.push((tracegrams.try_freeze_calibration(1).unwrap_err(), expected));
    }

    for (error, expected) in cases {
        assert_eq!(error.to_string(), expected);
    }
}

#[test]
fn freeze_publishes_one_complete_relaxed_bundle() {
    let mut builder = Tracegrams::builder();
    let entry = builder.stage("entry").unwrap();
    let db = builder.stage("db").unwrap();
    let tracegrams = builder.build().unwrap();

    for _ in 0..3 {
        let mut context = tracegrams.start_manual();
        tracegrams.record_elapsed(&mut context, entry, Duration::from_nanos(100));
        tracegrams.record_elapsed(&mut context, db, Duration::from_nanos(200));
    }

    let report = tracegrams.try_freeze_calibration(3).unwrap();
    assert_eq!(report.stages.len(), 2);
    let entry_report = report.stage(entry).unwrap();
    assert_eq!(entry_report.local.population, CalibrationPopulation::Local);
    assert_eq!(entry_report.local.samples, 3);
    assert_eq!(
        entry_report.local.availability(),
        CalibrationThresholdAvailability::Available
    );
    let estimate = entry_report.local.estimate.unwrap();
    assert_eq!(estimate.rank, 3);
    assert_eq!(estimate.samples, 3);
    assert_eq!(estimate.lower_bound, Some(100));
    assert_eq!(estimate.upper_bound, Some(108));
    assert_eq!(estimate.value, Some(104));
    assert_eq!(estimate.terminal, CalibrationTerminal::Finite);
    assert_eq!(
        entry_report.previous_cumulative.availability(),
        CalibrationThresholdAvailability::AbsentNoPredecessor
    );
    assert_eq!(entry_report.previous_cumulative.samples, 0);
    assert_eq!(entry_report.previous_cumulative.estimate, None);
    let db_previous = report.stage(db).unwrap().previous_cumulative;
    assert_eq!(db_previous.samples, 3);
    assert_eq!(
        db_previous.availability(),
        CalibrationThresholdAvailability::Available
    );

    let frozen = tracegrams.snapshot_relaxed();
    assert_eq!(frozen.calibration_state(), CalibrationState::Frozen);
    assert_eq!(frozen.calibration_report(), Some(&report));

    let before = frozen.sample_counts(entry).unwrap().calibration_local;
    tracegrams.record_elapsed(
        &mut tracegrams.start_manual(),
        entry,
        Duration::from_nanos(300),
    );
    let after = tracegrams.snapshot_relaxed();
    assert_eq!(
        after.sample_counts(entry).unwrap().calibration_local,
        before
    );
    assert_eq!(
        after
            .diagnostics()
            .calibration_samples_skipped_while_freezing,
        0
    );
    assert_eq!(
        tracegrams.try_freeze_calibration(3),
        Err(FreezeError::AlreadyFrozen)
    );
}

#[test]
fn empty_required_population_is_not_ready_even_with_a_zero_minimum() {
    let mut builder = Tracegrams::builder();
    let first = builder.stage("first").unwrap();
    let never_reached = builder.stage("never-reached").unwrap();
    let tracegrams = builder.build().unwrap();
    tracegrams.record_elapsed(
        &mut tracegrams.start_manual(),
        first,
        Duration::from_nanos(200),
    );

    assert_eq!(
        tracegrams.try_freeze_calibration(0),
        Err(FreezeError::NotReady {
            stage: never_reached,
            population: CalibrationPopulation::Local,
            samples: 0,
            minimum: 0,
        })
    );
    assert_eq!(
        tracegrams.snapshot_relaxed().calibration_state(),
        CalibrationState::Collecting
    );
}

#[test]
fn observed_previous_population_must_reach_the_minimum() {
    let mut builder = Tracegrams::builder();
    let first = builder.stage("first").unwrap();
    let destination = builder.stage("destination").unwrap();
    let tracegrams = builder.build().unwrap();

    // First marks alone make every required population ready while the
    // destination's previous-cumulative population stays absent.
    for stage in [first, destination] {
        for _ in 0..3 {
            tracegrams.record_elapsed(
                &mut tracegrams.start_manual(),
                stage,
                Duration::from_nanos(200),
            );
        }
    }
    let record_pair = || {
        let mut context = tracegrams.start_manual();
        tracegrams.record_elapsed(&mut context, first, Duration::from_nanos(100));
        tracegrams.record_elapsed(&mut context, destination, Duration::from_nanos(200));
    };

    record_pair();
    assert_eq!(
        tracegrams.try_freeze_calibration(3),
        Err(FreezeError::NotReady {
            stage: destination,
            population: CalibrationPopulation::PreviousCumulative,
            samples: 1,
            minimum: 3,
        })
    );
    assert_eq!(
        tracegrams.snapshot_relaxed().calibration_state(),
        CalibrationState::Collecting
    );

    record_pair();
    record_pair();
    let report = tracegrams.try_freeze_calibration(3).unwrap();
    assert_eq!(
        report
            .stage(destination)
            .unwrap()
            .previous_cumulative
            .samples,
        3
    );
    assert_eq!(
        report
            .stage(first)
            .unwrap()
            .previous_cumulative
            .availability(),
        CalibrationThresholdAvailability::AbsentNoPredecessor
    );
}

#[test]
fn lower_terminal_failure_retains_a_report_and_allows_finite_retry() {
    let mut builder = Tracegrams::builder();
    let stage = builder.stage("stage").unwrap();
    let tracegrams = builder.build().unwrap();
    tracegrams.record_elapsed(
        &mut tracegrams.start_manual(),
        stage,
        Duration::from_nanos(50),
    );

    let error = tracegrams.try_freeze_calibration(1).unwrap_err();
    let FreezeError::CalibrationRangeInsufficient {
        stage: failed_stage,
        population,
        terminal,
        report,
    } = error
    else {
        panic!("expected a typed lower-terminal failure");
    };
    assert_eq!(failed_stage, stage);
    assert_eq!(population, CalibrationPopulation::Local);
    assert_eq!(terminal, CalibrationTerminal::Lower);
    let local = report.stage(stage).unwrap().local;
    assert_eq!(
        local.availability(),
        CalibrationThresholdAvailability::RangeInsufficient
    );
    let estimate = local.estimate.unwrap();
    assert_eq!(estimate.terminal, CalibrationTerminal::Lower);
    assert_eq!(estimate.lower_bound, None);
    assert_eq!(estimate.upper_bound, Some(100));
    assert_eq!(estimate.value, None);
    assert_eq!(estimate.max_relative_error, None);
    let failed_snapshot = tracegrams.snapshot_relaxed();
    assert_eq!(
        failed_snapshot.calibration_state(),
        CalibrationState::Collecting
    );
    assert_eq!(failed_snapshot.calibration_report(), None);

    for _ in 0..100 {
        tracegrams.record_elapsed(
            &mut tracegrams.start_manual(),
            stage,
            Duration::from_nanos(100),
        );
    }
    let retry = tracegrams.try_freeze_calibration(1).unwrap();
    let estimate = retry.stage(stage).unwrap().local.estimate.unwrap();
    assert_eq!(estimate.terminal, CalibrationTerminal::Finite);
    assert!(estimate.value.is_some());
    assert!(estimate.max_relative_error.is_some());
}

#[test]
fn upper_terminal_failure_names_the_out_of_range_stage() {
    let mut builder = Tracegrams::builder();
    let stable = builder.stage("stable").unwrap();
    let out_of_range = builder.stage("out-of-range").unwrap();
    let tracegrams = builder.build().unwrap();
    let mut context = tracegrams.start_manual();
    tracegrams.record_elapsed(&mut context, stable, Duration::from_nanos(100));
    tracegrams.record_elapsed(&mut context, out_of_range, Duration::from_secs(10));

    assert!(matches!(
        tracegrams.try_freeze_calibration(1),
        Err(FreezeError::CalibrationRangeInsufficient {
            stage,
            terminal: CalibrationTerminal::Upper,
            ..
        }) if stage == out_of_range
    ));
    let snapshot = tracegrams.snapshot_relaxed();
    assert_eq!(snapshot.calibration_state(), CalibrationState::Collecting);
    assert_eq!(snapshot.calibration_report(), None);
}

#[test]
fn concurrent_freezer_callers_have_exactly_one_winner() {
    let mut builder = Tracegrams::builder();
    let mut stages = Vec::new();
    for index in 0..64 {
        stages.push(builder.stage(&format!("stage-{index}")).unwrap());
    }
    let tracegrams = Arc::new(builder.build().unwrap());
    let mut context = tracegrams.start_manual();
    for stage in stages {
        tracegrams.record_elapsed(&mut context, stage, Duration::from_nanos(100));
    }

    let barrier = Arc::new(Barrier::new(3));
    let mut callers = Vec::new();
    for _ in 0..2 {
        let tracegrams = Arc::clone(&tracegrams);
        let barrier = Arc::clone(&barrier);
        callers.push(std::thread::spawn(move || {
            barrier.wait();
            tracegrams.try_freeze_calibration(1)
        }));
    }
    barrier.wait();
    let results = callers
        .into_iter()
        .map(|caller| caller.join().unwrap())
        .collect::<Vec<_>>();

    assert_eq!(results.iter().filter(|result| result.is_ok()).count(), 1);
    assert_eq!(
        results
            .iter()
            .filter(|result| {
                matches!(
                    result,
                    Err(FreezeError::AlreadyFreezing | FreezeError::AlreadyFrozen)
                )
            })
            .count(),
        1
    );
    assert_eq!(
        tracegrams.snapshot_relaxed().calibration_state(),
        CalibrationState::Frozen
    );
}

#[test]
fn concurrent_snapshots_never_observe_a_partial_threshold_bundle() {
    let mut builder = Tracegrams::builder();
    let mut stages = Vec::new();
    for index in 0..64 {
        stages.push(builder.stage(&format!("stage-{index}")).unwrap());
    }
    let tracegrams = Arc::new(builder.build().unwrap());
    let mut context = tracegrams.start_manual();
    for stage in &stages {
        tracegrams.record_elapsed(&mut context, *stage, Duration::from_nanos(100));
    }

    let done = Arc::new(AtomicBool::new(false));
    let scans = Arc::new(AtomicUsize::new(0));
    let start = Arc::new(Barrier::new(2));
    let freezer_tracegrams = Arc::clone(&tracegrams);
    let freezer_done = Arc::clone(&done);
    let freezer_scans = Arc::clone(&scans);
    let freezer_start = Arc::clone(&start);
    let freezer = std::thread::spawn(move || {
        freezer_start.wait();
        let report = freezer_tracegrams.try_freeze_calibration(1).unwrap();
        // Require a snapshot that observes Frozen before ending the reader loop.
        let before = freezer_scans.load(Ordering::Acquire);
        while freezer_scans.load(Ordering::Acquire) == before {
            std::thread::yield_now();
        }
        freezer_done.store(true, Ordering::Release);
        report
    });

    let collecting = tracegrams.snapshot_relaxed();
    assert_eq!(collecting.calibration_state(), CalibrationState::Collecting);
    assert_eq!(collecting.calibration_report(), None);
    start.wait();
    while !done.load(Ordering::Acquire) {
        let snapshot = tracegrams.snapshot_relaxed();
        match snapshot.calibration_state() {
            CalibrationState::Collecting | CalibrationState::Freezing => {
                assert_eq!(snapshot.calibration_report(), None);
            }
            CalibrationState::Frozen => {
                assert_complete_bundle(&snapshot, stages.len());
                scans.fetch_add(1, Ordering::Release);
            }
            _ => unreachable!("calibration state is non-exhaustive"),
        }
    }
    let returned = freezer.join().unwrap();
    let frozen = tracegrams.snapshot_relaxed();
    assert_complete_bundle(&frozen, stages.len());
    assert_eq!(frozen.calibration_report(), Some(&returned));
}

#[test]
#[allow(clippy::too_many_lines)] // Keep the raced workload and its accounting together.
fn concurrent_writers_and_freezer_account_for_known_context_populations() {
    const WRITERS: usize = 4;
    const RACING: usize = 512;
    const ONLINE: usize = 128;
    let mut builder = Tracegrams::builder();
    let first = builder.stage("first").unwrap();
    let second = builder.stage("second").unwrap();
    let last = builder
        .stage("last-without-calibrated-predecessor")
        .unwrap();
    let tracegrams = builder.build().unwrap();
    let mut warmup = tracegrams.start_manual();
    tracegrams.record_elapsed(&mut warmup, first, Duration::from_nanos(100));
    tracegrams.finish_manual(warmup, second, Duration::from_nanos(200), Outcome::Success);
    tracegrams.finish_manual(
        tracegrams.start_manual(),
        last,
        Duration::from_nanos(300),
        Outcome::Error,
    );

    let start = Arc::new(Barrier::new(WRITERS + 1));
    let frozen = Arc::new(Barrier::new(WRITERS + 1));
    let mut writers = Vec::new();
    for _ in 0..WRITERS {
        let recorder = tracegrams.clone();
        let start = Arc::clone(&start);
        let frozen = Arc::clone(&frozen);
        writers.push(std::thread::spawn(move || {
            // Every racing context starts before the freezer is released.
            let contexts = (0..RACING)
                .map(|_| recorder.start_manual())
                .collect::<Vec<_>>();
            start.wait();
            for mut context in contexts {
                recorder.record_elapsed(&mut context, first, Duration::from_nanos(150));
                recorder.finish_manual(context, second, Duration::from_nanos(250), Outcome::Error);
            }
            frozen.wait();
            for _ in 0..ONLINE {
                let mut context = recorder.start_manual();
                recorder.record_elapsed(&mut context, first, Duration::from_nanos(150));
                recorder.record_elapsed(&mut context, second, Duration::from_nanos(250));
                recorder.finish_manual(context, last, Duration::from_nanos(350), Outcome::Success);
            }
        }));
    }
    start.wait();
    tracegrams.try_freeze_calibration(1).unwrap();
    frozen.wait();
    for writer in writers {
        writer.join().unwrap();
    }

    // Read final counters after join, not the approximately coincident scan
    // in FreezeReport. Pre-publication contexts intentionally have no online
    // samples; missing-previous-threshold skips are a separate population.
    let snapshot = tracegrams.snapshot_relaxed();
    let racing = (WRITERS * RACING) as u128;
    let online = (WRITERS * ONLINE) as u128;
    let mut calibration_marks = 0;
    for stage in [first, second] {
        let counts = snapshot.sample_counts(stage).unwrap();
        assert_eq!(counts.local, 1 + racing + online);
        assert_eq!(counts.cumulative_after, counts.local);
        assert!((1..=1 + racing).contains(&counts.calibration_local));
        assert_eq!(
            counts.calibration_cumulative_after,
            counts.calibration_local
        );
        assert_eq!(counts.online, online);
        calibration_marks += counts.calibration_local - 1;
    }
    let second_counts = snapshot.sample_counts(second).unwrap();
    assert_eq!(second_counts.cause, 1 + racing + online);
    assert_eq!(second_counts.incoming, second_counts.cause);
    assert_eq!(
        second_counts.calibration_previous_cumulative,
        second_counts.calibration_local
    );
    assert_eq!(
        snapshot
            .sample_counts(first)
            .unwrap()
            .calibration_previous_cumulative,
        0
    );
    let last_counts = snapshot.sample_counts(last).unwrap();
    assert_eq!(last_counts.local, 1 + online);
    assert_eq!(last_counts.cumulative_after, 1 + online);
    assert_eq!(last_counts.cause, online);
    assert_eq!(last_counts.incoming, online);
    assert_eq!(last_counts.calibration_local, 1);
    assert_eq!(last_counts.calibration_cumulative_after, 1);
    assert_eq!(last_counts.calibration_previous_cumulative, 0);
    assert_eq!(last_counts.online, 0);
    assert_eq!(snapshot.completion_counts(first).unwrap().success, 0);
    assert_eq!(snapshot.completion_counts(first).unwrap().error, 0);
    assert_eq!(snapshot.completion_counts(second).unwrap().success, 1);
    assert_eq!(
        u128::from(snapshot.completion_counts(second).unwrap().error),
        racing
    );
    assert_eq!(
        u128::from(snapshot.completion_counts(last).unwrap().success),
        online
    );
    assert_eq!(snapshot.completion_counts(last).unwrap().error, 1);
    let diagnostics = snapshot.diagnostics();
    let skipped = u128::from(diagnostics.calibration_samples_skipped_while_freezing);
    // The remainder consists of pre-freeze contexts recording in Frozen.
    assert!(calibration_marks + skipped <= 2 * racing);
    assert_eq!(
        u128::from(diagnostics.online_samples_skipped_missing_previous_threshold),
        online
    );
    assert_eq!(diagnostics.total(), skipped + online);
}

fn assert_complete_bundle(snapshot: &tracegrams::Snapshot, expected_stages: usize) {
    assert_eq!(snapshot.calibration_state(), CalibrationState::Frozen);
    let report = snapshot.calibration_report().unwrap();
    assert_eq!(report.stages.len(), expected_stages);
    assert!(report.stages.iter().all(|stage| {
        stage.local.availability() == CalibrationThresholdAvailability::Available
            && stage.cumulative_after.availability() == CalibrationThresholdAvailability::Available
    }));
}
