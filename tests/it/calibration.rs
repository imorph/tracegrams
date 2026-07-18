//! Calibration collection, readiness, and explicit-freeze contract tests.

use std::sync::atomic::{AtomicBool, Ordering};
use std::sync::{Arc, Barrier};
use std::time::Duration;

use tracegrams::{
    CalibrationConsistency, CalibrationPopulation, CalibrationState, CalibrationTerminal,
    CalibrationThresholdAvailability, FreezeCriteria, FreezeError, PopulationReadiness, Tracegrams,
};

#[test]
fn all_stage_freeze_publishes_one_complete_relaxed_bundle() {
    let mut builder = Tracegrams::builder();
    let entry = builder.stage("entry").unwrap();
    let db = builder.stage("db").unwrap();
    let tracegrams = builder.build().unwrap();

    for _ in 0..3 {
        let mut context = tracegrams.start_manual();
        tracegrams.record_elapsed(&mut context, entry, Duration::from_nanos(100));
        tracegrams.record_elapsed(&mut context, db, Duration::from_nanos(200));
    }

    let readiness = tracegrams.calibration_readiness(3);
    assert!(readiness.is_ready());
    assert!(matches!(
        readiness.stage(entry).unwrap().previous_cumulative(),
        PopulationReadiness::AbsentNoPredecessor
    ));
    assert!(matches!(
        readiness.stage(db).unwrap().previous_cumulative(),
        PopulationReadiness::Ready {
            samples: 3,
            minimum: 3
        }
    ));

    let report = tracegrams
        .try_freeze_calibration(FreezeCriteria::all_stages(3))
        .unwrap();
    assert_eq!(report.epoch(), Some(1));
    assert_eq!(report.consistency(), CalibrationConsistency::Relaxed);
    assert_eq!(report.stages().len(), 2);
    let entry_report = report.stage(entry).unwrap();
    assert_eq!(
        entry_report.local().population(),
        CalibrationPopulation::Local
    );
    assert_eq!(entry_report.local().samples(), 3);
    assert_eq!(
        entry_report.local().availability(),
        CalibrationThresholdAvailability::Available
    );
    let estimate = entry_report.local().estimate().unwrap();
    assert_eq!(estimate.rank(), 3);
    assert_eq!(estimate.samples(), 3);
    assert_eq!(estimate.lower_bound(), Some(100));
    assert_eq!(estimate.upper_bound(), Some(108));
    assert_eq!(estimate.value(), Some(104));
    assert_eq!(estimate.terminal(), CalibrationTerminal::Finite);
    assert_eq!(
        entry_report.previous_cumulative().availability(),
        CalibrationThresholdAvailability::AbsentNoPredecessor
    );
    assert_eq!(entry_report.previous_cumulative().samples(), 0);
    assert_eq!(entry_report.previous_cumulative().estimate(), None);

    let frozen = tracegrams.snapshot_relaxed();
    assert_eq!(frozen.calibration_state(), CalibrationState::Frozen);
    assert_eq!(frozen.calibration_epoch(), 1);
    assert_eq!(frozen.calibration_report(), Some(&report));

    let before = frozen.sample_counts(entry).unwrap().calibration_local();
    tracegrams.record_elapsed(
        &mut tracegrams.start_manual(),
        entry,
        Duration::from_nanos(300),
    );
    let after = tracegrams.snapshot_relaxed();
    assert_eq!(
        after.sample_counts(entry).unwrap().calibration_local(),
        before
    );
    assert_eq!(
        after
            .diagnostics()
            .calibration_samples_skipped_while_freezing(),
        0
    );
}

#[test]
fn previous_population_readiness_distinguishes_absent_insufficient_and_ready() {
    let mut builder = Tracegrams::builder();
    let first = builder.stage("first").unwrap();
    let destination = builder.stage("destination").unwrap();
    let tracegrams = builder.build().unwrap();

    for _ in 0..3 {
        tracegrams.record_elapsed(
            &mut tracegrams.start_manual(),
            destination,
            Duration::from_nanos(200),
        );
    }
    assert_eq!(
        tracegrams
            .calibration_readiness(3)
            .stage(destination)
            .unwrap()
            .previous_cumulative(),
        PopulationReadiness::AbsentNoPredecessor
    );

    let mut context = tracegrams.start_manual();
    tracegrams.record_elapsed(&mut context, first, Duration::from_nanos(100));
    tracegrams.record_elapsed(&mut context, destination, Duration::from_nanos(200));
    let one_previous = tracegrams.calibration_readiness(3);
    assert_eq!(
        one_previous
            .stage(destination)
            .unwrap()
            .previous_cumulative(),
        PopulationReadiness::Insufficient {
            samples: 1,
            minimum: 3,
        }
    );
    assert!(matches!(
        tracegrams.try_freeze_calibration(FreezeCriteria::all_stages(3)),
        Err(FreezeError::NotReady { .. })
    ));
    assert_eq!(
        tracegrams.snapshot_relaxed().calibration_state(),
        CalibrationState::Collecting
    );

    for _ in 0..2 {
        let mut context = tracegrams.start_manual();
        tracegrams.record_elapsed(&mut context, first, Duration::from_nanos(100));
        tracegrams.record_elapsed(&mut context, destination, Duration::from_nanos(200));
    }
    assert_eq!(
        tracegrams
            .calibration_readiness(3)
            .stage(destination)
            .unwrap()
            .previous_cumulative(),
        PopulationReadiness::Ready {
            samples: 3,
            minimum: 3,
        }
    );
}

#[test]
fn selected_stage_freeze_leaves_excluded_stages_uncalibrated() {
    let mut builder = Tracegrams::builder();
    let entry = builder.stage("entry").unwrap();
    let db = builder.stage("db").unwrap();
    let rare = builder.stage("rare").unwrap();
    let tracegrams = builder.build().unwrap();

    for _ in 0..2 {
        let mut context = tracegrams.start_manual();
        tracegrams.record_elapsed(&mut context, entry, Duration::from_nanos(100));
        tracegrams.record_elapsed(&mut context, db, Duration::from_nanos(200));
    }
    assert!(!tracegrams.calibration_readiness(2).is_ready());

    let report = tracegrams
        .try_freeze_calibration(FreezeCriteria::for_stages(&[entry, db], 2))
        .unwrap();
    assert!(report.stage(entry).is_some());
    assert_eq!(
        report
            .stage(entry)
            .unwrap()
            .previous_cumulative()
            .availability(),
        CalibrationThresholdAvailability::AbsentNoPredecessor
    );
    assert!(report.stage(db).is_some());
    assert!(report.stage(rare).is_none());
    assert_eq!(
        tracegrams.snapshot_relaxed().calibration_report().unwrap(),
        &report
    );
    assert_eq!(
        tracegrams.try_freeze_calibration(FreezeCriteria::all_stages(2)),
        Err(FreezeError::AlreadyFrozen { epoch: 1 })
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

    let error = tracegrams
        .try_freeze_calibration(FreezeCriteria::all_stages(1))
        .unwrap_err();
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
    assert_eq!(report.epoch(), None);
    let estimate = report.stage(stage).unwrap().local().estimate().unwrap();
    assert_eq!(estimate.terminal(), CalibrationTerminal::Lower);
    assert_eq!(estimate.lower_bound(), None);
    assert_eq!(estimate.upper_bound(), Some(100));
    assert_eq!(estimate.value(), None);
    assert_eq!(estimate.max_relative_error(), None);
    let failed_snapshot = tracegrams.snapshot_relaxed();
    assert_eq!(
        failed_snapshot.calibration_state(),
        CalibrationState::Collecting
    );
    assert_eq!(failed_snapshot.calibration_epoch(), 0);
    assert_eq!(failed_snapshot.calibration_report(), None);

    for _ in 0..100 {
        tracegrams.record_elapsed(
            &mut tracegrams.start_manual(),
            stage,
            Duration::from_nanos(100),
        );
    }
    let retry = tracegrams
        .try_freeze_calibration(FreezeCriteria::all_stages(1))
        .unwrap();
    let estimate = retry.stage(stage).unwrap().local().estimate().unwrap();
    assert_eq!(estimate.terminal(), CalibrationTerminal::Finite);
    assert!(estimate.value().is_some());
    assert!(estimate.max_relative_error().is_some());
}

#[test]
fn upper_terminal_failure_can_retry_with_an_amended_selector() {
    let mut builder = Tracegrams::builder();
    let stable = builder.stage("stable").unwrap();
    let out_of_range = builder.stage("out-of-range").unwrap();
    let tracegrams = builder.build().unwrap();
    let mut context = tracegrams.start_manual();
    tracegrams.record_elapsed(&mut context, stable, Duration::from_nanos(100));
    tracegrams.record_elapsed(&mut context, out_of_range, Duration::from_secs(10));

    assert!(matches!(
        tracegrams.try_freeze_calibration(FreezeCriteria::all_stages(1)),
        Err(FreezeError::CalibrationRangeInsufficient {
            stage,
            terminal: CalibrationTerminal::Upper,
            ..
        }) if stage == out_of_range
    ));
    assert_eq!(
        tracegrams.snapshot_relaxed().calibration_state(),
        CalibrationState::Collecting
    );

    let report = tracegrams
        .try_freeze_calibration(FreezeCriteria::for_stages(&[stable], 1))
        .unwrap();
    assert!(report.stage(stable).is_some());
    assert!(report.stage(out_of_range).is_none());
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
            tracegrams.try_freeze_calibration(FreezeCriteria::all_stages(1))
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
                    Err(FreezeError::AlreadyFreezing | FreezeError::AlreadyFrozen { epoch: 1 })
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
fn invalid_selected_stage_criteria_are_typed_and_do_not_change_state() {
    let mut builder = Tracegrams::builder();
    let stage = builder.stage("stage").unwrap();
    let tracegrams = builder.build().unwrap();
    let mut foreign_builder = Tracegrams::builder();
    let foreign = foreign_builder.stage("foreign").unwrap();

    assert_eq!(
        tracegrams.try_freeze_calibration(FreezeCriteria::for_stages(&[], 1)),
        Err(FreezeError::NoStagesSelected)
    );
    assert_eq!(
        tracegrams.try_freeze_calibration(FreezeCriteria::for_stages(&[foreign], 1)),
        Err(FreezeError::InvalidStageSelection { stage: foreign })
    );
    assert_eq!(
        tracegrams.try_freeze_calibration(FreezeCriteria::for_stages(&[stage, stage], 1)),
        Err(FreezeError::DuplicateStageSelection { stage })
    );
    let snapshot = tracegrams.snapshot_relaxed();
    assert_eq!(snapshot.calibration_state(), CalibrationState::Collecting);
    assert_eq!(snapshot.calibration_epoch(), 0);
    assert_eq!(snapshot.calibration_report(), None);
    assert_eq!(snapshot.diagnostics().total(), 0);
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
    let freezer_tracegrams = Arc::clone(&tracegrams);
    let freezer_done = Arc::clone(&done);
    let freezer = std::thread::spawn(move || {
        let report = freezer_tracegrams
            .try_freeze_calibration(FreezeCriteria::all_stages(1))
            .unwrap();
        freezer_done.store(true, Ordering::Release);
        report
    });

    while !done.load(Ordering::Acquire) {
        let snapshot = tracegrams.snapshot_relaxed();
        match snapshot.calibration_state() {
            CalibrationState::Collecting | CalibrationState::Freezing => {
                assert_eq!(snapshot.calibration_epoch(), 0);
                assert_eq!(snapshot.calibration_report(), None);
            }
            CalibrationState::Frozen => assert_complete_bundle(&snapshot, stages.len()),
            _ => unreachable!("calibration state is non-exhaustive"),
        }
    }
    let returned = freezer.join().unwrap();
    let frozen = tracegrams.snapshot_relaxed();
    assert_complete_bundle(&frozen, stages.len());
    assert_eq!(frozen.calibration_report(), Some(&returned));
}

fn assert_complete_bundle(snapshot: &tracegrams::Snapshot, expected_stages: usize) {
    assert_eq!(snapshot.calibration_state(), CalibrationState::Frozen);
    assert_eq!(snapshot.calibration_epoch(), 1);
    let report = snapshot.calibration_report().unwrap();
    assert_eq!(report.epoch(), Some(1));
    assert_eq!(report.stages().len(), expected_stages);
    assert!(report.stages().iter().all(|stage| {
        stage.local().availability() == CalibrationThresholdAvailability::Available
            && stage.cumulative_after().availability()
                == CalibrationThresholdAvailability::Available
    }));
}
