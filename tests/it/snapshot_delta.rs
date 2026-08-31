//! Relaxed snapshot and pure-delta contract tests.

use std::sync::Arc;
use std::sync::atomic::{AtomicBool, Ordering};
use std::time::Duration;

use tracegrams::{
    CalibrationPopulation, CalibrationState, Consistency, DeltaError, OnlineDeltaAvailability,
    Outcome, Tracegrams,
};

#[test]
fn quiesced_snapshot_owns_recorded_stage_data() {
    let mut builder = Tracegrams::builder();
    let parse = builder.stage("parse").unwrap();
    let db = builder.stage("db").unwrap();
    let tracegrams = builder.build().unwrap();

    let mut context = tracegrams.start_manual();
    tracegrams.record_elapsed(&mut context, parse, Duration::from_nanos(50));
    tracegrams.finish_manual(context, db, Duration::from_nanos(75), Outcome::Success);
    let snapshot = tracegrams.snapshot_relaxed();
    let snapshot = snapshot.clone();
    drop(tracegrams);

    assert_eq!(snapshot.consistency(), Consistency::Relaxed);
    assert_eq!(snapshot.stages()[0].name(), "parse");
    assert_eq!(snapshot.stages()[1].name(), "db");
    assert_eq!(snapshot.local_counts(parse).unwrap()[0], 1);
    assert_eq!(snapshot.local_counts(db).unwrap()[0], 1);
    assert_eq!(snapshot.cause_counts(db).unwrap()[0], 1);
    assert_eq!(snapshot.incoming_counts(db).unwrap()[1], 1);
    assert_eq!(snapshot.completion_counts(db).unwrap().success(), 1);
    assert_eq!(snapshot.completion_counts(db).unwrap().error(), 0);
}

#[test]
fn snapshot_exposes_read_plane_metadata_counts_and_configuration() {
    let mut builder = Tracegrams::builder();
    let parse = builder.stage("parse").unwrap();
    let db = builder.stage("db").unwrap();
    builder.tail_quantile(0.95).unwrap();
    let estimate = builder.estimated_memory().unwrap();
    let tracegrams = builder.build().unwrap();

    let mut context = tracegrams.start_manual();
    tracegrams.record_elapsed(&mut context, parse, Duration::from_nanos(50));
    tracegrams.finish_manual(context, db, Duration::from_nanos(75), Outcome::Error);
    let snapshot = tracegrams.snapshot_relaxed();

    assert_eq!(snapshot.stage_name(parse), Some("parse"));
    assert_eq!(snapshot.stage_name(db), Some("db"));
    assert_eq!(snapshot.bucket_bounds().len(), 63);
    assert_eq!(snapshot.calibration_bucket_bounds().len(), 249);
    assert!((snapshot.tail_quantile() - 0.95).abs() < f64::EPSILON);
    assert_eq!(snapshot.calibration_state(), CalibrationState::Collecting);
    assert_eq!(snapshot.calibration_epoch(), 0);
    assert_eq!(snapshot.memory_budget_bytes(), 8 * 1024 * 1024);
    assert_eq!(snapshot.memory_estimate(), estimate);

    let samples = snapshot.sample_counts(db).unwrap();
    assert_eq!(samples.local(), 1);
    assert_eq!(samples.cumulative_after(), 1);
    assert_eq!(samples.cause(), 1);
    assert_eq!(samples.incoming(), 1);
    assert_eq!(samples.calibration_local(), 1);
    assert_eq!(samples.calibration_cumulative_after(), 1);
    assert_eq!(samples.calibration_previous_cumulative(), 1);
    assert_eq!(samples.online(), 0);
    assert_eq!(
        snapshot
            .calibration_counts(db, CalibrationPopulation::PreviousCumulative)
            .unwrap()
            .iter()
            .sum::<u64>(),
        1
    );

    let availability = snapshot.score_availability(db).unwrap();
    assert!(availability.matrix_derived());
    assert!(!availability.calibrated_online());
    assert!(!snapshot.score_availability(parse).unwrap().matrix_derived());
    assert_eq!(snapshot.completion_counts(db).unwrap().error(), 1);
    assert_eq!(snapshot.diagnostics().total(), 0);
}

#[test]
fn delta_is_pure_checked_subtraction_for_an_incident_window() {
    let mut builder = Tracegrams::builder();
    let first = builder.stage("first").unwrap();
    let second = builder.stage("second").unwrap();
    let tracegrams = builder.build().unwrap();
    let before = tracegrams.snapshot_relaxed();

    let mut context = tracegrams.start_manual();
    tracegrams.record_elapsed(&mut context, first, Duration::from_nanos(50));
    tracegrams.finish_manual(context, second, Duration::from_nanos(75), Outcome::Success);
    let after = tracegrams.snapshot_relaxed();
    let window = after.delta(&before).unwrap();

    assert_eq!(window.consistency(), Consistency::Relaxed);
    assert_eq!(
        window.online_delta_availability(),
        OnlineDeltaAvailability::SameEpoch { epoch: 0 }
    );
    assert_eq!(window.local_counts(first).unwrap()[0], 1);
    assert_eq!(window.local_counts(second).unwrap()[0], 1);
    assert_eq!(window.cause_counts(second).unwrap()[0], 1);
    assert_eq!(window.completion_counts(second).unwrap().success(), 1);
    assert_eq!(window.sample_counts(second).unwrap().cause(), 1);
    assert!(window.score_availability(second).unwrap().matrix_derived());

    let empty = after.delta(&after).unwrap();
    assert_eq!(empty.sample_counts(first).unwrap().local(), 0);
    assert_eq!(empty.sample_counts(second).unwrap().cause(), 0);
    assert_eq!(empty.completion_counts(second).unwrap().success(), 0);
    assert_eq!(empty.diagnostics().total(), 0);
}

#[test]
fn delta_rejects_foreign_registries_and_counter_underflow() {
    let mut first_builder = Tracegrams::builder();
    let stage = first_builder.stage("stage").unwrap();
    let first = first_builder.build().unwrap();
    let empty = first.snapshot_relaxed();
    first.finish_manual(
        first.start_manual(),
        stage,
        Duration::from_nanos(50),
        Outcome::Success,
    );
    let populated = first.snapshot_relaxed();

    assert!(matches!(
        empty.delta(&populated),
        Err(DeltaError::CounterUnderflow {
            earlier: 1,
            later: 0
        })
    ));

    let mut second_builder = Tracegrams::builder();
    second_builder.stage("stage").unwrap();
    let foreign = second_builder.build().unwrap().snapshot_relaxed();
    assert!(matches!(
        populated.delta(&foreign),
        Err(DeltaError::RegistryMismatch)
    ));
}

#[test]
fn repeated_relaxed_snapshots_are_safe_during_concurrent_writes() {
    const REQUESTS: usize = 20_000;

    let mut builder = Tracegrams::builder();
    let first = builder.stage("first").unwrap();
    let second = builder.stage("second").unwrap();
    let tracegrams = builder.build().unwrap();
    let done = Arc::new(AtomicBool::new(false));
    let writer_done = Arc::clone(&done);
    let writer_tracegrams = tracegrams.clone();
    let writer = std::thread::spawn(move || {
        for _ in 0..REQUESTS {
            let mut context = writer_tracegrams.start_manual();
            writer_tracegrams.record_elapsed(&mut context, first, Duration::from_nanos(50));
            writer_tracegrams.finish_manual(
                context,
                second,
                Duration::from_nanos(75),
                Outcome::Success,
            );
        }
        writer_done.store(true, Ordering::Release);
    });

    let mut previous_first_samples = 0;
    while !done.load(Ordering::Acquire) {
        let snapshot = tracegrams.snapshot_relaxed();
        assert_eq!(snapshot.consistency(), Consistency::Relaxed);
        assert_eq!(snapshot.local_counts(first).unwrap().len(), 64);
        assert_eq!(snapshot.cause_counts(second).unwrap().len(), 64 * 64);
        let first_samples = snapshot.sample_counts(first).unwrap().local();
        assert!(first_samples >= previous_first_samples);
        assert!(first_samples <= REQUESTS as u128);
        previous_first_samples = first_samples;
    }
    writer.join().unwrap();

    let final_snapshot = tracegrams.snapshot_relaxed();
    assert_eq!(
        final_snapshot.sample_counts(first).unwrap().local(),
        REQUESTS as u128
    );
    assert_eq!(
        final_snapshot.sample_counts(second).unwrap().cause(),
        REQUESTS as u128
    );
    assert_eq!(
        final_snapshot.completion_counts(second).unwrap().success(),
        REQUESTS as u64
    );
}
