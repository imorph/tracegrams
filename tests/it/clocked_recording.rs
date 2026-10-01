//! Clocked-recording and context-invariant contract tests.

use std::mem::{needs_drop, size_of};
use std::time::Duration;

use tracegrams::{Ctx, Outcome, Tracegrams};

fn assert_send_sync<T: Send + Sync>(_: &T) {}

#[test]
fn clocked_context_satisfies_the_hot_path_layout_contract() {
    let mut builder = Tracegrams::builder();
    builder.stage("only").unwrap();
    let tracegrams = builder.build().unwrap();
    let context = tracegrams.start();

    assert_send_sync(&context);
    assert!(size_of::<Ctx>() <= 24);
    assert!(!needs_drop::<Ctx>());
}

#[test]
fn clocked_request_records_increasing_and_skipped_stages() {
    let mut builder = Tracegrams::builder();
    let parse = builder.stage("parse").unwrap();
    let _cache = builder.stage("cache").unwrap();
    let db = builder.stage("db").unwrap();
    let render = builder.stage("render").unwrap();
    let tracegrams = builder.build().unwrap();

    let mut context = tracegrams.start();
    tracegrams.mark(&mut context, parse);
    tracegrams.mark(&mut context, db);
    tracegrams.finish(context, render, Outcome::Success);

    let snapshot = tracegrams.snapshot_relaxed();
    assert_eq!(snapshot.sample_counts(parse).unwrap().local, 1);
    assert_eq!(snapshot.sample_counts(db).unwrap().local, 1);
    assert_eq!(snapshot.sample_counts(render).unwrap().local, 1);
    assert_eq!(snapshot.completion_counts(render).unwrap().success, 1);
}

#[test]
fn clocked_context_started_before_freeze_never_writes_online_counters() {
    let mut builder = Tracegrams::builder();
    let stage = builder.stage("stage").unwrap();
    let tracegrams = builder.build().unwrap();
    tracegrams.record_elapsed(
        &mut tracegrams.start_manual(),
        stage,
        Duration::from_nanos(100),
    );

    let old_context = tracegrams.start();
    tracegrams.try_freeze_calibration(1).unwrap();
    let frozen = tracegrams.snapshot_relaxed();
    assert_eq!(frozen.sample_counts(stage).unwrap().online, 0);

    tracegrams.finish(old_context, stage, Outcome::Success);
    let after_old = tracegrams.snapshot_relaxed();
    assert_eq!(after_old.sample_counts(stage).unwrap().local, 2);
    assert_eq!(after_old.sample_counts(stage).unwrap().online, 0);
    assert_eq!(after_old.completion_counts(stage).unwrap().success, 1);

    tracegrams.finish(tracegrams.start(), stage, Outcome::Success);
    let after_new = tracegrams.snapshot_relaxed();
    assert_eq!(after_new.sample_counts(stage).unwrap().local, 3);
    assert_eq!(after_new.sample_counts(stage).unwrap().online, 1);
    assert_eq!(after_new.completion_counts(stage).unwrap().success, 2);
    assert_eq!(after_new.diagnostics().total(), 0);
}

#[test]
fn clocked_context_can_move_between_threads() {
    let mut builder = Tracegrams::builder();
    let first = builder.stage("first").unwrap();
    let last = builder.stage("last").unwrap();
    let tracegrams = builder.build().unwrap();
    let context = tracegrams.start();
    let worker_tracegrams = tracegrams.clone();

    std::thread::spawn(move || {
        let mut context = context;
        worker_tracegrams.mark(&mut context, first);
        worker_tracegrams.finish(context, last, Outcome::Success);
    })
    .join()
    .unwrap();
}
