//! Deterministic manual-recording contract tests.

use std::time::Duration;

use tracegrams::{ManualCtx, Outcome, Tracegrams};

fn assert_send_sync<T: Send + Sync>(_: &T) {}

#[test]
fn manual_context_records_increasing_and_skipped_stages() {
    let mut builder = Tracegrams::builder();
    let parse = builder.stage("parse").unwrap();
    let _cache = builder.stage("cache").unwrap();
    let db = builder.stage("db").unwrap();
    let render = builder.stage("render").unwrap();
    let tracegrams = builder.build().unwrap();

    let mut context = tracegrams.start_manual();
    assert_send_sync(&context);
    assert!(!std::mem::needs_drop::<ManualCtx>());
    tracegrams.record_elapsed(&mut context, parse, Duration::from_micros(10));
    tracegrams.record_elapsed(&mut context, db, Duration::from_millis(2));
    tracegrams.finish_manual(context, render, Duration::from_micros(50), Outcome::Error);

    let snapshot = tracegrams.snapshot_relaxed();
    assert_eq!(snapshot.sample_counts(parse).unwrap().local(), 1);
    assert_eq!(snapshot.sample_counts(db).unwrap().local(), 1);
    assert_eq!(snapshot.sample_counts(render).unwrap().local(), 1);
    assert_eq!(snapshot.completion_counts(render).unwrap().error(), 1);
}

#[test]
fn a_manual_request_can_finish_at_its_first_mark() {
    let mut builder = Tracegrams::builder();
    let only = builder.stage("only").unwrap();
    let tracegrams = builder.build().unwrap();

    let context = tracegrams.start_manual();
    tracegrams.finish_manual(context, only, Duration::from_micros(10), Outcome::Success);

    let snapshot = tracegrams.snapshot_relaxed();
    assert_eq!(snapshot.sample_counts(only).unwrap().local(), 1);
    assert_eq!(snapshot.completion_counts(only).unwrap().success(), 1);
}
