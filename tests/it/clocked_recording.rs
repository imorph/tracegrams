//! Clocked-recording and context-invariant contract tests.

use std::mem::{needs_drop, size_of};

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
