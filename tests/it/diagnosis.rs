//! Matrix-derived score contract tests.

use std::time::Duration;

use tracegrams::{MatrixScore, Outcome, ScorePath, ScorePopulation, ScoreStatus, Tracegrams};

#[allow(clippy::cast_precision_loss)]
fn assert_score(score: MatrixScore, numerator: u128, denominator: u128) {
    assert_eq!(score.path(), ScorePath::MatrixDerived);
    assert_eq!(score.status(), ScoreStatus::Available);
    assert_eq!(score.numerator(), numerator);
    assert_eq!(score.denominator(), denominator);
    assert_eq!(score.value(), Some(numerator as f64 / denominator as f64));
}

#[test]
fn matrix_scores_match_the_predecessor_only_poc_fixture() {
    let mut builder = Tracegrams::builder();
    let first = builder.stage("first").unwrap();
    let destination = builder.stage("destination").unwrap();
    builder.tail_quantile(0.91).unwrap();
    let tracegrams = builder.build().unwrap();

    let populations = [
        (100, 100, 80),
        (100, 1_000_000, 10),
        (1_000_000, 100, 5),
        (1_000_000, 1_000_000, 5),
    ];
    for (previous_ns, local_ns, count) in populations {
        for _ in 0..count {
            let mut context = tracegrams.start_manual();
            tracegrams.record_elapsed(&mut context, first, Duration::from_nanos(previous_ns));
            tracegrams.finish_manual(
                context,
                destination,
                Duration::from_nanos(local_ns),
                Outcome::Success,
            );
        }
    }

    let scores = tracegrams
        .snapshot_relaxed()
        .matrix_scores(destination)
        .unwrap();

    assert_eq!(scores.path(), ScorePath::MatrixDerived);
    assert!(scores.path().known_drift());
    assert_eq!(scores.population(), ScorePopulation::PredecessorOnly);
    assert_eq!(scores.cause_samples(), 100);
    assert_eq!(scores.incoming_samples(), 100);
    let local_threshold = scores.thresholds().local().unwrap();
    assert_eq!(local_threshold.bucket(), 32);
    assert_eq!(local_threshold.rank(), 92);
    assert_eq!(local_threshold.samples(), 100);
    assert_eq!(
        scores.thresholds().previous_cumulative().unwrap().bucket(),
        32
    );
    assert_eq!(scores.thresholds().cumulative_after().unwrap().bucket(), 32);

    assert_score(scores.local_tail_origin_clean(), 10, 15);
    assert_score(scores.local_tail_origin_tail(), 5, 15);
    assert_score(scores.local_tail_rate_given_prev_tail(), 5, 10);
    assert_score(scores.local_tail_rate_given_prev_not_tail(), 10, 90);
    assert_score(scores.amplification_lift(), 450, 100);
    assert_score(scores.carry_through(), 5, 10);
    assert_score(scores.tail_onset(), 10, 20);
}

#[test]
fn first_only_stage_has_no_invented_matrix_scores() {
    let mut builder = Tracegrams::builder();
    let first = builder.stage("first").unwrap();
    let tracegrams = builder.build().unwrap();

    tracegrams.finish_manual(
        tracegrams.start_manual(),
        first,
        Duration::from_nanos(1_000_000),
        Outcome::Success,
    );

    let scores = tracegrams.snapshot_relaxed().matrix_scores(first).unwrap();
    assert_eq!(scores.population(), ScorePopulation::PredecessorOnly);
    assert_eq!(scores.cause_samples(), 0);
    assert_eq!(scores.incoming_samples(), 0);
    assert_eq!(scores.thresholds().local(), None);
    assert_eq!(scores.thresholds().previous_cumulative(), None);
    assert_eq!(scores.thresholds().cumulative_after(), None);

    for score in [
        scores.local_tail_origin_clean(),
        scores.local_tail_origin_tail(),
        scores.local_tail_rate_given_prev_tail(),
        scores.local_tail_rate_given_prev_not_tail(),
        scores.amplification_lift(),
        scores.carry_through(),
        scores.tail_onset(),
    ] {
        assert_eq!(score.status(), ScoreStatus::NoPredecessorPopulation);
        assert_eq!(score.value(), None);
    }
}

#[test]
fn mixed_stage_matrix_scores_stay_predecessor_only() {
    let mut builder = Tracegrams::builder();
    let first = builder.stage("first").unwrap();
    let destination = builder.stage("destination").unwrap();
    let tracegrams = builder.build().unwrap();

    let mut context = tracegrams.start_manual();
    tracegrams.record_elapsed(&mut context, first, Duration::from_nanos(100));
    tracegrams.finish_manual(
        context,
        destination,
        Duration::from_nanos(1_000),
        Outcome::Success,
    );
    let predecessor_only = tracegrams
        .snapshot_relaxed()
        .matrix_scores(destination)
        .unwrap();

    for _ in 0..100 {
        tracegrams.finish_manual(
            tracegrams.start_manual(),
            destination,
            Duration::from_secs(1),
            Outcome::Success,
        );
    }
    let mixed_snapshot = tracegrams.snapshot_relaxed();
    let mixed = mixed_snapshot.matrix_scores(destination).unwrap();

    assert_eq!(mixed.population(), ScorePopulation::PredecessorOnly);
    assert_eq!(mixed, predecessor_only);
    assert_eq!(
        mixed_snapshot.sample_counts(destination).unwrap().local(),
        101
    );
    assert_eq!(mixed.cause_samples(), 1);
    assert_eq!(mixed.incoming_samples(), 1);
}

#[test]
fn skipped_paths_score_the_destination_stage() {
    let mut builder = Tracegrams::builder();
    let first = builder.stage("first").unwrap();
    let skipped = builder.stage("skipped").unwrap();
    let destination = builder.stage("destination").unwrap();
    let tracegrams = builder.build().unwrap();

    let mut context = tracegrams.start_manual();
    tracegrams.record_elapsed(&mut context, first, Duration::from_nanos(100));
    tracegrams.finish_manual(
        context,
        destination,
        Duration::from_nanos(1_000),
        Outcome::Success,
    );

    let snapshot = tracegrams.snapshot_relaxed();
    let skipped_scores = snapshot.matrix_scores(skipped).unwrap();
    let destination_scores = snapshot.matrix_scores(destination).unwrap();

    assert_eq!(skipped_scores.cause_samples(), 0);
    assert_eq!(
        skipped_scores.tail_onset().status(),
        ScoreStatus::NoPredecessorPopulation
    );
    assert_eq!(destination_scores.stage(), destination);
    assert_eq!(destination_scores.cause_samples(), 1);
    assert_eq!(destination_scores.incoming_samples(), 1);
    assert_eq!(
        destination_scores.local_tail_origin_tail().status(),
        ScoreStatus::Available
    );
}

#[test]
fn zero_denominator_is_explicitly_unavailable() {
    let mut builder = Tracegrams::builder();
    let first = builder.stage("first").unwrap();
    let destination = builder.stage("destination").unwrap();
    let tracegrams = builder.build().unwrap();

    let mut context = tracegrams.start_manual();
    tracegrams.record_elapsed(&mut context, first, Duration::from_nanos(100));
    tracegrams.finish_manual(
        context,
        destination,
        Duration::from_nanos(100),
        Outcome::Success,
    );

    let scores = tracegrams
        .snapshot_relaxed()
        .matrix_scores(destination)
        .unwrap();
    let clean_rate = scores.local_tail_rate_given_prev_not_tail();
    assert_eq!(clean_rate.status(), ScoreStatus::ZeroDenominator);
    assert_eq!(clean_rate.numerator(), 0);
    assert_eq!(clean_rate.denominator(), 0);
    assert_eq!(clean_rate.value(), None);
    assert_eq!(
        scores.amplification_lift().status(),
        ScoreStatus::ZeroDenominator
    );
}

#[test]
fn snapshot_and_delta_matrix_scores_are_pure_and_deterministic() {
    let mut builder = Tracegrams::builder();
    let first = builder.stage("first").unwrap();
    let destination = builder.stage("destination").unwrap();
    let tracegrams = builder.build().unwrap();
    let before = tracegrams.snapshot_relaxed();

    let mut context = tracegrams.start_manual();
    tracegrams.record_elapsed(&mut context, first, Duration::from_nanos(100));
    tracegrams.finish_manual(
        context,
        destination,
        Duration::from_nanos(1_000),
        Outcome::Success,
    );
    let after = tracegrams.snapshot_relaxed();
    let expected = after.matrix_scores(destination).unwrap();

    assert_eq!(after.matrix_scores(destination).unwrap(), expected);
    assert_eq!(
        after
            .delta(&before)
            .unwrap()
            .matrix_scores(destination)
            .unwrap(),
        expected
    );

    tracegrams.finish_manual(
        tracegrams.start_manual(),
        destination,
        Duration::from_secs(1),
        Outcome::Success,
    );
    assert_eq!(after.matrix_scores(destination).unwrap(), expected);
}
