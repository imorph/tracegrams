//! Matrix-derived score contract tests.

use std::time::Duration;

use tracegrams::{
    CalibratedOnlineScore, Classification, DiagnoseConfig, DiagnoseError, FreezeCriteria,
    MatrixScore, Outcome, ScorePath, ScorePopulation, ScoreStatus, Tracegrams,
};

#[allow(clippy::cast_precision_loss)]
fn assert_score(score: MatrixScore, numerator: u128, denominator: u128) {
    assert_eq!(score.path(), ScorePath::MatrixDerived);
    assert_eq!(score.status(), ScoreStatus::Available);
    assert_eq!(score.numerator(), numerator);
    assert_eq!(score.denominator(), denominator);
    assert_eq!(score.value(), Some(numerator as f64 / denominator as f64));
}

#[allow(clippy::cast_precision_loss)]
fn assert_online_score(score: CalibratedOnlineScore, numerator: u128, denominator: u128) {
    assert_eq!(score.path(), ScorePath::CalibratedOnline);
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

#[test]
fn frozen_online_truth_cells_produce_the_exact_calibrated_formulas() {
    let mut builder = Tracegrams::builder();
    let first = builder.stage("first").unwrap();
    let destination = builder.stage("destination").unwrap();
    builder.tail_quantile(0.5).unwrap();
    let tracegrams = builder.build().unwrap();

    tracegrams.record_elapsed(
        &mut tracegrams.start_manual(),
        destination,
        Duration::from_nanos(200),
    );
    let mut context = tracegrams.start_manual();
    tracegrams.record_elapsed(&mut context, first, Duration::from_nanos(100));
    tracegrams.record_elapsed(&mut context, destination, Duration::from_nanos(200));
    tracegrams
        .try_freeze_calibration(FreezeCriteria::all_stages(1))
        .unwrap();

    for local_ns in [100, 1_000] {
        tracegrams.record_elapsed(
            &mut tracegrams.start_manual(),
            destination,
            Duration::from_nanos(local_ns),
        );
    }
    for (previous_ns, local_ns) in [(100, 100), (100, 1_000), (1_000, 100), (1_000, 1_000)] {
        let mut context = tracegrams.start_manual();
        tracegrams.record_elapsed(&mut context, first, Duration::from_nanos(previous_ns));
        tracegrams.record_elapsed(&mut context, destination, Duration::from_nanos(local_ns));
    }

    let snapshot = tracegrams.snapshot_relaxed();
    let scores = snapshot.calibrated_online_scores(destination).unwrap();
    assert_eq!(scores.path(), ScorePath::CalibratedOnline);
    assert!(!scores.path().known_drift());
    assert_eq!(scores.population(), ScorePopulation::AllReachedMarks);
    assert_eq!(scores.epoch(), 1);
    assert_eq!(scores.samples(), 6);
    assert_eq!(scores.first_samples(), 2);
    assert_eq!(scores.predecessor_samples(), 4);
    assert_eq!(scores.thresholds().previous_cumulative_ns(), Some(104));
    assert_online_score(scores.tail_onset(), 2, 4);
    assert_online_score(scores.local_tail_origin_clean(), 2, 3);
    assert_online_score(scores.local_tail_origin_tail(), 1, 3);
    assert_online_score(scores.local_tail_rate_given_prev_tail(), 1, 2);
    assert_online_score(scores.local_tail_rate_given_prev_not_tail(), 1, 2);
    assert_online_score(scores.amplification_lift(), 2, 2);
    assert_online_score(scores.carry_through(), 1, 2);
}

#[test]
fn first_marks_affect_only_clean_origin_and_onset_populations() {
    let mut builder = Tracegrams::builder();
    let only = builder.stage("only").unwrap();
    let tracegrams = builder.build().unwrap();
    tracegrams.record_elapsed(
        &mut tracegrams.start_manual(),
        only,
        Duration::from_nanos(100),
    );
    tracegrams
        .try_freeze_calibration(FreezeCriteria::all_stages(1))
        .unwrap();
    tracegrams.record_elapsed(
        &mut tracegrams.start_manual(),
        only,
        Duration::from_nanos(1_000),
    );

    let snapshot = tracegrams.snapshot_relaxed();
    let scores = snapshot.calibrated_online_scores(only).unwrap();
    assert_online_score(scores.tail_onset(), 1, 1);
    assert_online_score(scores.local_tail_origin_clean(), 1, 1);
    for score in [
        scores.local_tail_origin_tail(),
        scores.local_tail_rate_given_prev_tail(),
        scores.local_tail_rate_given_prev_not_tail(),
        scores.amplification_lift(),
        scores.carry_through(),
    ] {
        assert_eq!(score.status(), ScoreStatus::NoPredecessorPopulation);
        assert_eq!(score.value(), None);
    }

    let report = snapshot
        .diagnose(only, &DiagnoseConfig::experimental_defaults())
        .unwrap();
    assert_eq!(report.classification(), Classification::Onset);
}

#[test]
fn pre_freeze_context_never_writes_the_published_online_epoch() {
    let mut builder = Tracegrams::builder();
    let stage = builder.stage("stage").unwrap();
    let tracegrams = builder.build().unwrap();
    let mut old_context = tracegrams.start_manual();
    tracegrams.record_elapsed(
        &mut tracegrams.start_manual(),
        stage,
        Duration::from_nanos(100),
    );
    tracegrams
        .try_freeze_calibration(FreezeCriteria::all_stages(1))
        .unwrap();

    tracegrams.record_elapsed(&mut old_context, stage, Duration::from_nanos(1_000));
    let after_old = tracegrams.snapshot_relaxed();
    assert_eq!(after_old.sample_counts(stage).unwrap().local(), 2);
    assert_eq!(after_old.sample_counts(stage).unwrap().online(), 0);

    tracegrams.record_elapsed(
        &mut tracegrams.start_manual(),
        stage,
        Duration::from_nanos(1_000),
    );
    assert_eq!(
        tracegrams
            .snapshot_relaxed()
            .sample_counts(stage)
            .unwrap()
            .online(),
        1
    );
}

#[test]
fn predecessor_after_first_only_freeze_skips_the_whole_online_sample() {
    let mut builder = Tracegrams::builder();
    let first = builder.stage("first").unwrap();
    let destination = builder.stage("destination").unwrap();
    let tracegrams = builder.build().unwrap();
    tracegrams.record_elapsed(
        &mut tracegrams.start_manual(),
        destination,
        Duration::from_nanos(200),
    );
    tracegrams
        .try_freeze_calibration(FreezeCriteria::for_stages(&[destination], 1))
        .unwrap();

    let before = tracegrams.snapshot_relaxed();
    let mut context = tracegrams.start_manual();
    tracegrams.record_elapsed(&mut context, first, Duration::from_nanos(100));
    tracegrams.record_elapsed(&mut context, destination, Duration::from_nanos(1_000));
    let after = tracegrams.snapshot_relaxed();

    assert_eq!(after.sample_counts(destination).unwrap().online(), 0);
    assert_eq!(after.sample_counts(destination).unwrap().cause(), 1);
    assert_eq!(
        after
            .diagnose(first, &DiagnoseConfig::experimental_defaults())
            .unwrap_err(),
        DiagnoseError::NotCalibrated { stage: first }
    );
    assert_eq!(
        after
            .diagnostics()
            .online_samples_skipped_missing_previous_threshold(),
        before
            .diagnostics()
            .online_samples_skipped_missing_previous_threshold()
            + 1
    );
}

#[test]
fn frozen_online_zero_denominators_are_explicit() {
    let mut builder = Tracegrams::builder();
    let stage = builder.stage("stage").unwrap();
    let tracegrams = builder.build().unwrap();
    tracegrams.record_elapsed(
        &mut tracegrams.start_manual(),
        stage,
        Duration::from_nanos(100),
    );
    tracegrams
        .try_freeze_calibration(FreezeCriteria::all_stages(1))
        .unwrap();

    let scores = tracegrams
        .snapshot_relaxed()
        .calibrated_online_scores(stage)
        .unwrap();
    for score in [scores.tail_onset(), scores.local_tail_origin_clean()] {
        assert_eq!(score.status(), ScoreStatus::ZeroDenominator);
        assert_eq!(score.numerator(), 0);
        assert_eq!(score.denominator(), 0);
        assert_eq!(score.value(), None);
    }
}

#[test]
fn diagnosis_requires_calibration_and_classifies_from_numeric_scores() {
    let mut builder = Tracegrams::builder();
    let first = builder.stage("first").unwrap();
    let destination = builder.stage("destination").unwrap();
    builder.tail_quantile(0.5).unwrap();
    let tracegrams = builder.build().unwrap();
    let collecting = tracegrams.snapshot_relaxed();
    assert_eq!(
        collecting
            .diagnose(destination, &DiagnoseConfig::experimental_defaults())
            .unwrap_err(),
        DiagnoseError::NotCalibrated { stage: destination }
    );

    tracegrams.record_elapsed(
        &mut tracegrams.start_manual(),
        destination,
        Duration::from_nanos(200),
    );
    let mut context = tracegrams.start_manual();
    tracegrams.record_elapsed(&mut context, first, Duration::from_nanos(100));
    tracegrams.record_elapsed(&mut context, destination, Duration::from_nanos(200));
    tracegrams
        .try_freeze_calibration(FreezeCriteria::all_stages(1))
        .unwrap();
    tracegrams.record_elapsed(
        &mut tracegrams.start_manual(),
        destination,
        Duration::from_nanos(1_000),
    );
    for (previous_ns, local_ns) in [(100, 1_000), (1_000, 100), (1_000, 1_000)] {
        let mut context = tracegrams.start_manual();
        tracegrams.record_elapsed(&mut context, first, Duration::from_nanos(previous_ns));
        tracegrams.record_elapsed(&mut context, destination, Duration::from_nanos(local_ns));
    }
    let snapshot = tracegrams.snapshot_relaxed();

    let onset = DiagnoseConfig {
        onset_threshold: 0.5,
        amplification_lift_threshold: 2.0,
        carry_through_threshold: 0.75,
    };
    assert_eq!(
        snapshot
            .diagnose(destination, &onset)
            .unwrap()
            .classification(),
        Classification::Onset
    );
    let amplifier = DiagnoseConfig {
        onset_threshold: 0.75,
        amplification_lift_threshold: 0.5,
        carry_through_threshold: 0.75,
    };
    assert_eq!(
        snapshot
            .diagnose(destination, &amplifier)
            .unwrap()
            .classification(),
        Classification::Amplifier
    );
    let carry = DiagnoseConfig {
        onset_threshold: 0.75,
        amplification_lift_threshold: 0.75,
        carry_through_threshold: 0.5,
    };
    assert_eq!(
        snapshot
            .diagnose(destination, &carry)
            .unwrap()
            .classification(),
        Classification::CarryThrough
    );
    let inconclusive = DiagnoseConfig {
        onset_threshold: 0.75,
        amplification_lift_threshold: 0.75,
        carry_through_threshold: 0.75,
    };
    assert_eq!(
        snapshot
            .diagnose(destination, &inconclusive)
            .unwrap()
            .classification(),
        Classification::Inconclusive
    );
}

#[test]
fn diagnosis_report_and_display_are_pure_stable_and_path_explicit() {
    let mut builder = Tracegrams::builder();
    let stage = builder.stage("stage").unwrap();
    let tracegrams = builder.build().unwrap();
    tracegrams.record_elapsed(
        &mut tracegrams.start_manual(),
        stage,
        Duration::from_nanos(100),
    );
    tracegrams
        .try_freeze_calibration(FreezeCriteria::all_stages(1))
        .unwrap();
    let before = tracegrams.snapshot_relaxed();
    tracegrams.record_elapsed(
        &mut tracegrams.start_manual(),
        stage,
        Duration::from_nanos(1_000),
    );
    let snapshot = tracegrams.snapshot_relaxed();
    let config = DiagnoseConfig::experimental_defaults();

    let first = snapshot.diagnose(stage, &config).unwrap();
    let second = snapshot.diagnose(stage, &config).unwrap();
    let window = snapshot.delta(&before).unwrap();
    let window_report = window.diagnose(stage, &config).unwrap();
    assert_eq!(first, second);
    assert_eq!(
        first.calibrated_online().path(),
        ScorePath::CalibratedOnline
    );
    assert_eq!(first.matrix_derived().path(), ScorePath::MatrixDerived);
    assert_eq!(first.to_string(), second.to_string());
    assert_eq!(first.to_string(), window_report.to_string());
    assert_eq!(
        first.to_string(),
        concat!(
            "tracegrams diagnosis: stage\n",
            "path: calibrated-online\n",
            "  population: all-reached marks (first=1, predecessor=0)\n",
            "  epoch: 1\n",
            "  consistency: relaxed\n",
            "  thresholds_ns: local=104, previous_cumulative=n/a, cumulative_after=104\n",
            "  tail_onset: 1/1 = 1.000000\n",
            "  local_tail_origin_clean: 1/1 = 1.000000\n",
            "  local_tail_origin_tail: 0/0 = n/a (no predecessor population)\n",
            "  local_tail_rate_given_prev_tail: 0/0 = n/a (no predecessor population)\n",
            "  local_tail_rate_given_prev_not_tail: 0/0 = n/a (no predecessor population)\n",
            "  amplification_lift: 0/0 = n/a (no predecessor population)\n",
            "  carry_through: 0/0 = n/a (no predecessor population)\n",
            "path: matrix-derived (known drift)\n",
            "  population: predecessor-only (cause=0, incoming=0)\n",
            "  consistency: relaxed\n",
            "  threshold_buckets: local=n/a, previous_cumulative=n/a, cumulative_after=n/a\n",
            "  tail_onset: 0/0 = n/a (no predecessor population)\n",
            "  local_tail_origin_clean: 0/0 = n/a (no predecessor population)\n",
            "  local_tail_origin_tail: 0/0 = n/a (no predecessor population)\n",
            "  local_tail_rate_given_prev_tail: 0/0 = n/a (no predecessor population)\n",
            "  local_tail_rate_given_prev_not_tail: 0/0 = n/a (no predecessor population)\n",
            "  amplification_lift: 0/0 = n/a (no predecessor population)\n",
            "  carry_through: 0/0 = n/a (no predecessor population)\n",
            "classification (calibrated-online): onset\n",
            "classification (matrix-derived): inconclusive\n",
            "classification thresholds: onset>=0.700000, amplification_lift>=10.000000, ",
            "carry_through>=0.800000 (experimental PoC values)",
        )
    );
}

#[test]
fn cross_epoch_delta_keeps_matrix_scores_but_disables_online_diagnosis() {
    let mut builder = Tracegrams::builder();
    let first = builder.stage("first").unwrap();
    let destination = builder.stage("destination").unwrap();
    let tracegrams = builder.build().unwrap();
    let earlier = tracegrams.snapshot_relaxed();
    let mut context = tracegrams.start_manual();
    tracegrams.record_elapsed(&mut context, first, Duration::from_nanos(100));
    tracegrams.record_elapsed(&mut context, destination, Duration::from_nanos(200));
    tracegrams
        .try_freeze_calibration(FreezeCriteria::all_stages(1))
        .unwrap();
    let mut context = tracegrams.start_manual();
    tracegrams.record_elapsed(&mut context, first, Duration::from_nanos(100));
    tracegrams.record_elapsed(&mut context, destination, Duration::from_nanos(1_000));
    let delta = tracegrams.snapshot_relaxed().delta(&earlier).unwrap();

    let matrix_scores = delta.matrix_scores(destination).unwrap();
    assert_eq!(matrix_scores.cause_samples(), 2);
    assert_eq!(
        matrix_scores.classification(&DiagnoseConfig::experimental_defaults()),
        Classification::Inconclusive
    );
    assert_eq!(
        delta.calibrated_online_scores(destination).unwrap_err(),
        DiagnoseError::EpochMismatch {
            earlier: 0,
            later: 1,
        }
    );
    assert_eq!(
        delta
            .diagnose(destination, &DiagnoseConfig::experimental_defaults())
            .unwrap_err(),
        DiagnoseError::EpochMismatch {
            earlier: 0,
            later: 1,
        }
    );
}
