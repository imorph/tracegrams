//! Event-level differential properties for both diagnosis score paths.

use std::time::Duration;

use proptest::prelude::*;
use proptest::test_runner::{Config, RngAlgorithm, TestRng, TestRunner};
use tracegrams::{
    CalibratedOnlineScore, CalibrationEstimate, CalibrationTerminal, FreezeCriteria, MatrixScore,
    Outcome, ScoreStatus, StageId, Tracegrams,
};

const QUANTILE: f64 = 0.5;

#[derive(Clone, Copy, Debug)]
struct Event {
    previous: Option<u64>,
    local: u64,
}

impl Event {
    fn after(self) -> u64 {
        self.previous.unwrap_or(0).saturating_add(self.local)
    }
}

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
struct Expected {
    numerator: u128,
    denominator: u128,
    status: ScoreStatus,
}

fn probability(numerator: u128, denominator: u128) -> Expected {
    Expected {
        numerator,
        denominator,
        status: if denominator == 0 {
            ScoreStatus::ZeroDenominator
        } else {
            ScoreStatus::Available
        },
    }
}

fn lift(tail_local: u128, tail: u128, clean_local: u128, clean: u128) -> Expected {
    probability(tail_local * clean, tail * clean_local)
}

fn no_predecessor() -> Expected {
    Expected {
        numerator: 0,
        denominator: 0,
        status: ScoreStatus::NoPredecessorPopulation,
    }
}

fn bucket(value: u64, bounds: &[u64]) -> usize {
    bounds.iter().take_while(|bound| value >= **bound).count()
}

fn threshold(values: impl Iterator<Item = u64>, bounds: &[u64]) -> usize {
    let mut counts = vec![0_u128; bounds.len() + 1];
    for value in values {
        counts[bucket(value, bounds)] += 1;
    }
    let samples: u128 = counts.iter().sum();
    let rank = samples.div_ceil(2);
    let mut cumulative = 0;
    counts
        .iter()
        .position(|count| {
            cumulative += count;
            cumulative >= rank
        })
        .expect("the generated predecessor population is nonempty")
}

fn expected_scores(
    events: &[Event],
    previous_tail: impl Fn(u64) -> bool,
    local_tail: impl Fn(u64) -> bool,
    after_tail: impl Fn(u64) -> bool,
    first_marks: bool,
) -> [Expected; 7] {
    let predecessors = events
        .iter()
        .filter(|event| event.previous.is_some())
        .count();
    let mut local_total = 0;
    let mut local_clean = 0;
    let mut local_previous_clean = 0;
    let mut local_previous_tail = 0;
    let mut previous_clean = 0;
    let mut previous_tail_total = 0;
    let mut carry = 0;
    let mut after_total = 0;
    let mut onset = 0;
    for event in events {
        let local_is_tail = local_tail(event.local);
        let after_is_tail = after_tail(event.after());
        match event.previous {
            None if first_marks => {
                if local_is_tail {
                    local_total += 1;
                    local_clean += 1;
                }
                if after_is_tail {
                    after_total += 1;
                    onset += 1;
                }
            }
            Some(previous) => {
                let previous_is_tail = previous_tail(previous);
                if previous_is_tail {
                    previous_tail_total += 1;
                } else {
                    previous_clean += 1;
                }
                if local_is_tail {
                    local_total += 1;
                    if previous_is_tail {
                        local_previous_tail += 1;
                    } else {
                        local_clean += 1;
                        local_previous_clean += 1;
                    }
                } else if previous_is_tail {
                    carry += 1;
                }
                if after_is_tail {
                    after_total += 1;
                    if !previous_is_tail {
                        onset += 1;
                    }
                }
            }
            None => {}
        }
    }
    let predecessor_score = |score| {
        if predecessors == 0 {
            no_predecessor()
        } else {
            score
        }
    };
    if predecessors == 0 && !first_marks {
        return [no_predecessor(); 7];
    }
    [
        probability(local_clean, local_total),
        predecessor_score(probability(local_previous_tail, local_total)),
        predecessor_score(probability(local_previous_tail, previous_tail_total)),
        predecessor_score(probability(local_previous_clean, previous_clean)),
        predecessor_score(lift(
            local_previous_tail,
            previous_tail_total,
            local_previous_clean,
            previous_clean,
        )),
        predecessor_score(probability(carry, previous_tail_total)),
        probability(onset, after_total),
    ]
}

fn assert_matrix(actual: &[MatrixScore; 7], expected: &[Expected; 7]) {
    for (actual, expected) in actual.iter().zip(expected) {
        assert_eq!(actual.numerator(), expected.numerator);
        assert_eq!(actual.denominator(), expected.denominator);
        assert_eq!(actual.status(), expected.status);
    }
}

fn assert_online(actual: &[CalibratedOnlineScore; 7], expected: &[Expected; 7]) {
    for (actual, expected) in actual.iter().zip(expected) {
        assert_eq!(actual.numerator(), expected.numerator);
        assert_eq!(actual.denominator(), expected.denominator);
        assert_eq!(actual.status(), expected.status);
    }
}

fn record(recorder: &Tracegrams, first: StageId, destination: StageId, event: Event) {
    let mut context = recorder.start_manual();
    if let Some(previous) = event.previous {
        recorder.record_elapsed(&mut context, first, Duration::from_nanos(previous));
    }
    recorder.finish_manual(
        context,
        destination,
        Duration::from_nanos(event.local),
        Outcome::Success,
    );
}

fn assert_reported_calibration_rank(values: &[u64], bounds: &[u64], estimate: CalibrationEstimate) {
    let samples = values.len() as u128;
    let rank = samples.div_ceil(2);
    let selected = threshold(values.iter().copied(), bounds);
    let before = values
        .iter()
        .filter(|value| bucket(**value, bounds) < selected)
        .count() as u128;
    let through = values
        .iter()
        .filter(|value| bucket(**value, bounds) <= selected)
        .count() as u128;

    assert_eq!(estimate.samples(), samples);
    assert_eq!(estimate.rank(), rank);
    assert!(before < rank);
    assert!(through >= rank);
    assert_eq!(estimate.terminal(), CalibrationTerminal::Finite);
    assert_eq!(estimate.lower_bound(), Some(bounds[selected - 1]));
    assert_eq!(estimate.upper_bound(), Some(bounds[selected]));
    assert_eq!(bucket(estimate.value().unwrap(), bounds), selected);
}

#[allow(clippy::too_many_lines)]
fn run_case(mut calibration: Vec<(bool, u16, u16)>, scored: Vec<(bool, u16, u16)>) {
    // All generated values are on the finite grid; these anchors guarantee
    // both first-mark and predecessor-bearing calibration events are
    // represented.
    calibration.extend([(true, 100, 200), (false, 0, 400)]);
    let calibration = calibration
        .into_iter()
        .map(|(has_previous, previous, local)| Event {
            previous: has_previous.then_some(u64::from(previous)),
            local: u64::from(local),
        })
        .collect::<Vec<_>>();

    let mut builder = Tracegrams::builder();
    let first = builder.stage("first").unwrap();
    let destination = builder.stage("destination").unwrap();
    builder.tail_quantile(QUANTILE).unwrap();
    let recorder = builder.build().unwrap();
    for event in &calibration {
        record(&recorder, first, destination, *event);
    }
    let report = recorder
        .try_freeze_calibration(FreezeCriteria::all_stages(1))
        .unwrap();
    let stage_report = report.stage(destination).unwrap();
    let calibration_bounds = recorder
        .snapshot_relaxed()
        .calibration_bucket_bounds()
        .to_vec();
    for (values, estimate) in [
        (
            calibration
                .iter()
                .map(|event| event.local)
                .collect::<Vec<_>>(),
            stage_report.local().estimate().unwrap(),
        ),
        (
            calibration
                .iter()
                .map(|event| event.after())
                .collect::<Vec<_>>(),
            stage_report.cumulative_after().estimate().unwrap(),
        ),
        (
            calibration
                .iter()
                .filter_map(|event| event.previous)
                .collect::<Vec<_>>(),
            stage_report.previous_cumulative().estimate().unwrap(),
        ),
    ] {
        assert_reported_calibration_rank(&values, &calibration_bounds, estimate);
    }

    let local_ns = stage_report.local().estimate().unwrap().value().unwrap();
    let previous_ns = stage_report
        .previous_cumulative()
        .estimate()
        .unwrap()
        .value()
        .unwrap();
    let after_ns = stage_report
        .cumulative_after()
        .estimate()
        .unwrap()
        .value()
        .unwrap();
    let mut online = scored
        .into_iter()
        .map(|(has_previous, previous, local)| Event {
            previous: has_previous.then_some(u64::from(previous)),
            local: u64::from(local),
        })
        .collect::<Vec<_>>();
    // Equality witnesses make >= predicate mutations observable and exercise
    // the two first-mark score populations.
    online.extend([
        Event {
            previous: Some(previous_ns),
            local: local_ns,
        },
        Event {
            previous: None,
            local: after_ns,
        },
    ]);
    for event in &online {
        record(&recorder, first, destination, *event);
    }

    let snapshot = recorder.snapshot_relaxed();
    let bounds = snapshot.bucket_bounds();
    let all_predecessors = calibration
        .iter()
        .chain(&online)
        .copied()
        .filter(|event| event.previous.is_some())
        .collect::<Vec<_>>();
    let previous_bucket = threshold(
        all_predecessors.iter().filter_map(|event| event.previous),
        bounds,
    );
    let local_bucket = threshold(all_predecessors.iter().map(|event| event.local), bounds);
    let after_bucket = threshold(all_predecessors.iter().map(|event| event.after()), bounds);
    let matrix_expected = expected_scores(
        &all_predecessors,
        |value| bucket(value, bounds) >= previous_bucket,
        |value| bucket(value, bounds) >= local_bucket,
        |value| bucket(value, bounds) >= after_bucket,
        false,
    );
    let matrix = snapshot.matrix_scores(destination).unwrap();
    assert_eq!(
        matrix.thresholds().previous_cumulative().unwrap().bucket(),
        previous_bucket
    );
    assert_eq!(matrix.thresholds().local().unwrap().bucket(), local_bucket);
    assert_eq!(
        matrix.thresholds().cumulative_after().unwrap().bucket(),
        after_bucket
    );
    assert_matrix(
        &[
            matrix.local_tail_origin_clean(),
            matrix.local_tail_origin_tail(),
            matrix.local_tail_rate_given_prev_tail(),
            matrix.local_tail_rate_given_prev_not_tail(),
            matrix.amplification_lift(),
            matrix.carry_through(),
            matrix.tail_onset(),
        ],
        &matrix_expected,
    );

    let online_expected = expected_scores(
        &online,
        |value| value >= previous_ns,
        |value| value >= local_ns,
        |value| value >= after_ns,
        true,
    );
    let actual = snapshot.calibrated_online_scores(destination).unwrap();
    assert_online(
        &[
            actual.local_tail_origin_clean(),
            actual.local_tail_origin_tail(),
            actual.local_tail_rate_given_prev_tail(),
            actual.local_tail_rate_given_prev_not_tail(),
            actual.amplification_lift(),
            actual.carry_through(),
            actual.tail_onset(),
        ],
        &online_expected,
    );
}

#[test]
fn diagnosis_scores_match_raw_event_oracle() {
    let config = Config {
        cases: 64,
        failure_persistence: Some(Box::new(
            proptest::test_runner::FileFailurePersistence::Direct(
                "proptest-regressions/it/diagnosis_properties.txt",
            ),
        )),
        ..Config::default()
    };
    let rng = TestRng::from_seed(RngAlgorithm::ChaCha, &[0x5c; 32]);
    let mut runner = TestRunner::new_with_rng(config, rng);
    let calibration =
        prop::collection::vec((any::<bool>(), 100_u16..20_000, 100_u16..20_000), 0..=12);
    let scored = prop::collection::vec((any::<bool>(), 0_u16..20_000, 0_u16..20_000), 0..=20);
    runner
        .run(&(calibration, scored), |(calibration, scored)| {
            run_case(calibration, scored);
            Ok(())
        })
        .unwrap();
}

#[test]
fn first_only_populations_have_path_specific_statuses() {
    let mut builder = Tracegrams::builder();
    let destination = builder.stage("destination").unwrap();
    builder.tail_quantile(QUANTILE).unwrap();
    let recorder = builder.build().unwrap();
    let calibration = [
        Event {
            previous: None,
            local: 200,
        },
        Event {
            previous: None,
            local: 400,
        },
    ];
    for event in calibration {
        record(&recorder, destination, destination, event);
    }
    recorder
        .try_freeze_calibration(FreezeCriteria::all_stages(1))
        .unwrap();
    let online = [
        Event {
            previous: None,
            local: 200,
        },
        Event {
            previous: None,
            local: 800,
        },
    ];
    for event in online {
        record(&recorder, destination, destination, event);
    }

    let snapshot = recorder.snapshot_relaxed();
    let matrix = snapshot.matrix_scores(destination).unwrap();
    assert_matrix(
        &[
            matrix.local_tail_origin_clean(),
            matrix.local_tail_origin_tail(),
            matrix.local_tail_rate_given_prev_tail(),
            matrix.local_tail_rate_given_prev_not_tail(),
            matrix.amplification_lift(),
            matrix.carry_through(),
            matrix.tail_onset(),
        ],
        &[no_predecessor(); 7],
    );

    let report = snapshot.calibration_report().unwrap();
    let stage = report.stage(destination).unwrap();
    let expected = expected_scores(
        &online,
        |_| false,
        |value| value >= stage.local().estimate().unwrap().value().unwrap(),
        |value| {
            value
                >= stage
                    .cumulative_after()
                    .estimate()
                    .unwrap()
                    .value()
                    .unwrap()
        },
        true,
    );
    let actual = snapshot.calibrated_online_scores(destination).unwrap();
    assert_online(
        &[
            actual.local_tail_origin_clean(),
            actual.local_tail_origin_tail(),
            actual.local_tail_rate_given_prev_tail(),
            actual.local_tail_rate_given_prev_not_tail(),
            actual.amplification_lift(),
            actual.carry_through(),
            actual.tail_onset(),
        ],
        &expected,
    );
    assert_eq!(expected[0].status, ScoreStatus::Available);
    assert_eq!(expected[6].status, ScoreStatus::Available);
    for score in &expected[1..6] {
        assert_eq!(score.status, ScoreStatus::NoPredecessorPopulation);
    }
}

#[test]
fn amplification_lift_reports_zero_denominator() {
    let mut builder = Tracegrams::builder();
    let first = builder.stage("first").unwrap();
    let destination = builder.stage("destination").unwrap();
    builder.tail_quantile(QUANTILE).unwrap();
    let recorder = builder.build().unwrap();
    for event in [
        Event {
            previous: Some(100),
            local: 200,
        },
        Event {
            previous: Some(300),
            local: 400,
        },
    ] {
        record(&recorder, first, destination, event);
    }
    let report = recorder
        .try_freeze_calibration(FreezeCriteria::all_stages(1))
        .unwrap();
    let stage = report.stage(destination).unwrap();
    let previous = stage
        .previous_cumulative()
        .estimate()
        .unwrap()
        .value()
        .unwrap();
    let local = stage.local().estimate().unwrap().value().unwrap();
    record(
        &recorder,
        first,
        destination,
        Event {
            previous: Some(previous.saturating_sub(1)),
            local,
        },
    );
    assert_eq!(
        recorder
            .snapshot_relaxed()
            .calibrated_online_scores(destination)
            .unwrap()
            .amplification_lift()
            .status(),
        ScoreStatus::ZeroDenominator
    );
}
