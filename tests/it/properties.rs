//! Public-API differential properties for the recorder state machine.

use std::time::Duration;

use proptest::prelude::*;
use proptest::test_runner::{Config, RngAlgorithm, TestRng, TestRunner};
use tracegrams::{
    CalibrationPopulation, CalibrationState, FreezeCriteria, Outcome, Snapshot, StageId, Tracegrams,
};

const STAGES: usize = 3;
const WARMUP_REQUESTS: usize = 40;

#[derive(Clone, Copy, Debug)]
enum DurationSpec {
    Zero,
    Small(u16),
    Boundary { bound: u8, offset: i8 },
    Max,
    Overflow,
}

#[derive(Clone, Copy, Debug)]
enum MarkStage {
    Registered(u8),
    Foreign,
}

#[derive(Clone, Debug)]
struct Request {
    foreign_context: bool,
    marks: Vec<(MarkStage, DurationSpec)>,
    outcome: Outcome,
}

fn duration_strategy() -> impl Strategy<Value = DurationSpec> {
    prop_oneof![
        2 => Just(DurationSpec::Zero),
        8 => any::<u16>().prop_map(DurationSpec::Small),
        8 => (0_u8..63, -1_i8..=1).prop_map(|(bound, offset)| DurationSpec::Boundary { bound, offset }),
        1 => Just(DurationSpec::Max),
        1 => Just(DurationSpec::Overflow),
    ]
}

fn stage_strategy() -> impl Strategy<Value = MarkStage> {
    prop_oneof![
        9 => (0_u8..3).prop_map(MarkStage::Registered),
        1 => Just(MarkStage::Foreign),
    ]
}

fn request_strategy() -> impl Strategy<Value = Request> {
    (
        proptest::bool::weighted(0.08),
        prop::collection::vec((stage_strategy(), duration_strategy()), 1..=6),
        any::<bool>(),
    )
        .prop_map(|(foreign_context, marks, error)| Request {
            foreign_context,
            marks,
            outcome: if error {
                Outcome::Error
            } else {
                Outcome::Success
            },
        })
}

#[derive(Clone, Debug)]
struct StageModel {
    local: Vec<u64>,
    cumulative: Vec<u64>,
    cause: Vec<Vec<u64>>,
    incoming: Vec<Vec<u64>>,
    calibration: [Vec<u64>; 3],
    online: u64,
    success: u64,
    error: u64,
}

impl StageModel {
    fn new(buckets: usize, calibration_buckets: usize) -> Self {
        Self {
            local: vec![0; buckets],
            cumulative: vec![0; buckets],
            cause: vec![vec![0; buckets]; buckets],
            incoming: vec![vec![0; buckets]; buckets],
            calibration: std::array::from_fn(|_| vec![0; calibration_buckets]),
            online: 0,
            success: 0,
            error: 0,
        }
    }
}

#[derive(Clone, Copy, Debug, Default)]
struct DiagnosticsModel {
    invalid_stage: u64,
    invalid_context: u64,
    non_monotonic: u64,
    latency_overflow: u64,
    cumulative_overflow: u64,
    online_missing_previous_threshold: u64,
}

#[derive(Clone, Copy, Debug, Default)]
struct ContextModel {
    cumulative: u64,
    previous: Option<(usize, usize)>,
    frozen_epoch: bool,
}

struct Model {
    stages: Vec<StageModel>,
    bounds: Vec<u64>,
    calibration_bounds: Vec<u64>,
    frozen: bool,
    frozen_previous_threshold: Vec<bool>,
    diagnostics: DiagnosticsModel,
}

impl Model {
    fn new(snapshot: &Snapshot) -> Self {
        Self {
            stages: (0..STAGES)
                .map(|_| {
                    StageModel::new(
                        snapshot.bucket_bounds().len() + 1,
                        snapshot.calibration_bucket_bounds().len() + 1,
                    )
                })
                .collect(),
            bounds: snapshot.bucket_bounds().to_vec(),
            calibration_bounds: snapshot.calibration_bucket_bounds().to_vec(),
            frozen: false,
            frozen_previous_threshold: vec![false; STAGES],
            diagnostics: DiagnosticsModel::default(),
        }
    }

    // Deliberately independent of the crate's partition-point implementation.
    fn bucket(value: u64, bounds: &[u64]) -> usize {
        bounds.iter().take_while(|bound| value >= **bound).count()
    }

    fn increment(counter: &mut u64) {
        *counter = counter.saturating_add(1);
    }

    fn mark(
        &mut self,
        context: &mut ContextModel,
        foreign_context: bool,
        stage: MarkStage,
        duration: Duration,
        outcome: Option<Outcome>,
    ) {
        if foreign_context {
            Self::increment(&mut self.diagnostics.invalid_context);
            return;
        }
        let MarkStage::Registered(stage) = stage else {
            Self::increment(&mut self.diagnostics.invalid_stage);
            return;
        };
        let stage = usize::from(stage);
        if context
            .previous
            .is_some_and(|(previous, _)| stage <= previous)
        {
            Self::increment(&mut self.diagnostics.non_monotonic);
            return;
        }

        let (local, overflowed) = match u64::try_from(duration.as_nanos()) {
            Ok(value) => (value, false),
            Err(_) => (u64::MAX, true),
        };
        if overflowed {
            Self::increment(&mut self.diagnostics.latency_overflow);
        }
        let cumulative = if let Some(value) = context.cumulative.checked_add(local) {
            value
        } else {
            Self::increment(&mut self.diagnostics.cumulative_overflow);
            u64::MAX
        };
        let local_bucket = Self::bucket(local, &self.bounds);
        let cumulative_bucket = if context.previous.is_none() {
            local_bucket
        } else {
            Self::bucket(cumulative, &self.bounds)
        };
        let stage_model = &mut self.stages[stage];
        Self::increment(&mut stage_model.local[local_bucket]);
        Self::increment(&mut stage_model.cumulative[cumulative_bucket]);
        if let Some((_, previous_bucket)) = context.previous {
            Self::increment(&mut stage_model.cause[previous_bucket][local_bucket]);
            Self::increment(&mut stage_model.incoming[previous_bucket][cumulative_bucket]);
        }

        if self.frozen {
            if context.frozen_epoch {
                if context.previous.is_some() && !self.frozen_previous_threshold[stage] {
                    Self::increment(&mut self.diagnostics.online_missing_previous_threshold);
                } else {
                    Self::increment(&mut stage_model.online);
                }
            }
        } else {
            let local = Self::bucket(local, &self.calibration_bounds);
            let after = if context.previous.is_none() {
                local
            } else {
                Self::bucket(cumulative, &self.calibration_bounds)
            };
            Self::increment(&mut stage_model.calibration[0][local]);
            Self::increment(&mut stage_model.calibration[1][after]);
            if context.previous.is_some() {
                let previous = Self::bucket(context.cumulative, &self.calibration_bounds);
                Self::increment(&mut stage_model.calibration[2][previous]);
            }
        }
        match outcome {
            Some(Outcome::Success) => Self::increment(&mut stage_model.success),
            Some(Outcome::Error) => Self::increment(&mut stage_model.error),
            None => {}
            Some(_) => unreachable!("the public outcome currently has two variants"),
        }
        context.cumulative = cumulative;
        context.previous = Some((stage, cumulative_bucket));
    }

    fn freeze(&mut self) {
        self.frozen_previous_threshold = self
            .stages
            .iter()
            .map(|stage| stage.calibration[2].iter().any(|count| *count != 0))
            .collect();
        self.frozen = true;
    }
}

fn resolve_duration(spec: DurationSpec, bounds: &[u64]) -> Duration {
    match spec {
        DurationSpec::Zero => Duration::ZERO,
        DurationSpec::Small(value) => Duration::from_nanos(u64::from(value)),
        DurationSpec::Boundary { bound, offset } => {
            let bound = bounds[usize::from(bound) % bounds.len()];
            let value = match offset {
                -1 => bound.saturating_sub(1),
                0 => bound,
                _ => bound.saturating_add(1),
            };
            Duration::from_nanos(value)
        }
        DurationSpec::Max => Duration::from_nanos(u64::MAX),
        DurationSpec::Overflow => Duration::MAX,
    }
}

fn drive_request(
    recorder: &Tracegrams,
    foreign: &Tracegrams,
    stages: &[StageId],
    foreign_stage: StageId,
    model: &mut Model,
    request: &Request,
) {
    let owner = if request.foreign_context {
        foreign
    } else {
        recorder
    };
    let mut actual = Some(owner.start_manual());
    let mut expected = ContextModel {
        frozen_epoch: model.frozen,
        ..ContextModel::default()
    };
    for (index, (stage, duration)) in request.marks.iter().copied().enumerate() {
        let duration = resolve_duration(duration, &model.bounds);
        let actual_stage = match stage {
            MarkStage::Registered(index) => stages[usize::from(index)],
            MarkStage::Foreign => foreign_stage,
        };
        let outcome = (index + 1 == request.marks.len()).then_some(request.outcome);
        if let Some(outcome) = outcome {
            recorder.finish_manual(
                actual.take().expect("a request is finished only once"),
                actual_stage,
                duration,
                outcome,
            );
        } else {
            recorder.record_elapsed(
                actual
                    .as_mut()
                    .expect("unfinished request keeps its context"),
                actual_stage,
                duration,
            );
        }
        model.mark(
            &mut expected,
            request.foreign_context,
            stage,
            duration,
            outcome,
        );
        if outcome.is_some() {
            break;
        }
    }
}

fn assert_snapshot(snapshot: &Snapshot, stages: &[StageId], model: &Model) {
    // Sequential operations cover Collecting and Frozen; private deterministic
    // unit tests cover the transient Freezing state without a flaky spin race.
    assert_eq!(
        snapshot.calibration_state(),
        if model.frozen {
            CalibrationState::Frozen
        } else {
            CalibrationState::Collecting
        }
    );
    assert_eq!(snapshot.calibration_epoch(), u16::from(model.frozen));
    assert_eq!(snapshot.calibration_report().is_some(), model.frozen);
    for (index, stage) in stages.iter().copied().enumerate() {
        let expected = &model.stages[index];
        assert_eq!(snapshot.local_counts(stage).unwrap(), expected.local);
        assert_eq!(
            snapshot.cumulative_counts(stage).unwrap(),
            expected.cumulative
        );
        let cause = expected.cause.iter().flatten().copied().collect::<Vec<_>>();
        assert_eq!(snapshot.cause_counts(stage).unwrap(), cause);
        if index == 0 {
            assert!(snapshot.incoming_counts(stage).is_none());
        } else {
            let incoming = expected
                .incoming
                .iter()
                .flatten()
                .copied()
                .collect::<Vec<_>>();
            assert_eq!(snapshot.incoming_counts(stage).unwrap(), incoming);
        }
        for (population, expected_counts) in [
            (CalibrationPopulation::Local, &expected.calibration[0]),
            (
                CalibrationPopulation::CumulativeAfter,
                &expected.calibration[1],
            ),
            (
                CalibrationPopulation::PreviousCumulative,
                &expected.calibration[2],
            ),
        ] {
            assert_eq!(
                snapshot.calibration_counts(stage, population).unwrap(),
                expected_counts
            );
        }
        let samples = snapshot.sample_counts(stage).unwrap();
        assert_sample_counts(samples, expected, &cause);
        let availability = snapshot.score_availability(stage).unwrap();
        assert_eq!(
            availability.matrix_derived(),
            expected.cause.iter().flatten().any(|count| *count != 0)
                && expected.incoming.iter().flatten().any(|count| *count != 0)
        );
        assert_eq!(
            availability.calibrated_online(),
            model.frozen && expected.online != 0
        );
        let completions = snapshot.completion_counts(stage).unwrap();
        assert_eq!(
            (completions.success(), completions.error()),
            (expected.success, expected.error)
        );
    }
    let actual = snapshot.diagnostics();
    let expected = model.diagnostics;
    assert_eq!(actual.invalid_stage_marks(), expected.invalid_stage);
    assert_eq!(actual.invalid_context_marks(), expected.invalid_context);
    assert_eq!(actual.non_monotonic_marks(), expected.non_monotonic);
    assert_eq!(actual.latency_overflows(), expected.latency_overflow);
    assert_eq!(actual.cumulative_overflows(), expected.cumulative_overflow);
    assert_eq!(actual.clock_regressions(), 0);
    assert_eq!(actual.calibration_samples_skipped_while_freezing(), 0);
    assert_eq!(
        actual.online_samples_skipped_missing_previous_threshold(),
        expected.online_missing_previous_threshold
    );
}

fn assert_sample_counts(samples: tracegrams::SampleCounts, expected: &StageModel, cause: &[u64]) {
    assert_eq!(
        samples.local(),
        expected.local.iter().map(|v| u128::from(*v)).sum()
    );
    assert_eq!(
        samples.cumulative_after(),
        expected.cumulative.iter().map(|v| u128::from(*v)).sum()
    );
    assert_eq!(samples.cause(), cause.iter().map(|v| u128::from(*v)).sum());
    let incoming_total = expected
        .incoming
        .iter()
        .flatten()
        .map(|v| u128::from(*v))
        .sum();
    assert_eq!(samples.incoming(), incoming_total);
    assert_eq!(
        samples.calibration_local(),
        expected.calibration[0].iter().map(|v| u128::from(*v)).sum()
    );
    assert_eq!(
        samples.calibration_cumulative_after(),
        expected.calibration[1].iter().map(|v| u128::from(*v)).sum()
    );
    assert_eq!(
        samples.calibration_previous_cumulative(),
        expected.calibration[2].iter().map(|v| u128::from(*v)).sum()
    );
    assert_eq!(samples.online(), u128::from(expected.online));
}

fn run_case(before: &[Request], after: &[Request]) {
    let mut builder = Tracegrams::builder();
    let stages = [
        builder.stage("a").unwrap(),
        builder.stage("b").unwrap(),
        builder.stage("c").unwrap(),
    ];
    builder.tail_quantile(0.5).unwrap();
    let recorder = builder.build().unwrap();
    let mut foreign_builder = Tracegrams::builder();
    let foreign_stage = foreign_builder.stage("foreign").unwrap();
    let foreign = foreign_builder.build().unwrap();
    let mut model = Model::new(&recorder.snapshot_relaxed());

    for _ in 0..WARMUP_REQUESTS {
        let request = Request {
            foreign_context: false,
            marks: vec![
                (MarkStage::Registered(0), DurationSpec::Small(100)),
                (MarkStage::Registered(1), DurationSpec::Small(200)),
                (MarkStage::Registered(2), DurationSpec::Small(300)),
            ],
            outcome: Outcome::Success,
        };
        drive_request(
            &recorder,
            &foreign,
            &stages,
            foreign_stage,
            &mut model,
            &request,
        );
    }
    for request in before {
        drive_request(
            &recorder,
            &foreign,
            &stages,
            foreign_stage,
            &mut model,
            request,
        );
    }
    assert_snapshot(&recorder.snapshot_relaxed(), &stages, &model);

    let mut stale_actual = recorder.start_manual();
    let mut stale_model = ContextModel::default();
    recorder.record_elapsed(&mut stale_actual, stages[0], Duration::from_nanos(101));
    model.mark(
        &mut stale_model,
        false,
        MarkStage::Registered(0),
        Duration::from_nanos(101),
        None,
    );
    recorder
        .try_freeze_calibration(FreezeCriteria::all_stages(1))
        .unwrap();
    model.freeze();
    recorder.finish_manual(
        stale_actual,
        stages[1],
        Duration::from_nanos(202),
        Outcome::Error,
    );
    model.mark(
        &mut stale_model,
        false,
        MarkStage::Registered(1),
        Duration::from_nanos(202),
        Some(Outcome::Error),
    );
    for request in after {
        drive_request(
            &recorder,
            &foreign,
            &stages,
            foreign_stage,
            &mut model,
            request,
        );
    }
    assert_snapshot(&recorder.snapshot_relaxed(), &stages, &model);
}

#[test]
fn predecessor_online_sample_without_frozen_threshold_is_skipped() {
    let mut builder = Tracegrams::builder();
    let stages = [
        builder.stage("a").unwrap(),
        builder.stage("b").unwrap(),
        builder.stage("c").unwrap(),
    ];
    let recorder = builder.build().unwrap();
    let mut model = Model::new(&recorder.snapshot_relaxed());

    // Calibrate every stage only as a first mark, so no stage has a frozen
    // previous-cumulative threshold even though all-stages freeze can succeed.
    for (index, stage) in stages.iter().copied().enumerate() {
        let mut actual = recorder.start_manual();
        let mut expected = ContextModel::default();
        let duration = Duration::from_nanos(100 + index as u64);
        recorder.record_elapsed(&mut actual, stage, duration);
        model.mark(
            &mut expected,
            false,
            MarkStage::Registered(u8::try_from(index).unwrap()),
            duration,
            None,
        );
    }
    recorder
        .try_freeze_calibration(FreezeCriteria::all_stages(1))
        .unwrap();
    model.freeze();

    let mut actual = recorder.start_manual();
    let mut expected = ContextModel {
        frozen_epoch: true,
        ..ContextModel::default()
    };
    recorder.record_elapsed(&mut actual, stages[0], Duration::from_nanos(200));
    model.mark(
        &mut expected,
        false,
        MarkStage::Registered(0),
        Duration::from_nanos(200),
        None,
    );
    recorder.finish_manual(
        actual,
        stages[1],
        Duration::from_nanos(300),
        Outcome::Success,
    );
    model.mark(
        &mut expected,
        false,
        MarkStage::Registered(1),
        Duration::from_nanos(300),
        Some(Outcome::Success),
    );

    assert_snapshot(&recorder.snapshot_relaxed(), &stages, &model);
}

#[test]
fn recorder_state_machine_matches_independent_model() {
    let config = Config {
        cases: 96,
        failure_persistence: Some(Box::new(
            proptest::test_runner::FileFailurePersistence::Direct(
                "proptest-regressions/it/properties.txt",
            ),
        )),
        ..Config::default()
    };
    let rng = TestRng::from_seed(RngAlgorithm::ChaCha, &[0xa5; 32]);
    let mut runner = TestRunner::new_with_rng(config, rng);
    runner
        .run(
            &(
                prop::collection::vec(request_strategy(), 0..=15),
                prop::collection::vec(request_strategy(), 1..=15),
            ),
            |(before, after)| {
                run_case(&before, &after);
                Ok(())
            },
        )
        .unwrap();
}
