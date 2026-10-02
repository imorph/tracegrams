//! Public-API differential properties for the recorder state machine.

use std::time::Duration;

use super::support;
use proptest::prelude::*;
use tracegrams::{
    CalibrationPopulation, CalibrationState, CalibrationTerminal, DeltaError, DiagnoseConfig,
    DiagnoseError, FreezeError, ManualCtx, Outcome, Snapshot, StageId, Tracegrams,
};

const SLOTS: usize = 4;

#[derive(Clone, Copy, Debug)]
enum DurationSpec {
    Zero,
    Small(u16),
    Boundary {
        calibration: bool,
        bound: u8,
        offset: i8,
    },
    Max,
    Overflow,
}

#[derive(Clone, Copy, Debug)]
enum MarkStage {
    Registered(u8),
    Foreign,
}

#[derive(Clone, Copy, Debug)]
enum Operation {
    Start(usize, usize),
    Mark(usize, usize, MarkStage, DurationSpec),
    Finish(usize, usize, MarkStage, DurationSpec, bool),
    Drop(usize),
    Snapshot(usize),
    TryFreeze(usize, u64),
    Delta(usize, usize),
    Diagnose(usize, MarkStage),
}

fn duration_strategy() -> impl Strategy<Value = DurationSpec> {
    prop_oneof![
        2 => Just(DurationSpec::Zero),
        8 => any::<u16>().prop_map(DurationSpec::Small),
        8 => (any::<bool>(), any::<u8>(), -1_i8..=1).prop_map(|(calibration, bound, offset)| DurationSpec::Boundary { calibration, bound, offset }),
        1 => Just(DurationSpec::Max),
        1 => Just(DurationSpec::Overflow),
    ]
}

fn stage_strategy(stages: u8) -> impl Strategy<Value = MarkStage> {
    prop_oneof![
        9 => (0_u8..stages).prop_map(MarkStage::Registered),
        1 => Just(MarkStage::Foreign),
    ]
}

fn random_operations(stages: u8) -> impl Strategy<Value = Vec<Operation>> {
    prop::collection::vec(
        prop_oneof![
            3 => (0..SLOTS, 0..2_usize).prop_map(|(s, r)| Operation::Start(s, r)),
            5 => (0..SLOTS, 0..2_usize, stage_strategy(stages), duration_strategy()).prop_map(|(s,r,t,d)| Operation::Mark(s,r,t,d)),
            3 => (0..SLOTS, 0..2_usize, stage_strategy(stages), duration_strategy(), any::<bool>()).prop_map(|(s,r,t,d,e)| Operation::Finish(s,r,t,d,e)),
            1 => (0..SLOTS).prop_map(Operation::Drop),
            2 => (0..2_usize).prop_map(Operation::Snapshot),
            2 => (0..2_usize, 0..=6_u64).prop_map(|(r,n)| Operation::TryFreeze(r,n)),
            2 => (0..8_usize, 0..8_usize).prop_map(|(i,j)| Operation::Delta(i,j)),
            2 => (0..2_usize, stage_strategy(stages)).prop_map(|(r,t)| Operation::Diagnose(r,t)),
        ],
        0..=64,
    )
}

// Interleave four valid increasing paths, rather than spending most generated
// calls rejecting empty slots, repeated stages or foreign contexts.
fn valid_operations(stages: u8) -> impl Strategy<Value = Vec<Operation>> {
    prop::collection::vec(
        (
            0..2_usize,
            prop::collection::btree_set(0..stages, 1..=usize::from(stages)),
            prop::collection::vec(duration_strategy(), usize::from(stages)),
            any::<bool>(),
        ),
        SLOTS,
    )
    .prop_map(move |paths| {
        let mut ops = (0..SLOTS)
            .map(|s| Operation::Start(s, paths[s].0))
            .collect::<Vec<_>>();
        for step in 0..usize::from(stages) {
            for (slot, (owner, path, durations, error)) in paths.iter().enumerate() {
                if let Some(stage) = path.iter().nth(step) {
                    let mark = MarkStage::Registered(*stage);
                    ops.push(if step + 1 == path.len() {
                        Operation::Finish(slot, *owner, mark, durations[step], *error)
                    } else {
                        Operation::Mark(slot, *owner, mark, durations[step])
                    });
                }
            }
            ops.extend([Operation::Snapshot(0), Operation::TryFreeze(step % 2, 1)]);
        }
        ops
    })
}

fn sequence_strategy(stages: u8) -> impl Strategy<Value = Vec<Operation>> {
    (
        1..=3_u64,
        prop_oneof![3 => (100_u16..500).prop_map(DurationSpec::Small),
         1 => Just(DurationSpec::Small(50)), 1 => Just(DurationSpec::Max)],
        any::<bool>(),
        0..=4_u64,
        prop_oneof![valid_operations(stages), random_operations(stages)],
    )
        .prop_map(move |(samples, duration, first_only, minimum, traffic)| {
            let mut ops = Vec::new();
            for owner in 0..2 {
                ops.extend([
                    Operation::Snapshot(owner),
                    Operation::TryFreeze(owner, minimum),
                ]);
                for _ in 0..samples {
                    ops.push(Operation::Start(0, owner));
                    for stage in 0..stages {
                        if first_only {
                            ops.push(Operation::Start(0, owner));
                        }
                        let stage_id = MarkStage::Registered(stage);
                        ops.push(if first_only || stage + 1 == stages {
                            Operation::Finish(0, owner, stage_id, duration, false)
                        } else {
                            Operation::Mark(0, owner, stage_id, duration)
                        });
                    }
                }
                // Keep a context alive across both the generated attempt and retry.
                ops.extend([
                    Operation::Start(owner + 1, owner),
                    Operation::Snapshot(owner),
                    Operation::TryFreeze(owner, minimum),
                    Operation::TryFreeze(owner, 1),
                    Operation::Finish(
                        owner + 1,
                        owner,
                        MarkStage::Registered(stages - 1),
                        DurationSpec::Small(300),
                        true,
                    ),
                    Operation::Snapshot(owner),
                    Operation::Diagnose(owner, MarkStage::Registered(stages - 1)),
                ]);
            }
            // Include valid windows, cross-registry operands, and reverse windows.
            ops.extend([
                Operation::Delta(0, 0),
                Operation::Delta(2, 0),
                Operation::Delta(0, 2),
                Operation::Delta(2, 3),
            ]);
            ops.extend(traffic);
            ops.extend([Operation::Delta(7, 0), Operation::Delta(0, 7)]);
            ops
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
    started_frozen: bool,
}

#[derive(Clone)]
struct Model {
    stages: Vec<StageModel>,
    bounds: Vec<u64>,
    calibration_bounds: Vec<u64>,
    frozen: bool,
    frozen_previous_threshold: Vec<bool>,
    diagnostics: DiagnosticsModel,
    quantile: f64,
}

impl Model {
    fn new(snapshot: &Snapshot) -> Self {
        Self {
            stages: (0..snapshot.stages().len())
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
            frozen_previous_threshold: vec![false; snapshot.stages().len()],
            diagnostics: DiagnosticsModel::default(),
            quantile: snapshot.tail_quantile(),
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
            if context.started_frozen {
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

fn resolve_duration(spec: DurationSpec, model: &Model) -> Duration {
    match spec {
        DurationSpec::Zero => Duration::ZERO,
        DurationSpec::Small(value) => Duration::from_nanos(u64::from(value)),
        DurationSpec::Boundary {
            calibration,
            bound,
            offset,
        } => {
            let bounds = if calibration {
                &model.calibration_bounds
            } else {
                &model.bounds
            };
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

#[derive(Debug, PartialEq)]
enum FreezeFailure {
    Frozen,
    NotReady(StageId, CalibrationPopulation, u128, u64),
    Range(StageId, CalibrationPopulation, CalibrationTerminal),
}

const POPULATIONS: [CalibrationPopulation; 3] = [
    CalibrationPopulation::Local,
    CalibrationPopulation::CumulativeAfter,
    CalibrationPopulation::PreviousCumulative,
];

fn selected_bucket(counts: &[u64], quantile: f64) -> Option<(usize, u128)> {
    let samples = counts.iter().sum::<u64>();
    if samples == 0 {
        return None;
    }
    let rank = support::rank(samples, quantile);
    let mut remaining = rank;
    for (index, count) in counts.iter().enumerate() {
        if remaining <= u128::from(*count) {
            return Some((index, rank));
        }
        remaining -= u128::from(*count);
    }
    unreachable!("nonempty population contains the independently computed rank")
}

fn terminal(bucket: usize, bounds: &[u64]) -> CalibrationTerminal {
    if bucket == 0 {
        CalibrationTerminal::Lower
    } else if bucket == bounds.len() {
        CalibrationTerminal::Upper
    } else {
        CalibrationTerminal::Finite
    }
}

fn expected_freeze(model: &Model, stages: &[StageId], minimum: u64) -> Result<(), FreezeFailure> {
    if model.frozen {
        return Err(FreezeFailure::Frozen);
    }
    for (stage, counts) in stages.iter().zip(&model.stages) {
        for (index, population) in POPULATIONS.iter().enumerate() {
            let samples = counts.calibration[index]
                .iter()
                .map(|v| u128::from(*v))
                .sum::<u128>();
            if !(index == 2 && samples == 0) && (samples == 0 || samples < u128::from(minimum)) {
                return Err(FreezeFailure::NotReady(
                    *stage,
                    *population,
                    samples,
                    minimum,
                ));
            }
        }
    }
    for (stage, counts) in stages.iter().zip(&model.stages) {
        for (index, population) in POPULATIONS.iter().enumerate() {
            if let Some((bucket, _)) = selected_bucket(&counts.calibration[index], model.quantile) {
                let terminal = terminal(bucket, &model.calibration_bounds);
                if terminal != CalibrationTerminal::Finite {
                    return Err(FreezeFailure::Range(*stage, *population, terminal));
                }
            }
        }
    }
    Ok(())
}

fn check_freeze(recorder: &Tracegrams, stages: &[StageId], model: &mut Model, minimum: u64) {
    let expected = expected_freeze(model, stages, minimum);
    let actual = recorder.try_freeze_calibration(minimum);
    let report = match &actual {
        Ok(report) | Err(FreezeError::CalibrationRangeInsufficient { report, .. }) => Some(report),
        _ => None,
    };
    if let Some(report) = report {
        assert_eq!(report.stages.len(), stages.len());
        for (index, stage) in stages.iter().enumerate() {
            let reported = report.stage(*stage).unwrap();
            for (population, counts) in [
                reported.local,
                reported.cumulative_after,
                reported.previous_cumulative,
            ]
            .iter()
            .zip(&model.stages[index].calibration)
            {
                assert_eq!(
                    population.samples,
                    counts.iter().map(|v| u128::from(*v)).sum::<u128>()
                );
                match (population.estimate, selected_bucket(counts, model.quantile)) {
                    (None, None) => {}
                    (Some(estimate), Some((bucket, rank))) => {
                        assert_eq!(estimate.rank, rank);
                        assert_eq!(estimate.samples, population.samples);
                        assert_eq!(
                            estimate.terminal,
                            terminal(bucket, &model.calibration_bounds)
                        );
                    }
                    pair => panic!("estimate disagrees with population: {pair:?}"),
                }
            }
        }
    }
    let actual = actual.map(|_| ()).map_err(|error| match error {
        FreezeError::AlreadyFrozen => FreezeFailure::Frozen,
        FreezeError::NotReady {
            stage,
            population,
            samples,
            minimum,
        } => FreezeFailure::NotReady(stage, population, samples, minimum),
        FreezeError::CalibrationRangeInsufficient {
            stage,
            population,
            terminal,
            ..
        } => FreezeFailure::Range(stage, population, terminal),
        error => panic!("unexpected sequential freeze error: {error:?}"),
    });
    assert_eq!(actual, expected);
    if expected.is_ok() {
        model.freeze();
    }
    assert_snapshot(&recorder.snapshot_relaxed(), stages, model);
}

impl Model {
    fn counts_mut(&mut self) -> impl Iterator<Item = &mut u64> {
        let d = &mut self.diagnostics;
        self.stages
            .iter_mut()
            .flat_map(|s| {
                s.local
                    .iter_mut()
                    .chain(&mut s.cumulative)
                    .chain(s.cause.iter_mut().flatten())
                    .chain(s.incoming.iter_mut().flatten())
                    .chain(s.calibration.iter_mut().flatten())
                    .chain([&mut s.online, &mut s.success, &mut s.error])
            })
            .chain([
                &mut d.invalid_stage,
                &mut d.invalid_context,
                &mut d.non_monotonic,
                &mut d.latency_overflow,
                &mut d.cumulative_overflow,
                &mut d.online_missing_previous_threshold,
            ])
    }

    fn delta(&self, earlier: &Self) -> Option<Self> {
        let mut delta = self.clone();
        let mut earlier = earlier.clone();
        for (later, earlier) in delta.counts_mut().zip(earlier.counts_mut()) {
            *later = later.checked_sub(*earlier)?;
        }
        Some(delta)
    }
}

fn check_diagnosis(snapshot: &Snapshot, stage: StageId, index: Option<usize>, model: &Model) {
    let config = DiagnoseConfig::experimental_defaults();
    let actual = snapshot.diagnose(stage, &config);
    let error = if index.is_none() {
        Some(DiagnoseError::ForeignStage { stage })
    } else if snapshot.spans_freeze() {
        Some(DiagnoseError::WindowSpansFreeze)
    } else if !model.frozen {
        Some(DiagnoseError::NotCalibrated { stage })
    } else {
        None
    };
    if let Some(error) = error {
        assert_eq!(actual, Err(error));
    } else {
        let report = actual.unwrap();
        let counts = &model.stages[index.unwrap()];
        assert_eq!(report.stage(), stage);
        assert_eq!(report.config(), config);
        assert_eq!(
            report.calibrated_online().samples(),
            u128::from(counts.online)
        );
        assert_eq!(
            report.matrix_derived().cause_samples,
            counts
                .cause
                .iter()
                .flatten()
                .map(|v| u128::from(*v))
                .sum::<u128>()
        );
        assert_eq!(
            report.matrix_derived().incoming_samples,
            counts
                .incoming
                .iter()
                .flatten()
                .map(|v| u128::from(*v))
                .sum::<u128>()
        );
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
        let completions = snapshot.completion_counts(stage).unwrap();
        assert_eq!(
            (completions.success, completions.error),
            (expected.success, expected.error)
        );
    }
    let actual = snapshot.diagnostics();
    let expected = model.diagnostics;
    assert_eq!(actual.invalid_stage_marks, expected.invalid_stage);
    assert_eq!(actual.invalid_context_marks, expected.invalid_context);
    assert_eq!(actual.non_monotonic_marks, expected.non_monotonic);
    assert_eq!(actual.latency_overflows, expected.latency_overflow);
    assert_eq!(actual.cumulative_overflows, expected.cumulative_overflow);
    assert_eq!(actual.clock_regressions, 0);
    assert_eq!(actual.calibration_samples_skipped_while_freezing, 0);
    assert_eq!(
        actual.online_samples_skipped_missing_previous_threshold,
        expected.online_missing_previous_threshold
    );
}

fn assert_sample_counts(samples: tracegrams::SampleCounts, expected: &StageModel, cause: &[u64]) {
    assert_eq!(
        samples.local,
        expected.local.iter().map(|v| u128::from(*v)).sum()
    );
    assert_eq!(
        samples.cumulative_after,
        expected.cumulative.iter().map(|v| u128::from(*v)).sum()
    );
    assert_eq!(samples.cause, cause.iter().map(|v| u128::from(*v)).sum());
    let incoming_total = expected
        .incoming
        .iter()
        .flatten()
        .map(|v| u128::from(*v))
        .sum();
    assert_eq!(samples.incoming, incoming_total);
    assert_eq!(
        samples.calibration_local,
        expected.calibration[0].iter().map(|v| u128::from(*v)).sum()
    );
    assert_eq!(
        samples.calibration_cumulative_after,
        expected.calibration[1].iter().map(|v| u128::from(*v)).sum()
    );
    assert_eq!(
        samples.calibration_previous_cumulative,
        expected.calibration[2].iter().map(|v| u128::from(*v)).sum()
    );
    assert_eq!(samples.online, u128::from(expected.online));
}

#[allow(clippy::too_many_lines)] // Operation interpreter and its model transitions.
fn run_case(stage_count: u8, quantile: f64, operations: &[Operation]) {
    let registries = [(); 2].map(|()| {
        let mut builder = Tracegrams::builder();
        builder.tail_quantile(quantile).unwrap();
        let stages = (0..stage_count)
            .map(|i| builder.stage(&format!("stage-{i}")).unwrap())
            .collect::<Vec<_>>();
        (builder.build().unwrap(), stages)
    });
    let mut models = registries
        .each_ref()
        .map(|(recorder, _)| Model::new(&recorder.snapshot_relaxed()));
    let mut contexts: [Option<(usize, ManualCtx, ContextModel)>; SLOTS] =
        std::array::from_fn(|_| None);
    let mut history: Vec<(usize, Snapshot, Model)> = Vec::new();
    let stage_id = |owner: usize, stage: MarkStage| match stage {
        MarkStage::Registered(i) => registries[owner].1[usize::from(i)],
        MarkStage::Foreign => registries[1 - owner].1[0],
    };
    for operation in operations {
        match *operation {
            Operation::Start(slot, owner) => {
                contexts[slot] = Some((
                    owner,
                    registries[owner].0.start_manual(),
                    ContextModel {
                        started_frozen: models[owner].frozen,
                        ..ContextModel::default()
                    },
                ));
            }
            Operation::Drop(slot) => {
                contexts[slot] = None;
            }
            Operation::Mark(slot, owner, stage, duration)
            | Operation::Finish(slot, owner, stage, duration, _) => {
                let Some((context_owner, mut actual, mut expected)) = contexts[slot].take() else {
                    continue;
                };
                let outcome = if let Operation::Finish(_, _, _, _, error) = *operation {
                    Some(if error {
                        Outcome::Error
                    } else {
                        Outcome::Success
                    })
                } else {
                    None
                };
                let duration = resolve_duration(duration, &models[owner]);
                models[owner].mark(
                    &mut expected,
                    context_owner != owner,
                    stage,
                    duration,
                    outcome,
                );
                if let Some(outcome) = outcome {
                    registries[owner].0.finish_manual(
                        actual,
                        stage_id(owner, stage),
                        duration,
                        outcome,
                    );
                } else {
                    registries[owner].0.record_elapsed(
                        &mut actual,
                        stage_id(owner, stage),
                        duration,
                    );
                    contexts[slot] = Some((context_owner, actual, expected));
                }
            }
            Operation::Snapshot(owner) => {
                let snapshot = registries[owner].0.snapshot_relaxed();
                assert_snapshot(&snapshot, &registries[owner].1, &models[owner]);
                if history.len() == 8 {
                    history.remove(0);
                }
                history.push((owner, snapshot, models[owner].clone()));
            }
            Operation::TryFreeze(owner, minimum) => check_freeze(
                &registries[owner].0,
                &registries[owner].1,
                &mut models[owner],
                minimum,
            ),
            Operation::Diagnose(owner, stage) => {
                let index = match stage {
                    MarkStage::Registered(i) => Some(usize::from(i)),
                    MarkStage::Foreign => None,
                };
                check_diagnosis(
                    &registries[owner].0.snapshot_relaxed(),
                    stage_id(owner, stage),
                    index,
                    &models[owner],
                );
            }
            Operation::Delta(i, j) if !history.is_empty() => {
                let (owner, later, expected) = &history[i % history.len()];
                let (other, earlier, before) = &history[j % history.len()];
                let actual = later.delta(earlier);
                if owner != other {
                    assert!(matches!(actual, Err(DeltaError::RegistryMismatch)));
                } else if let Some(delta) = expected.delta(before) {
                    let actual = actual.unwrap();
                    assert_eq!(actual.spans_freeze(), expected.frozen != before.frozen);
                    assert_snapshot(&actual, &registries[*owner].1, &delta);
                    for (index, stage) in registries[*owner].1.iter().enumerate() {
                        check_diagnosis(&actual, *stage, Some(index), &delta);
                    }
                } else {
                    assert!(matches!(actual, Err(DeltaError::CounterUnderflow { .. })));
                }
            }
            Operation::Delta(..) => {}
        }
    }
    for (owner, (recorder, stages)) in registries.iter().enumerate() {
        assert_snapshot(&recorder.snapshot_relaxed(), stages, &models[owner]);
    }
    // Later writes and delta/diagnosis calls must not mutate retained snapshots.
    for (owner, snapshot, model) in history {
        assert_snapshot(&snapshot, &registries[owner].1, &model);
    }
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
    recorder.try_freeze_calibration(1).unwrap();
    model.freeze();

    let mut actual = recorder.start_manual();
    let mut expected = ContextModel {
        started_frozen: true,
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
fn recorder_sequences_retry_terminal_freezes_at_one_and_sixty_four_stages() {
    for stages in [1, 64] {
        for terminal in [DurationSpec::Small(50), DurationSpec::Max] {
            let mut ops = vec![Operation::Snapshot(0), Operation::TryFreeze(0, 0)];
            for stage in 0..stages {
                ops.extend([
                    Operation::Start(0, 0),
                    Operation::Finish(0, 0, MarkStage::Registered(stage), terminal, false),
                ]);
            }
            ops.extend([
                Operation::TryFreeze(0, 1),
                Operation::Snapshot(0),
                Operation::Start(1, 0),
            ]);
            // Two finite samples move the median off either terminal.
            for _ in 0..2 {
                ops.push(Operation::Start(0, 0));
                for stage in 0..stages {
                    ops.push(Operation::Mark(
                        0,
                        0,
                        MarkStage::Registered(stage),
                        DurationSpec::Small(200),
                    ));
                }
            }
            ops.extend([
                Operation::TryFreeze(0, 2),
                Operation::TryFreeze(0, 0),
                Operation::Finish(
                    1,
                    0,
                    MarkStage::Registered(stages - 1),
                    DurationSpec::Small(300),
                    true,
                ),
                Operation::Start(2, 0),
                Operation::Finish(
                    2,
                    0,
                    MarkStage::Registered(stages - 1),
                    DurationSpec::Small(900),
                    false,
                ),
                Operation::Snapshot(0),
                Operation::Delta(2, 0),
                Operation::Delta(2, 1),
                Operation::Delta(0, 2),
            ]);
            run_case(stages, 0.5, &ops);
        }
    }
}

#[test]
fn recorder_state_machine_matches_independent_model() {
    let mut runner = support::runner(96, 0xa5, "proptest-regressions/it/properties.txt");
    let strategy = (
        prop::sample::select(vec![1_u8, 2, 3, 8]),
        support::quantiles(),
    )
        .prop_flat_map(|(stages, quantile)| {
            (Just(stages), Just(quantile), sequence_strategy(stages))
        });
    runner
        .run(&strategy, |(stages, quantile, ops)| {
            run_case(stages, quantile, &ops);
            Ok(())
        })
        .unwrap();
}
