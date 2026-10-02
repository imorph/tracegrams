//! Checkpoint overhead, contention, snapshot, and memory screening harness.
//!
//! Timings are medians of aggregate-repeat averages, never per-operation
//! percentiles. First-mark cells use a fresh context and one finish-only
//! mark; transition cells record an intermediate stage and then finish at a
//! second stage. Pipelines of 3+ stages also exercise non-final predecessor
//! marks. Hot-bucket writers contend on the same counter cells, while the
//! transition spread visits the Cartesian product of previous/local buckets.
//! Clocked cells include all request-start and checkpoint clock reads.
//! Multiwriter ns/checkpoint is inverse aggregate throughput, not latency.

#![allow(
    clippy::cast_precision_loss,
    clippy::print_stderr,
    clippy::print_stdout
)]

use std::env;
use std::fs;
use std::hint::black_box;
use std::path::PathBuf;
use std::process::{self, Command};
use std::sync::{Arc, Barrier, mpsc};
use std::thread;
use std::time::{Duration, Instant, SystemTime, UNIX_EPOCH};

use serde::Serialize;
use tracegrams::{DiagnoseConfig, Outcome, Snapshot, StageId, Tracegrams};

const DEFAULT_CHECKPOINTS_PER_WRITER: u64 = 1_000_000;
const DEFAULT_WARMUP_CHECKPOINTS_PER_WRITER: u64 = 50_000;
const DEFAULT_REPEATS: usize = 7;
const DEFAULT_CONTENDED_WRITERS: usize = 16;
const HOT_DURATION_NS: u64 = 1_000;
const MANUAL_BUDGET_NS: f64 = 50.0;
const CLOCKED_BUDGET_NS: f64 = 120.0;
const MANUAL_TRANSITION_BUDGET_NS: f64 = 100.0;
const CLOCKED_TRANSITION_BUDGET_NS: f64 = 180.0;
const HOT_CELL_BUDGET_CHECKPOINTS_PER_SECOND: f64 = 2_000_000.0;
const SNAPSHOT_BUDGET_NS: f64 = 100_000_000.0;
const MEMORY_BUDGET_BYTES: usize = 8 * 1024 * 1024;

fn main() {
    // `cargo test --all-targets` executes harness-free benches. It should
    // validate every cell, not spend minutes screening debug performance.
    if cfg!(test) && cfg!(debug_assertions) {
        let options = Options {
            checkpoints_per_writer: 4097, // A full Cartesian cycle plus one request.
            warmup_checkpoints_per_writer: 3,
            repeats: 1,
            contended_writers: 2,
            ..Options::default()
        };
        black_box(run(&options).expect("benchmark smoke accounting must pass"));
        return;
    }

    let options = match Options::parse(env::args().skip(1)) {
        Ok(options) => options,
        Err(error) if error == usage() => {
            println!("{error}");
            return;
        }
        Err(error) => {
            eprintln!("{error}");
            process::exit(2);
        }
    };

    match run(&options) {
        Ok(report) => {
            print_summary(&report);
            if let Some(path) = &options.output {
                if let Err(error) = write_report(path, &report) {
                    eprintln!("failed to write {}: {error}", path.display());
                    process::exit(1);
                }
                println!("raw report: {}", path.display());
            }
            if options.assert_budgets && report.budgets.iter().any(|budget| !budget.passed) {
                process::exit(1);
            }
        }
        Err(error) => {
            eprintln!("{error}");
            process::exit(1);
        }
    }
}

#[derive(Clone, Debug)]
struct Options {
    checkpoints_per_writer: u64,
    warmup_checkpoints_per_writer: u64,
    repeats: usize,
    contended_writers: usize,
    output: Option<PathBuf>,
    assert_budgets: bool,
}

impl Default for Options {
    fn default() -> Self {
        Self {
            checkpoints_per_writer: DEFAULT_CHECKPOINTS_PER_WRITER,
            warmup_checkpoints_per_writer: DEFAULT_WARMUP_CHECKPOINTS_PER_WRITER,
            repeats: DEFAULT_REPEATS,
            contended_writers: DEFAULT_CONTENDED_WRITERS,
            output: None,
            assert_budgets: false,
        }
    }
}

impl Options {
    fn parse(args: impl IntoIterator<Item = String>) -> Result<Self, String> {
        let mut options = Self::default();
        let args = args.into_iter().collect::<Vec<_>>();
        let mut index = 0;
        while index < args.len() {
            let argument = &args[index];
            if argument == "--" || argument == "--bench" {
                index += 1;
                continue;
            }
            if argument == "--help" || argument == "-h" {
                return Err(usage().to_owned());
            }
            if argument == "--assert-budgets" {
                options.assert_budgets = true;
                index += 1;
                continue;
            }

            let value = args
                .get(index + 1)
                .ok_or_else(|| format!("missing value for {argument}\n\n{}", usage()))?;
            match argument.as_str() {
                "--checkpoints-per-writer" => {
                    options.checkpoints_per_writer = parse_positive(argument, value)?;
                }
                "--warmup-checkpoints-per-writer" => {
                    options.warmup_checkpoints_per_writer = parse_positive(argument, value)?;
                }
                "--repeats" => options.repeats = parse_positive(argument, value)?,
                "--contended-writers" => {
                    options.contended_writers = parse_positive(argument, value)?;
                }
                "--output" => options.output = Some(PathBuf::from(value)),
                _ => return Err(format!("unknown argument {argument}\n\n{}", usage())),
            }
            index += 2;
        }
        Ok(options)
    }
}

fn parse_positive<T>(name: &str, value: &str) -> Result<T, String>
where
    T: std::str::FromStr + PartialEq + From<u8>,
{
    let parsed = value
        .parse::<T>()
        .map_err(|_| format!("{name} must be a positive integer, got {value}"))?;
    if parsed == T::from(0) {
        Err(format!("{name} must be greater than zero"))
    } else {
        Ok(parsed)
    }
}

fn usage() -> &'static str {
    "usage: cargo bench --bench checkpoint -- [--checkpoints-per-writer N] \
[--warmup-checkpoints-per-writer N] [--repeats N] [--contended-writers N] \
[--output PATH] [--assert-budgets]"
}

#[derive(Clone, Copy)]
enum Operation {
    NoopManual,
    ManualCollecting,
    ManualFrozenOnline,
    NoopClocked,
    ClockedCollecting,
    ClockedFrozenOnline,
    ManualTransitionCollecting,
    ManualTransitionFrozenOnline,
    ClockedTransitionCollecting,
    ClockedTransitionFrozenOnline,
    ManualPipeline { stages: u8, frozen: bool },
    ClockedPipeline { stages: u8, frozen: bool },
}

impl Operation {
    const fn name(self) -> &'static str {
        match self {
            Self::NoopManual => "noop-manual",
            Self::ManualCollecting => "manual-collecting",
            Self::ManualFrozenOnline => "manual-frozen-online",
            Self::NoopClocked => "noop-clocked",
            Self::ClockedCollecting => "clocked-collecting",
            Self::ClockedFrozenOnline => "clocked-frozen-online",
            Self::ManualTransitionCollecting => "manual-transition-collecting",
            Self::ManualTransitionFrozenOnline => "manual-transition-frozen-online",
            Self::ClockedTransitionCollecting => "clocked-transition-collecting",
            Self::ClockedTransitionFrozenOnline => "clocked-transition-frozen-online",
            Self::ManualPipeline { frozen: false, .. } => "manual-pipeline-collecting",
            Self::ManualPipeline { frozen: true, .. } => "manual-pipeline-frozen-online",
            Self::ClockedPipeline { frozen: false, .. } => "clocked-pipeline-collecting",
            Self::ClockedPipeline { frozen: true, .. } => "clocked-pipeline-frozen-online",
        }
    }

    const fn instrumented(self) -> bool {
        !matches!(self, Self::NoopManual | Self::NoopClocked)
    }

    const fn frozen(self) -> bool {
        matches!(
            self,
            Self::ManualFrozenOnline
                | Self::ClockedFrozenOnline
                | Self::ManualTransitionFrozenOnline
                | Self::ClockedTransitionFrozenOnline
                | Self::ManualPipeline { frozen: true, .. }
                | Self::ClockedPipeline { frozen: true, .. }
        )
    }

    const fn transition(self) -> bool {
        matches!(
            self,
            Self::ManualTransitionCollecting
                | Self::ManualTransitionFrozenOnline
                | Self::ClockedTransitionCollecting
                | Self::ClockedTransitionFrozenOnline
        )
    }

    fn checkpoints_per_request(self) -> u64 {
        match self {
            Self::ManualPipeline { stages, .. } | Self::ClockedPipeline { stages, .. } => {
                u64::from(stages)
            }
            _ if self.transition() => 2,
            _ => 1,
        }
    }
}

#[derive(Clone, Copy)]
enum Pattern {
    HotBucket,
    BucketSpread,
}

impl Pattern {
    const fn name(self) -> &'static str {
        match self {
            Self::HotBucket => "hot-bucket",
            Self::BucketSpread => "deterministic-bucket-spread",
        }
    }
}

#[derive(Serialize)]
struct Environment {
    unix_time_seconds: u64,
    architecture: &'static str,
    operating_system: &'static str,
    cpu_count: usize,
    cpu_model: String,
    rustc_verbose_version: String,
    cargo_version: String,
    git_commit: String,
    git_dirty: bool,
    debug_assertions: bool,
}

#[derive(Serialize)]
struct Protocol {
    requests_per_writer: u64,
    warmup_requests_per_writer: u64,
    repeats: usize,
    contended_writers: usize,
    release_equivalent: bool,
    preregistered_shape: bool,
    operation_shape: &'static str,
    timing_interpretation: &'static str,
}

#[derive(Serialize)]
struct CellResult {
    name: String,
    operation: &'static str,
    pattern: &'static str,
    writers: usize,
    requests_per_writer: u64,
    checkpoints_per_request: u64,
    raw_elapsed_ns: Vec<u128>,
    repeat_average_ns_per_checkpoint: Vec<f64>,
    repeat_average_ns_per_request: Vec<f64>,
    repeat_aggregate_checkpoints_per_second: Vec<f64>,
    median_ns_per_checkpoint: f64,
    median_ns_per_request: f64,
    median_absolute_deviation_ns: f64,
    min_ns_per_checkpoint: f64,
    max_ns_per_checkpoint: f64,
    median_aggregate_checkpoints_per_second: f64,
    checksum: u128,
    anomaly_counters: Vec<u128>,
    nonzero_local_buckets: Vec<usize>,
    stage_validation: Vec<Vec<StageObservation>>,
    reader_snapshot_raw_ns: Vec<u128>,
}

#[derive(Debug, Serialize)]
struct StageObservation {
    stage_index: usize,
    local: u128,
    cumulative: u128,
    cause: u128,
    incoming: u128,
    completions: u128,
    online: u128,
    online_first: u128,
    online_predecessor: u128,
    nonzero_local_buckets: usize,
    nonzero_cause_cells: usize,
}

#[derive(Serialize)]
struct RejectionResult {
    name: &'static str,
    raw_elapsed_ns: Vec<u128>,
    median_ns_per_rejected_call: f64,
    rejected_calls_per_repeat: u64,
}

#[derive(Serialize)]
struct SnapshotResult {
    stages: usize,
    raw_elapsed_ns: Vec<u128>,
    median_ns: f64,
    median_absolute_deviation_ns: f64,
    exact_estimated_bytes: usize,
    checksum: u128,
}

#[derive(Serialize)]
struct MemoryResult {
    stages: usize,
    matrix_bytes: usize,
    calibration_bytes: usize,
    metadata_bytes: usize,
    exact_estimated_bytes: usize,
}

#[derive(Serialize)]
struct BudgetResult {
    metric: String,
    comparison: &'static str,
    budget: f64,
    observed: f64,
    unit: &'static str,
    passed: bool,
}

#[derive(Serialize)]
struct ScreeningReport {
    schema_version: u8,
    environment: Environment,
    protocol: Protocol,
    cells: Vec<CellResult>,
    rejections: Vec<RejectionResult>,
    snapshots: Vec<SnapshotResult>,
    memory: MemoryResult,
    budgets: Vec<BudgetResult>,
}

#[derive(Clone)]
struct RecorderSetup {
    tracegrams: Tracegrams,
    stages: Arc<[StageId]>,
    spread_durations: Arc<[Duration]>,
}

struct RunObservation {
    elapsed: Duration,
    worker_checksum: u128,
    anomaly_counters: u128,
    sample_delta: u128,
    completion_delta: u128,
    online_delta: u128,
    nonzero_local_buckets: usize,
    stages: Vec<StageObservation>,
    reader_snapshot_ns: Vec<u128>,
}

#[allow(clippy::too_many_lines)]
fn run(options: &Options) -> Result<ScreeningReport, String> {
    let preregistered_shape = options.checkpoints_per_writer >= DEFAULT_CHECKPOINTS_PER_WRITER
        && options.repeats >= DEFAULT_REPEATS
        && options.contended_writers == DEFAULT_CONTENDED_WRITERS;
    let release_equivalent = !cfg!(debug_assertions);
    if options.assert_budgets && (!preregistered_shape || !release_equivalent) {
        return Err(
            "--assert-budgets requires release-equivalent code, >=1M requests/writer, >=7 repeats, and exactly 16 contended writers"
                .to_owned(),
        );
    }

    let operations = [
        Operation::NoopManual,
        Operation::ManualCollecting,
        Operation::ManualFrozenOnline,
        Operation::NoopClocked,
        Operation::ClockedCollecting,
        Operation::ManualTransitionCollecting,
        Operation::ManualTransitionFrozenOnline,
        Operation::ClockedTransitionCollecting,
        Operation::ClockedFrozenOnline,
        Operation::ClockedTransitionFrozenOnline,
    ];
    let mut cells = Vec::new();
    for operation in operations {
        for writers in [1, options.contended_writers] {
            cells.push(measure_cell(
                operation,
                Pattern::HotBucket,
                writers,
                false,
                options,
            )?);
        }
    }
    cells.push(measure_cell(
        Operation::ManualCollecting,
        Pattern::BucketSpread,
        options.contended_writers,
        false,
        options,
    )?);
    cells.push(measure_cell(
        Operation::ManualCollecting,
        Pattern::HotBucket,
        options.contended_writers,
        true,
        options,
    )?);
    for operation in [
        Operation::ManualCollecting,
        Operation::ManualFrozenOnline,
        Operation::ManualTransitionCollecting,
        Operation::ManualTransitionFrozenOnline,
    ] {
        cells.push(measure_cell(
            operation,
            Pattern::BucketSpread,
            1,
            false,
            options,
        )?);
    }
    for stages in [3, 6, 32] {
        for frozen in [false, true] {
            for operation in [
                Operation::ManualPipeline { stages, frozen },
                Operation::ClockedPipeline { stages, frozen },
            ] {
                cells.push(measure_cell(
                    operation,
                    Pattern::HotBucket,
                    1,
                    false,
                    options,
                )?);
            }
            if stages > 3 {
                cells.push(measure_cell(
                    Operation::ManualPipeline { stages, frozen },
                    Pattern::BucketSpread,
                    1,
                    false,
                    options,
                )?);
            }
        }
    }
    let rejections = [false, true]
        .into_iter()
        .map(|foreign| measure_rejections(foreign, options))
        .collect::<Result<Vec<_>, _>>()?;

    let snapshots = [6, 32]
        .into_iter()
        .map(|stages| measure_snapshot(stages, options.repeats))
        .collect::<Result<Vec<_>, _>>()?;
    let memory = measure_memory(32)?;
    let budgets = evaluate_budgets(&cells, &snapshots, &memory, options.contended_writers);

    Ok(ScreeningReport {
        schema_version: 4,
        environment: environment(),
        protocol: Protocol {
            requests_per_writer: options.checkpoints_per_writer,
            warmup_requests_per_writer: options.warmup_checkpoints_per_writer,
            repeats: options.repeats,
            contended_writers: options.contended_writers,
            release_equivalent,
            preregistered_shape,
            operation_shape: "first-mark: start + finish; transition: start + mark(a) + finish(b); N-stage pipeline: start + N-1 marks + finish; ns/checkpoint = ns/request / N, not isolated middle-mark latency; clocked cells include all clock reads; spread transitions cycle all 64 x 64 previous/local bucket pairs",
            timing_interpretation: "median and dispersion across aggregate-repeat averages; not per-operation percentiles; multiwriter ns/checkpoint is inverse aggregate throughput; no no-op subtraction; rejected calls are secondary and ungated",
        },
        cells,
        rejections,
        snapshots,
        memory,
        budgets,
    })
}

#[allow(clippy::too_many_lines)]
fn measure_cell(
    operation: Operation,
    pattern: Pattern,
    writers: usize,
    reader_interference: bool,
    options: &Options,
) -> Result<CellResult, String> {
    let mut elapsed_samples = Vec::with_capacity(options.repeats);
    let mut ns_samples = Vec::with_capacity(options.repeats);
    let mut request_ns_samples = Vec::with_capacity(options.repeats);
    let mut throughput_samples = Vec::with_capacity(options.repeats);
    let mut anomaly_counters = Vec::with_capacity(options.repeats);
    let mut nonzero_local_buckets = Vec::with_capacity(options.repeats);
    let mut reader_snapshot_ns = Vec::new();
    let mut stage_validation = Vec::with_capacity(options.repeats);
    let mut expected_checksum = None;

    for _ in 0..options.repeats {
        let setup = operation
            .instrumented()
            .then(|| recorder_setup(operation.frozen(), operation.checkpoints_per_request()))
            .transpose()?;
        let warmup = run_concurrently(
            operation,
            pattern,
            writers,
            options.warmup_checkpoints_per_writer,
            false,
            setup.as_ref(),
        )?;
        black_box(warmup.worker_checksum);

        let observed = run_concurrently(
            operation,
            pattern,
            writers,
            options.checkpoints_per_writer,
            reader_interference,
            setup.as_ref(),
        )?;
        let total = u128::from(options.checkpoints_per_writer)
            * u128::try_from(writers).map_err(|_| "writer count does not fit u128")?;

        let checksum = observed
            .worker_checksum
            .wrapping_add(observed.sample_delta.wrapping_mul(3))
            .wrapping_add(observed.completion_delta.wrapping_mul(5))
            .wrapping_add(observed.online_delta.wrapping_mul(7));
        if expected_checksum
            .replace(checksum)
            .is_some_and(|expected| expected != checksum)
        {
            return Err(format!(
                "{} produced an unstable checksum",
                operation.name()
            ));
        }

        let elapsed_ns = observed.elapsed.as_nanos();
        let total_checkpoints = total * u128::from(operation.checkpoints_per_request());
        let total_f64 = total_checkpoints as f64;
        let elapsed_ns_f64 = elapsed_ns as f64;
        elapsed_samples.push(elapsed_ns);
        ns_samples.push(elapsed_ns_f64 / total_f64);
        request_ns_samples.push(elapsed_ns_f64 / total as f64);
        throughput_samples.push(total_f64 * 1_000_000_000.0 / elapsed_ns_f64);
        anomaly_counters.push(observed.anomaly_counters);
        nonzero_local_buckets.push(observed.nonzero_local_buckets);
        stage_validation.push(observed.stages);
        reader_snapshot_ns.extend(observed.reader_snapshot_ns);
    }

    let suffix = if reader_interference {
        "-reader-1hz"
    } else {
        ""
    };
    let shape = match operation.checkpoints_per_request() {
        1 => "first-mark".to_owned(),
        2 => "transition".to_owned(),
        stages => format!("pipeline-{stages}"),
    };
    Ok(CellResult {
        name: format!(
            "{shape}-{}-{}-{}w{suffix}",
            operation.name(),
            pattern.name(),
            writers
        ),
        operation: operation.name(),
        pattern: pattern.name(),
        writers,
        requests_per_writer: options.checkpoints_per_writer,
        checkpoints_per_request: operation.checkpoints_per_request(),
        raw_elapsed_ns: elapsed_samples,
        repeat_average_ns_per_checkpoint: ns_samples.clone(),
        median_ns_per_request: median(&request_ns_samples),
        repeat_average_ns_per_request: request_ns_samples,
        repeat_aggregate_checkpoints_per_second: throughput_samples.clone(),
        median_ns_per_checkpoint: median(&ns_samples),
        median_absolute_deviation_ns: median_absolute_deviation(&ns_samples),
        min_ns_per_checkpoint: minimum(&ns_samples),
        max_ns_per_checkpoint: maximum(&ns_samples),
        median_aggregate_checkpoints_per_second: median(&throughput_samples),
        checksum: expected_checksum.unwrap_or(0),
        anomaly_counters,
        nonzero_local_buckets,
        stage_validation,
        reader_snapshot_raw_ns: reader_snapshot_ns,
    })
}

fn recorder_setup(frozen: bool, stages: u64) -> Result<RecorderSetup, String> {
    let (tracegrams, stages) = build_recorder(usize::try_from(stages).unwrap())?;
    if frozen {
        let mut context = tracegrams.start_manual();
        for &stage in &stages {
            tracegrams.record_elapsed(&mut context, stage, Duration::from_nanos(HOT_DURATION_NS));
        }
        tracegrams
            .try_freeze_calibration(1)
            .map_err(|error| error.to_string())?;
    }
    let spread_durations = bucket_representatives(&tracegrams)
        .into_iter()
        .map(Duration::from_nanos)
        .collect::<Vec<_>>()
        .into();
    Ok(RecorderSetup {
        tracegrams,
        stages: stages.into(),
        spread_durations,
    })
}

fn bucket_representatives(tracegrams: &Tracegrams) -> Vec<u64> {
    let snapshot = tracegrams.snapshot_relaxed();
    let bounds = snapshot.bucket_bounds();
    let mut values = Vec::with_capacity(bounds.len() + 1);
    values.push(bounds[0] / 2);
    values.extend(
        bounds
            .windows(2)
            .map(|pair| pair[0] + (pair[1] - pair[0]) / 2),
    );
    values.push(*bounds.last().expect("the pinned table is non-empty"));
    values
}

#[allow(clippy::too_many_lines)]
fn run_concurrently(
    operation: Operation,
    pattern: Pattern,
    writers: usize,
    checkpoints_per_writer: u64,
    reader_interference: bool,
    setup: Option<&RecorderSetup>,
) -> Result<RunObservation, String> {
    let before = setup.map(|setup| setup.tracegrams.snapshot_relaxed());
    let participants = writers + 1 + usize::from(reader_interference);
    let barrier = Arc::new(Barrier::new(participants));
    let mut workers = Vec::with_capacity(writers);
    for _ in 0..writers {
        let barrier = Arc::clone(&barrier);
        let setup = setup.cloned();
        workers.push(thread::spawn(move || {
            barrier.wait();
            run_checkpoints(operation, pattern, checkpoints_per_writer, setup.as_ref())
        }));
    }

    let (stop_sender, reader) = if reader_interference {
        let tracegrams = setup
            .ok_or_else(|| "reader interference requires an instrumented cell".to_owned())?
            .tracegrams
            .clone();
        let barrier = Arc::clone(&barrier);
        let (sender, receiver) = mpsc::channel();
        let reader = thread::spawn(move || {
            barrier.wait();
            let mut elapsed = Vec::new();
            loop {
                let started = Instant::now();
                black_box(tracegrams.snapshot_relaxed());
                elapsed.push(started.elapsed().as_nanos());
                match receiver.recv_timeout(Duration::from_secs(1)) {
                    Ok(()) | Err(mpsc::RecvTimeoutError::Disconnected) => break,
                    Err(mpsc::RecvTimeoutError::Timeout) => {}
                }
            }
            elapsed
        });
        (Some(sender), Some(reader))
    } else {
        (None, None)
    };

    let started = Instant::now();
    barrier.wait();
    let mut worker_checksum = 0_u128;
    for worker in workers {
        worker_checksum = worker_checksum.wrapping_add(
            worker
                .join()
                .map_err(|_| "checkpoint worker panicked".to_owned())?,
        );
    }
    let elapsed = started.elapsed();
    drop(stop_sender);
    let reader_snapshot_ns = reader
        .map(|reader| {
            reader
                .join()
                .map_err(|_| "snapshot reader panicked".to_owned())
        })
        .transpose()?
        .unwrap_or_default();

    let mut stages = Vec::new();
    if let (Some(setup), Some(before)) = (setup, before.as_ref()) {
        let after = setup.tracegrams.snapshot_relaxed();
        let delta = after.delta(before).map_err(|error| error.to_string())?;
        if delta.diagnostics().total() != 0 {
            return Err(format!(
                "{} unexpected diagnostics: {:?}",
                operation.name(),
                delta.diagnostics()
            ));
        }
        stages = validate_stages(&delta, operation, pattern, checkpoints_per_writer, writers)?;
    }

    Ok(RunObservation {
        elapsed,
        worker_checksum,
        anomaly_counters: 0,
        sample_delta: stages.last().map_or(0, |stage| stage.local),
        completion_delta: stages.last().map_or(0, |stage| stage.completions),
        online_delta: stages.last().map_or(0, |stage| stage.online),
        nonzero_local_buckets: stages.last().map_or(0, |stage| stage.nonzero_local_buckets),
        stages,
        reader_snapshot_ns,
    })
}

#[allow(clippy::too_many_lines)]
fn validate_stages(
    delta: &Snapshot,
    operation: Operation,
    pattern: Pattern,
    requests: u64,
    writers: usize,
) -> Result<Vec<StageObservation>, String> {
    let total = u128::from(requests) * writers as u128;
    let nonzero = |counts: &[u64]| counts.iter().filter(|count| **count != 0).count();
    let mut stages = Vec::new();
    for (index, metadata) in delta.stages().iter().enumerate() {
        let destination = metadata.id;
        let samples = delta
            .sample_counts(destination)
            .expect("registered stage has sample counts");
        let completion = delta.completion_counts(destination).unwrap();
        let mut observed = StageObservation {
            stage_index: index,
            local: samples.local,
            cumulative: samples.cumulative_after,
            cause: samples.cause,
            incoming: samples.incoming,
            completions: u128::from(completion.success),
            online: samples.online,
            online_first: 0,
            online_predecessor: 0,
            nonzero_local_buckets: nonzero(delta.local_counts(destination).unwrap()),
            nonzero_cause_cells: nonzero(delta.cause_counts(destination).unwrap()),
        };
        let predecessors = if index == 0 { 0 } else { total };
        let online = if operation.frozen() { total } else { 0 };
        let calibration = if operation.frozen() { 0 } else { total };
        if samples.local != total
            || samples.cumulative_after != total
            || samples.cause != predecessors
            || samples.incoming != predecessors
            || observed.completions
                != if index + 1 == delta.stages().len() {
                    total
                } else {
                    0
                }
            || completion.error != 0
            || samples.online != online
            || samples.calibration_local != calibration
            || samples.calibration_cumulative_after != calibration
            || samples.calibration_previous_cumulative != if index == 0 { 0 } else { calibration }
        {
            return Err(format!(
                "{} stage {index} accounting mismatch: {samples:?}, {completion:?}, expected {total} requests",
                operation.name()
            ));
        }
        if operation.frozen() {
            let report = delta
                .diagnose(destination, &DiagnoseConfig::experimental_defaults())
                .map_err(|error| error.to_string())?;
            let scores = report.calibrated_online();
            observed.online_first = scores.first_samples;
            observed.online_predecessor = scores.predecessor_samples;
            if scores.predecessor_samples != predecessors
                || scores.first_samples != total - predecessors
            {
                return Err(format!(
                    "{} online population mismatch: {observed:?}",
                    operation.name()
                ));
            }
            // With a full Cartesian cycle, each previous/local predicate must
            // be exercised on both sides, not merely all samples classified.
            if operation.transition()
                && index == 1
                && matches!(pattern, Pattern::BucketSpread)
                && requests >= 4096
            {
                for rate in [
                    scores.scores.local_tail_rate_given_prev_tail,
                    scores.scores.local_tail_rate_given_prev_not_tail,
                ] {
                    if rate.numerator() == 0 || rate.numerator() >= rate.denominator() {
                        return Err(format!(
                            "frozen spread did not straddle both thresholds: {rate:?}"
                        ));
                    }
                }
            }
        }
        if matches!(pattern, Pattern::BucketSpread) {
            let expected_local = if operation.transition() && index == 1 {
                requests.div_ceil(64).min(64)
            } else {
                requests.min(64)
            };
            if observed.nonzero_local_buckets as u64 != expected_local
                || (operation.transition()
                    && index == 1
                    && observed.nonzero_cause_cells as u64 != requests.min(4096))
            {
                return Err(format!(
                    "{} spread mismatch: {observed:?}",
                    operation.name()
                ));
            }
        }
        stages.push(observed);
    }
    Ok(stages)
}

#[allow(clippy::too_many_lines)]
fn run_checkpoints(
    operation: Operation,
    pattern: Pattern,
    checkpoints: u64,
    setup: Option<&RecorderSetup>,
) -> u128 {
    let tracegrams = setup.map(|setup| &setup.tracegrams);
    let stage_a = setup.map(|setup| setup.stages[0]);
    let stage_b = setup.and_then(|setup| setup.stages.get(1).copied());
    let spread_durations = setup.map(|setup| &*setup.spread_durations);
    let mut checksum = 0_u128;
    for checkpoint in 0..checkpoints {
        let duration = match pattern {
            Pattern::HotBucket => Duration::from_nanos(HOT_DURATION_NS),
            Pattern::BucketSpread => {
                let durations = spread_durations.expect("spread cells have fixed durations");
                durations[usize::try_from(checkpoint).unwrap_or(0) % durations.len()]
            }
        };
        match operation {
            Operation::NoopManual => {
                black_box(duration);
            }
            Operation::ManualCollecting | Operation::ManualFrozenOnline => {
                let tracegrams = tracegrams.expect("manual cells have a recorder");
                let stage = stage_a.expect("manual cells have a stage");
                tracegrams.finish_manual(
                    black_box(tracegrams.start_manual()),
                    black_box(stage),
                    black_box(duration),
                    Outcome::Success,
                );
            }
            Operation::NoopClocked => {
                let started = Instant::now();
                black_box(Instant::now().checked_duration_since(started));
            }
            Operation::ClockedCollecting | Operation::ClockedFrozenOnline => {
                let tracegrams = tracegrams.expect("clocked cells have a recorder");
                let stage = stage_a.expect("clocked cells have a stage");
                tracegrams.finish(
                    black_box(tracegrams.start()),
                    black_box(stage),
                    Outcome::Success,
                );
            }
            Operation::ManualTransitionCollecting | Operation::ManualTransitionFrozenOnline => {
                let tracegrams = tracegrams.expect("manual transition cells have a recorder");
                let local = match pattern {
                    Pattern::HotBucket => duration,
                    Pattern::BucketSpread => {
                        // Previous varies every request; local varies every 64.
                        spread_durations.unwrap()[usize::try_from((checkpoint / 64) % 64).unwrap()]
                    }
                };
                let mut context = black_box(tracegrams.start_manual());
                tracegrams.record_elapsed(
                    &mut context,
                    black_box(stage_a.expect("transition cells have stage_a")),
                    black_box(duration),
                );
                tracegrams.finish_manual(
                    context,
                    black_box(stage_b.expect("transition cells have stage_b")),
                    black_box(local),
                    Outcome::Success,
                );
            }
            Operation::ClockedTransitionCollecting | Operation::ClockedTransitionFrozenOnline => {
                let tracegrams = tracegrams.expect("clocked transition cells have a recorder");
                let mut context = black_box(tracegrams.start());
                tracegrams.mark(
                    &mut context,
                    black_box(stage_a.expect("transition cells have stage_a")),
                );
                tracegrams.finish(
                    context,
                    black_box(stage_b.expect("transition cells have stage_b")),
                    Outcome::Success,
                );
            }
            Operation::ManualPipeline { .. } => {
                let setup = setup.unwrap();
                let (last, middle) = setup.stages.split_last().unwrap();
                let mut context = black_box(setup.tracegrams.start_manual());
                for (index, stage) in middle.iter().enumerate() {
                    let local = match pattern {
                        Pattern::HotBucket => duration,
                        // Stage offsets expand the active matrix working set;
                        // independence is tested in the two-stage spread cell.
                        Pattern::BucketSpread => {
                            setup.spread_durations
                                [(usize::try_from(checkpoint % 64).unwrap() + 17 * index) % 64]
                        }
                    };
                    setup.tracegrams.record_elapsed(
                        &mut context,
                        black_box(*stage),
                        black_box(local),
                    );
                }
                setup.tracegrams.finish_manual(
                    context,
                    black_box(*last),
                    black_box(duration),
                    Outcome::Success,
                );
            }
            Operation::ClockedPipeline { .. } => {
                let setup = setup.unwrap();
                let (last, middle) = setup.stages.split_last().unwrap();
                let mut context = black_box(setup.tracegrams.start());
                for stage in middle {
                    setup.tracegrams.mark(&mut context, black_box(*stage));
                }
                setup
                    .tracegrams
                    .finish(context, black_box(*last), Outcome::Success);
            }
        }
        checksum = checksum.wrapping_add(u128::from(checkpoint) + 1);
    }
    black_box(checksum)
}

fn measure_rejections(foreign: bool, options: &Options) -> Result<RejectionResult, String> {
    let (tracegrams, stages) = build_recorder(1)?;
    let (_other, other_stages) = build_recorder(1)?;
    let stage = if foreign { other_stages[0] } else { stages[0] };
    let mut context = tracegrams.start_manual();
    tracegrams.record_elapsed(
        &mut context,
        stages[0],
        Duration::from_nanos(HOT_DURATION_NS),
    );
    let mut raw_elapsed_ns = Vec::with_capacity(options.repeats);
    for repeat in 0..=options.repeats {
        let calls = if repeat == 0 {
            options.warmup_checkpoints_per_writer
        } else {
            options.checkpoints_per_writer
        };
        let before = tracegrams.snapshot_relaxed();
        let started = Instant::now();
        for _ in 0..calls {
            tracegrams.record_elapsed(
                black_box(&mut context),
                black_box(stage),
                black_box(Duration::from_nanos(HOT_DURATION_NS)),
            );
        }
        let elapsed = started.elapsed().as_nanos();
        let after = tracegrams.snapshot_relaxed();
        let delta = after.delta(&before).map_err(|error| error.to_string())?;
        let diagnostics = delta.diagnostics();
        let rejected = if foreign {
            diagnostics.invalid_stage_marks
        } else {
            diagnostics.non_monotonic_marks
        };
        if rejected != calls
            || diagnostics.total() != u128::from(calls)
            || after.sample_counts(stages[0]) != before.sample_counts(stages[0])
            || after.completion_counts(stages[0]) != before.completion_counts(stages[0])
        {
            return Err(format!(
                "rejected-mark accounting mismatch: {diagnostics:?}"
            ));
        }
        if repeat != 0 {
            raw_elapsed_ns.push(elapsed);
        }
    }
    let samples = raw_elapsed_ns
        .iter()
        .map(|ns| *ns as f64 / options.checkpoints_per_writer as f64)
        .collect::<Vec<_>>();
    Ok(RejectionResult {
        name: if foreign {
            "manual-foreign-stage"
        } else {
            "manual-non-monotonic"
        },
        median_ns_per_rejected_call: median(&samples),
        raw_elapsed_ns,
        rejected_calls_per_repeat: options.checkpoints_per_writer,
    })
}

fn measure_snapshot(stages: usize, repeats: usize) -> Result<SnapshotResult, String> {
    let (tracegrams, stage_ids) = build_recorder(stages)?;
    let mut context = tracegrams.start_manual();
    for stage in stage_ids {
        tracegrams.record_elapsed(&mut context, stage, Duration::from_nanos(HOT_DURATION_NS));
    }
    for _ in 0..3 {
        black_box(tracegrams.snapshot_relaxed());
    }

    let mut raw_elapsed_ns = Vec::with_capacity(repeats);
    let mut checksum = None;
    for _ in 0..repeats {
        let started = Instant::now();
        let snapshot = tracegrams.snapshot_relaxed();
        let elapsed = started.elapsed().as_nanos();
        let observed = snapshot
            .stages()
            .iter()
            .map(|stage| snapshot.sample_counts(stage.id).unwrap().local)
            .sum::<u128>()
            .wrapping_add(snapshot.memory_estimate().total_bytes() as u128);
        if checksum
            .replace(observed)
            .is_some_and(|expected| expected != observed)
        {
            return Err(format!("{stages}-stage snapshot checksum changed"));
        }
        black_box(&snapshot);
        raw_elapsed_ns.push(elapsed);
    }
    let samples = raw_elapsed_ns
        .iter()
        .map(|elapsed| *elapsed as f64)
        .collect::<Vec<_>>();
    let snapshot = tracegrams.snapshot_relaxed();
    Ok(SnapshotResult {
        stages,
        raw_elapsed_ns,
        median_ns: median(&samples),
        median_absolute_deviation_ns: median_absolute_deviation(&samples),
        exact_estimated_bytes: snapshot.memory_estimate().total_bytes(),
        checksum: checksum.unwrap_or(0),
    })
}

fn measure_memory(stages: usize) -> Result<MemoryResult, String> {
    let (tracegrams, _) = build_recorder(stages)?;
    let snapshot = tracegrams.snapshot_relaxed();
    let estimate = snapshot.memory_estimate();
    Ok(MemoryResult {
        stages,
        matrix_bytes: estimate.matrix_bytes,
        calibration_bytes: estimate.calibration_bytes,
        metadata_bytes: estimate.metadata_bytes,
        exact_estimated_bytes: estimate.total_bytes(),
    })
}

fn build_recorder(stages: usize) -> Result<(Tracegrams, Vec<StageId>), String> {
    let mut builder = Tracegrams::builder();
    let mut stage_ids = Vec::with_capacity(stages);
    for stage in 0..stages {
        stage_ids.push(
            builder
                .stage(&format!("stage-{stage}"))
                .map_err(|error| error.to_string())?,
        );
    }
    let tracegrams = builder.build().map_err(|error| error.to_string())?;
    Ok((tracegrams, stage_ids))
}

fn evaluate_budgets(
    cells: &[CellResult],
    snapshots: &[SnapshotResult],
    memory: &MemoryResult,
    contended_writers: usize,
) -> Vec<BudgetResult> {
    let at_most = |metric: String, budget: f64, observed: f64, unit| BudgetResult {
        metric,
        comparison: "<=",
        budget,
        observed,
        unit,
        passed: observed <= budget,
    };
    let mut budgets = Vec::new();
    for (operation, budget) in [
        ("manual-collecting", MANUAL_BUDGET_NS),
        ("manual-frozen-online", MANUAL_BUDGET_NS),
        ("clocked-collecting", CLOCKED_BUDGET_NS),
        ("clocked-frozen-online", CLOCKED_BUDGET_NS),
        ("manual-transition-collecting", MANUAL_TRANSITION_BUDGET_NS),
        (
            "manual-transition-frozen-online",
            MANUAL_TRANSITION_BUDGET_NS,
        ),
        (
            "clocked-transition-collecting",
            CLOCKED_TRANSITION_BUDGET_NS,
        ),
        (
            "clocked-transition-frozen-online",
            CLOCKED_TRANSITION_BUDGET_NS,
        ),
    ] {
        let cell = find_cell(cells, operation, "hot-bucket", 1, false);
        budgets.push(at_most(
            format!("{operation} uncontended median"),
            budget,
            cell.median_ns_per_checkpoint,
            "ns/checkpoint",
        ));
    }
    // T4 baseline, 2026-10-02 UTC, M3 Max, 1M requests x 7 repeats:
    // manual spread/pipelines <=22.64 ns/cp, clocked pipelines <=65.99.
    // Reuse the existing operation-class budgets after measuring these cells;
    // these are coarse regression guards, not fitted platform-specific targets.
    for cell in cells.iter().filter(|cell| {
        cell.writers == 1
            && (cell.checkpoints_per_request >= 3 || cell.pattern == Pattern::BucketSpread.name())
    }) {
        let budget = if cell.operation.starts_with("clocked-") {
            CLOCKED_TRANSITION_BUDGET_NS
        } else if cell.checkpoints_per_request == 1 {
            MANUAL_BUDGET_NS
        } else {
            MANUAL_TRANSITION_BUDGET_NS
        };
        budgets.push(at_most(
            format!("{} uncontended median", cell.name),
            budget,
            cell.median_ns_per_checkpoint,
            "ns/checkpoint",
        ));
    }
    for operation in [
        "manual-collecting",
        "manual-frozen-online",
        "clocked-collecting",
        "clocked-frozen-online",
    ] {
        let cell = find_cell(cells, operation, "hot-bucket", contended_writers, false);
        let observed = cell.median_aggregate_checkpoints_per_second;
        budgets.push(BudgetResult {
            metric: format!("{operation} {contended_writers}-writer hot-cell throughput"),
            comparison: ">=",
            budget: HOT_CELL_BUDGET_CHECKPOINTS_PER_SECOND,
            observed,
            unit: "checkpoints/s",
            passed: observed >= HOT_CELL_BUDGET_CHECKPOINTS_PER_SECOND,
        });
    }
    for snapshot in snapshots {
        budgets.push(at_most(
            format!("{}-stage snapshot median", snapshot.stages),
            SNAPSHOT_BUDGET_NS,
            snapshot.median_ns,
            "ns/snapshot",
        ));
    }
    budgets.push(at_most(
        "32-stage estimated memory".to_owned(),
        MEMORY_BUDGET_BYTES as f64,
        memory.exact_estimated_bytes as f64,
        "bytes",
    ));
    budgets
}

fn find_cell<'a>(
    cells: &'a [CellResult],
    operation: &str,
    pattern: &str,
    writers: usize,
    reader_interference: bool,
) -> &'a CellResult {
    cells
        .iter()
        .find(|cell| {
            cell.operation == operation
                && cell.pattern == pattern
                && cell.writers == writers
                && cell.name.ends_with("-reader-1hz") == reader_interference
        })
        .expect("the fixed screening matrix contains every gated cell")
}

fn environment() -> Environment {
    Environment {
        unix_time_seconds: SystemTime::now()
            .duration_since(UNIX_EPOCH)
            .unwrap_or_default()
            .as_secs(),
        architecture: env::consts::ARCH,
        operating_system: env::consts::OS,
        cpu_count: thread::available_parallelism().map_or(1, usize::from),
        cpu_model: cpu_model(),
        rustc_verbose_version: command_output("rustc", &["-Vv"]),
        cargo_version: command_output("cargo", &["-V"]),
        git_commit: command_output("git", &["rev-parse", "HEAD"]),
        git_dirty: !command_output("git", &["status", "--porcelain"]).is_empty(),
        debug_assertions: cfg!(debug_assertions),
    }
}

fn cpu_model() -> String {
    if env::consts::OS == "macos" {
        command_output("sysctl", &["-n", "machdep.cpu.brand_string"])
    } else if env::consts::OS == "linux" {
        command_output(
            "sh",
            &[
                "-c",
                "lscpu | awk -F: '/Model name/ {gsub(/^[ \\t]+/, \"\", $2); print $2; exit}'",
            ],
        )
    } else {
        String::new()
    }
}

fn command_output(program: &str, arguments: &[&str]) -> String {
    Command::new(program)
        .args(arguments)
        .output()
        .ok()
        .filter(|output| output.status.success())
        .and_then(|output| String::from_utf8(output.stdout).ok())
        .map(|output| output.trim().to_owned())
        .unwrap_or_default()
}

fn write_report(path: &PathBuf, report: &ScreeningReport) -> Result<(), String> {
    if let Some(parent) = path.parent() {
        fs::create_dir_all(parent).map_err(|error| error.to_string())?;
    }
    let json = serde_json::to_vec_pretty(report).map_err(|error| error.to_string())?;
    fs::write(path, json).map_err(|error| error.to_string())
}

fn print_summary(report: &ScreeningReport) {
    println!("tracegrams core screening");
    println!(
        "machine: {} {} ({})",
        report.environment.architecture,
        report.environment.operating_system,
        report.environment.cpu_model
    );
    println!(
        "protocol: {} requests/writer, {} warmup requests/writer, {} repeats, {} contended writers",
        report.protocol.requests_per_writer,
        report.protocol.warmup_requests_per_writer,
        report.protocol.repeats,
        report.protocol.contended_writers
    );
    println!("operation: {}", report.protocol.operation_shape);
    println!("timing: {}", report.protocol.timing_interpretation);
    println!();
    println!(
        "{:<72} {:>12} {:>12} {:>12} {:>14}",
        "cell", "ns/cp", "ns/request", "MAD ns/cp", "aggregate cp/s"
    );
    for cell in &report.cells {
        println!(
            "{:<72} {:>12.2} {:>12.2} {:>12.2} {:>14.0}",
            cell.name,
            cell.median_ns_per_checkpoint,
            cell.median_ns_per_request,
            cell.median_absolute_deviation_ns,
            cell.median_aggregate_checkpoints_per_second
        );
    }
    println!();
    for rejection in &report.rejections {
        println!(
            "{}: {:.2} ns/rejected call (secondary)",
            rejection.name, rejection.median_ns_per_rejected_call
        );
    }
    for snapshot in &report.snapshots {
        println!(
            "snapshot {:>2} stages: median {:.0} ns, MAD {:.0} ns, {} bytes",
            snapshot.stages,
            snapshot.median_ns,
            snapshot.median_absolute_deviation_ns,
            snapshot.exact_estimated_bytes
        );
    }
    println!();
    for budget in &report.budgets {
        println!(
            "{}: {} (observed {:.2} {} {} {:.2})",
            if budget.passed { "PASS" } else { "MISS" },
            budget.metric,
            budget.observed,
            budget.unit,
            budget.comparison,
            budget.budget
        );
    }
}

fn median(samples: &[f64]) -> f64 {
    let mut sorted = samples.to_vec();
    sorted.sort_by(f64::total_cmp);
    let middle = sorted.len() / 2;
    if sorted.len() % 2 == 0 {
        sorted[middle - 1] / 2.0 + sorted[middle] / 2.0
    } else {
        sorted[middle]
    }
}

fn median_absolute_deviation(samples: &[f64]) -> f64 {
    let sample_median = median(samples);
    let deviations = samples
        .iter()
        .map(|sample| (sample - sample_median).abs())
        .collect::<Vec<_>>();
    median(&deviations)
}

fn minimum(samples: &[f64]) -> f64 {
    samples.iter().copied().fold(f64::INFINITY, f64::min)
}

fn maximum(samples: &[f64]) -> f64 {
    samples.iter().copied().fold(f64::NEG_INFINITY, f64::max)
}
