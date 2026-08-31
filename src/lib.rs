//! Aggregate tail-latency propagation tracking without storing traces.
//!
//! Tracegrams records bounded aggregate latency transitions for a linear,
//! monotonically ordered request pipeline. A request carries a small context;
//! checkpoints update one fixed shared atomic recorder without retaining the
//! request or a trace.
//!
//! # Lifecycle
//!
//! Register stages once, collect a representative warm-up population, explicitly
//! freeze its tail thresholds, then diagnose a relaxed incident-window delta:
//!
//! ```
//! use std::time::Duration;
//! use tracegrams::{DiagnoseConfig, FreezeCriteria, Outcome, Tracegrams};
//!
//! # fn main() -> Result<(), Box<dyn std::error::Error>> {
//! let mut builder = Tracegrams::builder();
//! let parse = builder.stage("parse")?;
//! let database = builder.stage("database")?;
//! builder.tail_quantile(0.99)?;
//! let tracegrams = builder.build()?;
//!
//! // Manual timing keeps this example deterministic. Production handlers use
//! // Tracegrams::start, Tracegrams::mark, and Tracegrams::finish instead.
//! let mut warmup = tracegrams.start_manual();
//! tracegrams.record_elapsed(&mut warmup, parse, Duration::from_millis(1));
//! tracegrams.finish_manual(
//!     warmup,
//!     database,
//!     Duration::from_millis(3),
//!     Outcome::Success,
//! );
//! tracegrams.try_freeze_calibration(FreezeCriteria::all_stages(1))?;
//!
//! let before = tracegrams.snapshot_relaxed();
//! let mut incident = tracegrams.start_manual();
//! tracegrams.record_elapsed(&mut incident, parse, Duration::from_micros(100));
//! tracegrams.finish_manual(
//!     incident,
//!     database,
//!     Duration::from_millis(30),
//!     Outcome::Success,
//! );
//! let after = tracegrams.snapshot_relaxed();
//!
//! let window = after.delta(&before)?;
//! let policy = DiagnoseConfig::experimental_defaults();
//! let report = window.diagnose(database, &policy)?;
//! let display = report.to_string();
//! assert!(display.contains("tracegrams diagnosis: database"));
//! # Ok(())
//! # }
//! ```
//!
//! [`DiagnoseConfig::experimental_defaults`] is the synthetic proof-of-concept
//! policy, not a production default. Applications should supply named thresholds
//! appropriate to their workload.
//!
//! # Consistency and scope
//!
//! [`Tracegrams::snapshot_relaxed`] is an eventually consistent atomic scan, not
//! a single-instant cut. Matrix-derived scores cover predecessor-bearing marks
//! only and are explicitly labeled as a known-drift path; calibrated-online
//! scores also preserve clean-sentinel first marks. This v0 supports one explicit
//! calibration freeze and linear pipelines only; fan-out, retries, repeated
//! stages, and recalibration are outside its contract.

mod bucket;
mod calibration;
mod context;
mod diagnose;
mod init;
mod matrix;
mod recorder;
mod snapshot;

pub use calibration::{
    CalibrationConsistency, CalibrationEstimate, CalibrationPopulationReport, CalibrationReadiness,
    CalibrationTerminal, CalibrationThresholdAvailability, FreezeCriteria, FreezeError,
    FreezeReport, PopulationReadiness, StageCalibrationReadiness, StageCalibrationReport,
};
pub use context::{Ctx, ManualCtx, Outcome};
pub use diagnose::{
    BucketThreshold, CalibratedOnlineScore, CalibratedOnlineScores, CalibratedThresholds,
    Classification, DiagnoseConfig, DiagnoseError, DiagnosisReport, MatrixScore, MatrixScores,
    MatrixThresholds, ScorePath, ScorePopulation, ScoreStatus,
};
pub use init::{InitError, MemoryEstimate, StageId, Tracegrams, TracegramsBuilder};
pub use snapshot::{
    CalibrationPopulation, CalibrationState, CompletionCounts, Consistency, DeltaError,
    DeltaSnapshot, Diagnostics, OnlineDeltaAvailability, SampleCounts, ScoreAvailability, Snapshot,
    StageMetadata,
};

#[doc(hidden)]
pub mod internal {
    //! Unstable benchmark seam, not public API: `benches/checkpoint.rs` must
    //! measure the exact crate-private hot-path primitive.
    pub use crate::recorder::increment_counter;
}
