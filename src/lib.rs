//! Aggregate tail-latency propagation tracking without storing traces.

// These foundations are wired into the runtime in Stages 2 and 3.
#[allow(dead_code)]
mod bucket;
mod context;
mod diagnose;
mod init;
#[allow(dead_code)]
mod matrix;
mod recorder;
mod snapshot;

pub use context::{Ctx, ManualCtx, Outcome};
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
