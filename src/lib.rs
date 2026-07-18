//! Aggregate tail-latency propagation tracking without storing traces.

// These foundations are wired into the runtime in Stages 2 and 3.
#[allow(dead_code)]
mod bucket;
mod context;
mod diagnose;
#[allow(dead_code)]
mod matrix;
#[allow(dead_code)]
mod recorder;
