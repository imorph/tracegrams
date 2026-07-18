//! Executable map of the ratified v0.0 demo contract.
//!
//! Stage 0 baseline at `dd0b3cb` (2026-07-17): `just all`, exact
//! `cargo +1.85.0 check --locked --all-features`, and
//! `cargo package --allow-dirty` all passed; there were no pre-existing
//! failures to record.
//!
//! Traceability to `API_DESIGN.md` section 9:
//! - runtime registration, validation, and the default table: `init`;
//! - clocked `Ctx`, rejection rules, and total duration range: `clocked_recording`;
//! - deterministic `ManualCtx`: `manual_recording`;
//! - one fixed shared atomic backend: `manual_recording`;
//! - calibration collection, readiness, and explicit freeze: `calibration`;
//! - clean-sentinel first marks: `manual_recording` and `calibration`;
//! - relaxed snapshots and pure delta: `snapshot_delta`;
//! - matrix-derived and calibrated-online score paths: `diagnosis`;
//! - pure diagnosis and stable display report: `diagnosis`;
//! - performance screening: `benches/checkpoint.rs` (Stage 9);
//! - manual axum integration and service A/B: private experiment repo (Stage 10).
//!
//! Numeric contracts shared by those areas are unit-tested beside `bucket`,
//! `matrix`, and `recorder`. Context layout/trait assertions remain explicit
//! placeholders until `Ctx` exists in Stage 4, so they cannot pass for the
//! wrong type.

mod calibration;
mod clocked_recording;
mod diagnosis;
mod init;
mod manual_recording;
mod snapshot_delta;
