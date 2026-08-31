//! Integration tests for the public lifecycle, validation, snapshot, and
//! diagnosis contracts.
//!
//! Numeric contracts are unit-tested beside `bucket`, `matrix`, and
//! `recorder`. Context layout assertions live in `clocked_recording`;
//! non-`Clone`/non-`Copy` constraints are compile-fail examples on `Ctx`.

mod calibration;
mod clocked_recording;
mod diagnosis;
mod diagnosis_properties;
mod init;
mod manual_recording;
mod properties;
mod snapshot_delta;
