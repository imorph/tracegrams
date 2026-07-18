//! Clocked-recording and context-invariant contract tests land in Stage 4.
//!
//! The Stage 4 tests must assert `size_of::<Ctx>() <= 24`,
//! `!needs_drop::<Ctx>()`, and `Send + Sync`. Non-implementation of `Clone` and
//! `Copy` must be covered by compile-fail API examples rather than runtime
//! assertions.

#[test]
#[ignore = "Ctx is not implemented until Stage 4"]
fn context_invariants_placeholder() {}
