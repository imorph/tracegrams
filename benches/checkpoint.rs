//! Checkpoint recording benchmarks.
//!
//! Until Stage 9 lands the full section-8 screening protocol, this file holds
//! the counter-primitive comparison behind plan risk #2: the saturating CAS
//! increment used by the recorder against the wrapping `fetch_add` floor,
//! uncontended and with 16 threads hammering one hot cell.

use std::sync::atomic::{AtomicU64, Ordering};

use tracegrams::internal::increment_counter;

fn main() {
    divan::main();
}

#[divan::bench(threads = [1, 16])]
fn wrapping_fetch_add() -> u64 {
    static COUNTER: AtomicU64 = AtomicU64::new(0);
    COUNTER.fetch_add(1, Ordering::Relaxed)
}

#[divan::bench(threads = [1, 16])]
fn saturating_cas() {
    static COUNTER: AtomicU64 = AtomicU64::new(0);
    static OVERFLOWS: AtomicU64 = AtomicU64::new(0);
    increment_counter(&COUNTER, &OVERFLOWS);
}
