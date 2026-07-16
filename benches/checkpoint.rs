//! Checkpoint recording benchmarks.

fn main() {
    divan::main();
}

#[divan::bench]
fn checkpoint() -> u64 {
    divan::black_box(0)
}
