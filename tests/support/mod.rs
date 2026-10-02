//! Shared configuration and independent numeric oracles for property tests.

use proptest::prelude::*;
use proptest::test_runner::{Config, FileFailurePersistence, RngAlgorithm, TestRng, TestRunner};
use std::fmt::Write;

// A 64-digit hex seed preserves the existing 32-byte ChaCha defaults exactly.
// Print it before running so a failure (including a panic) remains reproducible.
#[allow(clippy::print_stderr)]
pub(crate) fn runner(cases: u32, seed_byte: u8, regression: &'static str) -> TestRunner {
    let cases = std::env::var("TRACEGRAMS_PROPTEST_CASES").map_or(cases, |value| {
        value
            .parse::<u32>()
            .expect("TRACEGRAMS_PROPTEST_CASES must be a positive u32")
    });
    assert!(cases > 0, "TRACEGRAMS_PROPTEST_CASES must be positive");
    let mut seed = [seed_byte; 32];
    if let Ok(value) = std::env::var("TRACEGRAMS_PROPTEST_SEED") {
        assert!(
            value.is_ascii() && value.len() == 64,
            "TRACEGRAMS_PROPTEST_SEED must be 64 hex digits"
        );
        for (byte, digits) in seed.iter_mut().zip(value.as_bytes().chunks_exact(2)) {
            *byte = u8::from_str_radix(std::str::from_utf8(digits).unwrap(), 16)
                .expect("TRACEGRAMS_PROPTEST_SEED must be 64 hex digits");
        }
    }
    let mut hex = String::new();
    for byte in seed {
        write!(hex, "{byte:02x}").unwrap();
    }
    eprintln!("{regression}: TRACEGRAMS_PROPTEST_CASES={cases} TRACEGRAMS_PROPTEST_SEED={hex}");
    TestRunner::new_with_rng(
        Config {
            cases,
            failure_persistence: Some(Box::new(FileFailurePersistence::Direct(regression))),
            // Bound reduction of an already-found failure, not the case budget.
            // Keep the failure and its seed even if shrinking is incomplete.
            max_shrink_time: 5_000,
            ..Config::default()
        },
        TestRng::from_seed(RngAlgorithm::ChaCha, &seed),
    )
}

// Exact binary rational by scaling, independent of the production bit decoder.
pub(crate) fn rank(samples: u64, quantile: f64) -> u128 {
    let mut numerator = quantile;
    let mut power = 0_u32;
    while numerator.fract() != 0.0 {
        numerator *= 2.0;
        power += 1;
    }
    #[allow(clippy::cast_possible_truncation, clippy::cast_sign_loss)]
    let product = u128::from(samples) * u128::from(numerator as u64);
    if power >= u128::BITS - product.leading_zeros() {
        1
    } else {
        product
            .div_ceil(1_u128 << power)
            .clamp(1, u128::from(samples))
    }
}

pub(crate) fn quantiles() -> impl Strategy<Value = f64> {
    prop_oneof![
        4 => prop::sample::select(vec![0.5, 0.9, 0.99, 1.0]),
        2 => (2_u32..=32).prop_flat_map(|n| (1..n, Just(n))).prop_flat_map(|(k, n)| {
            let q = f64::from(k) / f64::from(n);
            prop::sample::select(vec![f64::from_bits(q.to_bits() - 1), q, f64::from_bits(q.to_bits() + 1)])
        }),
    ]
}
