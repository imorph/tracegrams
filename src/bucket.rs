//! Pinned latency buckets and calibration quantile arithmetic.

pub(crate) const BUCKETS: usize = 64;
pub(crate) const CALIBRATION_BUCKETS: usize = 250;

const DEFAULT_BOUNDS: [u64; BUCKETS - 1] = [
    100,
    135,
    181,
    244,
    328,
    442,
    595,
    800,
    1_077,
    1_450,
    1_951,
    2_626,
    3_535,
    4_758,
    6_404,
    8_620,
    11_602,
    15_615,
    21_017,
    28_289,
    38_075,
    51_248,
    68_978,
    92_841,
    124_961,
    168_192,
    226_380,
    304_699,
    410_113,
    551_995,
    742_964,
    1_000_000,
    1_345_960,
    1_811_609,
    2_438_354,
    3_281_928,
    4_417_345,
    5_945_571,
    8_002_502,
    10_771_051,
    14_497_407,
    19_512_934,
    26_263_635,
    35_349_811,
    47_579_443,
    64_040_043,
    86_195_357,
    116_015_530,
    156_152_301,
    210_174_801,
    282_886_943,
    380_754_602,
    512_480_588,
    689_778_538,
    928_414_545,
    1_249_609_141,
    1_681_924_325,
    2_263_803_410,
    3_046_989_571,
    4_101_127_071,
    5_519_954_321,
    7_429_639_508,
    10_000_000_000,
];

// Each finite ordinary interval is split into four geometric calibration
// buckets, so a calibration bucket can be located from its ordinary bucket.
const CALIBRATION_BOUNDS: [u64; CALIBRATION_BUCKETS - 1] = [
    100,
    108,
    116,
    125,
    135,
    145,
    156,
    168,
    181,
    195,
    210,
    226,
    244,
    263,
    283,
    305,
    328,
    353,
    381,
    410,
    442,
    476,
    513,
    552,
    595,
    641,
    690,
    743,
    800,
    862,
    928,
    1_000,
    1_077,
    1_160,
    1_250,
    1_346,
    1_450,
    1_562,
    1_682,
    1_811,
    1_951,
    2_101,
    2_263,
    2_438,
    2_626,
    2_829,
    3_047,
    3_282,
    3_535,
    3_808,
    4_101,
    4_417,
    4_758,
    5_125,
    5_520,
    5_946,
    6_404,
    6_898,
    7_430,
    8_003,
    8_620,
    9_285,
    10_000,
    10_772,
    11_602,
    12_496,
    13_460,
    14_497,
    15_615,
    16_819,
    18_116,
    19_513,
    21_017,
    22_638,
    24_383,
    26_264,
    28_289,
    30_470,
    32_819,
    35_350,
    38_075,
    41_011,
    44_173,
    47_579,
    51_248,
    55_200,
    59_456,
    64_040,
    68_978,
    74_296,
    80_025,
    86_195,
    92_841,
    100_000,
    107_710,
    116_015,
    124_961,
    134_596,
    144_974,
    156_152,
    168_192,
    181_161,
    195_129,
    210_174,
    226_380,
    243_835,
    262_636,
    282_887,
    304_699,
    328_193,
    353_498,
    380_755,
    410_113,
    441_735,
    475_794,
    512_480,
    551_995,
    594_557,
    640_400,
    689_778,
    742_964,
    800_250,
    861_954,
    928_415,
    1_000_000,
    1_077_105,
    1_160_155,
    1_249_609,
    1_345_960,
    1_449_740,
    1_561_523,
    1_681_924,
    1_811_609,
    1_951_293,
    2_101_748,
    2_263_803,
    2_438_354,
    2_626_363,
    2_828_869,
    3_046_990,
    3_281_928,
    3_534_981,
    3_807_546,
    4_101_127,
    4_417_345,
    4_757_945,
    5_124_806,
    5_519_955,
    5_945_571,
    6_404_004,
    6_897_785,
    7_429_639,
    8_002_502,
    8_619_536,
    9_284_145,
    10_000_000,
    10_771_051,
    11_601_553,
    12_496_092,
    13_459_604,
    14_497_407,
    15_615_230,
    16_819_243,
    18_116_092,
    19_512_934,
    21_017_480,
    22_638_034,
    24_383_541,
    26_263_635,
    28_288_694,
    30_469_896,
    32_819_279,
    35_349_811,
    38_075_460,
    41_011_271,
    44_173_447,
    47_579_443,
    51_248_059,
    55_199_543,
    59_455_707,
    64_040_043,
    68_977_854,
    74_296_395,
    80_025_023,
    86_195_357,
    92_841_455,
    100_000_000,
    107_710_506,
    116_015_530,
    124_960_914,
    134_596_032,
    144_974_067,
    156_152_301,
    168_192_433,
    181_160_920,
    195_129_342,
    210_174_801,
    226_380_341,
    243_835_410,
    262_636_352,
    282_886_943,
    304_698_957,
    328_192_787,
    353_498_110,
    380_754_602,
    410_112_707,
    441_734_470,
    475_794_432,
    512_480_588,
    551_995_432,
    594_557_071,
    640_400_427,
    689_778_538,
    742_963_951,
    800_250_228,
    861_953_567,
    928_414_545,
    1_000_000_000,
    1_077_105_056,
    1_160_155_302,
    1_249_609_141,
    1_345_960_324,
    1_449_740_670,
    1_561_523_006,
    1_681_924_325,
    1_811_609_194,
    1_951_293_423,
    2_101_748_012,
    2_263_803_410,
    2_438_354_099,
    2_626_363_528,
    2_828_869_435,
    3_046_989_571,
    3_281_927_873,
    3_534_981_105,
    3_807_546_022,
    4_101_127_071,
    4_417_344_703,
    4_757_944_314,
    5_124_805_877,
    5_519_954_321,
    5_945_570_708,
    6_404_004_271,
    6_897_785_380,
    7_429_639_508,
    8_002_502_278,
    8_619_535_665,
    9_284_145_445,
    10_000_000_000,
];

pub(crate) const fn default_bounds() -> &'static [u64; BUCKETS - 1] {
    &DEFAULT_BOUNDS
}

pub(crate) const fn calibration_bounds() -> &'static [u64; CALIBRATION_BUCKETS - 1] {
    &CALIBRATION_BOUNDS
}

pub(crate) fn bucketize(value: u64, bounds: &[u64]) -> usize {
    if value < bounds[0] {
        return 0;
    }
    bounds.partition_point(|&bound| value >= bound)
}

pub(crate) fn default_bucketize(value: u64) -> usize {
    bucketize(value, &DEFAULT_BOUNDS)
}

#[inline]
pub(crate) fn calibration_bucketize(value: u64, default_bucket: usize) -> usize {
    if default_bucket == 0 {
        return 0;
    }
    if default_bucket == BUCKETS - 1 {
        return CALIBRATION_BUCKETS - 1;
    }

    // Search only the four calibration buckets nested inside the already
    // known ordinary bucket.
    let first = 1 + (default_bucket - 1) * 4;
    first + CALIBRATION_BOUNDS[first..first + 3].partition_point(|&bound| value >= bound)
}

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub(crate) struct RankSelection {
    pub(crate) bucket: usize,
    pub(crate) rank: u128,
    pub(crate) samples: u128,
}

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub(crate) enum TerminalStatus {
    Lower,
    Finite,
    Upper,
}

#[derive(Clone, Copy, Debug, PartialEq)]
pub(crate) struct QuantileEstimate {
    pub(crate) selection: RankSelection,
    pub(crate) lower_bound: Option<u64>,
    pub(crate) upper_bound: Option<u64>,
    pub(crate) value: Option<u64>,
    pub(crate) max_relative_error: Option<f64>,
    pub(crate) terminal: TerminalStatus,
}

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub(crate) enum QuantileError {
    InvalidQuantile,
    CounterTotalOverflow,
    ShapeMismatch,
}

pub(crate) fn nearest_rank(
    counts: &[u64],
    quantile: f64,
) -> Result<Option<RankSelection>, QuantileError> {
    if !quantile.is_finite() || quantile <= 0.0 || quantile > 1.0 {
        return Err(QuantileError::InvalidQuantile);
    }

    let samples = counts.iter().try_fold(0_u64, |total, count| {
        total
            .checked_add(*count)
            .ok_or(QuantileError::CounterTotalOverflow)
    })?;
    if samples == 0 {
        return Ok(None);
    }

    let rank = quantile_rank(samples, quantile);
    let mut cumulative = 0_u128;
    for (bucket, count) in counts.iter().enumerate() {
        cumulative += u128::from(*count);
        if cumulative >= rank {
            return Ok(Some(RankSelection {
                bucket,
                rank,
                samples: u128::from(samples),
            }));
        }
    }

    unreachable!("the checked total must be reached by its source counts");
}

// Computes `ceil(samples * quantile)` from the exact binary value of the
// `f64`, avoiding float rounding at rank boundaries.
fn quantile_rank(samples: u64, quantile: f64) -> u128 {
    let bits = quantile.to_bits();
    let exponent = ((bits >> 52) & 0x7ff) as u16;
    let mantissa = bits & ((1_u64 << 52) - 1);
    let (significand, denominator_shift) = if exponent == 0 {
        (mantissa, 1_074_u32)
    } else {
        ((1_u64 << 52) | mantissa, 1_075_u32 - u32::from(exponent))
    };
    let product = u128::from(samples) * u128::from(significand);

    if denominator_shift >= u128::BITS {
        return 1;
    }

    let denominator = 1_u128 << denominator_shift;
    let quotient = product / denominator;
    let rank = quotient + u128::from(product % denominator != 0);
    rank.max(1).min(u128::from(samples))
}

pub(crate) fn estimate_quantile(
    counts: &[u64],
    bounds: &[u64],
    quantile: f64,
) -> Result<Option<QuantileEstimate>, QuantileError> {
    if bounds.is_empty() || counts.len() != bounds.len() + 1 {
        return Err(QuantileError::ShapeMismatch);
    }

    let Some(selection) = nearest_rank(counts, quantile)? else {
        return Ok(None);
    };
    let bucket = selection.bucket;
    if bucket == 0 {
        return Ok(Some(QuantileEstimate {
            selection,
            lower_bound: None,
            upper_bound: Some(bounds[0]),
            value: None,
            max_relative_error: None,
            terminal: TerminalStatus::Lower,
        }));
    }
    if bucket == bounds.len() {
        return Ok(Some(QuantileEstimate {
            selection,
            lower_bound: bounds.last().copied(),
            upper_bound: None,
            value: None,
            max_relative_error: None,
            terminal: TerminalStatus::Upper,
        }));
    }

    let lower = bounds[bucket - 1];
    let upper = bounds[bucket];
    Ok(Some(QuantileEstimate {
        selection,
        lower_bound: Some(lower),
        upper_bound: Some(upper),
        value: finite_representative(lower, upper),
        max_relative_error: relative_error(lower, upper),
        terminal: TerminalStatus::Finite,
    }))
}

// The harmonic mean equalizes worst-case relative error at both bucket edges.
pub(crate) fn finite_representative(lower: u64, upper: u64) -> Option<u64> {
    if lower >= upper {
        return None;
    }

    let numerator = u128::from(lower)
        .checked_mul(u128::from(upper))?
        .checked_mul(2)?;
    let denominator = u128::from(lower) + u128::from(upper);
    let quotient = numerator / denominator;
    let remainder = numerator % denominator;
    let rounded = quotient + u128::from(remainder >= denominator.div_ceil(2));
    u64::try_from(rounded).ok()
}

#[allow(clippy::cast_precision_loss)]
fn relative_error(lower: u64, upper: u64) -> Option<f64> {
    (lower < upper).then(|| (upper - lower) as f64 / (u128::from(upper) + u128::from(lower)) as f64)
}

#[cfg(test)]
mod tests {
    use super::*;
    use proptest::prelude::*;
    use proptest::test_runner::{Config, RngAlgorithm, TestRng, TestRunner};

    const EXPECTED_DEFAULT_BOUNDS: [u64; BUCKETS - 1] =
        include!("../tests/fixtures/default_bounds.rs");
    const EXPECTED_CALIBRATION_BOUNDS: [u64; CALIBRATION_BUCKETS - 1] =
        include!("../tests/fixtures/calibration_bounds.rs");

    #[allow(clippy::cast_precision_loss)]
    fn quantile_strategy() -> impl Strategy<Value = f64> {
        prop_oneof![
            8 => (1_u64..=1.0_f64.to_bits()).prop_map(f64::from_bits),
            2 => (1_u64..=CALIBRATION_BUCKETS as u64).prop_flat_map(|divisor| {
                let reciprocal = 1.0 / divisor as f64;
                let bits = reciprocal.to_bits();
                prop::sample::select(
                    [bits - 1, bits, bits + 1]
                        .into_iter()
                        .map(f64::from_bits)
                        .filter(|quantile| *quantile <= 1.0)
                        .collect::<Vec<_>>(),
                )
            }),
            1 => Just(1.0),
            1 => prop::sample::select(vec![f64::from_bits(1), f64::MIN_POSITIVE]),
        ]
    }

    fn counts_strategy() -> impl Strategy<Value = Vec<u64>> {
        prop_oneof![
            8 => prop::collection::vec(0_u64..=1_000_000, 1..=CALIBRATION_BUCKETS),
            1 => (1_u64..=u64::MAX).prop_map(|samples| vec![samples]),
            1 => (1_usize..=CALIBRATION_BUCKETS, 1_u64..=u64::MAX)
                .prop_map(|(len, samples)| {
                    let mut counts = vec![0; len];
                    counts[0] = samples;
                    counts
                }),
            1 => (1_usize..=CALIBRATION_BUCKETS, 1_u64..=u64::MAX)
                .prop_map(|(len, samples)| {
                    let mut counts = vec![0; len];
                    counts[len - 1] = samples;
                    counts
                }),
            1 => (1_u64..=u64::MAX / 2).prop_map(|count| vec![count, 0, count]),
        ]
    }

    // Derive the exact binary rational without decoding the IEEE-754 fields,
    // then apply `ceil(samples * numerator / 2^denominator_power)`.
    fn rational_rank(samples: u64, quantile: f64) -> u128 {
        let mut numerator = quantile;
        let mut denominator_power = 0_u32;
        while numerator.fract() != 0.0 {
            numerator *= 2.0;
            denominator_power += 1;
        }

        #[allow(clippy::cast_possible_truncation, clippy::cast_sign_loss)]
        let numerator = numerator as u64;
        let product = u128::from(samples) * u128::from(numerator);
        let rank = if denominator_power >= u128::BITS - product.leading_zeros() {
            1
        } else {
            let quotient = product >> denominator_power;
            let remainder_mask = (1_u128 << denominator_power) - 1;
            quotient + u128::from(product & remainder_mask != 0)
        };
        rank.clamp(1, u128::from(samples))
    }

    fn assert_rank_case(counts: &[u64], quantile: f64) -> Result<RankSelection, TestCaseError> {
        let samples = counts.iter().copied().sum::<u64>();
        let selection = nearest_rank(counts, quantile).unwrap().unwrap();
        let production_rank = quantile_rank(samples, quantile);

        prop_assert!((1..=u128::from(samples)).contains(&selection.rank));
        prop_assert_eq!(selection.rank, rational_rank(samples, quantile));
        prop_assert_eq!(selection.rank, production_rank);

        let before = counts[..selection.bucket]
            .iter()
            .map(|count| u128::from(*count))
            .sum::<u128>();
        let through = before + u128::from(counts[selection.bucket]);
        prop_assert!(before < selection.rank);
        prop_assert!(through >= selection.rank);
        Ok(selection)
    }

    #[test]
    fn nearest_rank_boundary_corpus() {
        let predecessor_of_one = f64::from_bits(1.0_f64.to_bits() - 1);
        let cases = [
            (vec![1], 0.5),
            (vec![u64::MAX, 0], 0.5),
            (vec![0, u64::MAX], 0.5),
            (vec![1, 2, 3], predecessor_of_one),
            (vec![1, 2, 3], 1.0),
            (vec![1, 1, 1, 1], 0.25),
            (vec![1, 1, 1, 1], 0.5),
            (vec![1, 1, 1, 1], 0.75),
            (vec![1; 10], 0.3),
            (vec![1, 2, 3], f64::from_bits(1)),
            (vec![1, 2, 3], f64::MIN_POSITIVE),
        ];

        for (counts, quantile) in cases {
            assert_rank_case(&counts, quantile).unwrap();
        }
    }

    #[test]
    fn nearest_rank_matches_exact_rational_oracle_and_is_monotonic() {
        let config = Config {
            cases: 256,
            failure_persistence: Some(Box::new(
                proptest::test_runner::FileFailurePersistence::Direct(
                    "proptest-regressions/bucket.txt",
                ),
            )),
            ..Config::default()
        };
        let rng = TestRng::from_seed(RngAlgorithm::ChaCha, &[0x5a; 32]);
        let mut runner = TestRunner::new_with_rng(config, rng);
        let strategy = (counts_strategy(), quantile_strategy(), quantile_strategy());

        runner
            .run(&strategy, |(counts, first_quantile, second_quantile)| {
                let samples = counts.iter().copied().sum::<u64>();
                prop_assume!(samples > 0);
                let lower_quantile = first_quantile.min(second_quantile);
                let upper_quantile = first_quantile.max(second_quantile);
                let lower = assert_rank_case(&counts, lower_quantile)?;
                let upper = assert_rank_case(&counts, upper_quantile)?;

                prop_assert!(lower.rank <= upper.rank);
                prop_assert!(lower.bucket <= upper.bucket);
                Ok(())
            })
            .unwrap();
    }

    #[test]
    fn default_table_is_byte_for_byte_compatible_with_the_poc() {
        assert_eq!(*default_bounds(), EXPECTED_DEFAULT_BOUNDS);
    }

    #[test]
    fn default_bucketization_is_total_and_bounds_are_upper_exclusive() {
        let bounds = default_bounds();

        assert_eq!(bucketize(0, bounds), 0);
        assert_eq!(bucketize(99, bounds), 0);
        assert_eq!(bucketize(100, bounds), 1);
        assert_eq!(bucketize(134, bounds), 1);
        assert_eq!(bucketize(135, bounds), 2);
        assert_eq!(bucketize(10_000_000_000 - 1, bounds), BUCKETS - 2);
        assert_eq!(bucketize(10_000_000_000, bounds), BUCKETS - 1);
        assert_eq!(bucketize(u64::MAX, bounds), BUCKETS - 1);
    }

    #[test]
    fn calibration_table_matches_the_pinned_fixture() {
        assert_eq!(*calibration_bounds(), EXPECTED_CALIBRATION_BOUNDS);
    }

    #[test]
    #[allow(
        clippy::cast_possible_truncation,
        clippy::cast_precision_loss,
        clippy::cast_sign_loss
    )]
    fn calibration_fixture_uses_round_half_up_geometric_subdivisions() {
        let mut generated = Vec::with_capacity(CALIBRATION_BUCKETS - 1);
        for interval in EXPECTED_DEFAULT_BOUNDS.windows(2) {
            let lower = interval[0];
            let upper = interval[1];
            generated.push(lower);
            for subdivision in 1..4 {
                let ratio = upper as f64 / lower as f64;
                let value = lower as f64 * ratio.powf(f64::from(subdivision) / 4.0);
                generated.push(value.round() as u64);
            }
        }
        generated.push(*EXPECTED_DEFAULT_BOUNDS.last().unwrap());

        assert_eq!(generated, EXPECTED_CALIBRATION_BOUNDS);
        assert!(generated.windows(2).all(|pair| pair[0] < pair[1]));
    }

    #[test]
    fn calibration_bucketization_has_lower_and_upper_terminals() {
        let bounds = calibration_bounds();

        assert_eq!(bucketize(0, bounds), 0);
        assert_eq!(bucketize(99, bounds), 0);
        assert_eq!(bucketize(100, bounds), 1);
        assert_eq!(bucketize(107, bounds), 1);
        assert_eq!(bucketize(108, bounds), 2);
        assert_eq!(
            bucketize(10_000_000_000 - 1, bounds),
            CALIBRATION_BUCKETS - 2
        );
        assert_eq!(bucketize(10_000_000_000, bounds), CALIBRATION_BUCKETS - 1);
        assert_eq!(bucketize(u64::MAX, bounds), CALIBRATION_BUCKETS - 1);
    }

    #[test]
    fn calibration_bucketization_from_default_bucket_matches_the_pinned_grid() {
        for value in calibration_bounds()
            .iter()
            .flat_map(|bound| [bound - 1, *bound, bound + 1])
            .chain([0, u64::MAX])
        {
            let default_bucket = bucketize(value, default_bounds());
            assert_eq!(
                calibration_bucketize(value, default_bucket),
                bucketize(value, calibration_bounds()),
                "value={value}"
            );
        }
    }

    #[test]
    fn nearest_rank_handles_empty_singleton_and_population_boundaries() {
        assert_eq!(nearest_rank(&[], 0.99), Ok(None));
        assert_eq!(
            nearest_rank(&[0, 1, 0], 0.99),
            Ok(Some(RankSelection {
                bucket: 1,
                rank: 1,
                samples: 1,
            }))
        );
        assert_eq!(nearest_rank(&[1, 1, 1, 1], 0.5).unwrap().unwrap().bucket, 1);
        assert_eq!(
            nearest_rank(&[1, 1, 1, 1], 0.500_000_1)
                .unwrap()
                .unwrap()
                .bucket,
            2
        );
        assert_eq!(nearest_rank(&[1, 1, 1, 1], 1.0).unwrap().unwrap().bucket, 3);
    }

    #[test]
    fn invalid_quantiles_are_rejected() {
        for quantile in [
            f64::NAN,
            f64::INFINITY,
            f64::NEG_INFINITY,
            -0.5,
            0.0,
            1.000_000_1,
        ] {
            assert_eq!(
                nearest_rank(&[1], quantile),
                Err(QuantileError::InvalidQuantile)
            );
            assert_eq!(
                estimate_quantile(&[1, 0], &[100], quantile),
                Err(QuantileError::InvalidQuantile)
            );
        }
    }

    #[test]
    fn counter_total_overflow_is_reported() {
        assert_eq!(
            nearest_rank(&[u64::MAX, 1], 0.99),
            Err(QuantileError::CounterTotalOverflow)
        );
        assert_eq!(
            estimate_quantile(&[u64::MAX, u64::MAX], &[100], 0.99),
            Err(QuantileError::CounterTotalOverflow)
        );
    }

    #[test]
    fn estimate_rejects_mismatched_shapes() {
        assert_eq!(
            estimate_quantile(&[], &[], 0.99),
            Err(QuantileError::ShapeMismatch)
        );
        assert_eq!(
            estimate_quantile(&[1], &[], 0.99),
            Err(QuantileError::ShapeMismatch)
        );
        assert_eq!(
            estimate_quantile(&[1], &[100], 0.99),
            Err(QuantileError::ShapeMismatch)
        );
        assert_eq!(
            estimate_quantile(&[1, 2, 3], &[100], 0.99),
            Err(QuantileError::ShapeMismatch)
        );
    }

    #[test]
    fn selected_terminal_bucket_has_no_finite_estimate() {
        let bounds = calibration_bounds();
        let mut counts = [0; CALIBRATION_BUCKETS];

        counts[0] = 1;
        let lower = estimate_quantile(&counts, bounds, 0.99).unwrap().unwrap();
        assert_eq!(lower.terminal, TerminalStatus::Lower);
        assert_eq!(lower.lower_bound, None);
        assert_eq!(lower.upper_bound, Some(100));
        assert_eq!(lower.value, None);
        assert_eq!(lower.max_relative_error, None);

        counts[0] = 0;
        counts[CALIBRATION_BUCKETS - 1] = 1;
        let upper = estimate_quantile(&counts, bounds, 0.99).unwrap().unwrap();
        assert_eq!(upper.terminal, TerminalStatus::Upper);
        assert_eq!(upper.lower_bound, Some(10_000_000_000));
        assert_eq!(upper.upper_bound, None);
        assert_eq!(upper.value, None);
        assert_eq!(upper.max_relative_error, None);
    }

    #[test]
    fn finite_estimate_uses_u128_minimax_representative() {
        assert_eq!(
            finite_representative(9_284_145_445, 10_000_000_000),
            Some(9_628_785_959)
        );

        let mut counts = [0; CALIBRATION_BUCKETS];
        counts[CALIBRATION_BUCKETS - 2] = 1;
        let estimate = estimate_quantile(&counts, calibration_bounds(), 0.99)
            .unwrap()
            .unwrap();
        assert_eq!(estimate.terminal, TerminalStatus::Finite);
        assert_eq!(estimate.lower_bound, Some(9_284_145_445));
        assert_eq!(estimate.upper_bound, Some(10_000_000_000));
        assert_eq!(estimate.value, Some(9_628_785_959));
        assert!(
            (estimate.max_relative_error.unwrap() - 0.037_121_404_059_188_27).abs() < f64::EPSILON
        );
    }
}
