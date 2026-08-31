//! Pure matrix-derived and calibrated-online tail-propagation diagnosis.

use std::error::Error;
use std::fmt;

use crate::bucket::{BUCKETS, RankSelection, nearest_rank};
use crate::recorder::{ONLINE_COUNTERS_PER_STAGE, ONLINE_FIRST_COUNTERS};
use crate::{
    Consistency, DeltaSnapshot, OnlineDeltaAvailability, Snapshot, StageCalibrationReport, StageId,
};

/// The counter path from which a score was derived.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
#[non_exhaustive]
pub enum ScorePath {
    /// Approximate bucket thresholds and scores derived from transition matrices.
    MatrixDerived,
    /// Exact truth-table scores evaluated against frozen calibration thresholds.
    CalibratedOnline,
}

impl ScorePath {
    /// Returns whether this path has known compact-bucket threshold drift.
    pub const fn known_drift(self) -> bool {
        matches!(self, Self::MatrixDerived)
    }
}

/// The request population represented by a score path.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
#[non_exhaustive]
pub enum ScorePopulation {
    /// Only marks with an observed predecessor are represented.
    PredecessorOnly,
    /// All reached marks are represented, with first marks kept as clean sentinels.
    AllReachedMarks,
}

/// Availability of one numeric score.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
#[non_exhaustive]
pub enum ScoreStatus {
    /// The numerator, denominator, and ratio are available.
    Available,
    /// No predecessor-bearing samples were available for this score.
    NoPredecessorPopulation,
    /// The score's denominator is zero.
    ZeroDenominator,
    /// Relaxed inputs violated a probability invariant.
    InconsistentSnapshot,
    /// Threshold or ratio arithmetic exceeded its supported integer range.
    ArithmeticOverflow,
}

/// A nearest-rank threshold in the pinned ordinary bucket table.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub struct BucketThreshold {
    bucket: usize,
    rank: u128,
    samples: u128,
}

impl BucketThreshold {
    /// Returns the first bucket classified as tail.
    pub const fn bucket(self) -> usize {
        self.bucket
    }

    /// Returns the selected nearest rank.
    pub const fn rank(self) -> u128 {
        self.rank
    }

    /// Returns the predecessor-bearing population size used by the rank.
    pub const fn samples(self) -> u128 {
        self.samples
    }
}

/// Matrix-derived tail thresholds for one destination stage.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub struct MatrixThresholds {
    local: Option<BucketThreshold>,
    previous_cumulative: Option<BucketThreshold>,
    cumulative_after: Option<BucketThreshold>,
}

impl MatrixThresholds {
    /// Returns the local-latency threshold derived from the cause matrix.
    pub const fn local(self) -> Option<BucketThreshold> {
        self.local
    }

    /// Returns the previous-cumulative threshold derived from the cause matrix.
    pub const fn previous_cumulative(self) -> Option<BucketThreshold> {
        self.previous_cumulative
    }

    /// Returns the cumulative-after threshold derived from the incoming matrix.
    pub const fn cumulative_after(self) -> Option<BucketThreshold> {
        self.cumulative_after
    }
}

/// One numeric matrix-derived score and its integer rational terms.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct MatrixScore {
    numerator: u128,
    denominator: u128,
    value: Option<f64>,
    status: ScoreStatus,
}

impl MatrixScore {
    /// Returns the matrix-derived score path.
    pub const fn path(self) -> ScorePath {
        ScorePath::MatrixDerived
    }

    /// Returns this score's availability status.
    pub const fn status(self) -> ScoreStatus {
        self.status
    }

    /// Returns the integer numerator of this score's ratio.
    pub const fn numerator(self) -> u128 {
        self.numerator
    }

    /// Returns the integer denominator of this score's ratio.
    pub const fn denominator(self) -> u128 {
        self.denominator
    }

    /// Returns the ratio when the score is available.
    pub const fn value(self) -> Option<f64> {
        self.value
    }
}

/// Matrix-derived scores for one destination stage.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct MatrixScores {
    stage: StageId,
    consistency: Consistency,
    thresholds: MatrixThresholds,
    cause_samples: u128,
    incoming_samples: u128,
    local_tail_origin_clean: MatrixScore,
    local_tail_origin_tail: MatrixScore,
    local_tail_rate_given_prev_tail: MatrixScore,
    local_tail_rate_given_prev_not_tail: MatrixScore,
    amplification_lift: MatrixScore,
    carry_through: MatrixScore,
    tail_onset: MatrixScore,
}

impl MatrixScores {
    /// Returns the destination stage.
    pub const fn stage(self) -> StageId {
        self.stage
    }

    /// Returns the matrix-derived score path.
    pub const fn path(self) -> ScorePath {
        ScorePath::MatrixDerived
    }

    /// Returns the predecessor-only population represented by the matrices.
    pub const fn population(self) -> ScorePopulation {
        ScorePopulation::PredecessorOnly
    }

    /// Returns the relaxed consistency attached to the source snapshot.
    pub const fn consistency(self) -> Consistency {
        self.consistency
    }

    /// Returns nearest-rank bucket thresholds for this destination.
    pub const fn thresholds(self) -> MatrixThresholds {
        self.thresholds
    }

    /// Returns predecessor-bearing samples observed in the cause matrix.
    pub const fn cause_samples(self) -> u128 {
        self.cause_samples
    }

    /// Returns predecessor-bearing samples observed in the incoming matrix.
    pub const fn incoming_samples(self) -> u128 {
        self.incoming_samples
    }

    /// Returns `P(previous not tail | local tail)`.
    pub const fn local_tail_origin_clean(self) -> MatrixScore {
        self.local_tail_origin_clean
    }

    /// Returns `P(previous tail | local tail)`.
    pub const fn local_tail_origin_tail(self) -> MatrixScore {
        self.local_tail_origin_tail
    }

    /// Returns `P(local tail | previous tail)`.
    pub const fn local_tail_rate_given_prev_tail(self) -> MatrixScore {
        self.local_tail_rate_given_prev_tail
    }

    /// Returns `P(local tail | previous not tail)`.
    pub const fn local_tail_rate_given_prev_not_tail(self) -> MatrixScore {
        self.local_tail_rate_given_prev_not_tail
    }

    /// Returns the ratio between the previous-tail and previous-clean local-tail rates.
    pub const fn amplification_lift(self) -> MatrixScore {
        self.amplification_lift
    }

    /// Returns `P(local not tail | previous tail)`.
    pub const fn carry_through(self) -> MatrixScore {
        self.carry_through
    }

    /// Returns `P(previous not tail | cumulative after tail)`.
    pub const fn tail_onset(self) -> MatrixScore {
        self.tail_onset
    }

    /// Applies an explicit secondary classification policy to this matrix path.
    pub fn classification(&self, config: &DiagnoseConfig) -> Classification {
        classify_values(
            self.tail_onset.value,
            self.amplification_lift.value,
            self.carry_through.value,
            *config,
        )
    }
}

impl Snapshot {
    /// Purely derives predecessor-only approximate scores from matrix cells.
    pub fn matrix_scores(&self, stage: StageId) -> Option<MatrixScores> {
        derive_matrix_scores(
            stage,
            self.consistency(),
            self.tail_quantile(),
            self.cause_counts(stage)?,
            self.incoming_counts(stage),
        )
    }
}

impl DeltaSnapshot {
    /// Purely derives predecessor-only approximate scores for this window.
    pub fn matrix_scores(&self, stage: StageId) -> Option<MatrixScores> {
        derive_matrix_scores(
            stage,
            self.consistency(),
            self.tail_quantile(),
            self.cause_counts(stage)?,
            self.incoming_counts(stage),
        )
    }
}

fn derive_matrix_scores(
    stage: StageId,
    consistency: Consistency,
    tail_quantile: f64,
    cause: &[u64],
    incoming: Option<&[u64]>,
) -> Option<MatrixScores> {
    let cause_marginals = matrix_marginals(cause)?;
    let incoming_marginals = match incoming {
        Some(counts) => Some(matrix_marginals(counts)?),
        None => None,
    };
    let cause_samples = matrix_total(cause);
    let incoming_samples = incoming.map_or(0, matrix_total);

    let previous_cumulative = threshold(
        &cause_marginals.rows,
        cause_marginals.rows_overflowed,
        tail_quantile,
    );
    let local = threshold(
        &cause_marginals.columns,
        cause_marginals.columns_overflowed,
        tail_quantile,
    );
    let cumulative_after = incoming_marginals.map_or(Threshold::NoSamples, |marginals| {
        threshold(
            &marginals.columns,
            marginals.columns_overflowed,
            tail_quantile,
        )
    });
    let thresholds = MatrixThresholds {
        local: local.value(),
        previous_cumulative: previous_cumulative.value(),
        cumulative_after: cumulative_after.value(),
    };

    let cause_scores = cause_scores(cause, previous_cumulative, local);
    let tail_onset = incoming.map_or_else(
        || MatrixScore::unavailable(ScoreStatus::NoPredecessorPopulation),
        |counts| {
            tail_onset_score(
                counts,
                previous_cumulative,
                cumulative_after,
                incoming_samples,
            )
        },
    );

    Some(MatrixScores {
        stage,
        consistency,
        thresholds,
        cause_samples,
        incoming_samples,
        local_tail_origin_clean: cause_scores.local_tail_origin_clean,
        local_tail_origin_tail: cause_scores.local_tail_origin_tail,
        local_tail_rate_given_prev_tail: cause_scores.local_tail_rate_given_prev_tail,
        local_tail_rate_given_prev_not_tail: cause_scores.local_tail_rate_given_prev_not_tail,
        amplification_lift: cause_scores.amplification_lift,
        carry_through: cause_scores.carry_through,
        tail_onset,
    })
}

#[derive(Clone, Copy)]
struct MatrixMarginals {
    rows: [u64; BUCKETS],
    columns: [u64; BUCKETS],
    rows_overflowed: bool,
    columns_overflowed: bool,
}

fn matrix_marginals(counts: &[u64]) -> Option<MatrixMarginals> {
    if counts.len() != BUCKETS * BUCKETS {
        return None;
    }

    let mut rows = [0; BUCKETS];
    let mut columns = [0; BUCKETS];
    let mut rows_overflowed = false;
    let mut columns_overflowed = false;
    for (index, count) in counts.iter().copied().enumerate() {
        add_marginal(&mut rows, &mut rows_overflowed, index / BUCKETS, count);
        add_marginal(
            &mut columns,
            &mut columns_overflowed,
            index % BUCKETS,
            count,
        );
    }
    Some(MatrixMarginals {
        rows,
        columns,
        rows_overflowed,
        columns_overflowed,
    })
}

fn add_marginal(counts: &mut [u64; BUCKETS], overflowed: &mut bool, bucket: usize, count: u64) {
    if *overflowed {
        return;
    }
    let Some(sum) = counts[bucket].checked_add(count) else {
        *overflowed = true;
        return;
    };
    counts[bucket] = sum;
}

fn matrix_total(counts: &[u64]) -> u128 {
    counts.iter().map(|count| u128::from(*count)).sum()
}

#[derive(Clone, Copy)]
enum Threshold {
    Available(BucketThreshold),
    NoSamples,
    ArithmeticOverflow,
}

impl Threshold {
    const fn value(self) -> Option<BucketThreshold> {
        match self {
            Self::Available(threshold) => Some(threshold),
            Self::NoSamples | Self::ArithmeticOverflow => None,
        }
    }

    const fn bucket(self) -> Result<usize, ScoreStatus> {
        match self {
            Self::Available(threshold) => Ok(threshold.bucket),
            Self::NoSamples => Err(ScoreStatus::NoPredecessorPopulation),
            Self::ArithmeticOverflow => Err(ScoreStatus::ArithmeticOverflow),
        }
    }
}

fn threshold(counts: &[u64], overflowed: bool, quantile: f64) -> Threshold {
    if overflowed {
        return Threshold::ArithmeticOverflow;
    }
    match nearest_rank(counts, quantile) {
        Ok(Some(RankSelection {
            bucket,
            rank,
            samples,
        })) => Threshold::Available(BucketThreshold {
            bucket,
            rank,
            samples,
        }),
        Ok(None) => Threshold::NoSamples,
        Err(_) => Threshold::ArithmeticOverflow,
    }
}

#[derive(Clone, Copy)]
struct CauseScores {
    local_tail_origin_clean: MatrixScore,
    local_tail_origin_tail: MatrixScore,
    local_tail_rate_given_prev_tail: MatrixScore,
    local_tail_rate_given_prev_not_tail: MatrixScore,
    amplification_lift: MatrixScore,
    carry_through: MatrixScore,
}

fn cause_scores(
    counts: &[u64],
    previous_threshold: Threshold,
    local_threshold: Threshold,
) -> CauseScores {
    let unavailable = previous_threshold
        .bucket()
        .and_then(|previous| local_threshold.bucket().map(|local| (previous, local)));
    let Ok((previous_threshold, local_threshold)) = unavailable else {
        let status = unavailable.unwrap_err();
        let score = MatrixScore::unavailable(status);
        return CauseScores {
            local_tail_origin_clean: score,
            local_tail_origin_tail: score,
            local_tail_rate_given_prev_tail: score,
            local_tail_rate_given_prev_not_tail: score,
            amplification_lift: score,
            carry_through: score,
        };
    };

    let mut local_tail_from_prev_not_tail = 0_u128;
    let mut local_tail_from_prev_tail = 0_u128;
    let mut prev_not_tail_total = 0_u128;
    let mut prev_tail_total = 0_u128;
    let mut carry_through = 0_u128;

    for (index, count) in counts.iter().copied().enumerate() {
        let previous_tail = index / BUCKETS >= previous_threshold;
        let local_tail = index % BUCKETS >= local_threshold;
        let count = u128::from(count);
        match (previous_tail, local_tail) {
            (false, false) => prev_not_tail_total += count,
            (false, true) => {
                prev_not_tail_total += count;
                local_tail_from_prev_not_tail += count;
            }
            (true, false) => {
                prev_tail_total += count;
                carry_through += count;
            }
            (true, true) => {
                prev_tail_total += count;
                local_tail_from_prev_tail += count;
            }
        }
    }

    let local_tail_total = local_tail_from_prev_not_tail + local_tail_from_prev_tail;
    CauseScores {
        local_tail_origin_clean: MatrixScore::probability(
            local_tail_from_prev_not_tail,
            local_tail_total,
        ),
        local_tail_origin_tail: MatrixScore::probability(
            local_tail_from_prev_tail,
            local_tail_total,
        ),
        local_tail_rate_given_prev_tail: MatrixScore::probability(
            local_tail_from_prev_tail,
            prev_tail_total,
        ),
        local_tail_rate_given_prev_not_tail: MatrixScore::probability(
            local_tail_from_prev_not_tail,
            prev_not_tail_total,
        ),
        amplification_lift: MatrixScore::lift(
            local_tail_from_prev_tail,
            prev_tail_total,
            local_tail_from_prev_not_tail,
            prev_not_tail_total,
        ),
        carry_through: MatrixScore::probability(carry_through, prev_tail_total),
    }
}

fn tail_onset_score(
    counts: &[u64],
    previous_threshold: Threshold,
    cumulative_after_threshold: Threshold,
    incoming_samples: u128,
) -> MatrixScore {
    if incoming_samples == 0 {
        return MatrixScore::unavailable(ScoreStatus::NoPredecessorPopulation);
    }
    let thresholds = previous_threshold.bucket().and_then(|previous| {
        cumulative_after_threshold
            .bucket()
            .map(|after| (previous, after))
    });
    let Ok((previous_threshold, cumulative_after_threshold)) = thresholds else {
        return MatrixScore::unavailable(thresholds.unwrap_err());
    };

    let mut numerator = 0_u128;
    let mut denominator = 0_u128;
    for (index, count) in counts.iter().copied().enumerate() {
        let previous = index / BUCKETS;
        let cumulative_after = index % BUCKETS;
        if cumulative_after >= cumulative_after_threshold {
            let count = u128::from(count);
            denominator += count;
            if previous < previous_threshold {
                numerator += count;
            }
        }
    }
    MatrixScore::probability(numerator, denominator)
}

impl MatrixScore {
    const fn unavailable(status: ScoreStatus) -> Self {
        Self {
            numerator: 0,
            denominator: 0,
            value: None,
            status,
        }
    }

    fn probability(numerator: u128, denominator: u128) -> Self {
        let score = score_probability(numerator, denominator);
        Self {
            numerator: score.numerator,
            denominator: score.denominator,
            value: score.value,
            status: score.status,
        }
    }

    fn lift(
        tail_local_tail: u128,
        previous_tail: u128,
        clean_local_tail: u128,
        previous_clean: u128,
    ) -> Self {
        let score = score_lift(
            tail_local_tail,
            previous_tail,
            clean_local_tail,
            previous_clean,
        );
        Self {
            numerator: score.numerator,
            denominator: score.denominator,
            value: score.value,
            status: score.status,
        }
    }
}

#[derive(Clone, Copy)]
struct ScoreParts {
    numerator: u128,
    denominator: u128,
    value: Option<f64>,
    status: ScoreStatus,
}

fn score_probability(numerator: u128, denominator: u128) -> ScoreParts {
    if denominator == 0 {
        return ScoreParts {
            numerator,
            denominator,
            value: None,
            status: ScoreStatus::ZeroDenominator,
        };
    }
    if numerator > denominator {
        return ScoreParts {
            numerator,
            denominator,
            value: None,
            status: ScoreStatus::InconsistentSnapshot,
        };
    }
    score_ratio(numerator, denominator)
}

// Cross-multiplies the two rates to keep the ratio in integer terms until
// the final `f64` conversion.
fn score_lift(
    tail_local_tail: u128,
    previous_tail: u128,
    clean_local_tail: u128,
    previous_clean: u128,
) -> ScoreParts {
    let Some(numerator) = tail_local_tail.checked_mul(previous_clean) else {
        return unavailable_parts(ScoreStatus::ArithmeticOverflow);
    };
    let Some(denominator) = previous_tail.checked_mul(clean_local_tail) else {
        return unavailable_parts(ScoreStatus::ArithmeticOverflow);
    };
    if denominator == 0 {
        return ScoreParts {
            numerator,
            denominator,
            value: None,
            status: ScoreStatus::ZeroDenominator,
        };
    }
    score_ratio(numerator, denominator)
}

const fn unavailable_parts(status: ScoreStatus) -> ScoreParts {
    ScoreParts {
        numerator: 0,
        denominator: 0,
        value: None,
        status,
    }
}

#[allow(clippy::cast_precision_loss)]
fn score_ratio(numerator: u128, denominator: u128) -> ScoreParts {
    ScoreParts {
        numerator,
        denominator,
        value: Some(numerator as f64 / denominator as f64),
        status: ScoreStatus::Available,
    }
}

/// Frozen nanosecond thresholds used by one calibrated-online stage report.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub struct CalibratedThresholds {
    local: u64,
    previous_cumulative: Option<u64>,
    cumulative_after: u64,
}

impl CalibratedThresholds {
    /// Returns the frozen local-tail predicate threshold.
    pub const fn local_ns(self) -> u64 {
        self.local
    }

    /// Returns the frozen previous-cumulative predicate threshold when calibrated.
    pub const fn previous_cumulative_ns(self) -> Option<u64> {
        self.previous_cumulative
    }

    /// Returns the frozen cumulative-after-tail predicate threshold.
    pub const fn cumulative_after_ns(self) -> u64 {
        self.cumulative_after
    }
}

/// One calibrated-online score and its integer rational terms.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct CalibratedOnlineScore {
    numerator: u128,
    denominator: u128,
    value: Option<f64>,
    status: ScoreStatus,
}

impl CalibratedOnlineScore {
    /// Returns the calibrated-online score path.
    pub const fn path(self) -> ScorePath {
        ScorePath::CalibratedOnline
    }

    /// Returns this score's availability status.
    pub const fn status(self) -> ScoreStatus {
        self.status
    }

    /// Returns the integer numerator of this score's ratio.
    pub const fn numerator(self) -> u128 {
        self.numerator
    }

    /// Returns the integer denominator of this score's ratio.
    pub const fn denominator(self) -> u128 {
        self.denominator
    }

    /// Returns the ratio when the score is available.
    pub const fn value(self) -> Option<f64> {
        self.value
    }

    const fn from_parts(parts: ScoreParts) -> Self {
        Self {
            numerator: parts.numerator,
            denominator: parts.denominator,
            value: parts.value,
            status: parts.status,
        }
    }

    fn probability(numerator: u128, denominator: u128) -> Self {
        Self::from_parts(score_probability(numerator, denominator))
    }

    fn lift(
        tail_local_tail: u128,
        previous_tail: u128,
        clean_local_tail: u128,
        previous_clean: u128,
    ) -> Self {
        Self::from_parts(score_lift(
            tail_local_tail,
            previous_tail,
            clean_local_tail,
            previous_clean,
        ))
    }

    const fn no_predecessor() -> Self {
        Self::from_parts(unavailable_parts(ScoreStatus::NoPredecessorPopulation))
    }
}

/// Exact frozen-threshold scores for one reached-stage population.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct CalibratedOnlineScores {
    stage: StageId,
    consistency: Consistency,
    epoch: u16,
    thresholds: CalibratedThresholds,
    samples: u128,
    first_samples: u128,
    predecessor_samples: u128,
    local_tail_origin_clean: CalibratedOnlineScore,
    local_tail_origin_tail: CalibratedOnlineScore,
    local_tail_rate_given_prev_tail: CalibratedOnlineScore,
    local_tail_rate_given_prev_not_tail: CalibratedOnlineScore,
    amplification_lift: CalibratedOnlineScore,
    carry_through: CalibratedOnlineScore,
    tail_onset: CalibratedOnlineScore,
}

impl CalibratedOnlineScores {
    /// Returns the destination stage.
    pub const fn stage(self) -> StageId {
        self.stage
    }

    /// Returns the exact calibrated-online score path.
    pub const fn path(self) -> ScorePath {
        ScorePath::CalibratedOnline
    }

    /// Returns the all-reached population represented by the truth table.
    pub const fn population(self) -> ScorePopulation {
        ScorePopulation::AllReachedMarks
    }

    /// Returns the relaxed consistency attached to the source snapshot.
    pub const fn consistency(self) -> Consistency {
        self.consistency
    }

    /// Returns the frozen online epoch.
    pub const fn epoch(self) -> u16 {
        self.epoch
    }

    /// Returns the frozen thresholds.
    pub const fn thresholds(self) -> CalibratedThresholds {
        self.thresholds
    }

    /// Returns all classified first and predecessor-bearing marks.
    pub const fn samples(self) -> u128 {
        self.samples
    }

    /// Returns classified first marks.
    pub const fn first_samples(self) -> u128 {
        self.first_samples
    }

    /// Returns classified predecessor-bearing marks.
    pub const fn predecessor_samples(self) -> u128 {
        self.predecessor_samples
    }

    /// Returns `P(clean | local tail)` over first and predecessor marks.
    pub const fn local_tail_origin_clean(self) -> CalibratedOnlineScore {
        self.local_tail_origin_clean
    }

    /// Returns `P(previous tail | local tail)`; first marks count only in the denominator.
    pub const fn local_tail_origin_tail(self) -> CalibratedOnlineScore {
        self.local_tail_origin_tail
    }

    /// Returns `P(local tail | previous tail)`.
    pub const fn local_tail_rate_given_prev_tail(self) -> CalibratedOnlineScore {
        self.local_tail_rate_given_prev_tail
    }

    /// Returns `P(local tail | previous not tail)`.
    pub const fn local_tail_rate_given_prev_not_tail(self) -> CalibratedOnlineScore {
        self.local_tail_rate_given_prev_not_tail
    }

    /// Returns the ratio between previous-tail and previous-clean local-tail rates.
    pub const fn amplification_lift(self) -> CalibratedOnlineScore {
        self.amplification_lift
    }

    /// Returns `P(local not tail | previous tail)`.
    pub const fn carry_through(self) -> CalibratedOnlineScore {
        self.carry_through
    }

    /// Returns `P(clean | cumulative after tail)` over first and predecessor marks.
    pub const fn tail_onset(self) -> CalibratedOnlineScore {
        self.tail_onset
    }

    /// Applies an explicit secondary classification policy to the exact scores.
    pub fn classification(&self, config: &DiagnoseConfig) -> Classification {
        classify_values(
            self.tail_onset.value,
            self.amplification_lift.value,
            self.carry_through.value,
            *config,
        )
    }
}

#[derive(Clone, Copy, Debug, Default, Eq, PartialEq)]
struct OnlineTotals {
    first_samples: u128,
    predecessor_samples: u128,
    local_tail_total: u128,
    local_tail_clean: u128,
    local_tail_previous_clean: u128,
    local_tail_previous_tail: u128,
    previous_tail_total: u128,
    previous_clean_total: u128,
    carry_through: u128,
    cumulative_after_tail_total: u128,
    tail_onset: u128,
}

fn derive_calibrated_online_scores(
    stage: StageId,
    consistency: Consistency,
    epoch: u16,
    calibration: &StageCalibrationReport,
    counts: &[u64],
) -> Option<CalibratedOnlineScores> {
    if counts.len() != ONLINE_COUNTERS_PER_STAGE {
        return None;
    }
    let thresholds = CalibratedThresholds {
        local: calibration.local().estimate()?.value()?,
        previous_cumulative: calibration
            .previous_cumulative()
            .estimate()
            .and_then(crate::calibration::CalibrationEstimate::value),
        cumulative_after: calibration.cumulative_after().estimate()?.value()?,
    };
    let totals = online_totals(counts);

    let no_predecessor = CalibratedOnlineScore::no_predecessor();
    let (
        local_tail_origin_tail,
        local_tail_rate_given_prev_tail,
        local_tail_rate_given_prev_not_tail,
        amplification_lift,
        carry_through,
    ) = if totals.predecessor_samples == 0 {
        (
            no_predecessor,
            no_predecessor,
            no_predecessor,
            no_predecessor,
            no_predecessor,
        )
    } else {
        (
            CalibratedOnlineScore::probability(
                totals.local_tail_previous_tail,
                totals.local_tail_total,
            ),
            CalibratedOnlineScore::probability(
                totals.local_tail_previous_tail,
                totals.previous_tail_total,
            ),
            CalibratedOnlineScore::probability(
                totals.local_tail_previous_clean,
                totals.previous_clean_total,
            ),
            CalibratedOnlineScore::lift(
                totals.local_tail_previous_tail,
                totals.previous_tail_total,
                totals.local_tail_previous_clean,
                totals.previous_clean_total,
            ),
            CalibratedOnlineScore::probability(totals.carry_through, totals.previous_tail_total),
        )
    };

    Some(CalibratedOnlineScores {
        stage,
        consistency,
        epoch,
        thresholds,
        samples: totals.first_samples + totals.predecessor_samples,
        first_samples: totals.first_samples,
        predecessor_samples: totals.predecessor_samples,
        local_tail_origin_clean: CalibratedOnlineScore::probability(
            totals.local_tail_clean,
            totals.local_tail_total,
        ),
        local_tail_origin_tail,
        local_tail_rate_given_prev_tail,
        local_tail_rate_given_prev_not_tail,
        amplification_lift,
        carry_through,
        tail_onset: CalibratedOnlineScore::probability(
            totals.tail_onset,
            totals.cumulative_after_tail_total,
        ),
    })
}

// Cell indices encode cumulative-after/local/previous tail as bits 0/1/2;
// first-mark cells use only bits 0 and 1.
fn online_totals(counts: &[u64]) -> OnlineTotals {
    let mut totals = OnlineTotals::default();
    for (cell, count) in counts.iter().copied().enumerate() {
        let count = u128::from(count);
        let local_tail = cell & 2 != 0;
        let cumulative_after_tail = cell & 1 != 0;
        if cell < ONLINE_FIRST_COUNTERS {
            totals.first_samples += count;
            if local_tail {
                totals.local_tail_total += count;
                totals.local_tail_clean += count;
            }
            if cumulative_after_tail {
                totals.cumulative_after_tail_total += count;
                totals.tail_onset += count;
            }
            continue;
        }

        totals.predecessor_samples += count;
        let previous_tail = (cell - ONLINE_FIRST_COUNTERS) & 4 != 0;
        if previous_tail {
            totals.previous_tail_total += count;
        } else {
            totals.previous_clean_total += count;
        }
        if local_tail {
            totals.local_tail_total += count;
            if previous_tail {
                totals.local_tail_previous_tail += count;
            } else {
                totals.local_tail_clean += count;
                totals.local_tail_previous_clean += count;
            }
        } else if previous_tail {
            totals.carry_through += count;
        }
        if cumulative_after_tail {
            totals.cumulative_after_tail_total += count;
            if !previous_tail {
                totals.tail_onset += count;
            }
        }
    }
    totals
}

/// Named classification thresholds. There is deliberately no [`Default`] policy.
/// When multiple thresholds pass, onset takes precedence over amplifier, then carry-through.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct DiagnoseConfig {
    /// Minimum `tail_onset` score for [`Classification::Onset`].
    pub onset_threshold: f64,
    /// Minimum `amplification_lift` for [`Classification::Amplifier`].
    pub amplification_lift_threshold: f64,
    /// Minimum `carry_through` score for [`Classification::CarryThrough`].
    pub carry_through_threshold: f64,
}

impl DiagnoseConfig {
    /// Returns the synthetic-PoC policy; these values are not production defaults.
    pub const fn experimental_defaults() -> Self {
        Self {
            onset_threshold: 0.70,
            amplification_lift_threshold: 10.0,
            carry_through_threshold: 0.80,
        }
    }

    const fn is_experimental_defaults(self) -> bool {
        self.onset_threshold.to_bits() == 0.70_f64.to_bits()
            && self.amplification_lift_threshold.to_bits() == 10.0_f64.to_bits()
            && self.carry_through_threshold.to_bits() == 0.80_f64.to_bits()
    }
}

/// Secondary convenience label applied in onset, amplifier, carry-through
/// precedence; the first score meeting its threshold wins.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
#[non_exhaustive]
pub enum Classification {
    /// The tail-onset score met its threshold.
    Onset,
    /// The amplification-lift score met its threshold; tail onset did not.
    Amplifier,
    /// The carry-through score met its threshold; higher-precedence scores did not.
    CarryThrough,
    /// No available score met its classification threshold.
    Inconclusive,
}

impl fmt::Display for Classification {
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        formatter.write_str(match self {
            Self::Onset => "onset",
            Self::Amplifier => "amplifier",
            Self::CarryThrough => "carry-through",
            Self::Inconclusive => "inconclusive",
        })
    }
}

/// A typed failure to derive calibrated-online diagnosis.
#[derive(Clone, Debug, Eq, PartialEq)]
#[non_exhaustive]
pub enum DiagnoseError {
    /// Calibration has not frozen for the requested stage.
    NotCalibrated {
        /// Requested stage.
        stage: StageId,
    },
    /// A delta spans online epochs whose truth-table counters are incomparable.
    EpochMismatch {
        /// Earlier endpoint epoch.
        earlier: u16,
        /// Later endpoint epoch.
        later: u16,
    },
}

impl fmt::Display for DiagnoseError {
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::NotCalibrated { stage } => write!(formatter, "{stage:?} is not calibrated"),
            Self::EpochMismatch { earlier, later } => write!(
                formatter,
                "online diagnosis cannot span calibration epochs {earlier} and {later}"
            ),
        }
    }
}

impl Error for DiagnoseError {}

/// Numeric matrix and calibrated paths plus their secondary stage classification.
#[derive(Clone, Debug, PartialEq)]
pub struct DiagnosisReport {
    stage_name: Box<str>,
    config: DiagnoseConfig,
    matrix_derived: Box<MatrixScores>,
    calibrated_online: Box<CalibratedOnlineScores>,
    matrix_classification: Classification,
    classification: Classification,
}

impl DiagnosisReport {
    /// Returns the diagnosed destination stage.
    pub fn stage(&self) -> StageId {
        self.calibrated_online.stage
    }

    /// Returns its registered read-plane name.
    pub fn stage_name(&self) -> &str {
        &self.stage_name
    }

    /// Returns the caller-selected classification policy.
    pub const fn config(&self) -> DiagnoseConfig {
        self.config
    }

    /// Returns the predecessor-only known-drift matrix path.
    pub fn matrix_derived(&self) -> &MatrixScores {
        &self.matrix_derived
    }

    /// Returns the exact frozen-threshold online path.
    pub fn calibrated_online(&self) -> &CalibratedOnlineScores {
        &self.calibrated_online
    }

    /// Returns the matrix path's secondary convenience label.
    pub const fn matrix_classification(&self) -> Classification {
        self.matrix_classification
    }

    /// Returns the calibrated-online path's secondary convenience label.
    pub const fn classification(&self) -> Classification {
        self.classification
    }
}

impl Snapshot {
    /// Purely derives exact scores from frozen online truth-table counters.
    pub fn calibrated_online_scores(
        &self,
        stage: StageId,
    ) -> Result<CalibratedOnlineScores, DiagnoseError> {
        derive_snapshot_online_scores(self, stage)
    }

    /// Purely diagnoses one calibrated stage while retaining both score paths.
    pub fn diagnose(
        &self,
        stage: StageId,
        config: &DiagnoseConfig,
    ) -> Result<DiagnosisReport, DiagnoseError> {
        build_diagnosis(
            self.stage_name(stage),
            self.matrix_scores(stage).map(Box::new),
            self.calibrated_online_scores(stage).map(Box::new),
            *config,
            stage,
        )
    }
}

impl DeltaSnapshot {
    /// Purely derives exact online scores for a same-epoch window.
    pub fn calibrated_online_scores(
        &self,
        stage: StageId,
    ) -> Result<CalibratedOnlineScores, DiagnoseError> {
        match self.online_delta_availability() {
            OnlineDeltaAvailability::SameEpoch { .. } => derive_delta_online_scores(self, stage),
            OnlineDeltaAvailability::EpochMismatch { earlier, later } => {
                Err(DiagnoseError::EpochMismatch { earlier, later })
            }
        }
    }

    /// Purely diagnoses one calibrated same-epoch window with both score paths.
    pub fn diagnose(
        &self,
        stage: StageId,
        config: &DiagnoseConfig,
    ) -> Result<DiagnosisReport, DiagnoseError> {
        build_diagnosis(
            self.stage_name(stage),
            self.matrix_scores(stage).map(Box::new),
            self.calibrated_online_scores(stage).map(Box::new),
            *config,
            stage,
        )
    }
}

fn derive_snapshot_online_scores(
    snapshot: &Snapshot,
    stage: StageId,
) -> Result<CalibratedOnlineScores, DiagnoseError> {
    let calibration = snapshot
        .calibration_report()
        .and_then(|report| report.stage(stage))
        .ok_or(DiagnoseError::NotCalibrated { stage })?;
    derive_calibrated_online_scores(
        stage,
        snapshot.consistency(),
        snapshot.calibration_epoch(),
        &calibration,
        snapshot
            .online_counts(stage)
            .ok_or(DiagnoseError::NotCalibrated { stage })?,
    )
    .ok_or(DiagnoseError::NotCalibrated { stage })
}

fn derive_delta_online_scores(
    snapshot: &DeltaSnapshot,
    stage: StageId,
) -> Result<CalibratedOnlineScores, DiagnoseError> {
    let calibration = snapshot
        .calibration_report()
        .and_then(|report| report.stage(stage))
        .ok_or(DiagnoseError::NotCalibrated { stage })?;
    derive_calibrated_online_scores(
        stage,
        snapshot.consistency(),
        snapshot.calibration_epoch(),
        &calibration,
        snapshot
            .online_counts(stage)
            .ok_or(DiagnoseError::NotCalibrated { stage })?,
    )
    .ok_or(DiagnoseError::NotCalibrated { stage })
}

fn build_diagnosis(
    stage_name: Option<&str>,
    matrix_derived: Option<Box<MatrixScores>>,
    calibrated_online: Result<Box<CalibratedOnlineScores>, DiagnoseError>,
    config: DiagnoseConfig,
    stage: StageId,
) -> Result<DiagnosisReport, DiagnoseError> {
    let calibrated_online = calibrated_online?;
    let matrix_derived = matrix_derived.ok_or(DiagnoseError::NotCalibrated { stage })?;
    let stage_name = stage_name.ok_or(DiagnoseError::NotCalibrated { stage })?;
    let matrix_classification = matrix_derived.classification(&config);
    let classification = calibrated_online.classification(&config);
    Ok(DiagnosisReport {
        stage_name: stage_name.into(),
        config,
        matrix_derived,
        calibrated_online,
        matrix_classification,
        classification,
    })
}

fn classify_values(
    tail_onset: Option<f64>,
    amplification_lift: Option<f64>,
    carry_through: Option<f64>,
    config: DiagnoseConfig,
) -> Classification {
    if tail_onset.is_some_and(|value| value >= config.onset_threshold) {
        Classification::Onset
    } else if amplification_lift.is_some_and(|value| value >= config.amplification_lift_threshold) {
        Classification::Amplifier
    } else if carry_through.is_some_and(|value| value >= config.carry_through_threshold) {
        Classification::CarryThrough
    } else {
        Classification::Inconclusive
    }
}

impl fmt::Display for DiagnosisReport {
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        writeln!(formatter, "tracegrams diagnosis: {}", self.stage_name)?;
        display_online_scores(formatter, &self.calibrated_online)?;
        display_matrix_scores(formatter, &self.matrix_derived)?;
        writeln!(
            formatter,
            "classification (calibrated-online): {}",
            self.classification
        )?;
        writeln!(
            formatter,
            "classification (matrix-derived): {}",
            self.matrix_classification
        )?;
        write!(
            formatter,
            "classification thresholds: onset>={:.6}, amplification_lift>={:.6}, carry_through>={:.6}",
            self.config.onset_threshold,
            self.config.amplification_lift_threshold,
            self.config.carry_through_threshold,
        )?;
        if self.config.is_experimental_defaults() {
            formatter.write_str(" (experimental PoC values)")?;
        }
        Ok(())
    }
}

fn display_online_scores(
    formatter: &mut fmt::Formatter<'_>,
    scores: &CalibratedOnlineScores,
) -> fmt::Result {
    writeln!(formatter, "path: calibrated-online")?;
    writeln!(
        formatter,
        "  population: all-reached marks (first={}, predecessor={})",
        scores.first_samples, scores.predecessor_samples
    )?;
    writeln!(formatter, "  epoch: {}", scores.epoch)?;
    writeln!(formatter, "  consistency: relaxed")?;
    write!(
        formatter,
        "  thresholds_ns: local={}, previous_cumulative=",
        scores.thresholds.local
    )?;
    if let Some(previous) = scores.thresholds.previous_cumulative {
        write!(formatter, "{previous}")?;
    } else {
        formatter.write_str("n/a")?;
    }
    writeln!(
        formatter,
        ", cumulative_after={}",
        scores.thresholds.cumulative_after
    )?;
    display_online_score(formatter, "tail_onset", scores.tail_onset)?;
    display_online_score(
        formatter,
        "local_tail_origin_clean",
        scores.local_tail_origin_clean,
    )?;
    display_online_score(
        formatter,
        "local_tail_origin_tail",
        scores.local_tail_origin_tail,
    )?;
    display_online_score(
        formatter,
        "local_tail_rate_given_prev_tail",
        scores.local_tail_rate_given_prev_tail,
    )?;
    display_online_score(
        formatter,
        "local_tail_rate_given_prev_not_tail",
        scores.local_tail_rate_given_prev_not_tail,
    )?;
    display_online_score(formatter, "amplification_lift", scores.amplification_lift)?;
    display_online_score(formatter, "carry_through", scores.carry_through)
}

fn display_matrix_scores(formatter: &mut fmt::Formatter<'_>, scores: &MatrixScores) -> fmt::Result {
    writeln!(formatter, "path: matrix-derived (known drift)")?;
    writeln!(
        formatter,
        "  population: predecessor-only (cause={}, incoming={})",
        scores.cause_samples, scores.incoming_samples
    )?;
    writeln!(formatter, "  consistency: relaxed")?;
    write!(formatter, "  threshold_buckets: local=")?;
    display_optional_bucket(formatter, scores.thresholds.local)?;
    formatter.write_str(", previous_cumulative=")?;
    display_optional_bucket(formatter, scores.thresholds.previous_cumulative)?;
    formatter.write_str(", cumulative_after=")?;
    display_optional_bucket(formatter, scores.thresholds.cumulative_after)?;
    writeln!(formatter)?;
    display_matrix_score(formatter, "tail_onset", scores.tail_onset)?;
    display_matrix_score(
        formatter,
        "local_tail_origin_clean",
        scores.local_tail_origin_clean,
    )?;
    display_matrix_score(
        formatter,
        "local_tail_origin_tail",
        scores.local_tail_origin_tail,
    )?;
    display_matrix_score(
        formatter,
        "local_tail_rate_given_prev_tail",
        scores.local_tail_rate_given_prev_tail,
    )?;
    display_matrix_score(
        formatter,
        "local_tail_rate_given_prev_not_tail",
        scores.local_tail_rate_given_prev_not_tail,
    )?;
    display_matrix_score(formatter, "amplification_lift", scores.amplification_lift)?;
    display_matrix_score(formatter, "carry_through", scores.carry_through)
}

fn display_optional_bucket(
    formatter: &mut fmt::Formatter<'_>,
    threshold: Option<BucketThreshold>,
) -> fmt::Result {
    if let Some(threshold) = threshold {
        write!(formatter, "{}", threshold.bucket)
    } else {
        formatter.write_str("n/a")
    }
}

fn display_online_score(
    formatter: &mut fmt::Formatter<'_>,
    name: &str,
    score: CalibratedOnlineScore,
) -> fmt::Result {
    display_score_parts(
        formatter,
        name,
        score.numerator,
        score.denominator,
        score.value,
        score.status,
    )
}

fn display_matrix_score(
    formatter: &mut fmt::Formatter<'_>,
    name: &str,
    score: MatrixScore,
) -> fmt::Result {
    display_score_parts(
        formatter,
        name,
        score.numerator,
        score.denominator,
        score.value,
        score.status,
    )
}

fn display_score_parts(
    formatter: &mut fmt::Formatter<'_>,
    name: &str,
    numerator: u128,
    denominator: u128,
    value: Option<f64>,
    status: ScoreStatus,
) -> fmt::Result {
    write!(formatter, "  {name}: {numerator}/{denominator} = ")?;
    if let Some(value) = value {
        writeln!(formatter, "{value:.6}")
    } else {
        writeln!(formatter, "n/a ({})", score_status_name(status))
    }
}

const fn score_status_name(status: ScoreStatus) -> &'static str {
    match status {
        ScoreStatus::Available => "available",
        ScoreStatus::NoPredecessorPopulation => "no predecessor population",
        ScoreStatus::ZeroDenominator => "zero denominator",
        ScoreStatus::InconsistentSnapshot => "inconsistent relaxed snapshot",
        ScoreStatus::ArithmeticOverflow => "arithmetic overflow",
    }
}

#[cfg(test)]
mod tests {
    use std::time::Duration;

    use super::*;
    use crate::{FreezeCriteria, Tracegrams};

    #[test]
    fn all_twelve_online_truth_cells_feed_the_exact_formula_terms() {
        let mut builder = Tracegrams::builder();
        let first = builder.stage("first").unwrap();
        let destination = builder.stage("destination").unwrap();
        let tracegrams = builder.build().unwrap();
        let mut context = tracegrams.start_manual();
        tracegrams.record_elapsed(&mut context, first, Duration::from_nanos(100));
        tracegrams.record_elapsed(&mut context, destination, Duration::from_nanos(200));
        tracegrams
            .try_freeze_calibration(FreezeCriteria::all_stages(1))
            .unwrap();
        let snapshot = tracegrams.snapshot_relaxed();
        let calibration = snapshot
            .calibration_report()
            .unwrap()
            .stage(destination)
            .unwrap();
        let counts: [u64; ONLINE_COUNTERS_PER_STAGE] =
            std::array::from_fn(|cell| u64::try_from(cell + 1).unwrap());

        let scores = derive_calibrated_online_scores(
            destination,
            Consistency::Relaxed,
            1,
            &calibration,
            &counts,
        )
        .unwrap();

        assert_eq!(scores.samples(), 78);
        assert_eq!(scores.first_samples(), 10);
        assert_eq!(scores.predecessor_samples(), 68);
        assert_eq!(
            (
                scores.tail_onset().numerator(),
                scores.tail_onset().denominator()
            ),
            (20, 42)
        );
        assert_eq!(
            (
                scores.local_tail_origin_clean().numerator(),
                scores.local_tail_origin_clean().denominator(),
            ),
            (22, 45)
        );
        assert_eq!(
            (
                scores.local_tail_origin_tail().numerator(),
                scores.local_tail_origin_tail().denominator(),
            ),
            (23, 45)
        );
        assert_eq!(
            (
                scores.local_tail_rate_given_prev_tail().numerator(),
                scores.local_tail_rate_given_prev_tail().denominator(),
            ),
            (23, 42)
        );
        assert_eq!(
            (
                scores.local_tail_rate_given_prev_not_tail().numerator(),
                scores.local_tail_rate_given_prev_not_tail().denominator(),
            ),
            (15, 26)
        );
        assert_eq!(
            (
                scores.amplification_lift().numerator(),
                scores.amplification_lift().denominator(),
            ),
            (598, 630)
        );
        assert_eq!(
            (
                scores.carry_through().numerator(),
                scores.carry_through().denominator(),
            ),
            (19, 42)
        );
    }

    #[test]
    fn algebraically_skewed_relaxed_counts_are_inconclusive() {
        let score = MatrixScore::probability(2, 1);

        assert_eq!(score.status(), ScoreStatus::InconsistentSnapshot);
        assert_eq!(score.numerator(), 2);
        assert_eq!(score.denominator(), 1);
        assert_eq!(score.value(), None);
    }

    #[test]
    fn marginal_overflow_keeps_a_typed_score_report() {
        let mut builder = Tracegrams::builder();
        builder.stage("first").unwrap();
        let destination = builder.stage("destination").unwrap();
        let mut cause = vec![0_u64; BUCKETS * BUCKETS];
        cause[0] = u64::MAX;
        cause[1] = 1;

        let scores = derive_matrix_scores(
            destination,
            Consistency::Relaxed,
            0.99,
            &cause,
            Some(&cause),
        )
        .unwrap();

        assert_eq!(scores.cause_samples(), u128::from(u64::MAX) + 1);
        assert_eq!(
            scores.local_tail_origin_clean().status(),
            ScoreStatus::ArithmeticOverflow
        );
        assert_eq!(
            scores.tail_onset().status(),
            ScoreStatus::ArithmeticOverflow
        );
    }
}
