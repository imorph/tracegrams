//! Matrix-derived tail-propagation scores.

use crate::bucket::{BUCKETS, RankSelection, nearest_rank};
use crate::{Consistency, DeltaSnapshot, Snapshot, StageId};

/// The counter path from which a score was derived.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
#[non_exhaustive]
pub enum ScorePath {
    /// Approximate bucket thresholds and scores derived from transition matrices.
    MatrixDerived,
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
}

/// Availability of one numeric score.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
#[non_exhaustive]
pub enum ScoreStatus {
    /// The numerator, denominator, and ratio are available.
    Available,
    /// No predecessor-bearing samples were observed for the required matrix.
    NoPredecessorPopulation,
    /// The score's denominator is zero.
    ZeroDenominator,
    /// Relaxed inputs violated a probability invariant.
    InconsistentSnapshot,
    /// Exact threshold or ratio arithmetic exceeded its supported integer range.
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

/// One numeric matrix-derived score and its exact aggregate counts.
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

    /// Returns the exact aggregate numerator observed in the snapshot.
    pub const fn numerator(self) -> u128 {
        self.numerator
    }

    /// Returns the exact aggregate denominator observed in the snapshot.
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
        if denominator == 0 {
            return Self {
                numerator,
                denominator,
                value: None,
                status: ScoreStatus::ZeroDenominator,
            };
        }
        if numerator > denominator {
            return Self {
                numerator,
                denominator,
                value: None,
                status: ScoreStatus::InconsistentSnapshot,
            };
        }
        Self::ratio(numerator, denominator)
    }

    fn lift(
        tail_local_tail: u128,
        previous_tail: u128,
        clean_local_tail: u128,
        previous_clean: u128,
    ) -> Self {
        let Some(numerator) = tail_local_tail.checked_mul(previous_clean) else {
            return Self::unavailable(ScoreStatus::ArithmeticOverflow);
        };
        let Some(denominator) = previous_tail.checked_mul(clean_local_tail) else {
            return Self::unavailable(ScoreStatus::ArithmeticOverflow);
        };
        if denominator == 0 {
            return Self {
                numerator,
                denominator,
                value: None,
                status: ScoreStatus::ZeroDenominator,
            };
        }
        Self::ratio(numerator, denominator)
    }

    #[allow(clippy::cast_precision_loss)]
    fn ratio(numerator: u128, denominator: u128) -> Self {
        Self {
            numerator,
            denominator,
            value: Some(numerator as f64 / denominator as f64),
            status: ScoreStatus::Available,
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

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
        let mut builder = crate::Tracegrams::builder();
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
