//! Pure matrix-derived and calibrated-online tail-propagation diagnosis.

use std::error::Error;
use std::fmt;

use crate::bucket::{BUCKETS, BucketThreshold, nearest_rank};
use crate::recorder::{ONLINE_COUNTERS_PER_STAGE, ONLINE_FIRST_COUNTERS};
use crate::{Snapshot, StageCalibrationReport, StageId};

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
}

/// One numeric diagnosis score and its integer rational terms.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub struct Score {
    numerator: u128,
    denominator: u128,
    status: ScoreStatus,
}

impl Score {
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
    #[allow(clippy::cast_precision_loss)]
    pub fn value(self) -> Option<f64> {
        (self.status == ScoreStatus::Available)
            .then(|| self.numerator as f64 / self.denominator as f64)
    }

    const fn unavailable(status: ScoreStatus) -> Self {
        Self {
            numerator: 0,
            denominator: 0,
            status,
        }
    }

    /// Each numerator is counted in the same pass as a subset of its
    /// denominator, so a relaxed scan cannot produce a ratio above one.
    fn probability(numerator: u128, denominator: u128) -> Self {
        debug_assert!(numerator <= denominator);
        Self::ratio(numerator, denominator)
    }

    /// Cross-multiplies the two rates to keep the ratio in integer terms until
    /// the final `f64` conversion. Every term is a population total bounded by
    /// the `u64` counter contract, so each product fits `u128`.
    fn lift(
        tail_local_tail: u128,
        previous_tail: u128,
        clean_local_tail: u128,
        previous_clean: u128,
    ) -> Self {
        Self::ratio(
            tail_local_tail * previous_clean,
            previous_tail * clean_local_tail,
        )
    }

    const fn ratio(numerator: u128, denominator: u128) -> Self {
        Self {
            numerator,
            denominator,
            status: if denominator == 0 {
                ScoreStatus::ZeroDenominator
            } else {
                ScoreStatus::Available
            },
        }
    }
}

/// The seven diagnosis scores of one score path.
///
/// Both paths use the same formulas. The matrix-derived path counts only
/// predecessor-bearing marks. The calibrated-online path also counts first
/// marks as clean sentinels: they enter `tail_onset`, `local_tail_origin_clean`,
/// and the denominator of `local_tail_origin_tail`.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
#[non_exhaustive]
pub struct Scores {
    /// `P(previous not tail | cumulative after tail)`.
    pub tail_onset: Score,
    /// `P(previous not tail | local tail)`.
    pub local_tail_origin_clean: Score,
    /// `P(previous tail | local tail)`.
    pub local_tail_origin_tail: Score,
    /// `P(local tail | previous tail)`.
    pub local_tail_rate_given_prev_tail: Score,
    /// `P(local tail | previous not tail)`.
    pub local_tail_rate_given_prev_not_tail: Score,
    /// Ratio between the previous-tail and previous-clean local-tail rates.
    pub amplification_lift: Score,
    /// `P(local not tail | previous tail)`.
    pub carry_through: Score,
}

impl Scores {
    /// Applies an explicit secondary classification policy to these scores.
    pub fn classification(&self, config: &DiagnoseConfig) -> Classification {
        let meets = |score: Score, threshold: f64| {
            !threshold.is_nan() && score.value().is_some_and(|value| value >= threshold)
        };
        if meets(self.tail_onset, config.onset_threshold) {
            Classification::Onset
        } else if meets(self.amplification_lift, config.amplification_lift_threshold) {
            Classification::Amplifier
        } else if meets(self.carry_through, config.carry_through_threshold) {
            Classification::CarryThrough
        } else {
            Classification::Inconclusive
        }
    }

    const fn all(score: Score) -> Self {
        Self {
            tail_onset: score,
            local_tail_origin_clean: score,
            local_tail_origin_tail: score,
            local_tail_rate_given_prev_tail: score,
            local_tail_rate_given_prev_not_tail: score,
            amplification_lift: score,
            carry_through: score,
        }
    }

    const fn named(&self) -> [(&'static str, Score); 7] {
        [
            ("tail_onset", self.tail_onset),
            ("local_tail_origin_clean", self.local_tail_origin_clean),
            ("local_tail_origin_tail", self.local_tail_origin_tail),
            (
                "local_tail_rate_given_prev_tail",
                self.local_tail_rate_given_prev_tail,
            ),
            (
                "local_tail_rate_given_prev_not_tail",
                self.local_tail_rate_given_prev_not_tail,
            ),
            ("amplification_lift", self.amplification_lift),
            ("carry_through", self.carry_through),
        ]
    }
}

/// Matrix-derived tail thresholds for one destination stage.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
#[non_exhaustive]
pub struct MatrixThresholds {
    /// Local-latency threshold derived from the cause matrix.
    pub local: Option<BucketThreshold>,
    /// Previous-cumulative threshold derived from the cause matrix.
    pub previous_cumulative: Option<BucketThreshold>,
    /// Cumulative-after threshold derived from the incoming matrix.
    pub cumulative_after: Option<BucketThreshold>,
}

/// Predecessor-only approximate scores for one destination stage.
///
/// Thresholds are nearest-rank buckets of the matrix marginals, so this path
/// has known compact-bucket drift against the calibrated-online path.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
#[non_exhaustive]
pub struct MatrixScores {
    /// Destination stage.
    pub stage: StageId,
    /// Nearest-rank bucket thresholds for this destination.
    pub thresholds: MatrixThresholds,
    /// Predecessor-bearing samples observed in the cause matrix.
    pub cause_samples: u128,
    /// Predecessor-bearing samples observed in the incoming matrix.
    pub incoming_samples: u128,
    /// Scores over predecessor-bearing marks.
    pub scores: Scores,
}

impl Snapshot {
    /// Purely derives predecessor-only approximate scores from matrix cells.
    ///
    /// Unlike [`Snapshot::diagnose`], this path does not require calibration.
    pub fn matrix_scores(&self, stage: StageId) -> Option<MatrixScores> {
        Some(derive_matrix_scores(
            stage,
            self.tail_quantile(),
            self.cause_counts(stage)?,
            self.incoming_counts(stage),
        ))
    }
}

fn derive_matrix_scores(
    stage: StageId,
    tail_quantile: f64,
    cause: &[u64],
    incoming: Option<&[u64]>,
) -> MatrixScores {
    let (cause_rows, cause_columns) = matrix_marginals(cause);
    let previous_cumulative = nearest_rank(&cause_rows, tail_quantile);
    let local = nearest_rank(&cause_columns, tail_quantile);
    let cumulative_after =
        incoming.and_then(|counts| nearest_rank(&matrix_marginals(counts).1, tail_quantile));

    let tail_onset = match (incoming, previous_cumulative, cumulative_after) {
        (Some(counts), Some(previous), Some(after)) => {
            tail_onset_score(counts, previous.bucket, after.bucket)
        }
        _ => Score::unavailable(ScoreStatus::NoPredecessorPopulation),
    };
    let scores = match (previous_cumulative, local) {
        (Some(previous), Some(local)) => {
            cause_scores(cause, previous.bucket, local.bucket, tail_onset)
        }
        _ => Scores {
            tail_onset,
            ..Scores::all(Score::unavailable(ScoreStatus::NoPredecessorPopulation))
        },
    };

    MatrixScores {
        stage,
        thresholds: MatrixThresholds {
            local,
            previous_cumulative,
            cumulative_after,
        },
        cause_samples: matrix_total(cause),
        incoming_samples: incoming.map_or(0, matrix_total),
        scores,
    }
}

/// Returns row and column marginals. Each marginal is a sub-total of one
/// population, so it stays within the `u64` counter contract.
fn matrix_marginals(counts: &[u64]) -> ([u64; BUCKETS], [u64; BUCKETS]) {
    debug_assert_eq!(counts.len(), BUCKETS * BUCKETS);
    let mut rows = [0; BUCKETS];
    let mut columns = [0; BUCKETS];
    for (index, count) in counts.iter().copied().enumerate() {
        rows[index / BUCKETS] += count;
        columns[index % BUCKETS] += count;
    }
    (rows, columns)
}

fn matrix_total(counts: &[u64]) -> u128 {
    counts.iter().map(|count| u128::from(*count)).sum()
}

fn cause_scores(
    counts: &[u64],
    previous_threshold: usize,
    local_threshold: usize,
    tail_onset: Score,
) -> Scores {
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
    Scores {
        tail_onset,
        local_tail_origin_clean: Score::probability(
            local_tail_from_prev_not_tail,
            local_tail_total,
        ),
        local_tail_origin_tail: Score::probability(local_tail_from_prev_tail, local_tail_total),
        local_tail_rate_given_prev_tail: Score::probability(
            local_tail_from_prev_tail,
            prev_tail_total,
        ),
        local_tail_rate_given_prev_not_tail: Score::probability(
            local_tail_from_prev_not_tail,
            prev_not_tail_total,
        ),
        amplification_lift: Score::lift(
            local_tail_from_prev_tail,
            prev_tail_total,
            local_tail_from_prev_not_tail,
            prev_not_tail_total,
        ),
        carry_through: Score::probability(carry_through, prev_tail_total),
    }
}

fn tail_onset_score(
    counts: &[u64],
    previous_threshold: usize,
    cumulative_after_threshold: usize,
) -> Score {
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
    Score::probability(numerator, denominator)
}

/// Frozen nanosecond thresholds used by one calibrated-online stage report.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
#[non_exhaustive]
pub struct CalibratedThresholds {
    /// Frozen local-tail predicate threshold.
    pub local_ns: u64,
    /// Frozen previous-cumulative predicate threshold when calibrated.
    pub previous_cumulative_ns: Option<u64>,
    /// Frozen cumulative-after-tail predicate threshold.
    pub cumulative_after_ns: u64,
}

impl CalibratedThresholds {
    fn from_report(report: &StageCalibrationReport) -> Self {
        let finite = |estimate: Option<crate::CalibrationEstimate>| {
            estimate
                .and_then(|estimate| estimate.value)
                .expect("a published report holds finite required thresholds")
        };
        Self {
            local_ns: finite(report.local.estimate),
            previous_cumulative_ns: report
                .previous_cumulative
                .estimate
                .and_then(|estimate| estimate.value),
            cumulative_after_ns: finite(report.cumulative_after.estimate),
        }
    }
}

/// Exact frozen-threshold scores for one reached-stage population.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
#[non_exhaustive]
pub struct CalibratedOnlineScores {
    /// Destination stage.
    pub stage: StageId,
    /// Frozen thresholds.
    pub thresholds: CalibratedThresholds,
    /// Classified first marks.
    pub first_samples: u128,
    /// Classified predecessor-bearing marks.
    pub predecessor_samples: u128,
    /// Scores over first and predecessor-bearing marks.
    pub scores: Scores,
}

impl CalibratedOnlineScores {
    /// Returns all classified first and predecessor-bearing marks.
    pub const fn samples(&self) -> u128 {
        self.first_samples + self.predecessor_samples
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
    calibration: &StageCalibrationReport,
    counts: &[u64],
) -> CalibratedOnlineScores {
    debug_assert_eq!(counts.len(), ONLINE_COUNTERS_PER_STAGE);
    let totals = online_totals(counts);
    let first_mark_scores = Scores {
        tail_onset: Score::probability(totals.tail_onset, totals.cumulative_after_tail_total),
        local_tail_origin_clean: Score::probability(
            totals.local_tail_clean,
            totals.local_tail_total,
        ),
        ..Scores::all(Score::unavailable(ScoreStatus::NoPredecessorPopulation))
    };
    let scores = if totals.predecessor_samples == 0 {
        first_mark_scores
    } else {
        Scores {
            local_tail_origin_tail: Score::probability(
                totals.local_tail_previous_tail,
                totals.local_tail_total,
            ),
            local_tail_rate_given_prev_tail: Score::probability(
                totals.local_tail_previous_tail,
                totals.previous_tail_total,
            ),
            local_tail_rate_given_prev_not_tail: Score::probability(
                totals.local_tail_previous_clean,
                totals.previous_clean_total,
            ),
            amplification_lift: Score::lift(
                totals.local_tail_previous_tail,
                totals.previous_tail_total,
                totals.local_tail_previous_clean,
                totals.previous_clean_total,
            ),
            carry_through: Score::probability(totals.carry_through, totals.previous_tail_total),
            ..first_mark_scores
        }
    };

    CalibratedOnlineScores {
        stage,
        thresholds: CalibratedThresholds::from_report(calibration),
        first_samples: totals.first_samples,
        predecessor_samples: totals.predecessor_samples,
        scores,
    }
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
    ///
    /// `NaN` is never met. Negative values are accepted and are met by any
    /// available nonnegative score.
    pub onset_threshold: f64,
    /// Minimum `amplification_lift` for [`Classification::Amplifier`].
    ///
    /// `NaN` is never met. Negative values are accepted and are met by any
    /// available nonnegative lift.
    pub amplification_lift_threshold: f64,
    /// Minimum `carry_through` score for [`Classification::CarryThrough`].
    ///
    /// `NaN` is never met. Negative values are accepted and are met by any
    /// available nonnegative score.
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
    /// The stage does not belong to the snapshot's recorder.
    ForeignStage {
        /// Requested stage.
        stage: StageId,
    },
    /// Calibration has not frozen for the requested stage.
    NotCalibrated {
        /// Requested stage.
        stage: StageId,
    },
    /// A window spans the calibration freeze, so its online counters cover
    /// only part of the window.
    WindowSpansFreeze,
}

impl fmt::Display for DiagnoseError {
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::ForeignStage { stage } => {
                write!(formatter, "{stage:?} belongs to another recorder")
            }
            Self::NotCalibrated { stage } => write!(formatter, "{stage:?} is not calibrated"),
            Self::WindowSpansFreeze => {
                formatter.write_str("online diagnosis cannot span the calibration freeze")
            }
        }
    }
}

impl Error for DiagnoseError {}

/// Numeric matrix and calibrated paths plus their secondary stage classification.
#[derive(Clone, Debug, PartialEq)]
pub struct DiagnosisReport {
    stage_name: Box<str>,
    config: DiagnoseConfig,
    matrix_derived: MatrixScores,
    calibrated_online: CalibratedOnlineScores,
}

impl DiagnosisReport {
    /// Returns the diagnosed destination stage.
    pub const fn stage(&self) -> StageId {
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
    pub const fn matrix_derived(&self) -> &MatrixScores {
        &self.matrix_derived
    }

    /// Returns the exact frozen-threshold online path.
    pub const fn calibrated_online(&self) -> &CalibratedOnlineScores {
        &self.calibrated_online
    }

    /// Returns the matrix path's secondary convenience label.
    pub fn matrix_classification(&self) -> Classification {
        self.matrix_derived.scores.classification(&self.config)
    }

    /// Returns the calibrated-online path's secondary convenience label.
    pub fn classification(&self) -> Classification {
        self.calibrated_online.scores.classification(&self.config)
    }
}

impl Snapshot {
    /// Purely diagnoses one calibrated stage while retaining both score paths.
    pub fn diagnose(
        &self,
        stage: StageId,
        config: &DiagnoseConfig,
    ) -> Result<DiagnosisReport, DiagnoseError> {
        let index = self
            .stage_index(stage)
            .ok_or(DiagnoseError::ForeignStage { stage })?;
        let calibrated_online = self.calibrated_online_scores(stage)?;
        let Some(matrix_derived) = self.matrix_scores(stage) else {
            unreachable!("a validated stage has matrix storage");
        };
        Ok(DiagnosisReport {
            stage_name: self.stages()[index].name.clone(),
            config: *config,
            matrix_derived,
            calibrated_online,
        })
    }

    /// Derives exact scores from frozen online truth-table counters for a
    /// stage that belongs to this snapshot.
    pub(crate) fn calibrated_online_scores(
        &self,
        stage: StageId,
    ) -> Result<CalibratedOnlineScores, DiagnoseError> {
        if self.spans_freeze() {
            return Err(DiagnoseError::WindowSpansFreeze);
        }
        let calibration = self
            .calibration_report()
            .and_then(|report| report.stage(stage))
            .ok_or(DiagnoseError::NotCalibrated { stage })?;
        let counts = self
            .online_counts(stage)
            .expect("a validated stage has online storage");
        Ok(derive_calibrated_online_scores(stage, &calibration, counts))
    }
}

impl fmt::Display for DiagnosisReport {
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        let online = &self.calibrated_online;
        let matrix = &self.matrix_derived;
        writeln!(formatter, "tracegrams diagnosis: {}", self.stage_name)?;
        writeln!(formatter, "consistency: relaxed")?;

        writeln!(formatter, "path: calibrated-online")?;
        writeln!(
            formatter,
            "  population: all-reached marks (first={}, predecessor={})",
            online.first_samples, online.predecessor_samples
        )?;
        write!(
            formatter,
            "  thresholds_ns: local={}, previous_cumulative=",
            online.thresholds.local_ns
        )?;
        display_optional(formatter, online.thresholds.previous_cumulative_ns)?;
        writeln!(
            formatter,
            ", cumulative_after={}",
            online.thresholds.cumulative_after_ns
        )?;
        display_scores(formatter, &online.scores)?;

        writeln!(formatter, "path: matrix-derived (known drift)")?;
        writeln!(
            formatter,
            "  population: predecessor-only (cause={}, incoming={})",
            matrix.cause_samples, matrix.incoming_samples
        )?;
        let bucket = |threshold: Option<BucketThreshold>| threshold.map(|t| t.bucket);
        formatter.write_str("  threshold_buckets: local=")?;
        display_optional(formatter, bucket(matrix.thresholds.local))?;
        formatter.write_str(", previous_cumulative=")?;
        display_optional(formatter, bucket(matrix.thresholds.previous_cumulative))?;
        formatter.write_str(", cumulative_after=")?;
        display_optional(formatter, bucket(matrix.thresholds.cumulative_after))?;
        writeln!(formatter)?;
        display_scores(formatter, &matrix.scores)?;

        writeln!(
            formatter,
            "classification (calibrated-online): {}",
            self.classification()
        )?;
        writeln!(
            formatter,
            "classification (matrix-derived): {}",
            self.matrix_classification()
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

fn display_optional(
    formatter: &mut fmt::Formatter<'_>,
    value: Option<impl fmt::Display>,
) -> fmt::Result {
    match value {
        Some(value) => write!(formatter, "{value}"),
        None => formatter.write_str("n/a"),
    }
}

fn display_scores(formatter: &mut fmt::Formatter<'_>, scores: &Scores) -> fmt::Result {
    for (name, score) in scores.named() {
        write!(
            formatter,
            "  {name}: {}/{} = ",
            score.numerator, score.denominator
        )?;
        match score.value() {
            Some(value) => writeln!(formatter, "{value:.6}")?,
            None => writeln!(formatter, "n/a ({})", score_status_name(score.status))?,
        }
    }
    Ok(())
}

const fn score_status_name(status: ScoreStatus) -> &'static str {
    match status {
        ScoreStatus::Available => "available",
        ScoreStatus::NoPredecessorPopulation => "no predecessor population",
        ScoreStatus::ZeroDenominator => "zero denominator",
    }
}

#[cfg(test)]
mod tests {
    use std::time::Duration;

    use super::*;
    use crate::Tracegrams;

    #[test]
    fn all_twelve_online_truth_cells_feed_the_exact_formula_terms() {
        let mut builder = Tracegrams::builder();
        let first = builder.stage("first").unwrap();
        let destination = builder.stage("destination").unwrap();
        let tracegrams = builder.build().unwrap();
        let mut context = tracegrams.start_manual();
        tracegrams.record_elapsed(&mut context, first, Duration::from_nanos(100));
        tracegrams.record_elapsed(&mut context, destination, Duration::from_nanos(200));
        tracegrams.try_freeze_calibration(1).unwrap();
        let snapshot = tracegrams.snapshot_relaxed();
        let calibration = snapshot
            .calibration_report()
            .unwrap()
            .stage(destination)
            .unwrap();
        let counts: [u64; ONLINE_COUNTERS_PER_STAGE] =
            std::array::from_fn(|cell| u64::try_from(cell + 1).unwrap());

        let online = derive_calibrated_online_scores(destination, &calibration, &counts);

        assert_eq!(online.samples(), 78);
        assert_eq!(online.first_samples, 10);
        assert_eq!(online.predecessor_samples, 68);
        let terms = |score: Score| (score.numerator, score.denominator);
        let scores = online.scores;
        assert_eq!(terms(scores.tail_onset), (20, 42));
        assert_eq!(terms(scores.local_tail_origin_clean), (22, 45));
        assert_eq!(terms(scores.local_tail_origin_tail), (23, 45));
        assert_eq!(terms(scores.local_tail_rate_given_prev_tail), (23, 42));
        assert_eq!(terms(scores.local_tail_rate_given_prev_not_tail), (15, 26));
        assert_eq!(terms(scores.amplification_lift), (598, 630));
        assert_eq!(terms(scores.carry_through), (19, 42));
    }

    fn scores(tail_onset: u128, amplification_lift: u128, carry_through: u128) -> Scores {
        // Every ratio is `value / 1`, so `0` and `1` are the only rates.
        Scores {
            tail_onset: Score::ratio(tail_onset, 1),
            amplification_lift: Score::ratio(amplification_lift, 1),
            carry_through: Score::ratio(carry_through, 1),
            ..Scores::all(Score::unavailable(ScoreStatus::NoPredecessorPopulation))
        }
    }

    #[test]
    fn nan_classification_thresholds_are_never_met() {
        let classify = |config| scores(1, 1, 1).classification(&config);

        assert_eq!(
            classify(DiagnoseConfig {
                onset_threshold: f64::NAN,
                amplification_lift_threshold: 0.5,
                carry_through_threshold: 0.5,
            }),
            Classification::Amplifier
        );
        assert_eq!(
            classify(DiagnoseConfig {
                onset_threshold: f64::NAN,
                amplification_lift_threshold: f64::NAN,
                carry_through_threshold: 0.5,
            }),
            Classification::CarryThrough
        );
        assert_eq!(
            classify(DiagnoseConfig {
                onset_threshold: f64::NAN,
                amplification_lift_threshold: f64::NAN,
                carry_through_threshold: f64::NAN,
            }),
            Classification::Inconclusive
        );
    }

    #[test]
    fn negative_classification_thresholds_are_accepted_minima() {
        let classify = |config| scores(0, 0, 0).classification(&config);

        assert_eq!(
            classify(DiagnoseConfig {
                onset_threshold: -1.0,
                amplification_lift_threshold: f64::NAN,
                carry_through_threshold: f64::NAN,
            }),
            Classification::Onset
        );
        assert_eq!(
            classify(DiagnoseConfig {
                onset_threshold: f64::NAN,
                amplification_lift_threshold: -1.0,
                carry_through_threshold: f64::NAN,
            }),
            Classification::Amplifier
        );
        assert_eq!(
            classify(DiagnoseConfig {
                onset_threshold: f64::NAN,
                amplification_lift_threshold: f64::NAN,
                carry_through_threshold: -1.0,
            }),
            Classification::CarryThrough
        );
    }
}
