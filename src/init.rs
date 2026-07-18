//! Runtime stage registration and bounded recorder construction.

use std::error::Error;
use std::fmt;
use std::mem::size_of;
use std::sync::Arc;
use std::sync::atomic::{AtomicU64, Ordering};

use crate::bucket::{CALIBRATION_BUCKETS, calibration_bounds, default_bounds};
use crate::recorder::{FixedStorageLayout, Inner};

const DEFAULT_TAIL_QUANTILE: f64 = 0.99;
const DEFAULT_MEMORY_BUDGET_BYTES: usize = 8 * 1024 * 1024;
const MAX_STAGES: usize = 64;
const FIRST_REGISTRY_COOKIE: u64 = 1;
const MAX_REGISTRY_COOKIE: u64 = u32::MAX as u64;

static NEXT_REGISTRY_COOKIE: AtomicU64 = AtomicU64::new(FIRST_REGISTRY_COOKIE);

#[derive(Clone, Copy, Debug, Eq, Hash, PartialEq)]
pub(crate) struct RegistryCookie(u32);

/// An opaque, registry-branded identifier for a stage.
///
/// Identifiers are dense and ordered by registration. An identifier is valid
/// only with the [`Tracegrams`] instance built by the same builder.
#[derive(Clone, Copy, Eq, Hash, PartialEq)]
pub struct StageId {
    cookie: RegistryCookie,
    index: u8,
}

impl fmt::Debug for StageId {
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        formatter.debug_tuple("StageId").field(&self.index).finish()
    }
}

/// Itemized recorder memory estimate, excluding allocator rounding.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub struct MemoryEstimate {
    matrix: usize,
    calibration: usize,
    online: usize,
    bounds: usize,
    stage_metadata: usize,
    completion: usize,
    diagnostics: usize,
    recorder_metadata: usize,
    total: usize,
}

impl MemoryEstimate {
    /// Bytes used by ordinary distributions and transition matrices.
    pub const fn matrix_bytes(self) -> usize {
        self.matrix
    }

    /// Bytes used by calibration distributions.
    pub const fn calibration_bytes(self) -> usize {
        self.calibration
    }

    /// Bytes used by calibrated-online truth-table counters.
    pub const fn online_bytes(self) -> usize {
        self.online
    }

    /// Bytes used by the pinned ordinary and calibration bounds.
    pub const fn bounds_bytes(self) -> usize {
        self.bounds
    }

    /// Bytes used by stage-name metadata.
    pub const fn stage_metadata_bytes(self) -> usize {
        self.stage_metadata
    }

    /// Bytes used by success and error completion counters.
    pub const fn completion_bytes(self) -> usize {
        self.completion
    }

    /// Bytes used by hot-path diagnostic counters.
    pub const fn diagnostics_bytes(self) -> usize {
        self.diagnostics
    }

    /// Bytes used by fixed recorder and shared-handle metadata.
    pub const fn recorder_metadata_bytes(self) -> usize {
        self.recorder_metadata
    }

    /// Total bytes included in the estimate.
    pub const fn total_bytes(self) -> usize {
        self.total
    }
}

/// A structured failure while registering or building a recorder.
#[derive(Debug, PartialEq)]
#[non_exhaustive]
pub enum InitError {
    /// A stage name was empty.
    EmptyStageName,
    /// A stage name was already registered.
    DuplicateStageName {
        /// The duplicate name.
        name: String,
    },
    /// No stages were registered.
    NoStages,
    /// Registration exceeded the supported stage count.
    TooManyStages {
        /// Maximum supported stage count.
        maximum: usize,
    },
    /// The tail quantile was non-finite or outside `(0, 1]`.
    InvalidTailQuantile {
        /// The rejected quantile.
        value: f64,
    },
    /// The process-wide registry-cookie space was exhausted.
    RegistryCookieExhausted,
    /// Checked storage-size arithmetic overflowed.
    SizeOverflow,
    /// The recorder estimate exceeded the configured budget.
    MemoryBudgetExceeded {
        /// Estimated bytes required.
        required: usize,
        /// Configured byte budget.
        budget: usize,
    },
}

impl fmt::Display for InitError {
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::EmptyStageName => formatter.write_str("stage name must not be empty"),
            Self::DuplicateStageName { name } => {
                write!(formatter, "stage name {name:?} is already registered")
            }
            Self::NoStages => formatter.write_str("at least one stage must be registered"),
            Self::TooManyStages { maximum } => {
                write!(formatter, "at most {maximum} stages may be registered")
            }
            Self::InvalidTailQuantile { value } => {
                write!(formatter, "tail quantile {value:?} is outside (0, 1]")
            }
            Self::RegistryCookieExhausted => {
                formatter.write_str("registry-cookie space is exhausted")
            }
            Self::SizeOverflow => formatter.write_str("recorder size calculation overflowed"),
            Self::MemoryBudgetExceeded { required, budget } => write!(
                formatter,
                "recorder requires {required} bytes, exceeding the {budget}-byte budget"
            ),
        }
    }
}

impl Error for InitError {}

/// Builder for a fixed-size shared [`Tracegrams`] recorder.
#[derive(Debug)]
pub struct TracegramsBuilder {
    cookie: Option<RegistryCookie>,
    stage_names: Vec<Box<str>>,
    tail_quantile: f64,
    memory_budget_bytes: usize,
}

impl TracegramsBuilder {
    fn new() -> Self {
        Self {
            cookie: claim_registry_cookie(&NEXT_REGISTRY_COOKIE),
            stage_names: Vec::new(),
            tail_quantile: DEFAULT_TAIL_QUANTILE,
            memory_budget_bytes: DEFAULT_MEMORY_BUDGET_BYTES,
        }
    }

    /// Registers a stage and returns its dense, registry-branded identifier.
    pub fn stage(&mut self, name: &str) -> Result<StageId, InitError> {
        let cookie = self.cookie.ok_or(InitError::RegistryCookieExhausted)?;
        if name.is_empty() {
            return Err(InitError::EmptyStageName);
        }
        if self.stage_names.len() == MAX_STAGES {
            return Err(InitError::TooManyStages {
                maximum: MAX_STAGES,
            });
        }
        if self
            .stage_names
            .iter()
            .any(|registered| **registered == *name)
        {
            return Err(InitError::DuplicateStageName {
                name: name.to_owned(),
            });
        }

        let index = u8::try_from(self.stage_names.len()).map_err(|_| InitError::SizeOverflow)?;
        self.stage_names.push(Box::from(name));
        Ok(StageId { cookie, index })
    }

    /// Sets the nearest-rank tail quantile used for calibration.
    pub fn tail_quantile(&mut self, quantile: f64) -> Result<&mut Self, InitError> {
        if !quantile.is_finite() || quantile <= 0.0 || quantile > 1.0 {
            return Err(InitError::InvalidTailQuantile { value: quantile });
        }
        self.tail_quantile = quantile;
        Ok(self)
    }

    /// Sets the maximum recorder estimate accepted by [`Self::build`].
    pub const fn memory_budget_bytes(&mut self, budget: usize) -> &mut Self {
        self.memory_budget_bytes = budget;
        self
    }

    /// Estimates all fixed recorder-owned memory before building.
    pub fn estimated_memory(&self) -> Result<MemoryEstimate, InitError> {
        let layout = self.storage_layout()?;
        estimate_memory(layout, self.stage_names.iter().map(|name| name.len()))
    }

    /// Validates the configuration and allocates one fixed shared recorder.
    pub fn build(self) -> Result<Tracegrams, InitError> {
        let cookie = self.cookie.ok_or(InitError::RegistryCookieExhausted)?;
        let layout = self.storage_layout()?;
        let estimate = estimate_memory(layout, self.stage_names.iter().map(|name| name.len()))?;
        if estimate.total > self.memory_budget_bytes {
            return Err(InitError::MemoryBudgetExceeded {
                required: estimate.total,
                budget: self.memory_budget_bytes,
            });
        }

        Ok(Tracegrams {
            inner: Arc::new(Inner::new(
                cookie,
                self.stage_names.into_boxed_slice(),
                self.tail_quantile,
                self.memory_budget_bytes,
                estimate,
                layout,
            )),
        })
    }

    fn storage_layout(&self) -> Result<FixedStorageLayout, InitError> {
        if self.stage_names.is_empty() {
            return Err(InitError::NoStages);
        }
        FixedStorageLayout::new(self.stage_names.len()).ok_or(InitError::SizeOverflow)
    }
}

/// Cheap cloneable handle to one fixed-size shared recorder.
#[derive(Clone)]
pub struct Tracegrams {
    pub(crate) inner: Arc<Inner>,
}

impl Tracegrams {
    /// Starts a runtime registry builder.
    ///
    /// ```
    /// use tracegrams::Tracegrams;
    ///
    /// # fn main() -> Result<(), Box<dyn std::error::Error>> {
    /// let mut builder = Tracegrams::builder();
    /// let _parse = builder.stage("parse")?;
    /// let _db = builder.stage("db")?;
    /// builder.tail_quantile(0.99)?;
    /// builder.memory_budget_bytes(8 * 1024 * 1024);
    /// let _tracegrams = builder.build()?;
    /// # Ok(())
    /// # }
    /// ```
    pub fn builder() -> TracegramsBuilder {
        TracegramsBuilder::new()
    }
}

impl fmt::Debug for Tracegrams {
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        formatter
            .debug_struct("Tracegrams")
            .field("stages", &self.inner.stage_names.len())
            .field("tail_quantile", &self.inner.tail_quantile)
            .field("memory_budget_bytes", &self.inner.memory_budget_bytes)
            .field("memory_estimate", &self.inner.memory_estimate)
            .finish_non_exhaustive()
    }
}

fn claim_registry_cookie(source: &AtomicU64) -> Option<RegistryCookie> {
    let mut current = source.load(Ordering::Relaxed);
    loop {
        if current > MAX_REGISTRY_COOKIE {
            return None;
        }
        let cookie = u32::try_from(current).ok()?;
        match source.compare_exchange_weak(
            current,
            current + 1,
            Ordering::Relaxed,
            Ordering::Relaxed,
        ) {
            Ok(_) => return Some(RegistryCookie(cookie)),
            Err(observed) => current = observed,
        }
    }
}

fn estimate_memory(
    layout: FixedStorageLayout,
    stage_name_lengths: impl IntoIterator<Item = usize>,
) -> Result<MemoryEstimate, InitError> {
    let counter_bytes = size_of::<AtomicU64>();
    let matrix_bytes = checked_bytes(layout.matrix_counter_count(), counter_bytes)?;
    let calibration_bytes = checked_bytes(layout.calibration_counter_count(), counter_bytes)?;
    let online_bytes = checked_bytes(layout.online_counter_count(), counter_bytes)?;
    let completion_bytes = checked_bytes(layout.completion_counter_count(), counter_bytes)?;
    let diagnostics_bytes = checked_bytes(
        FixedStorageLayout::diagnostic_counter_count(),
        counter_bytes,
    )?;
    let bounds_count = default_bounds()
        .len()
        .checked_add(calibration_bounds().len())
        .ok_or(InitError::SizeOverflow)?;
    let bounds_bytes = checked_bytes(bounds_count, size_of::<u64>())?;

    let stage_headers = checked_bytes(layout.stage_count(), size_of::<Box<str>>())?;
    let stage_name_bytes = stage_name_lengths
        .into_iter()
        .try_fold(0_usize, |total, length| {
            total.checked_add(length).ok_or(InitError::SizeOverflow)
        })?;
    let stage_metadata_bytes = stage_headers
        .checked_add(stage_name_bytes)
        .ok_or(InitError::SizeOverflow)?;

    // `Arc` keeps two reference counts alongside `Inner`; allocator rounding
    // remains platform-specific and is intentionally outside this estimate.
    let recorder_metadata_bytes = size_of::<Inner>()
        .checked_add(2 * size_of::<usize>())
        .ok_or(InitError::SizeOverflow)?;
    let total_bytes = [
        matrix_bytes,
        calibration_bytes,
        online_bytes,
        bounds_bytes,
        stage_metadata_bytes,
        completion_bytes,
        diagnostics_bytes,
        recorder_metadata_bytes,
    ]
    .into_iter()
    .try_fold(0_usize, |total, bytes| {
        total.checked_add(bytes).ok_or(InitError::SizeOverflow)
    })?;

    Ok(MemoryEstimate {
        matrix: matrix_bytes,
        calibration: calibration_bytes,
        online: online_bytes,
        bounds: bounds_bytes,
        stage_metadata: stage_metadata_bytes,
        completion: completion_bytes,
        diagnostics: diagnostics_bytes,
        recorder_metadata: recorder_metadata_bytes,
        total: total_bytes,
    })
}

fn checked_bytes(items: usize, item_bytes: usize) -> Result<usize, InitError> {
    items.checked_mul(item_bytes).ok_or(InitError::SizeOverflow)
}

const _: () = assert!(CALIBRATION_BUCKETS == 250);

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn registration_order_defines_dense_ids_and_cookie_brands_them() {
        let mut first = Tracegrams::builder();
        let first_a = first.stage("a").unwrap();
        let first_b = first.stage("b").unwrap();
        let mut second = Tracegrams::builder();
        let second_a = second.stage("a").unwrap();

        assert_eq!(first_a.index, 0);
        assert_eq!(first_b.index, 1);
        assert_eq!(second_a.index, 0);
        assert_ne!(first_a.cookie, second_a.cookie);
    }

    #[test]
    fn cookie_allocation_never_wraps() {
        let source = AtomicU64::new(MAX_REGISTRY_COOKIE);

        assert_eq!(
            claim_registry_cookie(&source),
            Some(RegistryCookie(u32::MAX))
        );
        assert_eq!(claim_registry_cookie(&source), None);
        assert_eq!(source.load(Ordering::Relaxed), MAX_REGISTRY_COOKIE + 1);
    }

    #[test]
    fn exhausted_builder_returns_the_typed_error() {
        let mut builder = TracegramsBuilder {
            cookie: None,
            stage_names: Vec::new(),
            tail_quantile: DEFAULT_TAIL_QUANTILE,
            memory_budget_bytes: DEFAULT_MEMORY_BUDGET_BYTES,
        };

        assert_eq!(
            builder.stage("stage"),
            Err(InitError::RegistryCookieExhausted)
        );
        assert!(matches!(
            builder.build(),
            Err(InitError::RegistryCookieExhausted)
        ));
    }

    #[test]
    fn artificial_metadata_size_overflow_is_typed() {
        let layout = FixedStorageLayout::new(1).unwrap();

        assert_eq!(
            estimate_memory(layout, [usize::MAX]),
            Err(InitError::SizeOverflow)
        );
    }

    #[test]
    fn storage_is_fixed_and_clones_share_the_same_recorder() {
        let mut builder = Tracegrams::builder();
        for name in ["parse", "db", "render"] {
            builder.stage(name).unwrap();
        }
        let tracegrams = builder.build().unwrap();
        let cloned = tracegrams.clone();

        assert!(Arc::ptr_eq(&tracegrams.inner, &cloned.inner));
        assert_eq!(tracegrams.inner.stage_names.len(), 3);
        assert_eq!(
            tracegrams.inner.default_bounds.len(),
            default_bounds().len()
        );
        assert_eq!(
            tracegrams.inner.calibration_bounds.len(),
            calibration_bounds().len()
        );
        assert_eq!(
            tracegrams.inner.counters.len(),
            tracegrams.inner.layout.counter_count()
        );
    }
}
