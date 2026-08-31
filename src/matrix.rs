//! Checked indexing for the fixed shared counter storage.

use std::mem::size_of;
use std::sync::atomic::AtomicU64;

use crate::bucket::BUCKETS;

const MATRIX_CELLS: usize = BUCKETS * BUCKETS;

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub(crate) struct StorageLayout {
    stage_count: usize,
    distribution_cells: usize,
    cause_start: usize,
    incoming_start: usize,
    counter_count: usize,
}

impl StorageLayout {
    pub(crate) fn new(stage_count: usize) -> Option<Self> {
        if stage_count == 0 {
            return None;
        }

        let distribution_cells = stage_count.checked_mul(BUCKETS)?;
        let cause_cells = stage_count.checked_mul(MATRIX_CELLS)?;
        let incoming_cells = stage_count.checked_sub(1)?.checked_mul(MATRIX_CELLS)?;
        let cause_start = distribution_cells.checked_mul(2)?;
        let incoming_start = cause_start.checked_add(cause_cells)?;
        let counter_count = incoming_start.checked_add(incoming_cells)?;

        Some(Self {
            stage_count,
            distribution_cells,
            cause_start,
            incoming_start,
            counter_count,
        })
    }

    pub(crate) const fn counter_count(self) -> usize {
        self.counter_count
    }

    pub(crate) fn counter_bytes(self) -> Option<usize> {
        self.counter_count.checked_mul(size_of::<AtomicU64>())
    }

    pub(crate) fn local(self, stage: usize, bucket: usize) -> Option<usize> {
        self.distribution_index(0, stage, bucket)
    }

    pub(crate) fn local_index(self, stage: usize, bucket: usize) -> usize {
        self.distribution_index_unchecked(0, stage, bucket)
    }

    pub(crate) fn cumulative(self, stage: usize, bucket: usize) -> Option<usize> {
        self.distribution_index(self.distribution_cells, stage, bucket)
    }

    pub(crate) fn cumulative_index(self, stage: usize, bucket: usize) -> usize {
        self.distribution_index_unchecked(self.distribution_cells, stage, bucket)
    }

    pub(crate) fn cause(
        self,
        destination: usize,
        previous_cumulative: usize,
        local: usize,
    ) -> Option<usize> {
        self.matrix_index(self.cause_start, destination, previous_cumulative, local)
    }

    pub(crate) fn cause_index(
        self,
        destination: usize,
        previous_cumulative: usize,
        local: usize,
    ) -> usize {
        self.matrix_index_unchecked(self.cause_start, destination, previous_cumulative, local)
    }

    pub(crate) fn incoming(
        self,
        destination: usize,
        previous_cumulative: usize,
        cumulative_after: usize,
    ) -> Option<usize> {
        // The first registered stage cannot have a predecessor, so incoming
        // storage omits its matrix.
        let matrix = destination.checked_sub(1)?;
        if destination >= self.stage_count {
            return None;
        }
        self.matrix_index(
            self.incoming_start,
            matrix,
            previous_cumulative,
            cumulative_after,
        )
    }

    pub(crate) fn incoming_index(
        self,
        destination: usize,
        previous_cumulative: usize,
        cumulative_after: usize,
    ) -> usize {
        debug_assert!(destination > 0);
        self.matrix_index_unchecked(
            self.incoming_start,
            destination - 1,
            previous_cumulative,
            cumulative_after,
        )
    }

    fn distribution_index(self, start: usize, stage: usize, bucket: usize) -> Option<usize> {
        if stage >= self.stage_count || bucket >= BUCKETS {
            return None;
        }
        start
            .checked_add(stage.checked_mul(BUCKETS)?)?
            .checked_add(bucket)
    }

    fn distribution_index_unchecked(self, start: usize, stage: usize, bucket: usize) -> usize {
        debug_assert!(stage < self.stage_count);
        debug_assert!(bucket < BUCKETS);
        start + stage * BUCKETS + bucket
    }

    fn matrix_index(self, start: usize, matrix: usize, row: usize, column: usize) -> Option<usize> {
        if matrix >= self.stage_count || row >= BUCKETS || column >= BUCKETS {
            return None;
        }
        start
            .checked_add(matrix.checked_mul(MATRIX_CELLS)?)?
            .checked_add(row.checked_mul(BUCKETS)?)?
            .checked_add(column)
    }

    fn matrix_index_unchecked(
        self,
        start: usize,
        matrix: usize,
        row: usize,
        column: usize,
    ) -> usize {
        debug_assert!(matrix < self.stage_count);
        debug_assert!(row < BUCKETS);
        debug_assert!(column < BUCKETS);
        start + matrix * MATRIX_CELLS + row * BUCKETS + column
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn offsets_are_disjoint_and_cover_the_storage_exactly_once() {
        let stages = 3;
        let layout = StorageLayout::new(stages).unwrap();
        let mut visits = vec![0_u8; layout.counter_count()];
        let mut visit = |offset: usize| visits[offset] += 1;

        for stage in 0..stages {
            for bucket in 0..BUCKETS {
                visit(layout.local(stage, bucket).unwrap());
                visit(layout.cumulative(stage, bucket).unwrap());
            }
            for previous in 0..BUCKETS {
                for local in 0..BUCKETS {
                    visit(layout.cause(stage, previous, local).unwrap());
                }
            }
        }
        for destination in 1..stages {
            for previous in 0..BUCKETS {
                for cumulative_after in 0..BUCKETS {
                    visit(
                        layout
                            .incoming(destination, previous, cumulative_after)
                            .unwrap(),
                    );
                }
            }
        }

        assert!(visits.iter().all(|visits| *visits == 1));
    }

    #[test]
    fn cause_matrices_are_indexed_by_destination() {
        let layout = StorageLayout::new(3).unwrap();

        assert_eq!(
            layout.cause(2, 0, 0).unwrap() - layout.cause(1, 0, 0).unwrap(),
            BUCKETS * BUCKETS
        );
    }

    #[test]
    fn incoming_matrices_are_indexed_by_destination_minus_one() {
        let layout = StorageLayout::new(3).unwrap();

        assert_eq!(layout.incoming(0, 0, 0), None);
        assert_eq!(
            layout.incoming(2, 0, 0).unwrap() - layout.incoming(1, 0, 0).unwrap(),
            BUCKETS * BUCKETS
        );
        assert_eq!(layout.incoming(3, 0, 0), None);
    }

    #[test]
    fn checked_layout_arithmetic_covers_the_maximum_stage_count() {
        let six = StorageLayout::new(6).unwrap();
        assert_eq!(six.counter_count(), 45_824);
        assert_eq!(six.counter_bytes(), Some(366_592));

        let max = StorageLayout::new(64).unwrap();
        assert_eq!(max.counter_count(), 528_384);
        assert_eq!(max.counter_bytes(), Some(4_227_072));
        assert_eq!(StorageLayout::new(0), None);
        assert_eq!(StorageLayout::new(usize::MAX), None);
    }

    #[test]
    fn every_index_dimension_is_checked() {
        let layout = StorageLayout::new(2).unwrap();

        assert_eq!(layout.local(2, 0), None);
        assert_eq!(layout.local(0, BUCKETS), None);
        assert_eq!(layout.cumulative(2, 0), None);
        assert_eq!(layout.cause(2, 0, 0), None);
        assert_eq!(layout.cause(0, BUCKETS, 0), None);
        assert_eq!(layout.cause(0, 0, BUCKETS), None);
        assert_eq!(layout.incoming(1, BUCKETS, 0), None);
        assert_eq!(layout.incoming(1, 0, BUCKETS), None);
    }
}
