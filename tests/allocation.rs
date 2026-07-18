//! Allocation-counting contract test for the core hot plane.

#![allow(unsafe_code)]

use std::alloc::{GlobalAlloc, Layout, System};
use std::sync::atomic::{AtomicBool, AtomicUsize, Ordering};
use std::time::Duration;

use tracegrams::{FreezeCriteria, Outcome, StageId, Tracegrams};

struct CountingAllocator;

static COUNTING: AtomicBool = AtomicBool::new(false);
static ALLOCATIONS: AtomicUsize = AtomicUsize::new(0);

#[global_allocator]
static ALLOCATOR: CountingAllocator = CountingAllocator;

// SAFETY: every operation forwards the original allocator contract to `System`.
unsafe impl GlobalAlloc for CountingAllocator {
    unsafe fn alloc(&self, layout: Layout) -> *mut u8 {
        count_allocation();
        // SAFETY: forwarding the allocator contract unchanged to `System`.
        unsafe { System.alloc(layout) }
    }

    unsafe fn alloc_zeroed(&self, layout: Layout) -> *mut u8 {
        count_allocation();
        // SAFETY: forwarding the allocator contract unchanged to `System`.
        unsafe { System.alloc_zeroed(layout) }
    }

    unsafe fn realloc(&self, pointer: *mut u8, layout: Layout, size: usize) -> *mut u8 {
        count_allocation();
        // SAFETY: forwarding the allocator contract unchanged to `System`.
        unsafe { System.realloc(pointer, layout, size) }
    }

    unsafe fn dealloc(&self, pointer: *mut u8, layout: Layout) {
        // SAFETY: forwarding the allocator contract unchanged to `System`.
        unsafe { System.dealloc(pointer, layout) }
    }
}

fn count_allocation() {
    if COUNTING.load(Ordering::Relaxed) {
        ALLOCATIONS.fetch_add(1, Ordering::Relaxed);
    }
}

fn allocations_during<T>(operation: impl FnOnce() -> T) -> (T, usize) {
    ALLOCATIONS.store(0, Ordering::SeqCst);
    assert!(!COUNTING.swap(true, Ordering::SeqCst));
    let result = operation();
    COUNTING.store(false, Ordering::SeqCst);
    (result, ALLOCATIONS.load(Ordering::SeqCst))
}

fn one_stage_recorder() -> (Tracegrams, StageId) {
    let mut builder = Tracegrams::builder();
    let stage = builder.stage("stage").unwrap();
    (builder.build().unwrap(), stage)
}

fn frozen_recorder() -> (Tracegrams, StageId) {
    let (tracegrams, stage) = one_stage_recorder();
    tracegrams.record_elapsed(
        &mut tracegrams.start_manual(),
        stage,
        Duration::from_nanos(1_000),
    );
    tracegrams
        .try_freeze_calibration(FreezeCriteria::all_stages(1))
        .unwrap();
    (tracegrams, stage)
}

fn assert_zero_allocations_for_hot_plane(tracegrams: &Tracegrams, stage: StageId) {
    let (mut manual, allocations) = allocations_during(|| tracegrams.start_manual());
    assert_eq!(allocations, 0, "manual request start allocated");

    let ((), allocations) = allocations_during(|| {
        tracegrams.record_elapsed(&mut manual, stage, Duration::from_nanos(1_000));
    });
    assert_eq!(allocations, 0, "manual checkpoint allocated");

    let manual = tracegrams.start_manual();
    let ((), allocations) = allocations_during(|| {
        tracegrams.finish_manual(manual, stage, Duration::from_nanos(1_000), Outcome::Success);
    });
    assert_eq!(allocations, 0, "manual finish checkpoint allocated");

    let (mut clocked, allocations) = allocations_during(|| tracegrams.start());
    assert_eq!(allocations, 0, "clocked request start allocated");

    let ((), allocations) = allocations_during(|| tracegrams.mark(&mut clocked, stage));
    assert_eq!(allocations, 0, "clocked checkpoint allocated");

    let clocked = tracegrams.start();
    let ((), allocations) =
        allocations_during(|| tracegrams.finish(clocked, stage, Outcome::Success));
    assert_eq!(allocations, 0, "clocked finish checkpoint allocated");
}

#[test]
fn core_starts_and_checkpoints_allocate_nothing_after_build() {
    let (collecting, collecting_stage) = one_stage_recorder();
    assert_zero_allocations_for_hot_plane(&collecting, collecting_stage);

    let (frozen, frozen_stage) = frozen_recorder();
    assert_zero_allocations_for_hot_plane(&frozen, frozen_stage);
}
