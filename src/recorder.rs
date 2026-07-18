//! Checkpoint counter primitives.

use std::sync::atomic::{AtomicU64, Ordering};

pub(crate) fn increment_counter(counter: &AtomicU64, overflows: &AtomicU64) {
    if !increment_saturating(counter) {
        increment_saturating(overflows);
    }
}

fn increment_saturating(counter: &AtomicU64) -> bool {
    let mut current = counter.load(Ordering::Relaxed);
    loop {
        let Some(next) = current.checked_add(1) else {
            return false;
        };
        match counter.compare_exchange_weak(current, next, Ordering::Relaxed, Ordering::Relaxed) {
            Ok(_) => return true,
            Err(observed) => current = observed,
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn counter_saturates_and_accounts_for_every_overflow() {
        let counter = AtomicU64::new(u64::MAX - 1);
        let overflows = AtomicU64::new(0);

        increment_counter(&counter, &overflows);
        assert_eq!(counter.load(Ordering::Relaxed), u64::MAX);
        assert_eq!(overflows.load(Ordering::Relaxed), 0);

        increment_counter(&counter, &overflows);
        increment_counter(&counter, &overflows);
        assert_eq!(counter.load(Ordering::Relaxed), u64::MAX);
        assert_eq!(overflows.load(Ordering::Relaxed), 2);
    }

    #[test]
    fn overflow_accounting_counter_cannot_wrap() {
        let counter = AtomicU64::new(u64::MAX);
        let overflows = AtomicU64::new(u64::MAX);

        increment_counter(&counter, &overflows);
        assert_eq!(counter.load(Ordering::Relaxed), u64::MAX);
        assert_eq!(overflows.load(Ordering::Relaxed), u64::MAX);
    }
}
