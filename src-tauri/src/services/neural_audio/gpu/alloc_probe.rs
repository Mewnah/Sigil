//! Test-only heap-allocation counter (plan T2: "the GPU hot path makes zero heap allocations per
//! block, checked by a counting-allocator test").
//!
//! This is the crate's **only** `#[global_allocator]` in test builds (a binary can have one): it
//! forwards to the system allocator and counts allocations made on the calling thread while
//! [`allocations_during`] runs. Other hot-path tests (the benchmark's allocation-free check, P3-F)
//! reuse [`allocations_during`] instead of declaring a second allocator. Only Rust heap allocations
//! are seen; ONNX Runtime's C++ allocations go to the CRT heap directly. So a count of 0 proves "no
//! Rust heap allocation per tick", not "no host allocation": every path that calls ONNX Runtime per
//! tick (`RunWithBinding` on the bound path) still allocates inside it, and only a CUDA graph replay
//! (`gpu::plan`'s device path, no ONNX Runtime call per tick) is free of host allocations.

use std::alloc::{GlobalAlloc, Layout, System};
use std::cell::Cell;

struct CountingAllocator;

thread_local! {
    static COUNTING: Cell<bool> = const { Cell::new(false) };
    static COUNT: Cell<u64> = const { Cell::new(0) };
}

fn note_allocation() {
    // `try_with`: never panics while thread-local storage is being torn down.
    let _ = COUNTING.try_with(|on| {
        if on.get() {
            let _ = COUNT.try_with(|n| n.set(n.get() + 1));
        }
    });
}

// SAFETY: every call is forwarded unchanged to the system allocator; counting touches only
// const-initialised thread-locals (no allocation, no re-entrancy).
unsafe impl GlobalAlloc for CountingAllocator {
    unsafe fn alloc(&self, layout: Layout) -> *mut u8 {
        note_allocation();
        // SAFETY: forwarded contract.
        unsafe { System.alloc(layout) }
    }

    unsafe fn alloc_zeroed(&self, layout: Layout) -> *mut u8 {
        note_allocation();
        // SAFETY: forwarded contract.
        unsafe { System.alloc_zeroed(layout) }
    }

    unsafe fn realloc(&self, ptr: *mut u8, layout: Layout, new_size: usize) -> *mut u8 {
        note_allocation();
        // SAFETY: forwarded contract.
        unsafe { System.realloc(ptr, layout, new_size) }
    }

    unsafe fn dealloc(&self, ptr: *mut u8, layout: Layout) {
        // SAFETY: forwarded contract.
        unsafe { System.dealloc(ptr, layout) }
    }
}

#[global_allocator]
static GLOBAL: CountingAllocator = CountingAllocator;

/// Run `f` and return its result with the number of heap allocations (and reallocations) it made on
/// this thread.
pub(crate) fn allocations_during<R>(f: impl FnOnce() -> R) -> (R, u64) {
    COUNT.with(|n| n.set(0));
    COUNTING.with(|on| on.set(true));
    let r = f();
    COUNTING.with(|on| on.set(false));
    (r, COUNT.with(Cell::get))
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn counts_this_threads_allocations_only() {
        let ((), n) = allocations_during(|| {
            let v: Vec<u8> = Vec::with_capacity(64);
            std::hint::black_box(&v);
        });
        assert_eq!(n, 1);
        let ((), n) = allocations_during(|| {
            let x = std::hint::black_box(3u64) * 7;
            std::hint::black_box(x);
        });
        assert_eq!(n, 0);
        // Another thread's allocations are not counted.
        let ((), n) = allocations_during(|| {
            std::thread::scope(|s| {
                s.spawn(|| std::hint::black_box(vec![0u8; 1000]));
            });
        });
        // Spawning the scoped thread itself allocates on this thread; the vector does not count.
        assert!(n < 10, "{n}");
    }
}
