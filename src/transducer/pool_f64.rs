//! State allocation pool for float-weighted automata.
//!
//! This module provides `StatePoolF64`, a pool of reusable `StateF64` allocations
//! to avoid repeatedly allocating and deallocating states during query processing.
//!
//! # Performance
//!
//! [`StateF64`](super::StateF64) keeps typical position sets inline in a
//! `SmallVec`. Pooling is most valuable when a large frontier spills beyond
//! that inline capacity: clearing and retaining the state preserves its heap
//! allocation for a later transition.

use super::state_f64::StateF64;

/// Pool of reusable StateF64 allocations.
///
/// The pool maintains a collection of `StateF64` instances that can be reused
/// across multiple transitions within a single query, eliminating the need
/// to repeatedly allocate and discard spilled `SmallVec` storage.
///
/// # Usage
///
/// ```rust
/// use liblevenshtein::transducer::{Algorithm, PositionF64, StatePoolF64};
///
/// let mut pool = StatePoolF64::new();
/// assert_eq!(pool.pool_size(), 4); // Pre-warmed states.
///
/// // Acquire a state (from pool or allocate new)
/// let mut state = pool.acquire();
/// assert!(state.insert(
///     PositionF64::new(0, 0.0),
///     Algorithm::Standard,
///     4,
///     1.0,
/// ));
///
/// // Return to pool when done
/// pool.release(state);
/// let state = pool.acquire();
/// assert!(state.is_empty()); // `acquire` clears retained positions.
/// assert_eq!(pool.total_reuses(), 2);
/// ```
///
/// # Pool Management
///
/// - **Initial capacity:** 16 states
/// - **Maximum capacity:** 32 states (to prevent unbounded growth)
/// - **Reuse strategy:** LIFO (last-in, first-out) for cache locality
///
/// # Thread Safety
///
/// StatePoolF64 is NOT thread-safe. Each query should have its own pool.
/// For parallel queries, use thread-local pools or one pool per thread.
#[derive(Debug)]
pub struct StatePoolF64 {
    /// Recycled states ready for reuse
    pool: Vec<StateF64>,

    /// Statistics: Total allocations made
    allocations: usize,

    /// Statistics: Total reuses from pool
    reuses: usize,
}

impl StatePoolF64 {
    /// Maximum number of states to keep in the pool
    const MAX_POOL_SIZE: usize = 32;

    /// Initial capacity hint for the pool
    const INITIAL_CAPACITY: usize = 16;

    /// Create a new state pool with pre-warmed states.
    ///
    /// The pool is pre-warmed with 4 states to avoid initial allocation overhead.
    /// Based on profiling, most queries acquire 2-4 states during traversal.
    /// Pre-warming eliminates the cold-start penalty for the first few transitions.
    pub fn new() -> Self {
        const PREWARM_SIZE: usize = 4;
        let mut pool = Vec::with_capacity(Self::INITIAL_CAPACITY);

        // Pre-allocate states to avoid cold-start penalty
        for _ in 0..PREWARM_SIZE {
            pool.push(StateF64::new());
        }

        Self {
            pool,
            allocations: PREWARM_SIZE, // Count pre-warmed allocations
            reuses: 0,
        }
    }

    /// Acquire a state from the pool.
    ///
    /// If the pool has a recycled state available, it will be cleared and
    /// returned. Otherwise, a new state is allocated.
    ///
    /// # Performance
    ///
    /// - Pool hit: O(1) - pop from `Vec` + clear positions
    /// - Pool miss: O(1) - construct an inline-empty `SmallVec`
    ///
    /// Any spilled `SmallVec` allocation is reused when available, which is
    /// the primary performance benefit for large frontiers.
    #[inline]
    pub fn acquire(&mut self) -> StateF64 {
        if let Some(mut state) = self.pool.pop() {
            state.clear();
            self.reuses += 1;
            state
        } else {
            self.allocations += 1;
            StateF64::new()
        }
    }

    /// Release a state back to the pool for future reuse.
    ///
    /// If the pool is at maximum capacity, the state is dropped instead
    /// of being retained. This prevents unbounded memory growth.
    ///
    /// # Performance
    ///
    /// - O(1) - push to Vec (unless pool is full, then drop)
    ///
    /// The state's spilled `SmallVec` allocation is preserved for reuse.
    #[inline]
    pub fn release(&mut self, state: StateF64) {
        if self.pool.len() < Self::MAX_POOL_SIZE {
            self.pool.push(state);
        }
        // Otherwise drop the state (let it deallocate)
    }

    /// Get the current pool size (number of states available for reuse).
    pub fn pool_size(&self) -> usize {
        self.pool.len()
    }

    /// Get the total number of allocations made.
    pub fn total_allocations(&self) -> usize {
        self.allocations
    }

    /// Get the total number of reuses from the pool.
    pub fn total_reuses(&self) -> usize {
        self.reuses
    }

    /// Get the reuse rate as a percentage (0.0 to 1.0).
    ///
    /// Returns the ratio of reuses to total acquires.
    /// A higher rate indicates better pool efficiency.
    pub fn reuse_rate(&self) -> f64 {
        let total_acquires = self.allocations + self.reuses;
        if total_acquires == 0 {
            0.0
        } else {
            self.reuses as f64 / total_acquires as f64
        }
    }
}

impl Default for StatePoolF64 {
    fn default() -> Self {
        Self::new()
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::transducer::{Algorithm, PositionF64};

    #[test]
    fn test_pool_new() {
        let pool = StatePoolF64::new();
        // Pool is pre-warmed with 4 states
        assert_eq!(pool.pool_size(), 4);
        assert_eq!(pool.total_allocations(), 4);
        assert_eq!(pool.total_reuses(), 0);
    }

    #[test]
    fn test_pool_acquire_reuses_when_available() {
        let mut pool = StatePoolF64::new();

        // First acquire - reuse from pre-warmed
        let state = pool.acquire();
        assert!(state.is_empty());
        assert_eq!(pool.total_reuses(), 1);

        pool.release(state);
        assert_eq!(pool.pool_size(), 4);

        // Second acquire - reuse
        let state2 = pool.acquire();
        assert!(state2.is_empty());
        assert_eq!(pool.total_reuses(), 2);
    }

    #[test]
    fn test_pool_release_clears_state() {
        let mut pool = StatePoolF64::new();
        let query_length = 4;

        // Acquire a state and add positions
        let mut state = pool.acquire();
        state.insert(
            PositionF64::new(1, 0.0),
            Algorithm::Standard,
            query_length,
            1.0,
        );
        assert_eq!(state.len(), 1);

        // Release it
        pool.release(state);

        // Acquire it back - should be empty
        let state2 = pool.acquire();
        assert!(state2.is_empty());
    }

    #[test]
    fn test_pool_respects_max_size() {
        let mut pool = StatePoolF64::new();

        // Fill pool to max capacity
        for _ in 0..StatePoolF64::MAX_POOL_SIZE {
            pool.release(StateF64::new());
        }

        assert_eq!(pool.pool_size(), StatePoolF64::MAX_POOL_SIZE);

        // Try to add one more - should not increase pool size
        pool.release(StateF64::new());
        assert_eq!(pool.pool_size(), StatePoolF64::MAX_POOL_SIZE);
    }

    #[test]
    fn test_pool_reuse_rate() {
        let mut pool = StatePoolF64::new();

        // Pool pre-warmed with 4, but no acquires yet
        assert_eq!(pool.reuse_rate(), 0.0);

        // First acquire - reuse from pre-warmed
        let state1 = pool.acquire();
        pool.release(state1);

        // Second acquire - reuse
        let _state2 = pool.acquire();

        // 4 allocations + 2 reuses = 2/6 = 33.3% reuse rate
        let expected = 2.0 / 6.0;
        assert!((pool.reuse_rate() - expected).abs() < 1e-6);
    }
}
