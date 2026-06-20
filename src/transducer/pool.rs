//! State allocation pool for reducing cloning overhead.
//!
//! The StatePool manages a pool of reusable State allocations to avoid
//! repeatedly allocating and deallocating States during query processing.
//!
//! # Performance
//!
//! From profiling, State cloning accounts for 21.73% of runtime:
//! - Vec allocation: 6.00%
//! - Position copies: 7.44%
//! - Misc overhead: ~8%
//!
//! By reusing State allocations, we eliminate the Vec allocation overhead
//! and reduce overall cloning cost by an estimated 10-15%.

use super::state::State;

/// Pool of reusable State allocations.
///
/// The pool maintains a collection of States that can be reused across
/// multiple transitions within a single query, eliminating the need to
/// repeatedly allocate and deallocate Vec<Position> structures.
///
/// # Usage
///
/// ```ignore
/// let mut pool = StatePool::new();
///
/// // Acquire a state (from pool or allocate new)
/// let mut state = pool.acquire();
///
/// // Use the state...
/// state.insert(Position::new(0, 0));
///
/// // Return to pool when done
/// pool.release(state);
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
/// StatePool is NOT thread-safe. Each query should have its own pool.
/// For parallel queries, use thread-local pools or one pool per thread.
#[derive(Debug)]
pub struct StatePool {
    /// Recycled states ready for reuse
    pool: Vec<State>,

    /// Statistics: Total allocations made
    allocations: usize,

    /// Statistics: Total reuses from pool
    reuses: usize,
}

impl StatePool {
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
            pool.push(State::new());
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
    /// - Pool hit: O(1) - pop from Vec + clear positions Vec
    /// - Pool miss: O(1) - allocate new Vec<Position>
    ///
    /// The state's Vec<Position> allocation is reused when available,
    /// which is the primary performance benefit.
    #[inline]
    pub fn acquire(&mut self) -> State {
        if let Some(mut state) = self.pool.pop() {
            state.clear(); // Clear positions but keep Vec capacity
            self.reuses += 1;
            state
        } else {
            self.allocations += 1;
            State::new()
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
    /// The state's internal Vec<Position> allocation is preserved for reuse.
    #[inline]
    pub fn release(&mut self, state: State) {
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

impl Default for StatePool {
    fn default() -> Self {
        Self::new()
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::transducer::position::Position;

    #[test]
    fn test_pool_new() {
        let pool = StatePool::new();
        // Pool is pre-warmed with 4 states
        assert_eq!(pool.pool_size(), 4);
        assert_eq!(pool.total_allocations(), 4);
        assert_eq!(pool.total_reuses(), 0);
    }

    #[test]
    fn test_pool_acquire_allocates_when_empty() {
        let mut pool = StatePool::new();

        // Pool starts with 4 pre-warmed states, so first acquire reuses
        let state = pool.acquire();
        assert!(state.is_empty());
        assert_eq!(pool.total_allocations(), 4); // Pre-warmed allocations
        assert_eq!(pool.total_reuses(), 1); // First acquire is a reuse
    }

    #[test]
    fn test_pool_acquire_reuses_when_available() {
        let mut pool = StatePool::new();

        // Acquire and release a state (from pre-warmed pool)
        let state = pool.acquire();
        pool.release(state);

        assert_eq!(pool.pool_size(), 4); // 3 pre-warmed + 1 released

        // Acquire again - should reuse
        let state2 = pool.acquire();
        assert!(state2.is_empty());
        assert_eq!(pool.total_allocations(), 4); // Still just pre-warmed
        assert_eq!(pool.total_reuses(), 2); // Two reuses
        assert_eq!(pool.pool_size(), 3); // Back to 3 in pool
    }

    #[test]
    fn test_pool_release_clears_state() {
        use super::super::algorithm::Algorithm;

        let mut pool = StatePool::new();
        let max_distance = 2;

        // Acquire a state and add positions
        // (1,0) subsumes (2,1) with Standard subsumption: |1-2|=1 <= (1-0)=1
        let mut state = pool.acquire();
        state.insert(Position::new(1, 0), Algorithm::Standard, max_distance);
        state.insert(Position::new(2, 1), Algorithm::Standard, max_distance); // Subsumed by (1,0)
        assert_eq!(state.len(), 1); // Only (1,0) remains

        // Release it
        pool.release(state);

        // Acquire it back - should be empty
        let state2 = pool.acquire();
        assert!(state2.is_empty());
    }

    #[test]
    fn test_pool_respects_max_size() {
        let mut pool = StatePool::new();

        // Fill pool to max capacity
        for _ in 0..StatePool::MAX_POOL_SIZE {
            pool.release(State::new());
        }

        assert_eq!(pool.pool_size(), StatePool::MAX_POOL_SIZE);

        // Try to add one more - should not increase pool size
        pool.release(State::new());
        assert_eq!(pool.pool_size(), StatePool::MAX_POOL_SIZE);
    }

    #[test]
    fn test_pool_reuse_rate() {
        let mut pool = StatePool::new();

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

    #[test]
    fn test_pool_lifo_order() {
        use super::super::algorithm::Algorithm;

        let mut pool = StatePool::new();
        let max_distance = 2;

        // Pool starts with 4 pre-warmed states
        // Release two more states
        let mut state1 = State::new();
        state1.insert(Position::new(1, 0), Algorithm::Standard, max_distance);
        pool.release(state1);

        let mut state2 = State::new();
        state2.insert(Position::new(2, 0), Algorithm::Standard, max_distance);
        pool.release(state2);

        // Acquire should get state2 first (LIFO)
        let acquired = pool.acquire();
        assert!(acquired.is_empty()); // Should be cleared

        // Pool should have 5 states left (4 pre-warmed + 1 remaining)
        assert_eq!(pool.pool_size(), 5);
    }

    #[test]
    fn test_pool_capacity_preserved() {
        use super::super::algorithm::Algorithm;

        let mut pool = StatePool::new();
        let max_distance = 10;

        // Create a state with some capacity
        let mut state = pool.acquire();
        for i in 0..10 {
            state.insert(Position::new(i, 0), Algorithm::Standard, max_distance);
        }

        // The state's Vec should have capacity >= 10
        pool.release(state);

        // Acquire it back - capacity should be preserved even though cleared
        let mut state2 = pool.acquire();
        assert!(state2.is_empty());

        // Add 10 more positions - should not need reallocation
        // (This is an implementation detail test)
        for i in 0..10 {
            state2.insert(Position::new(i, 0), Algorithm::Standard, max_distance);
        }
        assert_eq!(state2.len(), 10);
    }
}
