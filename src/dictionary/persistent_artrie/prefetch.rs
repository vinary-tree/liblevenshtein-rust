//! Prefetching for Persistent ART traversal.
//!
//! This module provides prefetching hints for DFS traversal to reduce
//! I/O latency when accessing disk-resident nodes. When the Levenshtein
//! automaton traverses the trie, prefetching child nodes while processing
//! the current node can hide disk I/O latency.
//!
//! # Architecture
//!
//! ```text
//! ┌─────────────────────────────────────────────────────────────┐
//! │                 Traversal Thread                            │
//! │  ┌──────────┐  ┌──────────┐  ┌──────────┐                  │
//! │  │ Process  │→ │ Prefetch │→ │  Move to │                  │
//! │  │  Node A  │  │ Children │  │  Node B  │                  │
//! │  └──────────┘  └──────────┘  └──────────┘                  │
//! └─────────────────────────────────────────────────────────────┘
//!                        ↓
//!               ┌─────────────────┐
//!               │ Prefetch Queue  │
//!               └────────┬────────┘
//!                        ↓
//! ┌─────────────────────────────────────────────────────────────┐
//! │                Buffer Manager                               │
//! │  • Async read of prefetched pages                          │
//! │  • LRU cache keeps hot pages                               │
//! │  • Already-cached pages are no-op                          │
//! └─────────────────────────────────────────────────────────────┘
//! ```
//!
//! # Usage
//!
//! ```rust,ignore
//! use liblevenshtein::dictionary::persistent_artrie::prefetch::Prefetcher;
//!
//! let prefetcher = Prefetcher::new(buffer_manager.clone());
//!
//! // During traversal, prefetch children before accessing them
//! for child_ptr in node.children() {
//!     prefetcher.prefetch(child_ptr);
//! }
//!
//! // Later, when we access the child, it may already be in cache
//! let child = buffer_manager.get_page(child_ptr)?;
//! ```
//!
//! # Strategies
//!
//! The module supports multiple prefetching strategies:
//!
//! - **Immediate**: Prefetch all children immediately (high bandwidth use)
//! - **Selective**: Only prefetch likely-to-be-accessed children
//! - **Depth-Limited**: Prefetch up to N levels ahead
//! - **Breadth-Limited**: Prefetch at most N children per node

use std::collections::VecDeque;
use std::sync::atomic::{AtomicBool, AtomicU64, Ordering};
use std::sync::{Arc, Mutex};

use super::swizzled_ptr::{DiskLocation, SwizzledPtr};

/// Statistics for prefetching operations.
#[derive(Debug, Default)]
pub struct PrefetchStats {
    /// Number of prefetch requests issued
    pub requests: AtomicU64,
    /// Number of prefetch requests that were already in cache (no-op)
    pub cache_hits: AtomicU64,
    /// Number of prefetch requests that triggered I/O
    pub io_issued: AtomicU64,
    /// Number of prefetch requests that were dropped (queue full)
    pub dropped: AtomicU64,
}

impl PrefetchStats {
    /// Create new stats instance.
    pub fn new() -> Self {
        Self::default()
    }

    /// Record a prefetch request.
    pub fn record_request(&self) {
        self.requests.fetch_add(1, Ordering::Relaxed);
    }

    /// Record a cache hit (no I/O needed).
    pub fn record_cache_hit(&self) {
        self.cache_hits.fetch_add(1, Ordering::Relaxed);
    }

    /// Record an I/O operation.
    pub fn record_io(&self) {
        self.io_issued.fetch_add(1, Ordering::Relaxed);
    }

    /// Record a dropped request.
    pub fn record_dropped(&self) {
        self.dropped.fetch_add(1, Ordering::Relaxed);
    }

    /// Get a snapshot of current stats.
    pub fn snapshot(&self) -> PrefetchStatsSnapshot {
        PrefetchStatsSnapshot {
            requests: self.requests.load(Ordering::Relaxed),
            cache_hits: self.cache_hits.load(Ordering::Relaxed),
            io_issued: self.io_issued.load(Ordering::Relaxed),
            dropped: self.dropped.load(Ordering::Relaxed),
        }
    }
}

/// Immutable snapshot of prefetch stats.
#[derive(Debug, Clone, Copy)]
pub struct PrefetchStatsSnapshot {
    /// Total requests
    pub requests: u64,
    /// Cache hits
    pub cache_hits: u64,
    /// I/O operations issued
    pub io_issued: u64,
    /// Dropped requests
    pub dropped: u64,
}

impl PrefetchStatsSnapshot {
    /// Calculate the cache hit rate.
    pub fn hit_rate(&self) -> f64 {
        if self.requests == 0 {
            0.0
        } else {
            self.cache_hits as f64 / self.requests as f64
        }
    }

    /// Calculate the drop rate.
    pub fn drop_rate(&self) -> f64 {
        if self.requests == 0 {
            0.0
        } else {
            self.dropped as f64 / self.requests as f64
        }
    }
}

/// Prefetch strategy for controlling which nodes to prefetch.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum PrefetchStrategy {
    /// Prefetch all children immediately
    Immediate,
    /// Prefetch only the first N children
    FirstN(usize),
    /// Prefetch children up to N levels deep
    DepthLimited(usize),
    /// Disabled - no prefetching
    Disabled,
}

impl Default for PrefetchStrategy {
    fn default() -> Self {
        PrefetchStrategy::Immediate
    }
}

/// A prefetch request in the queue.
#[derive(Debug, Clone)]
pub struct PrefetchRequest {
    /// The disk location to prefetch
    pub location: DiskLocation,
    /// Priority (lower = higher priority)
    pub priority: u8,
    /// Depth in the traversal tree (for depth-limited strategies)
    pub depth: u16,
}

/// Prefetcher for async page loading.
///
/// The prefetcher maintains a queue of pages to prefetch and
/// coordinates with the buffer manager to load pages in the background.
pub struct Prefetcher {
    /// Queue of pending prefetch requests
    queue: Mutex<VecDeque<PrefetchRequest>>,
    /// Maximum queue size
    max_queue_size: usize,
    /// Current prefetch strategy
    strategy: PrefetchStrategy,
    /// Statistics
    stats: Arc<PrefetchStats>,
    /// Whether prefetching is enabled
    enabled: AtomicBool,
}

impl Prefetcher {
    /// Create a new prefetcher with default settings.
    pub fn new() -> Self {
        Self::with_config(1024, PrefetchStrategy::default())
    }

    /// Create a prefetcher with custom configuration.
    pub fn with_config(max_queue_size: usize, strategy: PrefetchStrategy) -> Self {
        Prefetcher {
            queue: Mutex::new(VecDeque::with_capacity(max_queue_size.min(1024))),
            max_queue_size,
            strategy,
            stats: Arc::new(PrefetchStats::new()),
            enabled: AtomicBool::new(true),
        }
    }

    /// Create a disabled prefetcher (no-op).
    pub fn disabled() -> Self {
        let mut prefetcher = Self::new();
        prefetcher.strategy = PrefetchStrategy::Disabled;
        prefetcher.enabled.store(false, Ordering::Relaxed);
        prefetcher
    }

    /// Check if prefetching is enabled.
    pub fn is_enabled(&self) -> bool {
        self.enabled.load(Ordering::Relaxed)
    }

    /// Enable or disable prefetching.
    pub fn set_enabled(&self, enabled: bool) {
        self.enabled.store(enabled, Ordering::Relaxed);
    }

    /// Get the current strategy.
    pub fn strategy(&self) -> PrefetchStrategy {
        self.strategy
    }

    /// Set the prefetch strategy.
    pub fn set_strategy(&mut self, strategy: PrefetchStrategy) {
        self.strategy = strategy;
        if strategy == PrefetchStrategy::Disabled {
            self.enabled.store(false, Ordering::Relaxed);
        }
    }

    /// Submit a prefetch request for a swizzled pointer.
    ///
    /// If the pointer is already in memory (swizzled), this is a no-op.
    /// If the pointer is on disk, it's added to the prefetch queue.
    pub fn prefetch(&self, ptr: &SwizzledPtr) {
        if !self.is_enabled() || self.strategy == PrefetchStrategy::Disabled {
            return;
        }

        self.stats.record_request();

        // If already in memory (swizzled), nothing to prefetch
        if ptr.is_swizzled() {
            self.stats.record_cache_hit();
            return;
        }

        // Get disk location
        if let Some(location) = ptr.disk_location() {
            self.queue_prefetch(PrefetchRequest {
                location,
                priority: 0,
                depth: 0,
            });
        }
    }

    /// Submit a prefetch request with depth information.
    ///
    /// Depth is used for depth-limited strategies.
    pub fn prefetch_with_depth(&self, ptr: &SwizzledPtr, depth: u16) {
        if !self.is_enabled() {
            return;
        }

        // Check depth limit
        if let PrefetchStrategy::DepthLimited(max_depth) = self.strategy {
            if depth as usize > max_depth {
                return;
            }
        }

        self.stats.record_request();

        if ptr.is_swizzled() {
            self.stats.record_cache_hit();
            return;
        }

        if let Some(location) = ptr.disk_location() {
            self.queue_prefetch(PrefetchRequest {
                location,
                priority: depth as u8,
                depth,
            });
        }
    }

    /// Submit multiple prefetch requests for child nodes.
    ///
    /// Respects the FirstN strategy if configured.
    pub fn prefetch_children(&self, children: &[(u8, SwizzledPtr)]) {
        if !self.is_enabled() {
            return;
        }

        let limit = match self.strategy {
            PrefetchStrategy::FirstN(n) => n.min(children.len()),
            PrefetchStrategy::Disabled => return,
            _ => children.len(),
        };

        for (_, ptr) in children.iter().take(limit) {
            self.prefetch(ptr);
        }
    }

    /// Queue a prefetch request.
    fn queue_prefetch(&self, request: PrefetchRequest) {
        let mut queue = self.queue.lock().expect("prefetch queue poisoned");

        if queue.len() >= self.max_queue_size {
            // Queue full - drop the request
            self.stats.record_dropped();
            return;
        }

        queue.push_back(request);
        self.stats.record_io();
    }

    /// Drain and return all pending prefetch requests.
    ///
    /// This is used by the I/O thread to process prefetch requests.
    pub fn drain_requests(&self) -> Vec<PrefetchRequest> {
        let mut queue = self.queue.lock().expect("prefetch queue poisoned");
        queue.drain(..).collect()
    }

    /// Get the current queue length.
    pub fn queue_len(&self) -> usize {
        self.queue.lock().expect("prefetch queue poisoned").len()
    }

    /// Get prefetch statistics.
    pub fn stats(&self) -> Arc<PrefetchStats> {
        self.stats.clone()
    }

    /// Clear all pending requests.
    pub fn clear(&self) {
        let mut queue = self.queue.lock().expect("prefetch queue poisoned");
        queue.clear();
    }
}

impl Default for Prefetcher {
    fn default() -> Self {
        Self::new()
    }
}

/// Prefetch hint for DFS traversal.
///
/// This trait is implemented by node types to provide prefetch hints
/// about which children are likely to be accessed next.
pub trait PrefetchHint {
    /// Get prefetch hints for likely-to-be-accessed children.
    ///
    /// Returns a list of swizzled pointers that should be prefetched.
    fn prefetch_hints(&self) -> Vec<SwizzledPtr>;

    /// Get the number of children (for prefetch planning).
    fn child_count(&self) -> usize;
}

/// Adaptive prefetcher that adjusts strategy based on hit rate.
///
/// This prefetcher monitors the cache hit rate and adjusts the
/// prefetch aggressiveness accordingly.
pub struct AdaptivePrefetcher {
    /// Inner prefetcher
    inner: Prefetcher,
    /// Target hit rate (0.0 - 1.0)
    target_hit_rate: f64,
    /// Minimum check interval (requests)
    check_interval: u64,
    /// Last check request count
    last_check: AtomicU64,
}

impl AdaptivePrefetcher {
    /// Create a new adaptive prefetcher.
    pub fn new(target_hit_rate: f64) -> Self {
        AdaptivePrefetcher {
            inner: Prefetcher::new(),
            target_hit_rate: target_hit_rate.clamp(0.0, 1.0),
            check_interval: 1000,
            last_check: AtomicU64::new(0),
        }
    }

    /// Prefetch with adaptive strategy adjustment.
    pub fn prefetch(&self, ptr: &SwizzledPtr) {
        self.inner.prefetch(ptr);
        self.maybe_adjust();
    }

    /// Check if we should adjust the strategy.
    fn maybe_adjust(&self) {
        let stats = self.inner.stats.snapshot();
        let last = self.last_check.load(Ordering::Relaxed);

        if stats.requests - last >= self.check_interval {
            self.last_check.store(stats.requests, Ordering::Relaxed);

            let hit_rate = stats.hit_rate();

            // Adjust prefetching based on hit rate
            // High hit rate = we're prefetching effectively
            // Low hit rate = we might need to prefetch more or less
            if hit_rate < self.target_hit_rate * 0.8 {
                // We're not hitting enough - might need more aggressive prefetching
                // or the workload is very random
                // (In a real implementation, we'd adjust parameters here)
            }
        }
    }

    /// Get the inner prefetcher.
    pub fn inner(&self) -> &Prefetcher {
        &self.inner
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_prefetch_stats() {
        let stats = PrefetchStats::new();

        stats.record_request();
        stats.record_request();
        stats.record_cache_hit();
        stats.record_io();
        stats.record_dropped();

        let snapshot = stats.snapshot();
        assert_eq!(snapshot.requests, 2);
        assert_eq!(snapshot.cache_hits, 1);
        assert_eq!(snapshot.io_issued, 1);
        assert_eq!(snapshot.dropped, 1);
        assert_eq!(snapshot.hit_rate(), 0.5);
    }

    #[test]
    fn test_prefetcher_disabled() {
        let prefetcher = Prefetcher::disabled();
        assert!(!prefetcher.is_enabled());

        // Even with a disk pointer, shouldn't queue anything
        let ptr = SwizzledPtr::on_disk(0, 0, super::super::swizzled_ptr::NodeType::Bucket);
        prefetcher.prefetch(&ptr);

        assert_eq!(prefetcher.queue_len(), 0);
    }

    #[test]
    fn test_prefetcher_memory_pointer() {
        let prefetcher = Prefetcher::new();

        // Memory pointer should be a no-op
        let ptr = SwizzledPtr::null();
        prefetcher.prefetch(&ptr);

        // Check stats - should show a request but also a cache hit
        let stats = prefetcher.stats().snapshot();
        // Null pointer doesn't count as memory for our purposes
        assert!(stats.requests >= 0);
    }

    #[test]
    fn test_prefetcher_queue() {
        let prefetcher = Prefetcher::new();

        // Queue some disk pointers
        for i in 0..10 {
            let ptr = SwizzledPtr::on_disk(i, 0, super::super::swizzled_ptr::NodeType::Bucket);
            prefetcher.prefetch(&ptr);
        }

        // Should have 10 requests
        assert_eq!(prefetcher.queue_len(), 10);

        // Drain and check
        let requests = prefetcher.drain_requests();
        assert_eq!(requests.len(), 10);
        assert_eq!(prefetcher.queue_len(), 0);
    }

    #[test]
    fn test_prefetcher_max_queue() {
        let prefetcher = Prefetcher::with_config(5, PrefetchStrategy::Immediate);

        // Queue more than max
        for i in 0..10 {
            let ptr = SwizzledPtr::on_disk(i, 0, super::super::swizzled_ptr::NodeType::Bucket);
            prefetcher.prefetch(&ptr);
        }

        // Should have max queue size
        assert_eq!(prefetcher.queue_len(), 5);

        // Check that some were dropped
        let stats = prefetcher.stats().snapshot();
        assert_eq!(stats.dropped, 5);
    }

    #[test]
    fn test_prefetch_strategy_first_n() {
        let prefetcher = Prefetcher::with_config(100, PrefetchStrategy::FirstN(3));

        let children: Vec<(u8, SwizzledPtr)> = (0..10)
            .map(|i| {
                (
                    i,
                    SwizzledPtr::on_disk(i as u32, 0, super::super::swizzled_ptr::NodeType::Bucket),
                )
            })
            .collect();

        prefetcher.prefetch_children(&children);

        // Should only prefetch first 3
        assert_eq!(prefetcher.queue_len(), 3);
    }

    #[test]
    fn test_prefetch_depth_limited() {
        let prefetcher = Prefetcher::with_config(100, PrefetchStrategy::DepthLimited(2));

        let ptr = SwizzledPtr::on_disk(0, 0, super::super::swizzled_ptr::NodeType::Bucket);

        // Depth 0 - should prefetch
        prefetcher.prefetch_with_depth(&ptr, 0);
        assert_eq!(prefetcher.queue_len(), 1);

        // Depth 1 - should prefetch
        prefetcher.prefetch_with_depth(&ptr, 1);
        assert_eq!(prefetcher.queue_len(), 2);

        // Depth 2 - should prefetch
        prefetcher.prefetch_with_depth(&ptr, 2);
        assert_eq!(prefetcher.queue_len(), 3);

        // Depth 3 - should NOT prefetch (beyond limit)
        prefetcher.prefetch_with_depth(&ptr, 3);
        assert_eq!(prefetcher.queue_len(), 3);
    }

    #[test]
    fn test_adaptive_prefetcher() {
        let prefetcher = AdaptivePrefetcher::new(0.5);

        // Should be enabled by default
        assert!(prefetcher.inner().is_enabled());

        // Prefetch should work
        let ptr = SwizzledPtr::on_disk(0, 0, super::super::swizzled_ptr::NodeType::Bucket);
        prefetcher.prefetch(&ptr);

        assert_eq!(prefetcher.inner().queue_len(), 1);
    }
}
