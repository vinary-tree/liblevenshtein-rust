//! Time series distance metrics and indexing.
//!
//! This module provides implementations for time series similarity measures,
//! particularly the Move-Split-Merge (MSM) metric, along with indexing structures
//! for efficient similarity search.
//!
//! # Move-Split-Merge (MSM) Metric
//!
//! The MSM metric, introduced by Stefan et al., is a metric for comparing time series
//! that is robust to time shifts and scaling. It defines three operations:
//!
//! - **Move**: Change a value by some amount. Cost = |change|
//! - **Split**: Duplicate a value into two consecutive identical elements. Cost = c (constant)
//! - **Merge**: Combine two consecutive equal-value elements into one. Cost = c (constant)
//!
//! ## Example
//!
//! ```rust
//! use liblevenshtein::time_series::MsmConfig;
//!
//! let config = MsmConfig::new(1.0);  // c = 1.0
//! let x = vec![1.0, 2.0, 3.0, 2.0];
//! let y = vec![1.0, 2.5, 2.0];
//!
//! let distance = config.distance(&x, &y);
//! println!("MSM distance: {}", distance);
//! ```
//!
//! # MSM Automaton
//!
//! In addition to the standard DP algorithm, this module provides an automaton-based
//! implementation of MSM. The automaton approach enables:
//!
//! - Early termination when cost exceeds threshold
//! - Future integration with trie-based time series indexing
//! - Alternative computational model for research purposes
//!
//! ## Automaton Example
//!
//! ```rust
//! use liblevenshtein::time_series::{MsmConfig, msm_distance_wavefront};
//!
//! let config = MsmConfig::new(1.0);
//! let x = vec![1.0, 2.0, 3.0];
//! let y = vec![1.5, 2.5, 3.5];
//!
//! // With threshold (returns None if distance exceeds threshold)
//! let distance = msm_distance_wavefront(&x, &y, &config, 2.0);
//! assert!(distance.is_some());
//! ```
//!
//! # Time Series Indexing
//!
//! The module provides trie-based indexing for efficient similarity search:
//!
//! - **Quantization**: Encode continuous values as discrete bins
//! - **Trie storage**: Efficient prefix-sharing using DynamicDawg
//! - **Hybrid search**: Approximate filtering + exact MSM verification
//!
//! ## Indexing Example
//!
//! ```rust
//! use liblevenshtein::time_series::{
//!     TimeSeriesIndex, HybridSearchIndex, QuantizationConfig, MsmConfig,
//! };
//!
//! // Simple quantized index
//! let config = QuantizationConfig::for_u8(0.0, 100.0);
//! let mut index = TimeSeriesIndex::from_series(config, &[
//!     vec![10.0, 20.0, 30.0],
//!     vec![15.0, 25.0, 35.0],
//! ]);
//!
//! // Approximate search
//! let results = index.search(&[12.0, 22.0, 32.0], 3);
//!
//! // Hybrid search with exact MSM verification
//! let quant_config = QuantizationConfig::for_u8(0.0, 100.0);
//! let msm_config = MsmConfig::new(1.0);
//! let mut hybrid = HybridSearchIndex::new(quant_config, msm_config);
//! hybrid.insert(0usize, &[10.0, 20.0, 30.0]);
//! let exact_results = hybrid.search_exact(&[12.0, 22.0, 32.0], 10.0);
//! ```
//!
//! # Encoding Options
//!
//! | Encoding | Precision | Use Case |
//! |----------|-----------|----------|
//! | Quantization (u8) | 256 levels | Fast approximate search |
//! | Quantization (u16) | 65K levels | Higher precision |
//! | Direct float bits | Exact | Exact matching |
//! | Delta encoding | Variable | Bounded local variation |
//! | SAX | Symbolic | Time series motifs |
//!
//! # Lower Bound Pruning
//!
//! For efficient search over large databases, lower bounds allow pruning
//! candidates without computing the expensive full MSM distance:
//!
//! ```rust
//! use liblevenshtein::time_series::{
//!     MsmConfig, euclidean_lb, length_lb, combined_lb,
//! };
//!
//! let x = vec![1.0, 2.0, 3.0, 4.0];
//! let y = vec![1.5, 2.5, 3.5, 4.5];
//! let c = 1.0;
//!
//! // Fast lower bounds
//! let lb_euclidean = euclidean_lb(&x, &y);
//! let lb_length = length_lb(&x, &y, c);
//! let lb_combined = combined_lb(&x, &y, c);
//!
//! // If lower bound exceeds threshold, skip expensive MSM computation
//! let threshold = 2.0;
//! if lb_combined > threshold {
//!     println!("Pruned: LB {} > threshold {}", lb_combined, threshold);
//! } else {
//!     let msm = MsmConfig::new(c).distance(&x, &y);
//!     println!("MSM distance: {}", msm);
//! }
//! ```
//!
//! # References
//!
//! - Stefan, Alexandra, et al. "The move-split-merge metric for time series."
//!   IEEE transactions on Knowledge and Data Engineering 25.6 (2012): 1425-1438.

mod encoding;
mod hybrid_search;
mod lower_bounds;
mod msm;
mod msm_position;
mod msm_state;
mod msm_transition;
mod trie_index;

// MSM metric exports
pub use msm::{MsmConfig, MsmResult};
pub use msm_position::{msm_subsumes, MsmPosition};
pub use msm_state::MsmState;
pub use msm_transition::{
    initial_msm_state, msm_distance_automaton, msm_distance_wavefront, transition_msm_position,
    transition_msm_state,
};

// Encoding exports
pub use encoding::QuantizationConfig;
pub use encoding::{delta_encoding, float_encoding, sax_encoding};

// Indexing exports
pub use hybrid_search::{HybridSearchIndex, HybridSearchIndexBuilder, HybridSearchStats};
pub use trie_index::{TimeSeriesIndex, TimeSeriesIndexBuilder, TimeSeriesIndexStats};

// Lower bound exports
#[cfg(feature = "rayon")]
pub use lower_bounds::search_with_lb_parallel;
pub use lower_bounds::{
    combined_lb, euclidean_lb, filter_by_lower_bound, l1_lb, length_lb, search_with_lb,
    search_with_lb_stats, LowerBoundConfig, LowerBoundStats, LowerBoundType,
};
