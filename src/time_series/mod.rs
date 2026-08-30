//! Time series distance metrics and indexing.
//!
//! This module provides implementations for time series similarity measures,
//! particularly Move-Split-Merge (MSM), Edit distance with Real Penalty (ERP),
//! Time Warp Edit Distance (TWED), discrete Fréchet, and explicitly banded
//! Dynamic Time Warping (DTW), along with indexing structures for efficient
//! similarity search.
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
//! # Edit distance with Real Penalty
//!
//! ERP uses absolute match costs and charges unmatched samples relative to a
//! fixed real gap value. [`crate::time_series::ErpTransducer`] supplies exact range and nearest
//! neighbour search over the same generic quantized-trie walker as MSM.
//!
//! ```rust
//! use liblevenshtein::time_series::{
//!     ErpConfig, ErpTransducer, QuantizationConfig,
//! };
//!
//! let references = vec![vec![1.0, 2.0], vec![1.0, 0.0, 2.0]];
//! let index = ErpTransducer::from_series(
//!     QuantizationConfig::for_u8(-10.0, 10.0),
//!     ErpConfig::new(0.0),
//!     &references,
//! );
//! assert_eq!(index.search_range(&[1.0, 2.0], 0.0).len(), 2);
//! ```
//!
//! # Time Warp Edit Distance
//!
//! TWED edits adjacent sample segments and penalizes temporal displacement.
//! [`crate::time_series::TwedConfig`] exposes the complete non-negative
//! parameter family, including the non-metric `nu = 0` regime. Validate
//! `nu > 0` with [`crate::time_series::MetricTwedConfig`] when a compile-time
//! metric witness is required.
//!
//! ```rust
//! use liblevenshtein::time_series::{
//!     MetricTwedConfig, MetricTwedTransducer, QuantizationConfig,
//! };
//!
//! let references = vec![vec![0.0, 1.0, 2.0], vec![0.0, 2.0, 3.0]];
//! let kernel = MetricTwedConfig::try_new(0.5, 1.0).unwrap();
//! let index = MetricTwedTransducer::from_series(
//!     QuantizationConfig::for_u8(0.0, 3.0),
//!     kernel,
//!     &references,
//! );
//! assert_eq!(index.search_range(&[0.0, 1.0, 2.0], 0.0), vec![(0, 0.0)]);
//! ```
//!
//! # Banded Dynamic Time Warping
//!
//! [`crate::time_series::DtwTransducer`] computes exact Dynamic Time Warping
//! (DTW) under a required symmetric Sakoe–Chiba band. Internal DP and
//! LB_Keogh costs are squared; public thresholds and results are square roots.
//! DTW is explicitly non-metric and must not be used in structures whose
//! pruning proof assumes the triangle inequality.
//!
//! ```rust
//! use liblevenshtein::time_series::{
//!     DtwConfig, DtwTransducer, QuantizationConfig,
//! };
//!
//! let references = vec![vec![0.0, 1.0, 2.0], vec![0.0, 1.0, 1.0, 2.0]];
//! let index = DtwTransducer::from_series(
//!     QuantizationConfig::for_u8(0.0, 10.0),
//!     DtwConfig::new(1),
//!     &references,
//! );
//! let zero_distance = index.search_range(&[0.0, 1.0, 2.0], 0.0);
//! assert_eq!(zero_distance.len(), 2);
//! assert!(zero_distance.iter().all(|(_, distance)| *distance == 0.0));
//! ```
//!
//! # Discrete Fréchet distance
//!
//! Discrete Fréchet minimizes the largest link in an order-preserving
//! coupling. [`crate::time_series::FrechetTransducer`] exercises the generic
//! walker with bottleneck (`max`) rather than additive path accumulation.
//!
//! ```rust
//! use liblevenshtein::time_series::{
//!     FrechetConfig, FrechetTransducer, QuantizationConfig,
//! };
//!
//! let references = vec![vec![1.0, 2.0], vec![1.0, 1.0, 2.0]];
//! let index = FrechetTransducer::from_series(
//!     QuantizationConfig::for_u8(-10.0, 10.0),
//!     FrechetConfig::new(),
//!     &references,
//! );
//! assert_eq!(index.search_range(&[1.0, 2.0], 0.0).len(), 2);
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
//! # Lower-Bound and Heuristic Pruning
//!
//! For efficient search over large databases, a proved lower bound can prune
//! candidates without computing the expensive full MSM distance. Prefix
//! Euclidean, L1, and Combined scores are also exported, but they are heuristics
//! for MSM: split/merge paths can be cheaper than pointwise prefix matching.
//!
//! ```rust
//! use liblevenshtein::time_series::{
//!     MsmConfig, length_lb, euclidean_lb,
//! };
//!
//! let x = vec![1.0, 2.0, 3.0, 4.0];
//! let y = vec![1.5, 2.5, 3.5, 4.5];
//! let c = 1.0;
//!
//! // Correctness-preserving lower bound.
//! let lb_length = length_lb(&x, &y, c);
//!
//! // If the proved lower bound exceeds the threshold, skip exact MSM.
//! let threshold = 2.0;
//! if lb_length > threshold {
//!     println!("Pruned: LB {} > threshold {}", lb_length, threshold);
//! } else {
//!     let msm = MsmConfig::new(c).distance(&x, &y);
//!     println!("MSM distance: {}", msm);
//! }
//!
//! // Heuristic scores can be useful for approximate workflows.
//! let _heuristic = euclidean_lb(&x, &y);
//! ```
//!
//! # References
//!
//! - Stefan, Alexandra, et al. "The move-split-merge metric for time series."
//!   IEEE transactions on Knowledge and Data Engineering 25.6 (2012): 1425-1438.

mod alignment;
mod approx_msm;
pub mod automaton;
pub mod bounded;
pub mod elastic;
mod encoding;
mod hybrid_search;
pub mod kernels;
mod lower_bounds;
mod metric_domains;
mod msm;
pub mod msm_interval;
mod msm_kernel;
mod msm_position;
mod msm_state;
mod msm_transducer;
mod msm_transition;
mod rolling;
mod timestamped_twed;
mod trie_index;
mod vector;

// MSM metric exports
pub use alignment::{MsmAlignmentStep, MsmAlignmentWitness, MsmWitnessReplayError};
pub use approx_msm::{paa_features, ApproxMsmConfig, ApproxMsmIndex};
pub use automaton::{
    ElasticOnlineAutomaton, ElasticOnlineObservation, ErpOnlineAutomaton, ErpOnlineObservation,
    OnlineAutomatonLimits, OnlineStepOutcome, TemporalArenaLimits, TemporalAutomatonError,
    TemporalStateId, TimestampedTwedOnlineAutomaton, TimestampedTwedOnlineObservation,
};
pub use bounded::{
    ExactDecision, IncompleteReason, NoWitness, Operand, OperationOutcome, PageBudget,
    ResourceKind, ResourceLedger, ResourceLimits, ResourceUsage, TemporalValidationError,
};
pub use metric_domains::{
    AuditedMetricTimeSeriesIndex, ErpQuotientSeries, FrechetStutterClass, MetricDomainError,
    MetricErpConfig, MetricErpTransducer, MetricFrechetTransducer,
};
pub use msm::{MetricMsmConfig, MetricMsmConfigError, MsmConfig, MsmConfigError, MsmResult};
pub use msm_kernel::{MetricMsmKernel, MsmKernel};
pub use msm_position::{msm_subsumes, MsmPosition};
pub use msm_state::MsmState;
pub use msm_transition::{
    initial_msm_state, msm_distance_automaton, msm_distance_wavefront, transition_msm_position,
    transition_msm_state,
};
pub use rolling::{BoundedRollingWindow, RollingWindowSnapshot, RollingWindowStep};

// Encoding exports
pub use elastic::{
    ElasticProductStateStats, ElasticSnapshot, ElasticSnapshotError, ElasticSnapshotIdentity,
    ElasticSnapshotKernel, ElasticSnapshotMetadata, ErpAutomatonRangeContinuation,
};
pub use encoding::QuantizationConfig;
pub use encoding::{delta_encoding, float_encoding, sax_encoding};

// Indexing exports
pub use hybrid_search::{HybridSearchIndex, HybridSearchIndexBuilder, HybridSearchStats};
pub use kernels::{
    erp_gap_mass_lower_bound, frechet_candidate_lower_bound, frechet_endpoint_lower_bound,
    frechet_one_sided_hausdorff_lower_bound, keogh_envelopes, lb_keogh, lb_keogh_squared,
    twed_length_lower_bound, DtwConfig, DtwKernel, DtwTransducer, ErpConfig, ErpKernel,
    ErpTransducer, FrechetConfig, FrechetKernel, FrechetTransducer, KeoghPlan, MetricTwedConfig,
    MetricTwedConfigError, MetricTwedKernel, MetricTwedTransducer, MetricUnitGridTwedConfig,
    MetricUnitGridTwedKernel, MetricUnitGridTwedTransducer, SoftDtwAnalysis, SoftDtwConfig,
    SoftDtwConfigError, TwedConfig, TwedKernel, TwedTransducer, UnitGridTwedConfig,
    UnitGridTwedKernel, UnitGridTwedTransducer,
};
pub use msm_transducer::{MetricMsmTransducer, MsmTransducer};
pub use timestamped_twed::{
    MetricTimestampedTwedConfig, TimestampUnit, TimestampedSeries, TimestampedTwedError,
};
pub use trie_index::{TimeSeriesIndex, TimeSeriesIndexBuilder, TimeSeriesIndexStats};
pub use vector::{
    GroundMetric, L1GroundMetric, L2GroundMetric, LinfGroundMetric, VectorFrechetMetric,
    VectorFrechetOnlineAutomaton, VectorFrechetOnlineObservation, VectorFrechetPath,
    VectorMetricError, VectorSample,
};

// Lower bound exports
#[cfg(feature = "rayon")]
pub use lower_bounds::search_with_lb_parallel;
pub use lower_bounds::{
    combined_lb, euclidean_lb, filter_by_lower_bound, l1_lb, length_lb, search_with_lb,
    search_with_lb_stats, LowerBoundConfig, LowerBoundStats, LowerBoundType,
};
