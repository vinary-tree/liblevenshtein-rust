//! Hybrid search combining trie-based filtering with exact MSM verification.
//!
//! This module provides efficient time series similarity search by:
//!
//! 1. **Candidate generation**: Use trie-based approximate search on quantized sequences
//! 2. **Verification**: Compute exact MSM distance on candidates
//!
//! This two-phase approach achieves both efficiency (fast filtering) and accuracy
//! (exact distance computation).
//!
//! # Example
//!
//! ```rust
//! use liblevenshtein::time_series::{
//!     HybridSearchIndex, QuantizationConfig, MsmConfig,
//! };
//!
//! // Create hybrid index
//! let quant_config = QuantizationConfig::for_u8(0.0, 100.0);
//! let msm_config = MsmConfig::new(1.0);
//! let mut index = HybridSearchIndex::new(quant_config, msm_config);
//!
//! // Insert time series
//! index.insert(0, &[10.0, 20.0, 30.0]);
//! index.insert(1, &[15.0, 25.0, 35.0]);
//! index.insert(2, &[50.0, 60.0, 70.0]);
//!
//! // Search with exact MSM verification
//! let results = index.search_exact(&[12.0, 22.0, 32.0], 5.0);
//! for (id, distance) in results {
//!     println!("Series {}: MSM distance = {:.2}", id, distance);
//! }
//! ```

use super::encoding::QuantizationConfig;
use super::lower_bounds::{LowerBoundConfig, LowerBoundStats, LowerBoundType};
use super::msm::MsmConfig;
use super::trie_index::TimeSeriesIndex;
use libdictenstein::DictionaryValue;
use std::collections::HashMap;

/// Hybrid search index combining trie filtering with exact MSM verification.
///
/// # Architecture
///
/// ```text
/// Query → Quantize → Trie Search → Candidates → Bound/Heuristic → MSM Verify → Results
///                      (fast)        (approx)        (fast)       (exact)
/// ```
///
/// # Performance Trade-offs
///
/// | Parameter | Effect on Candidates | Effect on Accuracy |
/// |-----------|---------------------|-------------------|
/// | More bins | Fewer candidates | Better filtering |
/// | Larger trie threshold | More candidates | Fewer missed results |
/// | Smaller MSM threshold | Fewer final results | More selective |
/// | Bound type | Varies | `LengthOnly` is safe; heuristic types can miss matches |
#[derive(Debug)]
pub struct HybridSearchIndex<V: DictionaryValue = usize> {
    /// Quantization-based trie index for candidate generation
    trie_index: TimeSeriesIndex<V>,

    /// MSM configuration for exact distance computation
    msm_config: MsmConfig,

    /// Original series storage (always needed for verification)
    originals: HashMap<V, Vec<f64>>,

    /// Multiplier for trie threshold based on MSM threshold
    /// trie_threshold = msm_threshold * trie_threshold_multiplier / bin_width
    trie_threshold_multiplier: f64,

    /// Lower-bound or heuristic configuration for pruning
    lb_config: LowerBoundConfig,

    /// Whether to use lower-bound or heuristic pruning
    use_lower_bounds: bool,
}

impl<V: DictionaryValue + std::hash::Hash + Eq + Copy> HybridSearchIndex<V> {
    /// Create a new hybrid search index.
    ///
    /// # Arguments
    ///
    /// * `quant_config` - Quantization configuration for trie indexing
    /// * `msm_config` - MSM configuration for exact distance computation
    pub fn new(quant_config: QuantizationConfig, msm_config: MsmConfig) -> Self {
        let c = msm_config.c;
        Self {
            trie_index: TimeSeriesIndex::new(quant_config),
            msm_config,
            originals: HashMap::new(),
            trie_threshold_multiplier: 2.0, // Conservative default
            lb_config: LowerBoundConfig::new(c),
            use_lower_bounds: true, // Enable by default
        }
    }

    /// Set the trie threshold multiplier.
    ///
    /// Higher values generate more candidates (fewer missed results, slower).
    /// Lower values generate fewer candidates (faster, may miss results).
    ///
    /// Default: 2.0
    pub fn set_trie_threshold_multiplier(&mut self, multiplier: f64) {
        assert!(multiplier > 0.0, "Multiplier must be positive");
        self.trie_threshold_multiplier = multiplier;
    }

    /// Enable or disable lower-bound / heuristic pruning.
    ///
    /// With the default `LengthOnly` bound this preserves exact verification of
    /// the generated candidate set. Euclidean, L1, and Combined are heuristic
    /// filters and may drop true MSM-near candidates.
    ///
    /// Default: enabled
    pub fn set_use_lower_bounds(&mut self, enable: bool) {
        self.use_lower_bounds = enable;
    }

    /// Set the bound or heuristic type to use for pruning.
    ///
    /// # Options
    ///
    /// - `LengthOnly`: Fastest, correctness-preserving pruning (O(1))
    /// - `EuclideanOnly`: Heuristic pruning (O(n), can miss true MSM matches)
    /// - `L1Only`: Heuristic pruning (O(n), can miss true MSM matches)
    /// - `Combined`: Heuristic pruning (O(n), can miss true MSM matches)
    ///
    /// Default: `LengthOnly`
    pub fn set_lower_bound_type(&mut self, lb_type: LowerBoundType) {
        self.lb_config.bounds = lb_type;
    }

    /// Get the current lower-bound / heuristic configuration.
    pub fn lower_bound_config(&self) -> &LowerBoundConfig {
        &self.lb_config
    }

    /// Check if lower-bound / heuristic pruning is enabled.
    pub fn uses_lower_bounds(&self) -> bool {
        self.use_lower_bounds
    }

    /// Get the quantization configuration.
    #[inline]
    pub fn quant_config(&self) -> &QuantizationConfig {
        self.trie_index.config()
    }

    /// Get the MSM configuration.
    #[inline]
    pub fn msm_config(&self) -> &MsmConfig {
        &self.msm_config
    }

    /// Get the number of indexed series.
    #[inline]
    pub fn len(&self) -> usize {
        self.originals.len()
    }

    /// Check if the index is empty.
    #[inline]
    pub fn is_empty(&self) -> bool {
        self.originals.is_empty()
    }

    /// Insert a time series with an associated value.
    ///
    /// # Arguments
    ///
    /// * `value` - The value to associate with this series
    /// * `series` - The time series data
    ///
    /// # Returns
    ///
    /// `true` if the series was newly inserted.
    pub fn insert(&mut self, value: V, series: &[f64]) -> bool {
        let is_new = !self.originals.contains_key(&value);
        self.trie_index.insert(value, series);
        self.originals.insert(value, series.to_vec());
        is_new
    }

    /// Get the original series for a value.
    pub fn get_original(&self, value: &V) -> Option<&[f64]> {
        self.originals.get(value).map(|v| v.as_slice())
    }

    /// Compute the trie threshold for a given MSM threshold.
    ///
    /// The trie threshold is an upper bound on the Levenshtein distance
    /// that corresponds to the MSM threshold, accounting for quantization error.
    fn compute_trie_threshold(&self, msm_threshold: f64) -> usize {
        let bin_width = self.trie_index.config().bin_width();
        // MSM distance roughly corresponds to sum of value differences
        // Levenshtein distance counts symbol mismatches
        // A MSM threshold of T allows value changes totaling T
        // In the worst case, each bin difference of 1 corresponds to bin_width value difference
        // So trie_threshold ≈ msm_threshold / bin_width * multiplier
        let threshold =
            (msm_threshold / bin_width * self.trie_threshold_multiplier).ceil() as usize;
        // Ensure at least 1 for any non-zero MSM threshold
        threshold.max(1)
    }

    /// Search for similar series with exact MSM verification.
    ///
    /// Pipeline:
    /// 1. Uses trie to find candidate series within approximate distance
    /// 2. (Optional) Filters candidates using the configured bound/heuristic
    /// 3. Computes exact MSM distance on remaining candidates
    /// 4. Returns only candidates within the exact threshold
    ///
    /// # Arguments
    ///
    /// * `query` - The query time series
    /// * `msm_threshold` - Maximum exact MSM distance
    ///
    /// # Returns
    ///
    /// Vector of (value, exact_distance) pairs for matching series,
    /// sorted by distance ascending.
    pub fn search_exact(&self, query: &[f64], msm_threshold: f64) -> Vec<(V, f64)> {
        // Phase 1: Get candidates from trie
        let trie_threshold = self.compute_trie_threshold(msm_threshold);
        let candidates = self.trie_index.search(query, trie_threshold);

        // Phase 2: Apply configured prefiltering and verify with MSM.
        let mut results: Vec<(V, f64)> = candidates
            .into_iter()
            .filter_map(|(value, _approx_dist)| {
                let original = self.originals.get(&value)?;

                // Phase 2a: Lower-bound / heuristic pruning.
                if self.use_lower_bounds {
                    let lb = self.lb_config.lower_bound(query, original);
                    if lb > msm_threshold {
                        return None; // Prune - configured score exceeds threshold.
                    }
                }

                // Phase 2b: Exact MSM verification
                let exact_dist = self.msm_config.distance(query, original);
                if exact_dist <= msm_threshold + 1e-9 {
                    Some((value, exact_dist))
                } else {
                    None
                }
            })
            .collect();

        // Sort by distance
        results.sort_by(|a, b| a.1.partial_cmp(&b.1).unwrap_or(std::cmp::Ordering::Equal));

        results
    }

    /// Search for k-nearest neighbors with exact MSM distance.
    ///
    /// # Arguments
    ///
    /// * `query` - The query time series
    /// * `k` - Number of nearest neighbors to return
    /// * `initial_threshold` - Initial MSM threshold for candidate generation
    ///
    /// # Returns
    ///
    /// Vector of (value, exact_distance) pairs for the k nearest series,
    /// sorted by distance ascending.
    pub fn search_knn(&self, query: &[f64], k: usize, initial_threshold: f64) -> Vec<(V, f64)> {
        if k == 0 || self.is_empty() {
            return Vec::new();
        }

        // Start with initial threshold
        let mut threshold = initial_threshold;
        let mut best_results: Vec<(V, f64)>;

        // Iteratively search with increasing threshold until we have k results
        loop {
            let results = self.search_exact(query, threshold);

            if results.len() >= k {
                // Found enough results
                return results.into_iter().take(k).collect();
            }

            // Keep best results so far
            best_results = results;

            // If we've searched everything, return what we have
            if threshold >= 1e10 {
                return best_results;
            }

            // Double the threshold and try again
            threshold *= 2.0;
        }
    }

    /// Brute-force search (for comparison/validation).
    ///
    /// Computes exact MSM distance to all series without using the trie.
    /// Use this to validate the hybrid search results.
    pub fn search_brute_force(&self, query: &[f64], msm_threshold: f64) -> Vec<(V, f64)> {
        let mut results: Vec<(V, f64)> = self
            .originals
            .iter()
            .filter_map(|(&value, original)| {
                let dist = self.msm_config.distance(query, original);
                if dist <= msm_threshold + 1e-9 {
                    Some((value, dist))
                } else {
                    None
                }
            })
            .collect();

        results.sort_by(|a, b| a.1.partial_cmp(&b.1).unwrap_or(std::cmp::Ordering::Equal));
        results
    }

    /// Get statistics about a search operation.
    pub fn search_stats(&self, query: &[f64], msm_threshold: f64) -> HybridSearchStats {
        let trie_threshold = self.compute_trie_threshold(msm_threshold);
        let candidates = self.trie_index.search(query, trie_threshold);
        let num_candidates = candidates.len();

        let mut pruned_by_lb = 0;
        let mut passed_lb = 0;
        let mut passed_exact = 0;

        for (value, _) in candidates {
            let original = match self.originals.get(&value) {
                Some(o) => o,
                None => continue,
            };

            // Check bound / heuristic if enabled.
            if self.use_lower_bounds {
                let lb = self.lb_config.lower_bound(query, original);
                if lb > msm_threshold {
                    pruned_by_lb += 1;
                    continue;
                }
            }
            passed_lb += 1;

            // Compute exact MSM
            let exact_dist = self.msm_config.distance(query, original);
            if exact_dist <= msm_threshold + 1e-9 {
                passed_exact += 1;
            }
        }

        HybridSearchStats {
            total_series: self.originals.len(),
            trie_threshold,
            num_candidates,
            pruned_by_lb,
            passed_lb,
            passed_exact,
            trie_pruning_rate: if self.originals.len() > 0 {
                1.0 - (num_candidates as f64 / self.originals.len() as f64)
            } else {
                0.0
            },
            lb_pruning_rate: if num_candidates > 0 {
                pruned_by_lb as f64 / num_candidates as f64
            } else {
                0.0
            },
            false_positive_rate: if passed_lb > 0 {
                (passed_lb - passed_exact) as f64 / passed_lb as f64
            } else {
                0.0
            },
        }
    }

    /// Get detailed statistics including prefilter breakdown.
    pub fn search_stats_detailed(
        &self,
        query: &[f64],
        msm_threshold: f64,
    ) -> (HybridSearchStats, LowerBoundStats) {
        let hybrid_stats = self.search_stats(query, msm_threshold);

        let lb_stats = LowerBoundStats {
            total_candidates: hybrid_stats.num_candidates,
            pruned_by_lb: hybrid_stats.pruned_by_lb,
            passed_lb: hybrid_stats.passed_lb,
            passed_exact: hybrid_stats.passed_exact,
            pruning_rate: hybrid_stats.lb_pruning_rate,
            false_positive_rate: hybrid_stats.false_positive_rate,
        };

        (hybrid_stats, lb_stats)
    }
}

/// Statistics from a hybrid search operation.
#[derive(Debug, Clone)]
pub struct HybridSearchStats {
    /// Total number of series in the index
    pub total_series: usize,

    /// Trie threshold used for candidate generation
    pub trie_threshold: usize,

    /// Number of candidates from trie search
    pub num_candidates: usize,

    /// Number of candidates pruned by the configured bound/heuristic
    pub pruned_by_lb: usize,

    /// Number of candidates that passed the configured prefilter
    pub passed_lb: usize,

    /// Number of candidates that passed exact MSM threshold
    pub passed_exact: usize,

    /// Trie pruning rate: 1 - (candidates / total)
    pub trie_pruning_rate: f64,

    /// Bound/heuristic pruning rate: pruned_by_lb / candidates
    pub lb_pruning_rate: f64,

    /// False positive rate after bound/heuristic filtering
    pub false_positive_rate: f64,
}

impl std::fmt::Display for HybridSearchStats {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        writeln!(f, "Hybrid Search Statistics:")?;
        writeln!(f, "  Total series: {}", self.total_series)?;
        writeln!(f, "  Trie threshold: {}", self.trie_threshold)?;
        writeln!(f, "  Candidates from trie: {}", self.num_candidates)?;
        writeln!(f, "  Pruned by configured prefilter: {}", self.pruned_by_lb)?;
        writeln!(f, "  Passed configured prefilter: {}", self.passed_lb)?;
        writeln!(f, "  Passed exact MSM: {}", self.passed_exact)?;
        writeln!(
            f,
            "  Trie pruning rate: {:.1}%",
            self.trie_pruning_rate * 100.0
        )?;
        writeln!(
            f,
            "  Configured prefilter pruning rate: {:.1}%",
            self.lb_pruning_rate * 100.0
        )?;
        writeln!(
            f,
            "  False positive rate: {:.1}%",
            self.false_positive_rate * 100.0
        )
    }
}

/// Builder for HybridSearchIndex.
#[derive(Debug, Clone)]
pub struct HybridSearchIndexBuilder {
    quant_config: Option<QuantizationConfig>,
    msm_config: Option<MsmConfig>,
    trie_threshold_multiplier: f64,
    use_lower_bounds: bool,
    lb_type: LowerBoundType,
}

impl HybridSearchIndexBuilder {
    /// Create a new builder.
    pub fn new() -> Self {
        Self {
            quant_config: None,
            msm_config: None,
            trie_threshold_multiplier: 2.0,
            use_lower_bounds: true,
            lb_type: LowerBoundType::LengthOnly,
        }
    }

    /// Set the quantization configuration.
    pub fn quant_config(mut self, config: QuantizationConfig) -> Self {
        self.quant_config = Some(config);
        self
    }

    /// Set quantization parameters directly.
    pub fn quantization(mut self, min: f64, max: f64, bins: u32) -> Self {
        self.quant_config = Some(QuantizationConfig::uniform(min, max, bins));
        self
    }

    /// Set the MSM configuration.
    pub fn msm_config(mut self, config: MsmConfig) -> Self {
        self.msm_config = Some(config);
        self
    }

    /// Set the MSM cost parameter c.
    pub fn msm_cost(mut self, c: f64) -> Self {
        self.msm_config = Some(MsmConfig::new(c));
        self
    }

    /// Set the trie threshold multiplier.
    pub fn trie_threshold_multiplier(mut self, multiplier: f64) -> Self {
        self.trie_threshold_multiplier = multiplier;
        self
    }

    /// Enable or disable lower-bound / heuristic pruning.
    pub fn use_lower_bounds(mut self, enable: bool) -> Self {
        self.use_lower_bounds = enable;
        self
    }

    /// Set the bound or heuristic type.
    pub fn lower_bound_type(mut self, lb_type: LowerBoundType) -> Self {
        self.lb_type = lb_type;
        self
    }

    /// Build the index.
    ///
    /// # Panics
    ///
    /// Panics if quantization config or MSM config is not set.
    pub fn build<V: DictionaryValue + std::hash::Hash + Eq + Copy>(self) -> HybridSearchIndex<V> {
        let quant_config = self.quant_config.expect("Quantization config must be set");
        let msm_config = self.msm_config.expect("MSM config must be set");

        let mut index = HybridSearchIndex::new(quant_config, msm_config);
        index.set_trie_threshold_multiplier(self.trie_threshold_multiplier);
        index.set_use_lower_bounds(self.use_lower_bounds);
        index.set_lower_bound_type(self.lb_type);
        index
    }
}

impl Default for HybridSearchIndexBuilder {
    fn default() -> Self {
        Self::new()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    const EPSILON: f64 = 1e-9;

    fn approx_eq(a: f64, b: f64) -> bool {
        (a - b).abs() < EPSILON
    }

    #[test]
    fn test_hybrid_index_creation() {
        let quant_config = QuantizationConfig::for_u8(0.0, 100.0);
        let msm_config = MsmConfig::new(1.0);
        let index: HybridSearchIndex<usize> = HybridSearchIndex::new(quant_config, msm_config);

        assert!(index.is_empty());
        assert_eq!(index.len(), 0);
    }

    #[test]
    fn test_insert_and_get_original() {
        let quant_config = QuantizationConfig::for_u8(0.0, 100.0);
        let msm_config = MsmConfig::new(1.0);
        let mut index = HybridSearchIndex::new(quant_config, msm_config);

        let series = vec![10.0, 20.0, 30.0];
        assert!(index.insert(0usize, &series));
        assert_eq!(index.len(), 1);

        let original = index.get_original(&0);
        assert!(original.is_some());
        assert_eq!(
            original.expect("expected Some original in test"),
            series.as_slice()
        );
    }

    #[test]
    fn test_search_exact_identical() {
        let quant_config = QuantizationConfig::for_u8(0.0, 100.0);
        let msm_config = MsmConfig::new(1.0);
        let mut index = HybridSearchIndex::new(quant_config, msm_config);

        let series = vec![10.0, 20.0, 30.0];
        index.insert(0usize, &series);

        // Exact match should have distance 0
        let results = index.search_exact(&series, 0.0);
        assert_eq!(results.len(), 1);
        assert_eq!(results[0].0, 0);
        assert!(approx_eq(results[0].1, 0.0));
    }

    #[test]
    fn test_search_exact_close_series() {
        let quant_config = QuantizationConfig::for_u8(0.0, 100.0);
        let msm_config = MsmConfig::new(1.0);
        let mut index = HybridSearchIndex::new(quant_config, msm_config);

        index.insert(0usize, &[10.0, 20.0, 30.0]);
        index.insert(1usize, &[11.0, 21.0, 31.0]); // Each element differs by 1
        index.insert(2usize, &[50.0, 60.0, 70.0]); // Far away

        // MSM distance of [10,20,30] to [11,21,31] is 3.0 (three moves of 1.0 each)
        let results = index.search_exact(&[10.0, 20.0, 30.0], 5.0);

        let found_ids: Vec<usize> = results.iter().map(|(id, _)| *id).collect();
        assert!(found_ids.contains(&0)); // Exact match
        assert!(found_ids.contains(&1)); // Close match

        // Series 2 is far (MSM distance = 120), should not be in results
        assert!(!found_ids.contains(&2));
    }

    #[test]
    fn test_search_exact_results_sorted() {
        let quant_config = QuantizationConfig::for_u8(0.0, 100.0);
        let msm_config = MsmConfig::new(1.0);
        let mut index = HybridSearchIndex::new(quant_config, msm_config);

        index.insert(0usize, &[10.0, 20.0, 30.0]);
        index.insert(1usize, &[12.0, 22.0, 32.0]); // Diff = 6
        index.insert(2usize, &[11.0, 21.0, 31.0]); // Diff = 3

        let results = index.search_exact(&[10.0, 20.0, 30.0], 10.0);

        // Results should be sorted by distance
        assert!(results.len() >= 2);
        assert!(results[0].1 <= results[1].1);
    }

    #[test]
    fn test_search_knn() {
        let quant_config = QuantizationConfig::for_u8(0.0, 100.0);
        let msm_config = MsmConfig::new(1.0);
        let mut index = HybridSearchIndex::new(quant_config, msm_config);

        index.insert(0usize, &[10.0, 20.0, 30.0]);
        index.insert(1usize, &[11.0, 21.0, 31.0]);
        index.insert(2usize, &[15.0, 25.0, 35.0]);
        index.insert(3usize, &[50.0, 60.0, 70.0]);

        let results = index.search_knn(&[10.0, 20.0, 30.0], 2, 5.0);

        assert_eq!(results.len(), 2);
        // First result should be exact match (distance 0)
        assert_eq!(results[0].0, 0);
        assert!(approx_eq(results[0].1, 0.0));
    }

    #[test]
    fn test_brute_force_matches_hybrid() {
        let quant_config = QuantizationConfig::for_u8(0.0, 100.0);
        let msm_config = MsmConfig::new(1.0);
        let mut index = HybridSearchIndex::new(quant_config, msm_config);

        index.insert(0usize, &[10.0, 20.0, 30.0]);
        index.insert(1usize, &[11.0, 21.0, 31.0]);
        index.insert(2usize, &[15.0, 25.0, 35.0]);

        let query = vec![12.0, 22.0, 32.0];
        let threshold = 10.0;

        let hybrid_results = index.search_exact(&query, threshold);
        let brute_results = index.search_brute_force(&query, threshold);

        // Both methods should return the same results
        assert_eq!(hybrid_results.len(), brute_results.len());

        for (hybrid, brute) in hybrid_results.iter().zip(brute_results.iter()) {
            assert_eq!(hybrid.0, brute.0);
            assert!(approx_eq(hybrid.1, brute.1));
        }
    }

    #[test]
    fn test_search_stats() {
        let quant_config = QuantizationConfig::for_u8(0.0, 100.0);
        let msm_config = MsmConfig::new(1.0);
        let mut index = HybridSearchIndex::new(quant_config, msm_config);

        for i in 0..10 {
            let series = vec![
                i as f64 * 10.0,
                i as f64 * 10.0 + 10.0,
                i as f64 * 10.0 + 20.0,
            ];
            index.insert(i, &series);
        }

        let stats = index.search_stats(&[25.0, 35.0, 45.0], 20.0);

        assert_eq!(stats.total_series, 10);
        assert!(stats.num_candidates > 0);
        assert!(stats.trie_pruning_rate >= 0.0 && stats.trie_pruning_rate <= 1.0);
        assert!(stats.lb_pruning_rate >= 0.0 && stats.lb_pruning_rate <= 1.0);
    }

    #[test]
    fn test_lower_bound_pruning() {
        let quant_config = QuantizationConfig::for_u8(0.0, 100.0);
        let msm_config = MsmConfig::new(1.0);
        let mut index = HybridSearchIndex::new(quant_config, msm_config);

        // Insert series with varying distances
        index.insert(0usize, &[10.0, 20.0, 30.0]); // Exact match
        index.insert(1usize, &[11.0, 21.0, 31.0]); // Close
        index.insert(2usize, &[50.0, 60.0, 70.0]); // Far

        // Search with safe lower-bound pruning enabled (default)
        let results_with_lb = index.search_exact(&[10.0, 20.0, 30.0], 5.0);

        // Disable lower-bound pruning and search again
        index.set_use_lower_bounds(false);
        let results_without_lb = index.search_exact(&[10.0, 20.0, 30.0], 5.0);

        // Results should be the same for the safe default bound.
        assert_eq!(results_with_lb.len(), results_without_lb.len());
    }

    #[test]
    fn test_lower_bound_type_config() {
        let quant_config = QuantizationConfig::for_u8(0.0, 100.0);
        let msm_config = MsmConfig::new(1.0);
        let mut index: HybridSearchIndex<usize> = HybridSearchIndex::new(quant_config, msm_config);

        // Default should be the correctness-preserving bound.
        assert_eq!(
            index.lower_bound_config().bounds,
            LowerBoundType::LengthOnly
        );

        // Heuristic filters remain opt-in.
        index.set_lower_bound_type(LowerBoundType::Combined);
        assert_eq!(index.lower_bound_config().bounds, LowerBoundType::Combined);
    }

    #[test]
    fn test_builder() {
        let index: HybridSearchIndex<usize> = HybridSearchIndexBuilder::new()
            .quantization(0.0, 100.0, 256)
            .msm_cost(1.0)
            .trie_threshold_multiplier(3.0)
            .build();

        assert!(index.is_empty());
    }

    #[test]
    fn test_compute_trie_threshold() {
        let quant_config = QuantizationConfig::for_u8(0.0, 256.0); // bin_width = 1.0
        let msm_config = MsmConfig::new(1.0);
        let index: HybridSearchIndex<usize> = HybridSearchIndex::new(quant_config, msm_config);

        // With bin_width = 1.0 and multiplier = 2.0:
        // threshold = ceil(msm_threshold / 1.0 * 2.0)
        assert_eq!(index.compute_trie_threshold(1.0), 2);
        assert_eq!(index.compute_trie_threshold(5.0), 10);
    }

    #[test]
    fn test_trie_threshold_multiplier() {
        let quant_config = QuantizationConfig::for_u8(0.0, 100.0);
        let msm_config = MsmConfig::new(1.0);
        let mut index: HybridSearchIndex<usize> = HybridSearchIndex::new(quant_config, msm_config);

        let threshold_2x = index.compute_trie_threshold(10.0);
        index.set_trie_threshold_multiplier(4.0);
        let threshold_4x = index.compute_trie_threshold(10.0);

        // 4x multiplier should give roughly 2x the threshold
        assert!(threshold_4x > threshold_2x);
    }
}
