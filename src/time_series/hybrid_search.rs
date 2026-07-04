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
use crate::numeric::nonnegative_ceil_to_usize;
use libdictenstein::DictionaryValue;
use std::collections::HashMap;

type BucketLocation = (usize, usize);

const DISTANCE_EPSILON: f64 = 1e-9;
const DEFAULT_RESULT_BUFFER_CAPACITY: usize = 64;
const DEFAULT_TRIE_THRESHOLD_MULTIPLIER: f64 = 2.0;

#[inline]
fn normalize_trie_threshold_multiplier(multiplier: f64) -> f64 {
    if multiplier.is_finite() && multiplier > 0.0 {
        multiplier
    } else {
        DEFAULT_TRIE_THRESHOLD_MULTIPLIER
    }
}

#[derive(Debug)]
struct HybridStoredSeries {
    series: Vec<f64>,
    bucket_location: BucketLocation,
}

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
    /// Quantization-based trie index for candidate-bucket generation.
    trie_index: TimeSeriesIndex<usize>,

    /// MSM configuration for exact distance computation
    msm_config: MsmConfig,

    /// Final bucket id -> all reference ids sharing that quantized key.
    buckets: Vec<Vec<V>>,

    /// Retired bucket ids that can be reused for future quantized keys.
    free_bucket_ids: Vec<usize>,

    /// Reference id -> original series and bucket slot for exact verification and upserts.
    originals: HashMap<V, HybridStoredSeries>,

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
        Self::with_capacity(quant_config, msm_config, 0)
    }

    /// Create a hybrid index and reserve metadata for `capacity` values.
    ///
    /// The quantized candidate buckets, original-series table, and underlying
    /// trie metadata are pre-sized for bulk loading.
    pub fn with_capacity(
        quant_config: QuantizationConfig,
        msm_config: MsmConfig,
        capacity: usize,
    ) -> Self {
        let msm_config = msm_config.normalized();
        let c = msm_config.split_merge_cost();
        Self {
            trie_index: TimeSeriesIndex::with_capacity(quant_config, capacity),
            msm_config,
            buckets: Vec::with_capacity(capacity),
            free_bucket_ids: Vec::new(),
            originals: HashMap::with_capacity(capacity),
            trie_threshold_multiplier: DEFAULT_TRIE_THRESHOLD_MULTIPLIER,
            lb_config: LowerBoundConfig::new(c),
            use_lower_bounds: true, // Enable by default
        }
    }

    /// Set the trie threshold multiplier.
    ///
    /// Higher values generate more candidates (fewer missed results, slower).
    /// Lower values generate fewer candidates (faster, may miss results).
    /// Non-positive or non-finite values fall back to the conservative default.
    ///
    /// Default: 2.0
    pub fn set_trie_threshold_multiplier(&mut self, multiplier: f64) {
        self.trie_threshold_multiplier = normalize_trie_threshold_multiplier(multiplier);
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
        let bucket_id = self.bucket_id_for_series(series);
        let is_new = !self.originals.contains_key(&value);

        let bucket_location = match self
            .originals
            .get(&value)
            .map(|stored| stored.bucket_location)
        {
            Some((old_bucket_id, old_slot)) if old_bucket_id == bucket_id => (bucket_id, old_slot),
            Some((old_bucket_id, old_slot)) => {
                if self.remove_from_bucket(&value, old_bucket_id, old_slot) {
                    self.retire_empty_bucket(old_bucket_id);
                }
                self.push_to_bucket(value, bucket_id)
            }
            None => self.push_to_bucket(value, bucket_id),
        };

        self.originals.insert(
            value,
            HybridStoredSeries {
                series: series.to_vec(),
                bucket_location,
            },
        );
        is_new
    }

    fn bucket_id_for_series(&mut self, series: &[f64]) -> usize {
        if let Some(existing) = self.trie_index.get(series) {
            return existing;
        }

        let bucket_id = match self.free_bucket_ids.pop() {
            Some(bucket_id) => bucket_id,
            None => {
                self.buckets.push(Vec::with_capacity(1));
                self.buckets.len() - 1
            }
        };
        debug_assert!(self.buckets[bucket_id].is_empty());

        let inserted = self.trie_index.insert(bucket_id, series);
        debug_assert!(inserted, "new hybrid bucket key should insert exactly once");
        bucket_id
    }

    fn push_to_bucket(&mut self, value: V, bucket_id: usize) -> BucketLocation {
        let slot = self.buckets[bucket_id].len();
        self.buckets[bucket_id].push(value);
        (bucket_id, slot)
    }

    fn remove_from_bucket(&mut self, value: &V, bucket_id: usize, slot: usize) -> bool {
        let (moved, became_empty) = {
            let bucket = &mut self.buckets[bucket_id];
            debug_assert!(bucket.get(slot).is_some_and(|stored| stored == value));
            let removed = bucket.swap_remove(slot);
            debug_assert!(&removed == value);
            (bucket.get(slot).copied(), bucket.is_empty())
        };
        if let Some(moved) = moved {
            if let Some(stored) = self.originals.get_mut(&moved) {
                stored.bucket_location = (bucket_id, slot);
            }
        }
        became_empty
    }

    fn retire_empty_bucket(&mut self, bucket_id: usize) {
        debug_assert!(self.buckets[bucket_id].is_empty());
        self.buckets[bucket_id] = Vec::new();

        let removed = self.trie_index.remove(bucket_id);
        debug_assert!(removed, "retired hybrid bucket should be indexed");
        if removed {
            self.free_bucket_ids.push(bucket_id);
        }
    }

    /// Remove a value from the index.
    ///
    /// This drops the stored original series, removes the value from its
    /// quantized bucket, and removes the empty bucket from candidate generation.
    /// Retired bucket slots are reused by future inserts, so repeated
    /// insert/remove cycles do not grow the bucket vector without bound.
    ///
    /// # Returns
    ///
    /// `true` if the value was present and removed, `false` otherwise.
    pub fn remove(&mut self, value: V) -> bool {
        let Some(stored) = self.originals.remove(&value) else {
            return false;
        };

        let (bucket_id, slot) = stored.bucket_location;
        if self.remove_from_bucket(&value, bucket_id, slot) {
            self.retire_empty_bucket(bucket_id);
        }
        true
    }

    /// Get the original series for a value.
    pub fn get_original(&self, value: &V) -> Option<&[f64]> {
        self.originals
            .get(value)
            .map(|stored| stored.series.as_slice())
    }

    /// Compute the trie threshold for a given MSM threshold.
    ///
    /// The trie threshold is an upper bound on the Levenshtein distance
    /// that corresponds to the MSM threshold, accounting for quantization error.
    fn compute_trie_threshold(&self, msm_threshold: f64) -> usize {
        if Self::is_invalid_threshold(msm_threshold) {
            return 0;
        }
        if msm_threshold.is_infinite() {
            return usize::MAX;
        }

        let bin_width = self.trie_index.config().bin_width();
        // MSM distance roughly corresponds to sum of value differences
        // Levenshtein distance counts symbol mismatches
        // A MSM threshold of T allows value changes totaling T
        // In the worst case, each bin difference of 1 corresponds to bin_width value difference
        // So trie_threshold ≈ msm_threshold / bin_width * multiplier
        let threshold = msm_threshold / bin_width * self.trie_threshold_multiplier;
        nonnegative_ceil_to_usize(threshold)
    }

    #[inline]
    fn is_invalid_threshold(msm_threshold: f64) -> bool {
        msm_threshold.is_nan() || msm_threshold < 0.0
    }

    #[inline]
    fn inclusive_cutoff(msm_threshold: f64) -> f64 {
        msm_threshold + DISTANCE_EPSILON
    }

    #[inline]
    fn result_capacity_hint(&self) -> usize {
        self.originals.len().min(DEFAULT_RESULT_BUFFER_CAPACITY)
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
        if Self::is_invalid_threshold(msm_threshold) || self.is_empty() {
            return Vec::new();
        }
        let cutoff = Self::inclusive_cutoff(msm_threshold);

        // Phase 1: Get candidates from trie
        let trie_threshold = self.compute_trie_threshold(msm_threshold);
        let candidate_buckets = self.trie_index.search(query, trie_threshold);

        // Phase 2: Apply configured prefiltering and verify with MSM.
        let mut results: Vec<(V, f64)> = Vec::with_capacity(self.result_capacity_hint());
        for (bucket_id, _approx_dist) in candidate_buckets {
            let Some(bucket) = self.buckets.get(bucket_id) else {
                continue;
            };
            for &value in bucket {
                let Some(stored) = self.originals.get(&value) else {
                    continue;
                };
                let original = &stored.series;

                // Phase 2a: Lower-bound / heuristic pruning.
                if self.use_lower_bounds {
                    let lb = self.lb_config.lower_bound(query, original);
                    if lb > cutoff {
                        continue; // Prune - configured score exceeds threshold.
                    }
                }

                // Phase 2b: Exact MSM verification
                let Some(exact_dist) = self
                    .msm_config
                    .distance_with_cutoff(query, original, cutoff)
                else {
                    continue;
                };
                if exact_dist <= cutoff {
                    results.push((value, exact_dist));
                }
            }
        }

        // Sort by distance
        results.sort_by(|a, b| a.1.total_cmp(&b.1));

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

        // Start with a valid threshold and make every expansion monotonic.
        let mut threshold = if initial_threshold.is_finite() && initial_threshold > 0.0 {
            initial_threshold
        } else {
            0.0
        };
        let mut best_results: Vec<(V, f64)>;

        // Iteratively search with increasing threshold until we have k results
        loop {
            let results = self.search_exact(query, threshold);

            if results.len() >= k {
                // Found enough results.
                let mut results = results;
                results.truncate(k);
                return results;
            }

            // Keep best results so far
            best_results = results;

            // If we've searched everything, return what we have
            if threshold >= 1e10 {
                return best_results;
            }

            // Double the threshold and try again. A zero/invalid initial
            // threshold must still make progress.
            let next_threshold = if threshold > 0.0 {
                (threshold * 2.0).min(1e10)
            } else {
                1.0
            };
            if next_threshold <= threshold {
                return best_results;
            }
            threshold = next_threshold;
        }
    }

    /// Brute-force search (for comparison/validation).
    ///
    /// Computes exact MSM distance to all series without using the trie.
    /// Use this to validate the hybrid search results.
    pub fn search_brute_force(&self, query: &[f64], msm_threshold: f64) -> Vec<(V, f64)> {
        if Self::is_invalid_threshold(msm_threshold) || self.is_empty() {
            return Vec::new();
        }
        let cutoff = Self::inclusive_cutoff(msm_threshold);

        // Iterate the deterministic `buckets` store (a `Vec<Vec<V>>` with stable
        // slot order) instead of the `originals` `HashMap` (whose `RandomState`
        // randomizes iteration order per process). Since the final `sort_by` is
        // a stable sort, equal-distance ties keep this reproducible bucket order
        // rather than a per-run-random one. Every id lives in exactly one bucket
        // slot, so this visits the same set exactly once.
        let mut results: Vec<(V, f64)> = Vec::with_capacity(self.result_capacity_hint());
        for bucket in &self.buckets {
            for &value in bucket {
                let Some(stored) = self.originals.get(&value) else {
                    continue;
                };
                let Some(dist) =
                    self.msm_config
                        .distance_with_cutoff(query, &stored.series, cutoff)
                else {
                    continue;
                };
                if dist <= cutoff {
                    results.push((value, dist));
                }
            }
        }

        results.sort_by(|a, b| a.1.total_cmp(&b.1));
        results
    }

    /// Get statistics about a search operation.
    pub fn search_stats(&self, query: &[f64], msm_threshold: f64) -> HybridSearchStats {
        if Self::is_invalid_threshold(msm_threshold) {
            return self.empty_search_stats(0);
        }
        let cutoff = Self::inclusive_cutoff(msm_threshold);

        let trie_threshold = self.compute_trie_threshold(msm_threshold);
        if self.is_empty() {
            return self.empty_search_stats(trie_threshold);
        }
        let candidate_buckets = self.trie_index.search(query, trie_threshold);

        let mut num_candidates = 0;
        let mut pruned_by_lb = 0;
        let mut passed_lb = 0;
        let mut passed_exact = 0;

        for (bucket_id, _) in candidate_buckets {
            let Some(bucket) = self.buckets.get(bucket_id) else {
                continue;
            };
            for &value in bucket {
                let Some(stored) = self.originals.get(&value) else {
                    continue;
                };
                let original = &stored.series;
                num_candidates += 1;

                // Check bound / heuristic if enabled.
                if self.use_lower_bounds {
                    let lb = self.lb_config.lower_bound(query, original);
                    if lb > cutoff {
                        pruned_by_lb += 1;
                        continue;
                    }
                }
                passed_lb += 1;

                // Compute exact MSM
                if let Some(exact_dist) = self
                    .msm_config
                    .distance_with_cutoff(query, original, cutoff)
                {
                    if exact_dist <= cutoff {
                        passed_exact += 1;
                    }
                }
            }
        }

        HybridSearchStats {
            total_series: self.originals.len(),
            trie_threshold,
            num_candidates,
            pruned_by_lb,
            passed_lb,
            passed_exact,
            trie_pruning_rate: if !self.originals.is_empty() {
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

    fn empty_search_stats(&self, trie_threshold: usize) -> HybridSearchStats {
        HybridSearchStats {
            total_series: self.originals.len(),
            trie_threshold,
            num_candidates: 0,
            pruned_by_lb: 0,
            passed_lb: 0,
            passed_exact: 0,
            trie_pruning_rate: if self.originals.is_empty() { 0.0 } else { 1.0 },
            lb_pruning_rate: 0.0,
            false_positive_rate: 0.0,
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
    quant_config: QuantizationConfig,
    msm_config: MsmConfig,
    trie_threshold_multiplier: f64,
    use_lower_bounds: bool,
    lb_type: LowerBoundType,
    capacity: usize,
}

impl HybridSearchIndexBuilder {
    /// Create a new builder.
    pub fn new() -> Self {
        Self {
            quant_config: QuantizationConfig::default(),
            msm_config: MsmConfig::default(),
            trie_threshold_multiplier: DEFAULT_TRIE_THRESHOLD_MULTIPLIER,
            use_lower_bounds: true,
            lb_type: LowerBoundType::LengthOnly,
            capacity: 0,
        }
    }

    /// Set the quantization configuration.
    pub fn quant_config(mut self, config: QuantizationConfig) -> Self {
        self.quant_config = config;
        self
    }

    /// Set quantization parameters directly.
    ///
    /// Invalid parameters leave the current configuration unchanged.
    pub fn quantization(mut self, min: f64, max: f64, bins: u32) -> Self {
        if let Some(config) = QuantizationConfig::try_uniform(min, max, bins) {
            self.quant_config = config;
        }
        self
    }

    /// Set the MSM configuration.
    pub fn msm_config(mut self, config: MsmConfig) -> Self {
        self.msm_config = config;
        self
    }

    /// Set the MSM cost parameter c.
    pub fn msm_cost(mut self, c: f64) -> Self {
        self.msm_config = MsmConfig::new(c);
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

    /// Reserve metadata capacity for at least `capacity` indexed values.
    pub fn capacity(mut self, capacity: usize) -> Self {
        self.capacity = capacity;
        self
    }

    /// Build the index.
    ///
    /// Uses [`QuantizationConfig::default`] and [`MsmConfig::default`] unless
    /// explicit configurations were provided.
    pub fn build<V: DictionaryValue + std::hash::Hash + Eq + Copy>(self) -> HybridSearchIndex<V> {
        let mut index =
            HybridSearchIndex::with_capacity(self.quant_config, self.msm_config, self.capacity);
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
    use std::collections::HashSet;

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
    fn test_hybrid_index_with_capacity_reserves_metadata() {
        let quant_config = QuantizationConfig::for_u8(0.0, 100.0);
        let msm_config = MsmConfig::new(1.0);
        let index: HybridSearchIndex<usize> =
            HybridSearchIndex::with_capacity(quant_config, msm_config, 8);

        assert!(index.buckets.capacity() >= 8);
        assert!(index.originals.capacity() >= 8);
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
    fn hybrid_brute_force_tie_order_is_deterministic() {
        // Many references at *equal* MSM distance to the query (identical stored
        // series under distinct ids). `search_brute_force` must derive its order
        // from the deterministic `buckets` store, not the randomized `originals`
        // HashMap: independent instances seed independent `RandomState`s, so a
        // HashMap-iterating brute force would order these equal-distance ties
        // differently between instances, while a bucket-iterating one is stable.
        let quant_config = QuantizationConfig::for_u8(0.0, 100.0);
        let msm_config = MsmConfig::new(1.0);
        let tied = [10.0, 20.0, 30.0];

        let build = || {
            let mut idx: HybridSearchIndex<usize> =
                HybridSearchIndex::new(quant_config.clone(), msm_config);
            for id in 0..16usize {
                idx.insert(id, &tied);
            }
            idx
        };

        let query = [10.0, 20.0, 30.0];
        let threshold = 10.0;

        let reference = build();
        let expected = reference.search_brute_force(&query, threshold);

        // All 16 ids returned at the same distance -> a genuine 16-way tie.
        assert_eq!(expected.len(), 16);
        assert!(expected.iter().all(|(_, d)| *d == expected[0].1));

        // Determinism across independent instances (fresh RandomState each time).
        for _ in 0..8 {
            let idx = build();
            assert_eq!(idx.search_brute_force(&query, threshold), expected);
        }
    }

    #[test]
    fn quantization_collisions_recover_all_values() {
        let quant_config = QuantizationConfig::for_u8(0.0, 100.0);
        let msm_config = MsmConfig::new(1.0);
        let mut index = HybridSearchIndex::new(quant_config, msm_config);

        assert!(index.insert(7usize, &[10.0, 20.0, 30.0]));
        assert!(index.insert(9usize, &[10.01, 20.01, 30.01]));

        assert_eq!(index.len(), 2);
        assert_eq!(index.buckets.iter().map(Vec::len).sum::<usize>(), 2);

        let results = index.search_exact(&[10.0, 20.0, 30.0], 0.1);
        let ids: HashSet<usize> = results.iter().map(|(id, _)| *id).collect();
        assert_eq!(ids, HashSet::from([7, 9]));

        let stats = index.search_stats(&[10.0, 20.0, 30.0], 0.1);
        assert_eq!(stats.num_candidates, 2);
        assert_eq!(stats.passed_exact, 2);
    }

    #[test]
    fn reinserting_value_relocates_bucket_membership() {
        let quant_config = QuantizationConfig::for_u8(0.0, 100.0);
        let msm_config = MsmConfig::new(1.0);
        let mut index = HybridSearchIndex::new(quant_config, msm_config);

        assert!(index.insert(7usize, &[10.0]));
        assert!(index.insert(9usize, &[10.01]));
        assert!(!index.insert(7usize, &[90.0]));

        assert_eq!(index.len(), 2);
        assert_eq!(index.buckets.iter().map(Vec::len).sum::<usize>(), 2);
        assert_eq!(index.get_original(&7), Some(&[90.0][..]));
        assert_eq!(index.get_original(&9), Some(&[10.01][..]));

        let old_bucket = index.trie_index.get(&[10.01]).unwrap();
        assert_eq!(index.buckets[old_bucket], vec![9]);
        assert_eq!(
            index.originals.get(&9).map(|stored| stored.bucket_location),
            Some((old_bucket, 0))
        );

        assert_eq!(index.search_exact(&[10.01], 0.0), vec![(9, 0.0)]);
        assert_eq!(index.search_exact(&[90.0], 0.0), vec![(7, 0.0)]);
    }

    #[test]
    fn remove_value_clears_bucket_membership_original_and_candidate_key() {
        let quant_config = QuantizationConfig::uniform(0.0, 100.0, 10);
        let msm_config = MsmConfig::new(1.0);
        let mut index = HybridSearchIndex::new(quant_config, msm_config);

        assert!(index.insert(7usize, &[10.0, 20.0]));
        let bucket_id = index.originals[&7].bucket_location.0;
        assert_eq!(
            index.trie_index.search(&[10.0, 20.0], 0),
            vec![(bucket_id, 0)]
        );

        assert!(index.remove(7));
        assert!(!index.remove(7));

        assert_eq!(index.len(), 0);
        assert_eq!(index.get_original(&7), None);
        assert!(index.search_exact(&[10.0, 20.0], 1.0).is_empty());
        assert!(index.search_brute_force(&[10.0, 20.0], 1.0).is_empty());
        assert!(index.trie_index.search(&[10.0, 20.0], 0).is_empty());
        assert!(index.buckets[bucket_id].is_empty());
        assert_eq!(index.buckets[bucket_id].capacity(), 0);
        assert!(index.free_bucket_ids.contains(&bucket_id));
    }

    #[test]
    fn remove_value_preserves_swapped_bucket_location() {
        let quant_config = QuantizationConfig::uniform(0.0, 100.0, 10);
        let msm_config = MsmConfig::new(1.0);
        let mut index = HybridSearchIndex::new(quant_config, msm_config);

        assert!(index.insert(7usize, &[10.0]));
        assert!(index.insert(9usize, &[10.01]));
        let shared_bucket = index.originals[&7].bucket_location.0;
        assert_eq!(index.buckets[shared_bucket], vec![7, 9]);

        assert!(index.remove(7));

        assert_eq!(index.len(), 1);
        assert_eq!(index.buckets[shared_bucket], vec![9]);
        assert_eq!(
            index.originals.get(&9).map(|stored| stored.bucket_location),
            Some((shared_bucket, 0))
        );
        assert_eq!(index.search_exact(&[10.01], 0.0), vec![(9, 0.0)]);

        assert!(!index.insert(9usize, &[90.0]));
        assert!(index.search_exact(&[10.01], 0.0).is_empty());
        assert_eq!(index.search_exact(&[90.0], 0.0), vec![(9, 0.0)]);
        assert!(index.trie_index.search(&[10.01], 0).is_empty());
        assert!(index.free_bucket_ids.contains(&shared_bucket));
    }

    #[test]
    fn insert_reuses_retired_bucket_slots() {
        let quant_config = QuantizationConfig::uniform(0.0, 100.0, 10);
        let msm_config = MsmConfig::new(1.0);
        let mut index = HybridSearchIndex::new(quant_config, msm_config);

        assert!(index.insert(7usize, &[10.0]));
        let retired_bucket = index.originals[&7].bucket_location.0;
        assert!(index.remove(7));

        assert!(index.insert(9usize, &[90.0]));

        assert_eq!(index.originals[&9].bucket_location, (retired_bucket, 0));
        assert!(index.free_bucket_ids.is_empty());
        assert!(index.trie_index.search(&[10.0], 0).is_empty());
        assert_eq!(
            index.trie_index.search(&[90.0], 0),
            vec![(retired_bucket, 0)]
        );
        assert_eq!(index.search_exact(&[90.0], 0.0), vec![(9, 0.0)]);
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
    fn test_search_knn_expands_from_zero_initial_threshold() {
        let quant_config = QuantizationConfig::for_u8(0.0, 100.0);
        let msm_config = MsmConfig::new(1.0);
        let mut index = HybridSearchIndex::new(quant_config, msm_config);

        index.insert(0usize, &[10.0, 20.0, 30.0]);
        index.insert(1usize, &[11.0, 21.0, 31.0]);
        index.insert(2usize, &[30.0, 40.0, 50.0]);

        let results = index.search_knn(&[10.0, 20.0, 30.0], 2, 0.0);

        assert_eq!(results.len(), 2);
        assert_eq!(results[0].0, 0);
        assert!(results[0].1 <= results[1].1);
    }

    #[test]
    fn test_search_knn_sanitizes_invalid_initial_threshold() {
        let quant_config = QuantizationConfig::for_u8(0.0, 100.0);
        let msm_config = MsmConfig::new(1.0);
        let mut index = HybridSearchIndex::new(quant_config, msm_config);

        index.insert(0usize, &[5.0, 5.0, 5.0]);
        index.insert(1usize, &[6.0, 6.0, 6.0]);

        let negative = index.search_knn(&[5.0, 5.0, 5.0], 2, -1.0);
        let nan = index.search_knn(&[5.0, 5.0, 5.0], 2, f64::NAN);

        assert_eq!(negative.len(), 2);
        assert_eq!(nan.len(), 2);
        assert_eq!(negative[0].0, 0);
        assert_eq!(nan[0].0, 0);
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
    fn test_search_cutoff_preserves_inclusive_epsilon() {
        let quant_config = QuantizationConfig::for_u8(0.0, 100.0);
        let msm_config = MsmConfig::new(1.0);
        let mut index = HybridSearchIndex::new(quant_config, msm_config);
        let query = [10.0, 20.0, 30.0];

        index.insert(0usize, &query);
        index.insert(1usize, &[10.0, 20.0, 30.0 + EPSILON / 2.0]);

        let hybrid_results = index.search_exact(&query, 0.0);
        let brute_results = index.search_brute_force(&query, 0.0);

        assert_eq!(hybrid_results, brute_results);
        assert_eq!(
            hybrid_results
                .iter()
                .map(|(value, _)| *value)
                .collect::<HashSet<_>>(),
            HashSet::from([0, 1])
        );

        let stats = index.search_stats(&query, 0.0);
        assert_eq!(stats.passed_exact, 2);
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
            .capacity(8)
            .build();

        assert!(index.is_empty());
        assert!(index.buckets.capacity() >= 8);
        assert!(index.originals.capacity() >= 8);
    }

    #[test]
    fn test_builder_default_configs() {
        let index: HybridSearchIndex<usize> = HybridSearchIndexBuilder::new().build();

        assert!(index.is_empty());
        assert_eq!(index.trie_index.config().min_value, 0.0);
        assert_eq!(index.trie_index.config().max_value, 1.0);
        assert_eq!(index.trie_index.config().num_bins, 256);
        assert_eq!(index.msm_config, MsmConfig::default());
        assert_eq!(index.trie_threshold_multiplier, 2.0);
        assert_eq!(
            index.lower_bound_config().bounds,
            LowerBoundType::LengthOnly
        );
    }

    #[test]
    fn test_builder_quantization_keeps_existing_config_for_invalid_parameters() {
        let index: HybridSearchIndex<usize> = HybridSearchIndexBuilder::new()
            .quantization(-1.0, 1.0, 64)
            .quantization(f64::NAN, 1.0, 10)
            .quantization(0.0, f64::INFINITY, 10)
            .quantization(10.0, 0.0, 10)
            .quantization(0.0, 1.0, 0)
            .build();

        assert_eq!(index.quant_config().min_value, -1.0);
        assert_eq!(index.quant_config().max_value, 1.0);
        assert_eq!(index.quant_config().num_bins, 64);
    }

    #[test]
    fn test_builder_normalizes_invalid_msm_cost() {
        let default_config = MsmConfig::default();

        let nan_index: HybridSearchIndex<usize> =
            HybridSearchIndexBuilder::new().msm_cost(f64::NAN).build();
        let negative_index: HybridSearchIndex<usize> =
            HybridSearchIndexBuilder::new().msm_cost(-5.0).build();
        let raw_index: HybridSearchIndex<usize> = HybridSearchIndex::new(
            QuantizationConfig::default(),
            MsmConfig { c: f64::INFINITY },
        );

        assert_eq!(nan_index.msm_config, default_config);
        assert_eq!(nan_index.lower_bound_config().c, default_config.c);
        assert_eq!(negative_index.msm_config.c, 0.0);
        assert_eq!(negative_index.lower_bound_config().c, 0.0);
        assert_eq!(raw_index.msm_config, default_config);
        assert_eq!(raw_index.lower_bound_config().c, default_config.c);
    }

    #[test]
    fn test_large_quantizer_is_coarsened_to_byte_bins() {
        let quant_config = QuantizationConfig::for_u16(0.0, 100.0);
        let msm_config = MsmConfig::new(1.0);
        let mut index: HybridSearchIndex<usize> = HybridSearchIndex::new(quant_config, msm_config);

        assert_eq!(index.quant_config().num_bins, 256);
        index.insert(0, &[10.0, 20.0, 30.0]);
        assert_eq!(index.search_exact(&[10.0, 20.0, 30.0], 0.0), vec![(0, 0.0)]);
    }

    #[test]
    fn test_compute_trie_threshold() {
        let quant_config = QuantizationConfig::for_u8(0.0, 256.0); // bin_width = 1.0
        let msm_config = MsmConfig::new(1.0);
        let index: HybridSearchIndex<usize> = HybridSearchIndex::new(quant_config, msm_config);

        // With bin_width = 1.0 and multiplier = 2.0:
        // threshold = ceil(msm_threshold / 1.0 * 2.0)
        assert_eq!(index.compute_trie_threshold(0.0), 0);
        assert_eq!(index.compute_trie_threshold(1.0), 2);
        assert_eq!(index.compute_trie_threshold(5.0), 10);
        assert_eq!(index.compute_trie_threshold(f64::INFINITY), usize::MAX);
        assert_eq!(index.compute_trie_threshold(f64::MAX), usize::MAX);
    }

    #[test]
    fn test_nonnegative_ceil_to_usize_bounds_before_casting() {
        let exact_large = 4_503_599_627_370_496.0;
        let expected_large = usize::try_from(4_503_599_627_370_496_u128).unwrap_or(usize::MAX);

        assert_eq!(nonnegative_ceil_to_usize(f64::NAN), 0);
        assert_eq!(nonnegative_ceil_to_usize(-1.0), 0);
        assert_eq!(nonnegative_ceil_to_usize(-0.0), 0);
        assert_eq!(nonnegative_ceil_to_usize(0.0), 0);
        assert_eq!(nonnegative_ceil_to_usize(0.1), 1);
        assert_eq!(nonnegative_ceil_to_usize(2.0), 2);
        assert_eq!(nonnegative_ceil_to_usize(2.1), 3);
        assert_eq!(nonnegative_ceil_to_usize(exact_large), expected_large);
        assert_eq!(nonnegative_ceil_to_usize(f64::INFINITY), usize::MAX);
        assert_eq!(nonnegative_ceil_to_usize(f64::MAX), usize::MAX);
    }

    #[test]
    fn test_invalid_thresholds_return_no_results_or_candidates() {
        let quant_config = QuantizationConfig::for_u8(0.0, 100.0);
        let msm_config = MsmConfig::new(1.0);
        let mut index: HybridSearchIndex<usize> = HybridSearchIndex::new(quant_config, msm_config);
        index.insert(0, &[10.0, 20.0, 30.0]);

        for threshold in [f64::NAN, -1.0] {
            assert_eq!(index.compute_trie_threshold(threshold), 0);
            assert!(index
                .search_exact(&[10.0, 20.0, 30.0], threshold)
                .is_empty());
            assert!(index
                .search_brute_force(&[10.0, 20.0, 30.0], threshold)
                .is_empty());

            let stats = index.search_stats(&[10.0, 20.0, 30.0], threshold);
            assert_eq!(stats.total_series, 1);
            assert_eq!(stats.trie_threshold, 0);
            assert_eq!(stats.num_candidates, 0);
            assert_eq!(stats.passed_exact, 0);
            assert_eq!(stats.trie_pruning_rate, 1.0);
        }
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

    #[test]
    fn test_invalid_trie_threshold_multipliers_fall_back_to_default() {
        let quant_config = QuantizationConfig::for_u8(0.0, 100.0);
        let msm_config = MsmConfig::new(1.0);
        let mut index: HybridSearchIndex<usize> = HybridSearchIndex::new(quant_config, msm_config);

        for multiplier in [0.0, -1.0, f64::NAN, f64::INFINITY, f64::NEG_INFINITY] {
            index.set_trie_threshold_multiplier(multiplier);
            assert_eq!(
                index.trie_threshold_multiplier,
                DEFAULT_TRIE_THRESHOLD_MULTIPLIER
            );
        }
    }

    #[test]
    fn test_builder_normalizes_invalid_trie_threshold_multiplier() {
        let index: HybridSearchIndex<usize> = HybridSearchIndexBuilder::new()
            .trie_threshold_multiplier(f64::NAN)
            .build();

        assert_eq!(
            index.trie_threshold_multiplier,
            DEFAULT_TRIE_THRESHOLD_MULTIPLIER
        );
    }
}
