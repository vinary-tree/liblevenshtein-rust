//! Time series trie indexing for efficient similarity search.
//!
//! This module provides trie-based indexing for time series using the existing
//! DynamicDawg infrastructure. Time series are encoded as discrete sequences
//! using quantization, then indexed for fast approximate search.
//!
//! # Architecture
//!
//! ```text
//! Time Series → Quantization → Byte Sequence → DynamicDawg
//!    [f64]         Config          [u8]         Index
//! ```
//!
//! # Example
//!
//! ```rust
//! use liblevenshtein::time_series::{TimeSeriesIndex, QuantizationConfig};
//!
//! // Create index with quantization config
//! let config = QuantizationConfig::uniform(0.0, 100.0, 256);
//! let mut index = TimeSeriesIndex::new(config);
//!
//! // Insert time series
//! index.insert(0, &[10.0, 20.0, 30.0]);
//! index.insert(1, &[15.0, 25.0, 35.0]);
//! index.insert(2, &[50.0, 60.0, 70.0]);
//!
//! // Query for similar series (within edit distance 2)
//! let query = vec![12.0, 22.0, 32.0];
//! let candidates = index.search(&query, 2);
//! // Returns IDs of similar series
//! ```
//!
//! # Search Modes
//!
//! - **Approximate search**: Uses Levenshtein distance on quantized sequences.
//!   Fast but may have false positives/negatives due to quantization.
//!
//! - **Hybrid search**: Uses approximate search for candidate generation,
//!   then verifies with exact MSM distance. Accurate but slower.

use super::encoding::QuantizationConfig;
use crate::transducer::transition::{
    initial_state, transition_state_pooled_ref, TransitionSettings,
};
use crate::transducer::{Algorithm, State, StatePool, Unrestricted};
use libdictenstein::dynamic_dawg::DynamicDawg;
use libdictenstein::{Dictionary, DictionaryNode, DictionaryValue, MappedDictionaryNode};
use std::collections::{HashMap, VecDeque};

struct ByteSearchNode<N> {
    node: N,
    state: State,
}

type BucketLocation = (usize, usize);

const DEFAULT_SEARCH_RESULT_CAPACITY: usize = 64;

/// A time series index using quantized trie storage.
///
/// The index stores time series as quantized byte sequences in a DynamicDawg,
/// enabling efficient approximate similarity search using Levenshtein automata.
///
/// # Type Parameters
///
/// - `V`: The value type associated with each time series (default: series ID as usize)
///
/// # Performance Characteristics
///
/// | Operation | Complexity |
/// |-----------|------------|
/// | Insert | O(L) where L = series length |
/// | Search | O(L × D) where D = edit distance threshold |
/// | Memory | Shared prefix compression via DAWG |
#[derive(Debug)]
pub struct TimeSeriesIndex<V: DictionaryValue = usize> {
    /// The underlying DAWG storing quantized sequences
    dawg: DynamicDawg<usize>,

    /// Values grouped by quantized byte sequence.
    buckets: Vec<Vec<V>>,

    /// Current bucket and slot for each indexed value.
    locations: HashMap<V, BucketLocation>,

    /// Quantization configuration
    config: QuantizationConfig,

    /// Original series storage (optional, for verification)
    /// Maps value to original series
    originals: HashMap<V, Vec<f64>>,

    /// Whether to store original series for exact verification
    store_originals: bool,

    /// Number of indexed series
    count: usize,
}

impl<V: DictionaryValue + std::hash::Hash + Eq + Copy> TimeSeriesIndex<V> {
    /// Create a new time series index with the given quantization config.
    ///
    /// # Arguments
    ///
    /// * `config` - Quantization configuration for encoding series. Configs
    ///   wider than 256 bins are coarsened to byte-compatible 256-bin configs.
    ///
    /// # Example
    ///
    /// ```rust
    /// use liblevenshtein::time_series::{TimeSeriesIndex, QuantizationConfig};
    ///
    /// let config = QuantizationConfig::for_u8(0.0, 100.0);
    /// let index: TimeSeriesIndex<usize> = TimeSeriesIndex::new(config);
    /// ```
    pub fn new(config: QuantizationConfig) -> Self {
        Self::with_capacity(config, 0)
    }

    /// Create a new time series index and reserve metadata for `capacity` values.
    ///
    /// This is useful when the caller knows the approximate number of inserts in
    /// advance. The DAWG grows normally, while the bucket table and value-location
    /// map avoid repeated reallocation during bulk loading.
    pub fn with_capacity(config: QuantizationConfig, capacity: usize) -> Self {
        Self::with_options_capacity(config, false, capacity)
    }

    /// Create a new index that also stores original series for exact verification.
    ///
    /// This enables hybrid search that uses approximate candidates for filtering
    /// then exact MSM distance for verification.
    pub fn new_with_verification(config: QuantizationConfig) -> Self {
        Self::with_verification_capacity(config, 0)
    }

    /// Create a verification-capable index and reserve metadata for `capacity` values.
    pub fn with_verification_capacity(config: QuantizationConfig, capacity: usize) -> Self {
        Self::with_options_capacity(config, true, capacity)
    }

    pub(crate) fn with_options_capacity(
        config: QuantizationConfig,
        store_originals: bool,
        capacity: usize,
    ) -> Self {
        let config = config.into_u8_compatible();
        Self {
            dawg: DynamicDawg::new(),
            buckets: Vec::with_capacity(capacity),
            locations: HashMap::with_capacity(capacity),
            config,
            originals: if store_originals {
                HashMap::with_capacity(capacity)
            } else {
                HashMap::new()
            },
            store_originals,
            count: 0,
        }
    }

    /// Get the quantization configuration.
    #[inline]
    pub fn config(&self) -> &QuantizationConfig {
        &self.config
    }

    /// Get the number of indexed series.
    #[inline]
    pub fn len(&self) -> usize {
        self.count
    }

    /// Check if the index is empty.
    #[inline]
    pub fn is_empty(&self) -> bool {
        self.count == 0
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
    /// `true` if the series was newly inserted, `false` if it already existed.
    ///
    /// # Example
    ///
    /// ```rust
    /// use liblevenshtein::time_series::{TimeSeriesIndex, QuantizationConfig};
    ///
    /// let config = QuantizationConfig::for_u8(0.0, 100.0);
    /// let mut index = TimeSeriesIndex::new(config);
    ///
    /// index.insert(0usize, &[10.0, 20.0, 30.0]);
    /// index.insert(1usize, &[15.0, 25.0, 35.0]);
    /// ```
    pub fn insert(&mut self, value: V, series: &[f64]) -> bool {
        let encoded = self.config.encode_u8(series);
        let bucket_id = self.bucket_id_for_encoded(&encoded);

        if let Some((old_bucket_id, old_slot)) = self.locations.get(&value).copied() {
            if old_bucket_id != bucket_id {
                self.remove_from_bucket(value, old_bucket_id, old_slot);
                self.push_to_bucket(value, bucket_id);
            }
            if self.store_originals {
                self.originals.insert(value, series.to_vec());
            }
            return false;
        }

        self.push_to_bucket(value, bucket_id);
        if self.store_originals {
            self.originals.insert(value, series.to_vec());
        }
        self.count += 1;
        true
    }

    fn bucket_id_for_encoded(&mut self, encoded: &[u8]) -> usize {
        if let Some(bucket_id) = self.dawg.get_bytes_value(encoded) {
            return bucket_id;
        }

        let bucket_id = self.buckets.len();
        self.buckets.push(Vec::with_capacity(1));
        let inserted = self.dawg.insert_bytes_with_value(encoded, bucket_id);
        debug_assert!(inserted, "new bucket id must be inserted for a new key");
        bucket_id
    }

    fn push_to_bucket(&mut self, value: V, bucket_id: usize) {
        let slot = self.buckets[bucket_id].len();
        self.buckets[bucket_id].push(value);
        self.locations.insert(value, (bucket_id, slot));
    }

    fn remove_from_bucket(&mut self, value: V, bucket_id: usize, slot: usize) {
        let bucket = &mut self.buckets[bucket_id];
        debug_assert!(bucket.get(slot).copied() == Some(value));
        let removed = bucket.swap_remove(slot);
        debug_assert!(removed == value);

        if let Some(swapped) = bucket.get(slot).copied() {
            self.locations.insert(swapped, (bucket_id, slot));
        } else if bucket.is_empty() {
            *bucket = Vec::new();
        }
    }

    /// Remove a value from the index.
    ///
    /// This removes the value from its quantized bucket and drops any stored
    /// original series. Empty buckets remain addressable from the DAWG so older
    /// bucket IDs stay stable, but their backing storage is released.
    ///
    /// # Returns
    ///
    /// `true` if the value was present and removed, `false` otherwise.
    pub fn remove(&mut self, value: V) -> bool {
        let Some((bucket_id, slot)) = self.locations.remove(&value) else {
            return false;
        };

        self.remove_from_bucket(value, bucket_id, slot);
        self.originals.remove(&value);
        self.count -= 1;
        true
    }

    /// Check if an exact series exists in the index.
    ///
    /// Note: Due to quantization, this checks for the quantized representation.
    pub fn contains(&self, series: &[f64]) -> bool {
        let encoded = self.config.encode_u8(series);
        self.dawg
            .get_bytes_value(&encoded)
            .and_then(|bucket_id| self.buckets.get(bucket_id))
            .is_some_and(|bucket| !bucket.is_empty())
    }

    /// Get the value associated with an exact series match.
    ///
    /// Note: Due to quantization, this looks up the quantized representation.
    pub fn get(&self, series: &[f64]) -> Option<V> {
        let encoded = self.config.encode_u8(series);
        self.dawg
            .get_bytes_value(&encoded)
            .and_then(|bucket_id| self.buckets.get(bucket_id))
            .and_then(|bucket| bucket.first().copied())
    }

    /// Search for similar series within a Levenshtein distance threshold.
    ///
    /// This performs approximate search on the quantized sequences.
    /// The returned candidates may include false positives due to quantization.
    ///
    /// # Arguments
    ///
    /// * `query` - The query time series
    /// * `max_distance` - Maximum edit distance on quantized sequences
    ///
    /// # Returns
    ///
    /// Vector of (value, edit_distance) pairs for matching series.
    ///
    /// # Example
    ///
    /// ```rust
    /// use liblevenshtein::time_series::{TimeSeriesIndex, QuantizationConfig};
    ///
    /// let config = QuantizationConfig::for_u8(0.0, 100.0);
    /// let mut index = TimeSeriesIndex::new(config);
    ///
    /// index.insert(0usize, &[10.0, 20.0, 30.0]);
    /// index.insert(1usize, &[15.0, 25.0, 35.0]);
    ///
    /// let results = index.search(&[12.0, 22.0, 32.0], 3);
    /// ```
    pub fn search(&self, query: &[f64], max_distance: usize) -> Vec<(V, usize)> {
        self.search_with_algorithm(query, max_distance, Algorithm::Standard)
    }

    /// Search using the adjacent-transposition OSA algorithm (restricted Damerau).
    ///
    /// Allows adjacent character transpositions at cost 1.
    pub fn search_transposition(&self, query: &[f64], max_distance: usize) -> Vec<(V, usize)> {
        self.search_with_algorithm(query, max_distance, Algorithm::Transposition)
    }

    /// Search using merge-and-split algorithm.
    ///
    /// Allows merging two adjacent characters or splitting one character into two.
    pub fn search_merge_split(&self, query: &[f64], max_distance: usize) -> Vec<(V, usize)> {
        self.search_with_algorithm(query, max_distance, Algorithm::MergeAndSplit)
    }

    /// Internal search implementation using the specified algorithm.
    fn search_with_algorithm(
        &self,
        query: &[f64],
        max_distance: usize,
        algorithm: Algorithm,
    ) -> Vec<(V, usize)> {
        let encoded = self.config.encode_u8(query);
        if max_distance == 0 {
            let Some(bucket) = self
                .dawg
                .get_bytes_value(&encoded)
                .and_then(|bucket_id| self.buckets.get(bucket_id))
            else {
                return Vec::new();
            };

            let mut results = Vec::with_capacity(bucket.len());
            results.extend(bucket.iter().map(|&value| (value, 0)));
            return results;
        }

        let settings = TransitionSettings::new(max_distance, algorithm, false);
        let mut pending = VecDeque::with_capacity(encoded.len().saturating_add(1));
        let mut results = Vec::with_capacity(self.count.min(DEFAULT_SEARCH_RESULT_CAPACITY));
        let mut state_pool = StatePool::new();

        pending.push_back(ByteSearchNode {
            node: self.dawg.root(),
            state: initial_state(encoded.len(), max_distance, algorithm),
        });

        while let Some(current) = pending.pop_front() {
            if current.node.is_final() {
                let distance = current
                    .state
                    .infer_distance(encoded.len())
                    .unwrap_or(usize::MAX);
                if distance <= max_distance {
                    if let Some(bucket_id) = current.node.value() {
                        if let Some(bucket) = self.buckets.get(bucket_id) {
                            results.extend(bucket.iter().map(|&value| (value, distance)));
                        }
                    }
                }
            }

            for (label, child) in current.node.edges() {
                if let Some(next_state) = transition_state_pooled_ref(
                    &current.state,
                    &mut state_pool,
                    &Unrestricted,
                    label,
                    &encoded,
                    settings,
                ) {
                    pending.push_back(ByteSearchNode {
                        node: child,
                        state: next_state,
                    });
                }
            }
        }

        results
    }

    /// Get candidates for exact MSM verification.
    ///
    /// Returns candidate values along with their original series data.
    /// Only available if the index was created with `new_with_verification`.
    ///
    /// # Arguments
    ///
    /// * `query` - The query time series
    /// * `max_distance` - Maximum edit distance for candidate filtering
    ///
    /// # Returns
    ///
    /// Vector of (value, original_series) pairs for verification.
    pub fn get_candidates_for_verification(
        &self,
        query: &[f64],
        max_distance: usize,
    ) -> Vec<(V, &[f64])> {
        if !self.store_originals {
            return Vec::new();
        }

        let candidates = self.search(query, max_distance);
        let mut verified = Vec::with_capacity(candidates.len());
        for (value, _) in candidates {
            if let Some(series) = self.originals.get(&value) {
                verified.push((value, series.as_slice()));
            }
        }
        verified
    }

    /// Get the original series for a value (if stored).
    pub fn get_original(&self, value: &V) -> Option<&[f64]> {
        self.originals.get(value).map(|v| v.as_slice())
    }

    /// Get statistics about the index.
    pub fn stats(&self) -> TimeSeriesIndexStats {
        TimeSeriesIndexStats {
            num_series: self.count,
            dawg_node_count: self.dawg.node_count(),
            stores_originals: self.store_originals,
            num_bins: self.config.num_bins,
            value_range: (self.config.min_value, self.config.max_value),
        }
    }
}

/// Default implementation for usize values.
impl TimeSeriesIndex<usize> {
    /// Create index and insert multiple series, auto-assigning IDs.
    ///
    /// # Example
    ///
    /// ```rust
    /// use liblevenshtein::time_series::{TimeSeriesIndex, QuantizationConfig};
    ///
    /// let config = QuantizationConfig::for_u8(0.0, 100.0);
    /// let series_data = vec![
    ///     vec![10.0, 20.0, 30.0],
    ///     vec![15.0, 25.0, 35.0],
    ///     vec![50.0, 60.0, 70.0],
    /// ];
    /// let index = TimeSeriesIndex::from_series(config, &series_data);
    /// assert_eq!(index.len(), 3);
    /// ```
    pub fn from_series(config: QuantizationConfig, series_list: &[Vec<f64>]) -> Self {
        let mut index = Self::with_capacity(config, series_list.len());
        for (id, series) in series_list.iter().enumerate() {
            index.insert(id, series);
        }
        index
    }

    /// Create index with verification support from multiple series.
    pub fn from_series_with_verification(
        config: QuantizationConfig,
        series_list: &[Vec<f64>],
    ) -> Self {
        let mut index = Self::with_verification_capacity(config, series_list.len());
        for (id, series) in series_list.iter().enumerate() {
            index.insert(id, series);
        }
        index
    }
}

/// Statistics about a time series index.
#[derive(Debug, Clone)]
pub struct TimeSeriesIndexStats {
    /// Number of indexed series
    pub num_series: usize,

    /// Number of nodes in the underlying DAWG
    pub dawg_node_count: usize,

    /// Whether original series are stored
    pub stores_originals: bool,

    /// Number of quantization bins
    pub num_bins: u32,

    /// Value range (min, max)
    pub value_range: (f64, f64),
}

impl std::fmt::Display for TimeSeriesIndexStats {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        writeln!(f, "TimeSeriesIndex Statistics:")?;
        writeln!(f, "  Series count: {}", self.num_series)?;
        writeln!(f, "  DAWG nodes: {}", self.dawg_node_count)?;
        writeln!(f, "  Stores originals: {}", self.stores_originals)?;
        writeln!(f, "  Quantization bins: {}", self.num_bins)?;
        writeln!(
            f,
            "  Value range: [{:.2}, {:.2}]",
            self.value_range.0, self.value_range.1
        )
    }
}

/// Builder for TimeSeriesIndex with customizable options.
#[derive(Debug, Clone)]
pub struct TimeSeriesIndexBuilder {
    config: QuantizationConfig,
    store_originals: bool,
    capacity: usize,
}

impl TimeSeriesIndexBuilder {
    /// Create a new builder.
    pub fn new() -> Self {
        Self {
            config: QuantizationConfig::default(),
            store_originals: false,
            capacity: 0,
        }
    }

    /// Set the quantization configuration.
    pub fn config(mut self, config: QuantizationConfig) -> Self {
        self.config = config;
        self
    }

    /// Set quantization parameters directly.
    ///
    /// Invalid parameters leave the current configuration unchanged.
    pub fn quantization(mut self, min: f64, max: f64, bins: u32) -> Self {
        if let Some(config) = QuantizationConfig::try_uniform(min, max, bins) {
            self.config = config;
        }
        self
    }

    /// Enable storing original series for verification.
    pub fn with_verification(mut self) -> Self {
        self.store_originals = true;
        self
    }

    /// Reserve metadata capacity for at least `capacity` indexed values.
    pub fn capacity(mut self, capacity: usize) -> Self {
        self.capacity = capacity;
        self
    }

    /// Auto-configure quantization from sample data.
    ///
    /// If the sample is empty, non-finite, or degenerate, the current
    /// configuration is kept.
    pub fn auto_config(mut self, sample_data: &[f64], bins: u32, margin: f64) -> Self {
        if let Some(config) = QuantizationConfig::from_data(sample_data, bins, margin) {
            self.config = config;
        }
        self
    }

    /// Build the index.
    ///
    /// Uses [`QuantizationConfig::default`] unless a configuration was provided.
    pub fn build<V: DictionaryValue + std::hash::Hash + Eq + Copy>(self) -> TimeSeriesIndex<V> {
        if self.store_originals {
            TimeSeriesIndex::with_verification_capacity(self.config, self.capacity)
        } else {
            TimeSeriesIndex::with_capacity(self.config, self.capacity)
        }
    }
}

impl Default for TimeSeriesIndexBuilder {
    fn default() -> Self {
        Self::new()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_index_creation() {
        let config = QuantizationConfig::for_u8(0.0, 100.0);
        let index: TimeSeriesIndex<usize> = TimeSeriesIndex::new(config);
        assert!(index.is_empty());
        assert_eq!(index.len(), 0);
    }

    #[test]
    fn test_insert_and_contains() {
        let config = QuantizationConfig::for_u8(0.0, 100.0);
        let mut index = TimeSeriesIndex::new(config);

        let series = vec![10.0, 20.0, 30.0];
        assert!(index.insert(0usize, &series));
        assert!(!index.is_empty());
        assert_eq!(index.len(), 1);

        // Contains the quantized version
        assert!(index.contains(&series));

        // The exact same values should always match
        assert!(index.contains(&[10.0, 20.0, 30.0]));
    }

    #[test]
    fn test_insert_duplicate() {
        let config = QuantizationConfig::for_u8(0.0, 100.0);
        let mut index = TimeSeriesIndex::new(config);

        let series = vec![10.0, 20.0, 30.0];
        assert!(index.insert(0usize, &series));
        // The same quantized series can carry multiple distinct values.
        assert!(index.insert(1usize, &series));
        assert_eq!(index.len(), 2);

        let mut found_ids: Vec<_> = index
            .search(&series, 0)
            .into_iter()
            .map(|(id, _)| id)
            .collect();
        found_ids.sort_unstable();
        assert_eq!(found_ids, vec![0, 1]);

        // Different series should insert as new
        let different_series = vec![50.0, 60.0, 70.0];
        assert!(index.insert(2usize, &different_series));
        assert_eq!(index.len(), 3);
    }

    #[test]
    fn test_get() {
        let config = QuantizationConfig::for_u8(0.0, 100.0);
        let mut index = TimeSeriesIndex::new(config);

        index.insert(42usize, &[10.0, 20.0, 30.0]);
        assert_eq!(index.get(&[10.0, 20.0, 30.0]), Some(42));
        assert_eq!(index.get(&[99.0, 99.0, 99.0]), None);
    }

    #[test]
    fn test_search() {
        let config = QuantizationConfig::for_u8(0.0, 100.0);
        let mut index = TimeSeriesIndex::new(config);

        // Insert several series
        index.insert(0usize, &[10.0, 20.0, 30.0]);
        index.insert(1usize, &[15.0, 25.0, 35.0]);
        index.insert(2usize, &[50.0, 60.0, 70.0]);

        // Search for similar series
        let results = index.search(&[12.0, 22.0, 32.0], 10);

        // Should find series 0 and 1 (close), but not 2 (far)
        let found_ids: Vec<usize> = results.iter().map(|(id, _)| *id).collect();
        assert!(found_ids.contains(&0));
        assert!(found_ids.contains(&1));
    }

    #[test]
    fn test_search_exact_match() {
        let config = QuantizationConfig::for_u8(0.0, 100.0);
        let mut index = TimeSeriesIndex::new(config);

        index.insert(0usize, &[10.0, 20.0, 30.0]);

        // Exact match should have distance 0
        let results = index.search(&[10.0, 20.0, 30.0], 0);
        assert_eq!(results.len(), 1);
        assert_eq!(results[0].0, 0);
        assert_eq!(results[0].1, 0);
    }

    #[test]
    fn test_search_exact_match_high_bins() {
        let config = QuantizationConfig::for_u8(0.0, 100.0);
        let mut index = TimeSeriesIndex::new(config);

        index.insert(0usize, &[90.0, 95.0, 99.0]);

        let results = index.search(&[90.0, 95.0, 99.0], 0);
        assert_eq!(results, vec![(0, 0)]);
    }

    #[test]
    fn test_with_verification() {
        let config = QuantizationConfig::for_u8(0.0, 100.0);
        let mut index = TimeSeriesIndex::new_with_verification(config);

        let series1 = vec![10.0, 20.0, 30.0];
        let series2 = vec![15.0, 25.0, 35.0];

        index.insert(0usize, &series1);
        index.insert(1usize, &series2);

        // Should be able to retrieve originals
        assert_eq!(index.get_original(&0), Some(series1.as_slice()));
        assert_eq!(index.get_original(&1), Some(series2.as_slice()));
    }

    #[test]
    fn quantization_collisions_recover_all_values_and_originals() {
        let config = QuantizationConfig::uniform(0.0, 100.0, 10);
        let mut index = TimeSeriesIndex::new_with_verification(config);

        let series1 = vec![10.0, 20.0, 30.0];
        let series2 = vec![10.1, 20.1, 30.1];

        assert!(index.insert(7usize, &series1));
        assert!(index.insert(9usize, &series2));
        assert_eq!(index.len(), 2);

        let mut results: Vec<_> = index.search(&series1, 0).into_iter().collect();
        results.sort_unstable();
        assert_eq!(results, vec![(7, 0), (9, 0)]);

        let mut candidates: Vec<_> = index
            .get_candidates_for_verification(&series1, 0)
            .into_iter()
            .map(|(id, series)| (id, series.to_vec()))
            .collect();
        candidates.sort_unstable_by_key(|(id, _)| *id);
        assert_eq!(candidates, vec![(7, series1), (9, series2)]);
    }

    #[test]
    fn reinserting_value_relocates_bucket_membership() {
        let config = QuantizationConfig::uniform(0.0, 100.0, 10);
        let mut index = TimeSeriesIndex::new_with_verification(config);

        assert!(index.insert(7usize, &[10.0]));
        assert!(!index.insert(7usize, &[90.0]));
        assert_eq!(index.len(), 1);

        assert!(!index.contains(&[10.0]));
        assert!(index.contains(&[90.0]));
        assert!(index.search(&[10.0], 0).is_empty());
        assert_eq!(index.search(&[90.0], 0), vec![(7, 0)]);
        assert_eq!(index.get_original(&7), Some(&[90.0][..]));
    }

    #[test]
    fn remove_value_clears_bucket_membership_and_original() {
        let config = QuantizationConfig::uniform(0.0, 100.0, 10);
        let mut index = TimeSeriesIndex::new_with_verification(config);

        assert!(index.insert(7usize, &[10.0, 20.0]));
        let bucket_id = index.locations[&7].0;
        assert!(index.buckets[bucket_id].capacity() > 0);
        assert_eq!(index.len(), 1);
        assert!(index.remove(7));
        assert!(!index.remove(7));

        assert_eq!(index.len(), 0);
        assert!(!index.contains(&[10.0, 20.0]));
        assert!(index.search(&[10.0, 20.0], 0).is_empty());
        assert_eq!(index.get(&[10.0, 20.0]), None);
        assert_eq!(index.get_original(&7), None);
        assert_eq!(index.buckets[bucket_id].capacity(), 0);
    }

    #[test]
    fn remove_value_preserves_swapped_bucket_location() {
        let config = QuantizationConfig::uniform(0.0, 100.0, 10);
        let mut index = TimeSeriesIndex::new_with_verification(config);

        assert!(index.insert(7usize, &[10.0]));
        assert!(index.insert(9usize, &[10.01]));
        assert_eq!(index.len(), 2);

        assert!(index.remove(7));
        assert_eq!(index.len(), 1);
        assert_eq!(index.search(&[10.0], 0), vec![(9, 0)]);

        assert!(!index.insert(9usize, &[90.0]));
        assert!(index.search(&[10.0], 0).is_empty());
        assert_eq!(index.search(&[90.0], 0), vec![(9, 0)]);
        assert_eq!(index.get_original(&9), Some(&[90.0][..]));
    }

    #[test]
    fn test_get_candidates_for_verification() {
        let config = QuantizationConfig::for_u8(0.0, 100.0);
        let mut index = TimeSeriesIndex::new_with_verification(config);

        index.insert(0usize, &[10.0, 20.0, 30.0]);
        index.insert(1usize, &[15.0, 25.0, 35.0]);

        let candidates = index.get_candidates_for_verification(&[12.0, 22.0, 32.0], 10);
        assert!(!candidates.is_empty());

        // Each candidate should have its original series
        for (_, series) in &candidates {
            assert_eq!(series.len(), 3);
        }
    }

    #[test]
    fn test_from_series() {
        let config = QuantizationConfig::for_u8(0.0, 100.0);
        let series_data = vec![
            vec![10.0, 20.0, 30.0],
            vec![15.0, 25.0, 35.0],
            vec![50.0, 60.0, 70.0],
        ];

        let index = TimeSeriesIndex::from_series(config, &series_data);
        assert_eq!(index.len(), 3);
    }

    #[test]
    fn from_series_reserves_index_metadata() {
        let config = QuantizationConfig::for_u8(0.0, 100.0);
        let series_data = vec![
            vec![10.0, 20.0, 30.0],
            vec![15.0, 25.0, 35.0],
            vec![50.0, 60.0, 70.0],
        ];

        let index: TimeSeriesIndex<usize> = TimeSeriesIndex::with_capacity(config.clone(), 8);
        assert!(index.buckets.capacity() >= 8);
        assert!(index.locations.capacity() >= 8);

        let index = TimeSeriesIndex::from_series(config.clone(), &series_data);
        assert!(index.buckets.capacity() >= series_data.len());
        assert!(index.locations.capacity() >= series_data.len());

        let index = TimeSeriesIndex::from_series_with_verification(config, &series_data);
        assert!(index.buckets.capacity() >= series_data.len());
        assert!(index.locations.capacity() >= series_data.len());
        assert!(index.originals.capacity() >= series_data.len());

        let index: TimeSeriesIndex<usize> = TimeSeriesIndexBuilder::new()
            .capacity(8)
            .with_verification()
            .build();
        assert!(index.buckets.capacity() >= 8);
        assert!(index.locations.capacity() >= 8);
        assert!(index.originals.capacity() >= 8);
    }

    #[test]
    fn test_stats() {
        let config = QuantizationConfig::for_u8(0.0, 100.0);
        let mut index = TimeSeriesIndex::new(config);

        index.insert(0usize, &[10.0, 20.0, 30.0]);
        index.insert(1usize, &[15.0, 25.0, 35.0]);

        let stats = index.stats();
        assert_eq!(stats.num_series, 2);
        assert!(stats.dawg_node_count > 0);
        assert!(!stats.stores_originals);
        assert_eq!(stats.num_bins, 256);
    }

    #[test]
    fn test_builder() {
        let index: TimeSeriesIndex<usize> = TimeSeriesIndexBuilder::new()
            .quantization(0.0, 100.0, 256)
            .with_verification()
            .build();

        assert!(index.is_empty());
        assert!(index.store_originals);
    }

    #[test]
    fn test_builder_default_config() {
        let index: TimeSeriesIndex<usize> = TimeSeriesIndexBuilder::new().build();

        assert!(index.is_empty());
        assert!(!index.store_originals);
        assert_eq!(index.config().min_value, 0.0);
        assert_eq!(index.config().max_value, 1.0);
        assert_eq!(index.config().num_bins, 256);
    }

    #[test]
    fn test_builder_auto_config_keeps_existing_config_for_invalid_data() {
        let index: TimeSeriesIndex<usize> = TimeSeriesIndexBuilder::new()
            .quantization(-1.0, 1.0, 64)
            .auto_config(&[], 256, 0.1)
            .build();

        assert_eq!(index.config().min_value, -1.0);
        assert_eq!(index.config().max_value, 1.0);
        assert_eq!(index.config().num_bins, 64);
    }

    #[test]
    fn test_builder_quantization_keeps_existing_config_for_invalid_parameters() {
        let index: TimeSeriesIndex<usize> = TimeSeriesIndexBuilder::new()
            .quantization(-1.0, 1.0, 64)
            .quantization(f64::NAN, 1.0, 10)
            .quantization(0.0, f64::INFINITY, 10)
            .quantization(10.0, 0.0, 10)
            .quantization(0.0, 1.0, 0)
            .build();

        assert_eq!(index.config().min_value, -1.0);
        assert_eq!(index.config().max_value, 1.0);
        assert_eq!(index.config().num_bins, 64);
    }

    #[test]
    fn test_new_coarsens_large_quantizer_to_byte_bins() {
        let mut index: TimeSeriesIndex<usize> =
            TimeSeriesIndex::new(QuantizationConfig::for_u16(0.0, 100.0));

        assert_eq!(index.config().num_bins, 256);
        assert!(index.insert(7, &[10.0, 20.0, 30.0]));
        assert!(index.contains(&[10.0, 20.0, 30.0]));
    }

    #[test]
    fn test_search_transposition() {
        let config = QuantizationConfig::for_u8(0.0, 100.0);
        let mut index = TimeSeriesIndex::new(config);

        // Series with swapped elements
        index.insert(0usize, &[10.0, 30.0, 20.0]); // 10, 30, 20
        index.insert(1usize, &[10.0, 20.0, 30.0]); // 10, 20, 30

        // Transposition search should find both with low distance
        let results = index.search_transposition(&[10.0, 20.0, 30.0], 2);
        assert!(!results.is_empty());
    }

    #[test]
    fn test_empty_series() {
        let config = QuantizationConfig::for_u8(0.0, 100.0);
        let mut index = TimeSeriesIndex::new(config);

        index.insert(0usize, &[]);
        assert_eq!(index.len(), 1);
        assert!(index.contains(&[]));
    }

    #[test]
    fn test_single_element_series() {
        let config = QuantizationConfig::for_u8(0.0, 100.0);
        let mut index = TimeSeriesIndex::new(config);

        index.insert(0usize, &[50.0]);
        index.insert(1usize, &[55.0]);

        // Exact match test - contains should work
        assert!(index.contains(&[50.0]));
        assert!(index.contains(&[55.0]));

        // Verify the index has 2 entries
        assert_eq!(index.len(), 2);
    }
}
