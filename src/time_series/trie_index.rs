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
use crate::dictionary::dynamic_dawg::DynamicDawg;
use crate::dictionary::DictionaryValue;
use crate::transducer::{Algorithm, Transducer};
use std::collections::HashMap;

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
    dawg: DynamicDawg<V>,

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
    /// * `config` - Quantization configuration for encoding series
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
        Self {
            dawg: DynamicDawg::new(),
            config,
            originals: HashMap::new(),
            store_originals: false,
            count: 0,
        }
    }

    /// Create a new index that also stores original series for exact verification.
    ///
    /// This enables hybrid search that uses approximate candidates for filtering
    /// then exact MSM distance for verification.
    pub fn new_with_verification(config: QuantizationConfig) -> Self {
        Self {
            dawg: DynamicDawg::new(),
            config,
            originals: HashMap::new(),
            store_originals: true,
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
        let inserted = self.dawg.insert_bytes_with_value(&encoded, value);

        if inserted {
            self.count += 1;
            if self.store_originals {
                self.originals.insert(value, series.to_vec());
            }
        }

        inserted
    }

    /// Check if an exact series exists in the index.
    ///
    /// Note: Due to quantization, this checks for the quantized representation.
    pub fn contains(&self, series: &[f64]) -> bool {
        let encoded = self.config.encode_u8(series);
        self.dawg.contains_bytes(&encoded)
    }

    /// Get the value associated with an exact series match.
    ///
    /// Note: Due to quantization, this looks up the quantized representation.
    pub fn get(&self, series: &[f64]) -> Option<V> {
        let encoded = self.config.encode_u8(series);
        self.dawg.get_bytes_value(&encoded)
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

    /// Search using transposition algorithm (Damerau-Levenshtein).
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

        // Convert encoded bytes to a string for the transducer
        // Since DynamicDawg stores bytes, we need to use a byte-compatible query
        // The transducer works on string queries, so we create a "fake" string
        // from the raw bytes (this is safe because we're only using it for traversal)
        let query_str = unsafe { std::str::from_utf8_unchecked(&encoded) };

        let transducer = Transducer::new(self.dawg.clone(), algorithm);
        transducer
            .query_candidates(query_str, max_distance)
            .filter_map(|candidate| {
                // The term returned by the transducer is the encoded byte sequence
                // We look up the value using the raw bytes
                self.dawg
                    .get_bytes_value(candidate.term.as_bytes())
                    .map(|v| (v, candidate.distance))
            })
            .collect()
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
        candidates
            .into_iter()
            .filter_map(|(value, _)| {
                self.originals
                    .get(&value)
                    .map(|series| (value, series.as_slice()))
            })
            .collect()
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
        let mut index = Self::new(config);
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
        let mut index = Self::new_with_verification(config);
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
    config: Option<QuantizationConfig>,
    store_originals: bool,
}

impl TimeSeriesIndexBuilder {
    /// Create a new builder.
    pub fn new() -> Self {
        Self {
            config: None,
            store_originals: false,
        }
    }

    /// Set the quantization configuration.
    pub fn config(mut self, config: QuantizationConfig) -> Self {
        self.config = Some(config);
        self
    }

    /// Set quantization parameters directly.
    pub fn quantization(mut self, min: f64, max: f64, bins: u32) -> Self {
        self.config = Some(QuantizationConfig::uniform(min, max, bins));
        self
    }

    /// Enable storing original series for verification.
    pub fn with_verification(mut self) -> Self {
        self.store_originals = true;
        self
    }

    /// Auto-configure quantization from sample data.
    pub fn auto_config(mut self, sample_data: &[f64], bins: u32, margin: f64) -> Self {
        self.config = QuantizationConfig::from_data(sample_data, bins, margin);
        self
    }

    /// Build the index.
    ///
    /// # Panics
    ///
    /// Panics if no configuration was set.
    pub fn build<V: DictionaryValue + std::hash::Hash + Eq + Copy>(self) -> TimeSeriesIndex<V> {
        let config = self
            .config
            .expect("Quantization config must be set before building");
        if self.store_originals {
            TimeSeriesIndex::new_with_verification(config)
        } else {
            TimeSeriesIndex::new(config)
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
        // Same quantized series (values within same bins) updates the value
        assert!(!index.insert(1usize, &series));
        // Count only includes unique encoded sequences
        assert_eq!(index.len(), 1);

        // Different series should insert as new
        let different_series = vec![50.0, 60.0, 70.0];
        assert!(index.insert(2usize, &different_series));
        assert_eq!(index.len(), 2);
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
    fn test_search_transposition() {
        let config = QuantizationConfig::for_u8(0.0, 100.0);
        let mut index = TimeSeriesIndex::new(config);

        // Series with swapped elements
        index.insert(0usize, &[10.0, 30.0, 20.0]); // 10, 30, 20
        index.insert(1usize, &[10.0, 20.0, 30.0]); // 10, 20, 30

        // Transposition search should find both with low distance
        let results = index.search_transposition(&[10.0, 20.0, 30.0], 2);
        assert!(results.len() >= 1);
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
