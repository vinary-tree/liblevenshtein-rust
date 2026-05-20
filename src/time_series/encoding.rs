//! Time series encoding for trie-based indexing.
//!
//! This module provides utilities for encoding time series as discrete sequences
//! that can be indexed using the existing trie infrastructure (DynamicDawg,
//! DynamicDawgChar, DoubleArrayTrie, etc.).
//!
//! # Encoding Approaches
//!
//! ## 1. Quantization (Lossy)
//!
//! Maps continuous float values to discrete bins:
//!
//! ```rust
//! use liblevenshtein::time_series::QuantizationConfig;
//!
//! let config = QuantizationConfig::uniform(0.0, 100.0, 256);
//! let series = vec![10.5, 25.3, 50.0, 75.8];
//! let encoded = config.encode_u8(&series);
//! // encoded: [26, 64, 127, 193] (approximate bin indices)
//! ```
//!
//! ## 2. Direct Float Encoding (Lossless)
//!
//! Stores float bit patterns directly as u32:
//!
//! ```rust
//! use liblevenshtein::time_series::float_encoding;
//!
//! let series = vec![1.0f32, 2.5, 3.14159];
//! let encoded = float_encoding::encode_f32_series(&series);
//! let decoded = float_encoding::decode_f32_series(&encoded);
//! assert_eq!(series, decoded);
//! ```
//!
//! # Choosing an Encoding
//!
//! | Encoding | Precision | Alphabet Size | Best For |
//! |----------|-----------|---------------|----------|
//! | Quantization u8 | Low (256 levels) | 256 | Fast approximate search |
//! | Quantization u16 | Medium (65K levels) | 65536 | Balance of speed/precision |
//! | Direct f32 | Exact | 2^32 | Exact matching, small datasets |
//!
//! # Integration with Tries
//!
//! - `encode_u8()` → Use with `DynamicDawg` (byte sequences)
//! - `encode_u32()` → Use with `DynamicDawgChar` (u32 sequences)

use std::fmt;

/// Configuration for quantizing time series values into discrete bins.
///
/// Quantization maps continuous float values to integer bins, enabling
/// use of discrete sequence data structures like tries.
///
/// # Quantization Schemes
///
/// - **Uniform**: Equal-width bins across the value range
/// - **Custom**: User-defined bin edges for non-uniform distributions
///
/// # Example
///
/// ```rust
/// use liblevenshtein::time_series::QuantizationConfig;
///
/// // Uniform quantization with 256 bins
/// let config = QuantizationConfig::uniform(0.0, 100.0, 256);
///
/// // Quantize a value
/// let bin = config.quantize(50.0);
/// assert_eq!(bin, 128); // 50.0 / (100.0/256) = 128
///
/// // Dequantize back to approximate value
/// let approx = config.dequantize(128);
/// assert!((approx - 50.0).abs() < 0.5);
/// ```
#[derive(Debug, Clone)]
pub struct QuantizationConfig {
    /// Minimum value in the expected range
    pub min_value: f64,

    /// Maximum value in the expected range
    pub max_value: f64,

    /// Number of quantization bins
    pub num_bins: u32,

    /// Bin width (computed from range and num_bins)
    bin_width: f64,

    /// Whether to clamp out-of-range values or use special bins
    pub clamp_outliers: bool,
}

impl QuantizationConfig {
    /// Create a uniform quantization configuration.
    ///
    /// # Arguments
    ///
    /// * `min_value` - Minimum expected value
    /// * `max_value` - Maximum expected value
    /// * `num_bins` - Number of quantization bins (max 2^32 - 1)
    ///
    /// # Panics
    ///
    /// Panics if `min_value >= max_value` or `num_bins == 0`.
    ///
    /// # Example
    ///
    /// ```rust
    /// use liblevenshtein::time_series::QuantizationConfig;
    ///
    /// let config = QuantizationConfig::uniform(0.0, 100.0, 256);
    /// ```
    pub fn uniform(min_value: f64, max_value: f64, num_bins: u32) -> Self {
        assert!(
            min_value < max_value,
            "min_value ({}) must be less than max_value ({})",
            min_value,
            max_value
        );
        assert!(num_bins > 0, "num_bins must be positive");

        let bin_width = (max_value - min_value) / num_bins as f64;

        Self {
            min_value,
            max_value,
            num_bins,
            bin_width,
            clamp_outliers: true,
        }
    }

    /// Create a configuration for byte encoding (256 bins).
    ///
    /// This is optimized for use with `DynamicDawg` which uses byte sequences.
    ///
    /// # Example
    ///
    /// ```rust
    /// use liblevenshtein::time_series::QuantizationConfig;
    ///
    /// let config = QuantizationConfig::for_u8(0.0, 100.0);
    /// assert_eq!(config.num_bins, 256);
    /// ```
    #[inline]
    pub fn for_u8(min_value: f64, max_value: f64) -> Self {
        Self::uniform(min_value, max_value, 256)
    }

    /// Create a configuration for u16 encoding (65536 bins).
    ///
    /// # Example
    ///
    /// ```rust
    /// use liblevenshtein::time_series::QuantizationConfig;
    ///
    /// let config = QuantizationConfig::for_u16(0.0, 100.0);
    /// assert_eq!(config.num_bins, 65536);
    /// ```
    #[inline]
    pub fn for_u16(min_value: f64, max_value: f64) -> Self {
        Self::uniform(min_value, max_value, 65536)
    }

    /// Create a configuration from data, automatically determining range.
    ///
    /// # Arguments
    ///
    /// * `data` - Sample data to determine value range
    /// * `num_bins` - Number of quantization bins
    /// * `margin` - Percentage margin to add to range (e.g., 0.1 for 10%)
    ///
    /// # Returns
    ///
    /// `None` if data is empty or contains only one unique value.
    ///
    /// # Example
    ///
    /// ```rust
    /// use liblevenshtein::time_series::QuantizationConfig;
    ///
    /// let data = vec![10.0, 20.0, 30.0, 40.0, 50.0];
    /// let config = QuantizationConfig::from_data(&data, 256, 0.1).unwrap();
    /// assert!(config.min_value < 10.0); // Has margin
    /// assert!(config.max_value > 50.0); // Has margin
    /// ```
    pub fn from_data(data: &[f64], num_bins: u32, margin: f64) -> Option<Self> {
        if data.is_empty() {
            return None;
        }

        let mut min_val = f64::INFINITY;
        let mut max_val = f64::NEG_INFINITY;

        for &v in data {
            if v.is_finite() {
                min_val = min_val.min(v);
                max_val = max_val.max(v);
            }
        }

        if !min_val.is_finite() || !max_val.is_finite() || min_val >= max_val {
            return None;
        }

        let range = max_val - min_val;
        let margin_amount = range * margin;

        Some(Self::uniform(
            min_val - margin_amount,
            max_val + margin_amount,
            num_bins,
        ))
    }

    /// Get the bin width.
    #[inline]
    pub fn bin_width(&self) -> f64 {
        self.bin_width
    }

    /// Quantize a single value to a bin index.
    ///
    /// # Returns
    ///
    /// Bin index in range `[0, num_bins)`.
    /// Out-of-range values are clamped if `clamp_outliers` is true.
    ///
    /// # Example
    ///
    /// ```rust
    /// use liblevenshtein::time_series::QuantizationConfig;
    ///
    /// let config = QuantizationConfig::uniform(0.0, 100.0, 100);
    /// assert_eq!(config.quantize(0.0), 0);
    /// assert_eq!(config.quantize(50.0), 50);
    /// assert_eq!(config.quantize(99.9), 99);
    /// assert_eq!(config.quantize(100.0), 99); // Clamped to max bin
    /// ```
    #[inline]
    pub fn quantize(&self, value: f64) -> u32 {
        if self.clamp_outliers {
            if value <= self.min_value {
                return 0;
            }
            if value >= self.max_value {
                return self.num_bins - 1;
            }
        }

        let normalized = (value - self.min_value) / self.bin_width;
        let bin = normalized.floor() as u32;

        // Clamp to valid range
        bin.min(self.num_bins - 1)
    }

    /// Quantize to u8 (for DynamicDawg compatibility).
    ///
    /// # Panics
    ///
    /// Panics if `num_bins > 256`.
    #[inline]
    pub fn quantize_u8(&self, value: f64) -> u8 {
        assert!(
            self.num_bins <= 256,
            "Cannot encode {} bins as u8 (max 256)",
            self.num_bins
        );
        self.quantize(value) as u8
    }

    /// Quantize to u16.
    ///
    /// # Panics
    ///
    /// Panics if `num_bins > 65536`.
    #[inline]
    pub fn quantize_u16(&self, value: f64) -> u16 {
        assert!(
            self.num_bins <= 65536,
            "Cannot encode {} bins as u16 (max 65536)",
            self.num_bins
        );
        self.quantize(value) as u16
    }

    /// Dequantize a bin index back to an approximate value.
    ///
    /// Returns the center of the bin.
    ///
    /// # Example
    ///
    /// ```rust
    /// use liblevenshtein::time_series::QuantizationConfig;
    ///
    /// let config = QuantizationConfig::uniform(0.0, 100.0, 100);
    /// let bin = config.quantize(50.5);
    /// let approx = config.dequantize(bin);
    /// assert!((approx - 50.5).abs() < 1.0);
    /// ```
    #[inline]
    pub fn dequantize(&self, bin: u32) -> f64 {
        let bin = bin.min(self.num_bins - 1);
        self.min_value + (bin as f64 + 0.5) * self.bin_width
    }

    /// Encode a time series as u8 bytes.
    ///
    /// For use with `DynamicDawg`.
    ///
    /// # Example
    ///
    /// ```rust
    /// use liblevenshtein::time_series::QuantizationConfig;
    ///
    /// let config = QuantizationConfig::for_u8(0.0, 100.0);
    /// let series = vec![0.0, 50.0, 100.0];
    /// let encoded = config.encode_u8(&series);
    /// assert_eq!(encoded.len(), 3);
    /// ```
    pub fn encode_u8(&self, series: &[f64]) -> Vec<u8> {
        series.iter().map(|&v| self.quantize_u8(v)).collect()
    }

    /// Encode a time series as u32 values.
    ///
    /// For use with `DynamicDawgChar`.
    pub fn encode_u32(&self, series: &[f64]) -> Vec<u32> {
        series.iter().map(|&v| self.quantize(v)).collect()
    }

    /// Decode a u8-encoded series back to approximate values.
    pub fn decode_u8(&self, encoded: &[u8]) -> Vec<f64> {
        encoded
            .iter()
            .map(|&bin| self.dequantize(bin as u32))
            .collect()
    }

    /// Decode a u32-encoded series back to approximate values.
    pub fn decode_u32(&self, encoded: &[u32]) -> Vec<f64> {
        encoded.iter().map(|&bin| self.dequantize(bin)).collect()
    }

    /// Compute the maximum quantization error.
    ///
    /// This is half the bin width - the maximum difference between
    /// an original value and its dequantized approximation.
    #[inline]
    pub fn max_error(&self) -> f64 {
        self.bin_width / 2.0
    }

    /// Compute the edit distance between bins that represents a given value difference.
    ///
    /// This helps map MSM costs to Levenshtein distances on quantized sequences.
    ///
    /// # Example
    ///
    /// ```rust
    /// use liblevenshtein::time_series::QuantizationConfig;
    ///
    /// let config = QuantizationConfig::uniform(0.0, 100.0, 100);
    /// // Values differing by 10.0 span approximately 10 bins
    /// let bin_diff = config.value_diff_to_bins(10.0);
    /// assert_eq!(bin_diff, 10);
    /// ```
    #[inline]
    pub fn value_diff_to_bins(&self, value_diff: f64) -> u32 {
        (value_diff.abs() / self.bin_width).ceil() as u32
    }
}

impl Default for QuantizationConfig {
    /// Default configuration: 256 bins for range [0, 1].
    fn default() -> Self {
        Self::for_u8(0.0, 1.0)
    }
}

impl fmt::Display for QuantizationConfig {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(
            f,
            "Quantization([{:.2}, {:.2}], {} bins, width={:.4})",
            self.min_value, self.max_value, self.num_bins, self.bin_width
        )
    }
}

/// Direct float encoding utilities for lossless representation.
///
/// These functions encode floats as their bit patterns, which can be stored
/// in u32-based tries. This is lossless but results in a large alphabet.
///
/// # Caveats
///
/// - IEEE 754 bit patterns don't sort lexicographically
/// - Negative numbers are bit-reversed relative to positives
/// - NaN values have multiple representations
///
/// This is suitable for exact matching but not for range queries.
pub mod float_encoding {
    /// Encode an f32 value as its bit pattern (u32).
    #[inline]
    pub fn encode_f32(value: f32) -> u32 {
        value.to_bits()
    }

    /// Decode a u32 bit pattern back to f32.
    #[inline]
    pub fn decode_f32(bits: u32) -> f32 {
        f32::from_bits(bits)
    }

    /// Encode an f64 value as its bit pattern (u64).
    #[inline]
    pub fn encode_f64(value: f64) -> u64 {
        value.to_bits()
    }

    /// Decode a u64 bit pattern back to f64.
    #[inline]
    pub fn decode_f64(bits: u64) -> f64 {
        f64::from_bits(bits)
    }

    /// Encode an f32 series as u32 bit patterns.
    ///
    /// For use with `DynamicDawgChar`.
    ///
    /// # Example
    ///
    /// ```rust
    /// use liblevenshtein::time_series::float_encoding;
    ///
    /// let series = vec![1.0f32, 2.5, 3.14159];
    /// let encoded = float_encoding::encode_f32_series(&series);
    /// assert_eq!(encoded.len(), 3);
    /// ```
    pub fn encode_f32_series(series: &[f32]) -> Vec<u32> {
        series.iter().map(|&v| encode_f32(v)).collect()
    }

    /// Decode a u32 series back to f32.
    pub fn decode_f32_series(encoded: &[u32]) -> Vec<f32> {
        encoded.iter().map(|&bits| decode_f32(bits)).collect()
    }

    /// Encode an f64 series as pairs of u32 (high, low).
    ///
    /// This doubles the sequence length but allows f64 storage in u32-based tries.
    pub fn encode_f64_series_as_u32_pairs(series: &[f64]) -> Vec<u32> {
        let mut encoded = Vec::with_capacity(series.len() * 2);
        for &v in series {
            let bits = encode_f64(v);
            encoded.push((bits >> 32) as u32); // High 32 bits
            encoded.push(bits as u32); // Low 32 bits
        }
        encoded
    }

    /// Decode pairs of u32 back to f64.
    pub fn decode_u32_pairs_to_f64(encoded: &[u32]) -> Vec<f64> {
        encoded
            .chunks_exact(2)
            .map(|pair| {
                let high = (pair[0] as u64) << 32;
                let low = pair[1] as u64;
                decode_f64(high | low)
            })
            .collect()
    }

    /// Order-preserving encoding for non-negative f32.
    ///
    /// This encoding preserves the natural ordering of non-negative floats,
    /// making it suitable for range queries in tries.
    ///
    /// # Note
    ///
    /// Only works correctly for values >= 0.0. Negative values should be
    /// handled separately or transformed.
    #[inline]
    pub fn encode_f32_ordered(value: f32) -> u32 {
        debug_assert!(value >= 0.0, "encode_f32_ordered requires non-negative values");
        let bits = value.to_bits();
        // For non-negative floats, the bit pattern is already ordered correctly
        bits
    }

    /// Order-preserving encoding for all f32 values.
    ///
    /// Transforms the IEEE 754 bit pattern so that the resulting u32 values
    /// sort in the same order as the original floats.
    #[inline]
    pub fn encode_f32_total_order(value: f32) -> u32 {
        let bits = value.to_bits();
        // If negative (sign bit set), flip all bits
        // If positive, flip only the sign bit
        if (bits as i32) < 0 {
            !bits
        } else {
            bits ^ 0x8000_0000
        }
    }

    /// Decode an order-preserving encoded u32 back to f32.
    #[inline]
    pub fn decode_f32_total_order(encoded: u32) -> f32 {
        // Reverse the transformation
        let bits = if (encoded & 0x8000_0000) != 0 {
            encoded ^ 0x8000_0000
        } else {
            !encoded
        };
        f32::from_bits(bits)
    }
}

/// Delta encoding for time series.
///
/// Instead of encoding absolute values, encode the differences between
/// consecutive values. This can reduce the effective alphabet size for
/// series with bounded local variation.
pub mod delta_encoding {
    use super::QuantizationConfig;

    /// Compute delta (differences) from a time series.
    ///
    /// Returns a vector of length `series.len() - 1` containing consecutive differences.
    pub fn compute_deltas(series: &[f64]) -> Vec<f64> {
        if series.len() < 2 {
            return Vec::new();
        }
        series.windows(2).map(|w| w[1] - w[0]).collect()
    }

    /// Reconstruct a series from its deltas and initial value.
    pub fn reconstruct_from_deltas(initial: f64, deltas: &[f64]) -> Vec<f64> {
        let mut series = Vec::with_capacity(deltas.len() + 1);
        series.push(initial);
        let mut current = initial;
        for &delta in deltas {
            current += delta;
            series.push(current);
        }
        series
    }

    /// Encode a time series using delta encoding with quantization.
    ///
    /// # Arguments
    ///
    /// * `series` - The time series to encode
    /// * `delta_config` - Quantization config for delta values
    ///
    /// # Returns
    ///
    /// Tuple of (initial value, encoded deltas).
    pub fn encode_deltas_u8(series: &[f64], delta_config: &QuantizationConfig) -> (f64, Vec<u8>) {
        if series.is_empty() {
            return (0.0, Vec::new());
        }
        let initial = series[0];
        let deltas = compute_deltas(series);
        let encoded = delta_config.encode_u8(&deltas);
        (initial, encoded)
    }

    /// Decode delta-encoded series.
    pub fn decode_deltas_u8(
        initial: f64,
        encoded: &[u8],
        delta_config: &QuantizationConfig,
    ) -> Vec<f64> {
        let deltas = delta_config.decode_u8(encoded);
        reconstruct_from_deltas(initial, &deltas)
    }
}

/// SAX (Symbolic Aggregate approXimation) encoding.
///
/// A popular time series discretization method that:
/// 1. Normalizes the series (z-score)
/// 2. Segments into windows and computes mean per window
/// 3. Maps means to alphabet symbols via quantiles
///
/// Reference: Lin, J., Keogh, E., Wei, L., & Lonardi, S. (2007).
/// Experiencing SAX: a novel symbolic representation of time series.
pub mod sax_encoding {
    /// Breakpoints for SAX alphabet sizes 2-10.
    /// These are z-score values that divide the normal distribution into equal areas.
    const SAX_BREAKPOINTS: &[&[f64]] = &[
        &[0.0],                                                         // alphabet_size = 2
        &[-0.43, 0.43],                                                 // 3
        &[-0.67, 0.0, 0.67],                                            // 4
        &[-0.84, -0.25, 0.25, 0.84],                                    // 5
        &[-0.97, -0.43, 0.0, 0.43, 0.97],                               // 6
        &[-1.07, -0.57, -0.18, 0.18, 0.57, 1.07],                       // 7
        &[-1.15, -0.67, -0.32, 0.0, 0.32, 0.67, 1.15],                  // 8
        &[-1.22, -0.76, -0.43, -0.14, 0.14, 0.43, 0.76, 1.22],          // 9
        &[-1.28, -0.84, -0.52, -0.25, 0.0, 0.25, 0.52, 0.84, 1.28],     // 10
    ];

    /// Get SAX breakpoints for a given alphabet size.
    pub fn get_breakpoints(alphabet_size: usize) -> Option<&'static [f64]> {
        if alphabet_size >= 2 && alphabet_size <= 10 {
            Some(SAX_BREAKPOINTS[alphabet_size - 2])
        } else {
            None
        }
    }

    /// Normalize a time series to zero mean and unit variance.
    pub fn normalize(series: &[f64]) -> Vec<f64> {
        if series.is_empty() {
            return Vec::new();
        }

        let n = series.len() as f64;
        let mean = series.iter().sum::<f64>() / n;
        let variance = series.iter().map(|&x| (x - mean).powi(2)).sum::<f64>() / n;
        let std_dev = variance.sqrt();

        if std_dev < 1e-10 {
            // Constant series - return zeros
            return vec![0.0; series.len()];
        }

        series.iter().map(|&x| (x - mean) / std_dev).collect()
    }

    /// Compute PAA (Piecewise Aggregate Approximation).
    ///
    /// Segments the series into `num_segments` windows and computes the mean of each.
    pub fn paa(series: &[f64], num_segments: usize) -> Vec<f64> {
        if series.is_empty() || num_segments == 0 {
            return Vec::new();
        }

        let n = series.len();
        if num_segments >= n {
            return series.to_vec();
        }

        let mut result = Vec::with_capacity(num_segments);
        let segment_size = n as f64 / num_segments as f64;

        for i in 0..num_segments {
            let start = (i as f64 * segment_size).floor() as usize;
            let end = ((i + 1) as f64 * segment_size).floor() as usize;
            let end = end.min(n);
            let segment = &series[start..end];
            let mean = segment.iter().sum::<f64>() / segment.len() as f64;
            result.push(mean);
        }

        result
    }

    /// Map a z-score value to a SAX symbol.
    fn zscore_to_symbol(z: f64, breakpoints: &[f64]) -> u8 {
        for (i, &bp) in breakpoints.iter().enumerate() {
            if z < bp {
                return i as u8;
            }
        }
        breakpoints.len() as u8
    }

    /// Encode a time series using SAX.
    ///
    /// # Arguments
    ///
    /// * `series` - The time series to encode
    /// * `num_segments` - Number of PAA segments (word length)
    /// * `alphabet_size` - Number of SAX symbols (2-10)
    ///
    /// # Returns
    ///
    /// SAX word as a vector of symbols (0 to alphabet_size-1).
    ///
    /// # Example
    ///
    /// ```rust
    /// use liblevenshtein::time_series::sax_encoding;
    ///
    /// let series = vec![1.0, 2.0, 3.0, 4.0, 5.0, 4.0, 3.0, 2.0];
    /// let sax_word = sax_encoding::encode(&series, 4, 4);
    /// assert_eq!(sax_word.len(), 4);
    /// ```
    pub fn encode(series: &[f64], num_segments: usize, alphabet_size: usize) -> Vec<u8> {
        let breakpoints = match get_breakpoints(alphabet_size) {
            Some(bp) => bp,
            None => return Vec::new(),
        };

        let normalized = normalize(series);
        let paa_values = paa(&normalized, num_segments);

        paa_values
            .iter()
            .map(|&z| zscore_to_symbol(z, breakpoints))
            .collect()
    }

    /// Compute MINDIST between two SAX words.
    ///
    /// This is a lower bound on the Euclidean distance between the original series.
    pub fn mindist(sax1: &[u8], sax2: &[u8], n: usize, alphabet_size: usize) -> f64 {
        if sax1.len() != sax2.len() || sax1.is_empty() {
            return f64::INFINITY;
        }

        let breakpoints = match get_breakpoints(alphabet_size) {
            Some(bp) => bp,
            None => return f64::INFINITY,
        };

        let w = sax1.len();
        let ratio = (n as f64 / w as f64).sqrt();

        let mut sum = 0.0;
        for (&s1, &s2) in sax1.iter().zip(sax2.iter()) {
            let diff = (s1 as i32 - s2 as i32).unsigned_abs() as usize;
            if diff > 1 {
                // Symbols are not adjacent - compute distance
                let larger = s1.max(s2) as usize;
                let smaller = s1.min(s2) as usize;
                // Distance between breakpoint[smaller] and breakpoint[larger-1]
                let dist = if larger > 0 && smaller < breakpoints.len() {
                    let bp_larger = if larger <= breakpoints.len() {
                        breakpoints[larger - 1]
                    } else {
                        f64::INFINITY
                    };
                    let bp_smaller = breakpoints[smaller];
                    (bp_larger - bp_smaller).abs()
                } else {
                    0.0
                };
                sum += dist * dist;
            }
        }

        ratio * sum.sqrt()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    const EPSILON: f64 = 1e-9;

    fn approx_eq(a: f64, b: f64) -> bool {
        (a - b).abs() < EPSILON
    }

    // ==================== QuantizationConfig Tests ====================

    #[test]
    fn test_uniform_quantization() {
        let config = QuantizationConfig::uniform(0.0, 100.0, 100);
        assert_eq!(config.num_bins, 100);
        assert!(approx_eq(config.bin_width, 1.0));
    }

    #[test]
    fn test_quantize_boundaries() {
        let config = QuantizationConfig::uniform(0.0, 100.0, 100);

        assert_eq!(config.quantize(0.0), 0);
        assert_eq!(config.quantize(0.5), 0);
        assert_eq!(config.quantize(1.0), 1);
        assert_eq!(config.quantize(99.0), 99);
        assert_eq!(config.quantize(99.9), 99);
        assert_eq!(config.quantize(100.0), 99); // Clamped
    }

    #[test]
    fn test_quantize_clamp_outliers() {
        let config = QuantizationConfig::uniform(0.0, 100.0, 100);

        assert_eq!(config.quantize(-10.0), 0);
        assert_eq!(config.quantize(110.0), 99);
    }

    #[test]
    fn test_dequantize_center() {
        let config = QuantizationConfig::uniform(0.0, 100.0, 100);

        // Bin 0 center should be 0.5
        assert!(approx_eq(config.dequantize(0), 0.5));
        // Bin 50 center should be 50.5
        assert!(approx_eq(config.dequantize(50), 50.5));
    }

    #[test]
    fn test_roundtrip() {
        let config = QuantizationConfig::uniform(0.0, 100.0, 1000);
        let original = 42.3;
        let quantized = config.quantize(original);
        let dequantized = config.dequantize(quantized);

        // Should be within max_error
        assert!((original - dequantized).abs() <= config.max_error() + EPSILON);
    }

    #[test]
    fn test_encode_decode_u8() {
        let config = QuantizationConfig::for_u8(0.0, 100.0);
        let series = vec![0.0, 25.0, 50.0, 75.0, 100.0];

        let encoded = config.encode_u8(&series);
        assert_eq!(encoded.len(), 5);

        let decoded = config.decode_u8(&encoded);
        for (orig, dec) in series.iter().zip(decoded.iter()) {
            assert!((orig - dec).abs() < 1.0);
        }
    }

    #[test]
    fn test_from_data() {
        let data = vec![10.0, 20.0, 30.0, 40.0, 50.0];
        let config = QuantizationConfig::from_data(&data, 256, 0.1).expect("test fixture: must be Some");

        // Range is 40, margin is 4, so total range is 48
        assert!(config.min_value < 10.0);
        assert!(config.max_value > 50.0);
    }

    #[test]
    fn test_from_data_empty() {
        let config = QuantizationConfig::from_data(&[], 256, 0.1);
        assert!(config.is_none());
    }

    #[test]
    fn test_value_diff_to_bins() {
        let config = QuantizationConfig::uniform(0.0, 100.0, 100);
        assert_eq!(config.value_diff_to_bins(10.0), 10);
        assert_eq!(config.value_diff_to_bins(0.5), 1);
        assert_eq!(config.value_diff_to_bins(0.0), 0);
    }

    // ==================== Float Encoding Tests ====================

    #[test]
    fn test_f32_encode_decode() {
        let values = vec![0.0f32, 1.0, -1.0, 3.14159, f32::MAX, f32::MIN];
        for v in values {
            let encoded = float_encoding::encode_f32(v);
            let decoded = float_encoding::decode_f32(encoded);
            assert_eq!(v.to_bits(), decoded.to_bits());
        }
    }

    #[test]
    fn test_f32_series_roundtrip() {
        let series = vec![1.0f32, 2.5, 3.14159, -0.001, 1000.0];
        let encoded = float_encoding::encode_f32_series(&series);
        let decoded = float_encoding::decode_f32_series(&encoded);
        assert_eq!(series, decoded);
    }

    #[test]
    fn test_f64_series_roundtrip() {
        let series = vec![1.0f64, 2.5, 3.14159265358979, -0.001, 1e100];
        let encoded = float_encoding::encode_f64_series_as_u32_pairs(&series);
        assert_eq!(encoded.len(), series.len() * 2);
        let decoded = float_encoding::decode_u32_pairs_to_f64(&encoded);
        assert_eq!(series, decoded);
    }

    #[test]
    fn test_f32_total_order() {
        // Test that total order encoding preserves ordering
        let values = vec![-100.0f32, -1.0, -0.001, 0.0, 0.001, 1.0, 100.0];
        let encoded: Vec<_> = values
            .iter()
            .map(|&v| float_encoding::encode_f32_total_order(v))
            .collect();

        // Check ordering is preserved
        for i in 0..encoded.len() - 1 {
            assert!(
                encoded[i] < encoded[i + 1],
                "{} should be < {}",
                values[i],
                values[i + 1]
            );
        }

        // Check roundtrip
        for &v in &values {
            let enc = float_encoding::encode_f32_total_order(v);
            let dec = float_encoding::decode_f32_total_order(enc);
            assert_eq!(v.to_bits(), dec.to_bits());
        }
    }

    // ==================== Delta Encoding Tests ====================

    #[test]
    fn test_compute_deltas() {
        let series = vec![1.0, 3.0, 6.0, 10.0];
        let deltas = delta_encoding::compute_deltas(&series);
        assert_eq!(deltas, vec![2.0, 3.0, 4.0]);
    }

    #[test]
    fn test_reconstruct_from_deltas() {
        let deltas = vec![2.0, 3.0, 4.0];
        let series = delta_encoding::reconstruct_from_deltas(1.0, &deltas);
        assert_eq!(series, vec![1.0, 3.0, 6.0, 10.0]);
    }

    #[test]
    fn test_delta_encoding_roundtrip() {
        let series = vec![1.0, 3.0, 6.0, 10.0, 8.0, 5.0];
        let delta_config = QuantizationConfig::uniform(-10.0, 10.0, 256);

        let (initial, encoded) = delta_encoding::encode_deltas_u8(&series, &delta_config);
        let decoded = delta_encoding::decode_deltas_u8(initial, &encoded, &delta_config);

        assert_eq!(series.len(), decoded.len());
        for (orig, dec) in series.iter().zip(decoded.iter()) {
            assert!((orig - dec).abs() < 0.5);
        }
    }

    // ==================== SAX Encoding Tests ====================

    #[test]
    fn test_sax_normalize() {
        let series = vec![1.0, 2.0, 3.0, 4.0, 5.0];
        let normalized = sax_encoding::normalize(&series);

        // Mean should be 0
        let mean: f64 = normalized.iter().sum::<f64>() / normalized.len() as f64;
        assert!(mean.abs() < EPSILON);

        // Variance should be ~1
        let variance: f64 =
            normalized.iter().map(|&x| x * x).sum::<f64>() / normalized.len() as f64;
        assert!((variance - 1.0).abs() < 0.1);
    }

    #[test]
    fn test_sax_paa() {
        let series = vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0];
        let paa = sax_encoding::paa(&series, 4);

        assert_eq!(paa.len(), 4);
        // First segment: mean of [1,2] = 1.5
        assert!((paa[0] - 1.5).abs() < EPSILON);
        // Second segment: mean of [3,4] = 3.5
        assert!((paa[1] - 3.5).abs() < EPSILON);
    }

    #[test]
    fn test_sax_encode() {
        let series = vec![1.0, 2.0, 3.0, 4.0, 5.0, 4.0, 3.0, 2.0];
        let sax_word = sax_encoding::encode(&series, 4, 4);

        assert_eq!(sax_word.len(), 4);
        // All symbols should be in range [0, 3]
        for &symbol in &sax_word {
            assert!(symbol < 4);
        }
    }

    #[test]
    fn test_sax_mindist() {
        // Same word should have distance 0
        let word1 = vec![0, 1, 2, 3];
        let word2 = vec![0, 1, 2, 3];
        assert!(approx_eq(sax_encoding::mindist(&word1, &word2, 100, 4), 0.0));

        // Adjacent symbols should have distance 0
        let word3 = vec![0, 1, 2, 3];
        let word4 = vec![1, 2, 3, 3]; // All within 1 of word3
        assert!(approx_eq(sax_encoding::mindist(&word3, &word4, 100, 4), 0.0));
    }

    #[test]
    fn test_sax_breakpoints() {
        for size in 2..=10 {
            let bp = sax_encoding::get_breakpoints(size);
            assert!(bp.is_some());
            assert_eq!(bp.expect("expected Some breakpoints in test").len(), size - 1);
        }

        assert!(sax_encoding::get_breakpoints(1).is_none());
        assert!(sax_encoding::get_breakpoints(11).is_none());
    }
}
