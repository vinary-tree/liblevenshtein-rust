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
    /// Panics if the bounds are non-finite, `min_value >= max_value`,
    /// `num_bins == 0`, or the resulting bin width is not finite and positive.
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
            min_value.is_finite() && max_value.is_finite() && min_value < max_value,
            "min_value ({}) must be less than max_value ({})",
            min_value,
            max_value
        );
        assert!(num_bins > 0, "num_bins must be positive");

        let bin_width = (max_value - min_value) / num_bins as f64;
        assert!(
            bin_width.is_finite() && bin_width > 0.0,
            "bin width must be finite and positive"
        );

        Self {
            min_value,
            max_value,
            num_bins,
            bin_width,
            clamp_outliers: true,
        }
    }

    /// Try to create a uniform quantization configuration.
    ///
    /// Returns `None` for non-finite bounds, non-increasing ranges, zero bins,
    /// or ranges whose bin width cannot be represented as a positive finite
    /// `f64`.
    pub fn try_uniform(min_value: f64, max_value: f64, num_bins: u32) -> Option<Self> {
        if !min_value.is_finite()
            || !max_value.is_finite()
            || min_value >= max_value
            || num_bins == 0
        {
            return None;
        }

        let bin_width = (max_value - min_value) / num_bins as f64;
        if !bin_width.is_finite() || bin_width <= 0.0 {
            return None;
        }

        Some(Self {
            min_value,
            max_value,
            num_bins,
            bin_width,
            clamp_outliers: true,
        })
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
    /// `None` if data is empty, has no finite non-constant range, uses zero
    /// bins, or has a margin that is non-finite or collapses the expanded range.
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
        if data.is_empty() || num_bins == 0 || !margin.is_finite() {
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
        if !margin_amount.is_finite() {
            return None;
        }

        let expanded_min = min_val - margin_amount;
        let expanded_max = max_val + margin_amount;
        if !expanded_min.is_finite() || !expanded_max.is_finite() || expanded_min >= expanded_max {
            return None;
        }

        Self::try_uniform(expanded_min, expanded_max, num_bins)
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
        if value.is_nan() {
            return 0;
        }
        if value == f64::NEG_INFINITY {
            return 0;
        }
        if value == f64::INFINITY {
            return self.num_bins - 1;
        }

        if self.clamp_outliers {
            if value <= self.min_value {
                return 0;
            }
            if value >= self.max_value {
                return self.num_bins - 1;
            }
        }

        let normalized = (value - self.min_value) / self.bin_width;
        Self::normalized_bin_floor(normalized, self.num_bins - 1)
    }

    #[inline]
    fn normalized_bin_floor(normalized: f64, max_bin: u32) -> u32 {
        if normalized.is_nan() || normalized <= 0.0 {
            return 0;
        }
        if normalized >= f64::from(max_bin) {
            return max_bin;
        }

        debug_assert!(normalized.is_finite());
        debug_assert!(normalized > 0.0);
        debug_assert!(normalized < f64::from(max_bin));
        normalized.floor() as u32
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
        u8::try_from(self.quantize(value))
            .expect("num_bins <= 256 guarantees quantized bin fits in u8")
    }

    /// Return a byte-encodable quantizer over the same value range.
    ///
    /// Byte-trie indexes can store at most 256 distinct bin ids. Wider
    /// quantizers are coarsened to 256 bins, preserving the range and outlier
    /// clamping policy. This can reduce pruning selectivity, but exact
    /// verification layers remain sound because wider bins produce admissible
    /// lower bounds.
    pub(crate) fn into_u8_compatible(self) -> Self {
        if self.num_bins <= 256 {
            return self;
        }

        let mut config = Self::uniform(self.min_value, self.max_value, 256);
        config.clamp_outliers = self.clamp_outliers;
        config
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
        u16::try_from(self.quantize(value))
            .expect("num_bins <= 65536 guarantees quantized bin fits in u16")
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

    /// Return the value interval `[lo, hi]` covering every concrete value that
    /// quantizes to `bin`. This is the admissible per-bin bound consumed by the
    /// interval-MSM transducer ([`crate::time_series::msm_interval`]): for any
    /// `v` with `self.quantize(v) == bin`, the interval satisfies `lo <= v <= hi`.
    ///
    /// Because [`quantize`](Self::quantize) folds out-of-range inputs into the
    /// extreme bins (everything `<= min_value` → bin `0`; everything `>=
    /// max_value` → bin `num_bins - 1`), those extreme bins must extend to ±∞
    /// for the bound to stay *sound* — otherwise a query value below `min_value`
    /// (which legitimately quantizes to bin 0) would be reported as outside
    /// bin 0's interval, inflating the lower bound and risking a dropped true
    /// match (a false negative). Interior bins use the tight half-open span
    /// `[min + bin·w, min + (bin+1)·w)`; the closed upper endpoint returned here
    /// is a harmless over-approximation (it only ever *loosens* the bound).
    ///
    /// # Example
    ///
    /// ```rust
    /// use liblevenshtein::time_series::QuantizationConfig;
    ///
    /// let config = QuantizationConfig::uniform(0.0, 100.0, 100);
    /// // Interior bin 50 spans [50.0, 51.0].
    /// let (lo, hi) = config.bin_bounds(50);
    /// assert!((lo - 50.0).abs() < 1e-9 && (hi - 51.0).abs() < 1e-9);
    /// // Extreme bins absorb outliers, so they extend to infinity.
    /// assert_eq!(config.bin_bounds(0).0, f64::NEG_INFINITY);
    /// assert_eq!(config.bin_bounds(99).1, f64::INFINITY);
    /// ```
    #[inline]
    pub fn bin_bounds(&self, bin: u32) -> (f64, f64) {
        let bin = bin.min(self.num_bins - 1);
        let lo = if bin == 0 {
            f64::NEG_INFINITY
        } else {
            self.min_value + bin as f64 * self.bin_width
        };
        let hi = if bin == self.num_bins - 1 {
            f64::INFINITY
        } else {
            self.min_value + (bin as f64 + 1.0) * self.bin_width
        };
        (lo, hi)
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
        let bins = value_diff.abs() / self.bin_width;
        Self::ceil_nonnegative_bins_to_u32(bins)
    }

    #[inline]
    fn ceil_nonnegative_bins_to_u32(bins: f64) -> u32 {
        if !bins.is_finite() || bins >= f64::from(u32::MAX) {
            u32::MAX
        } else {
            debug_assert!(bins >= 0.0);
            bins.ceil() as u32
        }
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
        let Some(capacity) = f64_pair_capacity(series.len()) else {
            return Vec::new();
        };
        let mut encoded = Vec::new();
        if encoded.try_reserve_exact(capacity).is_err() {
            return Vec::new();
        }

        for &v in series {
            let bits = encode_f64(v);
            encoded.push((bits >> 32) as u32); // High 32 bits
            encoded.push(bits as u32); // Low 32 bits
        }
        encoded
    }

    pub(super) fn f64_pair_capacity(series_len: usize) -> Option<usize> {
        series_len.checked_mul(2)
    }

    /// Decode pairs of u32 back to f64.
    pub fn decode_u32_pairs_to_f64(encoded: &[u32]) -> Vec<f64> {
        encoded
            .as_chunks::<2>()
            .0
            .iter()
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
        debug_assert!(
            value >= 0.0,
            "encode_f32_ordered requires non-negative values"
        );
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
        let Some(capacity) = reconstruction_capacity(deltas.len()) else {
            return Vec::new();
        };
        let mut series = Vec::new();
        if series.try_reserve_exact(capacity).is_err() {
            return Vec::new();
        }

        series.push(initial);
        let mut current = initial;
        for &delta in deltas {
            current += delta;
            series.push(current);
        }
        series
    }

    pub(super) fn reconstruction_capacity(delta_len: usize) -> Option<usize> {
        delta_len.checked_add(1)
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
        &[0.0],                                                     // alphabet_size = 2
        &[-0.43, 0.43],                                             // 3
        &[-0.67, 0.0, 0.67],                                        // 4
        &[-0.84, -0.25, 0.25, 0.84],                                // 5
        &[-0.97, -0.43, 0.0, 0.43, 0.97],                           // 6
        &[-1.07, -0.57, -0.18, 0.18, 0.57, 1.07],                   // 7
        &[-1.15, -0.67, -0.32, 0.0, 0.32, 0.67, 1.15],              // 8
        &[-1.22, -0.76, -0.43, -0.14, 0.14, 0.43, 0.76, 1.22],      // 9
        &[-1.28, -0.84, -0.52, -0.25, 0.0, 0.25, 0.52, 0.84, 1.28], // 10
    ];

    /// Get SAX breakpoints for a given alphabet size.
    pub fn get_breakpoints(alphabet_size: usize) -> Option<&'static [f64]> {
        if (2..=10).contains(&alphabet_size) {
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
        let mut result = Vec::new();
        if result.try_reserve_exact(num_segments).is_err() {
            return Vec::new();
        }

        for i in 0..num_segments {
            let start = scaled_segment_boundary(i, n, num_segments).min(n - 1);
            let end = scaled_segment_boundary(i + 1, n, num_segments)
                .max(start + 1)
                .min(n);
            let segment = &series[start..end];
            let mean = segment.iter().sum::<f64>() / segment.len() as f64;
            result.push(mean);
        }

        result
    }

    fn scaled_segment_boundary(segment: usize, series_len: usize, num_segments: usize) -> usize {
        let whole = segment * (series_len / num_segments);
        let fractional = ((segment as u128 * (series_len % num_segments) as u128)
            / num_segments as u128) as usize;
        whole + fractional
    }

    /// Map a z-score value to a SAX symbol.
    ///
    /// A `NaN` z-score maps to symbol `0`, matching
    /// [`super::QuantizationConfig::quantize`]'s `NaN` handling so the two
    /// encoders agree on degenerate input. Finite values and `±∞` are
    /// unaffected: the `z < bp` scan already routes `-∞` to symbol `0` and
    /// `+∞` to the top symbol (`breakpoints.len()`).
    fn zscore_to_symbol(z: f64, breakpoints: &[f64]) -> u8 {
        if z.is_nan() {
            return 0;
        }
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

    #[cfg(test)]
    mod sax_symbol_tests {
        use super::*;

        /// A `NaN` z-score must map to symbol `0`, matching
        /// [`super::super::QuantizationConfig::quantize`] (which maps both `NaN`
        /// and `-∞` to bin `0`). Finite values and `±∞` are unchanged.
        #[test]
        fn nan_zscore_maps_to_symbol_zero_matching_quantize() {
            // alphabet_size = 4 -> breakpoints [-0.67, 0.0, 0.67]; top symbol 3.
            let breakpoints = get_breakpoints(4).expect("alphabet size 4 is valid");
            let top = breakpoints.len() as u8;

            // The fix: NaN -> 0 (was previously the top symbol).
            assert_eq!(zscore_to_symbol(f64::NAN, breakpoints), 0);

            // Parity with QuantizationConfig::quantize on the degenerate inputs.
            let quant = super::super::QuantizationConfig::for_u8(-1.0, 1.0);
            assert_eq!(quant.quantize_u8(f64::NAN), 0);

            // Unchanged: -∞ -> 0, +∞ -> top, and finite values keep their bins.
            assert_eq!(zscore_to_symbol(f64::NEG_INFINITY, breakpoints), 0);
            assert_eq!(zscore_to_symbol(f64::INFINITY, breakpoints), top);
            assert_eq!(zscore_to_symbol(-1.0, breakpoints), 0);
            assert_eq!(zscore_to_symbol(0.5, breakpoints), 2);
        }
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
    fn test_try_uniform_rejects_invalid_ranges_without_panicking() {
        assert!(QuantizationConfig::try_uniform(f64::NAN, 1.0, 10).is_none());
        assert!(QuantizationConfig::try_uniform(0.0, f64::INFINITY, 10).is_none());
        assert!(QuantizationConfig::try_uniform(1.0, 1.0, 10).is_none());
        assert!(QuantizationConfig::try_uniform(2.0, 1.0, 10).is_none());
        assert!(QuantizationConfig::try_uniform(0.0, 1.0, 0).is_none());
        assert!(QuantizationConfig::try_uniform(-f64::MAX, f64::MAX, 10).is_none());

        let config = QuantizationConfig::try_uniform(-1.0, 1.0, 10)
            .expect("finite increasing range with bins should be valid");
        assert_eq!(config.min_value, -1.0);
        assert_eq!(config.max_value, 1.0);
        assert_eq!(config.num_bins, 10);
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
    fn test_quantize_non_finite_values_are_explicitly_clamped() {
        let config = QuantizationConfig::uniform(0.0, 100.0, 100);

        assert_eq!(config.quantize(f64::NAN), 0);
        assert_eq!(config.quantize(f64::NEG_INFINITY), 0);
        assert_eq!(config.quantize(f64::INFINITY), 99);
    }

    #[test]
    fn test_quantize_without_outlier_clamping_is_explicitly_bounded() {
        let mut config = QuantizationConfig::uniform(0.0, 100.0, 100);
        config.clamp_outliers = false;

        assert_eq!(config.quantize(-10.0), 0);
        assert_eq!(config.quantize(-f64::MAX), 0);
        assert_eq!(config.quantize(110.0), 99);
        assert_eq!(config.quantize(f64::MAX), 99);
    }

    #[test]
    fn test_quantize_typed_width_boundaries_are_checked() {
        let byte_config = QuantizationConfig::for_u8(0.0, 100.0);
        assert_eq!(byte_config.quantize_u8(-f64::MAX), 0);
        assert_eq!(byte_config.quantize_u8(0.0), 0);
        assert_eq!(byte_config.quantize_u8(100.0), u8::MAX);
        assert_eq!(byte_config.quantize_u8(f64::MAX), u8::MAX);

        let word_config = QuantizationConfig::for_u16(0.0, 100.0);
        assert_eq!(word_config.quantize_u16(-f64::MAX), 0);
        assert_eq!(word_config.quantize_u16(0.0), 0);
        assert_eq!(word_config.quantize_u16(100.0), u16::MAX);
        assert_eq!(word_config.quantize_u16(f64::MAX), u16::MAX);
    }

    #[test]
    fn test_bin_bounds_interior() {
        let config = QuantizationConfig::uniform(0.0, 100.0, 100);
        // Interior bin 50 spans [50.0, 51.0].
        let (lo, hi) = config.bin_bounds(50);
        assert!(approx_eq(lo, 50.0) && approx_eq(hi, 51.0));
    }

    #[test]
    fn test_bin_bounds_extreme_bins_are_infinite() {
        let config = QuantizationConfig::uniform(0.0, 100.0, 100);
        // Bin 0 absorbs everything <= min_value, so its lower bound is -inf.
        assert_eq!(config.bin_bounds(0).0, f64::NEG_INFINITY);
        assert!(approx_eq(config.bin_bounds(0).1, 1.0));
        // The last bin absorbs everything >= max_value, so its upper bound is +inf.
        assert!(approx_eq(config.bin_bounds(99).0, 99.0));
        assert_eq!(config.bin_bounds(99).1, f64::INFINITY);
    }

    #[test]
    fn test_bin_bounds_single_bin_is_unbounded_both_sides() {
        // With one bin, bin 0 is simultaneously first and last, so it must cover
        // the whole real line.
        let config = QuantizationConfig::uniform(0.0, 100.0, 1);
        let (lo, hi) = config.bin_bounds(0);
        assert_eq!(lo, f64::NEG_INFINITY);
        assert_eq!(hi, f64::INFINITY);
    }

    #[test]
    fn test_bin_bounds_clamps_out_of_range_index() {
        let config = QuantizationConfig::uniform(0.0, 100.0, 100);
        // A bin index >= num_bins is clamped to the last bin (defensive).
        assert_eq!(config.bin_bounds(1000), config.bin_bounds(99));
    }

    proptest::proptest! {
        /// Soundness: every concrete value lies within the interval of the bin it
        /// quantizes to. This is the Rust mirror of the Coq `quantize_in_bin_bounds`
        /// theorem; it is the property the interval-MSM lower bounds depend on.
        #[test]
        fn prop_bin_bounds_contains_quantized_value(
            v in -500.0f64..500.0,
            num_bins in 1u32..=256,
        ) {
            let config = QuantizationConfig::uniform(0.0, 100.0, num_bins);
            let bin = config.quantize(v);
            let (lo, hi) = config.bin_bounds(bin);
            proptest::prop_assert!(lo <= v, "lo {lo} > v {v} (bin {bin}, num_bins {num_bins})");
            proptest::prop_assert!(v <= hi, "v {v} > hi {hi} (bin {bin}, num_bins {num_bins})");
        }
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
        let config =
            QuantizationConfig::from_data(&data, 256, 0.1).expect("test fixture: must be Some");

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
    fn test_from_data_invalid_config_returns_none() {
        let data = vec![10.0, 20.0, 30.0];

        assert!(QuantizationConfig::from_data(&data, 0, 0.1).is_none());
        assert!(QuantizationConfig::from_data(&data, 256, f64::NAN).is_none());
        assert!(QuantizationConfig::from_data(&data, 256, f64::INFINITY).is_none());
        assert!(QuantizationConfig::from_data(&data, 256, f64::MAX).is_none());
        assert!(QuantizationConfig::from_data(&data, 256, -0.5).is_none());
        assert!(QuantizationConfig::from_data(&data, 256, -1.0).is_none());
    }

    #[test]
    fn test_from_data_allows_valid_shrinking_margin() {
        let data = vec![10.0, 20.0, 30.0];
        let config = QuantizationConfig::from_data(&data, 10, -0.25)
            .expect("negative margin above -0.5 preserves a positive range");

        assert!(approx_eq(config.min_value, 15.0));
        assert!(approx_eq(config.max_value, 25.0));
        assert_eq!(config.num_bins, 10);
    }

    #[test]
    fn test_value_diff_to_bins() {
        let config = QuantizationConfig::uniform(0.0, 100.0, 100);
        assert_eq!(config.value_diff_to_bins(10.0), 10);
        assert_eq!(config.value_diff_to_bins(0.5), 1);
        assert_eq!(config.value_diff_to_bins(0.0), 0);
        assert_eq!(config.value_diff_to_bins(f64::NAN), u32::MAX);
        assert_eq!(config.value_diff_to_bins(f64::INFINITY), u32::MAX);
        assert_eq!(config.value_diff_to_bins(f64::NEG_INFINITY), u32::MAX);
        assert_eq!(config.value_diff_to_bins(f64::MAX), u32::MAX);
        assert_eq!(
            config.value_diff_to_bins(f64::from(u32::MAX) - 1.0),
            u32::MAX - 1
        );
        assert_eq!(config.value_diff_to_bins(f64::from(u32::MAX)), u32::MAX);
    }

    // ==================== Float Encoding Tests ====================

    #[test]
    fn test_f32_encode_decode() {
        let values = vec![0.0f32, 1.0, -1.0, std::f32::consts::PI, f32::MAX, f32::MIN];
        for v in values {
            let encoded = float_encoding::encode_f32(v);
            let decoded = float_encoding::decode_f32(encoded);
            assert_eq!(v.to_bits(), decoded.to_bits());
        }
    }

    #[test]
    fn test_f32_series_roundtrip() {
        let series = vec![1.0f32, 2.5, std::f32::consts::PI, -0.001, 1000.0];
        let encoded = float_encoding::encode_f32_series(&series);
        let decoded = float_encoding::decode_f32_series(&encoded);
        assert_eq!(series, decoded);
    }

    #[test]
    fn test_f64_series_roundtrip() {
        let series = vec![1.0f64, 2.5, std::f64::consts::PI, -0.001, 1e100];
        let encoded = float_encoding::encode_f64_series_as_u32_pairs(&series);
        assert_eq!(encoded.len(), series.len() * 2);
        let decoded = float_encoding::decode_u32_pairs_to_f64(&encoded);
        assert_eq!(series, decoded);
    }

    #[test]
    fn test_f64_pair_capacity_rejects_overflow() {
        assert_eq!(float_encoding::f64_pair_capacity(0), Some(0));
        assert_eq!(float_encoding::f64_pair_capacity(3), Some(6));
        assert_eq!(float_encoding::f64_pair_capacity(usize::MAX), None);
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
    fn test_delta_reconstruction_capacity_rejects_overflow() {
        assert_eq!(delta_encoding::reconstruction_capacity(0), Some(1));
        assert_eq!(delta_encoding::reconstruction_capacity(3), Some(4));
        assert_eq!(delta_encoding::reconstruction_capacity(usize::MAX), None);
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
    fn test_sax_paa_preserves_requested_segment_count_for_short_series() {
        let series = vec![1.0, 2.0, 3.0];
        let paa = sax_encoding::paa(&series, 8);

        assert_eq!(paa.len(), 8);
        assert!(paa.iter().all(|value| value.is_finite()));
        assert_eq!(paa, vec![1.0, 1.0, 1.0, 2.0, 2.0, 2.0, 3.0, 3.0]);
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
    fn test_sax_encode_preserves_requested_word_length_for_short_series() {
        let series = vec![1.0, 2.0, 3.0];
        let sax_word = sax_encoding::encode(&series, 8, 4);

        assert_eq!(sax_word.len(), 8);
        assert!(sax_word.iter().all(|&symbol| symbol < 4));
    }

    #[test]
    fn test_sax_mindist() {
        // Same word should have distance 0
        let word1 = vec![0, 1, 2, 3];
        let word2 = vec![0, 1, 2, 3];
        assert!(approx_eq(
            sax_encoding::mindist(&word1, &word2, 100, 4),
            0.0
        ));

        // Adjacent symbols should have distance 0
        let word3 = vec![0, 1, 2, 3];
        let word4 = vec![1, 2, 3, 3]; // All within 1 of word3
        assert!(approx_eq(
            sax_encoding::mindist(&word3, &word4, 100, 4),
            0.0
        ));
    }

    #[test]
    fn test_sax_breakpoints() {
        for size in 2..=10 {
            let bp = sax_encoding::get_breakpoints(size);
            assert!(bp.is_some());
            assert_eq!(
                bp.expect("expected Some breakpoints in test").len(),
                size - 1
            );
        }

        assert!(sax_encoding::get_breakpoints(1).is_none());
        assert!(sax_encoding::get_breakpoints(11).is_none());
    }
}
