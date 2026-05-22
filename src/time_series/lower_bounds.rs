//! Lower bound functions for MSM distance.
//!
//! Lower bounds enable efficient pruning during similarity search:
//! if `lower_bound(X, Y) > threshold`, then `msm(X, Y) > threshold`,
//! allowing us to skip the expensive full MSM computation.
//!
//! # Theory
//!
//! For a lower bound function `lb` to be valid for MSM, it must satisfy:
//!
//! ```text
//! lb(X, Y) <= MSM(X, Y)  for all X, Y
//! ```
//!
//! The tighter the bound (closer to MSM), the more effective pruning becomes.
//!
//! # Available Lower Bounds
//!
//! | Function | Complexity | Tightness | Best For |
//! |----------|------------|-----------|----------|
//! | `euclidean_lb` | O(min(m,n)) | Medium | Same-length series |
//! | `length_lb` | O(1) | Weak | Quick pre-filter |
//! | `combined_lb` | O(min(m,n)) | Best | General use |
//!
//! # Example
//!
//! ```rust
//! use liblevenshtein::time_series::{MsmConfig, euclidean_lb, length_lb};
//!
//! let x = vec![1.0, 2.0, 3.0, 4.0];
//! let y = vec![1.5, 2.5, 3.5, 4.5];
//! let c = 1.0;
//!
//! let lb_euclidean = euclidean_lb(&x, &y);
//! let lb_length = length_lb(&x, &y, c);
//!
//! // Both bounds are valid (less than or equal to actual MSM)
//! let config = MsmConfig::new(c);
//! let actual_msm = config.distance(&x, &y);
//!
//! assert!(lb_euclidean <= actual_msm);
//! assert!(lb_length <= actual_msm);
//! ```

use super::msm::MsmConfig;

/// Euclidean distance lower bound for MSM.
///
/// When series have the same length, the Euclidean distance (L2 norm)
/// is a lower bound for MSM because MSM's move operation has cost
/// `|x_i - y_j|`, and the optimal alignment for equal-length series
/// uses only move operations when values differ.
///
/// For different-length series, we compute the Euclidean distance over
/// the minimum length prefix, which is still a valid lower bound.
///
/// # Arguments
///
/// * `x` - First time series
/// * `y` - Second time series
///
/// # Returns
///
/// Lower bound on MSM distance.
///
/// # Complexity
///
/// O(min(len(x), len(y)))
///
/// # Proof of Validity
///
/// For equal-length series X = (x_1, ..., x_n) and Y = (y_1, ..., y_n):
///
/// - Euclidean distance: `E(X,Y) = sqrt(sum((x_i - y_i)^2))`
/// - Sum of absolute differences: `S(X,Y) = sum(|x_i - y_i|)`
/// - By the triangle inequality: `E(X,Y) <= S(X,Y)`
/// - Any MSM alignment costs at least `S(X,Y)` for moves alone
/// - Therefore: `E(X,Y) <= MSM(X,Y)`
///
/// For unequal-length series, we use the prefix property: the distance
/// for a prefix is at most the distance for the full series.
///
/// # Example
///
/// ```rust
/// use liblevenshtein::time_series::euclidean_lb;
///
/// let x = vec![1.0, 2.0, 3.0];
/// let y = vec![2.0, 3.0, 4.0];
///
/// // sqrt((2-1)^2 + (3-2)^2 + (4-3)^2) = sqrt(3) ≈ 1.732
/// let lb = euclidean_lb(&x, &y);
/// assert!((lb - 1.732).abs() < 0.01);
/// ```
pub fn euclidean_lb(x: &[f64], y: &[f64]) -> f64 {
    if x.is_empty() || y.is_empty() {
        if x.is_empty() && y.is_empty() {
            return 0.0;
        }
        // Empty vs non-empty has infinite MSM distance
        return f64::INFINITY;
    }

    let min_len = x.len().min(y.len());
    let sum_sq: f64 = x[..min_len]
        .iter()
        .zip(y[..min_len].iter())
        .map(|(&xi, &yi)| (xi - yi).powi(2))
        .sum();

    sum_sq.sqrt()
}

/// Length-based lower bound for MSM.
///
/// When series have different lengths, at least `|len(x) - len(y)|`
/// split or merge operations are needed. Each costs at least `c`,
/// giving a lower bound of `|len(x) - len(y)| * c`.
///
/// # Arguments
///
/// * `x` - First time series
/// * `y` - Second time series
/// * `c` - The MSM split/merge cost constant
///
/// # Returns
///
/// Lower bound on MSM distance.
///
/// # Complexity
///
/// O(1)
///
/// # Proof of Validity
///
/// Let m = len(X), n = len(Y), and assume m > n (WLOG).
///
/// To transform X to Y (or vice versa), we need to either:
/// - Merge (m - n) pairs of elements, or
/// - Use a combination of moves that effectively does the same
///
/// Each merge operation costs at least `c` (from the C function).
/// Therefore: `MSM(X, Y) >= |m - n| * c`
///
/// # Example
///
/// ```rust
/// use liblevenshtein::time_series::length_lb;
///
/// let x = vec![1.0, 2.0, 3.0, 4.0, 5.0]; // len = 5
/// let y = vec![1.0, 2.0, 3.0];           // len = 3
/// let c = 1.0;
///
/// // |5 - 3| * 1.0 = 2.0
/// let lb = length_lb(&x, &y, c);
/// assert!((lb - 2.0).abs() < 1e-9);
/// ```
pub fn length_lb(x: &[f64], y: &[f64], c: f64) -> f64 {
    if x.is_empty() && y.is_empty() {
        return 0.0;
    }
    if x.is_empty() || y.is_empty() {
        return f64::INFINITY;
    }

    let len_diff = (x.len() as isize - y.len() as isize).unsigned_abs();
    len_diff as f64 * c
}

/// Combined lower bound using both Euclidean and length bounds.
///
/// Takes the maximum of available lower bounds for the tightest result.
///
/// # Arguments
///
/// * `x` - First time series
/// * `y` - Second time series
/// * `c` - The MSM split/merge cost constant
///
/// # Returns
///
/// Maximum of all applicable lower bounds.
///
/// # Complexity
///
/// O(min(len(x), len(y)))
///
/// # Example
///
/// ```rust
/// use liblevenshtein::time_series::combined_lb;
///
/// let x = vec![1.0, 2.0, 3.0, 4.0];
/// let y = vec![1.0, 2.0];
/// let c = 1.0;
///
/// let lb = combined_lb(&x, &y, c);
/// // Uses max(euclidean_lb, length_lb)
/// ```
pub fn combined_lb(x: &[f64], y: &[f64], c: f64) -> f64 {
    euclidean_lb(x, y).max(length_lb(x, y, c))
}

/// Sum of absolute differences lower bound.
///
/// For equal-length series, this is typically tighter than Euclidean
/// because it directly represents the move costs without the square root.
///
/// ```text
/// L1(X, Y) = sum(|x_i - y_i|)
/// ```
///
/// This equals the MSM distance when no splits/merges are used.
///
/// # Arguments
///
/// * `x` - First time series
/// * `y` - Second time series
///
/// # Returns
///
/// Sum of absolute differences for the overlapping prefix.
///
/// # Complexity
///
/// O(min(len(x), len(y)))
pub fn l1_lb(x: &[f64], y: &[f64]) -> f64 {
    if x.is_empty() || y.is_empty() {
        if x.is_empty() && y.is_empty() {
            return 0.0;
        }
        return f64::INFINITY;
    }

    let min_len = x.len().min(y.len());
    x[..min_len]
        .iter()
        .zip(y[..min_len].iter())
        .map(|(&xi, &yi)| (xi - yi).abs())
        .sum()
}

/// Configuration for lower bound-based pruning.
#[derive(Debug, Clone, Copy)]
pub struct LowerBoundConfig {
    /// The MSM split/merge cost constant
    pub c: f64,
    /// Which lower bounds to use
    pub bounds: LowerBoundType,
}

/// Which lower bounds to compute.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum LowerBoundType {
    /// Use only length-based bound (fastest, weakest)
    LengthOnly,
    /// Use only Euclidean bound
    EuclideanOnly,
    /// Use only L1 bound
    L1Only,
    /// Use combined bounds (max of all, tightest)
    Combined,
}

impl LowerBoundConfig {
    /// Create a new configuration.
    pub fn new(c: f64) -> Self {
        Self {
            c,
            bounds: LowerBoundType::Combined,
        }
    }

    /// Compute the lower bound based on configuration.
    pub fn lower_bound(&self, x: &[f64], y: &[f64]) -> f64 {
        match self.bounds {
            LowerBoundType::LengthOnly => length_lb(x, y, self.c),
            LowerBoundType::EuclideanOnly => euclidean_lb(x, y),
            LowerBoundType::L1Only => l1_lb(x, y),
            LowerBoundType::Combined => euclidean_lb(x, y)
                .max(length_lb(x, y, self.c))
                .max(l1_lb(x, y)),
        }
    }
}

/// Filter candidates using lower bound pruning.
///
/// Returns only candidates whose lower bound distance is at or below threshold.
///
/// # Arguments
///
/// * `query` - The query time series
/// * `candidates` - Iterator of (value, series) pairs to filter
/// * `threshold` - Maximum MSM distance threshold
/// * `lb_config` - Lower bound configuration
///
/// # Returns
///
/// Iterator of candidates that pass the lower bound filter.
pub fn filter_by_lower_bound<'a, V: Clone + 'a>(
    query: &'a [f64],
    candidates: impl Iterator<Item = (V, &'a [f64])> + 'a,
    threshold: f64,
    lb_config: LowerBoundConfig,
) -> impl Iterator<Item = (V, &'a [f64])> + 'a {
    candidates.filter(move |(_, series)| lb_config.lower_bound(query, series) <= threshold)
}

/// Brute-force search with lower bound pruning (sequential).
///
/// Uses lower bounds to skip computing full MSM for candidates
/// that cannot possibly be within the threshold.
///
/// # Arguments
///
/// * `query` - The query time series
/// * `database` - Slice of (value, series) pairs
/// * `threshold` - Maximum MSM distance
/// * `msm_config` - MSM configuration
///
/// # Returns
///
/// Vector of (value, distance) pairs within threshold, sorted by distance.
pub fn search_with_lb<V: Clone>(
    query: &[f64],
    database: &[(V, Vec<f64>)],
    threshold: f64,
    msm_config: &MsmConfig,
) -> Vec<(V, f64)> {
    let lb_config = LowerBoundConfig::new(msm_config.c);

    let mut results: Vec<(V, f64)> = database
        .iter()
        .filter_map(|(value, series)| {
            // First check lower bound
            if lb_config.lower_bound(query, series) > threshold {
                return None;
            }

            // Compute exact MSM
            let dist = msm_config.distance(query, series);
            if dist <= threshold + 1e-9 {
                Some((value.clone(), dist))
            } else {
                None
            }
        })
        .collect();

    results.sort_by(|a, b| a.1.partial_cmp(&b.1).unwrap_or(std::cmp::Ordering::Equal));
    results
}

/// Brute-force search with lower bound pruning (parallel with Rayon).
///
/// Parallelizes both lower bound filtering and MSM computation.
///
/// # Arguments
///
/// * `query` - The query time series
/// * `database` - Slice of (value, series) pairs
/// * `threshold` - Maximum MSM distance
/// * `msm_config` - MSM configuration
///
/// # Returns
///
/// Vector of (value, distance) pairs within threshold, sorted by distance.
#[cfg(feature = "rayon")]
pub fn search_with_lb_parallel<V: Clone + Send + Sync>(
    query: &[f64],
    database: &[(V, Vec<f64>)],
    threshold: f64,
    msm_config: &MsmConfig,
) -> Vec<(V, f64)> {
    use rayon::prelude::*;

    let lb_config = LowerBoundConfig::new(msm_config.c);

    let mut results: Vec<(V, f64)> = database
        .par_iter()
        .filter_map(|(value, series)| {
            // First check lower bound
            if lb_config.lower_bound(query, series) > threshold {
                return None;
            }

            // Compute exact MSM
            let dist = msm_config.distance(query, series);
            if dist <= threshold + 1e-9 {
                Some((value.clone(), dist))
            } else {
                None
            }
        })
        .collect();

    results.sort_by(|a, b| a.1.partial_cmp(&b.1).unwrap_or(std::cmp::Ordering::Equal));
    results
}

/// Statistics from lower bound pruning.
#[derive(Debug, Clone)]
pub struct LowerBoundStats {
    /// Total candidates evaluated
    pub total_candidates: usize,
    /// Candidates pruned by lower bound
    pub pruned_by_lb: usize,
    /// Candidates that passed lower bound
    pub passed_lb: usize,
    /// Candidates that passed exact MSM
    pub passed_exact: usize,
    /// Pruning efficiency: pruned / total
    pub pruning_rate: f64,
    /// False positive rate: (passed_lb - passed_exact) / passed_lb
    pub false_positive_rate: f64,
}

/// Search with detailed statistics collection.
pub fn search_with_lb_stats<V: Clone>(
    query: &[f64],
    database: &[(V, Vec<f64>)],
    threshold: f64,
    msm_config: &MsmConfig,
) -> (Vec<(V, f64)>, LowerBoundStats) {
    let lb_config = LowerBoundConfig::new(msm_config.c);

    let total_candidates = database.len();
    let mut pruned_by_lb = 0;
    let mut passed_lb = 0;
    let mut passed_exact = 0;

    let mut results: Vec<(V, f64)> = Vec::new();

    for (value, series) in database {
        // Check lower bound
        if lb_config.lower_bound(query, series) > threshold {
            pruned_by_lb += 1;
            continue;
        }
        passed_lb += 1;

        // Compute exact MSM
        let dist = msm_config.distance(query, series);
        if dist <= threshold + 1e-9 {
            passed_exact += 1;
            results.push((value.clone(), dist));
        }
    }

    results.sort_by(|a, b| a.1.partial_cmp(&b.1).unwrap_or(std::cmp::Ordering::Equal));

    let stats = LowerBoundStats {
        total_candidates,
        pruned_by_lb,
        passed_lb,
        passed_exact,
        pruning_rate: if total_candidates > 0 {
            pruned_by_lb as f64 / total_candidates as f64
        } else {
            0.0
        },
        false_positive_rate: if passed_lb > 0 {
            (passed_lb - passed_exact) as f64 / passed_lb as f64
        } else {
            0.0
        },
    };

    (results, stats)
}

impl std::fmt::Display for LowerBoundStats {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        writeln!(f, "Lower Bound Pruning Statistics:")?;
        writeln!(f, "  Total candidates: {}", self.total_candidates)?;
        writeln!(f, "  Pruned by LB: {}", self.pruned_by_lb)?;
        writeln!(f, "  Passed LB: {}", self.passed_lb)?;
        writeln!(f, "  Passed exact: {}", self.passed_exact)?;
        writeln!(f, "  Pruning rate: {:.1}%", self.pruning_rate * 100.0)?;
        writeln!(
            f,
            "  False positive rate: {:.1}%",
            self.false_positive_rate * 100.0
        )
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    const EPSILON: f64 = 1e-9;

    fn approx_eq(a: f64, b: f64) -> bool {
        if a.is_infinite() && b.is_infinite() {
            return a.signum() == b.signum();
        }
        (a - b).abs() < EPSILON
    }

    fn approx_le(a: f64, b: f64) -> bool {
        a <= b + EPSILON
    }

    #[test]
    fn test_euclidean_lb_identical() {
        let x = vec![1.0, 2.0, 3.0, 4.0];
        assert!(approx_eq(euclidean_lb(&x, &x), 0.0));
    }

    #[test]
    fn test_euclidean_lb_simple() {
        let x = vec![1.0, 2.0, 3.0];
        let y = vec![2.0, 3.0, 4.0];
        // sqrt(1 + 1 + 1) = sqrt(3) ≈ 1.732
        let lb = euclidean_lb(&x, &y);
        assert!((lb - 3.0_f64.sqrt()).abs() < 0.01);
    }

    #[test]
    fn test_euclidean_lb_empty() {
        let x: Vec<f64> = vec![];
        let y = vec![1.0, 2.0, 3.0];
        assert!(euclidean_lb(&x, &y).is_infinite());
        assert!(euclidean_lb(&y, &x).is_infinite());
        assert!(approx_eq(euclidean_lb(&x, &x), 0.0));
    }

    #[test]
    fn test_euclidean_lb_different_lengths() {
        let x = vec![1.0, 2.0, 3.0, 4.0];
        let y = vec![1.0, 2.0];
        // Uses only first 2 elements: sqrt(0 + 0) = 0
        let lb = euclidean_lb(&x, &y);
        assert!(approx_eq(lb, 0.0));
    }

    #[test]
    fn test_length_lb_same_length() {
        let x = vec![1.0, 2.0, 3.0];
        let y = vec![4.0, 5.0, 6.0];
        assert!(approx_eq(length_lb(&x, &y, 1.0), 0.0));
    }

    #[test]
    fn test_length_lb_different_lengths() {
        let x = vec![1.0, 2.0, 3.0, 4.0, 5.0];
        let y = vec![1.0, 2.0, 3.0];
        let c = 1.0;
        // |5 - 3| * 1.0 = 2.0
        assert!(approx_eq(length_lb(&x, &y, c), 2.0));
    }

    #[test]
    fn test_length_lb_different_c() {
        let x = vec![1.0, 2.0, 3.0, 4.0];
        let y = vec![1.0];
        // |4 - 1| = 3 length difference
        assert!(approx_eq(length_lb(&x, &y, 1.0), 3.0));
        assert!(approx_eq(length_lb(&x, &y, 2.0), 6.0));
        assert!(approx_eq(length_lb(&x, &y, 0.5), 1.5));
    }

    #[test]
    fn test_l1_lb_identical() {
        let x = vec![1.0, 2.0, 3.0];
        assert!(approx_eq(l1_lb(&x, &x), 0.0));
    }

    #[test]
    fn test_l1_lb_simple() {
        let x = vec![1.0, 2.0, 3.0];
        let y = vec![2.0, 3.0, 4.0];
        // |1| + |1| + |1| = 3
        assert!(approx_eq(l1_lb(&x, &y), 3.0));
    }

    #[test]
    fn test_combined_lb() {
        let x = vec![1.0, 2.0, 3.0, 4.0];
        let y = vec![1.0, 2.0];
        let c = 1.0;

        let combined = combined_lb(&x, &y, c);
        let euclidean = euclidean_lb(&x, &y);
        let length = length_lb(&x, &y, c);

        // Combined should be max
        assert!(approx_eq(combined, euclidean.max(length)));
    }

    #[test]
    fn test_lb_validity_vs_msm() {
        let config = MsmConfig::new(1.0);
        let test_cases = vec![
            (vec![1.0, 2.0, 3.0], vec![1.0, 2.0, 3.0]),
            (vec![1.0, 2.0, 3.0], vec![2.0, 3.0, 4.0]),
            (vec![1.0, 2.0, 3.0], vec![1.0, 2.0]),
            (vec![1.0, 2.0], vec![1.0, 2.0, 3.0]),
            (vec![1.0, 5.0, 2.0], vec![1.0, 2.0, 5.0]),
        ];

        for (x, y) in test_cases {
            let msm = config.distance(&x, &y);
            let euclidean = euclidean_lb(&x, &y);
            let length = length_lb(&x, &y, 1.0);
            let l1 = l1_lb(&x, &y);
            let combined = combined_lb(&x, &y, 1.0);

            assert!(
                approx_le(euclidean, msm),
                "Euclidean LB {} > MSM {} for {:?} vs {:?}",
                euclidean,
                msm,
                x,
                y
            );
            assert!(
                approx_le(length, msm),
                "Length LB {} > MSM {} for {:?} vs {:?}",
                length,
                msm,
                x,
                y
            );
            assert!(
                approx_le(l1, msm),
                "L1 LB {} > MSM {} for {:?} vs {:?}",
                l1,
                msm,
                x,
                y
            );
            assert!(
                approx_le(combined, msm),
                "Combined LB {} > MSM {} for {:?} vs {:?}",
                combined,
                msm,
                x,
                y
            );
        }
    }

    #[test]
    fn test_search_with_lb() {
        let config = MsmConfig::new(1.0);
        let database: Vec<(usize, Vec<f64>)> = vec![
            (0, vec![1.0, 2.0, 3.0]),
            (1, vec![1.1, 2.1, 3.1]),
            (2, vec![10.0, 20.0, 30.0]),
            (3, vec![1.0, 2.0]),
            (4, vec![1.0, 2.0, 3.0, 4.0]),
        ];

        let query = vec![1.0, 2.0, 3.0];
        let results = search_with_lb(&query, &database, 1.0, &config);

        // Should find id=0 (exact match) and id=1 (0.3 distance)
        let found_ids: Vec<usize> = results.iter().map(|(id, _)| *id).collect();
        assert!(found_ids.contains(&0));
        assert!(found_ids.contains(&1));
        assert!(!found_ids.contains(&2)); // Too far
    }

    #[test]
    fn test_search_with_lb_stats() {
        let config = MsmConfig::new(1.0);
        let database: Vec<(usize, Vec<f64>)> = vec![
            (0, vec![1.0, 2.0, 3.0]),
            (1, vec![100.0, 200.0, 300.0]), // Will be pruned by LB
            (2, vec![1.5, 2.5, 3.5]),
            (3, vec![50.0, 60.0, 70.0]), // Will be pruned by LB
        ];

        let query = vec![1.0, 2.0, 3.0];
        let (results, stats) = search_with_lb_stats(&query, &database, 2.0, &config);

        assert_eq!(stats.total_candidates, 4);
        assert!(stats.pruned_by_lb >= 1); // At least some pruning happened
        assert!(results.len() >= 1);
    }

    #[test]
    fn test_lower_bound_config() {
        let x = vec![1.0, 2.0, 3.0, 4.0];
        let y = vec![1.0, 2.0];

        let config = LowerBoundConfig {
            c: 1.0,
            bounds: LowerBoundType::LengthOnly,
        };
        let lb_length = config.lower_bound(&x, &y);
        assert!(approx_eq(lb_length, 2.0)); // |4-2| * 1.0

        let config = LowerBoundConfig {
            c: 1.0,
            bounds: LowerBoundType::EuclideanOnly,
        };
        let lb_euclidean = config.lower_bound(&x, &y);
        assert!(approx_eq(lb_euclidean, 0.0)); // First 2 elements match

        let config = LowerBoundConfig {
            c: 1.0,
            bounds: LowerBoundType::Combined,
        };
        let lb_combined = config.lower_bound(&x, &y);
        assert!(approx_eq(lb_combined, 2.0)); // max(0, 2, 0) = 2
    }

    #[test]
    fn test_filter_by_lower_bound() {
        let query = vec![1.0, 2.0, 3.0];
        let candidates: Vec<(usize, Vec<f64>)> = vec![
            (0, vec![1.0, 2.0, 3.0]),       // Close
            (1, vec![100.0, 200.0, 300.0]), // Far
            (2, vec![1.5, 2.5, 3.5]),       // Close
        ];

        let lb_config = LowerBoundConfig::new(1.0);
        let candidate_refs: Vec<(usize, &[f64])> = candidates
            .iter()
            .map(|(id, series)| (*id, series.as_slice()))
            .collect();

        let filtered: Vec<_> =
            filter_by_lower_bound(&query, candidate_refs.into_iter(), 5.0, lb_config).collect();

        // Should include close series, exclude far series
        let found_ids: Vec<usize> = filtered.iter().map(|(id, _)| *id).collect();
        assert!(found_ids.contains(&0));
        assert!(!found_ids.contains(&1)); // 100-1 = 99 > threshold of 5
        assert!(found_ids.contains(&2));
    }

    #[test]
    fn test_l1_tighter_than_euclidean() {
        // L1 is tighter than Euclidean for vectors with many small differences
        let x = vec![1.0, 2.0, 3.0, 4.0];
        let y = vec![2.0, 3.0, 4.0, 5.0];

        let l1 = l1_lb(&x, &y); // 4.0
        let euclidean = euclidean_lb(&x, &y); // 2.0

        assert!(l1 > euclidean);

        // Both should still be valid lower bounds
        let config = MsmConfig::new(1.0);
        let msm = config.distance(&x, &y);
        assert!(l1 <= msm + EPSILON);
        assert!(euclidean <= msm + EPSILON);
    }
}
