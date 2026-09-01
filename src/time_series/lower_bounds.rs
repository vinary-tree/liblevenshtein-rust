//! Lower-bound and heuristic prefilter functions for MSM distance.
//!
//! A pruning lower bound must satisfy `lb(X, Y) <= MSM(X, Y)`. If such a bound
//! exceeds a threshold, the candidate can be skipped without changing exact
//! range-search results.
//!
//! # Theory
//!
//! MSM split/merge paths can be much cheaper than pointwise diagonal matching.
//! Therefore prefix Euclidean and L1 distances are useful heuristics, but they
//! are not general correctness-preserving lower bounds for MSM. For example,
//! with `c = 1`, `[0, 100]` and `[0, 0, 100]` have MSM distance `1` via a split,
//! while prefix Euclidean/L1 are `100`.
//!
//! # Available Bounds
//!
//! | Function | Complexity | Pruning status |
//! |----------|------------|----------------|
//! | `length_lb` | O(1) | correctness-preserving |
//! | `euclidean_lb` | O(min(m,n)) | heuristic only |
//! | `l1_lb` | O(min(m,n)) | heuristic only |
//! | `combined_lb` | O(min(m,n)) | heuristic only because it includes heuristic bounds |
//!
//! # Example
//!
//! ```rust
//! use liblevenshtein::time_series::{MsmConfig, length_lb};
//!
//! let x = vec![1.0, 2.0, 3.0, 4.0];
//! let y = vec![1.5, 2.5, 3.5, 4.5];
//! let c = 1.0;
//!
//! let lb_length = length_lb(&x, &y, c);
//!
//! // The length bound is safe for pruning.
//! let config = MsmConfig::new(c);
//! let actual_msm = config.distance(&x, &y);
//!
//! assert!(lb_length <= actual_msm);
//! ```

use super::msm::MsmConfig;

const DEFAULT_RESULT_BUFFER_CAPACITY: usize = 64;

/// Prefix Euclidean distance heuristic for MSM.
///
/// This is not a correctness-preserving lower bound for general MSM because
/// split/merge paths can avoid large pointwise prefix differences.
///
/// # Arguments
///
/// * `x` - First time series
/// * `y` - Second time series
///
/// # Returns
///
/// A heuristic score. Do not use it for exact pruning unless false negatives
/// are acceptable.
///
/// # Complexity
///
/// O(min(len(x), len(y)))
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
/// Each merge operation costs at least the effective non-negative split/merge
/// cost `c` (from the C function).
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

    let len_diff = x.len().abs_diff(y.len());
    len_diff as f64 * MsmConfig::new(c).c
}

/// Combined heuristic using Euclidean and length scores.
///
/// This is heuristic-only because it includes [`euclidean_lb`].
///
/// # Arguments
///
/// * `x` - First time series
/// * `y` - Second time series
/// * `c` - The MSM split/merge cost constant
///
/// # Returns
///
/// Maximum of the component scores.
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

/// Prefix sum-of-absolute-differences heuristic.
///
/// This is not a correctness-preserving lower bound for general MSM.
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

/// Configuration for lower-bound or heuristic pruning.
#[derive(Debug, Clone, Copy)]
pub struct LowerBoundConfig {
    /// The MSM split/merge cost constant
    pub c: f64,
    /// Which bound/heuristic to use
    pub bounds: LowerBoundType,
}

/// Which bound or heuristic to compute.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum LowerBoundType {
    /// Use only length-based bound (fastest, correctness-preserving)
    LengthOnly,
    /// Use only prefix Euclidean heuristic (can cause false negatives)
    EuclideanOnly,
    /// Use only prefix L1 heuristic (can cause false negatives)
    L1Only,
    /// Use combined heuristic (can cause false negatives)
    Combined,
}

impl LowerBoundType {
    /// Whether this type is safe to use for correctness-preserving pruning.
    #[inline]
    pub fn is_proven_safe_for_pruning(self) -> bool {
        matches!(self, LowerBoundType::LengthOnly)
    }
}

impl LowerBoundConfig {
    /// Create a new configuration.
    pub fn new(c: f64) -> Self {
        Self {
            c: MsmConfig::new(c).c,
            bounds: LowerBoundType::LengthOnly,
        }
    }

    /// Compute the configured bound or heuristic score.
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

/// Filter candidates using the configured bound or heuristic.
///
/// Returns only candidates whose configured score is at or below threshold.
/// Exact callers should use a [`LowerBoundConfig`] whose
/// [`LowerBoundType::is_proven_safe_for_pruning`] is `true`.
///
/// # Arguments
///
/// * `query` - The query time series
/// * `candidates` - Iterator of (value, series) pairs to filter
/// * `threshold` - Maximum MSM distance threshold
/// * `lb_config` - Bound/heuristic configuration
///
/// # Returns
///
/// Iterator of candidates that pass the configured prefilter.
pub fn filter_by_lower_bound<'a, V: Clone + 'a>(
    query: &'a [f64],
    candidates: impl Iterator<Item = (V, &'a [f64])> + 'a,
    threshold: f64,
    lb_config: LowerBoundConfig,
) -> impl Iterator<Item = (V, &'a [f64])> + 'a {
    candidates.filter(move |(_, series)| lb_config.lower_bound(query, series) <= threshold)
}

#[inline]
fn is_invalid_threshold(threshold: f64) -> bool {
    threshold.is_nan() || threshold < 0.0
}

#[inline]
fn inclusive_cutoff(threshold: f64) -> f64 {
    threshold
}

#[inline]
fn result_capacity_hint(candidate_count: usize) -> usize {
    candidate_count.min(DEFAULT_RESULT_BUFFER_CAPACITY)
}

/// Brute-force search with safe lower-bound pruning (sequential).
///
/// Uses the default [`LowerBoundConfig`], which is correctness-preserving.
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
    if is_invalid_threshold(threshold) || database.is_empty() {
        return Vec::new();
    }

    let lb_config = LowerBoundConfig::new(msm_config.split_merge_cost());
    let cutoff = inclusive_cutoff(threshold);

    let mut results: Vec<(V, f64)> = Vec::with_capacity(result_capacity_hint(database.len()));
    for (value, series) in database {
        // First check the safe length lower bound.
        if lb_config.lower_bound(query, series) > cutoff {
            continue;
        }

        // Compute exact MSM only while the candidate can still qualify.
        let Some(dist) = msm_config.distance_with_cutoff(query, series, cutoff) else {
            continue;
        };
        if dist <= cutoff {
            results.push((value.clone(), dist));
        }
    }

    results.sort_by(|a, b| a.1.total_cmp(&b.1));
    results
}

/// Brute-force search with safe lower-bound pruning (parallel with Rayon).
///
/// Parallelizes both safe lower-bound filtering and MSM computation.
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

    if is_invalid_threshold(threshold) || database.is_empty() {
        return Vec::new();
    }

    let lb_config = LowerBoundConfig::new(msm_config.split_merge_cost());
    let cutoff = inclusive_cutoff(threshold);

    let mut results: Vec<(V, f64)> = database
        .par_iter()
        .filter_map(|(value, series)| {
            // First check the safe length lower bound.
            if lb_config.lower_bound(query, series) > cutoff {
                return None;
            }

            // Compute exact MSM only while the candidate can still qualify.
            msm_config
                .distance_with_cutoff(query, series, cutoff)
                .filter(|&dist| dist <= cutoff)
                .map(|dist| (value.clone(), dist))
        })
        .collect();

    results.sort_by(|a, b| a.1.total_cmp(&b.1));
    results
}

/// Statistics from lower-bound pruning.
#[derive(Debug, Clone)]
pub struct LowerBoundStats {
    /// Total candidates evaluated
    pub total_candidates: usize,
    /// Candidates pruned by the safe lower bound
    pub pruned_by_lb: usize,
    /// Candidates that passed the safe lower bound
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
    let total_candidates = database.len();
    if is_invalid_threshold(threshold) || database.is_empty() {
        return (Vec::new(), empty_lower_bound_stats(total_candidates));
    }

    let lb_config = LowerBoundConfig::new(msm_config.split_merge_cost());
    let cutoff = inclusive_cutoff(threshold);
    let mut pruned_by_lb = 0;
    let mut passed_lb = 0;
    let mut passed_exact = 0;

    let mut results: Vec<(V, f64)> = Vec::with_capacity(result_capacity_hint(database.len()));

    for (value, series) in database {
        // Check the safe length lower bound.
        if lb_config.lower_bound(query, series) > cutoff {
            pruned_by_lb += 1;
            continue;
        }
        passed_lb += 1;

        // Compute exact MSM only while the candidate can still qualify.
        let Some(dist) = msm_config.distance_with_cutoff(query, series, cutoff) else {
            continue;
        };
        if dist <= cutoff {
            passed_exact += 1;
            results.push((value.clone(), dist));
        }
    }

    results.sort_by(|a, b| a.1.total_cmp(&b.1));

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

fn empty_lower_bound_stats(total_candidates: usize) -> LowerBoundStats {
    LowerBoundStats {
        total_candidates,
        pruned_by_lb: 0,
        passed_lb: 0,
        passed_exact: 0,
        pruning_rate: 0.0,
        false_positive_rate: 0.0,
    }
}

impl std::fmt::Display for LowerBoundStats {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        writeln!(f, "Lower Bound Pruning Statistics:")?;
        writeln!(f, "  Total candidates: {}", self.total_candidates)?;
        writeln!(f, "  Pruned by safe bound: {}", self.pruned_by_lb)?;
        writeln!(f, "  Passed safe bound: {}", self.passed_lb)?;
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
        assert!(approx_eq(length_lb(&y, &x, c), 2.0));
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
    fn test_length_lb_normalizes_invalid_cost() {
        let x = vec![1.0, 2.0, 3.0, 4.0];
        let y = vec![1.0];

        assert!(approx_eq(length_lb(&x, &y, -1.0), 0.0));
        assert!(approx_eq(length_lb(&x, &y, f64::NAN), 3.0));
        assert!(approx_eq(
            LowerBoundConfig::new(f64::INFINITY).c,
            MsmConfig::default().c
        ));
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
    fn test_heuristic_bounds_are_not_general_msm_lower_bounds() {
        let config = MsmConfig::new(1.0);
        let x = vec![0.0, 100.0];
        let y = vec![0.0, 0.0, 100.0];
        let msm = config.distance(&x, &y);

        assert!(approx_eq(msm, 1.0));
        assert!(euclidean_lb(&x, &y) > msm);
        assert!(l1_lb(&x, &y) > msm);
        assert!(combined_lb(&x, &y, config.c) > msm);
        assert!(length_lb(&x, &y, config.c) <= msm + EPSILON);
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
    fn test_search_with_lb_uses_the_exact_inclusive_cutoff() {
        let config = MsmConfig::new(1.0);
        let query = vec![10.0, 20.0, 30.0];
        let database: Vec<(usize, Vec<f64>)> = vec![
            (0, query.clone()),
            (1, vec![10.0, 20.0, 30.0_f64.next_up()]),
            (2, vec![100.0, 100.0, 100.0]),
        ];

        let results = search_with_lb(&query, &database, 0.0, &config);
        let found_ids: Vec<usize> = results.iter().map(|(id, _)| *id).collect();
        assert_eq!(found_ids, vec![0]);

        #[cfg(feature = "rayon")]
        assert_eq!(
            search_with_lb_parallel(&query, &database, 0.0, &config),
            results
        );

        let (stats_results, stats) = search_with_lb_stats(&query, &database, 0.0, &config);
        assert_eq!(stats_results, results);
        assert_eq!(stats.total_candidates, 3);
        assert_eq!(stats.passed_exact, 1);
    }

    #[test]
    fn test_search_with_lb_invalid_thresholds_are_empty() {
        let config = MsmConfig::new(1.0);
        let query = vec![1.0, 2.0, 3.0];
        let database: Vec<(usize, Vec<f64>)> = vec![(0, query.clone())];

        for threshold in [f64::NAN, -1.0] {
            assert!(search_with_lb(&query, &database, threshold, &config).is_empty());

            #[cfg(feature = "rayon")]
            assert!(search_with_lb_parallel(&query, &database, threshold, &config).is_empty());

            let (results, stats) = search_with_lb_stats(&query, &database, threshold, &config);
            assert!(results.is_empty());
            assert_eq!(stats.total_candidates, 1);
            assert_eq!(stats.pruned_by_lb, 0);
            assert_eq!(stats.passed_lb, 0);
            assert_eq!(stats.passed_exact, 0);
        }
    }

    #[test]
    fn test_search_with_lb_stats() {
        let config = MsmConfig::new(1.0);
        let database: Vec<(usize, Vec<f64>)> = vec![
            (0, vec![1.0, 2.0, 3.0]),
            (1, vec![100.0, 200.0, 300.0, 400.0, 500.0, 600.0]), // length-pruned
            (2, vec![1.5, 2.5, 3.5]),
            (3, vec![50.0, 60.0, 70.0]),
        ];

        let query = vec![1.0, 2.0, 3.0];
        let (results, stats) = search_with_lb_stats(&query, &database, 2.0, &config);

        assert_eq!(stats.total_candidates, 4);
        assert!(stats.pruned_by_lb >= 1); // At least one safe length prune happened
        assert!(!results.is_empty());
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

        let lb_config = LowerBoundConfig {
            c: 1.0,
            bounds: LowerBoundType::Combined,
        };
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

        // This relation is only between the heuristics themselves; neither is
        // used as a general correctness-preserving MSM pruning bound.
    }
}
