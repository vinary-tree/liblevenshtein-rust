//! Move-Split-Merge (MSM) metric for time series.
//!
//! This module implements the MSM distance metric as described in:
//!
//! Stefan, Alexandra, et al. "The move-split-merge metric for time series."
//! IEEE transactions on Knowledge and Data Engineering 25.6 (2012): 1425-1438.
//!
//! # Algorithm
//!
//! The MSM metric is computed using dynamic programming with the recurrence:
//!
//! ```text
//! Cost(i,j) = min{
//!     Cost(i-1, j-1) + |x_i - y_j|,           // Move
//!     Cost(i-1, j) + C(x_i, x_{i-1}, y_j),    // Merge-like
//!     Cost(i, j-1) + C(y_j, x_i, y_{j-1})     // Split-like
//! }
//! ```
//!
//! where C(a, b, c) = c if b ≤ a ≤ c OR b ≥ a ≥ c, else c + min(|a-b|, |a-c|)

use std::fmt;

use super::bounded::{
    ExactDecision, IncompleteReason, NoWitness, Operand, OperationOutcome, ResourceKind,
    ResourceLedger, ResourceLimits, TemporalValidationError,
};

const CUTOFF_EPSILON: f64 = 1e-9;
const DEFAULT_MSM_C: f64 = 1.0;

#[inline]
pub(super) fn series_values_are_finite(series: &[f64]) -> bool {
    series.iter().all(|value| value.is_finite())
}

#[inline]
pub(super) fn series_pair_values_are_finite(x: &[f64], y: &[f64]) -> bool {
    series_values_are_finite(x) && series_values_are_finite(y)
}

#[inline]
pub(super) fn msm_dp_len(series_len: usize) -> Option<usize> {
    series_len.checked_add(1)
}

#[inline]
pub(super) fn msm_dp_shape(row_series_len: usize, col_series_len: usize) -> Option<(usize, usize)> {
    Some((msm_dp_len(row_series_len)?, msm_dp_len(col_series_len)?))
}

pub(super) fn msm_infinity_matrix(
    row_series_len: usize,
    col_series_len: usize,
) -> Option<Vec<Vec<f64>>> {
    let (rows, cols) = msm_dp_shape(row_series_len, col_series_len)?;
    Some(vec![vec![f64::INFINITY; cols]; rows])
}

#[inline]
fn normalize_msm_cost(c: f64) -> f64 {
    if c.is_finite() {
        c.max(0.0)
    } else {
        DEFAULT_MSM_C
    }
}

/// Configuration for MSM distance computation.
///
/// # Fields
///
/// - `c`: The constant cost for split and merge operations. Normalized to a
///   finite, non-negative value by the constructors and methods in this module.
///
/// # Example
///
/// ```rust
/// use liblevenshtein::time_series::MsmConfig;
///
/// let config = MsmConfig::new(1.0);
/// let distance = config.distance(&[1.0, 2.0, 3.0], &[1.0, 2.5, 3.0]);
/// ```
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct MsmConfig {
    /// The constant cost for split/merge operations.
    /// This corresponds to 'c' in the paper.
    pub c: f64,
}

impl MsmConfig {
    /// Construct an exact raw-domain MSM configuration without normalization.
    ///
    /// Zero is accepted for exact scoring but yields a pseudometric, not a
    /// metric. Use [`MetricMsmConfig::try_new`] when metric laws are required.
    pub fn try_new(c: f64) -> Result<Self, MsmConfigError> {
        if !c.is_finite() || c < 0.0 {
            return Err(MsmConfigError::InvalidSplitMergeCost);
        }
        Ok(Self { c })
    }

    /// Create a new MSM configuration.
    ///
    /// # Arguments
    ///
    /// * `c` - The constant cost for split/merge operations. Finite negative
    ///   values are clamped to `0.0`; non-finite values use the default cost.
    ///
    /// # Example
    ///
    /// ```rust
    /// use liblevenshtein::time_series::MsmConfig;
    ///
    /// let config = MsmConfig::new(1.0);
    /// assert_eq!(config.c, 1.0);
    /// ```
    #[inline]
    pub fn new(c: f64) -> Self {
        Self {
            c: normalize_msm_cost(c),
        }
    }

    /// Create a configuration with the default cost c = 1.0.
    ///
    /// This is equivalent to `MsmConfig::new(1.0)`.
    #[inline]
    pub fn default_cost() -> Self {
        Self::new(DEFAULT_MSM_C)
    }

    /// Return the effective split/merge cost.
    ///
    /// This protects algorithms from malformed configurations constructed
    /// directly with the public `c` field.
    #[inline]
    pub fn split_merge_cost(&self) -> f64 {
        normalize_msm_cost(self.c)
    }

    /// Return a configuration whose public fields match their effective values.
    #[inline]
    pub fn normalized(self) -> Self {
        Self::new(self.c)
    }

    /// Compute the MSM distance between two time series.
    ///
    /// This implements the exact dynamic programming recurrence from Figure 10
    /// of the Stefan et al. paper using the two-row implementation in
    /// [`Self::distance_optimized`]. The public default therefore keeps
    /// `O(mn)` time while using `O(min(m, n))` space. Use
    /// [`Self::distance_with_matrix`] when the full DP matrix is needed for
    /// debugging or alignment inspection.
    ///
    /// # Arguments
    ///
    /// * `x` - The first time series.
    /// * `y` - The second time series.
    ///
    /// # Returns
    ///
    /// The MSM distance between x and y.
    ///
    /// # Special Cases
    ///
    /// - If both series are empty, returns 0.0.
    /// - If one series is empty and the other is not, returns infinity
    ///   (since we can't transform an empty series to a non-empty one
    ///   using only Move, Split, and Merge operations).
    ///
    /// # Example
    ///
    /// ```rust
    /// use liblevenshtein::time_series::MsmConfig;
    ///
    /// let config = MsmConfig::new(1.0);
    ///
    /// // Identical series have distance 0
    /// assert_eq!(config.distance(&[1.0, 2.0, 3.0], &[1.0, 2.0, 3.0]), 0.0);
    ///
    /// // Single element difference uses Move cost
    /// assert_eq!(config.distance(&[1.0], &[2.0]), 1.0);  // |1.0 - 2.0| = 1.0
    /// ```
    pub fn distance(&self, x: &[f64], y: &[f64]) -> f64 {
        self.distance_optimized(x, y)
    }

    /// Compute the MSM distance with space optimization.
    ///
    /// Uses O(min(m, n)) space instead of O(mn) by keeping only two rows/columns
    /// of the DP matrix at a time.
    ///
    /// # Arguments
    ///
    /// * `x` - The first time series.
    /// * `y` - The second time series.
    ///
    /// # Returns
    ///
    /// The MSM distance between x and y.
    pub fn distance_optimized(&self, x: &[f64], y: &[f64]) -> f64 {
        // Ensure x is the longer series for space optimization
        if x.len() < y.len() {
            return self.distance_optimized(y, x);
        }

        let m = x.len();
        let n = y.len();

        // Handle empty series
        if m == 0 && n == 0 {
            return 0.0;
        }
        if n == 0 {
            return f64::INFINITY;
        }
        if !series_pair_values_are_finite(x, y) {
            return f64::INFINITY;
        }

        // Use two rows: previous and current
        let Some(row_len) = msm_dp_len(n) else {
            return f64::INFINITY;
        };
        let mut prev = vec![f64::INFINITY; row_len];
        let mut curr = vec![f64::INFINITY; row_len];

        // Initialize first row (i=1)
        prev[1] = (x[0] - y[0]).abs();
        for j in 2..=n {
            prev[j] = prev[j - 1] + self.c_func(y[j - 1], x[0], y[j - 2]);
        }

        // Process remaining rows
        for i in 2..=m {
            // Initialize first column of current row
            curr[1] = prev[1] + self.c_func(x[i - 1], x[i - 2], y[0]);

            for j in 2..=n {
                let move_cost = prev[j - 1] + (x[i - 1] - y[j - 1]).abs();
                let merge_cost = prev[j] + self.c_func(x[i - 1], x[i - 2], y[j - 1]);
                let split_cost = curr[j - 1] + self.c_func(y[j - 1], x[i - 1], y[j - 2]);

                curr[j] = move_cost.min(merge_cost).min(split_cost);
            }

            std::mem::swap(&mut prev, &mut curr);
        }

        prev[n]
    }

    /// Compute the MSM distance only when it is at most `max_cost`.
    ///
    /// This is the exact two-row dynamic program with a safe early-abandon
    /// check: if every live cell in a completed row exceeds `max_cost`, all
    /// future paths must also exceed it because MSM operation costs are
    /// non-negative. Returns `None` exactly when the distance is greater than
    /// a finite cutoff; a `+∞` cutoff returns the same value as
    /// [`Self::distance_optimized`].
    pub fn distance_with_cutoff(&self, x: &[f64], y: &[f64], max_cost: f64) -> Option<f64> {
        if max_cost.is_nan() || max_cost < -CUTOFF_EPSILON {
            return None;
        }
        if max_cost.is_infinite() && max_cost.is_sign_positive() {
            return Some(self.distance_optimized(x, y));
        }

        // Ensure x is the longer series for space optimization.
        if x.len() < y.len() {
            return self.distance_with_cutoff(y, x, max_cost);
        }

        let m = x.len();
        let n = y.len();
        let cutoff = max_cost + CUTOFF_EPSILON;

        if m == 0 && n == 0 {
            return (0.0 <= cutoff).then_some(0.0);
        }
        if n == 0 {
            return None;
        }
        if !series_pair_values_are_finite(x, y) {
            return None;
        }

        let row_len = msm_dp_len(n)?;
        let mut prev = vec![f64::INFINITY; row_len];
        let mut curr = vec![f64::INFINITY; row_len];

        prev[1] = (x[0] - y[0]).abs();
        let mut row_min = prev[1];
        for j in 2..=n {
            prev[j] = prev[j - 1] + self.c_func(y[j - 1], x[0], y[j - 2]);
            row_min = row_min.min(prev[j]);
        }
        if row_min > cutoff {
            return None;
        }

        for i in 2..=m {
            curr[1] = prev[1] + self.c_func(x[i - 1], x[i - 2], y[0]);
            row_min = curr[1];

            for j in 2..=n {
                let move_cost = prev[j - 1] + (x[i - 1] - y[j - 1]).abs();
                let merge_cost = prev[j] + self.c_func(x[i - 1], x[i - 2], y[j - 1]);
                let split_cost = curr[j - 1] + self.c_func(y[j - 1], x[i - 1], y[j - 2]);

                curr[j] = move_cost.min(merge_cost).min(split_cost);
                row_min = row_min.min(curr[j]);
            }

            if row_min > cutoff {
                return None;
            }

            std::mem::swap(&mut prev, &mut curr);
        }

        let distance = prev[n];
        (distance <= cutoff).then_some(distance)
    }

    /// Strict, fail-closed MSM cutoff decision under explicit resource limits.
    ///
    /// The complete score-only calculation reserves its worst-case cell work
    /// before allocating. It therefore never begins an operation that cannot
    /// finish within the declared non-resumable distance budget.
    pub fn distance_bounded(
        &self,
        query: &[f64],
        candidate: &[f64],
        cutoff: f64,
        limits: ResourceLimits,
    ) -> Result<OperationOutcome<ExactDecision>, TemporalValidationError> {
        if !self.c.is_finite() || self.c < 0.0 {
            return Err(TemporalValidationError::InvalidConfiguration(
                "MSM split/merge cost must be finite and nonnegative",
            ));
        }
        if cutoff.is_nan() || cutoff < 0.0 || (cutoff.is_infinite() && cutoff.is_sign_negative()) {
            return Err(TemporalValidationError::InvalidCutoff);
        }

        let mut ledger = ResourceLedger::new(limits);
        ledger.validate_finite_series(Operand::Query, query)?;
        ledger.validate_finite_series(Operand::Candidate, candidate)?;

        if query.is_empty() || candidate.is_empty() {
            let value = if query.is_empty() && candidate.is_empty() {
                if 0.0 <= cutoff {
                    ExactDecision::WithinCutoff {
                        distance: 0.0,
                        witness: NoWitness,
                    }
                } else {
                    ExactDecision::AboveCutoff
                }
            } else {
                ExactDecision::NoFiniteAlignment
            };
            return Ok(OperationOutcome::Complete {
                value,
                usage: ledger.usage(),
            });
        }

        let Some(dp_cells) = query.len().checked_mul(candidate.len()) else {
            return Ok(OperationOutcome::Incomplete {
                partial: None,
                reason: IncompleteReason::ArithmeticOverflow {
                    resource: ResourceKind::DpCells,
                },
                continuation: None,
                usage: ledger.usage(),
            });
        };
        let scratch_rows = query.len().min(candidate.len()).checked_add(1);
        let scratch_bytes = scratch_rows
            .and_then(|len| len.checked_mul(2))
            .and_then(|len| len.checked_mul(std::mem::size_of::<f64>()));
        let Some(scratch_bytes) = scratch_bytes else {
            return Ok(OperationOutcome::Incomplete {
                partial: None,
                reason: IncompleteReason::ArithmeticOverflow {
                    resource: ResourceKind::ScratchBytes,
                },
                continuation: None,
                usage: ledger.usage(),
            });
        };

        if let Err(reason) = ledger.charge_many(&[
            (ResourceKind::DpCells, dp_cells),
            (ResourceKind::WorkUnits, dp_cells),
            (ResourceKind::ScratchBytes, scratch_bytes),
        ]) {
            return Ok(OperationOutcome::Incomplete {
                partial: None,
                reason,
                continuation: None,
                usage: ledger.usage(),
            });
        }

        let value = match self.distance_with_cutoff(query, candidate, cutoff) {
            Some(distance) if distance.is_finite() => ExactDecision::WithinCutoff {
                distance,
                witness: NoWitness,
            },
            Some(_) => {
                return Ok(OperationOutcome::Incomplete {
                    partial: None,
                    reason: IncompleteReason::NumericOverflow,
                    continuation: None,
                    usage: ledger.usage(),
                });
            }
            None if cutoff.is_infinite() => {
                return Ok(OperationOutcome::Incomplete {
                    partial: None,
                    reason: IncompleteReason::NumericOverflow,
                    continuation: None,
                    usage: ledger.usage(),
                });
            }
            None => ExactDecision::AboveCutoff,
        };
        Ok(OperationOutcome::Complete {
            value,
            usage: ledger.usage(),
        })
    }

    /// Compute the MSM distance and return the full DP matrix for debugging.
    ///
    /// # Arguments
    ///
    /// * `x` - The first time series.
    /// * `y` - The second time series.
    ///
    /// # Returns
    ///
    /// An `MsmResult` containing the distance and the DP matrix.
    pub fn distance_with_matrix(&self, x: &[f64], y: &[f64]) -> MsmResult {
        let m = x.len();
        let n = y.len();

        // Handle empty series
        if m == 0 && n == 0 {
            return MsmResult {
                distance: 0.0,
                matrix: vec![vec![0.0]],
            };
        }
        if m == 0 || n == 0 {
            return MsmResult {
                distance: f64::INFINITY,
                matrix: msm_infinity_matrix(m, n).unwrap_or_default(),
            };
        }
        if !series_pair_values_are_finite(x, y) {
            return MsmResult {
                distance: f64::INFINITY,
                matrix: msm_infinity_matrix(m, n).unwrap_or_default(),
            };
        }

        // Allocate DP matrix
        let Some(mut cost) = msm_infinity_matrix(m, n) else {
            return MsmResult {
                distance: f64::INFINITY,
                matrix: Vec::new(),
            };
        };

        // Base case
        cost[1][1] = (x[0] - y[0]).abs();

        // Initialize first column
        for i in 2..=m {
            cost[i][1] = cost[i - 1][1] + self.c_func(x[i - 1], x[i - 2], y[0]);
        }

        // Initialize first row
        for j in 2..=n {
            cost[1][j] = cost[1][j - 1] + self.c_func(y[j - 1], x[0], y[j - 2]);
        }

        // Fill the DP matrix
        for i in 2..=m {
            for j in 2..=n {
                let move_cost = cost[i - 1][j - 1] + (x[i - 1] - y[j - 1]).abs();
                let merge_cost = cost[i - 1][j] + self.c_func(x[i - 1], x[i - 2], y[j - 1]);
                let split_cost = cost[i][j - 1] + self.c_func(y[j - 1], x[i - 1], y[j - 2]);
                cost[i][j] = move_cost.min(merge_cost).min(split_cost);
            }
        }

        MsmResult {
            distance: cost[m][n],
            matrix: cost,
        }
    }

    /// The C function from the paper.
    ///
    /// C(a, b, c) determines the cost of a split/merge-like operation.
    ///
    /// - If b ≤ a ≤ c OR b ≥ a ≥ c (a is between b and c), then C(a,b,c) = c_const
    /// - Otherwise, C(a,b,c) = c_const + min(|a-b|, |a-c|)
    ///
    /// # Arguments
    ///
    /// * `a` - The value being inserted/removed
    /// * `b` - The adjacent value in the first operand
    /// * `c` - The adjacent value in the second operand
    ///
    /// # Returns
    ///
    /// The cost for this split/merge operation.
    #[inline]
    pub fn c_func(&self, a: f64, b: f64, c: f64) -> f64 {
        let split_merge_cost = self.split_merge_cost();
        if !a.is_finite() || !b.is_finite() || !c.is_finite() {
            return f64::INFINITY;
        }

        // Check if a is between b and c (inclusive)
        let between_bc = (b <= a && a <= c) || (b >= a && a >= c);

        if between_bc {
            split_merge_cost
        } else {
            split_merge_cost + (a - b).abs().min((a - c).abs())
        }
    }
}

impl Default for MsmConfig {
    fn default() -> Self {
        Self::default_cost()
    }
}

/// Validation error for an exact raw-domain MSM configuration.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum MsmConfigError {
    /// The cost was negative or nonfinite.
    InvalidSplitMergeCost,
}

impl fmt::Display for MsmConfigError {
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        formatter.write_str("MSM split/merge cost must be finite and nonnegative")
    }
}

impl std::error::Error for MsmConfigError {}

/// MSM configuration whose documented nonempty domain is a metric space.
#[derive(Clone, Copy, Debug, Default, PartialEq)]
pub struct MetricMsmConfig(MsmConfig);

impl MetricMsmConfig {
    /// Construct a metric-domain MSM configuration with finite `c > 0`.
    pub fn try_new(c: f64) -> Result<Self, MetricMsmConfigError> {
        if !c.is_finite() || c <= 0.0 {
            return Err(MetricMsmConfigError::NonPositiveSplitMergeCost);
        }
        Ok(Self(MsmConfig { c }))
    }

    /// Borrow the validated raw MSM configuration.
    #[inline]
    pub fn as_config(&self) -> &MsmConfig {
        &self.0
    }

    /// Return the validated positive split/merge cost.
    #[inline]
    pub fn split_merge_cost(&self) -> f64 {
        self.0.c
    }

    /// Compute an exact, fail-closed cutoff decision on the nonempty metric domain.
    pub fn distance_bounded(
        &self,
        query: &[f64],
        candidate: &[f64],
        cutoff: f64,
        limits: ResourceLimits,
    ) -> Result<OperationOutcome<ExactDecision>, TemporalValidationError> {
        if query.is_empty() || candidate.is_empty() {
            return Err(TemporalValidationError::EmptyMetricSeries);
        }
        self.0.distance_bounded(query, candidate, cutoff, limits)
    }
}

impl TryFrom<MsmConfig> for MetricMsmConfig {
    type Error = MetricMsmConfigError;

    fn try_from(config: MsmConfig) -> Result<Self, Self::Error> {
        Self::try_new(config.c)
    }
}

/// Validation error for [`MetricMsmConfig`].
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum MetricMsmConfigError {
    /// The split/merge cost was zero, negative, or nonfinite.
    NonPositiveSplitMergeCost,
}

impl fmt::Display for MetricMsmConfigError {
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        formatter.write_str("metric MSM requires a finite split/merge cost greater than zero")
    }
}

impl std::error::Error for MetricMsmConfigError {}

/// Result of MSM distance computation with the DP matrix.
///
/// This is useful for debugging and understanding the alignment.
#[derive(Debug, Clone)]
pub struct MsmResult {
    /// The final MSM distance.
    pub distance: f64,

    /// The DP matrix. Entry (i, j) contains the cost of aligning
    /// x[0..i] with y[0..j]. The matrix is 1-indexed, so row 0 and
    /// column 0 are not meaningful (contain infinity).
    pub matrix: Vec<Vec<f64>>,
}

impl fmt::Display for MsmResult {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        writeln!(f, "MSM Distance: {}", self.distance)?;
        writeln!(f, "DP Matrix:")?;
        for (i, row) in self.matrix.iter().enumerate() {
            write!(f, "  [{:2}] ", i)?;
            for cell in row {
                if cell.is_infinite() {
                    write!(f, "  inf ")?;
                } else {
                    write!(f, "{:5.2} ", cell)?;
                }
            }
            writeln!(f)?;
        }
        Ok(())
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    const EPSILON: f64 = 1e-10;

    fn approx_eq(a: f64, b: f64) -> bool {
        (a - b).abs() < EPSILON
    }

    fn same_distance(a: f64, b: f64) -> bool {
        a == b || approx_eq(a, b)
    }

    fn assert_msm_case(config: &MsmConfig, x: &[f64], y: &[f64], expected: f64) {
        let standard = config.distance(x, y);
        let optimized = config.distance_optimized(x, y);
        let reverse = config.distance(y, x);

        assert!(
            approx_eq(standard, expected),
            "MSM({:?}, {:?}) = {}, expected {}",
            x,
            y,
            standard,
            expected
        );
        assert!(
            approx_eq(optimized, expected),
            "optimized MSM({:?}, {:?}) = {}, expected {}",
            x,
            y,
            optimized,
            expected
        );
        assert!(
            approx_eq(reverse, expected),
            "reverse MSM({:?}, {:?}) = {}, expected {}",
            y,
            x,
            reverse,
            expected
        );
    }

    #[test]
    fn test_msm_dp_dimension_helpers_reject_overflow() {
        assert_eq!(msm_dp_len(0), Some(1));
        assert_eq!(msm_dp_len(usize::MAX), None);
        assert_eq!(msm_dp_shape(2, 3), Some((3, 4)));
        assert_eq!(msm_dp_shape(usize::MAX, 0), None);
        assert_eq!(msm_dp_shape(0, usize::MAX), None);

        let matrix = msm_infinity_matrix(1, 2).expect("small matrix should allocate");
        assert_eq!(matrix.len(), 2);
        assert_eq!(matrix[0].len(), 3);
        assert!(matrix.iter().flatten().all(|value| value.is_infinite()));
    }

    #[test]
    fn test_identical_series() {
        let config = MsmConfig::new(1.0);
        let x = vec![1.0, 2.0, 3.0, 4.0, 5.0];
        assert!(approx_eq(config.distance(&x, &x), 0.0));
    }

    #[test]
    fn test_single_element_move() {
        let config = MsmConfig::new(1.0);
        // Move cost = |1.0 - 2.0| = 1.0
        assert!(approx_eq(config.distance(&[1.0], &[2.0]), 1.0));
        assert!(approx_eq(config.distance(&[0.0], &[5.0]), 5.0));
    }

    #[test]
    fn test_empty_series() {
        let config = MsmConfig::new(1.0);
        assert!(approx_eq(config.distance(&[], &[]), 0.0));
        assert!(config.distance(&[], &[1.0]).is_infinite());
        assert!(config.distance(&[1.0], &[]).is_infinite());
    }

    #[test]
    fn test_c_function_between() {
        let config = MsmConfig::new(1.0);

        // a is between b and c
        assert!(approx_eq(config.c_func(2.0, 1.0, 3.0), 1.0)); // 1 <= 2 <= 3
        assert!(approx_eq(config.c_func(2.0, 3.0, 1.0), 1.0)); // 1 <= 2 <= 3 (reversed)
        assert!(approx_eq(config.c_func(1.0, 1.0, 1.0), 1.0)); // All equal
    }

    #[test]
    fn test_c_function_outside() {
        let config = MsmConfig::new(1.0);

        // a is outside [b, c]
        // a=0, b=1, c=3: |0-1|=1, |0-3|=3, min=1, cost = 1 + 1 = 2
        assert!(approx_eq(config.c_func(0.0, 1.0, 3.0), 2.0));

        // a=5, b=1, c=3: |5-1|=4, |5-3|=2, min=2, cost = 1 + 2 = 3
        assert!(approx_eq(config.c_func(5.0, 1.0, 3.0), 3.0));
    }

    #[test]
    fn test_symmetry() {
        let config = MsmConfig::new(1.0);
        let x = vec![1.0, 2.0, 3.0, 2.0, 1.0];
        let y = vec![1.0, 3.0, 2.0];

        // MSM is symmetric
        let d_xy = config.distance(&x, &y);
        let d_yx = config.distance(&y, &x);
        assert!(approx_eq(d_xy, d_yx), "d(x,y)={} != d(y,x)={}", d_xy, d_yx);
    }

    #[test]
    fn test_triangle_inequality() {
        let config = MsmConfig::new(1.0);
        let x = vec![1.0, 2.0, 3.0];
        let y = vec![1.5, 2.5, 3.5];
        let z = vec![2.0, 3.0, 4.0];

        let d_xz = config.distance(&x, &z);
        let d_xy = config.distance(&x, &y);
        let d_yz = config.distance(&y, &z);

        assert!(
            d_xz <= d_xy + d_yz + EPSILON,
            "Triangle inequality violated: d(x,z)={} > d(x,y)+d(y,z)={}",
            d_xz,
            d_xy + d_yz
        );
    }

    #[test]
    fn test_default_distance_matches_matrix_path() {
        let config = MsmConfig::new(1.0);
        let cases: &[(&[f64], &[f64])] = &[
            (&[], &[]),
            (&[], &[1.0]),
            (&[1.0], &[]),
            (&[1.0, 2.0, 3.0, 4.0, 5.0], &[1.5, 2.5, 3.5, 4.5]),
            (&[0.0, 100.0], &[0.0, 0.0, 100.0]),
            (&[3.0, 1.0, 4.0, 1.0], &[2.0, 7.0]),
        ];

        for &(x, y) in cases {
            let default = config.distance(x, y);
            let optimized = config.distance_optimized(x, y);
            let matrix = config.distance_with_matrix(x, y).distance;

            assert!(
                same_distance(default, optimized),
                "default distance {default} != optimized distance {optimized} for {x:?}, {y:?}"
            );
            assert!(
                same_distance(default, matrix),
                "default distance {default} != matrix distance {matrix} for {x:?}, {y:?}"
            );
        }
    }

    #[test]
    fn test_cutoff_matches_standard_when_within_threshold() {
        let config = MsmConfig::new(1.0);
        let x = vec![1.0, 2.0, 3.0, 4.0, 5.0];
        let y = vec![1.5, 2.5, 3.5, 4.5];
        let expected = config.distance(&x, &y);

        let got = config.distance_with_cutoff(&x, &y, expected);
        assert!(
            got.is_some_and(|d| approx_eq(d, expected)),
            "cutoff result {got:?} did not match exact {expected}"
        );

        let reversed = config.distance_with_cutoff(&y, &x, expected);
        assert!(
            reversed.is_some_and(|d| approx_eq(d, expected)),
            "reversed cutoff result {reversed:?} did not match exact {expected}"
        );
    }

    #[test]
    fn test_cutoff_rejects_distances_above_threshold() {
        let config = MsmConfig::new(1.0);
        let x = vec![1.0, 2.0, 3.0, 4.0, 5.0];
        let y = vec![8.0, 9.0, 10.0, 11.0, 12.0];
        let exact = config.distance(&x, &y);

        assert!(config.distance_with_cutoff(&x, &y, exact - 0.5).is_none());
        assert_eq!(config.distance_with_cutoff(&[], &[1.0], 100.0), None);
        assert_eq!(
            config.distance_with_cutoff(&[], &[1.0], f64::INFINITY),
            Some(f64::INFINITY)
        );
        assert_eq!(config.distance_with_cutoff(&[], &[], 0.0), Some(0.0));
        assert_eq!(config.distance_with_cutoff(&[], &[], -1.0), None);
    }

    #[test]
    fn test_with_matrix() {
        let config = MsmConfig::new(1.0);
        let x = vec![1.0, 2.0, 3.0];
        let y = vec![1.0, 2.0, 3.0];

        let result = config.distance_with_matrix(&x, &y);
        assert!(approx_eq(result.distance, 0.0));
        assert_eq!(result.matrix.len(), 4); // (m+1) rows
        assert_eq!(result.matrix[0].len(), 4); // (n+1) columns
    }

    #[test]
    fn test_different_c_values() {
        let x = vec![1.0, 1.0, 2.0];
        let y = vec![1.0, 2.0];

        // With c=0.5
        let config1 = MsmConfig::new(0.5);
        let d1 = config1.distance(&x, &y);

        // With c=2.0
        let config2 = MsmConfig::new(2.0);
        let d2 = config2.distance(&x, &y);

        // Higher c should generally lead to higher distances
        // (unless the optimal path doesn't use split/merge)
        assert!(d1 <= d2 || approx_eq(d1, d2));
    }

    #[test]
    fn test_invalid_c_is_normalized() {
        assert_eq!(MsmConfig::new(-1.0).c, 0.0);
        assert_eq!(MsmConfig::new(f64::NAN), MsmConfig::default());
        assert_eq!(MsmConfig::new(f64::INFINITY), MsmConfig::default());
        assert_eq!(MsmConfig::new(f64::NEG_INFINITY), MsmConfig::default());
    }

    #[test]
    fn test_raw_malformed_c_uses_effective_cost() {
        let negative = MsmConfig { c: -2.0 };
        let infinite = MsmConfig { c: f64::INFINITY };

        assert_eq!(negative.split_merge_cost(), 0.0);
        assert_eq!(negative.c_func(5.0, 1.0, 3.0), 2.0);
        assert_eq!(infinite.split_merge_cost(), MsmConfig::default().c);
        assert_eq!(
            infinite.c_func(5.0, 1.0, 3.0),
            MsmConfig::default().c_func(5.0, 1.0, 3.0)
        );
    }

    #[test]
    fn test_non_finite_series_values_have_infinite_distance() {
        let config = MsmConfig::new(1.0);
        let cases: &[(&[f64], &[f64])] = &[
            (&[f64::NAN], &[1.0]),
            (&[f64::INFINITY], &[1.0]),
            (&[1.0], &[f64::NEG_INFINITY]),
            (&[1.0, f64::NAN], &[1.0, 2.0]),
        ];

        for &(x, y) in cases {
            assert!(
                config.distance(x, y).is_infinite(),
                "distance for {x:?}, {y:?}"
            );
            assert!(
                config.distance_optimized(x, y).is_infinite(),
                "optimized distance for {x:?}, {y:?}"
            );

            let matrix = config.distance_with_matrix(x, y);
            assert!(
                matrix.distance.is_infinite(),
                "matrix distance for {x:?}, {y:?}"
            );
        }
    }

    #[test]
    fn test_cutoff_rejects_non_finite_series_values_without_nan() {
        let config = MsmConfig::new(1.0);

        assert_eq!(
            config.distance_with_cutoff(&[f64::NAN], &[1.0], 100.0),
            None
        );
        assert_eq!(
            config.distance_with_cutoff(&[f64::NAN], &[1.0], f64::INFINITY),
            Some(f64::INFINITY)
        );
    }

    #[test]
    fn test_c_function_rejects_non_finite_point_values() {
        let config = MsmConfig::new(1.0);

        assert!(config.c_func(f64::NAN, 1.0, 2.0).is_infinite());
        assert!(config.c_func(1.0, f64::INFINITY, 2.0).is_infinite());
        assert!(config.c_func(1.0, 2.0, f64::NEG_INFINITY).is_infinite());
    }

    #[test]
    fn test_known_optimal_operation_cases() {
        let config = MsmConfig::new(1.0);

        assert_msm_case(&config, &[1.0, 2.0], &[1.0, 2.0], 0.0);
        assert_msm_case(&config, &[1.0, 2.0, 3.0], &[2.0, 3.0, 4.0], 3.0);
        assert_msm_case(&config, &[0.0, 100.0], &[0.0, 0.0, 100.0], 1.0);
        assert_msm_case(&config, &[0.0, 100.0], &[0.0, 100.0, 100.0], 1.0);
    }
}
