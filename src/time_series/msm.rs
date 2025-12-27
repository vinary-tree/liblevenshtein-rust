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

/// Configuration for MSM distance computation.
///
/// # Fields
///
/// - `c`: The constant cost for split and merge operations. Must be non-negative.
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
    /// Create a new MSM configuration.
    ///
    /// # Arguments
    ///
    /// * `c` - The constant cost for split/merge operations. Must be >= 0.
    ///
    /// # Panics
    ///
    /// Panics if `c` is negative.
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
        assert!(c >= 0.0, "Split/merge cost c must be non-negative, got {}", c);
        Self { c }
    }

    /// Create a configuration with the default cost c = 1.0.
    ///
    /// This is equivalent to `MsmConfig::new(1.0)`.
    #[inline]
    pub fn default_cost() -> Self {
        Self::new(1.0)
    }

    /// Compute the MSM distance between two time series.
    ///
    /// This implements the O(mn) dynamic programming algorithm from Figure 10
    /// of the Stefan et al. paper.
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
        let m = x.len();
        let n = y.len();

        // Handle empty series
        if m == 0 && n == 0 {
            return 0.0;
        }
        if m == 0 || n == 0 {
            // Can't transform empty to non-empty with Move/Split/Merge
            return f64::INFINITY;
        }

        // Allocate DP matrix (1-indexed, so size is (m+1) x (n+1))
        let mut cost = vec![vec![f64::INFINITY; n + 1]; m + 1];

        // Base case: Cost(1,1) = |x_0 - y_0|
        cost[1][1] = (x[0] - y[0]).abs();

        // Initialize first column (j=1): accumulate using the C function
        for i in 2..=m {
            // Cost(i, 1) = Cost(i-1, 1) + C(x_{i-1}, x_{i-2}, y_0)
            cost[i][1] = cost[i - 1][1] + self.c_func(x[i - 1], x[i - 2], y[0]);
        }

        // Initialize first row (i=1): accumulate using the C function
        for j in 2..=n {
            // Cost(1, j) = Cost(1, j-1) + C(y_{j-1}, x_0, y_{j-2})
            cost[1][j] = cost[1][j - 1] + self.c_func(y[j - 1], x[0], y[j - 2]);
        }

        // Fill the DP matrix
        for i in 2..=m {
            for j in 2..=n {
                // Move: diagonal transition with cost |x_i - y_j|
                let move_cost = cost[i - 1][j - 1] + (x[i - 1] - y[j - 1]).abs();

                // Merge-like: vertical transition with C(x_i, x_{i-1}, y_j)
                let merge_cost = cost[i - 1][j] + self.c_func(x[i - 1], x[i - 2], y[j - 1]);

                // Split-like: horizontal transition with C(y_j, x_i, y_{j-1})
                let split_cost = cost[i][j - 1] + self.c_func(y[j - 1], x[i - 1], y[j - 2]);

                cost[i][j] = move_cost.min(merge_cost).min(split_cost);
            }
        }

        cost[m][n]
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

        // Use two rows: previous and current
        let mut prev = vec![f64::INFINITY; n + 1];
        let mut curr = vec![f64::INFINITY; n + 1];

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
                matrix: vec![vec![f64::INFINITY; n + 1]; m + 1],
            };
        }

        // Allocate DP matrix
        let mut cost = vec![vec![f64::INFINITY; n + 1]; m + 1];

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
        // Check if a is between b and c (inclusive)
        let between_bc = (b <= a && a <= c) || (b >= a && a >= c);

        if between_bc {
            self.c
        } else {
            self.c + (a - b).abs().min((a - c).abs())
        }
    }
}

impl Default for MsmConfig {
    fn default() -> Self {
        Self::default_cost()
    }
}

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
    fn test_optimized_matches_standard() {
        let config = MsmConfig::new(1.0);
        let x = vec![1.0, 2.0, 3.0, 4.0, 5.0];
        let y = vec![1.5, 2.5, 3.5, 4.5];

        let d1 = config.distance(&x, &y);
        let d2 = config.distance_optimized(&x, &y);

        assert!(
            approx_eq(d1, d2),
            "Standard distance {} != optimized distance {}",
            d1,
            d2
        );
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
    #[should_panic(expected = "must be non-negative")]
    fn test_negative_c_panics() {
        MsmConfig::new(-1.0);
    }

    #[test]
    fn test_paper_example() {
        // Example from the paper (if we can find specific test cases)
        // For now, test with simple known cases

        let config = MsmConfig::new(1.0);

        // Two-element series
        let x = vec![1.0, 2.0];
        let y = vec![1.0, 2.0];
        assert!(approx_eq(config.distance(&x, &y), 0.0));

        // Shift all values by 1
        let x = vec![1.0, 2.0, 3.0];
        let y = vec![2.0, 3.0, 4.0];
        // Optimal: move each element by 1, cost = 1+1+1 = 3
        assert!(approx_eq(config.distance(&x, &y), 3.0));
    }
}
