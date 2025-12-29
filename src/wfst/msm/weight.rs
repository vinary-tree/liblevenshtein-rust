//! Custom MSM weight for data-dependent cost WFST.
//!
//! Unlike [`TropicalWeight`] which only tracks accumulated cost, [`MsmWeight`]
//! also tracks the last consumed values from both series, which are required
//! for computing the data-dependent C() function during transitions.
//!
//! # Architecture
//!
//! The MSM metric uses three operations:
//! - **Move**: Cost = |x_i - y_j| (diagonal transition)
//! - **Merge-like**: Cost = C(x_i, x_{i-1}, y_j) (vertical transition)
//! - **Split-like**: Cost = C(y_j, x_i, y_{j-1}) (horizontal transition)
//!
//! The C(a, b, c) function depends on actual data values, making the weight
//! computation inherently different from Levenshtein or other operation-level metrics.
//!
//! # Semiring Properties
//!
//! `MsmWeight` forms a semiring under:
//! - **⊕ (plus)**: min over cost (keeping the lowest-cost weight)
//! - **⊗ (times)**: accumulation along paths (additive cost, value propagation)
//! - **0 (zero)**: Infinite cost (identity for ⊕)
//! - **1 (one)**: Zero cost (identity for ⊗)

use lling_llang::prelude::{Semiring, TropicalWeight};
use std::cmp::Ordering;
use std::fmt;
use std::hash::{Hash, Hasher};

/// Weight for MSM WFST with data-dependent costs.
///
/// This weight extends `TropicalWeight` by tracking the last consumed values
/// from both the query and target series, which are needed for computing
/// the C() function in merge-like and split-like operations.
#[derive(Clone, Copy)]
pub struct MsmWeight {
    /// Accumulated cost so far (tropical semiring value).
    cost: f64,

    /// The last query value consumed (x_{i-1}).
    /// Used for C() in merge-like operations.
    last_query_value: f64,

    /// The last target value consumed (y_{j-1}).
    /// Used for C() in split-like operations.
    last_target_value: f64,

    /// The c constant for split/merge operations.
    c_const: f64,
}

impl MsmWeight {
    /// Create a new MSM weight.
    #[inline]
    pub fn new(cost: f64, last_query_value: f64, last_target_value: f64, c_const: f64) -> Self {
        Self {
            cost,
            last_query_value,
            last_target_value,
            c_const,
        }
    }

    /// Create an initial weight for starting a path.
    #[inline]
    pub fn initial(first_query_value: f64, first_target_value: f64, c_const: f64) -> Self {
        Self {
            cost: 0.0,
            last_query_value: first_query_value,
            last_target_value: first_target_value,
            c_const,
        }
    }

    /// Get the accumulated cost.
    #[inline]
    pub fn cost(&self) -> f64 {
        self.cost
    }

    /// Get the last query value.
    #[inline]
    pub fn last_query_value(&self) -> f64 {
        self.last_query_value
    }

    /// Get the last target value.
    #[inline]
    pub fn last_target_value(&self) -> f64 {
        self.last_target_value
    }

    /// Get the c constant.
    #[inline]
    pub fn c_const(&self) -> f64 {
        self.c_const
    }

    /// Compute the C(a, b, c) function for MSM.
    ///
    /// C(a, b, c) = c_const           if b ≤ a ≤ c OR b ≥ a ≥ c
    ///            = c_const + min(|a-b|, |a-c|)  otherwise
    #[inline]
    pub fn c_func(&self, a: f64, b: f64, c: f64) -> f64 {
        let between_bc = (b <= a && a <= c) || (b >= a && a >= c);
        if between_bc {
            self.c_const
        } else {
            self.c_const + (a - b).abs().min((a - c).abs())
        }
    }

    /// Apply a Move operation: consume one element from both series.
    ///
    /// Cost = current + |query_value - target_value|
    #[inline]
    pub fn with_move(&self, query_value: f64, target_value: f64) -> Self {
        Self {
            cost: self.cost + (query_value - target_value).abs(),
            last_query_value: query_value,
            last_target_value: target_value,
            c_const: self.c_const,
        }
    }

    /// Apply a Merge-like operation: consume one query element only.
    ///
    /// Cost = current + C(query_value, last_query_value, last_target_value)
    #[inline]
    pub fn with_merge(&self, query_value: f64) -> Self {
        let c_cost = self.c_func(query_value, self.last_query_value, self.last_target_value);
        Self {
            cost: self.cost + c_cost,
            last_query_value: query_value,
            last_target_value: self.last_target_value,
            c_const: self.c_const,
        }
    }

    /// Apply a Split-like operation: consume one target element only.
    ///
    /// Cost = current + C(target_value, last_query_value, last_target_value)
    #[inline]
    pub fn with_split(&self, target_value: f64) -> Self {
        let c_cost = self.c_func(target_value, self.last_query_value, self.last_target_value);
        Self {
            cost: self.cost + c_cost,
            last_query_value: self.last_query_value,
            last_target_value: target_value,
            c_const: self.c_const,
        }
    }

    /// Convert to a TropicalWeight (loses value tracking).
    #[inline]
    pub fn to_tropical(&self) -> TropicalWeight {
        TropicalWeight::new(self.cost)
    }

    /// Create from a TropicalWeight with default values.
    #[inline]
    pub fn from_tropical(weight: TropicalWeight, c_const: f64) -> Self {
        Self {
            cost: weight.value(),
            last_query_value: 0.0,
            last_target_value: 0.0,
            c_const,
        }
    }
}

impl Default for MsmWeight {
    fn default() -> Self {
        Self {
            cost: 0.0,
            last_query_value: 0.0,
            last_target_value: 0.0,
            c_const: 1.0,
        }
    }
}

impl fmt::Debug for MsmWeight {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(
            f,
            "MsmWeight({:.3}, qv={:.2}, tv={:.2}, c={:.1})",
            self.cost, self.last_query_value, self.last_target_value, self.c_const
        )
    }
}

impl fmt::Display for MsmWeight {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        if self.cost.is_infinite() {
            write!(f, "∞")
        } else {
            write!(f, "{:.3}", self.cost)
        }
    }
}

impl PartialEq for MsmWeight {
    fn eq(&self, other: &Self) -> bool {
        const EPSILON: f64 = 1e-9;
        (self.cost - other.cost).abs() < EPSILON
            && (self.last_query_value - other.last_query_value).abs() < EPSILON
            && (self.last_target_value - other.last_target_value).abs() < EPSILON
            && (self.c_const - other.c_const).abs() < EPSILON
    }
}

impl Eq for MsmWeight {}

impl PartialOrd for MsmWeight {
    fn partial_cmp(&self, other: &Self) -> Option<Ordering> {
        Some(self.cmp(other))
    }
}

impl Ord for MsmWeight {
    fn cmp(&self, other: &Self) -> Ordering {
        self.cost
            .partial_cmp(&other.cost)
            .unwrap_or(Ordering::Equal)
    }
}

impl Hash for MsmWeight {
    fn hash<H: Hasher>(&self, state: &mut H) {
        // Hash the cost with reasonable precision
        let cost_bits = (self.cost * 1_000_000.0) as i64;
        cost_bits.hash(state);
    }
}

impl Semiring for MsmWeight {
    /// Zero element: infinite cost (identity for ⊕).
    fn zero() -> Self {
        Self {
            cost: f64::INFINITY,
            last_query_value: 0.0,
            last_target_value: 0.0,
            c_const: 1.0,
        }
    }

    /// One element: zero cost (identity for ⊗).
    fn one() -> Self {
        Self {
            cost: 0.0,
            last_query_value: 0.0,
            last_target_value: 0.0,
            c_const: 1.0,
        }
    }

    /// Check if this is the zero element.
    fn is_zero(&self) -> bool {
        self.cost.is_infinite() && self.cost.is_sign_positive()
    }

    /// Check if this is the one element.
    fn is_one(&self) -> bool {
        self.cost == 0.0
    }

    /// Plus operation (⊕): minimum cost.
    ///
    /// For MSM, we take the weight with minimum cost. The value tracking
    /// is preserved from the winning weight.
    fn plus(&self, other: &Self) -> Self {
        if self.cost <= other.cost {
            *self
        } else {
            *other
        }
    }

    /// Times operation (⊗): path extension.
    ///
    /// For path composition, we add costs and propagate the value tracking
    /// from the extension weight (other).
    fn times(&self, other: &Self) -> Self {
        Self {
            cost: self.cost + other.cost,
            // Propagate values from the later weight in the path
            last_query_value: if other.cost == 0.0 {
                self.last_query_value
            } else {
                other.last_query_value
            },
            last_target_value: if other.cost == 0.0 {
                self.last_target_value
            } else {
                other.last_target_value
            },
            c_const: self.c_const,
        }
    }

    /// Approximate equality for floating point comparison.
    fn approx_eq(&self, other: &Self, epsilon: f64) -> bool {
        if self.is_zero() && other.is_zero() {
            return true;
        }
        if self.is_zero() || other.is_zero() {
            return false;
        }
        (self.cost - other.cost).abs() <= epsilon
    }

    /// Natural ordering: smaller cost is better.
    fn natural_less(&self, other: &Self) -> Option<bool> {
        Some(self.cost < other.cost)
    }

    /// Serialize to bytes.
    fn to_bytes(&self) -> Vec<u8> {
        self.cost.to_le_bytes().to_vec()
    }
}

impl lling_llang::semiring::NumericalWeight for MsmWeight {
    #[inline]
    fn numerical_value(&self) -> f64 {
        self.cost
    }
}

/// Convert MsmWeight to TropicalWeight for composition with standard WFSTs.
impl From<MsmWeight> for TropicalWeight {
    fn from(msm: MsmWeight) -> Self {
        TropicalWeight::new(msm.cost)
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
    fn test_new() {
        let w = MsmWeight::new(1.5, 2.0, 3.0, 1.0);
        assert!(approx_eq(w.cost(), 1.5));
        assert!(approx_eq(w.last_query_value(), 2.0));
        assert!(approx_eq(w.last_target_value(), 3.0));
        assert!(approx_eq(w.c_const(), 1.0));
    }

    #[test]
    fn test_initial() {
        let w = MsmWeight::initial(1.0, 2.0, 0.5);
        assert!(approx_eq(w.cost(), 0.0));
        assert!(approx_eq(w.last_query_value(), 1.0));
        assert!(approx_eq(w.last_target_value(), 2.0));
        assert!(approx_eq(w.c_const(), 0.5));
    }

    #[test]
    fn test_c_func_between() {
        let w = MsmWeight::initial(0.0, 0.0, 1.0);

        // a is between b and c
        assert!(approx_eq(w.c_func(2.0, 1.0, 3.0), 1.0)); // 1 <= 2 <= 3
        assert!(approx_eq(w.c_func(2.0, 3.0, 1.0), 1.0)); // 1 <= 2 <= 3 (reversed)
        assert!(approx_eq(w.c_func(1.0, 1.0, 1.0), 1.0)); // All equal
    }

    #[test]
    fn test_c_func_outside() {
        let w = MsmWeight::initial(0.0, 0.0, 1.0);

        // a outside [b, c]
        // a=0, b=1, c=3: |0-1|=1, |0-3|=3, min=1, cost = 1 + 1 = 2
        assert!(approx_eq(w.c_func(0.0, 1.0, 3.0), 2.0));

        // a=5, b=1, c=3: |5-1|=4, |5-3|=2, min=2, cost = 1 + 2 = 3
        assert!(approx_eq(w.c_func(5.0, 1.0, 3.0), 3.0));
    }

    #[test]
    fn test_with_move() {
        let w = MsmWeight::initial(1.0, 2.0, 1.0);
        let w2 = w.with_move(1.5, 2.5);

        // Cost = |1.5 - 2.5| = 1.0
        assert!(approx_eq(w2.cost(), 1.0));
        assert!(approx_eq(w2.last_query_value(), 1.5));
        assert!(approx_eq(w2.last_target_value(), 2.5));
    }

    #[test]
    fn test_with_merge() {
        let w = MsmWeight::initial(1.0, 2.0, 1.0);
        let w2 = w.with_merge(1.5);

        // C(1.5, 1.0, 2.0) - 1.5 is between 1.0 and 2.0, so cost = c_const = 1.0
        assert!(approx_eq(w2.cost(), 1.0));
        assert!(approx_eq(w2.last_query_value(), 1.5));
        assert!(approx_eq(w2.last_target_value(), 2.0)); // unchanged
    }

    #[test]
    fn test_with_split() {
        let w = MsmWeight::initial(1.0, 2.0, 1.0);
        let w2 = w.with_split(2.5);

        // C(2.5, 1.0, 2.0) - 2.5 > 2.0, so cost = c_const + min(|2.5-1.0|, |2.5-2.0|)
        // = 1.0 + min(1.5, 0.5) = 1.5
        assert!(approx_eq(w2.cost(), 1.5));
        assert!(approx_eq(w2.last_query_value(), 1.0)); // unchanged
        assert!(approx_eq(w2.last_target_value(), 2.5));
    }

    #[test]
    fn test_semiring_zero() {
        let zero = MsmWeight::zero();
        assert!(zero.is_zero());
        assert!(zero.cost().is_infinite());
    }

    #[test]
    fn test_semiring_one() {
        let one = MsmWeight::one();
        assert!(one.is_one());
        assert!(approx_eq(one.cost(), 0.0));
    }

    #[test]
    fn test_semiring_plus() {
        let w1 = MsmWeight::new(1.0, 0.0, 0.0, 1.0);
        let w2 = MsmWeight::new(2.0, 0.0, 0.0, 1.0);

        let result = w1.plus(&w2);
        assert!(approx_eq(result.cost(), 1.0)); // min(1.0, 2.0) = 1.0
    }

    #[test]
    fn test_semiring_times() {
        let w1 = MsmWeight::new(1.0, 1.0, 2.0, 1.0);
        let w2 = MsmWeight::new(0.5, 1.5, 2.5, 1.0);

        let result = w1.times(&w2);
        assert!(approx_eq(result.cost(), 1.5)); // 1.0 + 0.5 = 1.5
        assert!(approx_eq(result.last_query_value(), 1.5)); // from w2
        assert!(approx_eq(result.last_target_value(), 2.5)); // from w2
    }

    #[test]
    fn test_to_tropical() {
        let w = MsmWeight::new(2.5, 1.0, 2.0, 1.0);
        let t = w.to_tropical();
        assert!(approx_eq(t.value(), 2.5));
    }

    #[test]
    fn test_ordering() {
        let w1 = MsmWeight::new(1.0, 0.0, 0.0, 1.0);
        let w2 = MsmWeight::new(2.0, 0.0, 0.0, 1.0);
        let w3 = MsmWeight::new(1.0, 0.0, 0.0, 1.0); // Same values as w1

        assert!(w1 < w2);
        assert!(w1 == w3); // Same cost and tracked values

        // Weights with same cost but different tracked values are not equal
        let w4 = MsmWeight::new(1.0, 1.0, 1.0, 1.0);
        assert!(w1 != w4);

        // But for ordering, they should be equivalent (neither is less than the other)
        assert!(!(w1 < w4) && !(w4 < w1));
    }

    #[test]
    fn test_display() {
        let w = MsmWeight::new(1.5, 0.0, 0.0, 1.0);
        assert_eq!(format!("{}", w), "1.500");

        let zero = MsmWeight::zero();
        assert_eq!(format!("{}", zero), "∞");
    }
}
