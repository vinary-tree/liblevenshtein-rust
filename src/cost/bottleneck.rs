//! Minimax/bottleneck path costs.

use super::CostMonoid;
use std::cmp::Ordering;

/// Bottleneck-cost monoid over [`f64`], combining a path with `max`.
///
/// The lawful domain is finite, non-negative values plus positive infinity.
/// This is the accumulation rule used by discrete Fréchet-style minimax DPs.
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq, Hash)]
pub struct BottleneckCost;

impl CostMonoid for BottleneckCost {
    type Cost = f64;

    const ZERO: Self::Cost = 0.0;
    const TOP: Self::Cost = f64::INFINITY;
    const EPSILON: Self::Cost = 1.0e-9;

    #[inline(always)]
    fn combine(accumulated: Self::Cost, step: Self::Cost) -> Self::Cost {
        if accumulated == Self::TOP || step == Self::TOP {
            return Self::TOP;
        }
        if Self::compare(accumulated, step) == Ordering::Less {
            step
        } else {
            accumulated
        }
    }

    #[inline(always)]
    fn compare(a: Self::Cost, b: Self::Cost) -> Ordering {
        a.total_cmp(&b)
    }

    #[inline(always)]
    fn within(cost: Self::Cost, threshold: Self::Cost) -> bool {
        if cost.is_nan() || threshold.is_nan() {
            return false;
        }
        Self::compare(cost, threshold) != Ordering::Greater
    }

    #[inline]
    fn canonical_state_key(cost: Self::Cost) -> Option<u64> {
        (!cost.is_nan()).then(|| if cost == 0.0 { 0 } else { cost.to_bits() })
    }
}
