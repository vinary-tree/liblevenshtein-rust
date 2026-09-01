//! Additive real-valued costs.

use super::CostMonoid;
use std::cmp::Ordering;

/// Additive weighted-cost monoid over [`f64`].
///
/// The lawful domain is finite, non-negative values plus positive infinity.
/// [`f64::total_cmp`] supplies deterministic total ordering; NaN remains
/// orderable for defensive data-structure behavior but is outside the algebra's
/// lawful domain.
///
/// IEEE-754 addition is not bitwise associative for arbitrary operands. The
/// algebraic L1 proof therefore concerns the corresponding non-negative real
/// model. A Flocq proof bounds binary64 round-to-nearest-even reassociation;
/// tests separately pin exact associativity on same-scale dyadic inputs and a
/// forward-error envelope across non-negative finite binary64 bit patterns.
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq, Hash)]
pub struct WeightedCost;

impl CostMonoid for WeightedCost {
    type Cost = f64;

    const ZERO: Self::Cost = 0.0;
    const TOP: Self::Cost = f64::INFINITY;
    const EPSILON: Self::Cost = 1.0e-9;

    #[inline(always)]
    fn combine(accumulated: Self::Cost, step: Self::Cost) -> Self::Cost {
        if accumulated == Self::TOP || step == Self::TOP {
            Self::TOP
        } else {
            accumulated + step
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
