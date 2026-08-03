//! Saturating unit edit costs.

use super::CostMonoid;
use std::cmp::Ordering;

/// Unit-cost additive monoid over [`usize`].
///
/// Addition saturates at [`usize::MAX`], which is the distinguished `TOP`.
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq, Hash)]
pub struct UnitCost;

impl CostMonoid for UnitCost {
    type Cost = usize;

    const ZERO: Self::Cost = 0;
    const TOP: Self::Cost = usize::MAX;
    const EPSILON: Self::Cost = 0;

    #[inline(always)]
    fn combine(accumulated: Self::Cost, step: Self::Cost) -> Self::Cost {
        accumulated.saturating_add(step)
    }

    #[inline(always)]
    fn compare(a: Self::Cost, b: Self::Cost) -> Ordering {
        a.cmp(&b)
    }

    #[inline(always)]
    fn within(cost: Self::Cost, threshold: Self::Cost) -> bool {
        cost <= threshold
    }
}
