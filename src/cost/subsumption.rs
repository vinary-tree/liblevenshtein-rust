//! Shared structural subsumption for unit and weighted position families.
//!
//! The two position representations intentionally remain distinct. This seam
//! centralizes only their common Standard/OSA/MergeSplit decision tree while
//! keeping exact integer and canonical floating arithmetic explicit.

use super::{CostMonoid, UnitCost, WeightedCost};
use std::cmp::Ordering;

/// Carrier operations used by the shared structural relation.
///
/// This internal extension keeps subtraction and index realignment out of the
/// public [`CostMonoid`] contract: those operations are not monoid laws. Each
/// implementation exactly preserves its legacy machine comparison.
pub(crate) trait SubsumptionCost: CostMonoid {
    fn non_greater(lhs: Self::Cost, rhs: Self::Cost) -> bool;

    fn realignment_fits(
        lhs: Self::Cost,
        rhs: Self::Cost,
        index_difference: usize,
        maximum_index_operation_cost: Self::Cost,
    ) -> bool;

    fn strictly_less(lhs: Self::Cost, rhs: Self::Cost) -> bool;
}

impl SubsumptionCost for UnitCost {
    #[inline(always)]
    fn non_greater(lhs: usize, rhs: usize) -> bool {
        lhs <= rhs
    }

    #[inline(always)]
    fn realignment_fits(
        lhs: usize,
        rhs: usize,
        index_difference: usize,
        maximum_index_operation_cost: usize,
    ) -> bool {
        index_difference.saturating_mul(maximum_index_operation_cost) <= rhs - lhs
    }

    #[inline(always)]
    fn strictly_less(lhs: usize, rhs: usize) -> bool {
        lhs < rhs
    }
}

impl SubsumptionCost for WeightedCost {
    #[inline(always)]
    fn non_greater(lhs: f64, rhs: f64) -> bool {
        matches!(
            lhs.partial_cmp(&rhs),
            Some(Ordering::Less | Ordering::Equal)
        )
    }

    #[inline(always)]
    fn realignment_fits(
        lhs: f64,
        rhs: f64,
        index_difference: usize,
        maximum_index_operation_cost: f64,
    ) -> bool {
        let slack = rhs - lhs;
        index_difference as f64 * maximum_index_operation_cost <= slack
    }

    #[inline(always)]
    fn strictly_less(lhs: f64, rhs: f64) -> bool {
        matches!(lhs.partial_cmp(&rhs), Some(Ordering::Less))
    }
}

/// Structural relation shared by unit and weighted positions.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub(crate) enum SubsumptionMode {
    Standard,
    Transposition,
    MergeSplit,
}

/// Return whether the left position safely dominates the right position.
///
/// `maximum_index_operation_cost` is one for unit costs and
/// `max(insertion, deletion)` for the weighted family. The carrier extension
/// preserves exact carrier arithmetic; this function owns
/// all variant-state branching.
#[inline(always)]
#[allow(clippy::too_many_arguments)]
pub(crate) fn subsumes_with<M: SubsumptionCost>(
    lhs_index: usize,
    lhs_cost: M::Cost,
    lhs_special: bool,
    rhs_index: usize,
    rhs_cost: M::Cost,
    rhs_special: bool,
    mode: SubsumptionMode,
    query_length: usize,
    maximum_index_operation_cost: M::Cost,
) -> bool {
    if !M::non_greater(lhs_cost, rhs_cost) {
        return false;
    }

    match mode {
        SubsumptionMode::Standard => M::realignment_fits(
            lhs_cost,
            rhs_cost,
            lhs_index.abs_diff(rhs_index),
            maximum_index_operation_cost,
        ),
        SubsumptionMode::Transposition => match (lhs_special, rhs_special) {
            (true, true) => lhs_index == rhs_index,
            (true, false) | (false, true) => false,
            (false, false) => M::realignment_fits(
                lhs_cost,
                rhs_cost,
                lhs_index.abs_diff(rhs_index),
                maximum_index_operation_cost,
            ),
        },
        SubsumptionMode::MergeSplit => {
            if lhs_special != rhs_special || lhs_index > query_length {
                return false;
            }
            if lhs_special && lhs_index >= query_length && rhs_index < query_length {
                return false;
            }
            M::strictly_less(lhs_cost, rhs_cost) && lhs_index == rhs_index
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use proptest::prelude::*;

    #[allow(clippy::too_many_arguments)]
    fn legacy_unit(
        lhs_index: usize,
        lhs_cost: usize,
        lhs_special: bool,
        rhs_index: usize,
        rhs_cost: usize,
        rhs_special: bool,
        mode: SubsumptionMode,
        query_length: usize,
        maximum_index_operation_cost: usize,
    ) -> bool {
        if lhs_cost > rhs_cost {
            return false;
        }
        match mode {
            SubsumptionMode::Standard => {
                lhs_index
                    .abs_diff(rhs_index)
                    .saturating_mul(maximum_index_operation_cost)
                    <= rhs_cost - lhs_cost
            }
            SubsumptionMode::Transposition => match (lhs_special, rhs_special) {
                (true, true) => lhs_index == rhs_index,
                (true, false) | (false, true) => false,
                (false, false) => {
                    lhs_index
                        .abs_diff(rhs_index)
                        .saturating_mul(maximum_index_operation_cost)
                        <= rhs_cost - lhs_cost
                }
            },
            SubsumptionMode::MergeSplit => {
                lhs_special == rhs_special
                    && lhs_index <= query_length
                    && !(lhs_special && lhs_index >= query_length && rhs_index < query_length)
                    && lhs_cost < rhs_cost
                    && lhs_index == rhs_index
            }
        }
    }

    #[allow(clippy::too_many_arguments)]
    fn legacy_weighted(
        lhs_index: usize,
        lhs_cost: f64,
        lhs_special: bool,
        rhs_index: usize,
        rhs_cost: f64,
        rhs_special: bool,
        mode: SubsumptionMode,
        query_length: usize,
        maximum_index_operation_cost: f64,
    ) -> bool {
        if lhs_cost > rhs_cost {
            return false;
        }
        let realignment_fits = || {
            lhs_index.abs_diff(rhs_index) as f64 * maximum_index_operation_cost
                <= rhs_cost - lhs_cost
        };
        match mode {
            SubsumptionMode::Standard => realignment_fits(),
            SubsumptionMode::Transposition => match (lhs_special, rhs_special) {
                (true, true) => lhs_index == rhs_index,
                (true, false) | (false, true) => false,
                (false, false) => realignment_fits(),
            },
            SubsumptionMode::MergeSplit => {
                lhs_special == rhs_special
                    && lhs_index <= query_length
                    && !(lhs_special && lhs_index >= query_length && rhs_index < query_length)
                    && lhs_cost < rhs_cost
                    && lhs_index == rhs_index
            }
        }
    }

    fn mode_strategy() -> impl Strategy<Value = SubsumptionMode> {
        prop_oneof![
            Just(SubsumptionMode::Standard),
            Just(SubsumptionMode::Transposition),
            Just(SubsumptionMode::MergeSplit),
        ]
    }

    proptest! {
        #[test]
        fn unit_helper_is_extensionally_equal_to_the_two_legacy_formulas(
            lhs_index in 0usize..40,
            lhs_cost in 0usize..30,
            lhs_special in any::<bool>(),
            rhs_index in 0usize..40,
            rhs_cost in 0usize..30,
            rhs_special in any::<bool>(),
            mode in mode_strategy(),
            query_length in 0usize..40,
            maximum_index_operation_cost in 0usize..5,
        ) {
            prop_assert_eq!(
                subsumes_with::<UnitCost>(
                    lhs_index, lhs_cost, lhs_special, rhs_index, rhs_cost, rhs_special,
                    mode, query_length, maximum_index_operation_cost,
                ),
                legacy_unit(
                    lhs_index, lhs_cost, lhs_special, rhs_index, rhs_cost, rhs_special,
                    mode, query_length, maximum_index_operation_cost,
                ),
            );
        }

        #[test]
        fn weighted_helper_is_extensionally_equal_to_the_legacy_formula(
            lhs_index in 0usize..40,
            lhs_ticks in 0u16..3_000,
            lhs_special in any::<bool>(),
            rhs_index in 0usize..40,
            rhs_ticks in 0u16..3_000,
            rhs_special in any::<bool>(),
            mode in mode_strategy(),
            query_length in 0usize..40,
            operation_ticks in 0u16..500,
        ) {
            let lhs_cost = f64::from(lhs_ticks) / 100.0;
            let rhs_cost = f64::from(rhs_ticks) / 100.0;
            let maximum_index_operation_cost = f64::from(operation_ticks) / 100.0;
            prop_assert_eq!(
                subsumes_with::<WeightedCost>(
                    lhs_index, lhs_cost, lhs_special, rhs_index, rhs_cost, rhs_special,
                    mode, query_length, maximum_index_operation_cost,
                ),
                legacy_weighted(
                    lhs_index, lhs_cost, lhs_special, rhs_index, rhs_cost, rhs_special,
                    mode, query_length, maximum_index_operation_cost,
                ),
            );
        }
    }
}
