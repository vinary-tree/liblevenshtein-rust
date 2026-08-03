use crate::cost::{CostScale, ScaleError};
use crate::transducer::variant::{AutomatonVariant, TransitionCtx};
use crate::transducer::{Position, PositionKind};
use smallvec::SmallVec;

/// Exact fixed-point costs for the affine-gap automaton.
///
/// A gap containing `k` units costs `gap_open + k * gap_extend`.  The
/// parameters retain their [`CostScale`] so public APIs can convert costs at
/// the boundary while the transition and subsumption kernels compare exact
/// integers.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub struct AffineGapParams {
    gap_open: usize,
    gap_extend: usize,
    substitution: usize,
    scale: CostScale,
}

impl AffineGapParams {
    /// Derive the least exact decimal scale and convert all three costs.
    pub fn new(gap_open: f64, gap_extend: f64, substitution: f64) -> Result<Self, ScaleError> {
        let scale = CostScale::for_weights([gap_open, gap_extend, substitution])?;
        Self::with_scale(scale, gap_open, gap_extend, substitution)
    }

    /// Convert costs with a caller-selected exact scale.
    pub fn with_scale(
        scale: CostScale,
        gap_open: f64,
        gap_extend: f64,
        substitution: f64,
    ) -> Result<Self, ScaleError> {
        Ok(Self {
            gap_open: scale.to_scaled(gap_open)?,
            gap_extend: scale.to_scaled(gap_extend)?,
            substitution: scale.to_scaled(substitution)?,
            scale,
        })
    }

    /// Construct parameters whose costs are already expressed in `scale`.
    pub const fn from_scaled(
        scale: CostScale,
        gap_open: usize,
        gap_extend: usize,
        substitution: usize,
    ) -> Self {
        Self {
            gap_open,
            gap_extend,
            substitution,
            scale,
        }
    }

    /// Fixed-point scale shared by every cost in this parameter set.
    pub const fn scale(self) -> CostScale {
        self.scale
    }

    /// Scaled gap-open cost.
    pub const fn gap_open(self) -> usize {
        self.gap_open
    }

    /// Scaled cost for each unit in a gap, including its first unit.
    pub const fn gap_extend(self) -> usize {
        self.gap_extend
    }

    /// Scaled substitution cost.
    pub const fn substitution(self) -> usize {
        self.substitution
    }

    /// Convert a real-valued query budget into this exact scale.
    pub fn scale_cost(self, cost: f64) -> Result<usize, ScaleError> {
        self.scale.to_scaled(cost)
    }

    /// Convert an exact automaton cost into a presentation value.
    pub fn unscale_cost(self, cost: usize) -> f64 {
        self.scale.from_scaled(cost)
    }

    #[inline(always)]
    fn gap_step(self, source: PositionKind, target: PositionKind) -> Option<usize> {
        if source == target {
            Some(self.gap_extend)
        } else {
            self.gap_open.checked_add(self.gap_extend)
        }
    }
}

/// Three-layer Gotoh automaton (`M`, query gap, dictionary gap).
#[derive(Debug, Clone, Copy)]
pub(crate) struct AffineV;

impl AffineV {
    #[inline(always)]
    fn layer_precedes(lhs: PositionKind, rhs: PositionKind) -> bool {
        lhs == rhs || rhs == PositionKind::Normal
    }

    #[inline(always)]
    fn checked_cost(position: Position, increment: usize, maximum: usize) -> Option<usize> {
        position
            .num_errors
            .checked_add(increment)
            .filter(|cost| *cost <= maximum)
    }
}

impl AutomatonVariant for AffineV {
    type Params = AffineGapParams;

    #[inline(always)]
    fn successors(
        position: Position,
        characteristic_vector: &[bool],
        ctx: &TransitionCtx<Self::Params>,
        out: &mut SmallVec<[Position; 4]>,
    ) {
        debug_assert!(out.is_empty());

        if ctx.prefix_mode && position.term_index >= ctx.query_length {
            out.push(position);
            return;
        }

        if position.term_index < ctx.query_length {
            let substitution = if characteristic_vector.first() == Some(&true) {
                0
            } else {
                ctx.params.substitution
            };
            if let (Some(term_index), Some(cost)) = (
                position.term_index.checked_add(1),
                Self::checked_cost(position, substitution, ctx.max_distance),
            ) {
                out.push(Position::new(term_index, cost));
            }
        }

        if let Some(increment) = ctx
            .params
            .gap_step(position.kind(), PositionKind::AffineDictGap)
        {
            if let Some(cost) = Self::checked_cost(position, increment, ctx.max_distance) {
                out.push(Position::new_affine_dict_gap(position.term_index, cost));
            }
        }
    }

    #[inline(always)]
    fn epsilon_successors(
        position: Position,
        ctx: &TransitionCtx<Self::Params>,
        out: &mut SmallVec<[Position; 4]>,
    ) {
        if position.term_index >= ctx.query_length {
            return;
        }
        let Some(increment) = ctx
            .params
            .gap_step(position.kind(), PositionKind::AffineQueryGap)
        else {
            return;
        };
        if let (Some(term_index), Some(cost)) = (
            position.term_index.checked_add(1),
            Self::checked_cost(position, increment, ctx.max_distance),
        ) {
            out.push(Position::new_affine_query_gap(term_index, cost));
        }
    }

    #[inline(always)]
    fn subsumes(lhs: &Position, rhs: &Position, ctx: &TransitionCtx<Self::Params>) -> bool {
        if lhs.term_index != rhs.term_index {
            // B-5's forward inequality is a valid semantic bound, but enabling
            // it in this closure architecture also requires a fused
            // skip-and-consume successor. Without that successor, canonicalizing
            // an epsilon representative can hide the only matching cv offset.
            // The backwards direction is additionally vulnerable to splitting
            // a gap run. B-4 therefore remains intentionally same-index.
            return false;
        }
        (Self::layer_precedes(lhs.kind(), rhs.kind()) && lhs.num_errors <= rhs.num_errors)
            || lhs
                .num_errors
                .checked_add(ctx.params.gap_open)
                .is_some_and(|cost| cost <= rhs.num_errors)
    }

    #[inline(always)]
    fn finish_cost(
        position: &Position,
        query_length: usize,
        params: Self::Params,
    ) -> Option<usize> {
        let remaining = query_length.saturating_sub(position.term_index);
        if remaining == 0 {
            return Some(position.num_errors);
        }
        let opening = if position.kind() == PositionKind::AffineQueryGap {
            0
        } else {
            params.gap_open
        };
        remaining
            .checked_mul(params.gap_extend)
            .and_then(|run| position.num_errors.checked_add(opening)?.checked_add(run))
    }

    #[inline(always)]
    fn skip_window(position: &Position, ctx: &TransitionCtx<Self::Params>) -> usize {
        let remaining_query = ctx.query_length.saturating_sub(position.term_index);
        let remaining_cost = ctx.max_distance.saturating_sub(position.num_errors);
        let operation_budget = remaining_cost
            .checked_div(ctx.params.gap_extend)
            .unwrap_or(remaining_query);
        operation_budget
            .saturating_add(1)
            .min(remaining_query.saturating_add(1))
            .max(1)
    }

    #[inline(always)]
    fn supports_zero_distance_fast_path(ctx: &TransitionCtx<Self::Params>) -> bool {
        ctx.params.substitution > 0
            && ctx.params.gap_extend > 0
            && ctx
                .params
                .gap_open
                .checked_add(ctx.params.gap_extend)
                .is_some_and(|cost| cost > 0)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use proptest::prelude::*;
    use std::collections::HashMap;

    fn params(open: usize, extend: usize, substitution: usize) -> AffineGapParams {
        AffineGapParams::from_scaled(
            CostScale::new(1).expect("unit scale"),
            open,
            extend,
            substitution,
        )
    }

    #[test]
    fn decimal_parameters_are_exactly_scaled() {
        let params = AffineGapParams::new(0.5, 0.25, 1.0).expect("finite decimals");
        assert_eq!(params.scale().denominator(), 4);
        assert_eq!(params.gap_open(), 2);
        assert_eq!(params.gap_extend(), 1);
        assert_eq!(params.substitution(), 4);
        assert_eq!(params.scale_cost(1.5), Ok(6));
        assert_eq!(params.unscale_cost(6), 1.5);
    }

    #[test]
    fn same_index_layer_order_and_switch_penalty_are_executable() {
        let ctx = TransitionCtx::new(4, 20, false, params(3, 1, 2));
        let query_gap = Position::new_affine_query_gap(2, 4);
        let normal = Position::new(2, 4);
        let dict_gap = Position::new_affine_dict_gap(2, 4);

        assert!(AffineV::subsumes(&query_gap, &normal, &ctx));
        assert!(AffineV::subsumes(&dict_gap, &normal, &ctx));
        assert!(!AffineV::subsumes(&query_gap, &dict_gap, &ctx));
        assert!(!AffineV::subsumes(&dict_gap, &query_gap, &ctx));
        assert!(AffineV::subsumes(
            &Position::new_affine_query_gap(2, 1),
            &Position::new_affine_dict_gap(2, 4),
            &ctx
        ));
    }

    #[test]
    fn trailing_query_gap_extends_without_reopening() {
        let params = params(3, 2, 4);
        assert_eq!(
            AffineV::finish_cost(&Position::new_affine_query_gap(1, 5), 4, params),
            Some(11)
        );
        assert_eq!(
            AffineV::finish_cost(&Position::new(1, 5), 4, params),
            Some(14)
        );
    }

    #[test]
    fn scaled_window_depends_on_operations_not_raw_cost_units() {
        let params = AffineGapParams::from_scaled(
            CostScale::new(1_000).expect("non-zero"),
            1_000,
            250,
            1_000,
        );
        let ctx = TransitionCtx::new(10_000, 2_000, false, params);
        assert_eq!(AffineV::skip_window(&Position::new(0, 1_000), &ctx), 5);
    }

    fn kind(tag: u8) -> PositionKind {
        match tag % 3 {
            0 => PositionKind::Normal,
            1 => PositionKind::AffineQueryGap,
            _ => PositionKind::AffineDictGap,
        }
    }

    fn completion_cost(
        query: &[u8],
        dictionary: &[u8],
        query_index: usize,
        dictionary_index: usize,
        incoming: PositionKind,
        params: AffineGapParams,
        memo: &mut HashMap<(usize, usize, PositionKind), usize>,
    ) -> usize {
        if let Some(cost) = memo.get(&(query_index, dictionary_index, incoming)) {
            return *cost;
        }
        if query_index == query.len() && dictionary_index == dictionary.len() {
            return 0;
        }

        let mut best = usize::MAX;
        if query_index < query.len() && dictionary_index < dictionary.len() {
            let substitution = if query[query_index] == dictionary[dictionary_index] {
                0
            } else {
                params.substitution
            };
            best = best.min(
                substitution
                    + completion_cost(
                        query,
                        dictionary,
                        query_index + 1,
                        dictionary_index + 1,
                        PositionKind::Normal,
                        params,
                        memo,
                    ),
            );
        }
        if query_index < query.len() {
            best = best.min(
                params
                    .gap_step(incoming, PositionKind::AffineQueryGap)
                    .expect("small generated cost")
                    + completion_cost(
                        query,
                        dictionary,
                        query_index + 1,
                        dictionary_index,
                        PositionKind::AffineQueryGap,
                        params,
                        memo,
                    ),
            );
        }
        if dictionary_index < dictionary.len() {
            best = best.min(
                params
                    .gap_step(incoming, PositionKind::AffineDictGap)
                    .expect("small generated cost")
                    + completion_cost(
                        query,
                        dictionary,
                        query_index,
                        dictionary_index + 1,
                        PositionKind::AffineDictGap,
                        params,
                        memo,
                    ),
            );
        }
        memo.insert((query_index, dictionary_index, incoming), best);
        best
    }

    proptest! {
        #![proptest_config(ProptestConfig::with_cases(2_000))]

        #[test]
        fn b4_subsumption_is_sound_for_every_generated_suffix(
            query in prop::collection::vec(0u8..=2, 0..=6),
            suffix in prop::collection::vec(0u8..=2, 0..=6),
            raw_index in 0usize..=6,
            left_cost in 0usize..=20,
            right_cost in 0usize..=20,
            left_tag in 0u8..3,
            right_tag in 0u8..3,
            open in 0usize..=5,
            extend in 0usize..=5,
            substitution in 0usize..=5,
        ) {
            let index = raw_index.min(query.len());
            let params = params(open, extend, substitution);
            let left = Position::with_kind(index, left_cost, kind(left_tag), 0);
            let right = Position::with_kind(index, right_cost, kind(right_tag), 0);
            let ctx = TransitionCtx::new(query.len(), 100, false, params);

            if AffineV::subsumes(&left, &right, &ctx) {
                let left_total = left_cost + completion_cost(
                    &query, &suffix, index, 0, left.kind(), params, &mut HashMap::new()
                );
                let right_total = right_cost + completion_cost(
                    &query, &suffix, index, 0, right.kind(), params, &mut HashMap::new()
                );
                prop_assert!(left_total <= right_total);
            }
        }

        #[test]
        fn uniform_switch_penalty_is_an_executable_invariant(
            left_tag in 0u8..3,
            right_tag in 0u8..3,
            target_tag in 1u8..3,
            open in 0usize..=20,
            extend in 0usize..=20,
        ) {
            let params = params(open, extend, 1);
            let left = params.gap_step(kind(left_tag), kind(target_tag)).expect("small costs");
            let right = params.gap_step(kind(right_tag), kind(target_tag)).expect("small costs");
            prop_assert!(left <= right + open);
        }

        #[test]
        fn operation_window_contains_every_affordable_run(
            maximum in 0usize..=10_000,
            raw_cost in 0usize..=10_000,
            extend in 1usize..=1_000,
            operations in 0usize..=100,
        ) {
            let cost = raw_cost.min(maximum);
            let params = params(0, extend, 1);
            let ctx = TransitionCtx::new(20_000, maximum, false, params);
            let position = Position::new(0, cost);
            let window = AffineV::skip_window(&position, &ctx);
            if cost.checked_add(operations * extend).is_some_and(|total| total <= maximum) {
                prop_assert!(operations < window);
            }
        }
    }
}
