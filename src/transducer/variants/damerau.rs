use super::{unit_finish_cost, unit_skip_window, StandardV};
use crate::transducer::transition::transition_damerau_into;
use crate::transducer::variant::{AutomatonVariant, PositionKind, TransitionCtx};
use crate::transducer::Position;
use smallvec::SmallVec;

/// Unit-cost, unrestricted Damerau–Levenshtein variant.
///
/// A [`PositionKind::DamerauPending`] representative stores the positive query
/// endpoint delta in [`Position::aux`]. The entry cost prepays the transposition
/// and intervening query deletions; subsequent dictionary units add insertion
/// cost until the deferred endpoint resolves.
#[derive(Debug, Clone, Copy)]
pub(crate) struct DamerauV;

impl AutomatonVariant for DamerauV {
    type Params = ();

    #[inline(always)]
    fn successors(
        position: Position,
        characteristic_vector: &[bool],
        ctx: &TransitionCtx<()>,
        out: &mut SmallVec<[Position; 4]>,
    ) {
        transition_damerau_into(
            &position,
            characteristic_vector,
            ctx.query_length,
            ctx.max_distance,
            ctx.prefix_mode,
            out,
        );
    }

    #[inline(always)]
    fn epsilon_successors(
        position: Position,
        ctx: &TransitionCtx<()>,
        out: &mut SmallVec<[Position; 4]>,
    ) {
        if position.kind() == PositionKind::Normal {
            StandardV::epsilon_successors(position, ctx, out);
        }
    }

    #[inline(always)]
    fn subsumes(lhs: &Position, rhs: &Position, _ctx: &TransitionCtx<()>) -> bool {
        if lhs.num_errors > rhs.num_errors {
            return false;
        }

        match (lhs.kind(), rhs.kind()) {
            (PositionKind::Normal, PositionKind::Normal) => {
                lhs.term_index.abs_diff(rhs.term_index) <= rhs.num_errors - lhs.num_errors
            }
            (PositionKind::DamerauPending, PositionKind::DamerauPending) => {
                lhs.term_index == rhs.term_index && lhs.aux() == rhs.aux()
            }
            _ => false,
        }
    }

    #[inline(always)]
    fn finish_cost(position: &Position, query_length: usize, (): ()) -> Option<usize> {
        (position.kind() == PositionKind::Normal)
            .then(|| unit_finish_cost(position, query_length))
            .flatten()
    }

    #[inline(always)]
    fn skip_window(position: &Position, ctx: &TransitionCtx<()>) -> usize {
        unit_skip_window(position, ctx)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::transducer::transition::transition_window_size;
    use proptest::prelude::*;

    fn characteristic_vector(
        position: Position,
        unit: u8,
        query: &[u8],
        budget: usize,
    ) -> Vec<bool> {
        let window = transition_window_size(budget, query.len());
        (0..window)
            .map(|offset| {
                position
                    .term_index
                    .checked_add(offset)
                    .and_then(|index| query.get(index))
                    .is_some_and(|candidate| *candidate == unit)
            })
            .collect()
    }

    fn epsilon_closure_raw(positions: &mut Vec<Position>, ctx: &TransitionCtx<()>) {
        let mut cursor = 0;
        while cursor < positions.len() {
            let mut successors = SmallVec::<[Position; 4]>::new();
            DamerauV::epsilon_successors(positions[cursor], ctx, &mut successors);
            for successor in successors {
                if !positions.contains(&successor) {
                    positions.push(successor);
                }
            }
            cursor += 1;
        }
    }

    /// Evaluate one representative without applying subsumption. This is the
    /// executable residual-language $`F(p,v)`$ used by the soundness contract.
    fn completion_cost(
        start: Position,
        suffix: &[u8],
        query: &[u8],
        budget: usize,
    ) -> Option<usize> {
        let ctx = TransitionCtx::unit(query.len(), budget, false);
        let mut positions = vec![start];

        for unit in suffix {
            epsilon_closure_raw(&mut positions, &ctx);
            let mut next = Vec::new();
            for position in positions {
                let cv = characteristic_vector(position, *unit, query, budget);
                let mut successors = SmallVec::<[Position; 4]>::new();
                DamerauV::successors(position, &cv, &ctx, &mut successors);
                for successor in successors {
                    if !next.contains(&successor) {
                        next.push(successor);
                    }
                }
            }
            positions = next;
            if positions.is_empty() {
                return None;
            }
        }

        positions
            .iter()
            .filter_map(|position| DamerauV::finish_cost(position, query.len(), ()))
            .filter(|cost| *cost <= budget)
            .min()
    }

    fn byte_sequence(max_len: usize) -> impl Strategy<Value = Vec<u8>> {
        prop::collection::vec(0u8..=3, 0..=max_len)
    }

    proptest! {
        #![proptest_config(ProptestConfig::with_cases(2_000))]

        #[test]
        fn entry_extend_resolve_and_epsilon_laws_are_executable(
            origin in 0usize..=8,
            delta_seed in 1u8..=3,
            budget in 1usize..=3,
            initial_error_seed in 0usize..=2,
        ) {
            let delta = delta_seed.min(budget as u8);
            let delta_usize = usize::from(delta);
            let initial_error = initial_error_seed.min(budget - delta_usize);
            let query_length = origin + delta_usize + 2;
            let ctx = TransitionCtx::unit(query_length, budget, false);
            let normal = Position::new(origin, initial_error);
            let mut cv = vec![false; delta_usize + 1];
            cv[delta_usize] = true;
            let mut entered = SmallVec::<[Position; 4]>::new();
            DamerauV::successors(normal, &cv, &ctx, &mut entered);

            let pending = entered
                .iter()
                .find(|position| {
                    position.kind() == PositionKind::DamerauPending
                        && position.aux() == delta
                })
                .copied()
                .expect("guarded entry emits the selected delta");
            prop_assert_eq!(pending.term_index, origin);
            prop_assert_eq!(pending.num_errors, initial_error + delta_usize);

            let mut epsilon = SmallVec::<[Position; 4]>::new();
            DamerauV::epsilon_successors(pending, &ctx, &mut epsilon);
            prop_assert!(epsilon.is_empty());

            let mut resolving_cv = vec![false; delta_usize + 1];
            resolving_cv[0] = true;
            let mut resolved = SmallVec::<[Position; 4]>::new();
            DamerauV::successors(pending, &resolving_cv, &ctx, &mut resolved);
            let has_resolution = resolved.iter().any(|position| {
                position.kind() == PositionKind::Normal
                    && position.term_index == origin + delta_usize + 1
                    && position.num_errors == initial_error + delta_usize
            });
            prop_assert!(has_resolution);

            if pending.num_errors < budget {
                let mut extended = SmallVec::<[Position; 4]>::new();
                DamerauV::successors(pending, &[false], &ctx, &mut extended);
                let has_extension = extended.iter().any(|position| {
                    position.kind() == PositionKind::DamerauPending
                        && position.term_index == origin
                        && position.aux() == delta
                        && position.num_errors == pending.num_errors + 1
                });
                prop_assert!(has_extension);
            }
        }

        #[test]
        fn subsumption_never_raises_any_generated_suffix_completion(
            query in byte_sequence(7),
            suffix in byte_sequence(6),
            budget in 0usize..=3,
            origin_seed in 0usize..=8,
            error_seed in 0usize..=3,
            gap_seed in 0usize..=3,
            pending in any::<bool>(),
            delta in 1u8..=3,
            backwards in any::<bool>(),
        ) {
            let origin = origin_seed.min(query.len());
            let lhs_error = error_seed.min(budget);
            let gap = gap_seed.min(budget - lhs_error);
            let rhs_error = lhs_error + gap;
            let (lhs, rhs) = if pending {
                (
                    Position::new_damerau_pending(origin, lhs_error, delta),
                    Position::new_damerau_pending(origin, rhs_error, delta),
                )
            } else {
                let rhs_index = if backwards {
                    origin.saturating_sub(gap)
                } else {
                    origin.saturating_add(gap).min(query.len())
                };
                (Position::new(origin, lhs_error), Position::new(rhs_index, rhs_error))
            };
            let ctx = TransitionCtx::unit(query.len(), budget, false);
            prop_assert!(DamerauV::subsumes(&lhs, &rhs, &ctx));

            let lhs_completion = completion_cost(lhs, &suffix, &query, budget);
            let rhs_completion = completion_cost(rhs, &suffix, &query, budget);
            if let Some(rhs_cost) = rhs_completion {
                prop_assert!(
                    lhs_completion.is_some_and(|lhs_cost| lhs_cost <= rhs_cost),
                    "lhs={lhs:?}, rhs={rhs:?}, query={query:?}, suffix={suffix:?}, lhs_cost={lhs_completion:?}, rhs_cost={rhs_cost}"
                );
            }
        }

        #[test]
        fn mixed_or_unequal_pending_continuations_are_incomparable(
            origin in 0usize..=8,
            lhs_error in 0usize..=3,
            rhs_error in 0usize..=3,
            lhs_delta in 1u8..=3,
            rhs_delta in 1u8..=3,
        ) {
            let budget = lhs_error.max(rhs_error);
            let ctx = TransitionCtx::unit(origin + 5, budget, false);
            let normal = Position::new(origin, lhs_error);
            let lhs_pending = Position::new_damerau_pending(origin, lhs_error, lhs_delta);
            let rhs_pending = Position::new_damerau_pending(origin, rhs_error, rhs_delta);

            prop_assert!(!DamerauV::subsumes(&normal, &rhs_pending, &ctx));
            prop_assert!(!DamerauV::subsumes(&rhs_pending, &normal, &ctx));
            if lhs_delta != rhs_delta {
                prop_assert!(!DamerauV::subsumes(&lhs_pending, &rhs_pending, &ctx));
                prop_assert!(!DamerauV::subsumes(&rhs_pending, &lhs_pending, &ctx));
            }
        }
    }
}
