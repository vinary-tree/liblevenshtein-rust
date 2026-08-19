use super::{unit_finish_cost, unit_skip_window};
use crate::cost::{subsumes_with, SubsumptionMode, UnitCost};
use crate::transducer::transition::transition_merge_split_into;
use crate::transducer::variant::{AutomatonVariant, SubsumptionScope, TransitionCtx};
use crate::transducer::Position;
use smallvec::SmallVec;

/// Unit-cost merge-and-split variant.
#[derive(Debug, Clone, Copy)]
pub(crate) struct MergeSplitV;

impl AutomatonVariant for MergeSplitV {
    type Params = ();

    // Merge/split dominance requires equal term indices. Specialness and cost
    // remain checked by `subsumes`, so this narrows only the search domain.
    const SUBSUMPTION_SCOPE: SubsumptionScope = SubsumptionScope::SameTermIndex;

    #[inline(always)]
    fn successors(
        position: Position,
        characteristic_vector: &[bool],
        ctx: &TransitionCtx<()>,
        out: &mut SmallVec<[Position; 4]>,
    ) {
        transition_merge_split_into(
            &position,
            characteristic_vector,
            ctx.query_length,
            ctx.max_distance,
            ctx.prefix_mode,
            out,
        );
    }

    #[inline(always)]
    fn subsumes(lhs: &Position, rhs: &Position, ctx: &TransitionCtx<()>) -> bool {
        subsumes_with::<UnitCost>(
            lhs.term_index,
            lhs.num_errors,
            lhs.is_special(),
            rhs.term_index,
            rhs.num_errors,
            rhs.is_special(),
            SubsumptionMode::MergeSplit,
            ctx.query_length,
            1,
        )
    }

    #[inline(always)]
    fn finish_cost(position: &Position, query_length: usize, (): ()) -> Option<usize> {
        unit_finish_cost(position, query_length)
    }

    #[inline(always)]
    fn skip_window(position: &Position, ctx: &TransitionCtx<()>) -> usize {
        unit_skip_window(position, ctx)
    }
}
