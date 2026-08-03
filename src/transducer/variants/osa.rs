use super::{unit_finish_cost, unit_skip_window};
use crate::cost::{subsumes_with, SubsumptionMode, UnitCost};
use crate::transducer::transition::transition_transposition_into;
use crate::transducer::variant::{AutomatonVariant, TransitionCtx};
use crate::transducer::Position;
use smallvec::SmallVec;

/// Optimal-string-alignment variant with adjacent transposition.
#[derive(Debug, Clone, Copy)]
pub(crate) struct OsaV;

impl AutomatonVariant for OsaV {
    type Params = ();

    #[inline(always)]
    fn successors(
        position: Position,
        characteristic_vector: &[bool],
        ctx: &TransitionCtx<()>,
        out: &mut SmallVec<[Position; 4]>,
    ) {
        transition_transposition_into(
            &position,
            characteristic_vector,
            ctx.query_length,
            ctx.max_distance,
            ctx.prefix_mode,
            out,
        );
    }

    #[inline(always)]
    fn subsumes(lhs: &Position, rhs: &Position, _ctx: &TransitionCtx<()>) -> bool {
        subsumes_with::<UnitCost>(
            lhs.term_index,
            lhs.num_errors,
            lhs.is_special(),
            rhs.term_index,
            rhs.num_errors,
            rhs.is_special(),
            SubsumptionMode::Transposition,
            usize::MAX,
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
