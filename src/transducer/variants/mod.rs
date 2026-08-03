//! Built-in monomorphized automaton variants.

mod affine;
mod damerau;
mod merge_split;
mod osa;
mod standard;

pub(crate) use damerau::DamerauV;
pub(crate) use merge_split::MergeSplitV;
pub(crate) use osa::OsaV;
pub(crate) use standard::StandardV;

use super::variant::TransitionCtx;
use super::Position;

#[inline(always)]
fn unit_finish_cost(position: &Position, query_length: usize) -> Option<usize> {
    if position.is_special() {
        return None;
    }
    let remaining = query_length.saturating_sub(position.term_index);
    position.num_errors.checked_add(remaining)
}

#[inline(always)]
fn unit_skip_window(position: &Position, ctx: &TransitionCtx<()>) -> usize {
    let _ = position;
    let () = ctx.params;
    super::transition::transition_window_size(ctx.max_distance, ctx.query_length)
}
pub use affine::AffineGapParams;
pub(crate) use affine::AffineV;
