//! Compile-time automaton variants and the compact position-kind tag.
//!
//! [`Algorithm`](super::Algorithm) remains the public runtime selector. The
//! transition layer converts it to a [`VariantSpec`] once per dictionary edge,
//! then monomorphizes the per-position kernel through [`AutomatonVariant`].

use super::variants::{AffineV, DamerauV, MergeSplitV, OsaV, StandardV};
use super::{Algorithm, Position};
use smallvec::SmallVec;

/// Continuation state represented by a [`Position`].
///
/// `Normal`, `OsaTransposing`, and `Splitting` serve the three legacy
/// algorithms. The remaining tags carry the history needed by the true
/// Damerau and affine-gap variants while preserving the compact position seam.
#[repr(u8)]
#[derive(Debug, Default, Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Hash)]
pub enum PositionKind {
    /// Ordinary edit-distance frontier position.
    #[default]
    Normal = 0,
    /// First dictionary symbol of an OSA adjacent transposition was consumed.
    OsaTransposing = 1,
    /// First dictionary symbol of a merge/split continuation was consumed.
    Splitting = 2,
    /// Affine query-gap layer: the previous operation consumed a query unit.
    AffineQueryGap = 3,
    /// Affine dictionary-gap layer: the previous operation consumed a dictionary unit.
    AffineDictGap = 4,
    /// True Damerau macro-transition continuation; `aux` carries its delta.
    DamerauPending = 5,
}

impl PositionKind {
    /// Whether this tag denotes an unfinished multi-edge operation.
    #[inline(always)]
    pub const fn is_special(self) -> bool {
        !matches!(self, Self::Normal)
    }
}

/// Immutable context shared by one monomorphized transition kernel.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub(crate) struct TransitionCtx<P: Copy> {
    pub(crate) query_length: usize,
    pub(crate) max_distance: usize,
    pub(crate) prefix_mode: bool,
    pub(crate) params: P,
}

impl<P: Copy> TransitionCtx<P> {
    #[inline(always)]
    pub(crate) const fn new(
        query_length: usize,
        max_distance: usize,
        prefix_mode: bool,
        params: P,
    ) -> Self {
        Self {
            query_length,
            max_distance,
            prefix_mode,
            params,
        }
    }
}

impl TransitionCtx<()> {
    #[inline(always)]
    pub(crate) const fn unit(query_length: usize, max_distance: usize, prefix_mode: bool) -> Self {
        Self::new(query_length, max_distance, prefix_mode, ())
    }
}

/// Compile-time policy for one automaton family.
///
/// # Soundness contract
///
/// If `subsumes(lhs, rhs, ctx)` returns `true`, dropping `rhs` must not
/// increase the minimum cost reported for any possible dictionary suffix.
pub(crate) trait AutomatonVariant: Copy + 'static {
    type Params: Copy;

    /// Fill `out` with successors of `position`.
    ///
    /// `out` must be empty on entry. Keeping that invariant at the caller lets
    /// LLVM specialize single-successor paths without a redundant length reset.
    fn successors(
        position: Position,
        characteristic_vector: &[bool],
        ctx: &TransitionCtx<Self::Params>,
        out: &mut SmallVec<[Position; 4]>,
    );

    #[inline(always)]
    fn epsilon_successors(
        position: Position,
        ctx: &TransitionCtx<Self::Params>,
        out: &mut SmallVec<[Position; 4]>,
    ) {
        if position.num_errors < ctx.max_distance && position.term_index < ctx.query_length {
            if let (Some(term_index), Some(num_errors)) = (
                position.term_index.checked_add(1),
                position.num_errors.checked_add(1),
            ) {
                out.push(Position::new(term_index, num_errors));
            }
        }
    }

    fn subsumes(lhs: &Position, rhs: &Position, ctx: &TransitionCtx<Self::Params>) -> bool;

    fn finish_cost(position: &Position, query_length: usize, params: Self::Params)
        -> Option<usize>;

    /// Maximum query window required by this position at the current budget.
    fn skip_window(position: &Position, ctx: &TransitionCtx<Self::Params>) -> usize;

    /// Whether the exact-match shortcut is valid at a zero budget.
    #[inline(always)]
    fn supports_zero_distance_fast_path(_ctx: &TransitionCtx<Self::Params>) -> bool {
        true
    }
}

/// Runtime selector consumed once at the dictionary-edge boundary.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub(crate) enum VariantSpec {
    Standard,
    Osa,
    MergeSplit,
    Damerau,
    /// Parameterized affine-gap variant, dispatched through its typed entry point.
    AffineGap,
}

impl From<Algorithm> for VariantSpec {
    #[inline(always)]
    fn from(value: Algorithm) -> Self {
        match value {
            Algorithm::Standard => Self::Standard,
            Algorithm::Transposition => Self::Osa,
            Algorithm::MergeAndSplit => Self::MergeSplit,
            Algorithm::DamerauLevenshtein => Self::Damerau,
        }
    }
}

/// Bind a runtime [`VariantSpec`] to a compile-time variant type exactly once.
macro_rules! with_variant {
    ($spec:expr, |$variant:ident| $body:block) => {{
        match $spec {
            $crate::transducer::variant::VariantSpec::Standard => {
                type $variant = $crate::transducer::variants::StandardV;
                $body
            }
            $crate::transducer::variant::VariantSpec::Osa => {
                type $variant = $crate::transducer::variants::OsaV;
                $body
            }
            $crate::transducer::variant::VariantSpec::MergeSplit => {
                type $variant = $crate::transducer::variants::MergeSplitV;
                $body
            }
            $crate::transducer::variant::VariantSpec::Damerau => {
                type $variant = $crate::transducer::variants::DamerauV;
                $body
            }
            $crate::transducer::variant::VariantSpec::AffineGap => {
                unreachable!("affine-gap dispatch requires AffineGapParams")
            }
        }
    }};
}

pub(crate) use with_variant;

const _: fn() = || {
    fn variants_are_copy<T: Copy>() {}
    variants_are_copy::<StandardV>();
    variants_are_copy::<OsaV>();
    variants_are_copy::<MergeSplitV>();
    variants_are_copy::<DamerauV>();
    variants_are_copy::<AffineV>();
};
