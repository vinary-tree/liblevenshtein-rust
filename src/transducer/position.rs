//! Position in the Levenshtein automaton.

use super::algorithm::Algorithm;
use super::variant::{with_variant, AutomatonVariant, PositionKind, TransitionCtx, VariantSpec};
use std::cmp::Ordering;

/// A position in the Levenshtein automaton state.
///
/// A position represents a location `(term_index, num_errors)` in the
/// automaton, indicating we've consumed `term_index` characters from
/// the query term with `num_errors` accumulated errors.
///
/// The compact `kind` and `aux` fields distinguish unfinished variant states
/// without widening the legacy 64-bit layout.
///
/// # Performance
///
/// Position is `Copy` (two `usize` fields plus two one-byte tags and
/// target-dependent padding) to eliminate allocation overhead when copying
/// positions during state transitions. On supported 64-bit targets it remains
/// exactly 24 bytes.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub struct Position {
    /// Index into the query term (characters consumed)
    pub term_index: usize,

    /// Number of accumulated edit operations
    pub num_errors: usize,

    /// Variant continuation tag. Kept private so constructors preserve the
    /// relationship between the tag and `aux`.
    kind: PositionKind,

    /// Variant-specific payload. Zero for all three legacy variants.
    aux: u8,
}

#[cfg(target_pointer_width = "64")]
const _: [(); 24] = [(); std::mem::size_of::<Position>()];

impl Position {
    /// Create a new position
    #[inline(always)]
    pub fn new(term_index: usize, num_errors: usize) -> Self {
        Self {
            term_index,
            num_errors,
            kind: PositionKind::Normal,
            aux: 0,
        }
    }

    /// Create a new OSA transposition position.
    ///
    /// This historical constructor predates typed continuation kinds. New
    /// variant implementations should use `with_kind` so their continuation
    /// language is explicit.
    #[deprecated(since = "0.9.1", note = "use typed variant constructors")]
    #[inline(always)]
    pub fn new_special(term_index: usize, num_errors: usize) -> Self {
        Self::new_osa_transposing(term_index, num_errors)
    }

    /// Create an OSA transposition-in-progress position.
    #[inline(always)]
    pub const fn new_osa_transposing(term_index: usize, num_errors: usize) -> Self {
        Self::with_kind(term_index, num_errors, PositionKind::OsaTransposing, 0)
    }

    /// Create a merge/split continuation position.
    #[inline(always)]
    pub const fn new_splitting(term_index: usize, num_errors: usize) -> Self {
        Self::with_kind(term_index, num_errors, PositionKind::Splitting, 0)
    }

    /// Create a position inside a gap that consumes query units.
    #[inline(always)]
    pub(crate) const fn new_affine_query_gap(term_index: usize, cost: usize) -> Self {
        Self::with_kind(term_index, cost, PositionKind::AffineQueryGap, 0)
    }

    /// Create a position inside a gap that consumes dictionary units.
    #[inline(always)]
    pub(crate) const fn new_affine_dict_gap(term_index: usize, cost: usize) -> Self {
        Self::with_kind(term_index, cost, PositionKind::AffineDictGap, 0)
    }

    /// Create an unrestricted Damerau–Levenshtein continuation.
    ///
    /// `delta` is the positive distance from the current query index to the
    /// transposition's deferred query endpoint. Zero is invalid because it
    /// would not describe a transposition macro.
    #[inline(always)]
    pub(crate) const fn new_damerau_pending(
        term_index: usize,
        num_errors: usize,
        delta: u8,
    ) -> Self {
        debug_assert!(delta > 0);
        Self::with_kind(term_index, num_errors, PositionKind::DamerauPending, delta)
    }

    #[inline(always)]
    pub(crate) const fn with_kind(
        term_index: usize,
        num_errors: usize,
        kind: PositionKind,
        aux: u8,
    ) -> Self {
        Self {
            term_index,
            num_errors,
            kind,
            aux,
        }
    }

    /// Whether this is an unfinished multi-edge operation.
    #[inline(always)]
    pub const fn is_special(&self) -> bool {
        self.kind.is_special()
    }

    /// Return the typed continuation kind.
    #[inline(always)]
    pub const fn kind(&self) -> PositionKind {
        self.kind
    }

    /// Return the variant-specific one-byte payload.
    #[inline(always)]
    pub const fn aux(&self) -> u8 {
        self.aux
    }

    /// Check if this position subsumes another position.
    ///
    /// Position `p1` subsumes `p2` if all candidates reachable from `p2`
    /// are also reachable from `p1`. This allows pruning redundant states.
    ///
    /// The subsumption logic differs by algorithm:
    ///
    /// # Standard Algorithm
    /// `p1` at (i, e) subsumes `p2` at (j, f) if:
    /// - `|i - j| <= (f - e)`
    ///
    /// # Transposition Algorithm
    /// Complex logic considering special positions (transposition states):
    /// - If both special: must be at same position
    /// - If exactly one position is special, neither can subsume the other
    /// - Otherwise: standard formula
    ///
    /// # MergeAndSplit Algorithm
    /// - Positions with different special-state variants cannot subsume
    /// - Final split-in-progress positions cannot subsume non-final positions
    /// - Otherwise: standard formula with strictly fewer errors
    ///
    /// Based on C++ implementation in subsumes.cpp
    ///
    /// # Parameters
    /// - `query_length`: Length of the query term (n in C++/Java code)
    pub fn subsumes(&self, other: &Position, algorithm: Algorithm, query_length: usize) -> bool {
        let ctx = TransitionCtx::unit(query_length, 0, false);
        with_variant!(VariantSpec::from(algorithm), |V| {
            V::subsumes(self, other, &ctx)
        })
    }

    /// Compare positions for sorting (lexicographic order)
    ///
    /// Order: term index, accumulated cost, continuation kind, then `aux`.
    pub fn compare(&self, other: &Position) -> Ordering {
        self.term_index
            .cmp(&other.term_index)
            .then_with(|| self.num_errors.cmp(&other.num_errors))
            .then_with(|| self.kind.cmp(&other.kind))
            .then_with(|| self.aux.cmp(&other.aux))
    }
}

impl PartialOrd for Position {
    fn partial_cmp(&self, other: &Self) -> Option<Ordering> {
        Some(self.cmp(other))
    }
}

impl Ord for Position {
    fn cmp(&self, other: &Self) -> Ordering {
        self.compare(other)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_position_creation() {
        let pos = Position::new(5, 2);
        assert_eq!(pos.term_index, 5);
        assert_eq!(pos.num_errors, 2);
        assert!(!pos.is_special());
        assert_eq!(pos.kind(), PositionKind::Normal);
        assert_eq!(pos.aux(), 0);
    }

    #[test]
    fn test_position_special() {
        let pos = Position::new_osa_transposing(3, 1);
        assert!(pos.is_special());
        assert_eq!(pos.kind(), PositionKind::OsaTransposing);
    }

    #[test]
    #[allow(deprecated)]
    fn deprecated_special_constructor_maps_to_osa_kind() {
        let pos = Position::new_special(3, 1);
        assert_eq!(pos, Position::new_osa_transposing(3, 1));
    }

    #[test]
    fn test_position_subsumption_standard() {
        // Standard algorithm subsumption: |i - j| <= (f - e)
        let max_distance = 3; // Max distance for tests

        // (5, 2) should subsume (5, 3) - same position, fewer errors
        // |5-5| = 0 <= (3-2) = 1 ✓
        let p1 = Position::new(5, 2);
        let p2 = Position::new(5, 3);
        assert!(
            p1.subsumes(&p2, Algorithm::Standard, max_distance),
            "p1(5,2) should subsume p2(5,3)"
        );

        // (5, 2) SHOULD subsume (4, 3) - |5-4| = 1 <= (3-2) = 1
        let p3 = Position::new(5, 2);
        let p4 = Position::new(4, 3);
        assert!(
            p3.subsumes(&p4, Algorithm::Standard, max_distance),
            "p3(5,2) should subsume p4(4,3) per C++ logic"
        );

        // (3, 2) should subsume (3, 2) - same position and errors
        let p5 = Position::new(3, 2);
        let p6 = Position::new(3, 2);
        assert!(
            p5.subsumes(&p6, Algorithm::Standard, max_distance),
            "p5(3,2) should subsume p6(3,2)"
        );

        // (3, 3) should NOT subsume (5, 2) - |3-5| = 2 > (2-3) = -1 (and e > f)
        let p7 = Position::new(3, 3);
        let p8 = Position::new(5, 2);
        assert!(
            !p7.subsumes(&p8, Algorithm::Standard, max_distance),
            "p7(3,3) should NOT subsume p8(5,2)"
        );
    }

    #[test]
    fn test_position_subsumption_transposition() {
        // Transposition subsumption tests
        let max_distance = 3; // Max distance for tests

        // Both special: must be at same position
        let p1 = Position::new_osa_transposing(5, 2);
        let p2 = Position::new_osa_transposing(5, 3);
        assert!(
            p1.subsumes(&p2, Algorithm::Transposition, max_distance),
            "special(5,2) should subsume special(5,3) - same position"
        );

        let p3 = Position::new_osa_transposing(5, 2);
        let p4 = Position::new_osa_transposing(6, 3);
        assert!(
            !p3.subsumes(&p4, Algorithm::Transposition, max_distance),
            "special(5,2) should NOT subsume special(6,3) - different position"
        );

        // lhs special, rhs not: special should NEVER subsume non-special
        // Defect fix: Special positions (transposition-in-progress) and normal positions
        // represent fundamentally different computational paths that cannot be interchanged.
        let p5 = Position::new_osa_transposing(5, 2);
        let p6 = Position::new(5, 3);
        assert!(
            !p5.subsumes(&p6, Algorithm::Transposition, max_distance),
            "special(5,2) should NOT subsume normal(5,3) - different computational paths"
        );

        // Regression test for defect case: special(0,2) subsuming normal(0,2) caused false negatives
        // Test case: dict=["auou"], query="ou", max_dist=2
        // Defect: After processing 'u', (0,2,special) was subsuming (0,2), eliminating valid paths
        let p5a = Position::new_osa_transposing(0, 2);
        let p5b = Position::new(0, 2);
        assert!(
            !p5a.subsumes(&p5b, Algorithm::Transposition, max_distance),
            "special(0,2) should NOT subsume normal(0,2) - same errors, different paths"
        );

        // lhs normal, rhs special: normal cannot subsume special (transposition-in-progress)
        let p7 = Position::new(5, 2);
        let p8 = Position::new_osa_transposing(4, 3);
        assert!(
            !p7.subsumes(&p8, Algorithm::Transposition, max_distance),
            "normal(5,2) should NOT subsume special(4,3) - special positions are transposition-in-progress"
        );

        let p9 = Position::new(5, 2);
        let p10 = Position::new_osa_transposing(6, 3);
        assert!(
            !p9.subsumes(&p10, Algorithm::Transposition, max_distance),
            "normal(5,2) should NOT subsume special(6,3) - special positions are transposition-in-progress"
        );

        // Neither special: standard formula
        let p11 = Position::new(5, 2);
        let p12 = Position::new(4, 3);
        assert!(
            p11.subsumes(&p12, Algorithm::Transposition, max_distance),
            "normal(5,2) should subsume normal(4,3) - standard formula"
        );
    }

    #[test]
    fn test_transposition_special_and_normal_states_do_not_cross_subsume() {
        let query_length = 4;

        for term_index in 0..=query_length {
            for num_errors in 0..=2 {
                let normal = Position::new(term_index, num_errors);
                let special = Position::new_osa_transposing(term_index, num_errors);

                assert!(
                    !normal.subsumes(&special, Algorithm::Transposition, query_length),
                    "normal position {:?} must not subsume special {:?}",
                    normal,
                    special
                );
                assert!(
                    !special.subsumes(&normal, Algorithm::Transposition, query_length),
                    "special position {:?} must not subsume normal {:?}",
                    special,
                    normal
                );
            }
        }
    }

    #[test]
    fn test_position_subsumption_merge_split() {
        // MergeAndSplit subsumption tests
        let query_length = 5;

        // Different variant states cannot subsume each other.
        let p1 = Position::new_splitting(5, 2);
        let p2 = Position::new(5, 3);
        assert!(
            !p1.subsumes(&p2, Algorithm::MergeAndSplit, query_length),
            "special(5,2) should NOT subsume normal(5,3) for MergeAndSplit"
        );

        let p2a = Position::new(5, 2);
        let p2b = Position::new_splitting(4, 3);
        assert!(
            !p2a.subsumes(&p2b, Algorithm::MergeAndSplit, query_length),
            "normal(5,2) should NOT subsume special(4,3) for MergeAndSplit"
        );

        // Final split-in-progress states cannot prune non-final pending splits.
        let p2c = Position::new_splitting(5, 1);
        let p2d = Position::new_splitting(4, 2);
        assert!(
            !p2c.subsumes(&p2d, Algorithm::MergeAndSplit, query_length),
            "final special(5,1) should NOT subsume non-final special(4,2)"
        );

        // Cross-index pruning is intentionally disabled for MergeAndSplit.
        let p3 = Position::new(5, 2);
        let p4 = Position::new(4, 3);
        assert!(
            !p3.subsumes(&p4, Algorithm::MergeAndSplit, query_length),
            "normal(5,2) should NOT subsume normal(4,3)"
        );

        // Same variant, same index, strictly fewer errors can prune.
        let p5 = Position::new_splitting(5, 2);
        let p6 = Position::new_splitting(5, 3);
        assert!(
            p5.subsumes(&p6, Algorithm::MergeAndSplit, query_length),
            "special(5,2) should subsume special(5,3)"
        );
    }

    #[test]
    fn test_position_ordering() {
        let p1 = Position::new(3, 1);
        let p2 = Position::new(3, 2);
        let p3 = Position::new(4, 1);

        assert!(p1 < p2);
        assert!(p1 < p3);
        assert!(p2 < p3);
    }

    #[test]
    fn ordering_distinguishes_kind_and_aux_payload() {
        let normal = Position::new(3, 1);
        let osa = Position::new_osa_transposing(3, 1);
        let split = Position::new_splitting(3, 1);
        let pending_1 = Position::with_kind(3, 1, PositionKind::DamerauPending, 1);
        let pending_2 = Position::with_kind(3, 1, PositionKind::DamerauPending, 2);

        assert!(normal < osa);
        assert!(osa < split);
        assert!(split < pending_1);
        assert!(pending_1 < pending_2);
    }

    #[test]
    fn legacy_64_bit_layout_is_a_compile_time_contract() {
        #[cfg(target_pointer_width = "64")]
        assert_eq!(std::mem::size_of::<Position>(), 24);
    }
}
