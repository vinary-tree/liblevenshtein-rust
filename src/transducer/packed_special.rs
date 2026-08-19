//! Two-plane packed frontiers for unit-cost automata with continuation states.
//!
//! A positional MergeSplit frontier distinguishes ordinary positions from a
//! split-in-progress continuation. For short queries, both exact-cost planes
//! fit in two machine words. The raw two-plane relation is interned through the
//! shared exact-label DFA, so dictionary traversal still carries one eight-byte
//! state identifier.

use super::packed_dfa::{ExactLabelDfa, ExactLabelDfaRow};
use super::packed_lanes::PackedEditLaneLayout;
use super::transition::TransitionSettings;
use super::{Algorithm, SubstitutionPolicy, SubstitutionPolicyFor};
use libdictenstein::CharUnit;
use std::marker::PhantomData;

#[derive(Clone, Copy, Debug, Hash, PartialEq, Eq)]
pub(crate) struct SpecialFrontier {
    normal: u64,
    splitting: u64,
}

/// Compile-time recurrence for one exact two-plane continuation automaton.
///
/// The marker is statically dispatched: sharing storage and DFA mechanics
/// does not add a vtable or a function-pointer call to dictionary traversal.
pub(crate) trait SpecialKernel {
    const ALGORITHM: Algorithm;

    fn record_query();
    fn record_fallback_policy();
    fn record_fallback_prefix();
    fn record_fallback_width();

    fn step(
        layout: PackedEditLaneLayout,
        source: SpecialFrontier,
        repeated_matches: u64,
    ) -> Option<SpecialFrontier>;
}

#[derive(Debug)]
pub(crate) struct MergeSplitKernel;

impl SpecialKernel for MergeSplitKernel {
    const ALGORITHM: Algorithm = Algorithm::MergeAndSplit;

    #[inline(always)]
    fn record_query() {
        crate::causal_perf::record_packed_merge_split_queries(1);
    }

    #[inline(always)]
    fn record_fallback_policy() {
        crate::causal_perf::record_packed_merge_split_fallback_policy(1);
    }

    #[inline(always)]
    fn record_fallback_prefix() {
        crate::causal_perf::record_packed_merge_split_fallback_prefix(1);
    }

    #[inline(always)]
    fn record_fallback_width() {
        crate::causal_perf::record_packed_merge_split_fallback_width(1);
    }

    #[inline(always)]
    fn step(
        layout: PackedEditLaneLayout,
        source: SpecialFrontier,
        repeated_matches: u64,
    ) -> Option<SpecialFrontier> {
        step_merge_split(layout, source, repeated_matches)
    }
}

#[derive(Debug)]
pub(crate) struct OsaKernel;

impl SpecialKernel for OsaKernel {
    const ALGORITHM: Algorithm = Algorithm::Transposition;

    #[inline(always)]
    fn record_query() {
        crate::causal_perf::record_packed_osa_queries(1);
    }

    #[inline(always)]
    fn record_fallback_policy() {
        crate::causal_perf::record_packed_osa_fallback_policy(1);
    }

    #[inline(always)]
    fn record_fallback_prefix() {
        crate::causal_perf::record_packed_osa_fallback_prefix(1);
    }

    #[inline(always)]
    fn record_fallback_width() {
        crate::causal_perf::record_packed_osa_fallback_width(1);
    }

    #[inline(always)]
    fn step(
        layout: PackedEditLaneLayout,
        source: SpecialFrontier,
        repeated_matches: u64,
    ) -> Option<SpecialFrontier> {
        step_osa(layout, source, repeated_matches)
    }
}

/// Exact two-plane transition machine shared by continuation automata.
pub(crate) struct PackedSpecialMachine<U: CharUnit, K: SpecialKernel> {
    layout: PackedEditLaneLayout,
    dfa: ExactLabelDfa<U, SpecialFrontier>,
    kernel: PhantomData<K>,
}

pub(crate) type PackedMergeSplitMachine<U> = PackedSpecialMachine<U, MergeSplitKernel>;
pub(crate) type PackedOsaMachine<U> = PackedSpecialMachine<U, OsaKernel>;

impl<U: CharUnit, K: SpecialKernel> PackedSpecialMachine<U, K> {
    #[cfg(test)]
    pub(crate) fn eligible<P>(query_length: usize, settings: TransitionSettings) -> bool
    where
        P: SubstitutionPolicy,
    {
        settings.algorithm == K::ALGORITHM
            && !P::MAY_MATCH_DISTINCT_UNITS
            && !settings.prefix_mode
            && PackedEditLaneLayout::eligible(query_length, settings.max_distance)
    }

    pub(crate) fn new<P>(query: &[U], settings: TransitionSettings) -> Option<Self>
    where
        P: SubstitutionPolicy,
    {
        if settings.algorithm != K::ALGORITHM {
            return None;
        }
        if P::MAY_MATCH_DISTINCT_UNITS {
            K::record_fallback_policy();
            return None;
        }
        if settings.prefix_mode {
            K::record_fallback_prefix();
            return None;
        }
        let Some(layout) = PackedEditLaneLayout::new(query.len(), settings.max_distance, false)
        else {
            K::record_fallback_width();
            return None;
        };
        let seed = SpecialFrontier {
            normal: layout.exact_seed(),
            splitting: 0,
        };
        let dfa = ExactLabelDfa::new(query, layout.lane_starts(), seed);
        K::record_query();
        Some(Self {
            layout,
            dfa,
            kernel: PhantomData,
        })
    }

    #[inline(always)]
    pub(crate) fn seed(&self) -> u64 {
        self.dfa.seed()
    }

    #[inline(always)]
    pub(crate) fn step<P>(
        &mut self,
        source: u64,
        _policy: &P,
        label: U,
        query: &[U],
        settings: TransitionSettings,
    ) -> Option<u64>
    where
        P: SubstitutionPolicy + SubstitutionPolicyFor<U>,
    {
        debug_assert_eq!(query.len(), self.layout.query_length());
        debug_assert_eq!(settings.max_distance, self.layout.max_distance());
        debug_assert_eq!(settings.algorithm, K::ALGORITHM);
        debug_assert!(!settings.prefix_mode);
        debug_assert!(!P::MAY_MATCH_DISTINCT_UNITS);
        self.step_prepared(source, label)
    }

    #[inline(always)]
    pub(crate) fn step_prepared(&mut self, source: u64, label: U) -> Option<u64> {
        let layout = self.layout;
        self.dfa.step(source, label, |frontier, repeated_matches| {
            K::step(layout, frontier, repeated_matches)
        })
    }

    #[inline(always)]
    pub(crate) fn prepare_source_row(&self, source: u64) -> Option<ExactLabelDfaRow> {
        Some(self.dfa.prepare_row(source))
    }

    #[cfg(feature = "perf-instrumentation")]
    #[inline(always)]
    pub(crate) fn source_row_label_is_class_zero(&self, label: U) -> bool {
        self.dfa.source_row_label_is_class_zero(label)
    }

    #[inline(always)]
    pub(crate) fn step_prepared_source_row(
        &mut self,
        source: &mut ExactLabelDfaRow,
        label: U,
    ) -> Option<u64> {
        let layout = self.layout;
        self.dfa
            .step_in_row(source, label, |frontier, repeated_matches| {
                K::step(layout, frontier, repeated_matches)
            })
    }

    pub(crate) fn complete_distance(&self, frontier: u64) -> Option<usize> {
        self.layout
            .complete_distance(self.dfa.frontier(frontier).normal)
    }

    pub(crate) fn min_distance(&self, frontier: u64) -> Option<usize> {
        let frontier = self.dfa.frontier(frontier);
        self.layout
            .min_distance(frontier.normal | frontier.splitting)
    }

    pub(crate) fn max_consumed(&self, frontier: u64) -> usize {
        let frontier = self.dfa.frontier(frontier);
        self.layout
            .max_consumed(frontier.normal | frontier.splitting)
    }

    #[cfg(feature = "perf-instrumentation")]
    pub(crate) fn active_len(&self, frontier: u64) -> usize {
        let frontier = self.dfa.frontier(frontier);
        frontier.normal.count_ones() as usize + frontier.splitting.count_ones() as usize
    }

    pub(crate) fn canonical_state(&self, frontier: u64) -> super::State {
        use super::variant::with_variant;
        use super::variant::TransitionCtx;
        use super::{Position, State};

        let frontier = self.dfa.frontier(frontier);
        let ctx = TransitionCtx::unit(
            self.layout.query_length(),
            self.layout.max_distance(),
            false,
        );
        let mut state = State::new();
        with_variant!(super::variant::VariantSpec::from(K::ALGORITHM), |V| {
            self.layout
                .for_each_set_position(frontier.normal, |edit, position| {
                    state.insert_with::<V>(Position::new(position, edit), &ctx);
                });
            self.layout
                .for_each_set_position(frontier.splitting, |edit, position| {
                    let position = match K::ALGORITHM {
                        Algorithm::Transposition => Position::new_osa_transposing(position, edit),
                        Algorithm::MergeAndSplit => Position::new_splitting(position, edit),
                        Algorithm::Standard | Algorithm::DamerauLevenshtein => {
                            unreachable!("packed special kernels require a continuation algorithm")
                        }
                    };
                    state.insert_with::<V>(position, &ctx);
                });
        });
        state
    }
}

#[inline(always)]
fn step_osa(
    layout: PackedEditLaneLayout,
    source: SpecialFrontier,
    repeated_matches: u64,
) -> Option<SpecialFrontier> {
    if layout.max_distance() == 0 {
        let normal = ((source.normal & repeated_matches) << 1) & layout.active_mask();
        return (normal != 0).then_some(SpecialFrontier {
            normal,
            splitting: 0,
        });
    }

    let nonterminal = layout.nonterminal_bits();
    let error_sources = layout.error_source_bits();
    let matches = source.normal & repeated_matches & nonterminal;
    let mismatches = source.normal & !repeated_matches;
    let editable = mismatches & error_sources;

    // Packed normal lanes deliberately retain the exact deletion closure used
    // by the Standard kernel. OSA's positional oracle, however, starts a
    // transposition only from its canonical normal antichain. Gate that one
    // cross-plane operation through the equivalent bitwise antichain so a
    // deletion-redundant representative cannot manufacture a continuation.
    let canonical = canonical_normal_frontier(layout, source.normal);
    let transposition_sources = canonical
        & !repeated_matches
        & (repeated_matches >> 1)
        & error_sources
        & layout.two_unit_source_bits();

    let unclosed_normal = (matches << 1)
        | (editable << layout.lane_width())
        | ((editable & nonterminal) << (layout.lane_width() + 1))
        | ((source.splitting & repeated_matches & layout.two_unit_source_bits()) << 2);
    let normal = layout.close_exact_deletions(unclosed_normal);
    let transposing = (transposition_sources << layout.lane_width()) & layout.active_mask();
    ((normal | transposing) != 0).then_some(SpecialFrontier {
        normal,
        splitting: transposing,
    })
}

/// Remove normal positions dominated by a lower exact-cost lane.
#[inline]
fn canonical_normal_frontier(layout: PackedEditLaneLayout, packed: u64) -> u64 {
    let mut canonical = 0u64;
    for edit in 0..=layout.max_distance() {
        let mut dominated = 0u64;
        for lower_edit in 0..edit {
            let radius = edit - lower_edit;
            let lower = layout.lane(packed, lower_edit);
            let mut expanded = lower;
            for shift in 1..=radius {
                expanded |= lower << shift;
                expanded |= lower >> shift;
            }
            dominated |= expanded & layout.lane_mask();
        }
        let lane = layout.lane(packed, edit) & !dominated;
        canonical |= lane << (edit * layout.lane_width());
    }
    canonical
}

#[inline(always)]
fn step_merge_split(
    layout: PackedEditLaneLayout,
    source: SpecialFrontier,
    repeated_matches: u64,
) -> Option<SpecialFrontier> {
    if layout.max_distance() == 0 {
        let normal = ((source.normal & repeated_matches) << 1) & layout.active_mask();
        return (normal != 0).then_some(SpecialFrontier {
            normal,
            splitting: 0,
        });
    }

    let nonterminal = layout.nonterminal_bits();
    let error_sources = layout.error_source_bits();
    let matches = source.normal & repeated_matches & nonterminal;
    let mismatches = source.normal & !repeated_matches;
    let editable = mismatches & error_sources;

    let unclosed_normal = (matches << 1)
        | (editable << layout.lane_width())
        | ((editable & nonterminal) << (layout.lane_width() + 1))
        | ((editable & layout.two_unit_source_bits()) << (layout.lane_width() + 2))
        | ((source.splitting & nonterminal) << 1);
    let splitting = ((editable & nonterminal) << layout.lane_width()) & layout.active_mask();

    // Deleting from a split-in-progress continuation returns to the normal
    // plane. Further deletions then follow the ordinary exact-cost closure.
    let special_delete_seed =
        (splitting & error_sources & nonterminal) << (layout.lane_width() + 1);
    let normal = layout.close_exact_deletions(unclosed_normal | special_delete_seed);
    ((normal | splitting) != 0).then_some(SpecialFrontier { normal, splitting })
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::transducer::Unrestricted;

    #[test]
    fn eligibility_is_exact_at_two_plane_width_boundaries() {
        for (distance, largest) in [(0, 63), (1, 31), (2, 20), (3, 15)] {
            let settings = TransitionSettings::new(distance, Algorithm::MergeAndSplit, false);
            assert!(PackedMergeSplitMachine::<u8>::eligible::<Unrestricted>(
                largest, settings
            ));
            assert!(!PackedMergeSplitMachine::<u8>::eligible::<Unrestricted>(
                largest + 1,
                settings
            ));

            let settings = TransitionSettings::new(distance, Algorithm::Transposition, false);
            assert!(PackedOsaMachine::<u8>::eligible::<Unrestricted>(
                largest, settings
            ));
            assert!(!PackedOsaMachine::<u8>::eligible::<Unrestricted>(
                largest + 1,
                settings
            ));
        }
    }

    #[test]
    fn split_and_merge_paths_finish_at_one_edit() {
        let settings = TransitionSettings::new(1, Algorithm::MergeAndSplit, false);

        let mut split = PackedMergeSplitMachine::new::<Unrestricted>(b"a", settings).unwrap();
        let first = split
            .step(split.seed(), &Unrestricted, b'x', b"a", settings)
            .unwrap();
        let second = split
            .step(first, &Unrestricted, b'y', b"a", settings)
            .unwrap();
        assert_eq!(split.complete_distance(second), Some(1));

        let mut merge = PackedMergeSplitMachine::new::<Unrestricted>(b"ab", settings).unwrap();
        let merged = merge
            .step(merge.seed(), &Unrestricted, b'x', b"ab", settings)
            .unwrap();
        assert_eq!(merge.complete_distance(merged), Some(1));
    }

    #[test]
    fn osa_swap_completes_including_from_the_top_error_lane() {
        let settings = TransitionSettings::new(1, Algorithm::Transposition, false);
        let mut machine = PackedOsaMachine::new::<Unrestricted>(b"ab", settings).unwrap();
        let first = machine.step_prepared(machine.seed(), b'b').unwrap();
        let complete = machine.step_prepared(first, b'a').unwrap();
        assert_eq!(machine.complete_distance(complete), Some(1));
    }

    #[test]
    fn osa_begin_uses_only_the_canonical_normal_antichain() {
        let settings = TransitionSettings::new(2, Algorithm::Transposition, false);
        let mut machine = PackedOsaMachine::new::<Unrestricted>(b"aab", settings).unwrap();
        let after_b = machine.step_prepared(machine.seed(), b'b').unwrap();
        let state = machine.canonical_state(after_b);
        assert!(
            state.positions().iter().all(|position| !position.is_special()),
            "a deletion-redundant normal representative started a spurious transposition: {state:?}"
        );
    }

    #[test]
    fn osa_repeated_adjacent_match_and_failed_completion_do_not_leave_history() {
        let settings = TransitionSettings::new(1, Algorithm::Transposition, false);

        let mut repeated = PackedOsaMachine::new::<Unrestricted>(b"aa", settings).unwrap();
        let after_a = repeated.step_prepared(repeated.seed(), b'a').unwrap();
        assert!(repeated
            .canonical_state(after_a)
            .positions()
            .iter()
            .all(|position| !position.is_special()));

        let mut failed = PackedOsaMachine::new::<Unrestricted>(b"ab", settings).unwrap();
        let after_b = failed.step_prepared(failed.seed(), b'b').unwrap();
        assert!(failed
            .canonical_state(after_b)
            .positions()
            .iter()
            .any(|position| position.is_special()));
        if let Some(after_x) = failed.step_prepared(after_b, b'x') {
            assert!(failed
                .canonical_state(after_x)
                .positions()
                .iter()
                .all(|position| !position.is_special()));
        }
    }

    #[test]
    fn osa_does_not_begin_at_a_match_beyond_the_adjacent_unit() {
        let settings = TransitionSettings::new(2, Algorithm::Transposition, false);
        let mut machine = PackedOsaMachine::new::<Unrestricted>(b"abc", settings).unwrap();
        let after_c = machine.step_prepared(machine.seed(), b'c').unwrap();
        assert!(machine
            .canonical_state(after_c)
            .positions()
            .iter()
            .all(|position| !position.is_special()));
    }

    #[test]
    fn osa_empty_and_three_edit_separator_boundaries_are_exact() {
        let empty_settings = TransitionSettings::new(1, Algorithm::Transposition, false);
        let mut empty = PackedOsaMachine::new::<Unrestricted>(b"", empty_settings).unwrap();
        assert_eq!(empty.complete_distance(empty.seed()), Some(0));
        let one_insertion = empty.step_prepared(empty.seed(), b'x').unwrap();
        assert_eq!(empty.complete_distance(one_insertion), Some(1));

        let within = |distance| {
            let settings = TransitionSettings::new(distance, Algorithm::Transposition, false);
            let mut machine = PackedOsaMachine::new::<Unrestricted>(b"CA", settings).unwrap();
            let mut frontier = Some(machine.seed());
            for label in b"ABC" {
                frontier = frontier.and_then(|source| machine.step_prepared(source, *label));
            }
            frontier.and_then(|frontier| machine.complete_distance(frontier))
        };
        assert_eq!(within(2), None);
        assert_eq!(within(3), Some(3));
    }
}
