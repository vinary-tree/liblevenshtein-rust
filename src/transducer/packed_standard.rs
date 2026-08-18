//! One-word bounded Levenshtein NFA for short Standard queries.
//!
//! The positional automaton represents a frontier as a collection of
//! `(query position, edit count)` pairs. For the overwhelmingly common small
//! edit budgets, the same reachability relation fits in one `u64`: lane `e`
//! contains query positions reachable at exactly `e` edits. A transition is
//! then a fixed number of shifts, masks, and Boolean operations.
//!
//! If `R[e]` is the epsilon-closed exact-cost source lane and `M` marks query
//! positions equivalent to the dictionary label, the consuming transition is
//!
//! ```text
//! T[0] = ((R[0] & M) << 1) | prefix_sink(R[0])
//! T[e] = ((R[e] & M) << 1) | R[e-1] | (R[e-1] << 1)
//!      | prefix_sink(R[e])
//! ```
//!
//! The terms are match, insertion, substitution, and the zero-cost suffix sink
//! used by prefix/substring dictionary walks. A closed-form epsilon closure
//! then shifts the consumed frontier by `k * (lane_width + 1)` for `k` query
//! deletions. This retains exact edit costs without cumulative lane promotion.

use super::packed_lanes::PackedEditLaneLayout;
use super::transition::TransitionSettings;
use super::{Algorithm, SubstitutionPolicy, SubstitutionPolicyFor};
use libdictenstein::CharUnit;
use rustc_hash::FxHashMap;

const DIRECT_MASK_SLOTS: usize = 256;
const DIRECT_PRESENT_WORDS: usize = DIRECT_MASK_SLOTS / u64::BITS as usize;

#[derive(Debug)]
struct DenseExactMasks<U: CharUnit> {
    direct: [u64; DIRECT_MASK_SLOTS],
    overflow: FxHashMap<U, u64>,
}

impl<U: CharUnit> DenseExactMasks<U> {
    fn empty() -> Self {
        Self {
            direct: [0; DIRECT_MASK_SLOTS],
            overflow: FxHashMap::default(),
        }
    }

    #[inline(always)]
    fn get(&self, unit: U) -> u64 {
        match unit.to_dense_index() {
            Some(slot) => self.direct[usize::from(slot)],
            None => self.overflow.get(&unit).copied().unwrap_or(0),
        }
    }

    fn insert_or(&mut self, unit: U, mask: u64) {
        match unit.to_dense_index() {
            Some(slot) => self.direct[usize::from(slot)] |= mask,
            None => *self.overflow.entry(unit).or_default() |= mask,
        }
    }
}

#[derive(Debug)]
enum ExactMaskResolver<U: CharUnit> {
    Empty,
    Single { unit: U, repeated_mask: u64 },
    Dense(Box<DenseExactMasks<U>>),
}

impl<U: CharUnit> ExactMaskResolver<U> {
    fn new(query: &[U], lane_starts: u64) -> Self {
        let Some(&first) = query.first() else {
            return Self::Empty;
        };

        let mut single_mask = 0u64;
        let mut all_equal = true;
        for (position, &unit) in query.iter().enumerate() {
            single_mask |= lane_starts << position;
            all_equal &= unit == first;
        }
        if all_equal {
            return Self::Single {
                unit: first,
                repeated_mask: single_mask,
            };
        }

        let mut masks = DenseExactMasks::empty();
        for (position, &unit) in query.iter().enumerate() {
            masks.insert_or(unit, lane_starts << position);
        }
        Self::Dense(Box::new(masks))
    }

    #[inline(always)]
    fn get(&self, unit: U) -> u64 {
        match self {
            Self::Empty => 0,
            Self::Single {
                unit: stored,
                repeated_mask,
            } => u64::from(*stored == unit).wrapping_mul(*repeated_mask),
            Self::Dense(masks) => masks.get(unit),
        }
    }
}

/// Query-local equivalence classes for exact matching.
///
/// Class zero denotes every unit absent from the query.  Every distinct query
/// unit receives one non-zero class whose payload is the repeated packed match
/// mask for that unit.  Dense byte-range units use one array load; arbitrary
/// Unicode scalars and `u64` tokens retain the generic hash-map fallback.
#[derive(Debug)]
struct ExactLabelClasses<U: CharUnit> {
    direct: [u8; DIRECT_MASK_SLOTS],
    overflow: FxHashMap<U, u8>,
    repeated_masks: Box<[u64]>,
}

impl<U: CharUnit> ExactLabelClasses<U> {
    fn new(query: &[U], lane_starts: u64) -> Self {
        let mut direct = [0u8; DIRECT_MASK_SLOTS];
        let mut overflow = FxHashMap::default();
        let mut repeated_masks = vec![0u64];

        for (position, &unit) in query.iter().enumerate() {
            let existing = match unit.to_dense_index() {
                Some(slot) => direct[usize::from(slot)],
                None => overflow.get(&unit).copied().unwrap_or(0),
            };
            let class = if existing == 0 {
                let class = u8::try_from(repeated_masks.len())
                    .expect("a packed query has fewer than 256 label classes");
                repeated_masks.push(0);
                match unit.to_dense_index() {
                    Some(slot) => direct[usize::from(slot)] = class,
                    None => {
                        overflow.insert(unit, class);
                    }
                }
                class
            } else {
                existing
            };
            repeated_masks[usize::from(class)] |= lane_starts << position;
        }

        Self {
            direct,
            overflow,
            repeated_masks: repeated_masks.into_boxed_slice(),
        }
    }

    #[inline(always)]
    fn class_for(&self, unit: U) -> usize {
        usize::from(match unit.to_dense_index() {
            Some(slot) => self.direct[usize::from(slot)],
            None => self.overflow.get(&unit).copied().unwrap_or(0),
        })
    }

    #[inline(always)]
    fn class_count(&self) -> usize {
        self.repeated_masks.len()
    }

    #[inline(always)]
    fn repeated_mask(&self, class: usize) -> u64 {
        self.repeated_masks[class]
    }
}

const DFA_TARGET_DEAD: u32 = u32::MAX;
const DFA_TARGET_UNCOMPUTED: u32 = u32::MAX - 1;

/// Lazy deterministic form of the packed bounded NFA.
///
/// Queued frontiers become dense state identifiers.  The first visit to a
/// `(state, label-class)` cell performs the packed recurrence and interns its
/// target frontier; every subsequent visit is two array lookups.  This avoids
/// hashing on the steady-state hit path while generating only rows reached by
/// the concrete dictionary walk.
#[derive(Debug)]
struct PackedStandardDfa<U: CharUnit> {
    classes: ExactLabelClasses<U>,
    frontiers: Vec<u64>,
    frontier_ids: FxHashMap<u64, u32>,
    targets: Vec<u32>,
}

impl<U: CharUnit> PackedStandardDfa<U> {
    fn new(query: &[U], layout: PackedEditLaneLayout) -> Self {
        let classes = ExactLabelClasses::new(query, layout.lane_starts());
        let seed = layout.exact_seed();
        let mut frontier_ids = FxHashMap::default();
        frontier_ids.insert(seed, 0);
        let targets = vec![DFA_TARGET_UNCOMPUTED; classes.class_count()];
        crate::causal_perf::record_packed_dfa_queries(1);
        crate::causal_perf::record_packed_dfa_states_interned(1);
        Self {
            classes,
            frontiers: vec![seed],
            frontier_ids,
            targets,
        }
    }

    #[inline(always)]
    fn seed(&self) -> u64 {
        0
    }

    #[inline(always)]
    fn frontier(&self, state: u64) -> u64 {
        self.frontiers[usize::try_from(state).expect("packed DFA state exceeds usize")]
    }

    #[inline(always)]
    fn step(&mut self, state: u64, label: U, layout: PackedEditLaneLayout) -> Option<u64> {
        let state = usize::try_from(state).expect("packed DFA state exceeds usize");
        let class = self.classes.class_for(label);
        let cell = state
            .checked_mul(self.classes.class_count())
            .and_then(|row| row.checked_add(class))
            .expect("packed DFA transition index overflow");
        let cached = self.targets[cell];
        if cached != DFA_TARGET_UNCOMPUTED {
            crate::causal_perf::record_packed_dfa_transition_hits(1);
            return (cached != DFA_TARGET_DEAD).then_some(u64::from(cached));
        }

        crate::causal_perf::record_packed_dfa_transition_misses(1);
        let source = self.frontiers[state];
        let repeated_matches = self.classes.repeated_mask(class);
        let target = step_word_exact_layout(layout, source, repeated_matches);
        let encoded = match target {
            None => DFA_TARGET_DEAD,
            Some(frontier) => match self.frontier_ids.get(&frontier).copied() {
                Some(id) => id,
                None => {
                    let id = u32::try_from(self.frontiers.len())
                        .expect("packed DFA state identifier exceeds u32");
                    assert!(
                        id < DFA_TARGET_UNCOMPUTED,
                        "packed DFA state identifier space exhausted"
                    );
                    self.frontiers.push(frontier);
                    self.frontier_ids.insert(frontier, id);
                    self.targets.extend(std::iter::repeat_n(
                        DFA_TARGET_UNCOMPUTED,
                        self.classes.class_count(),
                    ));
                    crate::causal_perf::record_packed_dfa_states_interned(1);
                    id
                }
            },
        };
        self.targets[cell] = encoded;
        (encoded != DFA_TARGET_DEAD).then_some(u64::from(encoded))
    }
}

#[derive(Debug)]
struct PolicyMaskCache<U: CharUnit> {
    direct: [u64; DIRECT_MASK_SLOTS],
    direct_present: [u64; DIRECT_PRESENT_WORDS],
    overflow: FxHashMap<U, u64>,
}

impl<U: CharUnit> PolicyMaskCache<U> {
    fn empty() -> Self {
        Self {
            direct: [0; DIRECT_MASK_SLOTS],
            direct_present: [0; DIRECT_PRESENT_WORDS],
            overflow: FxHashMap::default(),
        }
    }

    #[inline(always)]
    fn get(&self, unit: U) -> Option<u64> {
        match unit.to_dense_index() {
            Some(slot) => {
                let slot = usize::from(slot);
                let present = self.direct_present[slot / u64::BITS as usize]
                    & (1u64 << (slot % u64::BITS as usize));
                (present != 0).then_some(self.direct[slot])
            }
            None => self.overflow.get(&unit).copied(),
        }
    }

    fn insert(&mut self, unit: U, mask: u64) {
        match unit.to_dense_index() {
            Some(slot) => {
                let slot = usize::from(slot);
                self.direct[slot] = mask;
                self.direct_present[slot / u64::BITS as usize] |=
                    1u64 << (slot % u64::BITS as usize);
            }
            None => {
                self.overflow.insert(unit, mask);
            }
        }
    }
}

#[derive(Debug)]
enum ActiveMaskResolver<U: CharUnit> {
    Exact(ExactMaskResolver<U>),
    Policy(Option<Box<PolicyMaskCache<U>>>),
}

/// Former collision-tagged mask index retained only for same-binary causal
/// comparison. Production builds contain only the dense active resolver.
#[cfg(feature = "resource-profiling")]
#[derive(Debug)]
struct LegacyQueryMaskIndex<U: CharUnit> {
    direct: [Option<(U, u64)>; DIRECT_MASK_SLOTS],
    overflow: FxHashMap<U, u64>,
}

#[cfg(feature = "resource-profiling")]
impl<U: CharUnit> LegacyQueryMaskIndex<U> {
    fn empty() -> Self {
        Self {
            direct: [None; DIRECT_MASK_SLOTS],
            overflow: FxHashMap::default(),
        }
    }

    fn exact(query: &[U], lane_starts: u64) -> Self {
        let mut index = Self::empty();
        for (position, &unit) in query.iter().enumerate() {
            let bit = lane_starts << position;
            if let Some(mask) = index.get_mut(unit) {
                *mask |= bit;
            } else {
                index.insert(unit, bit);
            }
        }
        index
    }

    #[inline(always)]
    fn slot(unit: U) -> usize {
        unit.to_dat_offset() & (DIRECT_MASK_SLOTS - 1)
    }

    #[inline(always)]
    fn get(&self, unit: U) -> Option<u64> {
        match self.direct[Self::slot(unit)] {
            None => None,
            Some((stored, mask)) if stored == unit => Some(mask),
            Some(_) => self.overflow.get(&unit).copied(),
        }
    }

    fn get_mut(&mut self, unit: U) -> Option<&mut u64> {
        let slot = Self::slot(unit);
        match self.direct[slot] {
            None => None,
            Some((stored, _)) if stored == unit => {
                self.direct[slot].as_mut().map(|entry| &mut entry.1)
            }
            Some(_) => self.overflow.get_mut(&unit),
        }
    }

    fn insert(&mut self, unit: U, mask: u64) {
        let slot = Self::slot(unit);
        match self.direct[slot] {
            None => self.direct[slot] = Some((unit, mask)),
            Some((stored, _)) if stored == unit => self.direct[slot] = Some((unit, mask)),
            Some(_) => {
                self.overflow.insert(unit, mask);
            }
        }
    }
}

#[cfg(feature = "resource-profiling")]
struct LegacyPackedMasks<U: CharUnit> {
    exact: LegacyQueryMaskIndex<U>,
    policy: LegacyQueryMaskIndex<U>,
}

/// Packed Standard-Levenshtein transition machine.
pub(crate) struct PackedStandardMachine<U: CharUnit> {
    layout: PackedEditLaneLayout,
    /// Recurrence-only resolver. Exact-match DFA queries do not construct the
    /// redundant 2 KiB `u64` mask table.
    masks: Option<ActiveMaskResolver<U>>,
    dfa: Option<Box<PackedStandardDfa<U>>>,
    #[cfg(feature = "resource-profiling")]
    legacy_masks: Box<LegacyPackedMasks<U>>,
    #[cfg(feature = "resource-profiling")]
    use_legacy_masks: bool,
    #[cfg(any(feature = "perf-instrumentation", feature = "resource-profiling"))]
    use_word_kernel: bool,
    #[cfg(any(feature = "perf-instrumentation", feature = "resource-profiling"))]
    use_exact_cost_lanes: bool,
}

impl<U: CharUnit> PackedStandardMachine<U> {
    pub(crate) fn eligible(query_length: usize, settings: TransitionSettings) -> bool {
        if settings.algorithm != Algorithm::Standard {
            return false;
        }
        PackedEditLaneLayout::eligible(query_length, settings.max_distance)
    }

    pub(crate) fn new<P>(query: &[U], settings: TransitionSettings) -> Option<Self>
    where
        P: SubstitutionPolicy,
    {
        if !Self::eligible(query.len(), settings) {
            return None;
        }
        let layout =
            PackedEditLaneLayout::new(query.len(), settings.max_distance, settings.prefix_mode)?;
        #[cfg(any(feature = "perf-instrumentation", feature = "resource-profiling"))]
        let use_word_kernel = !word_kernel_disabled();
        #[cfg(any(feature = "perf-instrumentation", feature = "resource-profiling"))]
        let use_exact_cost_lanes = use_word_kernel && exact_cost_lanes_enabled();
        #[cfg(any(feature = "perf-instrumentation", feature = "resource-profiling"))]
        let use_dfa = use_exact_cost_lanes && packed_dfa_enabled();
        #[cfg(not(any(feature = "perf-instrumentation", feature = "resource-profiling")))]
        let use_dfa = true;
        let dfa = (!P::MAY_MATCH_DISTINCT_UNITS && use_dfa)
            .then(|| Box::new(PackedStandardDfa::new(query, layout)));
        let masks = dfa.is_none().then(|| {
            if P::MAY_MATCH_DISTINCT_UNITS {
                ActiveMaskResolver::Policy(None)
            } else {
                ActiveMaskResolver::Exact(ExactMaskResolver::new(query, layout.lane_starts()))
            }
        });
        Some(Self {
            layout,
            masks,
            dfa,
            #[cfg(feature = "resource-profiling")]
            legacy_masks: Box::new(LegacyPackedMasks {
                exact: LegacyQueryMaskIndex::exact(query, layout.lane_starts()),
                policy: LegacyQueryMaskIndex::empty(),
            }),
            #[cfg(feature = "resource-profiling")]
            use_legacy_masks: legacy_mask_index_enabled(),
            #[cfg(any(feature = "perf-instrumentation", feature = "resource-profiling"))]
            use_word_kernel,
            #[cfg(any(feature = "perf-instrumentation", feature = "resource-profiling"))]
            use_exact_cost_lanes,
        })
    }

    pub(crate) fn seed(&self) -> u64 {
        if let Some(dfa) = &self.dfa {
            return dfa.seed();
        }
        #[cfg(any(feature = "perf-instrumentation", feature = "resource-profiling"))]
        if !self.use_exact_cost_lanes {
            return self.layout.cumulative_seed();
        }
        self.layout.exact_seed()
    }

    #[inline(always)]
    pub(crate) fn step<P>(
        &mut self,
        source: u64,
        policy: &P,
        label: U,
        query: &[U],
        settings: TransitionSettings,
    ) -> Option<u64>
    where
        P: SubstitutionPolicy + SubstitutionPolicyFor<U>,
    {
        debug_assert_eq!(query.len(), self.layout.query_length());
        debug_assert_eq!(settings.max_distance, self.layout.max_distance());
        debug_assert_eq!(settings.algorithm, Algorithm::Standard);
        debug_assert_eq!(settings.prefix_mode, self.layout.terminal_bits() != 0);

        if let Some(dfa) = &mut self.dfa {
            return dfa.step(source, label, self.layout);
        }

        #[cfg(feature = "resource-profiling")]
        let matches = if self.use_legacy_masks {
            if P::MAY_MATCH_DISTINCT_UNITS {
                match self.legacy_masks.policy.get(label) {
                    Some(mask) => mask,
                    None => {
                        let mask = matching_query_mask(policy, label, query);
                        let repeated = self.layout.lane_starts() * mask;
                        self.legacy_masks.policy.insert(label, repeated);
                        repeated
                    }
                }
            } else {
                self.legacy_masks.exact.get(label).unwrap_or(0)
            }
        } else if P::MAY_MATCH_DISTINCT_UNITS {
            let Some(ActiveMaskResolver::Policy(cache)) = &mut self.masks else {
                unreachable!("packed mask resolver disagrees with substitution policy")
            };
            let cache = cache.get_or_insert_with(|| Box::new(PolicyMaskCache::empty()));
            match cache.get(label) {
                Some(mask) => mask,
                None => {
                    let mask = matching_query_mask(policy, label, query);
                    let repeated = self.layout.lane_starts() * mask;
                    cache.insert(label, repeated);
                    repeated
                }
            }
        } else {
            let Some(ActiveMaskResolver::Exact(resolver)) = &self.masks else {
                unreachable!("packed mask resolver disagrees with substitution policy")
            };
            resolver.get(label)
        };

        #[cfg(not(feature = "resource-profiling"))]
        let matches = if P::MAY_MATCH_DISTINCT_UNITS {
            let Some(ActiveMaskResolver::Policy(cache)) = &mut self.masks else {
                unreachable!("packed mask resolver disagrees with substitution policy")
            };
            let cache = cache.get_or_insert_with(|| Box::new(PolicyMaskCache::empty()));
            match cache.get(label) {
                Some(mask) => mask,
                None => {
                    let mask = matching_query_mask(policy, label, query);
                    let repeated = self.layout.lane_starts() * mask;
                    cache.insert(label, repeated);
                    repeated
                }
            }
        } else {
            let Some(ActiveMaskResolver::Exact(resolver)) = &self.masks else {
                unreachable!("packed mask resolver disagrees with substitution policy")
            };
            resolver.get(label)
        };

        #[cfg(any(feature = "perf-instrumentation", feature = "resource-profiling"))]
        if !self.use_word_kernel {
            return self.step_lanewise(
                source,
                matches & (self.layout.lane_mask() >> 1),
                settings.prefix_mode,
            );
        }

        #[cfg(any(feature = "perf-instrumentation", feature = "resource-profiling"))]
        if !self.use_exact_cost_lanes {
            return self.step_word_cumulative(source, matches);
        }

        self.step_word_exact(source, matches)
    }

    /// Apply every exact-cost edit lane simultaneously and close deletions in
    /// one fixed expression. `nonterminal_bits` prevents a shifted terminal
    /// bit from crossing a lane boundary.
    #[inline(always)]
    fn step_word_exact(&self, source: u64, repeated_matches: u64) -> Option<u64> {
        step_word_exact_layout(self.layout, source, repeated_matches)
    }

    /// Former cumulative-lane kernel retained as a same-binary causal control
    /// and as the positional word-kernel oracle in tests.
    #[cfg(any(test, feature = "perf-instrumentation", feature = "resource-profiling"))]
    #[inline(always)]
    fn step_word_cumulative(&self, source: u64, repeated_matches: u64) -> Option<u64> {
        if self.layout.max_distance() == 0 {
            let target = (((source & repeated_matches) << 1)
                | (source & self.layout.terminal_bits()))
                & self.layout.lane_mask();
            return (target != 0).then_some(target);
        }

        let initial = (((source & repeated_matches) << 1)
            | (source << self.layout.lane_width())
            | ((source & self.layout.nonterminal_bits()) << (self.layout.lane_width() + 1))
            | (source & self.layout.terminal_bits()))
            & self.layout.active_mask();

        let target = match self.layout.max_distance() {
            1 => self.close_up_one_lane(initial),
            2 => self.close_up_one_lane(self.close_up_one_lane(initial)),
            3 => self.close_up_one_lane(self.close_up_one_lane(self.close_up_one_lane(initial))),
            _ => unreachable!("packed Standard distance exceeds three"),
        };

        ((target & self.layout.top_lane_mask()) != 0).then_some(target)
    }

    #[cfg(any(test, feature = "perf-instrumentation", feature = "resource-profiling"))]
    #[inline(always)]
    fn close_up_one_lane(&self, target: u64) -> u64 {
        (target
            | (target << self.layout.lane_width())
            | ((target & self.layout.nonterminal_bits()) << (self.layout.lane_width() + 1)))
            & self.layout.active_mask()
    }

    #[cfg(any(test, feature = "perf-instrumentation", feature = "resource-profiling"))]
    #[inline(always)]
    fn step_lanewise(&self, source: u64, matches: u64, prefix_mode: bool) -> Option<u64> {
        let mut target = 0u64;
        let source_zero = self.lane(source, 0);
        let mut previous_target =
            ((source_zero & matches) << 1) | self.prefix_sink(source_zero, prefix_mode);
        previous_target &= self.layout.lane_mask();
        target |= previous_target;

        for edit in 1..=self.layout.max_distance() {
            let source_lane = self.lane(source, edit);
            let cheaper_source = self.lane(source, edit - 1);
            let lane = (((source_lane & matches) << 1)
                | cheaper_source
                | (cheaper_source << 1)
                | previous_target
                | (previous_target << 1)
                | self.prefix_sink(source_lane, prefix_mode))
                & self.layout.lane_mask();
            target |= lane << (edit * self.layout.lane_width());
            previous_target = lane;
        }

        (self.lane(target, self.layout.max_distance()) != 0).then_some(target)
    }

    #[cfg(any(test, feature = "perf-instrumentation", feature = "resource-profiling"))]
    #[inline(always)]
    fn lane(&self, packed: u64, edit: usize) -> u64 {
        self.layout.lane(packed, edit)
    }

    #[cfg(any(test, feature = "perf-instrumentation", feature = "resource-profiling"))]
    #[inline(always)]
    fn prefix_sink(&self, lane: u64, prefix_mode: bool) -> u64 {
        if prefix_mode {
            lane & self.layout.end_bit()
        } else {
            0
        }
    }

    pub(crate) fn complete_distance(&self, frontier: u64) -> Option<usize> {
        self.layout.complete_distance(self.frontier_bits(frontier))
    }

    /// Packed prefix finality is language-correct, but the legacy positional
    /// prefix scheduler observes its canonical antichain representation. That
    /// scheduler therefore selects the positional arm at construction.
    pub(crate) fn prefix_distance(&self, frontier: u64) -> Option<usize> {
        self.layout.complete_distance(self.frontier_bits(frontier))
    }

    pub(crate) fn min_distance(&self, frontier: u64) -> Option<usize> {
        self.layout.min_distance(self.frontier_bits(frontier))
    }

    pub(crate) fn max_consumed(&self, frontier: u64) -> usize {
        let frontier = self.frontier_bits(frontier);
        #[cfg(any(feature = "perf-instrumentation", feature = "resource-profiling"))]
        if !self.use_exact_cost_lanes {
            let lane = self.lane(frontier, self.layout.max_distance());
            return if lane == 0 {
                0
            } else {
                (u64::BITS - 1 - lane.leading_zeros()) as usize
            };
        }
        self.layout.max_consumed(frontier)
    }

    #[inline(always)]
    fn frontier_bits(&self, frontier: u64) -> u64 {
        match &self.dfa {
            Some(dfa) => dfa.frontier(frontier),
            None => frontier,
        }
    }

    #[cfg(feature = "perf-instrumentation")]
    pub(crate) fn active_len(&self, frontier: u64) -> usize {
        let frontier = self.frontier_bits(frontier);
        #[cfg(any(feature = "perf-instrumentation", feature = "resource-profiling"))]
        if !self.use_exact_cost_lanes {
            return self.lane(frontier, self.layout.max_distance()).count_ones() as usize;
        }
        self.layout.active_len(frontier)
    }
}

#[inline(always)]
fn step_word_exact_layout(
    layout: PackedEditLaneLayout,
    source: u64,
    repeated_matches: u64,
) -> Option<u64> {
    if layout.max_distance() == 0 {
        let target = (((source & repeated_matches) << 1) | (source & layout.terminal_bits()))
            & layout.lane_mask();
        return (target != 0).then_some(target);
    }
    let initial = ((source & repeated_matches) << 1)
        | (source << layout.lane_width())
        | ((source & layout.nonterminal_bits()) << (layout.lane_width() + 1))
        | (source & layout.terminal_bits());
    let target = layout.close_exact_deletions(initial);
    (target != 0).then_some(target)
}

#[inline]
fn matching_query_mask<U, P>(policy: &P, label: U, query: &[U]) -> u64
where
    U: CharUnit,
    P: SubstitutionPolicy + SubstitutionPolicyFor<U>,
{
    let mut mask = 0u64;
    for (position, &query_unit) in query.iter().enumerate() {
        if label == query_unit || policy.is_allowed_for(label, query_unit) {
            mask |= 1u64 << position;
        }
    }
    mask
}

#[cfg(feature = "resource-profiling")]
fn legacy_mask_index_enabled() -> bool {
    std::env::var_os("LIBLEVENSHTEIN_CAUSAL_USE_LEGACY_PACKED_MASKS").is_some()
}

#[cfg(any(feature = "perf-instrumentation", feature = "resource-profiling"))]
fn word_kernel_disabled() -> bool {
    use std::sync::OnceLock;
    static DISABLED: OnceLock<bool> = OnceLock::new();
    *DISABLED.get_or_init(|| {
        std::env::var_os("LIBLEVENSHTEIN_CAUSAL_DISABLE_PACKED_WORD_KERNEL").is_some()
    })
}

#[cfg(any(feature = "perf-instrumentation", feature = "resource-profiling"))]
fn exact_cost_lanes_enabled() -> bool {
    #[cfg(feature = "resource-profiling")]
    {
        std::env::var_os("LIBLEVENSHTEIN_CAUSAL_DISABLE_EXACT_COST_PACKED_LANES").is_none()
    }
    #[cfg(not(feature = "resource-profiling"))]
    {
        true
    }
}

#[cfg(any(feature = "perf-instrumentation", feature = "resource-profiling"))]
fn packed_dfa_enabled() -> bool {
    #[cfg(feature = "resource-profiling")]
    {
        std::env::var_os("LIBLEVENSHTEIN_CAUSAL_DISABLE_PACKED_DFA").is_none()
    }
    #[cfg(not(feature = "resource-profiling"))]
    {
        true
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::transducer::Unrestricted;

    fn levenshtein(left: &[u8], right: &[u8]) -> usize {
        let mut row: Vec<usize> = (0..=right.len()).collect();
        for (left_index, &left_unit) in left.iter().enumerate() {
            let mut diagonal = row[0];
            row[0] = left_index + 1;
            for (right_index, &right_unit) in right.iter().enumerate() {
                let above = row[right_index + 1];
                row[right_index + 1] = (row[right_index + 1] + 1)
                    .min(row[right_index] + 1)
                    .min(diagonal + usize::from(left_unit != right_unit));
                diagonal = above;
            }
        }
        row[right.len()]
    }

    fn words(alphabet: &[u8], max_len: usize) -> Vec<Vec<u8>> {
        let mut words = vec![Vec::new()];
        let mut level = vec![Vec::new()];
        for _ in 0..max_len {
            let mut next = Vec::new();
            for prefix in &level {
                for &unit in alphabet {
                    let mut word = prefix.clone();
                    word.push(unit);
                    words.push(word.clone());
                    next.push(word);
                }
            }
            level = next;
        }
        words
    }

    #[test]
    fn whole_word_kernel_matches_lanewise_oracle_for_arbitrary_frontiers() {
        let mut random = 0x73d5_4a91_c82e_6f0bu64;
        for max_distance in 0..=3 {
            let max_width = u64::BITS as usize / (max_distance + 1);
            for lane_width in 1..=max_width {
                let query = vec![b'x'; lane_width - 1];
                for prefix_mode in [false, true] {
                    let settings =
                        TransitionSettings::new(max_distance, Algorithm::Standard, prefix_mode);
                    let machine = PackedStandardMachine::new::<Unrestricted>(&query, settings)
                        .expect("boundary-sized packed machine");
                    let match_domain = machine.layout.lane_mask() >> 1;
                    for _ in 0..10_000 {
                        random ^= random << 13;
                        random ^= random >> 7;
                        random ^= random << 17;
                        let source = random;
                        random ^= random << 13;
                        random ^= random >> 7;
                        random ^= random << 17;
                        let matches = random & match_domain;
                        let repeated_matches = machine.layout.lane_starts() * matches;
                        assert_eq!(
                            machine.step_word_cumulative(source, repeated_matches),
                            machine.step_lanewise(source, matches, prefix_mode),
                            "distance={max_distance} width={lane_width} prefix={prefix_mode} source={source:#018x} matches={matches:#018x}",
                        );
                    }
                }
            }
        }
    }

    fn exact_to_cumulative(layout: PackedEditLaneLayout, exact: u64) -> u64 {
        let mut cumulative = 0u64;
        let mut reachable = 0u64;
        for edit in 0..=layout.max_distance() {
            reachable |= layout.lane(exact, edit);
            cumulative |= reachable << (edit * layout.lane_width());
        }
        cumulative
    }

    #[test]
    fn exact_cost_kernel_matches_cumulative_oracle_for_arbitrary_closed_frontiers() {
        let mut random = 0xa62b_d71e_9534_c80fu64;
        for max_distance in 0..=PackedEditLaneLayout::MAX_DISTANCE {
            let max_width = u64::BITS as usize / (max_distance + 1);
            for lane_width in 1..=max_width {
                let query = vec![b'x'; lane_width - 1];
                for prefix_mode in [false, true] {
                    let settings =
                        TransitionSettings::new(max_distance, Algorithm::Standard, prefix_mode);
                    let machine = PackedStandardMachine::new::<Unrestricted>(&query, settings)
                        .expect("boundary-sized packed machine");
                    let match_domain = machine.layout.lane_mask() >> 1;
                    for _ in 0..10_000 {
                        random ^= random << 13;
                        random ^= random >> 7;
                        random ^= random << 17;
                        let exact_source = machine
                            .layout
                            .close_exact_deletions(random & machine.layout.active_mask());
                        let cumulative_source = exact_to_cumulative(machine.layout, exact_source);
                        random ^= random << 13;
                        random ^= random >> 7;
                        random ^= random << 17;
                        let matches = random & match_domain;
                        let repeated_matches = machine.layout.lane_starts() * matches;
                        let exact_target = machine
                            .step_word_exact(exact_source, repeated_matches)
                            .map(|target| exact_to_cumulative(machine.layout, target));
                        let cumulative_target =
                            machine.step_word_cumulative(cumulative_source, repeated_matches);
                        assert_eq!(
                            exact_target, cumulative_target,
                            "distance={max_distance} width={lane_width} prefix={prefix_mode} exact={exact_source:#018x} matches={matches:#018x}",
                        );
                    }
                }
            }
        }
    }

    #[test]
    fn dense_and_sparse_exact_masks_do_not_alias() {
        let chars = ExactMaskResolver::<char>::new(&['\0', '\u{0100}', '\0'], 1);
        assert_eq!(chars.get('\0'), 0b101);
        assert_eq!(chars.get('\u{0100}'), 0b010);
        assert_eq!(chars.get('x'), 0);

        let high = 1u64 << 32;
        let units = ExactMaskResolver::<u64>::new(&[0, high, 0], 1);
        assert_eq!(units.get(0), 0b101);
        assert_eq!(units.get(high), 0b010);
        assert_eq!(units.get(u64::MAX), 0);
    }

    #[test]
    fn policy_cache_distinguishes_cached_zero_from_absence() {
        let mut cache = PolicyMaskCache::<char>::empty();
        assert_eq!(cache.get('x'), None);
        cache.insert('x', 0);
        assert_eq!(cache.get('x'), Some(0));

        cache.insert('\u{0100}', 0x55);
        assert_eq!(cache.get('\u{0100}'), Some(0x55));
        assert_eq!(cache.get('\0'), None);
    }

    #[test]
    fn active_resolver_keeps_packed_machines_compact() {
        assert!(std::mem::size_of::<PackedStandardMachine<u8>>() <= 256);
        assert!(std::mem::size_of::<PackedStandardMachine<char>>() <= 256);
        assert!(std::mem::size_of::<PackedStandardMachine<u64>>() <= 256);
    }

    #[test]
    fn packed_complete_distances_match_dynamic_programming_exhaustively() {
        let corpus = words(b"ab", 6);
        for query in &corpus {
            for max_distance in 0..=3 {
                let settings = TransitionSettings::new(max_distance, Algorithm::Standard, false);
                let Some(mut machine) = PackedStandardMachine::new::<Unrestricted>(query, settings)
                else {
                    continue;
                };
                for term in &corpus {
                    let mut frontier = Some(machine.seed());
                    for &label in term {
                        frontier = frontier.and_then(|source| {
                            machine.step(source, &Unrestricted, label, query, settings)
                        });
                    }
                    let expected = levenshtein(term, query);
                    let actual = frontier
                        .and_then(|state| machine.complete_distance(state))
                        .filter(|distance| *distance <= max_distance);
                    assert_eq!(
                        actual,
                        (expected <= max_distance).then_some(expected),
                        "query={query:?} term={term:?} max_distance={max_distance}",
                    );
                }
            }
        }
    }

    #[test]
    fn bit_sixty_three_is_a_valid_terminal_position() {
        let query = vec![b'x'; 63];
        let settings = TransitionSettings::new(0, Algorithm::Standard, false);
        let mut machine = PackedStandardMachine::new::<Unrestricted>(&query, settings)
            .expect("63-unit distance-zero query fits one word");
        let mut frontier = machine.seed();
        for &label in &query {
            frontier = machine
                .step(frontier, &Unrestricted, label, &query, settings)
                .expect("exact path remains live");
        }
        assert_eq!(machine.complete_distance(frontier), Some(0));
        assert_eq!(machine.max_consumed(frontier), 63);
    }
}
