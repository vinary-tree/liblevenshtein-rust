//! State transition logic for Levenshtein automata.

use super::packed_standard::PackedStandardMachine;
use super::variant::{with_variant, AutomatonVariant, TransitionCtx, VariantSpec};
use super::variants::{AffineGapParams, AffineV};
use super::{
    Algorithm, Position, PositionKind, State, StatePool, SubstitutionPolicy, SubstitutionPolicyFor,
};
use libdictenstein::CharUnit;
use rustc_hash::FxHashMap;
use smallvec::SmallVec;
use std::ptr::NonNull;
use std::sync::Arc;

/// Configuration shared by pooled state transitions.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct TransitionSettings {
    /// Maximum edit distance.
    pub max_distance: usize,
    /// Edit algorithm variant.
    pub algorithm: Algorithm,
    /// Whether positions beyond the query length are free matches.
    pub prefix_mode: bool,
}

impl TransitionSettings {
    /// Create transition settings.
    pub const fn new(max_distance: usize, algorithm: Algorithm, prefix_mode: bool) -> Self {
        Self {
            max_distance,
            algorithm,
            prefix_mode,
        }
    }
}

/// Configuration shared by pooled affine-gap transitions.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub(crate) struct AffineTransitionSettings {
    /// Maximum exact scaled cost.
    pub max_cost: usize,
    /// Exact affine-gap parameters.
    pub params: AffineGapParams,
    /// Whether positions beyond the query length are free matches.
    pub prefix_mode: bool,
}

impl AffineTransitionSettings {
    /// Create affine transition settings.
    pub(crate) const fn new(max_cost: usize, params: AffineGapParams, prefix_mode: bool) -> Self {
        Self {
            max_cost,
            params,
            prefix_mode,
        }
    }
}

/// Compute the characteristic vector for a position in the query.
///
/// The characteristic vector indicates which characters in a window
/// of the query term can be consumed without error when matching the
/// dictionary character.
///
/// # Arguments
/// * `policy` - Substitution policy for character matching
/// * `dict_unit` - Character unit from the dictionary edge
/// * `query` - Query term units (bytes or chars)
/// * `window_size` - Size of the window (typically max_distance + 1)
/// * `offset` - Base offset in query
/// # Returns
/// Slice of booleans indicating positions where characters can match without error.
/// This includes both exact matches AND policy-allowed substitutions.
/// Uses stack storage for the common max-distance ≤ 7 case and grows only when
/// callers request a larger edit window.
///
/// # Semantics
///
/// For unrestricted policies, only exact character matches are allowed (traditional Levenshtein).
/// For restricted policies, the policy determines if a substitution should be cost-free.
/// This treats policy-allowed substitutions as "equivalences" rather than edits.
///
/// # Performance
///
#[inline]
fn characteristic_vector<'a, U: CharUnit, P: SubstitutionPolicy + SubstitutionPolicyFor<U>>(
    policy: &P,
    dict_unit: U,
    query: &[U],
    window_size: usize,
    offset: usize,
    output: &'a mut SmallVec<[bool; 8]>,
) -> &'a [bool] {
    output.clear();
    output.reserve(window_size);

    // The characteristic vector shows which query positions can consume dict_unit without error.
    // This includes:
    // 1. Exact character matches (query[i] == dict_unit)
    // 2. Policy-allowed substitutions (policy.is_allowed(...))
    //
    // For Unrestricted policy: is_allowed always returns false (no zero-cost substitutions)
    // For Restricted policy: is_allowed checks the substitution set

    for i in 0..window_size {
        let matched = if let Some(query_unit) = checked_query_index(offset, i)
            .and_then(|query_idx| query.get(query_idx))
            .copied()
        {
            query_unit == dict_unit || is_substitution_allowed(policy, dict_unit, query_unit)
        } else {
            false
        };
        output.push(matched);
    }

    output.as_slice()
}

/// Helper function to check if a substitution is allowed.
///
/// This function uses the `SubstitutionPolicyFor` marker trait to safely
/// check substitutions for both byte-level (u8) and character-level (char) operations.
///
/// # Type Safety
///
/// The `P: SubstitutionPolicyFor<U>` bound ensures compile-time type safety:
/// - `Restricted<'a>` can only be used with `U = u8`
/// - `RestrictedChar<'a>` can only be used with `U = char`
/// - `Unrestricted` works with both `u8` and `char`
///
/// # Zero-Cost Abstraction
///
/// For `Unrestricted` policy, this function compiles to a constant `false` and
/// is optimized away entirely by the compiler through inlining and constant propagation.
#[inline(always)]
fn is_substitution_allowed<U: CharUnit, P: SubstitutionPolicy + SubstitutionPolicyFor<U>>(
    policy: &P,
    dict_unit: U,
    query_unit: U,
) -> bool {
    // Safe! No transmute needed - uses the correct trait method for the unit type
    policy.is_allowed_for(dict_unit, query_unit)
}

#[inline(always)]
fn checked_query_index(offset: usize, window_index: usize) -> Option<usize> {
    offset.checked_add(window_index)
}

#[inline(always)]
fn checked_window_end(start: usize, limit: usize, len: usize) -> Option<usize> {
    let end = start.checked_add(limit).unwrap_or(len).min(len);
    (start <= end).then_some(end)
}

#[inline(always)]
fn checked_successor_or_max(value: usize) -> usize {
    match value.checked_add(1) {
        Some(next) => next,
        None => usize::MAX,
    }
}

#[inline(always)]
pub(crate) fn transition_window_size(max_distance: usize, query_length: usize) -> usize {
    let distance_window = checked_successor_or_max(max_distance);
    let query_window = checked_successor_or_max(query_length).max(1);
    distance_window.min(query_window)
}

/// Query-local cache of complete label/query equivalence vectors.
///
/// Values include a false suffix as wide as any unit-cost transition window.
/// A transition can therefore borrow an offset window directly, including at
/// the end of the query, without rebuilding a `SmallVec` for every state
/// position. The cache is unit- and substitution-policy-generic.
pub(crate) struct CharacteristicCache<U: CharUnit> {
    direct: [Option<(U, u32)>; 256],
    overflow: FxHashMap<U, u32>,
    classes: FxHashMap<Arc<[bool]>, u32>,
    class_patterns: Vec<Arc<[bool]>>,
    query_length: usize,
    padding: usize,
}

#[derive(Clone, Copy, Debug, Hash, PartialEq, Eq)]
pub(crate) struct GeneratedStateId(usize);

/// Opaque eight-byte frontier carried by unit-cost dictionary traversals.
///
/// Packed Standard queries store their complete reachability relation directly
/// in this word. Positional fallbacks store a query-local generated-state ID.
/// Keeping the representation opaque lets every traversal scheduler share one
/// transition façade without growing its queue entries.
#[repr(transparent)]
#[derive(Clone, Copy, Debug, Hash, PartialEq, Eq)]
pub(crate) struct UnitCostFrontier(u64);

/// Final-distance interpretation for a unit-cost frontier.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub(crate) enum FinishMode {
    /// Match the entire dictionary term against the entire query.
    Complete,
    /// A suffix-based dictionary may accept any live query position.
    Substring,
    /// The entire query must have been consumed, while the dictionary term may
    /// continue through the zero-cost terminal sink.
    Prefix,
}

/// Query-local unit-cost transition façade.
///
/// Eligible short Standard queries use a one-word bounded NFA. Every other
/// algorithm, budget, or query width keeps the established positional engine
/// as an exact fallback. The enum is selected once per query; schedulers never
/// branch on or duplicate the two representations themselves.
// Keep the positional engine inline. Boxing it adds an allocation and pointer
// indirection to every query solely to reduce this query-local stack value.
#[allow(clippy::large_enum_variant)]
pub(crate) enum UnitCostMachine<U: CharUnit> {
    PackedStandard(PackedStandardMachine<U>),
    Positional(CachedUnitTransitions<U>),
}

impl<U: CharUnit> UnitCostMachine<U> {
    pub(crate) fn unseeded_positional(query_length: usize, max_distance: usize) -> Self {
        Self::Positional(CachedUnitTransitions::new(query_length, max_distance))
    }

    pub(crate) fn seeded<P>(query: &[U], settings: TransitionSettings) -> (Self, UnitCostFrontier)
    where
        P: SubstitutionPolicy,
    {
        if !packed_standard_disabled() {
            if let Some(machine) = PackedStandardMachine::new::<P>(query, settings) {
                let seed = UnitCostFrontier(machine.seed());
                crate::causal_perf::record_packed_standard_queries(1);
                return (Self::PackedStandard(machine), seed);
            }
        }
        Self::seeded_positional(query.len(), settings)
    }

    pub(crate) fn seeded_positional(
        query_length: usize,
        settings: TransitionSettings,
    ) -> (Self, UnitCostFrontier) {
        let initial = initial_state(query_length, settings.max_distance, settings.algorithm);
        let mut transitions = CachedUnitTransitions::new(query_length, settings.max_distance);
        let root = transitions.seed_generated_state(&initial, settings);
        crate::causal_perf::record_positional_unit_queries(1);
        (
            Self::Positional(transitions),
            UnitCostFrontier(
                u64::try_from(root.0).expect("generated state identifier exceeds u64"),
            ),
        )
    }

    /// Apply one unit-cost transition. The packed arm remains small enough to
    /// inline into dictionary traversal; the substantially larger positional
    /// engine is outlined behind a cold-for-this-query call boundary.
    #[inline(always)]
    pub(crate) fn step<P>(
        &mut self,
        source: UnitCostFrontier,
        pool: &mut StatePool,
        policy: &P,
        label: U,
        query: &[U],
        settings: TransitionSettings,
    ) -> Option<UnitCostFrontier>
    where
        P: SubstitutionPolicy + SubstitutionPolicyFor<U>,
    {
        #[cfg(feature = "resource-profiling")]
        if monolithic_unit_step_enabled() {
            return self.step_monolithic(source, pool, policy, label, query, settings);
        }

        match self {
            Self::PackedStandard(machine) => {
                crate::causal_perf::record_transition_attempts(1);
                crate::causal_perf::record_packed_standard_transition_attempts(1);
                let target = machine
                    .step(source.0, policy, label, query, settings)
                    .map(UnitCostFrontier);
                if target.is_none() {
                    crate::causal_perf::record_packed_standard_transition_dead(1);
                }
                target
            }
            Self::Positional(transitions) => {
                Self::step_positional(transitions, source, pool, policy, label, query, settings)
            }
        }
    }

    #[inline(never)]
    fn step_positional<P>(
        transitions: &mut CachedUnitTransitions<U>,
        source: UnitCostFrontier,
        pool: &mut StatePool,
        policy: &P,
        label: U,
        query: &[U],
        settings: TransitionSettings,
    ) -> Option<UnitCostFrontier>
    where
        P: SubstitutionPolicy + SubstitutionPolicyFor<U>,
    {
        transitions
            .transition_generated(
                GeneratedStateId(
                    usize::try_from(source.0).expect("generated state identifier exceeds usize"),
                ),
                pool,
                policy,
                label,
                query,
                settings,
            )
            .map(|id| {
                UnitCostFrontier(
                    u64::try_from(id.0).expect("generated state identifier exceeds u64"),
                )
            })
    }

    /// Same-binary reference for the former mixed hot/cold façade. This arm is
    /// compiled only for causal resource profiling and intentionally retains
    /// the positional body beside the packed arm so its code shape matches the
    /// pre-outline implementation.
    #[cfg(feature = "resource-profiling")]
    #[inline(never)]
    fn step_monolithic<P>(
        &mut self,
        source: UnitCostFrontier,
        pool: &mut StatePool,
        policy: &P,
        label: U,
        query: &[U],
        settings: TransitionSettings,
    ) -> Option<UnitCostFrontier>
    where
        P: SubstitutionPolicy + SubstitutionPolicyFor<U>,
    {
        match self {
            Self::PackedStandard(machine) => {
                crate::causal_perf::record_transition_attempts(1);
                crate::causal_perf::record_packed_standard_transition_attempts(1);
                let target = machine
                    .step(source.0, policy, label, query, settings)
                    .map(UnitCostFrontier);
                if target.is_none() {
                    crate::causal_perf::record_packed_standard_transition_dead(1);
                }
                target
            }
            Self::Positional(transitions) => transitions
                .transition_generated(
                    GeneratedStateId(
                        usize::try_from(source.0)
                            .expect("generated state identifier exceeds usize"),
                    ),
                    pool,
                    policy,
                    label,
                    query,
                    settings,
                )
                .map(|id| {
                    UnitCostFrontier(
                        u64::try_from(id.0).expect("generated state identifier exceeds u64"),
                    )
                }),
        }
    }

    pub(crate) fn finish_distance(
        &self,
        frontier: UnitCostFrontier,
        mode: FinishMode,
        query_length: usize,
    ) -> Option<usize> {
        match self {
            Self::PackedStandard(machine) => match mode {
                FinishMode::Complete => machine.complete_distance(frontier.0),
                FinishMode::Substring => machine.min_distance(frontier.0),
                FinishMode::Prefix => machine.prefix_distance(frontier.0),
            },
            Self::Positional(transitions) => {
                let state = transitions.generated_frontier_state(GeneratedStateId(
                    usize::try_from(frontier.0).expect("generated state identifier exceeds usize"),
                ));
                match mode {
                    FinishMode::Complete => state.infer_distance(query_length),
                    FinishMode::Substring => state.min_distance(),
                    FinishMode::Prefix => state.infer_prefix_distance(query_length),
                }
            }
        }
    }

    pub(crate) fn min_distance(&self, frontier: UnitCostFrontier) -> Option<usize> {
        match self {
            Self::PackedStandard(machine) => machine.min_distance(frontier.0),
            Self::Positional(transitions) => transitions
                .generated_frontier_state(GeneratedStateId(
                    usize::try_from(frontier.0).expect("generated state identifier exceeds usize"),
                ))
                .min_distance(),
        }
    }

    pub(crate) fn max_consumed(&self, frontier: UnitCostFrontier) -> usize {
        match self {
            Self::PackedStandard(machine) => machine.max_consumed(frontier.0),
            Self::Positional(transitions) => transitions
                .generated_frontier_state(GeneratedStateId(
                    usize::try_from(frontier.0).expect("generated state identifier exceeds usize"),
                ))
                .positions()
                .iter()
                .map(|position| position.term_index)
                .max()
                .unwrap_or(0),
        }
    }

    #[cfg(feature = "perf-instrumentation")]
    pub(crate) fn active_len(&self, frontier: UnitCostFrontier) -> usize {
        match self {
            Self::PackedStandard(machine) => machine.active_len(frontier.0),
            Self::Positional(transitions) => {
                transitions.generated_position_count(GeneratedStateId(
                    usize::try_from(frontier.0).expect("generated state identifier exceeds usize"),
                ))
            }
        }
    }

    #[cfg(feature = "perf-instrumentation")]
    pub(crate) fn frontier_storage_bytes(&self, frontier: UnitCostFrontier) -> usize {
        match self {
            Self::PackedStandard(_) => std::mem::size_of::<UnitCostFrontier>(),
            Self::Positional(_) => self
                .active_len(frontier)
                .saturating_mul(std::mem::size_of::<Position>()),
        }
    }

    pub(crate) fn transition_affine<P>(
        &mut self,
        state: &State,
        pool: &mut StatePool,
        policy: &P,
        label: U,
        query: &[U],
        settings: AffineTransitionSettings,
    ) -> Option<State>
    where
        P: SubstitutionPolicy + SubstitutionPolicyFor<U>,
    {
        match self {
            Self::Positional(transitions) => {
                transitions.transition_affine(state, pool, policy, label, query, settings)
            }
            Self::PackedStandard(_) => {
                unreachable!("affine queries always select the positional transition machine")
            }
        }
    }

    #[cfg(test)]
    pub(crate) fn seeded_packed_for_test<P>(
        query: &[U],
        settings: TransitionSettings,
    ) -> Option<(Self, UnitCostFrontier)>
    where
        P: SubstitutionPolicy,
    {
        PackedStandardMachine::new::<P>(query, settings).map(|machine| {
            let root = UnitCostFrontier(machine.seed());
            (Self::PackedStandard(machine), root)
        })
    }
}

#[inline]
fn packed_standard_disabled() -> bool {
    #[cfg(any(feature = "perf-instrumentation", feature = "resource-profiling"))]
    {
        use std::sync::OnceLock;
        static DISABLED: OnceLock<bool> = OnceLock::new();
        *DISABLED.get_or_init(|| {
            std::env::var_os("LIBLEVENSHTEIN_CAUSAL_DISABLE_PACKED_STANDARD").is_some()
        })
    }
    #[cfg(not(any(feature = "perf-instrumentation", feature = "resource-profiling")))]
    {
        false
    }
}

#[cfg(feature = "resource-profiling")]
#[inline(always)]
fn monolithic_unit_step_enabled() -> bool {
    use std::sync::OnceLock;
    static ENABLED: OnceLock<bool> = OnceLock::new();
    *ENABLED.get_or_init(|| {
        std::env::var_os("LIBLEVENSHTEIN_CAUSAL_USE_MONOLITHIC_UNIT_STEP").is_some()
    })
}

#[cfg(feature = "resource-profiling")]
fn jagged_generated_targets_enabled() -> bool {
    std::env::var_os("LIBLEVENSHTEIN_CAUSAL_USE_JAGGED_GENERATED_TARGETS").is_some()
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
struct GeneratedTarget(usize);

impl GeneratedTarget {
    const UNCOMPUTED: Self = Self(usize::MAX);
    const EMPTY: Self = Self(usize::MAX - 1);

    #[inline(always)]
    const fn state(id: GeneratedStateId) -> Self {
        Self(id.0)
    }

    #[inline(always)]
    const fn state_id(self) -> GeneratedStateId {
        debug_assert!(self.0 < Self::EMPTY.0);
        GeneratedStateId(self.0)
    }
}

impl GeneratedStateId {
    #[inline]
    fn next(sources: &[Box<[Position]>]) -> Self {
        let id = sources.len();
        assert!(
            id < GeneratedTarget::EMPTY.0,
            "query-local generated state identifiers exhausted"
        );
        Self(id)
    }
}

/// Contiguous row-major transition targets for generated positional states.
///
/// Canonical position slices remain separately boxed because queued `State`
/// values borrow their stable addresses. Targets have no such stability
/// requirement, so keeping them in one allocation removes the second pointer
/// chase from the overwhelmingly common cache-hit path. The row width is a
/// fixed power of two derived from query length. Exact matching never exceeds
/// that bound; custom substitution policies may create more classes, whose
/// genuinely sparse cells use the overflow map without widening every row.
struct DenseGeneratedTargets {
    values: Vec<GeneratedTarget>,
    row_shift: u32,
    overflow: FxHashMap<(GeneratedStateId, usize), GeneratedTarget>,
}

impl DenseGeneratedTargets {
    fn new(initial_class_capacity: usize) -> Self {
        let capacity = initial_class_capacity
            .max(1)
            .checked_next_power_of_two()
            .expect("query-local characteristic class capacity exceeds usize");
        Self {
            values: Vec::new(),
            row_shift: capacity.trailing_zeros(),
            overflow: FxHashMap::default(),
        }
    }

    #[inline(always)]
    fn stride(&self) -> usize {
        1usize << self.row_shift
    }

    #[inline(always)]
    fn index(&self, state: GeneratedStateId, class: usize) -> usize {
        debug_assert!(class < self.stride());
        (state.0 << self.row_shift) | class
    }

    fn push_row(&mut self) {
        self.values.extend(std::iter::repeat_n(
            GeneratedTarget::UNCOMPUTED,
            self.stride(),
        ));
    }

    #[inline(always)]
    fn get(&self, state: GeneratedStateId, class: usize) -> GeneratedTarget {
        if class >= self.stride() {
            return self
                .overflow
                .get(&(state, class))
                .copied()
                .unwrap_or(GeneratedTarget::UNCOMPUTED);
        }
        self.values[self.index(state, class)]
    }

    fn set(&mut self, state: GeneratedStateId, class: usize, target: GeneratedTarget) {
        if class >= self.stride() {
            self.overflow.insert((state, class), target);
            return;
        }
        let index = self.index(state, class);
        self.values[index] = target;
    }

    fn clear(&mut self) {
        self.values.clear();
        self.overflow.clear();
    }
}

/// Query-local unit-transition engine shared by every iterator surface.
///
/// It owns the lazy characteristic cache and centralizes the epsilon-closed
/// queued-state invariant. Concrete unit, substitution-policy, and automaton
/// variant types are still monomorphized; this abstraction introduces no
/// trait objects or indirect calls in the transition loop.
pub(crate) struct CachedUnitTransitions<U: CharUnit> {
    cache: CharacteristicCache<U>,
    generated_config: Option<(Algorithm, bool)>,
    generated_sources: Vec<Box<[Position]>>,
    dense_targets: DenseGeneratedTargets,
    #[cfg(feature = "resource-profiling")]
    jagged_targets: Vec<Vec<GeneratedTarget>>,
    #[cfg(feature = "resource-profiling")]
    use_jagged_targets: bool,
    generated_states: FxHashMap<u64, SmallVec<[GeneratedStateId; 1]>>,
    max_distance: usize,
}

impl<U: CharUnit> CachedUnitTransitions<U> {
    pub(crate) fn new(query_length: usize, max_distance: usize) -> Self {
        Self {
            cache: CharacteristicCache::new(query_length, max_distance),
            generated_config: None,
            generated_sources: Vec::new(),
            dense_targets: DenseGeneratedTargets::new(query_length.saturating_add(1)),
            #[cfg(feature = "resource-profiling")]
            jagged_targets: Vec::new(),
            #[cfg(feature = "resource-profiling")]
            use_jagged_targets: jagged_generated_targets_enabled(),
            generated_states: FxHashMap::default(),
            max_distance,
        }
    }

    fn push_generated_source(&mut self, source: Box<[Position]>) -> GeneratedStateId {
        let id = GeneratedStateId::next(&self.generated_sources);
        self.generated_sources.push(source);
        #[cfg(feature = "resource-profiling")]
        if self.use_jagged_targets {
            self.jagged_targets.push(Vec::new());
        } else {
            self.dense_targets.push_row();
        }
        #[cfg(not(feature = "resource-profiling"))]
        self.dense_targets.push_row();
        id
    }

    #[inline(always)]
    fn cached_generated_target(&self, source: GeneratedStateId, class: usize) -> GeneratedTarget {
        #[cfg(feature = "resource-profiling")]
        if self.use_jagged_targets {
            return self.jagged_targets[source.0]
                .get(class)
                .copied()
                .unwrap_or(GeneratedTarget::UNCOMPUTED);
        }
        self.dense_targets.get(source, class)
    }

    #[inline]
    fn store_generated_target(
        &mut self,
        source: GeneratedStateId,
        class: usize,
        target: GeneratedTarget,
    ) {
        #[cfg(feature = "resource-profiling")]
        if self.use_jagged_targets {
            let row = &mut self.jagged_targets[source.0];
            if row.len() <= class {
                row.resize(class + 1, GeneratedTarget::UNCOMPUTED);
            }
            row[class] = target;
            return;
        }
        self.dense_targets.set(source, class, target);
    }

    fn clear_generated_table(&mut self) {
        self.generated_sources.clear();
        self.dense_targets.clear();
        #[cfg(feature = "resource-profiling")]
        self.jagged_targets.clear();
    }

    /// Intern the root of a unit-cost query and return its compact frontier ID.
    ///
    /// IDs are local to this transition engine and keep the traversal queue
    /// independent of the canonical position allocation's representation.
    pub(crate) fn seed_generated_state(
        &mut self,
        state: &State,
        settings: TransitionSettings,
    ) -> GeneratedStateId {
        debug_assert_eq!(settings.max_distance, self.max_distance);
        let config = (settings.algorithm, settings.prefix_mode);
        match self.generated_config {
            Some(existing) => assert_eq!(existing, config, "generated transition mode changed"),
            None => self.generated_config = Some(config),
        }

        if self.generated_sources.is_empty() {
            let id = self.push_generated_source(state.positions().into());
            debug_assert_eq!(id, GeneratedStateId(0));
            self.generated_states
                .entry(state.transition_fingerprint())
                .or_default()
                .push(id);
            id
        } else {
            self.intern_positions(state.positions(), state.transition_fingerprint())
        }
    }

    /// Transition a compact query-local frontier without materializing a
    /// `State` in the dictionary traversal queue.
    #[inline]
    pub(crate) fn transition_generated<P>(
        &mut self,
        source: GeneratedStateId,
        pool: &mut StatePool,
        policy: &P,
        dict_unit: U,
        query: &[U],
        settings: TransitionSettings,
    ) -> Option<GeneratedStateId>
    where
        P: SubstitutionPolicy + SubstitutionPolicyFor<U>,
    {
        debug_assert_eq!(settings.max_distance, self.max_distance);
        let cacheable = self.generated_config == Some((settings.algorithm, settings.prefix_mode));
        let characteristic_class = self.cache.class_for(policy, dict_unit, query);

        if cacheable {
            let target = self.cached_generated_target(source, characteristic_class as usize);
            if target != GeneratedTarget::UNCOMPUTED {
                crate::causal_perf::record_transition_attempts(1);
                crate::causal_perf::record_generated_transition_hits(1);
                return (target != GeneratedTarget::EMPTY).then(|| target.state_id());
            }
        }

        crate::causal_perf::record_generated_transition_misses(1);
        let matches = self.cache.matches(characteristic_class);
        let source_state = State::from_canonical_positions(
            self.generated_positions(source),
            Self::nonzero_state_tag(source),
        );
        let generated = transition_epsilon_closed_state_pooled_cached(
            &source_state,
            pool,
            matches,
            query.len(),
            settings,
        );
        let target = match generated {
            Some(state) => {
                let id = self.intern_positions(state.positions(), state.transition_fingerprint());
                pool.release(state);
                GeneratedTarget::state(id)
            }
            None => GeneratedTarget::EMPTY,
        };
        let result = (target != GeneratedTarget::EMPTY).then(|| target.state_id());
        if cacheable {
            self.store_generated_target(source, characteristic_class as usize, target);
        }
        result
    }

    #[inline(always)]
    pub(crate) fn generated_state(&self, id: GeneratedStateId) -> State {
        State::from_canonical_positions(
            self.generated_positions(id),
            u64::try_from(id.0).expect("generated state identifier exceeds u64"),
        )
    }

    #[inline(always)]
    pub(crate) fn generated_frontier_state(&self, id: GeneratedStateId) -> State {
        State::from_canonical_positions(self.generated_positions(id), Self::nonzero_state_tag(id))
    }

    #[inline(always)]
    #[cfg(feature = "perf-instrumentation")]
    pub(crate) fn generated_position_count(&self, id: GeneratedStateId) -> usize {
        self.generated_sources[id.0].len()
    }

    #[inline(always)]
    fn nonzero_state_tag(id: GeneratedStateId) -> u64 {
        u64::try_from(id.0)
            .expect("generated state identifier exceeds u64")
            .checked_add(1)
            .expect("generated state identifier exhausts nonzero tags")
    }

    #[inline]
    pub(crate) fn transition<P>(
        &mut self,
        state: &State,
        pool: &mut StatePool,
        policy: &P,
        dict_unit: U,
        query: &[U],
        settings: TransitionSettings,
    ) -> Option<State>
    where
        P: SubstitutionPolicy + SubstitutionPolicyFor<U>,
    {
        debug_assert_eq!(settings.max_distance, self.max_distance);
        let characteristic_class = self.cache.class_for(policy, dict_unit, query);
        let config = (settings.algorithm, settings.prefix_mode);
        if self.generated_config.is_none() {
            self.generated_config = Some(config);
            let id = self.push_generated_source(state.positions().into());
            debug_assert_eq!(id, GeneratedStateId(0));
        }
        let row_index = usize::try_from(state.generated_id()).ok();
        let cacheable = self.generated_config == Some(config)
            && row_index
                .and_then(|index| self.generated_sources.get(index))
                .is_some_and(|source| {
                    if state.generated_id() == 0 {
                        source.as_ref() == state.positions()
                    } else {
                        std::ptr::eq(source.as_ref(), state.positions())
                    }
                });

        if let Some(index) = row_index.filter(|_| cacheable) {
            let target = self
                .cached_generated_target(GeneratedStateId(index), characteristic_class as usize);
            if target != GeneratedTarget::UNCOMPUTED {
                crate::causal_perf::record_transition_attempts(1);
                crate::causal_perf::record_generated_transition_hits(1);
                return (target != GeneratedTarget::EMPTY)
                    .then(|| self.generated_state(target.state_id()));
            }
        };

        crate::causal_perf::record_generated_transition_misses(1);
        let matches = self.cache.matches(characteristic_class);
        let mut generated = transition_epsilon_closed_state_pooled_cached(
            state,
            pool,
            matches,
            query.len(),
            settings,
        );
        if cacheable {
            let target = match generated.as_mut() {
                Some(target) => {
                    let (id, positions) = self.intern_generated_state(target);
                    let raw_id =
                        u64::try_from(id.0).expect("query-local generated state count exceeds u64");
                    if let Some(reusable) = target.adopt_canonical_positions(positions, raw_id) {
                        pool.release(reusable);
                    }
                    GeneratedTarget::state(id)
                }
                None => GeneratedTarget::EMPTY,
            };
            self.store_generated_target(
                GeneratedStateId(row_index.expect("cacheable row")),
                characteristic_class as usize,
                target,
            );
        }
        generated
    }

    fn generated_positions(&self, id: GeneratedStateId) -> NonNull<[Position]> {
        NonNull::from(self.generated_sources[id.0].as_ref())
    }

    fn intern_generated_state(
        &mut self,
        state: &mut State,
    ) -> (GeneratedStateId, NonNull<[Position]>) {
        let fingerprint = state.transition_fingerprint();
        if let Some(states) = self.generated_states.get(&fingerprint) {
            if let Some(&id) = states
                .iter()
                .find(|&&id| self.generated_sources[id.0].as_ref() == state.positions())
            {
                return (id, self.generated_positions(id));
            }
        }

        let id = self.push_generated_source(state.positions().into());
        self.generated_states
            .entry(fingerprint)
            .or_default()
            .push(id);
        (id, self.generated_positions(id))
    }

    fn intern_positions(&mut self, positions: &[Position], fingerprint: u64) -> GeneratedStateId {
        if let Some(states) = self.generated_states.get(&fingerprint) {
            if let Some(&id) = states
                .iter()
                .find(|&&id| self.generated_sources[id.0].as_ref() == positions)
            {
                return id;
            }
        }

        let id = self.push_generated_source(positions.into());
        self.generated_states
            .entry(fingerprint)
            .or_default()
            .push(id);
        id
    }

    /// Transition an affine-gap state through the same query-local
    /// characteristic cache and epsilon-closed queue invariant.
    #[inline]
    pub(crate) fn transition_affine<P>(
        &mut self,
        state: &State,
        pool: &mut StatePool,
        policy: &P,
        dict_unit: U,
        query: &[U],
        settings: AffineTransitionSettings,
    ) -> Option<State>
    where
        P: SubstitutionPolicy + SubstitutionPolicyFor<U>,
    {
        debug_assert_eq!(settings.max_cost, self.max_distance);

        // A unit-cost result may borrow its positions from `generated_sources`.
        // Mixed internal use is unusual, but it is legal: materialize that
        // input before invalidating the unit-cost table below. Ordinary query
        // surfaces select exactly one transition mode and never pay this copy.
        let materialized_state = state
            .borrows_canonical_positions()
            .then(|| State::from_positions(state.positions().to_vec()));
        let state = materialized_state.as_ref().unwrap_or(state);

        // Characteristic-class identifiers are local to the current padding
        // layout. Affine queries may widen that layout, so discard any unit-
        // cost generated table before remapping classes. Normal query
        // surfaces select exactly one mode; this protects mixed internal use.
        if self.generated_config.take().is_some() {
            self.clear_generated_table();
            self.generated_states.clear();
        }

        // An affine query gap can inspect every remaining query unit when its
        // extension cost is zero. Grow the false suffix once before borrowing
        // the label vector so every legal window remains a direct slice.
        self.cache.ensure_padding(query.len().saturating_add(1));
        let (matches, _) = self.cache.matches_for(policy, dict_unit, query);
        let ctx = TransitionCtx::new(
            query.len(),
            settings.max_cost,
            settings.prefix_mode,
            settings.params,
        );
        let characteristics = CachedCharacteristics::new(matches, query.len());
        transition_epsilon_closed_state_pooled_with::<AffineV, _>(
            state,
            pool,
            &characteristics,
            &ctx,
        )
    }
}

impl<U: CharUnit> CharacteristicCache<U> {
    pub(crate) fn new(query_length: usize, max_distance: usize) -> Self {
        Self {
            direct: std::array::from_fn(|_| None),
            overflow: FxHashMap::default(),
            classes: FxHashMap::default(),
            class_patterns: Vec::new(),
            query_length,
            padding: transition_window_size(max_distance, query_length),
        }
    }

    /// Classify one dictionary unit without borrowing its full characteristic
    /// vector. Generated-table hits need only this compact identifier.
    #[inline]
    pub(crate) fn class_for<P>(&mut self, policy: &P, dict_unit: U, query: &[U]) -> u32
    where
        P: SubstitutionPolicy + SubstitutionPolicyFor<U>,
    {
        debug_assert_eq!(query.len(), self.query_length);
        let direct_index = dict_unit.to_dat_offset();
        if direct_index < self.direct.len() {
            let slot = &mut self.direct[direct_index];
            if let Some((unit, class)) = slot {
                if *unit == dict_unit {
                    return *class;
                }
            }
            if slot.is_none() {
                let class = Self::build_pattern(
                    &mut self.classes,
                    &mut self.class_patterns,
                    self.padding,
                    policy,
                    dict_unit,
                    query,
                );
                *slot = Some((dict_unit, class));
                return class;
            }
        }

        let classes = &mut self.classes;
        let class_patterns = &mut self.class_patterns;
        let padding = self.padding;
        self.overflow
            .entry(dict_unit)
            .or_insert_with(|| {
                Self::build_pattern(classes, class_patterns, padding, policy, dict_unit, query)
            })
            .to_owned()
    }

    #[inline(always)]
    fn matches(&self, class: u32) -> &[bool] {
        self.class_patterns[class as usize].as_ref()
    }

    #[inline]
    pub(crate) fn matches_for<P>(&mut self, policy: &P, dict_unit: U, query: &[U]) -> (&[bool], u32)
    where
        P: SubstitutionPolicy + SubstitutionPolicyFor<U>,
    {
        let class = self.class_for(policy, dict_unit, query);
        (self.matches(class), class)
    }

    fn build_pattern<P>(
        classes: &mut FxHashMap<Arc<[bool]>, u32>,
        class_patterns: &mut Vec<Arc<[bool]>>,
        padding: usize,
        policy: &P,
        dict_unit: U,
        query: &[U],
    ) -> u32
    where
        P: SubstitutionPolicy + SubstitutionPolicyFor<U>,
    {
        let mut matches = Vec::with_capacity(query.len().saturating_add(padding));
        matches.extend(query.iter().copied().map(|query_unit| {
            query_unit == dict_unit || policy.is_allowed_for(dict_unit, query_unit)
        }));
        matches.resize(query.len().saturating_add(padding), false);
        let matches: Arc<[bool]> = matches.into();
        let class = match classes.get(matches.as_ref()) {
            Some(class) => *class,
            None => {
                let class = u32::try_from(classes.len())
                    .expect("query characteristic class count exceeds u32");
                classes.insert(Arc::clone(&matches), class);
                class_patterns.push(Arc::clone(&matches));
                class
            }
        };
        class
    }

    /// Ensure cached vectors contain at least `padding` false units after the
    /// query. Existing vectors are extended in place only when a specialized
    /// automaton requires a wider look-ahead window.
    pub(crate) fn ensure_padding(&mut self, padding: usize) {
        if padding <= self.padding {
            return;
        }

        let length = self.query_length.saturating_add(padding);
        self.classes.clear();
        for (class, pattern) in self.class_patterns.iter_mut().enumerate() {
            let mut extended = pattern.as_ref().to_vec();
            extended.resize(length, false);
            *pattern = extended.into();
            let previous = self.classes.insert(
                Arc::clone(pattern),
                u32::try_from(class).expect("query characteristic class count exceeds u32"),
            );
            debug_assert!(previous.is_none());
        }
        self.padding = padding;
    }
}

#[inline(always)]
fn remaining_error_window(max_distance: usize, num_errors: usize) -> usize {
    match max_distance.checked_sub(num_errors) {
        Some(remaining) => checked_successor_or_max(remaining),
        None => 0,
    }
}

#[inline(always)]
fn checked_position_with_offsets(
    term_index: usize,
    term_offset: usize,
    num_errors: usize,
    error_offset: usize,
) -> Option<Position> {
    let term_index = term_index.checked_add(term_offset)?;
    let num_errors = num_errors.checked_add(error_offset)?;
    Some(Position::new(term_index, num_errors))
}

#[inline(always)]
fn checked_special_position_with_offsets(
    term_index: usize,
    term_offset: usize,
    num_errors: usize,
    error_offset: usize,
    kind: PositionKind,
) -> Option<Position> {
    let term_index = term_index.checked_add(term_offset)?;
    let num_errors = num_errors.checked_add(error_offset)?;
    Some(Position::with_kind(term_index, num_errors, kind, 0))
}

#[inline(always)]
fn checked_deletion_position(
    term_index: usize,
    num_errors: usize,
    match_offset: usize,
) -> Option<Position> {
    let term_offset = match_offset.checked_add(1)?;
    checked_position_with_offsets(term_index, term_offset, num_errors, match_offset)
}

#[inline(always)]
fn push_checked_position(next: &mut SmallVec<[Position; 4]>, position: Option<Position>) {
    if let Some(position) = position {
        next.push(position);
    }
}

#[inline(always)]
fn has_query_units(term_index: usize, count: usize, query_length: usize) -> bool {
    term_index
        .checked_add(count)
        .is_some_and(|end| end <= query_length)
}

/// Transition a position given a characteristic vector.
///
/// Computes all possible next positions from the current position
/// after consuming a dictionary character, considering the query
/// term through the characteristic vector.
///
/// # Standard Algorithm
///
/// From position `(i, e)` (term_index=i, num_errors=e), we can reach:
/// - `(i+1, e)` if `query[i]` matches `dict_char` (no error)
/// - `(i+1, e+1)` via substitution (if different)
/// - `(i, e+1)` via insertion (dictionary char, no query advance)
/// - `(i+1, e+1)` via deletion (skip query char)
///
/// The characteristic vector tells us whether query[offset+k] matches
/// for k in [0..window_size).
///
/// # Prefix Mode
///
/// When `prefix_mode` is true, characters beyond query_length are treated
/// as free matches (no errors added), enabling autocomplete/prefix matching.
#[inline]
pub fn transition_position(
    position: &Position,
    characteristic_vector: &[bool],
    query_length: usize,
    max_distance: usize,
    algorithm: Algorithm,
    prefix_mode: bool,
) -> SmallVec<[Position; 4]> {
    algorithm.assert_supported_max_distance(max_distance);
    match algorithm {
        Algorithm::Standard => transition_standard(
            position,
            characteristic_vector,
            query_length,
            max_distance,
            prefix_mode,
        ),
        Algorithm::Transposition => transition_transposition(
            position,
            characteristic_vector,
            query_length,
            max_distance,
            prefix_mode,
        ),
        Algorithm::MergeAndSplit => transition_merge_split(
            position,
            characteristic_vector,
            query_length,
            max_distance,
            prefix_mode,
        ),
        Algorithm::DamerauLevenshtein => transition_damerau(
            position,
            characteristic_vector,
            query_length,
            max_distance,
            prefix_mode,
        ),
    }
}

/// Standard algorithm transition (insert, delete, substitute)
///
/// Returns SmallVec to avoid heap allocations for small result sets.
/// Most transitions produce 2-3 positions, so we stack-allocate up to 4.
///
/// # Prefix Matching Support
///
/// Find the index of the first true value in characteristic_vector[start..start+limit]
/// Returns None if no true value is found within the range.
///
/// This corresponds to the `index_of` function in the C++ implementation.
#[inline]
fn index_of_match(cv: &[bool], start: usize, limit: usize) -> Option<usize> {
    let end = checked_window_end(start, limit, cv.len())?;
    cv.get(start..end)?.iter().position(|&matched| matched)
}

/// Standard Levenshtein position transition function.
///
/// This implementation follows the C++/Java logic exactly, including the
/// multi-character deletion optimization via `index_of`.
///
/// When `prefix_mode` is true and term_index >= query_length, additional dictionary
/// characters are treated as free matches, keeping the position "stuck" at query_length
/// with the same error count.
#[inline]
pub(crate) fn transition_standard(
    position: &Position,
    cv: &[bool],
    query_length: usize,
    max_distance: usize,
    prefix_mode: bool,
) -> SmallVec<[Position; 4]> {
    let mut next = SmallVec::new();
    transition_standard_into(
        position,
        cv,
        query_length,
        max_distance,
        prefix_mode,
        &mut next,
    );
    next
}

/// Fill a caller-owned buffer with Standard Levenshtein successors.
///
/// Keeping ownership at the caller avoids constructing and then assigning a
/// temporary `SmallVec` inside [`AutomatonVariant::successors`].
#[inline(always)]
pub(crate) fn transition_standard_into(
    position: &Position,
    cv: &[bool],
    query_length: usize,
    max_distance: usize,
    prefix_mode: bool,
    next: &mut SmallVec<[Position; 4]>,
) {
    debug_assert!(next.is_empty());
    let i = position.term_index;
    let e = position.num_errors;
    let h = 0; // cv is offset-adjusted to start at position i, so h = 0
    let w = cv.len();

    // Prefix matching: if enabled and we've consumed the full query, treat any character as a free match
    if prefix_mode && i >= query_length {
        next.push(Position::new(i, e));
        return;
    }

    // Case 1: e < max_distance (can still add errors)
    if e < max_distance {
        // Subcase 1a: At least 2 characters remain in query (h + 2 <= w)
        if h + 2 <= w {
            let a = remaining_error_window(max_distance, e);
            let b = w - h;
            let k = a.min(b);

            match index_of_match(cv, h, k) {
                Some(0) => {
                    // Immediate match at cv[h]
                    push_checked_position(next, checked_position_with_offsets(i, 1, e, 0));
                }
                Some(j) => {
                    // Match found at cv[h + j]
                    // Return: insertion, substitution, and multi-character deletion
                    push_checked_position(next, checked_position_with_offsets(i, 0, e, 1));
                    push_checked_position(next, checked_position_with_offsets(i, 1, e, 1));
                    push_checked_position(next, checked_deletion_position(i, e, j));
                }
                None => {
                    // No match found in range
                    push_checked_position(next, checked_position_with_offsets(i, 0, e, 1));
                    push_checked_position(next, checked_position_with_offsets(i, 1, e, 1));
                }
            }
        }
        // Subcase 1b: Exactly 1 character remains (h + 1 == w)
        else if h + 1 == w {
            if cv[h] {
                // Match at last position
                push_checked_position(next, checked_position_with_offsets(i, 1, e, 0));
            } else {
                // No match at last position
                push_checked_position(next, checked_position_with_offsets(i, 0, e, 1));
                push_checked_position(next, checked_position_with_offsets(i, 1, e, 1));
            }
        }
        // Subcase 1c: Past the end of query (h >= w)
        else {
            // Only insertion is possible
            push_checked_position(next, checked_position_with_offsets(i, 0, e, 1));
        }
    }
    // Case 2: e == max_distance (at max errors, only exact matches allowed)
    else if e == max_distance && h < w && cv[h] {
        push_checked_position(next, checked_position_with_offsets(i, 1, max_distance, 0));
    }
}

/// Unrestricted Damerau–Levenshtein position transition.
///
/// Normal positions retain every Standard successor and may start a
/// Lowrance–Wagner macro transition. Pending positions either resolve their
/// deferred query endpoint or consume another intervening dictionary unit.
#[inline]
pub(crate) fn transition_damerau(
    position: &Position,
    cv: &[bool],
    query_length: usize,
    max_distance: usize,
    prefix_mode: bool,
) -> SmallVec<[Position; 4]> {
    let mut next = SmallVec::new();
    transition_damerau_into(
        position,
        cv,
        query_length,
        max_distance,
        prefix_mode,
        &mut next,
    );
    next
}

/// Fill a caller-owned buffer with unrestricted Damerau successors.
#[inline(always)]
pub(crate) fn transition_damerau_into(
    position: &Position,
    cv: &[bool],
    query_length: usize,
    max_distance: usize,
    prefix_mode: bool,
    next: &mut SmallVec<[Position; 4]>,
) {
    debug_assert!(next.is_empty());

    match position.kind() {
        PositionKind::Normal => {
            transition_standard_into(position, cv, query_length, max_distance, prefix_mode, next);

            let Some(remaining_budget) = max_distance.checked_sub(position.num_errors) else {
                return;
            };
            let max_delta = remaining_budget
                .min(cv.len().saturating_sub(1))
                .min(usize::from(u8::MAX));

            for (delta, &matches) in cv.iter().enumerate().take(max_delta + 1).skip(1) {
                if !matches {
                    continue;
                }
                let Some(num_errors) = position.num_errors.checked_add(delta) else {
                    continue;
                };
                let Ok(delta) = u8::try_from(delta) else {
                    continue;
                };
                next.push(Position::new_damerau_pending(
                    position.term_index,
                    num_errors,
                    delta,
                ));
            }
        }
        PositionKind::DamerauPending => {
            let delta = usize::from(position.aux());

            if cv.first().copied().unwrap_or(false) {
                let term_offset = delta.checked_add(1);
                if let Some(term_index) =
                    term_offset.and_then(|offset| position.term_index.checked_add(offset))
                {
                    next.push(Position::new(term_index, position.num_errors));
                }
            }

            if let Some(num_errors) = position.num_errors.checked_add(1) {
                if num_errors <= max_distance {
                    next.push(Position::new_damerau_pending(
                        position.term_index,
                        num_errors,
                        position.aux(),
                    ));
                }
            }
        }
        _ => {}
    }
}

/// Transposition algorithm transition (adds swap of adjacent chars)
///
/// This implements the transposition Levenshtein distance which allows
/// swapping of two adjacent characters as a single edit operation.
#[inline]
pub(crate) fn transition_transposition(
    position: &Position,
    cv: &[bool],
    query_length: usize,
    max_distance: usize,
    prefix_mode: bool,
) -> SmallVec<[Position; 4]> {
    let mut next = SmallVec::new();
    transition_transposition_into(
        position,
        cv,
        query_length,
        max_distance,
        prefix_mode,
        &mut next,
    );
    next
}

/// Fill a caller-owned buffer with OSA successors.
#[inline(always)]
pub(crate) fn transition_transposition_into(
    position: &Position,
    cv: &[bool],
    query_length: usize,
    max_distance: usize,
    prefix_mode: bool,
    next: &mut SmallVec<[Position; 4]>,
) {
    debug_assert!(next.is_empty());
    let i = position.term_index;
    let e = position.num_errors;
    let t = position.is_special(); // Transposition flag
    let h = 0; // cv is offset-adjusted
    let w = cv.len();

    // Prefix matching
    if prefix_mode && i >= query_length {
        next.push(Position::new(i, e));
        return;
    }

    // Case 1: e == 0 (no errors yet)
    if e == 0 && max_distance > 0 {
        if h + 2 <= w {
            let a = remaining_error_window(max_distance, 0);
            let b = w - h;
            let k = a.min(b);

            match index_of_match(cv, h, k) {
                Some(0) => {
                    // Immediate match
                    push_checked_position(next, checked_position_with_offsets(i, 1, 0, 0));
                }
                Some(1) => {
                    // Match at next position - potential transposition
                    next.push(Position::new(i, 1)); // insertion
                    next.push(Position::with_kind(i, 1, PositionKind::OsaTransposing, 0));
                    push_checked_position(next, checked_position_with_offsets(i, 1, 1, 0));
                    push_checked_position(next, checked_position_with_offsets(i, 2, 1, 0));
                }
                Some(j) => {
                    // Match found at position j > 1
                    next.push(Position::new(i, 1)); // insertion
                    push_checked_position(next, checked_position_with_offsets(i, 1, 1, 0));
                    push_checked_position(next, checked_deletion_position(i, 0, j));
                }
                None => {
                    // No match found
                    next.push(Position::new(i, 1)); // insertion
                    push_checked_position(next, checked_position_with_offsets(i, 1, 1, 0));
                }
            }
        } else if h + 1 == w {
            if cv[h] {
                push_checked_position(next, checked_position_with_offsets(i, 1, 0, 0));
            } else {
                next.push(Position::new(i, 1));
                push_checked_position(next, checked_position_with_offsets(i, 1, 1, 0));
            }
        } else {
            next.push(Position::new(i, 1));
        }
    }
    // Case 2: 1 <= e < max_distance
    else if e >= 1 && e < max_distance {
        if h + 2 <= w {
            if !t {
                // Not in transposition state
                let a = remaining_error_window(max_distance, e);
                let b = w - h;
                let k = a.min(b);

                match index_of_match(cv, h, k) {
                    Some(0) => {
                        push_checked_position(next, checked_position_with_offsets(i, 1, e, 0));
                    }
                    Some(1) => {
                        push_checked_position(next, checked_position_with_offsets(i, 0, e, 1));
                        push_checked_position(
                            next,
                            checked_special_position_with_offsets(
                                i,
                                0,
                                e,
                                1,
                                PositionKind::OsaTransposing,
                            ),
                        );
                        push_checked_position(next, checked_position_with_offsets(i, 1, e, 1));
                        push_checked_position(next, checked_position_with_offsets(i, 2, e, 1));
                    }
                    Some(j) => {
                        push_checked_position(next, checked_position_with_offsets(i, 0, e, 1));
                        push_checked_position(next, checked_position_with_offsets(i, 1, e, 1));
                        push_checked_position(next, checked_deletion_position(i, e, j));
                    }
                    None => {
                        push_checked_position(next, checked_position_with_offsets(i, 0, e, 1));
                        push_checked_position(next, checked_position_with_offsets(i, 1, e, 1));
                    }
                }
            } else {
                // In transposition state (is_special == true)
                if cv[h] {
                    // Complete the transposition by matching
                    push_checked_position(next, checked_position_with_offsets(i, 2, e, 0));
                }
                // else: no valid transitions from failed transposition
            }
        } else if h + 1 == w {
            if cv[h] {
                push_checked_position(next, checked_position_with_offsets(i, 1, e, 0));
            } else {
                push_checked_position(next, checked_position_with_offsets(i, 0, e, 1));
                push_checked_position(next, checked_position_with_offsets(i, 1, e, 1));
            }
        } else {
            push_checked_position(next, checked_position_with_offsets(i, 0, e, 1));
        }
    }
    // Case 3: e == max_distance (at max errors)
    else if e == max_distance {
        if h < w && !t {
            if cv[h] {
                push_checked_position(next, checked_position_with_offsets(i, 1, max_distance, 0));
            }
            // else: no transitions at max distance without match
        } else if h + 2 <= w && t && cv[h] {
            // Complete transposition at max distance
            push_checked_position(next, checked_position_with_offsets(i, 2, max_distance, 0));
        }
    }
}

/// Merge and split algorithm transition
///
/// This implements merge-and-split operations:
/// - Merge: Two query characters combine into one dictionary character
/// - Split: One query character expands into two dictionary characters
#[inline]
pub(crate) fn transition_merge_split(
    position: &Position,
    cv: &[bool],
    query_length: usize,
    max_distance: usize,
    prefix_mode: bool,
) -> SmallVec<[Position; 4]> {
    let mut next = SmallVec::new();
    transition_merge_split_into(
        position,
        cv,
        query_length,
        max_distance,
        prefix_mode,
        &mut next,
    );
    next
}

/// Fill a caller-owned buffer with merge/split successors.
#[inline(always)]
pub(crate) fn transition_merge_split_into(
    position: &Position,
    cv: &[bool],
    query_length: usize,
    max_distance: usize,
    prefix_mode: bool,
    next: &mut SmallVec<[Position; 4]>,
) {
    debug_assert!(next.is_empty());
    let i = position.term_index;
    let e = position.num_errors;
    let s = position.is_special(); // Special flag for merge/split
    let h = 0; // cv is offset-adjusted
    let w = cv.len();

    // Prefix matching
    if prefix_mode && i >= query_length {
        next.push(Position::new(i, e));
        return;
    }

    // Case 1: e == 0 (no errors yet)
    if e == 0 && max_distance > 0 {
        if h + 2 <= w {
            if cv[h] {
                // Immediate match
                push_checked_position(next, checked_position_with_offsets(i, 1, e, 0));
            } else {
                // No match - add error operations including merge/split
                push_checked_position(next, checked_position_with_offsets(i, 0, e, 1));
                // Split operation: one query char becomes two dict chars (only if we have 1 char available)
                if i < query_length {
                    push_checked_position(
                        next,
                        checked_special_position_with_offsets(i, 0, e, 1, PositionKind::Splitting),
                    );
                }
                push_checked_position(next, checked_position_with_offsets(i, 1, e, 1));
                // Merge operation: skip 2 query chars (only if we have 2 chars available)
                if has_query_units(i, 2, query_length) {
                    push_checked_position(next, checked_position_with_offsets(i, 2, e, 1));
                }
            }
        } else if h + 1 == w {
            if cv[h] {
                push_checked_position(next, checked_position_with_offsets(i, 1, e, 0));
            } else {
                push_checked_position(next, checked_position_with_offsets(i, 0, e, 1));
                // Split operation: one query char becomes two dict chars (only if we have 1 char available)
                if i < query_length {
                    push_checked_position(
                        next,
                        checked_special_position_with_offsets(i, 0, e, 1, PositionKind::Splitting),
                    );
                }
                push_checked_position(next, checked_position_with_offsets(i, 1, e, 1));
            }
        } else {
            push_checked_position(next, checked_position_with_offsets(i, 0, e, 1));
        }
    }
    // Case 2: e < max_distance (can still add errors)
    else if e < max_distance {
        if h + 2 <= w {
            if !s {
                // Not in special state
                if cv[h] {
                    push_checked_position(next, checked_position_with_offsets(i, 1, e, 0));
                } else {
                    push_checked_position(next, checked_position_with_offsets(i, 0, e, 1));
                    // Split operation: one query char becomes two dict chars (only if we have 1 char available)
                    if i < query_length {
                        push_checked_position(
                            next,
                            checked_special_position_with_offsets(
                                i,
                                0,
                                e,
                                1,
                                PositionKind::Splitting,
                            ),
                        );
                    }
                    push_checked_position(next, checked_position_with_offsets(i, 1, e, 1));
                    // Merge operation: skip 2 query chars (only if we have 2 chars available)
                    if has_query_units(i, 2, query_length) {
                        push_checked_position(next, checked_position_with_offsets(i, 2, e, 1));
                    }
                }
            } else {
                // In special state (completing split)
                push_checked_position(next, checked_position_with_offsets(i, 1, e, 0));
            }
        } else if h + 1 == w {
            if !s {
                if cv[h] {
                    push_checked_position(next, checked_position_with_offsets(i, 1, e, 0));
                } else {
                    push_checked_position(next, checked_position_with_offsets(i, 0, e, 1));
                    // Split operation: one query char becomes two dict chars (only if we have 1 char available)
                    if i < query_length {
                        push_checked_position(
                            next,
                            checked_special_position_with_offsets(
                                i,
                                0,
                                e,
                                1,
                                PositionKind::Splitting,
                            ),
                        );
                    }
                    push_checked_position(next, checked_position_with_offsets(i, 1, e, 1));
                }
            } else {
                // Special state at boundary
                push_checked_position(next, checked_position_with_offsets(i, 1, e, 0));
            }
        } else {
            push_checked_position(next, checked_position_with_offsets(i, 0, e, 1));
        }
    }
    // Case 3: e == max_distance (at max errors)
    else if e == max_distance && h < w {
        if !s {
            if cv[h] {
                push_checked_position(next, checked_position_with_offsets(i, 1, max_distance, 0));
            }
            // else: no transitions at max distance without match
        } else {
            // Special state: can advance even at max distance (completing split)
            push_checked_position(next, checked_position_with_offsets(i, 1, e, 0));
        }
    }
}

/// Compute epsilon closure: add positions reachable by deletion (skipping query chars)
/// without consuming dictionary characters.
///
/// Optimized version that modifies the state in-place to avoid cloning.
#[inline]
fn epsilon_closure_mut<V: AutomatonVariant>(state: &mut State, ctx: &TransitionCtx<V::Params>) {
    // Pre-allocate with typical size to avoid reallocation
    let mut to_process: SmallVec<[Position; 8]> = SmallVec::with_capacity(8);

    // Start with current positions
    for pos in state.positions() {
        to_process.push(*pos);
    }

    let mut processed = 0;
    let mut epsilon = SmallVec::<[Position; 4]>::new();
    while processed < to_process.len() {
        let position = to_process[processed];
        processed += 1;

        epsilon.clear();
        V::epsilon_successors(position, ctx, &mut epsilon);
        for deleted in epsilon.iter().copied() {
            // Try to insert - State.insert handles deduplication efficiently
            // Only add to to_process if it was actually inserted (new position)
            if state.insert_with::<V>(deleted, ctx) {
                to_process.push(deleted);
            }
        }
    }
}

/// Wrapper that creates a new state and applies epsilon closure
fn epsilon_closure<V: AutomatonVariant>(state: &State, ctx: &TransitionCtx<V::Params>) -> State {
    let mut result = state.clone();
    epsilon_closure_mut::<V>(&mut result, ctx);
    result
}

/// Compute epsilon closure from source into target state (pool-friendly).
///
/// This function copies positions from the source state into the target state
/// and then applies epsilon closure in-place. The target state is cleared first.
///
/// # Performance
///
/// This is optimized for use with StatePool:
/// - Target state's Vec allocation is reused
/// - Position is Copy, so no clone overhead
/// - Eliminates one State::clone compared to epsilon_closure()
#[inline]
fn epsilon_closure_into<V: AutomatonVariant>(
    source: &State,
    target: &mut State,
    ctx: &TransitionCtx<V::Params>,
) {
    // Copy positions from source to target
    target.copy_from(source);

    // Apply epsilon closure in-place
    epsilon_closure_mut::<V>(target, ctx);
}

#[inline]
fn transition_zero_distance_into<
    U: CharUnit,
    P: SubstitutionPolicy + SubstitutionPolicyFor<U>,
    V: AutomatonVariant,
>(
    state: &State,
    target: &mut State,
    policy: &P,
    dict_unit: U,
    query: &[U],
    ctx: &TransitionCtx<V::Params>,
) {
    let query_length = ctx.query_length;

    for position in state.positions() {
        if ctx.prefix_mode && position.term_index >= query_length {
            target.insert_with::<V>(Position::new(position.term_index, position.num_errors), ctx);
            continue;
        }

        if position.num_errors != 0 {
            continue;
        }

        let Some(&query_unit) = query.get(position.term_index) else {
            continue;
        };

        if query_unit == dict_unit || policy.is_allowed_for(dict_unit, query_unit) {
            if let Some(next_position) =
                checked_position_with_offsets(position.term_index, 1, position.num_errors, 0)
            {
                target.insert_with::<V>(next_position, ctx);
            }
        }
    }
}

#[inline]
fn supports_zero_distance_fast_path(state: &State) -> bool {
    state
        .positions()
        .iter()
        .all(|position| !position.is_special() && position.num_errors == 0)
}

/// Transition an entire state given a dictionary character unit.
///
/// Computes the next state by transitioning all positions in the
/// current state and merging the results.
pub fn transition_state<U: CharUnit, P: SubstitutionPolicy + SubstitutionPolicyFor<U>>(
    state: &State,
    policy: P,
    dict_unit: U,
    query: &[U],
    max_distance: usize,
    algorithm: Algorithm,
    prefix_mode: bool,
) -> Option<State> {
    algorithm.assert_supported_max_distance(max_distance);
    let ctx = TransitionCtx::unit(query.len(), max_distance, prefix_mode);
    with_variant!(VariantSpec::from(algorithm), |V| {
        transition_state_inner::<U, P, V>(state, &policy, dict_unit, query, &ctx)
    })
}

#[inline]
fn transition_state_inner<
    U: CharUnit,
    P: SubstitutionPolicy + SubstitutionPolicyFor<U>,
    V: AutomatonVariant,
>(
    state: &State,
    policy: &P,
    dict_unit: U,
    query: &[U],
    ctx: &TransitionCtx<V::Params>,
) -> Option<State> {
    crate::causal_perf::record_transition_attempts(1);
    if ctx.max_distance == 0
        && V::supports_zero_distance_fast_path(ctx)
        && supports_zero_distance_fast_path(state)
    {
        let mut next_state = State::new();
        transition_zero_distance_into::<U, P, V>(
            state,
            &mut next_state,
            policy,
            dict_unit,
            query,
            ctx,
        );
        return (!next_state.is_empty()).then_some(next_state);
    }

    // First, compute epsilon closure to handle deletions
    let expanded_state = epsilon_closure::<V>(state, ctx);

    let mut next_state = State::new();
    let mut cv = SmallVec::<[bool; 8]>::new();
    let mut next_positions = SmallVec::<[Position; 4]>::new();

    for position in expanded_state.positions() {
        let offset = position.term_index;
        let window_size = V::skip_window(position, ctx);
        let cv = characteristic_vector(policy, dict_unit, query, window_size, offset, &mut cv);

        next_positions.clear();
        V::successors(*position, cv, ctx, &mut next_positions);
        for next_pos in next_positions.iter().copied() {
            next_state.insert_with::<V>(next_pos, ctx);
        }
    }

    if next_state.is_empty() {
        None
    } else {
        Some(next_state)
    }
}

/// Transition a state using a StatePool for allocation reuse (optimized).
///
/// This is the pool-aware version of `transition_state` that eliminates
/// State cloning overhead by reusing allocations from the pool.
///
/// # Performance
///
/// Compared to `transition_state`:
/// - Eliminates Vec allocation for expanded_state (reuses from pool)
/// - Eliminates Vec allocation for next_state (reuses from pool)
/// - Reduces State::clone overhead by ~6-10% of total runtime
///
/// # Arguments
///
/// * `state` - Current automaton state
/// * `pool` - State pool for allocation reuse
/// * `policy` - Substitution policy for character matching
/// * `dict_unit` - Dictionary character unit to transition on
/// * `query` - Query term units (bytes or chars)
/// * `settings` - Maximum distance, algorithm, and prefix-mode configuration
///
/// # Returns
///
/// The next state after transitioning, or None if no valid transitions exist.
/// The returned state is acquired from the pool (caller should release it when done).
#[inline]
pub fn transition_state_pooled<U: CharUnit, P: SubstitutionPolicy + SubstitutionPolicyFor<U>>(
    state: &State,
    pool: &mut StatePool,
    policy: P,
    dict_unit: U,
    query: &[U],
    settings: TransitionSettings,
) -> Option<State> {
    transition_state_pooled_ref(state, pool, &policy, dict_unit, query, settings)
}

/// Transition a state using a borrowed policy and a StatePool for allocation reuse.
///
/// This preserves the public by-value `transition_state_pooled` API while letting
/// internal iterators avoid cloning non-ZST policies once per explored edge.
#[inline]
pub(crate) fn transition_state_pooled_ref<
    U: CharUnit,
    P: SubstitutionPolicy + SubstitutionPolicyFor<U>,
>(
    state: &State,
    pool: &mut StatePool,
    policy: &P,
    dict_unit: U,
    query: &[U],
    settings: TransitionSettings,
) -> Option<State> {
    settings
        .algorithm
        .assert_supported_max_distance(settings.max_distance);
    let ctx = TransitionCtx::unit(query.len(), settings.max_distance, settings.prefix_mode);
    with_variant!(VariantSpec::from(settings.algorithm), |V| {
        transition_state_pooled_inner::<U, P, V>(state, pool, policy, dict_unit, query, &ctx)
    })
}

/// Transition from a state that is already epsilon-closed and close the
/// accepted result before returning it.
///
/// This is the query-iterator fast path. Its caller maintains the invariant
/// that the initial state and every queued successor are epsilon-closed for
/// the same transition context. A dictionary node can therefore share one
/// closure across all of its outgoing labels instead of copying and closing
/// the source independently for every edge.
#[inline]
pub(crate) fn transition_epsilon_closed_state_pooled_cached(
    state: &State,
    pool: &mut StatePool,
    matches: &[bool],
    query_length: usize,
    settings: TransitionSettings,
) -> Option<State> {
    settings
        .algorithm
        .assert_supported_max_distance(settings.max_distance);
    let ctx = TransitionCtx::unit(query_length, settings.max_distance, settings.prefix_mode);
    let characteristics = CachedCharacteristics {
        matches,
        query_length,
    };
    with_variant!(VariantSpec::from(settings.algorithm), |V| {
        transition_epsilon_closed_state_pooled_with::<V, _>(state, pool, &characteristics, &ctx)
    })
}

pub(crate) trait CharacteristicProvider {
    fn query_length(&self) -> usize;

    fn window<'a>(
        &'a self,
        offset: usize,
        window_size: usize,
        scratch: &'a mut SmallVec<[bool; 8]>,
    ) -> &'a [bool];
}

pub(crate) struct OnDemandCharacteristics<'a, U: CharUnit, P> {
    policy: &'a P,
    dict_unit: U,
    query: &'a [U],
}

impl<'a, U: CharUnit, P> OnDemandCharacteristics<'a, U, P> {
    pub(crate) const fn new(policy: &'a P, dict_unit: U, query: &'a [U]) -> Self {
        Self {
            policy,
            dict_unit,
            query,
        }
    }
}

impl<U, P> CharacteristicProvider for OnDemandCharacteristics<'_, U, P>
where
    U: CharUnit,
    P: SubstitutionPolicy + SubstitutionPolicyFor<U>,
{
    #[inline(always)]
    fn query_length(&self) -> usize {
        self.query.len()
    }

    #[inline(always)]
    fn window<'a>(
        &'a self,
        offset: usize,
        window_size: usize,
        scratch: &'a mut SmallVec<[bool; 8]>,
    ) -> &'a [bool] {
        characteristic_vector(
            self.policy,
            self.dict_unit,
            self.query,
            window_size,
            offset,
            scratch,
        )
    }
}

pub(crate) struct CachedCharacteristics<'a> {
    matches: &'a [bool],
    query_length: usize,
}

impl<'a> CachedCharacteristics<'a> {
    pub(crate) const fn new(matches: &'a [bool], query_length: usize) -> Self {
        Self {
            matches,
            query_length,
        }
    }
}

impl CharacteristicProvider for CachedCharacteristics<'_> {
    #[inline(always)]
    fn query_length(&self) -> usize {
        self.query_length
    }

    #[inline(always)]
    fn window<'a>(
        &'a self,
        offset: usize,
        window_size: usize,
        scratch: &'a mut SmallVec<[bool; 8]>,
    ) -> &'a [bool] {
        let start = offset.min(self.query_length);
        if let Some(end) = start.checked_add(window_size) {
            if let Some(window) = self.matches.get(start..end) {
                return window;
            }
        }

        // Defensive fallback for a caller whose requested window exceeds the
        // cache's construction bound. Public distance validation normally
        // makes this unreachable, but retaining it keeps extreme inputs safe.
        scratch.clear();
        scratch.resize(window_size, false);
        scratch.as_slice()
    }
}

#[inline]
fn transition_state_pooled_inner<
    U: CharUnit,
    P: SubstitutionPolicy + SubstitutionPolicyFor<U>,
    V: AutomatonVariant,
>(
    state: &State,
    pool: &mut StatePool,
    policy: &P,
    dict_unit: U,
    query: &[U],
    ctx: &TransitionCtx<V::Params>,
) -> Option<State> {
    let characteristics = OnDemandCharacteristics {
        policy,
        dict_unit,
        query,
    };
    transition_state_pooled_with::<V, _>(state, pool, &characteristics, ctx)
}

#[inline]
fn transition_state_pooled_with<V: AutomatonVariant, C: CharacteristicProvider>(
    state: &State,
    pool: &mut StatePool,
    characteristics: &C,
    ctx: &TransitionCtx<V::Params>,
) -> Option<State> {
    crate::causal_perf::record_transition_attempts(1);
    if ctx.max_distance == 0
        && V::supports_zero_distance_fast_path(ctx)
        && supports_zero_distance_fast_path(state)
    {
        let mut next_state = pool.acquire();
        transition_zero_distance_with_characteristics::<V, C>(
            state,
            &mut next_state,
            characteristics,
            ctx,
        );

        if next_state.is_empty() {
            pool.release(next_state);
            return None;
        }

        return Some(next_state);
    }

    // Acquire a state from pool for epsilon closure (reuses allocation!)
    let mut expanded_state = pool.acquire();

    // Compute epsilon closure into the pooled state (no clone!)
    crate::causal_perf::record_epsilon_input_positions(state.len() as u64);
    epsilon_closure_into::<V>(state, &mut expanded_state, ctx);
    crate::causal_perf::record_epsilon_output_positions(expanded_state.len() as u64);

    // Acquire another state from pool for next state
    let mut next_state = pool.acquire();
    transition_epsilon_closed_state_into::<V, C>(
        &expanded_state,
        &mut next_state,
        characteristics,
        ctx,
    );

    // Return expanded_state to pool (no longer needed)
    pool.release(expanded_state);

    if next_state.is_empty() {
        // Return empty state to pool
        pool.release(next_state);
        None
    } else {
        Some(next_state)
    }
}

/// Pooled transition kernel for the epsilon-closed queued-state invariant.
///
/// The source closure is label-independent, so the iterator stores it once.
/// The non-empty output is closed in place before it is queued. Both this
/// path and the compatibility path above share the same monomorphized
/// per-position transition kernel.
#[inline]
fn transition_epsilon_closed_state_pooled_with<V: AutomatonVariant, C: CharacteristicProvider>(
    state: &State,
    pool: &mut StatePool,
    characteristics: &C,
    ctx: &TransitionCtx<V::Params>,
) -> Option<State> {
    crate::causal_perf::record_transition_attempts(1);
    let mut next_state = pool.acquire();

    if ctx.max_distance == 0
        && V::supports_zero_distance_fast_path(ctx)
        && supports_zero_distance_fast_path(state)
    {
        transition_zero_distance_with_characteristics::<V, C>(
            state,
            &mut next_state,
            characteristics,
            ctx,
        );
    } else {
        transition_epsilon_closed_state_into::<V, C>(state, &mut next_state, characteristics, ctx);
    }

    if next_state.is_empty() {
        pool.release(next_state);
        return None;
    }

    // Establish the invariant consumed by the next dictionary node. Closing
    // here performs the work once per accepted state rather than once per
    // outgoing edge of that state.
    crate::causal_perf::record_epsilon_input_positions(next_state.len() as u64);
    epsilon_closure_mut::<V>(&mut next_state, ctx);
    crate::causal_perf::record_epsilon_output_positions(next_state.len() as u64);

    Some(next_state)
}

/// Consume one dictionary unit from an epsilon-closed source state.
///
/// Keeping this as the single generic inner loop avoids duplicating the hot
/// algorithm logic between the public compatibility path and the queued-state
/// fast path. `V` and `C` remain statically dispatched and monomorphized.
#[inline]
fn transition_epsilon_closed_state_into<V: AutomatonVariant, C: CharacteristicProvider>(
    state: &State,
    next_state: &mut State,
    characteristics: &C,
    ctx: &TransitionCtx<V::Params>,
) {
    let mut cv = SmallVec::<[bool; 8]>::new();
    let mut next_positions = SmallVec::<[Position; 4]>::new();

    for position in state.positions() {
        let offset = position.term_index;
        let window_size = V::skip_window(position, ctx);
        crate::causal_perf::record_characteristic_vectors(1);
        crate::causal_perf::record_characteristic_units(window_size as u64);
        let cv = characteristics.window(offset, window_size, &mut cv);

        next_positions.clear();
        V::successors(*position, cv, ctx, &mut next_positions);
        crate::causal_perf::record_successor_candidates(next_positions.len() as u64);
        for next_pos in next_positions.iter().copied() {
            next_state.insert_with::<V>(next_pos, ctx);
        }
    }
}

#[inline]
fn transition_zero_distance_with_characteristics<V: AutomatonVariant, C: CharacteristicProvider>(
    state: &State,
    target: &mut State,
    characteristics: &C,
    ctx: &TransitionCtx<V::Params>,
) {
    let query_length = characteristics.query_length();
    let mut scratch = SmallVec::<[bool; 8]>::new();

    for position in state.positions() {
        if ctx.prefix_mode && position.term_index >= query_length {
            target.insert_with::<V>(Position::new(position.term_index, position.num_errors), ctx);
            continue;
        }
        if position.num_errors != 0 {
            continue;
        }
        if characteristics
            .window(position.term_index, 1, &mut scratch)
            .first()
            .copied()
            .unwrap_or(false)
        {
            if let Some(next_position) =
                checked_position_with_offsets(position.term_index, 1, position.num_errors, 0)
            {
                target.insert_with::<V>(next_position, ctx);
            }
        }
    }
}

/// Create the initial state for a query.
///
/// The initial state contains positions representing all possible
/// ways to start matching (including initial errors via deletions/insertions).
pub fn initial_state(query_length: usize, max_distance: usize, algorithm: Algorithm) -> State {
    algorithm.assert_supported_max_distance(max_distance);
    let ctx = TransitionCtx::unit(query_length, max_distance, false);
    with_variant!(VariantSpec::from(algorithm), |V| {
        initial_state_with::<V>(&ctx)
    })
}

#[inline]
fn initial_state_with<V: AutomatonVariant>(ctx: &TransitionCtx<V::Params>) -> State {
    let mut state = State::new();

    // Start at position (0, 0) - no errors, beginning of query
    state.insert_with::<V>(Position::new(0, 0), ctx);

    // Seed every legal query-prefix gap. For unit-cost variants this is the
    // historical `(i, i)` prefix; affine uses the same fixpoint shape with its
    // layer-aware open/extension costs.
    epsilon_closure_mut::<V>(&mut state, ctx);

    state
}

/// Create the initial state for an exact scaled affine-gap query.
pub(crate) fn initial_state_affine(
    query_length: usize,
    max_cost: usize,
    params: AffineGapParams,
) -> State {
    // Keep the runtime seam's marker executable even though parameters require
    // the typed path below rather than the unit-parameter dispatch macro.
    let _spec = VariantSpec::AffineGap;
    let ctx = TransitionCtx::new(query_length, max_cost, false, params);
    initial_state_with::<AffineV>(&ctx)
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::transducer::{Restricted, SubstitutionSet, Unrestricted};

    fn short_words<U: CharUnit>(alphabet: &[U], max_len: usize) -> Vec<Vec<U>> {
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

    fn assert_packed_matches_positional<U, P>(
        corpus: &[Vec<U>],
        max_distance: usize,
        prefix_mode: bool,
        policy: &P,
    ) where
        U: CharUnit,
        P: SubstitutionPolicy + SubstitutionPolicyFor<U>,
    {
        let settings = TransitionSettings::new(max_distance, Algorithm::Standard, prefix_mode);
        for query in corpus {
            let Some((mut packed, packed_root)) =
                UnitCostMachine::seeded_packed_for_test::<P>(query, settings)
            else {
                continue;
            };
            for term in corpus {
                let (mut positional, positional_root) =
                    UnitCostMachine::seeded_positional(query.len(), settings);
                let mut packed_frontier = Some(packed_root);
                let mut positional_frontier = Some(positional_root);
                let mut packed_pool = StatePool::new();
                let mut positional_pool = StatePool::new();
                for &label in term {
                    packed_frontier = packed_frontier.and_then(|frontier| {
                        packed.step(frontier, &mut packed_pool, policy, label, query, settings)
                    });
                    positional_frontier = positional_frontier.and_then(|frontier| {
                        positional.step(
                            frontier,
                            &mut positional_pool,
                            policy,
                            label,
                            query,
                            settings,
                        )
                    });
                }
                assert_eq!(
                    packed_frontier.is_some(),
                    positional_frontier.is_some(),
                    "query={query:?} term={term:?} d={max_distance} prefix={prefix_mode}",
                );
                let (Some(packed_frontier), Some(positional_frontier)) =
                    (packed_frontier, positional_frontier)
                else {
                    continue;
                };
                assert_eq!(
                    packed.min_distance(packed_frontier),
                    positional.min_distance(positional_frontier),
                    "minimum distance: query={query:?} term={term:?} d={max_distance} prefix={prefix_mode}",
                );
                for mode in [FinishMode::Complete, FinishMode::Substring] {
                    assert_eq!(
                        packed
                            .finish_distance(packed_frontier, mode, query.len())
                            .filter(|distance| *distance <= max_distance),
                        positional
                            .finish_distance(positional_frontier, mode, query.len())
                            .filter(|distance| *distance <= max_distance),
                        "finish={mode:?}: query={query:?} term={term:?} d={max_distance} prefix={prefix_mode}",
                    );
                }
            }
        }
    }

    #[test]
    fn packed_and_positional_standard_frontiers_are_exhaustively_equivalent() {
        let byte_words = short_words(b"ab", 5);
        let char_words = short_words(&['a', 'é'], 4);
        let token_words = short_words(&[7u64, u64::MAX], 4);
        for max_distance in 0..=3 {
            for prefix_mode in [false, true] {
                assert_packed_matches_positional(
                    &byte_words,
                    max_distance,
                    prefix_mode,
                    &Unrestricted,
                );
                assert_packed_matches_positional(
                    &char_words,
                    max_distance,
                    prefix_mode,
                    &Unrestricted,
                );
                assert_packed_matches_positional(
                    &token_words,
                    max_distance,
                    prefix_mode,
                    &Unrestricted,
                );
            }
        }
    }

    #[test]
    fn packed_directional_restricted_masks_match_the_positional_oracle() {
        let corpus = short_words(b"abc", 4);
        let mut substitutions = SubstitutionSet::new();
        substitutions.allow_byte(b'a', b'b');
        substitutions.allow_byte(b'c', b'a');
        let policy = Restricted::new(&substitutions);
        for max_distance in 0..=3 {
            for prefix_mode in [false, true] {
                assert_packed_matches_positional(&corpus, max_distance, prefix_mode, &policy);
            }
        }
    }

    #[test]
    fn compact_generated_transition_cells_are_one_machine_word() {
        assert_eq!(
            std::mem::size_of::<GeneratedStateId>(),
            std::mem::size_of::<usize>()
        );
        assert_eq!(
            std::mem::size_of::<GeneratedTarget>(),
            std::mem::size_of::<usize>()
        );
        const {
            assert!(GeneratedTarget::EMPTY.0 < GeneratedTarget::UNCOMPUTED.0);
        }
    }

    #[test]
    fn dense_generated_targets_keep_custom_policy_classes_in_sparse_overflow() {
        let mut targets = DenseGeneratedTargets::new(2);
        let row_zero = GeneratedStateId(0);
        let row_one = GeneratedStateId(1);
        targets.push_row();
        targets.push_row();
        targets.set(row_zero, 0, GeneratedTarget::state(GeneratedStateId(7)));
        targets.set(row_one, 1, GeneratedTarget::EMPTY);

        targets.set(row_zero, 5, GeneratedTarget::state(GeneratedStateId(9)));

        assert_eq!(targets.stride(), 2);
        assert_eq!(
            targets.get(row_zero, 0),
            GeneratedTarget::state(GeneratedStateId(7))
        );
        assert_eq!(targets.get(row_one, 1), GeneratedTarget::EMPTY);
        assert_eq!(
            targets.get(row_zero, 5),
            GeneratedTarget::state(GeneratedStateId(9))
        );
        assert_eq!(targets.get(row_one, 5), GeneratedTarget::UNCOMPUTED);

        let row_two = GeneratedStateId(2);
        targets.push_row();
        for class in 0..targets.stride() {
            assert_eq!(targets.get(row_two, class), GeneratedTarget::UNCOMPUTED);
        }
        assert_eq!(targets.get(row_two, 5), GeneratedTarget::UNCOMPUTED);
    }

    #[test]
    fn test_characteristic_vector() {
        let query = b"test";
        let policy = Unrestricted;
        let mut buffer = SmallVec::<[bool; 8]>::new();

        let cv = characteristic_vector(&policy, b't', query, 3, 0, &mut buffer);
        assert_eq!(cv, &[true, false, false]);

        let cv = characteristic_vector(&policy, b'e', query, 3, 0, &mut buffer);
        assert_eq!(cv, &[false, true, false]);

        let cv = characteristic_vector(&policy, b's', query, 3, 1, &mut buffer);
        assert_eq!(cv, &[false, true, false]);
    }

    #[test]
    fn test_characteristic_vector_supports_large_windows() {
        let query = b"abcdefghijklmnop";
        let policy = Unrestricted;
        let mut buffer = SmallVec::<[bool; 8]>::new();

        let cv = characteristic_vector(&policy, b'j', query, 12, 0, &mut buffer);

        assert_eq!(cv.len(), 12);
        assert!(cv[9]);
    }

    #[test]
    fn characteristic_cache_extends_existing_vectors_for_specialized_windows() {
        let query = b"test";
        let policy = Unrestricted;
        let mut cache = CharacteristicCache::new(query.len(), 1);

        let before = cache.matches_for(&policy, b't', query).0.to_vec();
        assert_eq!(before.len(), query.len() + 2);

        cache.ensure_padding(8);
        let after = cache.matches_for(&policy, b't', query).0;
        assert_eq!(after.len(), query.len() + 8);
        assert_eq!(&after[..query.len()], &[true, false, false, true]);
        assert!(after[query.len()..].iter().all(|matched| !matched));
    }

    #[test]
    fn characteristic_cache_handles_colliding_u64_direct_offsets_exactly() {
        let high = (1u64 << 32) | 1;
        let query = [1u64, high, 2];
        let policy = Unrestricted;
        let mut cache = CharacteristicCache::new(query.len(), 2);

        let low_matches = cache.matches_for(&policy, 1, &query).0.to_vec();
        let high_matches = cache.matches_for(&policy, high, &query).0.to_vec();

        assert_eq!(&low_matches[..query.len()], &[true, false, false]);
        assert_eq!(&high_matches[..query.len()], &[false, true, false]);
        assert_eq!(cache.matches_for(&policy, 1, &query).0, low_matches);
        assert_eq!(cache.matches_for(&policy, high, &query).0, high_matches);
    }

    #[test]
    fn generated_transition_table_matches_positional_kernel_for_every_unit_variant() {
        const ALGORITHMS: [Algorithm; 4] = [
            Algorithm::Standard,
            Algorithm::Transposition,
            Algorithm::MergeAndSplit,
            Algorithm::DamerauLevenshtein,
        ];
        let query = b"abca";
        let labels = *b"abcx";
        let policy = Unrestricted;

        for algorithm in ALGORITHMS {
            for max_distance in 0..=3 {
                for prefix_mode in [false, true] {
                    let settings = TransitionSettings::new(max_distance, algorithm, prefix_mode);
                    let mut generated = CachedUnitTransitions::new(query.len(), max_distance);
                    let initial = initial_state(query.len(), max_distance, algorithm);
                    let mut frontier = vec![(initial.clone(), initial)];
                    let mut generated_pool = StatePool::new();
                    let mut reference_pool = StatePool::new();

                    for _depth in 0..4 {
                        let mut next_frontier = Vec::new();
                        for (generated_state, reference_state) in frontier {
                            for label in labels {
                                let actual = generated.transition(
                                    &generated_state,
                                    &mut generated_pool,
                                    &policy,
                                    label,
                                    query,
                                    settings,
                                );
                                let mut matches = query
                                    .iter()
                                    .map(|query_unit| *query_unit == label)
                                    .collect::<Vec<_>>();
                                matches.resize(
                                    query.len() + transition_window_size(max_distance, query.len()),
                                    false,
                                );
                                let expected = transition_epsilon_closed_state_pooled_cached(
                                    &reference_state,
                                    &mut reference_pool,
                                    &matches,
                                    query.len(),
                                    settings,
                                );

                                assert_eq!(
                                    actual.as_ref().map(State::positions),
                                    expected.as_ref().map(State::positions),
                                    "algorithm={algorithm:?} distance={max_distance} prefix={prefix_mode} label={label:?}",
                                );
                                if let (Some(actual), Some(expected)) = (actual, expected) {
                                    next_frontier.push((actual, expected));
                                }
                            }
                        }
                        frontier = next_frontier;
                    }
                }
            }
        }
    }

    #[test]
    fn affine_transition_materializes_a_borrowed_unit_cost_frontier_before_reset() {
        let query = b"abca";
        let policy = Unrestricted;
        let max_cost = 2;
        let mut mixed = CachedUnitTransitions::new(query.len(), max_cost);
        let mut mixed_pool = StatePool::new();
        let initial = initial_state(query.len(), max_cost, Algorithm::Standard);
        let borrowed = mixed
            .transition(
                &initial,
                &mut mixed_pool,
                &policy,
                b'a',
                query,
                TransitionSettings::new(max_cost, Algorithm::Standard, false),
            )
            .expect("unit-cost transition");
        assert!(borrowed.borrows_canonical_positions());

        let owned = State::from_positions(borrowed.positions().to_vec());
        let params = AffineGapParams::new(1.0, 1.0, 1.0).expect("unit affine costs");
        let settings = AffineTransitionSettings::new(max_cost, params, false);
        let actual =
            mixed.transition_affine(&borrowed, &mut mixed_pool, &policy, b'b', query, settings);

        let mut reference = CachedUnitTransitions::new(query.len(), max_cost);
        let mut reference_pool = StatePool::new();
        let expected = reference.transition_affine(
            &owned,
            &mut reference_pool,
            &policy,
            b'b',
            query,
            settings,
        );
        assert_eq!(
            actual.as_ref().map(State::positions),
            expected.as_ref().map(State::positions)
        );
    }

    #[test]
    fn checked_transition_arithmetic_helpers_handle_boundaries() {
        assert_eq!(checked_query_index(usize::MAX, 1), None);
        assert_eq!(checked_query_index(usize::MAX - 1, 1), Some(usize::MAX));

        assert_eq!(checked_window_end(1, usize::MAX, 4), Some(4));
        assert_eq!(checked_window_end(5, 1, 4), None);

        assert_eq!(transition_window_size(usize::MAX, 0), 1);
        assert_eq!(transition_window_size(2, usize::MAX), 3);

        assert_eq!(remaining_error_window(5, 2), 4);
        assert_eq!(remaining_error_window(usize::MAX, 0), usize::MAX);
        assert_eq!(remaining_error_window(2, 5), 0);

        assert_eq!(checked_position_with_offsets(usize::MAX, 1, 0, 0), None);
        assert_eq!(checked_position_with_offsets(0, 1, usize::MAX, 1), None);
        assert_eq!(checked_deletion_position(usize::MAX, 0, 0), None);
        assert_eq!(checked_deletion_position(0, 0, usize::MAX), None);
        assert!(!has_query_units(usize::MAX, 1, usize::MAX));
    }

    #[test]
    fn damerau_budget_ceiling_accepts_the_largest_representable_delta() {
        let state = initial_state(
            0,
            Algorithm::MAX_DAMERAU_DISTANCE,
            Algorithm::DamerauLevenshtein,
        );
        assert_eq!(state.len(), 1);
    }

    #[test]
    #[should_panic(expected = "Algorithm::DamerauLevenshtein supports max_distance <= 255")]
    fn damerau_budget_ceiling_rejects_incomplete_semantics() {
        let _ = transition_position(
            &Position::new(0, 0),
            &[false; 257],
            257,
            Algorithm::MAX_DAMERAU_DISTANCE + 1,
            Algorithm::DamerauLevenshtein,
            false,
        );
    }

    #[test]
    fn characteristic_vector_overflowing_offset_is_unmatched() {
        let query = b"a";
        let policy = Unrestricted;
        let mut buffer = SmallVec::<[bool; 8]>::new();

        let cv = characteristic_vector(&policy, b'a', query, 2, usize::MAX, &mut buffer);

        assert_eq!(cv, &[false, false]);
    }

    #[test]
    fn transition_standard_drops_overflowing_successor() {
        let pos = Position::new(usize::MAX, 0);
        let cv = vec![true];

        let next = transition_standard(&pos, &cv, usize::MAX, 1, false);

        assert!(next.is_empty());
    }

    #[test]
    fn transition_merge_split_rejects_overflowing_two_unit_check() {
        assert!(!has_query_units(usize::MAX - 1, 2, usize::MAX));

        let pos = Position::new(usize::MAX - 1, 0);
        let cv = vec![false, false];
        let next = transition_merge_split(&pos, &cv, usize::MAX, 1, false);

        assert!(!next.contains(&Position::new(0, 1)));
        assert!(next.contains(&Position::new(usize::MAX, 1)));
        assert!(next.contains(&Position::new(usize::MAX - 1, 1)));
    }

    #[test]
    fn test_transition_standard_match() {
        let pos = Position::new(0, 0);
        let cv = vec![true, false, false]; // Matches at position 0
        let query_length = 4; // e.g., "test"
        let next = transition_standard(&pos, &cv, query_length, 2, false);

        // Should advance with no error on match
        assert!(next.contains(&Position::new(1, 0)));
    }

    #[test]
    fn test_transition_standard_operations() {
        let pos = Position::new(1, 0);
        let cv = vec![false, false, true]; // No match at position 1
        let query_length = 4; // e.g., "test"
        let next = transition_standard(&pos, &cv, query_length, 2, false);

        // Should include:
        // - Substitution: (2, 1)
        // - Insertion: (1, 1)
        // - Deletion: (2, 1)
        assert!(next.len() >= 2);
        assert!(next.contains(&Position::new(1, 1))); // Insertion
    }

    #[test]
    fn test_initial_state() {
        let state = initial_state(5, 2, Algorithm::Standard);

        // With Standard subsumption, (0,0) subsumes both (1,1) and (2,2)
        // because |0-1|=1 <= (1-0)=1 and |0-2|=2 <= (2-0)=2
        // So only (0,0) remains in the initial state.
        //
        // This is correct: (0,0) can reach everything that (1,1) and (2,2)
        // can reach, so keeping only (0,0) is sufficient and more efficient.
        assert_eq!(state.len(), 1);
        assert!(state.positions().contains(&Position::new(0, 0)));
    }

    #[test]
    fn test_transition_state() {
        let query = b"test";
        let max_distance = 2;
        let mut state = State::new();
        state.insert(Position::new(0, 0), Algorithm::Standard, max_distance);
        let policy = Unrestricted;

        let next = transition_state(
            &state,
            policy,
            b't',
            query,
            max_distance,
            Algorithm::Standard,
            false,
        );
        assert!(next.is_some());

        let next_state = next.expect("test fixture: transition produces Some (asserted above)");
        // Should have advanced after matching 't'
        assert!(next_state.positions().iter().any(|p| p.term_index > 0));
    }

    #[test]
    fn test_zero_distance_transition_preserves_restricted_substitution() {
        let query = b"kat";
        let state = initial_state(query.len(), 0, Algorithm::Standard);

        let mut substitutions = SubstitutionSet::new();
        substitutions.allow('c', 'k');
        let policy = Restricted::new(&substitutions);

        let next = transition_state(&state, policy, b'c', query, 0, Algorithm::Standard, false)
            .expect("restricted c->k substitution should advance at distance zero");

        assert!(next.positions().contains(&Position::new(1, 0)));

        let mut pool = StatePool::new();
        let pooled = transition_state_pooled(
            &state,
            &mut pool,
            policy,
            b'c',
            query,
            TransitionSettings::new(0, Algorithm::Standard, false),
        )
        .expect("pooled restricted c->k substitution should advance at distance zero");

        assert_eq!(pooled.positions(), next.positions());

        let mut pool = StatePool::new();
        let borrowed = transition_state_pooled_ref(
            &state,
            &mut pool,
            &policy,
            b'c',
            query,
            TransitionSettings::new(0, Algorithm::Standard, false),
        )
        .expect("borrowed-policy pooled restricted c->k substitution should advance");

        assert_eq!(borrowed.positions(), next.positions());
    }

    #[test]
    fn test_zero_distance_transition_rejects_unmatched_unit() {
        let query = b"kat";
        let state = initial_state(query.len(), 0, Algorithm::Standard);
        let policy = Unrestricted;

        assert!(
            transition_state(&state, policy, b'c', query, 0, Algorithm::Standard, false).is_none()
        );
    }

    #[test]
    fn test_zero_distance_transition_preserves_prefix_mode_after_query() {
        let query = b"cat";
        let state = State::single(Position::new(query.len(), 0));
        let policy = Unrestricted;

        let next = transition_state(&state, policy, b's', query, 0, Algorithm::Standard, true)
            .expect("prefix mode should keep accepting after query is consumed");

        assert_eq!(next.positions(), &[Position::new(query.len(), 0)]);

        assert!(
            transition_state(&state, policy, b's', query, 0, Algorithm::Standard, false).is_none()
        );
    }

    #[test]
    fn test_zero_distance_transition_falls_back_for_special_positions() {
        let query = b"a";
        let state = State::single(Position::new_osa_transposing(0, 0));
        let policy = Unrestricted;

        let next = transition_state(
            &state,
            policy,
            b'x',
            query,
            0,
            Algorithm::MergeAndSplit,
            false,
        )
        .expect("special merge/split state should use the general transition path");

        assert_eq!(next.positions(), &[Position::new(1, 0)]);
    }
}
