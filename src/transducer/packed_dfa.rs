//! Shared lazy deterministic core for exact-label packed automata.

use libdictenstein::CharUnit;
use rustc_hash::FxHashMap;
use std::hash::Hash;
#[cfg(feature = "resource-profiling")]
use std::sync::OnceLock;

const DIRECT_LABEL_SLOTS: usize = 256;
const DFA_TARGET_DEAD: u32 = u32::MAX;
const DFA_TARGET_UNCOMPUTED: u32 = u32::MAX - 1;

#[cfg(feature = "resource-profiling")]
#[inline(always)]
fn class_zero_row_cache_enabled() -> bool {
    static ENABLED: OnceLock<bool> = OnceLock::new();
    *ENABLED.get_or_init(|| {
        std::env::var_os("LIBLEVENSHTEIN_CAUSAL_DISABLE_CLASS_ZERO_ROW_CACHE").is_none()
    })
}

/// Proof that one query-local DFA state owns a complete row in `targets`.
///
/// Dictionary expansion fixes the source state while it visits every sibling
/// edge. Carrying this copyable row descriptor lets that loop derive the row
/// base once instead of repeating state conversion and multiplication for
/// every edge. It borrows no storage, so lazy miss-path interning may still
/// grow the DFA vectors safely.
#[derive(Clone, Copy, Debug)]
pub(crate) struct ExactLabelDfaRow {
    frontier_index: usize,
    target_start: usize,
    /// Encoded class-zero result after the first sibling label absent from the
    /// query. `DFA_TARGET_DEAD` is a valid cached result; `None` alone means the
    /// row has not probed class zero yet.
    class_zero_target: Option<u32>,
}

/// Query-local equivalence classes for exact matching.
///
/// Class zero denotes every unit absent from the query. Every distinct query
/// unit receives one non-zero class whose payload is the repeated packed match
/// mask for that unit. Dense byte-range units use one array load; arbitrary
/// Unicode scalars and `u64` tokens retain the generic hash-map fallback.
#[derive(Debug)]
struct ExactLabelClasses<U: CharUnit> {
    direct: [u8; DIRECT_LABEL_SLOTS],
    overflow: FxHashMap<U, u8>,
    repeated_masks: Box<[u64]>,
}

impl<U: CharUnit> ExactLabelClasses<U> {
    fn new(query: &[U], lane_starts: u64) -> Self {
        let mut direct = [0u8; DIRECT_LABEL_SLOTS];
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

/// Lazy deterministic exact-label transition table shared by packed kernels.
///
/// The generic frontier remains owned by the specialized automaton. The
/// recurrence closure is generic at each call site, so the compiler
/// monomorphizes it into the miss path without a function pointer or vtable.
#[derive(Debug)]
pub(super) struct ExactLabelDfa<U, F>
where
    U: CharUnit,
    F: Copy + Eq + Hash,
{
    classes: ExactLabelClasses<U>,
    frontiers: Vec<F>,
    frontier_ids: FxHashMap<F, u32>,
    targets: Vec<u32>,
    #[cfg(feature = "resource-profiling")]
    class_zero_row_cache: bool,
}

impl<U, F> ExactLabelDfa<U, F>
where
    U: CharUnit,
    F: Copy + Eq + Hash,
{
    pub(super) fn new(query: &[U], lane_starts: u64, seed: F) -> Self {
        let classes = ExactLabelClasses::new(query, lane_starts);
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
            #[cfg(feature = "resource-profiling")]
            class_zero_row_cache: class_zero_row_cache_enabled(),
        }
    }

    #[inline(always)]
    fn uses_class_zero_row_cache(&self) -> bool {
        #[cfg(feature = "resource-profiling")]
        {
            self.class_zero_row_cache
        }
        #[cfg(not(feature = "resource-profiling"))]
        {
            true
        }
    }

    #[inline(always)]
    pub(super) fn seed(&self) -> u64 {
        0
    }

    #[inline(always)]
    pub(super) fn frontier(&self, state: u64) -> F {
        self.frontiers[usize::try_from(state).expect("packed DFA state exceeds usize")]
    }

    #[inline(always)]
    pub(super) fn prepare_row(&self, state: u64) -> ExactLabelDfaRow {
        let frontier_index = usize::try_from(state).expect("packed DFA state exceeds usize");
        let target_start = frontier_index
            .checked_mul(self.classes.class_count())
            .expect("packed DFA transition row overflow");
        debug_assert!(target_start < self.targets.len());
        ExactLabelDfaRow {
            frontier_index,
            target_start,
            class_zero_target: None,
        }
    }

    /// Classify one source-row label for untimed reuse diagnostics.
    #[cfg(feature = "perf-instrumentation")]
    #[inline(always)]
    pub(super) fn source_row_label_is_class_zero(&self, label: U) -> bool {
        self.classes.class_for(label) == 0
    }

    #[inline(always)]
    pub(super) fn step<R>(&mut self, state: u64, label: U, recurrence: R) -> Option<u64>
    where
        R: FnOnce(F, u64) -> Option<F>,
    {
        let mut row = self.prepare_row(state);
        self.step_in_row(&mut row, label, recurrence)
    }

    #[inline(always)]
    pub(super) fn step_in_row<R>(
        &mut self,
        row: &mut ExactLabelDfaRow,
        label: U,
        recurrence: R,
    ) -> Option<u64>
    where
        R: FnOnce(F, u64) -> Option<F>,
    {
        let class = self.classes.class_for(label);
        if class == 0 && self.uses_class_zero_row_cache() {
            if let Some(cached) = row.class_zero_target {
                return (cached != DFA_TARGET_DEAD).then_some(u64::from(cached));
            }
        }
        let cell = row.target_start + class;
        crate::causal_perf::record_packed_dfa_physical_target_probes(1);
        let cached = self.targets[cell];
        if cached != DFA_TARGET_UNCOMPUTED {
            crate::causal_perf::record_packed_dfa_transition_hits(1);
            if class == 0 && self.uses_class_zero_row_cache() {
                row.class_zero_target = Some(cached);
            }
            return (cached != DFA_TARGET_DEAD).then_some(u64::from(cached));
        }

        crate::causal_perf::record_packed_dfa_transition_misses(1);
        let source = self.frontiers[row.frontier_index];
        let repeated_matches = self.classes.repeated_mask(class);
        let target = recurrence(source, repeated_matches);
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
        if class == 0 && self.uses_class_zero_row_cache() {
            row.class_zero_target = Some(encoded);
        }
        (encoded != DFA_TARGET_DEAD).then_some(u64::from(encoded))
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::hash::{Hash, Hasher};

    #[derive(Clone, Copy, Debug, Eq, PartialEq)]
    struct CollidingFrontier(u8);

    impl Hash for CollidingFrontier {
        fn hash<H: Hasher>(&self, state: &mut H) {
            0u8.hash(state);
        }
    }

    #[test]
    fn exact_classes_keep_dense_and_overflow_units_distinct() {
        let chars = ExactLabelClasses::new(&['\0', '\u{0100}', '\0'], 1);
        let dense = chars.class_for('\0');
        let overflow = chars.class_for('\u{0100}');
        assert_ne!(dense, overflow);
        assert_ne!(dense, 0);
        assert_ne!(overflow, 0);
        assert_eq!(chars.class_for('x'), 0);
        assert_eq!(chars.repeated_mask(dense), 0b101);
        assert_eq!(chars.repeated_mask(overflow), 0b010);

        let high = 1u64 << 32;
        let units = ExactLabelClasses::new(&[0, high, 0], 1);
        let dense = units.class_for(0);
        let overflow = units.class_for(high);
        assert_ne!(dense, overflow);
        assert_eq!(units.class_for(u64::MAX), 0);
        assert_eq!(units.repeated_mask(dense), 0b101);
        assert_eq!(units.repeated_mask(overflow), 0b010);
    }

    #[test]
    fn frontier_hash_collisions_do_not_alias_dense_state_ids() {
        let mut dfa = ExactLabelDfa::new(b"ab", 1, CollidingFrontier(0));
        let left = dfa
            .step(0, b'a', |_, _| Some(CollidingFrontier(1)))
            .expect("first frontier remains live");
        let right = dfa
            .step(0, b'b', |_, _| Some(CollidingFrontier(2)))
            .expect("second frontier remains live");
        assert_ne!(left, right);
        assert_eq!(dfa.frontier(left), CollidingFrontier(1));
        assert_eq!(dfa.frontier(right), CollidingFrontier(2));

        let mut source_row = dfa.prepare_row(left);
        let reused = dfa
            .step_in_row(&mut source_row, b'b', |_, _| Some(CollidingFrontier(2)))
            .expect("existing frontier is interned exactly");
        assert_eq!(reused, right);
    }

    #[test]
    fn prepared_rows_remain_valid_when_lazy_interning_grows_storage() {
        let mut dfa = ExactLabelDfa::new(b"abc", 1, 0u64);
        let mut root_row = dfa.prepare_row(0);

        // Grow every backing vector after capturing the copyable row. The row
        // contains only stable indices, never references into reallocating
        // storage.
        for state in 0..64u64 {
            let label = b"abc"[usize::try_from(state % 3).unwrap()];
            let frontier = state + 1;
            assert_eq!(
                dfa.step(state, label, |_, _| Some(frontier)),
                Some(frontier)
            );
        }

        let target = dfa
            .step_in_row(&mut root_row, b'x', |source, repeated| {
                assert_eq!(source, 0);
                assert_eq!(repeated, 0);
                Some(1_000)
            })
            .expect("class-zero transition remains live");
        assert_eq!(dfa.frontier(target), 1_000);
        assert_eq!(
            dfa.step_in_row(&mut root_row, b'x', |_, _| unreachable!(
                "cached transition"
            )),
            Some(target)
        );
    }
}
