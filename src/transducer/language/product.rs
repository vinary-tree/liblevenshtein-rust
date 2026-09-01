//! Standard-Levenshtein distance to a regular language.

use super::LanguageAutomaton;
use smallvec::SmallVec;
use std::marker::PhantomData;

/// A cost-indexed language frontier.
///
/// `levels[e]` is the union of language states reachable at exactly edit cost
/// `e`, after canonicalization removes states already reachable more cheaply.
/// Consequently the number of levels is always at most `max_distance + 1`,
/// regardless of input history or language nondeterminism.
#[derive(Clone, Debug, PartialEq, Eq, Hash)]
pub struct Frontier<S> {
    levels: SmallVec<[Option<S>; 4]>,
}

impl<S> Frontier<S> {
    /// Number of cost levels. Empty levels still count toward the configured
    /// budget and preserve the stable `level == cost` mapping.
    pub fn len(&self) -> usize {
        self.levels.len()
    }

    /// Whether every cost level is empty.
    pub fn is_empty(&self) -> bool {
        self.levels.iter().all(Option::is_none)
    }

    /// Inspect the state set at exact cost `level`.
    pub fn level(&self, level: usize) -> Option<&S> {
        self.levels.get(level).and_then(Option::as_ref)
    }
}

/// Product of a finite-language automaton and unit-cost Levenshtein edits.
#[derive(Clone, Debug)]
pub struct LanguageProduct<U, L> {
    language: L,
    max_distance: u8,
    _unit: PhantomData<fn(U)>,
}

impl<U, L> LanguageProduct<U, L>
where
    L: LanguageAutomaton<U>,
{
    /// Construct a product bounded by `max_distance` edits.
    pub fn new(language: L, max_distance: u8) -> Self {
        Self {
            language,
            max_distance,
            _unit: PhantomData,
        }
    }

    /// Borrow the underlying language automaton.
    pub fn language(&self) -> &L {
        &self.language
    }

    /// Maximum edit distance.
    pub fn max_distance(&self) -> u8 {
        self.max_distance
    }

    fn empty_frontier(&self) -> Frontier<L::StateSet> {
        Frontier {
            levels: (0..=self.max_distance).map(|_| None).collect(),
        }
    }

    fn union_level(
        &self,
        frontier: &mut Frontier<L::StateSet>,
        level: usize,
        states: &L::StateSet,
    ) {
        if self.language.is_empty(states) {
            return;
        }
        match &mut frontier.levels[level] {
            Some(existing) => self.language.union_into(existing, states),
            slot @ None => *slot = Some(states.clone()),
        }
    }

    fn canonicalize(&self, frontier: &mut Frontier<L::StateSet>) {
        let mut covered = self.language.empty();
        for level in &mut frontier.levels {
            let Some(states) = level else {
                continue;
            };
            self.language.subtract(states, &covered);
            if self.language.is_empty(states) {
                *level = None;
            } else {
                self.language.union_into(&mut covered, states);
            }
        }
    }

    fn deletion_closure(&self, frontier: &mut Frontier<L::StateSet>) {
        for level in 0..usize::from(self.max_distance) {
            let Some(states) = frontier.levels[level].clone() else {
                continue;
            };
            let advanced = self.language.advance(&states);
            self.union_level(frontier, level + 1, &advanced);
            self.canonicalize(frontier);
        }
    }

    /// Initial frontier, including all pattern deletions within budget.
    pub fn initial_frontier(&self) -> Frontier<L::StateSet> {
        let mut frontier = self.empty_frontier();
        let initial = self.language.initial();
        self.union_level(&mut frontier, 0, &initial);
        self.deletion_closure(&mut frontier);
        frontier
    }

    /// Advance a frontier by one dictionary/input unit.
    pub fn step(&self, frontier: &Frontier<L::StateSet>, unit: &U) -> Frontier<L::StateSet> {
        let mut next = self.empty_frontier();
        for level in 0..frontier.levels.len() {
            let Some(states) = frontier.levels[level].as_ref() else {
                continue;
            };

            let matched = self.language.step(states, unit);
            self.union_level(&mut next, level, &matched);

            if level < usize::from(self.max_distance) {
                // Insertion consumes the input unit without moving the language.
                self.union_level(&mut next, level + 1, states);
                // Substitution consumes any one language symbol.
                let substituted = self.language.advance(states);
                self.union_level(&mut next, level + 1, &substituted);
            }
        }
        self.canonicalize(&mut next);
        self.deletion_closure(&mut next);
        next
    }

    /// Merge two frontiers by language-state union at each cost level.
    pub fn merge(
        &self,
        left: &Frontier<L::StateSet>,
        right: &Frontier<L::StateSet>,
    ) -> Frontier<L::StateSet> {
        let mut merged = self.empty_frontier();
        for level in 0..merged.levels.len() {
            if let Some(states) = left.levels[level].as_ref() {
                self.union_level(&mut merged, level, states);
            }
            if let Some(states) = right.levels[level].as_ref() {
                self.union_level(&mut merged, level, states);
            }
        }
        self.canonicalize(&mut merged);
        merged
    }

    /// Least accepting cost represented by `frontier`.
    pub fn min_accepting_distance(&self, frontier: &Frontier<L::StateSet>) -> Option<u8> {
        frontier
            .levels
            .iter()
            .enumerate()
            .find_map(|(cost, states)| {
                states
                    .as_ref()
                    .filter(|states| self.language.is_accepting(states))
                    .and_then(|_| u8::try_from(cost).ok())
            })
    }

    /// Exact bounded distance from a unit sequence to the recognized language.
    pub fn distance_to_language<I>(&self, input: I) -> Option<u8>
    where
        I: IntoIterator<Item = U>,
    {
        let mut frontier = self.initial_frontier();
        for unit in input {
            frontier = self.step(&frontier, &unit);
            if frontier.is_empty() {
                return None;
            }
        }
        self.min_accepting_distance(&frontier)
    }

    /// Whether the sequence is within the configured distance of the language.
    pub fn accepts<I>(&self, input: I) -> bool
    where
        I: IntoIterator<Item = U>,
    {
        self.distance_to_language(input).is_some()
    }
}
