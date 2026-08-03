//! Stateful subtree-pruning visitors for depth-first dictionary traversals.
//!
//! A prefix pruner is deliberately a balanced DFS visitor rather than a
//! predicate. Incremental scorers commonly push one dynamic-programming column
//! in [`PrefixPruner::enter`] and pop it in [`PrefixPruner::leave`]. A breadth-
//! first traversal cannot honor that stack discipline without copying or
//! replaying scorer state for every queued path.

use libdictenstein::CharUnit;
use rustc_hash::FxHashSet;
use std::hash::Hash;

/// A balanced visitor that may reject an entire dictionary subtree.
///
/// `enter` and `leave` are paired even when `enter` returns `false`: the
/// traversal immediately calls `leave` before moving to the next sibling. This
/// lets implementations unconditionally push state before deciding whether the
/// resulting prefix can still produce a useful completion.
pub trait PrefixPruner<U: CharUnit> {
    /// Compare a dictionary unit with the next structural query unit.
    ///
    /// The default is exact equality. Scorers with a different matching
    /// relation, such as case-insensitive fzf matching, may override this so
    /// the structural subsequence walk and the scorer cannot disagree.
    fn matches_query_unit(&self, candidate: U, query: U) -> bool {
        candidate == query
    }

    /// Enter the child reached by `unit` at one-based `depth`.
    ///
    /// Return `false` to skip that child and its complete subtree.
    fn enter(&mut self, unit: U, depth: usize) -> bool;

    /// Leave the child previously reported to [`enter`](Self::enter).
    fn leave(&mut self, unit: U, depth: usize);

    /// Whether a structurally accepted final prefix belongs to the pruner's
    /// terminal set. This is distinct from [`accept`](Self::accept), whose
    /// `Option` represents the presence of a score rather than membership.
    fn permits_accept(&mut self, prefix: &[U]) -> bool {
        let _ = prefix;
        true
    }

    /// Observe an accepted dictionary prefix and optionally attach a score.
    ///
    /// Returning `None` does not reject the match; it means that the structural
    /// visitor has no score to report.
    fn accept(&mut self, prefix: &[U]) -> Option<f64> {
        let _ = prefix;
        None
    }
}

/// Zero-sized default that never prunes and never scores.
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub struct NoPruning;

impl<U: CharUnit> PrefixPruner<U> for NoPruning {
    #[inline(always)]
    fn enter(&mut self, _unit: U, _depth: usize) -> bool {
        true
    }

    #[inline(always)]
    fn leave(&mut self, _unit: U, _depth: usize) {}
}

/// Exact prefix set produced by a conservative source filter.
///
/// Every prefix of every admitted term is retained. Therefore rejecting a
/// missing prefix cannot remove a term admitted by the source filter. This is
/// useful for threading an [`NgramIndex`](crate::filter::NgramIndex) or
/// [`HybridMatcher`](crate::filter::HybridMatcher) candidate set *inside* a
/// dictionary DFS rather than filtering materialized matches afterward.
#[derive(Clone, Debug)]
pub struct AllowedPrefixes<U>
where
    U: CharUnit + Eq + Hash,
{
    prefixes: FxHashSet<Vec<U>>,
    terms: FxHashSet<Vec<U>>,
    current: Vec<U>,
}

impl<U> AllowedPrefixes<U>
where
    U: CharUnit + Eq + Hash,
{
    /// Build the downward-closed prefix set for `terms`.
    pub fn new<I, T>(terms: I) -> Self
    where
        I: IntoIterator<Item = T>,
        T: AsRef<[U]>,
    {
        let terms: Vec<Vec<U>> = terms
            .into_iter()
            .map(|term| term.as_ref().to_vec())
            .collect();
        let mut prefixes = FxHashSet::default();
        prefixes.insert(Vec::new());
        for term in &terms {
            for length in 1..=term.len() {
                prefixes.insert(term[..length].to_vec());
            }
        }
        Self {
            prefixes,
            terms: terms.into_iter().collect(),
            current: Vec::new(),
        }
    }

    /// Number of distinct retained prefixes, including the empty prefix.
    pub fn len(&self) -> usize {
        self.prefixes.len()
    }

    /// Whether only the empty prefix is retained.
    pub fn is_empty(&self) -> bool {
        self.prefixes.len() == 1
    }
}

impl<U> PrefixPruner<U> for AllowedPrefixes<U>
where
    U: CharUnit + Eq + Hash,
{
    fn enter(&mut self, unit: U, depth: usize) -> bool {
        debug_assert_eq!(depth, self.current.len() + 1);
        self.current.push(unit);
        self.prefixes.contains(&self.current)
    }

    fn leave(&mut self, unit: U, depth: usize) {
        debug_assert_eq!(depth, self.current.len());
        debug_assert_eq!(self.current.last().copied(), Some(unit));
        self.current.pop();
    }

    fn permits_accept(&mut self, prefix: &[U]) -> bool {
        self.terms.contains(prefix)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn no_pruning_is_zero_sized() {
        assert_eq!(std::mem::size_of::<NoPruning>(), 0);
    }

    #[test]
    fn allowed_prefixes_are_downward_closed_and_balanced() {
        let mut pruner = AllowedPrefixes::new([b"alpha".as_slice(), b"alpine".as_slice()]);
        assert!(pruner.enter(b'a', 1));
        assert!(pruner.enter(b'l', 2));
        assert!(!pruner.enter(b'z', 3));
        pruner.leave(b'z', 3);
        assert!(pruner.enter(b'p', 3));
        pruner.leave(b'p', 3);
        pruner.leave(b'l', 2);
        pruner.leave(b'a', 1);
        assert!(pruner.current.is_empty());
    }
}
