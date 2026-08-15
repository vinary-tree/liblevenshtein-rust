//! Prefix-shared, subsequence-constrained dictionary DFS.
//!
//! The query must occur in order in a yielded dictionary term. Dictionary
//! units between query matches are skipped without a bound. This is a
//! structural traversal only: scoring and ranking belong to a caller-supplied
//! [`PrefixPruner`](super::PrefixPruner).

use super::{NoPruning, PrefixPruner};
use libdictenstein::{CharUnit, Dictionary, DictionaryNode};

/// Work counters for a subsequence traversal.
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub struct SubsequenceQueryStats {
    /// Nodes whose finality and outgoing edges were inspected.
    pub nodes_visited: usize,
    /// Outgoing edges considered before external prefix pruning.
    pub edges_enumerated: usize,
    /// Edges rejected with their entire subtree by the external pruner.
    pub subtrees_pruned: usize,
}

/// One dictionary term containing the query as a subsequence.
#[derive(Clone, Debug, PartialEq)]
pub struct SubsequenceMatch<U: CharUnit> {
    /// Raw dictionary units in root-to-leaf order.
    pub units: Vec<U>,
    /// Optional score reported by the prefix visitor.
    pub score: Option<f64>,
}

struct Frame<N: DictionaryNode> {
    edges: std::vec::IntoIter<(N::Unit, N)>,
    matched: usize,
    entered_by: Option<N::Unit>,
    is_final: bool,
    final_checked: bool,
}

impl<N: DictionaryNode> Frame<N> {
    fn root(node: N) -> Self {
        let mut edges = Vec::with_capacity(node.edge_count().unwrap_or(0));
        let is_final = node.visit_edges_and_finality(|label, child| edges.push((label, child)));
        Self {
            edges: edges.into_iter(),
            matched: 0,
            entered_by: None,
            is_final,
            final_checked: false,
        }
    }

    fn child(node: N, entered_by: N::Unit, matched: usize) -> Self {
        let mut edges = Vec::with_capacity(node.edge_count().unwrap_or(0));
        let is_final = node.visit_edges_and_finality(|label, child| edges.push((label, child)));
        Self {
            edges: edges.into_iter(),
            matched,
            entered_by: Some(entered_by),
            is_final,
            final_checked: false,
        }
    }
}

/// Lazy explicit-stack DFS for dictionary terms containing a query subsequence.
pub struct SubsequenceQueryIterator<N, P = NoPruning>
where
    N: DictionaryNode,
    P: PrefixPruner<N::Unit>,
{
    query: Vec<N::Unit>,
    stack: Vec<Frame<N>>,
    prefix: Vec<N::Unit>,
    pruner: Option<P>,
    stats: SubsequenceQueryStats,
}

impl<N> SubsequenceQueryIterator<N, NoPruning>
where
    N: DictionaryNode,
{
    /// Traverse from `root` without external subtree pruning.
    pub fn new(root: N, query: Vec<N::Unit>) -> Self {
        Self::with_pruner(root, query, NoPruning)
    }

    /// Construct from any compatible dictionary without external pruning.
    pub fn from_dictionary<D>(dictionary: &D, query: Vec<N::Unit>) -> Self
    where
        D: Dictionary<Node = N>,
    {
        Self::new(dictionary.root(), query)
    }
}

impl<N, P> SubsequenceQueryIterator<N, P>
where
    N: DictionaryNode,
    P: PrefixPruner<N::Unit>,
{
    /// Traverse from `root` with a stateful, balanced prefix visitor.
    pub fn with_pruner(root: N, query: Vec<N::Unit>, pruner: P) -> Self {
        Self {
            query,
            stack: vec![Frame::root(root)],
            prefix: Vec::new(),
            pruner: Some(pruner),
            stats: SubsequenceQueryStats {
                nodes_visited: 1,
                ..SubsequenceQueryStats::default()
            },
        }
    }

    /// Snapshot traversal counters.
    pub fn stats(&self) -> SubsequenceQueryStats {
        self.stats
    }

    /// Borrow the visitor, for example to inspect scorer counters.
    pub fn pruner(&self) -> &P {
        self.pruner
            .as_ref()
            .expect("the prefix pruner is present until the iterator is consumed")
    }

    /// Consume the iterator and return a balanced visitor.
    pub fn into_pruner(mut self) -> P {
        self.unwind();
        self.pruner
            .take()
            .expect("the prefix pruner is present until it is extracted")
    }

    fn pruner_mut(&mut self) -> &mut P {
        self.pruner
            .as_mut()
            .expect("the prefix pruner is present while iteration is active")
    }

    fn unwind(&mut self) {
        while let Some(frame) = self.stack.pop() {
            if let Some(unit) = frame.entered_by {
                let depth = self.prefix.len();
                debug_assert_eq!(self.prefix.pop(), Some(unit));
                if let Some(pruner) = self.pruner.as_mut() {
                    pruner.leave(unit, depth);
                }
            }
        }
    }
}

impl<N, P> Iterator for SubsequenceQueryIterator<N, P>
where
    N: DictionaryNode,
    N::Unit: CharUnit,
    P: PrefixPruner<N::Unit>,
{
    type Item = SubsequenceMatch<N::Unit>;

    fn next(&mut self) -> Option<Self::Item> {
        loop {
            self.stack.last()?;

            let accepts_subsequence = {
                let frame = self
                    .stack
                    .last_mut()
                    .expect("the DFS stack was observed non-empty");
                if frame.final_checked {
                    false
                } else {
                    frame.final_checked = true;
                    frame.is_final && frame.matched == self.query.len()
                }
            };
            if accepts_subsequence {
                let prefix = self.prefix.clone();
                if self.pruner_mut().permits_accept(&prefix) {
                    let score = self.pruner_mut().accept(&prefix);
                    return Some(SubsequenceMatch {
                        units: prefix,
                        score,
                    });
                }
            }

            let edge = {
                let frame = self
                    .stack
                    .last_mut()
                    .expect("the DFS stack was observed non-empty");
                frame
                    .edges
                    .next()
                    .map(|(unit, child)| (unit, child, frame.matched))
            };
            if let Some((unit, child, parent_matched)) = edge {
                self.stats.edges_enumerated = self.stats.edges_enumerated.saturating_add(1);
                let depth = self.prefix.len().saturating_add(1);
                if !self.pruner_mut().enter(unit, depth) {
                    self.stats.subtrees_pruned = self.stats.subtrees_pruned.saturating_add(1);
                    self.pruner_mut().leave(unit, depth);
                    continue;
                }

                let matches_next = parent_matched < self.query.len()
                    && self
                        .pruner
                        .as_ref()
                        .expect("the prefix pruner is present while iteration is active")
                        .matches_query_unit(unit, self.query[parent_matched]);
                let matched = parent_matched + usize::from(matches_next);
                self.prefix.push(unit);
                self.stack.push(Frame::child(child, unit, matched));
                self.stats.nodes_visited = self.stats.nodes_visited.saturating_add(1);
                continue;
            }

            let frame = self.stack.pop().expect("the stack was observed non-empty");
            if let Some(unit) = frame.entered_by {
                let depth = self.prefix.len();
                debug_assert_eq!(self.prefix.pop(), Some(unit));
                self.pruner_mut().leave(unit, depth);
            }
        }
    }
}

impl<N, P> Drop for SubsequenceQueryIterator<N, P>
where
    N: DictionaryNode,
    P: PrefixPruner<N::Unit>,
{
    fn drop(&mut self) {
        self.unwind();
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::transducer::AllowedPrefixes;
    use libdictenstein::double_array_trie::DoubleArrayTrie;

    #[test]
    fn subsequence_boundaries_and_pruning_are_exact() {
        let dictionary = DoubleArrayTrie::from_terms(["", "abc", "axbyc", "acb", "zzz"]);
        let all: Vec<_> = SubsequenceQueryIterator::from_dictionary(&dictionary, b"abc".to_vec())
            .map(|item| String::from_utf8(item.units).unwrap())
            .collect();
        assert_eq!(all, ["abc", "axbyc"]);

        let pruner = AllowedPrefixes::new([b"axbyc".as_slice()]);
        let mut pruned =
            SubsequenceQueryIterator::with_pruner(dictionary.root(), b"abc".to_vec(), pruner);
        assert_eq!(
            pruned
                .by_ref()
                .map(|item| String::from_utf8(item.units).unwrap())
                .collect::<Vec<_>>(),
            ["axbyc"]
        );
        assert!(pruned.stats().subtrees_pruned > 0);
    }

    #[test]
    fn empty_query_accepts_every_final_and_dfs_events_balance() {
        let dictionary = DoubleArrayTrie::from_terms(["", "a", "ab"]);
        let matches: Vec<_> = SubsequenceQueryIterator::from_dictionary(&dictionary, Vec::new())
            .map(|item| item.units)
            .collect();
        assert_eq!(matches, [b"".to_vec(), b"a".to_vec(), b"ab".to_vec()]);
    }

    #[derive(Default)]
    struct AsciiCaseInsensitive;

    impl PrefixPruner<u8> for AsciiCaseInsensitive {
        fn matches_query_unit(&self, candidate: u8, query: u8) -> bool {
            candidate.eq_ignore_ascii_case(&query)
        }

        fn enter(&mut self, _unit: u8, _depth: usize) -> bool {
            true
        }

        fn leave(&mut self, _unit: u8, _depth: usize) {}
    }

    #[test]
    fn structural_matching_relation_is_owned_by_the_visitor() {
        let dictionary = DoubleArrayTrie::from_terms(["FooBar", "far", "bar"]);
        let matches: Vec<_> = SubsequenceQueryIterator::with_pruner(
            dictionary.root(),
            b"fb".to_vec(),
            AsciiCaseInsensitive,
        )
        .map(|item| String::from_utf8(item.units).expect("dictionary terms are UTF-8"))
        .collect();

        assert_eq!(matches, ["FooBar"]);
    }
}
