//! Query iterator using zipper-based traversal.
//!
//! A zipper focus is adapted to the same compact, statically dispatched product
//! scheduler used by ordinary dictionary queries. Backends may project labels
//! before constructing accepted child foci, so an automaton-pruned edge pays no
//! zipper path allocation. Results remain relative to the supplied focus.

use crate::transducer::{Algorithm, Candidate, QueryIterator};
use libdictenstein::zipper::DictZipper;
use libdictenstein::ZipperTraversalNode;

/// Query iterator using zipper-based BFS traversal.
///
/// This iterator composes a dictionary zipper with the production compact
/// query machine and traverses reachable product focuses in breadth-first
/// order.
///
/// # Type Parameters
///
/// - `D`: Dictionary zipper type (must have `Unit = u8`)
///
/// # Performance
///
/// - **Queue item**: one opaque dictionary focus, one parent-arena cursor, and
///   one eight-byte query frontier
/// - **Search memory**: reached canonical states + observed transitions + live
///   BFS breadth + result paths; the queue never multiplies frontier width
/// - **Time**: proportional to inspected reachable edges plus first-observation
///   transition construction; worst case still inspects the dictionary
///
/// # Examples
///
/// ```rust
/// # #[cfg(feature = "pathmap-backend")]
/// # {
/// # // Note: This example requires the 'pathmap-backend' feature
/// use liblevenshtein::dictionary::pathmap::PathMapDictionary;
/// use liblevenshtein::dictionary::pathmap_zipper::PathMapZipper;
/// use liblevenshtein::transducer::{Algorithm, ZipperQueryIterator};
///
/// let dict = PathMapDictionary::<()>::new();
/// dict.insert("cat");
/// dict.insert("dog");
/// dict.insert("car");
///
/// let dict_zipper = PathMapZipper::new_from_dict(&dict);
///
/// let iter = ZipperQueryIterator::new(
///     dict_zipper,
///     "cat",
///     1,
///     Algorithm::Standard
/// );
///
/// let results: Vec<_> = iter.collect();
/// assert!(results.iter().any(|c| c.term == "cat"));
/// assert!(results.iter().any(|c| c.term == "car"));
/// # }
/// ```
pub struct ZipperQueryIterator<D>
where
    D: DictZipper<Unit = u8> + Send + Sync,
{
    inner: QueryIterator<ZipperTraversalNode<D>, Candidate>,
}

impl<D> ZipperQueryIterator<D>
where
    D: DictZipper<Unit = u8> + Send + Sync,
{
    /// Create a new zipper-based query iterator.
    ///
    /// # Arguments
    ///
    /// * `dict_zipper` - Dictionary zipper at root position
    /// * `query` - Query string to match against
    /// * `max_distance` - Maximum edit distance threshold
    /// * `algorithm` - Levenshtein algorithm variant
    ///
    /// # Examples
    ///
    /// ```rust
    /// # #[cfg(feature = "pathmap-backend")]
    /// # {
    /// # // Note: This example requires the 'pathmap-backend' feature
    /// use liblevenshtein::dictionary::pathmap::PathMapDictionary;
    /// use liblevenshtein::dictionary::pathmap_zipper::PathMapZipper;
    /// use liblevenshtein::transducer::{Algorithm, ZipperQueryIterator};
    ///
    /// let dict = PathMapDictionary::<()>::new();
    /// dict.insert("test");
    ///
    /// let dict_zipper = PathMapZipper::new_from_dict(&dict);
    /// let iter = ZipperQueryIterator::new(
    ///     dict_zipper,
    ///     "test",
    ///     1,
    ///     Algorithm::Standard
    /// );
    ///
    /// let results: Vec<_> = iter.collect();
    /// assert_eq!(results.len(), 1);
    /// assert_eq!(results[0].term, "test");
    /// # }
    /// ```
    pub fn new(dict_zipper: D, query: &str, max_distance: usize, algorithm: Algorithm) -> Self {
        Self {
            inner: QueryIterator::new(
                dict_zipper.into_traversal_node(),
                query.to_owned(),
                max_distance,
                algorithm,
            ),
        }
    }
}

impl<D> Iterator for ZipperQueryIterator<D>
where
    D: DictZipper<Unit = u8> + Send + Sync,
{
    type Item = Candidate;

    #[inline]
    fn next(&mut self) -> Option<Self::Item> {
        self.inner.next()
    }

    fn size_hint(&self) -> (usize, Option<usize>) {
        (0, None)
    }
}

impl<D> std::iter::FusedIterator for ZipperQueryIterator<D> where
    D: DictZipper<Unit = u8> + Send + Sync
{
}

#[cfg(all(test, feature = "pathmap-backend"))]
mod tests {
    use super::*;
    use crate::distance::{
        create_memo_cache, damerau_levenshtein_distance, merge_and_split_distance,
        standard_distance, transposition_distance,
    };
    use libdictenstein::pathmap::zipper::PathMapZipper;
    use libdictenstein::pathmap::PathMapDictionary;
    use libdictenstein::Dictionary;
    use proptest::prelude::*;
    use std::sync::atomic::{AtomicUsize, Ordering};
    use std::sync::Arc;

    #[derive(Clone)]
    struct CountingWideZipper {
        label: Option<u8>,
        children_constructed: Arc<AtomicUsize>,
    }

    impl CountingWideZipper {
        fn root(children_constructed: Arc<AtomicUsize>) -> Self {
            Self {
                label: None,
                children_constructed,
            }
        }

        fn child(&self, label: u8) -> Self {
            self.children_constructed.fetch_add(1, Ordering::Relaxed);
            Self {
                label: Some(label),
                children_constructed: Arc::clone(&self.children_constructed),
            }
        }
    }

    impl DictZipper for CountingWideZipper {
        type Unit = u8;

        fn is_final(&self) -> bool {
            self.label.is_some()
        }

        fn descend(&self, label: Self::Unit) -> Option<Self> {
            self.label.is_none().then(|| self.child(label))
        }

        fn children(&self) -> impl Iterator<Item = (Self::Unit, Self)> {
            let children = if self.label.is_none() {
                (u8::MIN..=u8::MAX)
                    .map(|label| (label, self.child(label)))
                    .collect()
            } else {
                Vec::new()
            };
            children.into_iter()
        }

        fn filter_map_children<T, P, F>(&self, mut project: P, mut visitor: F)
        where
            P: FnMut(Self::Unit) -> Option<T>,
            F: FnMut(Self::Unit, Self, T),
        {
            if self.label.is_none() {
                for label in u8::MIN..=u8::MAX {
                    if let Some(projected) = project(label) {
                        visitor(label, self.child(label), projected);
                    }
                }
            }
        }

        fn path(&self) -> Vec<Self::Unit> {
            self.label.into_iter().collect()
        }
    }

    #[test]
    fn test_exact_match() {
        let dict = PathMapDictionary::<()>::new();
        dict.insert("cat");

        let dict_zipper = PathMapZipper::new_from_dict(&dict);
        let iter = ZipperQueryIterator::new(dict_zipper, "cat", 0, Algorithm::Standard);

        let results: Vec<_> = iter.collect();
        assert_eq!(results.len(), 1);
        assert_eq!(results[0].term, "cat");
        assert_eq!(results[0].distance, 0);
    }

    #[test]
    fn test_fuzzy_match() {
        let dict = PathMapDictionary::<()>::new();
        dict.insert("cat");
        dict.insert("car");
        dict.insert("dog");

        let dict_zipper = PathMapZipper::new_from_dict(&dict);
        let iter = ZipperQueryIterator::new(dict_zipper, "cat", 1, Algorithm::Standard);

        let results: Vec<_> = iter.collect();

        // Should find "cat" (distance 0) and "car" (distance 1)
        assert!(results.iter().any(|c| c.term == "cat" && c.distance == 0));
        assert!(results.iter().any(|c| c.term == "car" && c.distance == 1));

        // Should NOT find "dog" (distance 3)
        assert!(!results.iter().any(|c| c.term == "dog"));
    }

    #[test]
    fn test_multiple_distances() {
        let dict = PathMapDictionary::<()>::new();
        dict.insert("cat");
        dict.insert("at");
        dict.insert("ca");

        let dict_zipper = PathMapZipper::new_from_dict(&dict);
        let iter = ZipperQueryIterator::new(dict_zipper, "cat", 1, Algorithm::Standard);

        let results: Vec<_> = iter.collect();

        // Should find all three terms
        assert_eq!(results.len(), 3);

        // "cat" should have distance 0
        assert!(results.iter().any(|c| c.term == "cat" && c.distance == 0));

        // Both shorter terms require one full-string deletion. Compare to the
        // independent oracle so prefix-only scoring cannot return unnoticed.
        for term in ["at", "ca"] {
            let expected = standard_distance("cat", term);
            assert!(
                results
                    .iter()
                    .any(|candidate| candidate.term == term && candidate.distance == expected),
                "missing oracle distance for {term}"
            );
        }
    }

    #[test]
    fn test_empty_dictionary() {
        let dict = PathMapDictionary::<()>::new();

        let dict_zipper = PathMapZipper::new_from_dict(&dict);
        let iter = ZipperQueryIterator::new(dict_zipper, "cat", 1, Algorithm::Standard);

        let results: Vec<_> = iter.collect();
        assert_eq!(results.len(), 0);
    }

    #[test]
    fn test_no_matches_within_distance() {
        let dict = PathMapDictionary::<()>::new();
        dict.insert("cat");
        dict.insert("car");

        let dict_zipper = PathMapZipper::new_from_dict(&dict);
        // Query "dog" with max distance 1 - should not match "cat" or "car"
        let iter = ZipperQueryIterator::new(dict_zipper, "dog", 1, Algorithm::Standard);

        let results: Vec<_> = iter.collect();
        assert_eq!(results.len(), 0);
    }

    #[test]
    fn test_transposition_algorithm() {
        let dict = PathMapDictionary::<()>::new();
        dict.insert("abcd");
        dict.insert("bacd");

        let dict_zipper = PathMapZipper::new_from_dict(&dict);
        let iter = ZipperQueryIterator::new(dict_zipper, "abcd", 1, Algorithm::Transposition);

        let results: Vec<_> = iter.collect();

        // Should find "abcd" (exact) and "bacd" (transposition of first two chars)
        assert!(results.iter().any(|c| c.term == "abcd"));
        assert!(results.iter().any(|c| c.term == "bacd"));
    }

    #[test]
    fn damerau_empty_final_respects_zero_cutoff() {
        let dict = PathMapDictionary::<()>::new();
        dict.insert("");
        dict.insert("a");

        let dict_zipper = PathMapZipper::new_from_dict(&dict);
        let results: Vec<_> =
            ZipperQueryIterator::new(dict_zipper, "a", 0, Algorithm::DamerauLevenshtein).collect();

        assert_eq!(results.len(), 1);
        assert_eq!(results[0].term, "a");
        assert_eq!(results[0].distance, 0);
    }

    #[test]
    fn automaton_projection_constructs_only_viable_wide_children() {
        let constructions = Arc::new(AtomicUsize::new(0));
        let zipper = CountingWideZipper::root(Arc::clone(&constructions));

        let results: Vec<_> =
            ZipperQueryIterator::new(zipper, "a", 0, Algorithm::Standard).collect();

        assert_eq!(
            results,
            vec![Candidate {
                term: "a".to_owned(),
                distance: 0,
            }]
        );
        assert_eq!(constructions.load(Ordering::Relaxed), 1);
    }

    proptest! {
        #[test]
        fn every_exact_ascii_projection_constructs_one_of_256_children(
            wanted in 0x20_u8..=0x7e,
        ) {
            let constructions = Arc::new(AtomicUsize::new(0));
            let zipper = CountingWideZipper::root(Arc::clone(&constructions));
            let query = String::from_utf8(vec![wanted]).expect("generated ASCII is UTF-8");

            let results: Vec<_> =
                ZipperQueryIterator::new(zipper, &query, 0, Algorithm::Standard).collect();

            prop_assert_eq!(results.len(), 1);
            prop_assert_eq!(results[0].term.as_bytes(), &[wanted]);
            prop_assert_eq!(results[0].distance, 0);
            prop_assert_eq!(constructions.load(Ordering::Relaxed), 1);
        }
    }

    #[test]
    fn non_root_focus_returns_relative_terms() {
        let dict = PathMapDictionary::<()>::new();
        for term in ["scope/car", "scope/cat", "elsewhere"] {
            dict.insert(term);
        }

        let mut focus = PathMapZipper::new_from_dict(&dict);
        for &label in b"scope/" {
            focus = focus.descend(label).expect("scope prefix exists");
        }

        let results: Vec<_> =
            ZipperQueryIterator::new(focus, "cat", 1, Algorithm::Standard).collect();
        assert_eq!(
            results,
            vec![
                Candidate {
                    term: "car".to_owned(),
                    distance: 1,
                },
                Candidate {
                    term: "cat".to_owned(),
                    distance: 0,
                },
            ]
        );
    }

    #[test]
    fn captured_focus_is_snapshot_isolated_from_later_mutation() {
        let dict = PathMapDictionary::<()>::new();
        dict.insert("before");
        let captured = PathMapZipper::new_from_dict(&dict);
        dict.insert("after");

        let old_results: Vec<_> =
            ZipperQueryIterator::new(captured, "after", 0, Algorithm::Standard).collect();
        let fresh_results: Vec<_> = ZipperQueryIterator::new(
            PathMapZipper::new_from_dict(&dict),
            "after",
            0,
            Algorithm::Standard,
        )
        .collect();

        assert!(old_results.is_empty());
        assert_eq!(fresh_results.len(), 1);
        assert_eq!(fresh_results[0].term, "after");
    }

    #[test]
    fn deep_focus_product_is_iterative_on_a_small_stack() {
        const DEPTH: usize = 8_192;
        let term = "a".repeat(DEPTH);
        let dict = PathMapDictionary::<()>::new();
        dict.insert(&term);
        let focus = PathMapZipper::new_from_dict(&dict);

        std::thread::Builder::new()
            .stack_size(256 * 1024)
            .spawn(move || {
                let results: Vec<_> =
                    ZipperQueryIterator::new(focus, &term, 0, Algorithm::Standard).collect();
                assert_eq!(results.len(), 1);
                assert_eq!(results[0].term.len(), DEPTH);
                assert_eq!(results[0].distance, 0);
            })
            .expect("small-stack test thread starts")
            .join()
            .expect("iterative product does not overflow the process stack");
    }

    #[test]
    fn test_larger_dictionary() {
        let dict = PathMapDictionary::<()>::new();
        let words = vec![
            "cat", "car", "card", "care", "careful", "dog", "door", "dot", "test", "testing",
            "tester",
        ];
        for word in &words {
            dict.insert(word);
        }

        let dict_zipper = PathMapZipper::new_from_dict(&dict);
        let iter = ZipperQueryIterator::new(dict_zipper, "car", 1, Algorithm::Standard);

        let results: Vec<_> = iter.collect();

        // Should find "car", "cat", "card", "care"
        assert!(results.iter().any(|c| c.term == "car"));
        assert!(results.iter().any(|c| c.term == "cat"));
        assert!(results.iter().any(|c| c.term == "card"));
        assert!(results.iter().any(|c| c.term == "care"));

        // Should have at least 4 results
        assert!(results.len() >= 4);
    }

    #[test]
    fn exhaustive_small_product_matches_independent_oracles() {
        fn binary_terms(max_len: usize) -> Vec<String> {
            let mut terms = vec![String::new()];
            for len in 1..=max_len {
                for bits in 0..(1usize << len) {
                    let term = (0..len)
                        .map(|offset| if bits & (1 << offset) == 0 { 'a' } else { 'b' })
                        .collect();
                    terms.push(term);
                }
            }
            terms
        }

        let terms = binary_terms(3);
        let dict = PathMapDictionary::<()>::new();
        for term in &terms {
            dict.insert(term);
        }

        let merge_split_cache = create_memo_cache();
        for algorithm in [
            Algorithm::Standard,
            Algorithm::Transposition,
            Algorithm::MergeAndSplit,
            Algorithm::DamerauLevenshtein,
        ] {
            for query in &terms {
                for cutoff in 0..=3 {
                    let mut expected: Vec<_> = terms
                        .iter()
                        .filter_map(|term| {
                            let distance = match algorithm {
                                Algorithm::Standard => standard_distance(query, term),
                                Algorithm::Transposition => transposition_distance(query, term),
                                Algorithm::MergeAndSplit => {
                                    merge_and_split_distance(query, term, &merge_split_cache)
                                }
                                Algorithm::DamerauLevenshtein => {
                                    damerau_levenshtein_distance(query, term)
                                }
                            };
                            (distance <= cutoff).then(|| (term.clone(), distance))
                        })
                        .collect();
                    expected.sort_unstable();

                    let dict_zipper = PathMapZipper::new_from_dict(&dict);
                    let ordered_actual: Vec<_> =
                        ZipperQueryIterator::new(dict_zipper, query, cutoff, algorithm)
                            .map(|candidate| (candidate.term, candidate.distance))
                            .collect();
                    let ordered_direct: Vec<_> = QueryIterator::<_, Candidate>::new(
                        dict.root(),
                        query.clone(),
                        cutoff,
                        algorithm,
                    )
                    .map(|candidate| (candidate.term, candidate.distance))
                    .collect();

                    assert_eq!(
                        ordered_actual, ordered_direct,
                        "zipper and node product order differ for algorithm={algorithm:?}, query={query:?}, cutoff={cutoff}",
                    );

                    let mut actual = ordered_actual;
                    actual.sort_unstable();

                    assert_eq!(
                        actual, expected,
                        "lazy product mismatch for algorithm={algorithm:?}, query={query:?}, cutoff={cutoff}",
                    );
                }
            }
        }
    }
}
