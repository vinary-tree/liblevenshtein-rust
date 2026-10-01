//! WallBreaker algorithm for approximate string matching with large error bounds.
//!
//! This module implements the WallBreaker algorithm from:
//! > "WallBreaker - overcoming the wall effect in similarity search"
//! > (Gerdjikov, Mihov, Mitankin, Schulz - EDBT/ICDT 2013)
//!
//! # The Wall Effect Problem
//!
//! Traditional Levenshtein automata traverse dictionaries left-to-right.
//! With error bound `b`, the first `b` steps must explore ALL prefixes up
//! to length `b` before any filtering occurs. This creates a "wall" that
//! limits performance for large error bounds.
//!
//! # WallBreaker Solution
//!
//! WallBreaker overcomes the wall by:
//!
//! 1. **Splitting** the query into pieces based on the pigeonhole principle
//! 2. **Finding exact matches** for each piece using SCDAWG substring search
//! 3. **Verifying** each original complete member with bounded edit distance
//! 4. **Deduplicating** and yielding only dictionary members
//!
//! The separate [`BidirectionalExtension`](crate::wallbreaker::BidirectionalExtension) helper remains public for native
//! experimentation, but it is not used for qualified `WallBreakerQuery`
//! results: reconstructing a complete term from suffix-graph labels can
//! fabricate nonmembers. Short queries (including empty) use the dictionary's
//! exact empty-substring enumeration because no nonempty piece is guaranteed.
//!
//! # Piece Count by Algorithm (Formally Verified)
//!
//! The number of pieces depends on the edit distance algorithm:
//!
//! - **Standard Levenshtein**: `k+1` pieces suffice
//!   - Each operation (insert, delete, substitute) corrupts at most 1 piece
//!
//! - **Transposition (optimal string alignment)**: `2k+1` pieces required
//!   - Adjacent transpositions can corrupt 2 pieces when spanning boundaries
//!   - Proven in `WallBreakerPigeonhole.v` with counterexample for k=2
//!
//! - **MergeAndSplit**: `2k+1` pieces required
//!   - Merge operations span 2 characters, can corrupt 2 pieces at boundaries
//!   - Proven in `WallBreakerPigeonhole.v` with counterexample for k=2
//!
//! # Performance
//!
//! Substring filtering can help when a surviving piece is selective. Short
//! queries and unrestricted Damerau use an exact full-term scan, however,
//! and complete-term verification is required in every path. Historical
//! speedup projections predate the correctness repair and are not a current
//! benchmark qualification.
//!
//! # Example
//!
//! ```rust
//! use liblevenshtein::dictionary::scdawg::Scdawg;
//! use liblevenshtein::wallbreaker::WallBreaker;
//! use liblevenshtein::transducer::Algorithm;
//!
//! // Build SCDAWG dictionary
//! let dict = Scdawg::<()>::from_terms(vec!["cathedral", "category", "catering"]);
//!
//! // Create WallBreaker with max distance 2 (Standard algorithm by default)
//! let wb = WallBreaker::new(&dict, 2);
//!
//! // Or explicitly specify an algorithm for Transposition or MergeAndSplit
//! let wb = WallBreaker::with_algorithm(&dict, 2, Algorithm::Transposition);
//!
//! // Find approximate matches
//! for result in wb.query("cathedrel") {
//!     println!("{} (distance {})", result.term, result.distance);
//! }
//! // Output: cathedral (distance 1)
//! ```

mod extension;
mod pattern_splitter;
mod query_iterator;

pub use extension::{BidirectionalExtension, ExtensionState};
pub use pattern_splitter::{PatternPiece, PatternSplitter};
pub use query_iterator::{WallBreakerQuery, WallBreakerResult};

use crate::transducer::Algorithm;
use libdictenstein::substring::{BidirectionalDictionaryNode, SubstringDictionary};
use libdictenstein::Dictionary;

/// WallBreaker approximate string matcher.
///
/// Wraps a [`SubstringDictionary`] (typically an SCDAWG) and provides
/// approximate matching using exact substring candidates and bounded distance
/// verification. The bidirectional extension helper is not part of this
/// qualified result path.
///
/// # Algorithm Support
///
/// WallBreaker supports all three Levenshtein algorithm variants with
/// algorithm-specific piece counts (formally verified in `WallBreakerPigeonhole.v`):
///
/// - **Standard**: `k+1` pieces (default)
/// - **Transposition**: `2k+1` pieces
/// - **MergeAndSplit**: `2k+1` pieces
///
/// # Type Parameters
///
/// * `D` - Dictionary type that implements both [`Dictionary`] and [`SubstringDictionary`]
///
/// # Example
///
/// ```rust
/// use liblevenshtein::dictionary::scdawg::Scdawg;
/// use liblevenshtein::wallbreaker::WallBreaker;
/// use liblevenshtein::transducer::Algorithm;
///
/// let dict = Scdawg::<()>::from_terms(["hello", "world", "help"]);
///
/// // Standard algorithm (default)
/// let wb = WallBreaker::new(&dict, 1);
///
/// // Transposition algorithm
/// let wb = WallBreaker::with_algorithm(&dict, 1, Algorithm::Transposition);
///
/// let results: Vec<_> = wb.query("helo").collect();
/// assert!(results.iter().any(|r| r.term == "hello"));
/// ```
pub struct WallBreaker<'a, D>
where
    D: Dictionary + SubstringDictionary,
    D::Node: BidirectionalDictionaryNode,
    <D::Node as crate::dictionary::DictionaryNode>::Unit: Into<u32>,
{
    dictionary: &'a D,
    max_distance: usize,
    algorithm: Algorithm,
    splitter: PatternSplitter,
}

impl<'a, D> WallBreaker<'a, D>
where
    D: Dictionary + SubstringDictionary,
    D::Node: BidirectionalDictionaryNode,
    <D::Node as crate::dictionary::DictionaryNode>::Unit: Into<u32>,
{
    /// Create a new WallBreaker with the given dictionary and max distance.
    ///
    /// Uses the Standard Levenshtein algorithm by default.
    /// For Transposition or MergeAndSplit, use [`with_algorithm`](Self::with_algorithm).
    ///
    /// # Arguments
    ///
    /// * `dictionary` - The SCDAWG or other substring-searchable dictionary
    /// * `max_distance` - Maximum Levenshtein distance for matches
    ///
    /// # Example
    ///
    /// ```rust
    /// use liblevenshtein::dictionary::scdawg::Scdawg;
    /// use liblevenshtein::wallbreaker::WallBreaker;
    ///
    /// let dict = Scdawg::<()>::from_terms(["test"]);
    /// let wb = WallBreaker::new(&dict, 2);
    /// assert_eq!(wb.max_distance(), 2);
    /// ```
    pub fn new(dictionary: &'a D, max_distance: usize) -> Self {
        Self::with_algorithm(dictionary, max_distance, Algorithm::Standard)
    }

    /// Create a new WallBreaker with a specific algorithm.
    ///
    /// # Algorithm-Specific Piece Counts (Formally Verified)
    ///
    /// - **Standard**: `k+1` pieces (each operation corrupts ≤1 piece)
    /// - **Transposition**: `2k+1` pieces (transpositions can corrupt 2 pieces)
    /// - **MergeAndSplit**: `2k+1` pieces (merge/split can corrupt 2 pieces)
    ///
    /// These piece counts are proven correct in `WallBreakerPigeonhole.v`.
    ///
    /// # Arguments
    ///
    /// * `dictionary` - The SCDAWG or other substring-searchable dictionary
    /// * `max_distance` - Maximum edit distance for matches
    /// * `algorithm` - The edit distance algorithm to use
    ///
    /// # Example
    ///
    /// ```rust
    /// use liblevenshtein::dictionary::scdawg::Scdawg;
    /// use liblevenshtein::transducer::Algorithm;
    /// use liblevenshtein::wallbreaker::WallBreaker;
    ///
    /// let dict = Scdawg::<()>::from_terms(["test"]);
    ///
    /// // For OSA (adjacent-transposition) matching
    /// let wb = WallBreaker::with_algorithm(&dict, 2, Algorithm::Transposition);
    /// ```
    pub fn with_algorithm(dictionary: &'a D, max_distance: usize, algorithm: Algorithm) -> Self {
        WallBreaker {
            dictionary,
            max_distance,
            algorithm,
            splitter: PatternSplitter::new(max_distance, algorithm),
        }
    }

    /// Query the dictionary for approximate matches.
    ///
    /// Returns an iterator over (term, distance) pairs for all dictionary
    /// terms within `max_distance` of the query.
    ///
    /// # Arguments
    ///
    /// * `query` - The query string to match
    ///
    /// # Returns
    ///
    /// An iterator yielding `(String, usize)` pairs where:
    /// - `String` is the matched dictionary term
    /// - `usize` is the Levenshtein distance from query to term
    ///
    /// # Example
    ///
    /// ```rust
    /// use liblevenshtein::dictionary::scdawg::Scdawg;
    /// use liblevenshtein::wallbreaker::WallBreaker;
    ///
    /// let dict = Scdawg::<()>::from_terms(["test"]);
    /// let wb = WallBreaker::new(&dict, 2);
    /// for result in wb.query("tset") {
    ///     println!("{} at distance {}", result.term, result.distance);
    /// }
    /// ```
    pub fn query(&self, query: &str) -> WallBreakerQuery<'_, D> {
        WallBreakerQuery::new(self.dictionary, query, self.max_distance, &self.splitter)
    }

    /// Get the maximum distance configured for this WallBreaker.
    pub fn max_distance(&self) -> usize {
        self.max_distance
    }

    /// Get the algorithm configured for this WallBreaker.
    pub fn algorithm(&self) -> Algorithm {
        self.algorithm
    }

    /// Update the maximum distance.
    ///
    /// This preserves the current algorithm and updates the pattern splitter
    /// with the algorithm-specific piece count.
    pub fn set_max_distance(&mut self, max_distance: usize) {
        self.max_distance = max_distance;
        self.splitter = PatternSplitter::new(max_distance, self.algorithm);
    }

    /// Update the algorithm.
    ///
    /// This updates the pattern splitter with the new algorithm-specific piece count.
    pub fn set_algorithm(&mut self, algorithm: Algorithm) {
        self.algorithm = algorithm;
        self.splitter = PatternSplitter::new(self.max_distance, algorithm);
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use libdictenstein::scdawg::Scdawg;
    use libdictenstein::scdawg::ScdawgChar;

    #[test]
    fn test_wallbreaker_basic() {
        let dict = Scdawg::<()>::from_terms(vec!["hello", "world", "help"]);
        let wb = WallBreaker::new(&dict, 1);

        let results: Vec<_> = wb.query("helo").collect();
        assert!(!results.is_empty());
        assert!(results.iter().any(|r| r.term == "hello"));
    }

    #[test]
    fn unicode_wallbreaker_matches_independent_standard_distance_oracle() {
        let terms = [
            "", "a", "b", "café", "cafe", "préfixe", "suffixe", "αβγ", "αXγ",
        ];
        let dictionary = ScdawgChar::<()>::from_terms(terms);
        let mut mismatches = Vec::new();
        for query in ["", "a", "café", "cafe", "préfixe", "αβγ", "αβδ"] {
            for bound in 0..=2 {
                let matcher = WallBreaker::new(&dictionary, bound);
                let mut observed: Vec<_> = matcher
                    .query(query)
                    .map(|result| (result.term, result.distance))
                    .collect();
                observed.sort();
                let mut expected: Vec<_> = terms
                    .iter()
                    .filter_map(|term| {
                        let distance = crate::distance::standard_distance(query, term);
                        (distance <= bound).then(|| ((*term).to_owned(), distance))
                    })
                    .collect();
                expected.sort();
                if observed != expected {
                    mismatches.push(format!("query={query:?}, bound={bound}: observed={observed:?}, expected={expected:?}"));
                }
            }
        }
        assert!(mismatches.is_empty(), "{}", mismatches.join("\n"));
    }

    #[test]
    fn wallbreaker_matches_exhaustive_small_unicode_oracles_for_all_variants() {
        let mut terms = vec![String::new()];
        let alphabet = ['a', 'b', 'é'];
        let mut generation = vec![String::new()];
        for _ in 0..4 {
            generation = generation
                .iter()
                .flat_map(|prefix| alphabet.iter().map(move |unit| format!("{prefix}{unit}")))
                .collect();
            terms.extend(generation.iter().cloned());
        }
        let dictionary = ScdawgChar::<()>::from_terms(terms.iter().map(String::as_str));
        for query in &terms {
            for algorithm in [
                Algorithm::Standard,
                Algorithm::Transposition,
                Algorithm::MergeAndSplit,
                Algorithm::DamerauLevenshtein,
            ] {
                for bound in 0..=2 {
                    let mut observed: Vec<_> =
                        WallBreaker::with_algorithm(&dictionary, bound, algorithm)
                            .query(query)
                            .map(|result| (result.term, result.distance))
                            .collect();
                    observed.sort();
                    let mut expected: Vec<_> = terms
                        .iter()
                        .filter_map(|term| {
                            let distance = match algorithm {
                                Algorithm::Standard => {
                                    crate::distance::standard_distance(query, term)
                                }
                                Algorithm::Transposition => {
                                    crate::distance::transposition_distance(query, term)
                                }
                                Algorithm::MergeAndSplit => {
                                    crate::distance::merge_and_split_distance(
                                        query,
                                        term,
                                        &crate::distance::create_memo_cache(),
                                    )
                                }
                                Algorithm::DamerauLevenshtein => {
                                    crate::distance::damerau_levenshtein_distance(query, term)
                                }
                            };
                            (distance <= bound).then(|| (term.clone(), distance))
                        })
                        .collect();
                    expected.sort();
                    assert_eq!(
                        observed, expected,
                        "query={query:?}, bound={bound}, algorithm={algorithm:?}"
                    );
                }
            }
        }
    }

    #[test]
    fn wallbreaker_query_pins_revision_and_deduplicates_repeated_occurrences() {
        let dictionary = ScdawgChar::<()>::from_terms(["banana", "bandana", "cabana"]);
        let matcher = WallBreaker::new(&dictionary, 2);
        let pending = matcher.query("banana");
        let expected: Vec<_> = matcher.query("banana").collect();
        dictionary.insert("bananas");
        let observed: Vec<_> = pending.collect();
        assert_eq!(observed, expected, "query-start root must stay pinned");
        let mut distinct = std::collections::HashSet::new();
        assert!(observed.iter().all(|result| distinct.insert(&result.term)));
        assert!(matcher
            .query("banana")
            .any(|result| result.term == "bananas"));
        assert_eq!(
            matcher.query("banana").collect::<Vec<_>>(),
            matcher.query("banana").collect::<Vec<_>>(),
            "traversal order is deterministic"
        );
    }

    #[test]
    fn seeded_long_unicode_queries_match_randomized_distance_oracles() {
        let alphabet = ['a', 'b', 'é', '猫'];
        let mut seed = 0x4d65_7267_6544_6177_u64;
        let mut terms = Vec::new();
        for _ in 0..64 {
            let mut term = String::new();
            for _ in 0..8 {
                seed = seed.wrapping_mul(6364136223846793005).wrapping_add(1);
                term.push(alphabet[((seed >> 32) & 3) as usize]);
            }
            if !terms.contains(&term) {
                terms.push(term);
            }
        }
        let dictionary = ScdawgChar::<()>::from_terms(terms.iter().map(String::as_str));
        for base in terms.iter().take(20) {
            let mut changed: Vec<char> = base.chars().collect();
            changed[2] = alphabet[(alphabet
                .iter()
                .position(|unit| *unit == changed[2])
                .unwrap()
                + 1)
                % 4];
            changed[5] = alphabet[(alphabet
                .iter()
                .position(|unit| *unit == changed[5])
                .unwrap()
                + 1)
                % 4];
            let query: String = changed.into_iter().collect();
            for algorithm in [
                Algorithm::Standard,
                Algorithm::Transposition,
                Algorithm::MergeAndSplit,
            ] {
                let mut observed: Vec<_> = WallBreaker::with_algorithm(&dictionary, 2, algorithm)
                    .query(&query)
                    .map(|result| (result.term, result.distance))
                    .collect();
                observed.sort();
                let mut expected: Vec<_> = terms
                    .iter()
                    .filter_map(|term| {
                        let distance = match algorithm {
                            Algorithm::Standard => crate::distance::standard_distance(&query, term),
                            Algorithm::Transposition => {
                                crate::distance::transposition_distance(&query, term)
                            }
                            Algorithm::MergeAndSplit => crate::distance::merge_and_split_distance(
                                &query,
                                term,
                                &crate::distance::create_memo_cache(),
                            ),
                            Algorithm::DamerauLevenshtein => unreachable!(),
                        };
                        (distance <= 2).then(|| (term.clone(), distance))
                    })
                    .collect();
                expected.sort();
                assert_eq!(
                    observed, expected,
                    "query={query:?}, algorithm={algorithm:?}"
                );
            }
        }
    }

    #[test]
    fn long_exact_unicode_query_uses_bounded_stack() {
        let term = "é".repeat(2048);
        let dictionary = ScdawgChar::<()>::from_terms([term.as_str()]);
        let observed: Vec<_> = WallBreaker::new(&dictionary, 0).query(&term).collect();
        assert_eq!(observed, vec![WallBreakerResult::new(term, 0)]);
    }

    #[test]
    fn test_wallbreaker_exact_match() {
        let dict = Scdawg::<()>::from_terms(vec!["hello", "world"]);
        let wb = WallBreaker::new(&dict, 0);

        let results: Vec<_> = wb.query("hello").collect();
        assert_eq!(results.len(), 1);
        assert_eq!(results[0].term, "hello");
        assert_eq!(results[0].distance, 0);
    }

    #[test]
    fn query_owns_one_revision_across_every_pattern_piece() {
        let dict = Scdawg::<()>::from_terms(["abcd", "omega"]);
        let wb = WallBreaker::new(&dict, 1);
        let mut captured = wb.query("abcd");

        assert_eq!(
            captured.next().map(|result| result.term),
            Some("abcd".into())
        );
        assert!(dict.insert("abxd"));
        assert!(captured.all(|result| result.term != "abxd"));

        let fresh: Vec<_> = wb.query("abcd").collect();
        assert!(fresh.iter().any(|result| result.term == "abxd"));
    }

    #[test]
    fn test_wallbreaker_no_match() {
        let dict = Scdawg::<()>::from_terms(vec!["hello", "world"]);
        let wb = WallBreaker::new(&dict, 1);

        let results: Vec<_> = wb.query("xyz").collect();
        assert!(results.is_empty());
    }

    #[test]
    fn test_wallbreaker_distance_2() {
        let dict = Scdawg::<()>::from_terms(vec!["cathedral"]);
        let wb = WallBreaker::new(&dict, 2);

        // "cathedrel" has distance 1 from "cathedral" (e->a)
        let results: Vec<_> = wb.query("cathedrel").collect();
        assert!(results.iter().any(|r| r.term == "cathedral"));
    }

    #[test]
    fn test_wallbreaker_multiple_terms() {
        let terms = vec![
            "cathedral",
            "category",
            "catering",
            "catastrophe",
            "catalog",
        ];
        let dict = Scdawg::<()>::from_terms(terms);
        let wb = WallBreaker::new(&dict, 2);

        // Test various queries
        let results: Vec<_> = wb.query("cathedrel").collect();
        assert!(results.iter().any(|r| r.term == "cathedral"));

        let results: Vec<_> = wb.query("caterng").collect();
        assert!(results.iter().any(|r| r.term == "catering"));
    }

    // Algorithm-specific tests

    #[test]
    fn test_wallbreaker_with_algorithm() {
        let dict = Scdawg::<()>::from_terms(vec!["hello", "world", "help"]);

        // Test all algorithm types can be created
        let wb_std = WallBreaker::with_algorithm(&dict, 1, Algorithm::Standard);
        let wb_trans = WallBreaker::with_algorithm(&dict, 1, Algorithm::Transposition);
        let wb_ms = WallBreaker::with_algorithm(&dict, 1, Algorithm::MergeAndSplit);

        // Verify algorithms are set correctly
        assert!(matches!(wb_std.algorithm(), Algorithm::Standard));
        assert!(matches!(wb_trans.algorithm(), Algorithm::Transposition));
        assert!(matches!(wb_ms.algorithm(), Algorithm::MergeAndSplit));
    }

    #[test]
    fn test_wallbreaker_algorithm_getter() {
        let dict = Scdawg::<()>::from_terms(vec!["test"]);

        // Default is Standard
        let wb = WallBreaker::new(&dict, 1);
        assert!(matches!(wb.algorithm(), Algorithm::Standard));

        // Explicit algorithm
        let wb = WallBreaker::with_algorithm(&dict, 1, Algorithm::Transposition);
        assert!(matches!(wb.algorithm(), Algorithm::Transposition));
    }

    #[test]
    fn test_wallbreaker_set_algorithm() {
        let dict = Scdawg::<()>::from_terms(vec!["test"]);
        let mut wb = WallBreaker::new(&dict, 2);

        // Initial algorithm is Standard
        assert!(matches!(wb.algorithm(), Algorithm::Standard));

        // Change to Transposition
        wb.set_algorithm(Algorithm::Transposition);
        assert!(matches!(wb.algorithm(), Algorithm::Transposition));

        // Change to MergeAndSplit
        wb.set_algorithm(Algorithm::MergeAndSplit);
        assert!(matches!(wb.algorithm(), Algorithm::MergeAndSplit));
    }

    #[test]
    fn test_wallbreaker_set_max_distance_preserves_algorithm() {
        let dict = Scdawg::<()>::from_terms(vec!["test"]);
        let mut wb = WallBreaker::with_algorithm(&dict, 2, Algorithm::Transposition);

        // Change max distance
        wb.set_max_distance(4);

        // Algorithm should be preserved
        assert!(matches!(wb.algorithm(), Algorithm::Transposition));
        assert_eq!(wb.max_distance(), 4);
    }

    #[test]
    fn test_wallbreaker_transposition_finds_matches() {
        let dict = Scdawg::<()>::from_terms(vec!["hello", "world", "help"]);
        let wb = WallBreaker::with_algorithm(&dict, 1, Algorithm::Transposition);

        // Should find matches using the configured transposition verifier.
        let results: Vec<_> = wb.query("helo").collect();
        assert!(!results.is_empty());
    }

    #[test]
    fn test_wallbreaker_merge_and_split_finds_matches() {
        let dict = Scdawg::<()>::from_terms(vec!["hello", "world", "help"]);
        let wb = WallBreaker::with_algorithm(&dict, 1, Algorithm::MergeAndSplit);

        // Should find matches using the configured merge/split verifier.
        let results: Vec<_> = wb.query("helo").collect();
        assert!(!results.is_empty());
    }
}
