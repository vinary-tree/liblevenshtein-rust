//! Levenshtein distance algorithm variants.

use crate::transducer::OperationSet;

/// Levenshtein distance algorithm type.
///
/// Different algorithms support different edit operations and are
/// suited for different use cases.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
#[non_exhaustive]
#[cfg_attr(
    feature = "serialization",
    derive(serde::Serialize, serde::Deserialize)
)]
#[derive(Default)]
pub enum Algorithm {
    /// Standard Levenshtein distance.
    ///
    /// Supports three edit operations:
    /// - Insert: add a character
    /// - Delete: remove a character
    /// - Substitute: replace one character with another
    ///
    /// This is the classic edit distance metric.
    #[default]
    Standard,

    /// Optimal string alignment distance with adjacent transposition.
    ///
    /// Extends Standard with:
    /// - Transpose: swap two adjacent characters
    ///
    /// Useful for catching common typos where adjacent letters are swapped.
    /// Unlike unrestricted Damerau–Levenshtein distance, this restricted
    /// recurrence cannot edit a substring more than once and is not a metric.
    #[doc(alias = "OSA")]
    #[doc(alias = "restricted-damerau")]
    Transposition,

    /// Levenshtein distance with merge and split operations.
    ///
    /// Extends Standard with:
    /// - Merge: combine two characters into one
    /// - Split: expand one character into two
    ///
    /// Useful for OCR errors and other character-level transformations.
    MergeAndSplit,

    /// Unrestricted Damerau–Levenshtein distance.
    ///
    /// Unlike [`Transposition`](Self::Transposition), edit operations may act
    /// on substrings already changed by an earlier operation. The resulting
    /// distance is a metric and separates from optimal string alignment on
    /// examples such as `d("CA", "ABC") = 2`.
    #[doc(alias = "true-damerau")]
    #[doc(alias = "unrestricted-damerau")]
    DamerauLevenshtein,
}

impl Algorithm {
    /// Largest exactly representable budget for unrestricted Damerau–Levenshtein.
    ///
    /// Pending macro transitions store their positive query-endpoint delta in
    /// one byte so [`Position`](crate::transducer::Position) remains 24 bytes.
    /// Practical fuzzy-search budgets are normally 1–3. Public transition
    /// entry points fail explicitly above this ceiling instead of silently
    /// omitting unrepresentable edit scripts.
    pub const MAX_DAMERAU_DISTANCE: usize = u8::MAX as usize;

    /// Get a human-readable name for this algorithm
    pub fn name(&self) -> &'static str {
        match self {
            Algorithm::Standard => "standard",
            Algorithm::Transposition => "transposition",
            Algorithm::MergeAndSplit => "merge-and-split",
            Algorithm::DamerauLevenshtein => "damerau-levenshtein",
        }
    }

    /// Check if this algorithm supports transposition operations
    pub fn supports_transposition(&self) -> bool {
        matches!(
            self,
            Algorithm::Transposition | Algorithm::DamerauLevenshtein
        )
    }

    /// Assert that `max_distance` is exactly representable by this variant.
    ///
    /// Existing transition APIs predate fallible algorithm selection, so an
    /// unsupported true-Damerau budget is a contract violation and panics.
    /// Other variants have no variant-specific representation ceiling.
    #[inline]
    pub(crate) fn assert_supported_max_distance(self, max_distance: usize) {
        assert!(
            self != Self::DamerauLevenshtein || max_distance <= Self::MAX_DAMERAU_DISTANCE,
            "Algorithm::DamerauLevenshtein supports max_distance <= {}; got {max_distance}",
            Self::MAX_DAMERAU_DISTANCE,
        );
    }

    /// Check if this algorithm supports merge/split operations
    pub fn supports_merge_split(&self) -> bool {
        matches!(self, Algorithm::MergeAndSplit)
    }

    /// Whether the distance is a metric and therefore satisfies the triangle inequality.
    ///
    /// [`Transposition`](Algorithm::Transposition) implements optimal string
    /// alignment (restricted Damerau distance). It is symmetric and separates
    /// distinct strings, but it violates the triangle inequality—for example,
    /// `d("CA", "ABC") = 3` while `d("CA", "AC") + d("AC", "ABC") = 2`.
    /// Standard Levenshtein and the generic, symmetric Merge-and-Split distance
    /// are metrics.
    ///
    /// This classification concerns metric-tree pruning. Trie dynamic-programming
    /// walkers may instead rely on an admissible lower bound and non-negative
    /// step costs, which do not require a metric.
    pub const fn is_metric(&self) -> bool {
        matches!(
            self,
            Algorithm::Standard | Algorithm::MergeAndSplit | Algorithm::DamerauLevenshtein
        )
    }

    /// Convert this algorithm's local edit repertoire to an `OperationSet`.
    ///
    /// Maps the enum variant to the corresponding operation-set configuration.
    /// This enables backward compatibility with the generalized operations framework.
    ///
    /// # History-dependent projection
    ///
    /// [`DamerauLevenshtein`](Self::DamerauLevenshtein) projects to the same
    /// local repertoire as [`Transposition`](Self::Transposition). An
    /// `OperationSet` describes alignments and cannot preserve the former's
    /// edit-history composition. Running that projection in a generalized
    /// automaton therefore computes optimal string alignment, **not** true
    /// Damerau–Levenshtein. Retain the `Algorithm` selector and use the
    /// dedicated unit-cost transducer when history-dependent semantics matter.
    ///
    /// # Returns
    ///
    /// An `OperationSet` containing the operations for this algorithm.
    ///
    /// # Examples
    ///
    /// ```rust
    /// # use liblevenshtein::transducer::Algorithm;
    /// let alg = Algorithm::Standard;
    /// let ops = alg.to_operation_set();
    /// assert_eq!(ops.len(), 4);  // Match, Substitute, Insert, Delete
    ///
    /// let alg = Algorithm::Transposition;
    /// let ops = alg.to_operation_set();
    /// assert_eq!(ops.len(), 5);  // Standard + Transposition
    /// ```
    pub fn to_operation_set(&self) -> OperationSet {
        match self {
            Algorithm::Standard => OperationSet::standard(),
            Algorithm::Transposition => OperationSet::with_transposition(),
            Algorithm::MergeAndSplit => OperationSet::with_merge_split(),
            // True Damerau and OSA have the same local operation repertoire;
            // OperationSet captures that repertoire, not history-dependent
            // composition. GeneralizedAutomaton over this projection computes
            // OSA, so callers needing true-Damerau semantics must retain this
            // Algorithm and use the dedicated transducer.
            Algorithm::DamerauLevenshtein => OperationSet::with_transposition(),
        }
    }
}

impl std::fmt::Display for Algorithm {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.write_str(self.name())
    }
}

impl std::str::FromStr for Algorithm {
    type Err = String;

    fn from_str(s: &str) -> Result<Self, Self::Err> {
        match s.to_lowercase().as_str() {
            "standard" => Ok(Algorithm::Standard),
            "transposition" | "trans" => Ok(Algorithm::Transposition),
            "merge-and-split" | "mergesplit" | "merge" => Ok(Algorithm::MergeAndSplit),
            "damerau-levenshtein" | "damerau_levenshtein" | "damerau" | "true-damerau" => {
                Ok(Algorithm::DamerauLevenshtein)
            }
            _ => Err(format!(
                "Unknown algorithm: {}. Valid options: standard, transposition, merge-and-split, damerau-levenshtein",
                s
            )),
        }
    }
}

impl From<Algorithm> for OperationSet {
    /// Convert an Algorithm to an OperationSet
    ///
    /// Enables seamless conversion from the legacy enum-based API to the
    /// generalized operations framework. This is the preferred conversion path.
    ///
    /// # Examples
    ///
    /// ```rust
    /// # use liblevenshtein::transducer::{Algorithm, OperationSet};
    /// let ops: OperationSet = Algorithm::Standard.into();
    /// assert_eq!(ops.len(), 4);
    /// ```
    fn from(algorithm: Algorithm) -> Self {
        algorithm.to_operation_set()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_to_operation_set_standard() {
        let ops = Algorithm::Standard.to_operation_set();
        assert_eq!(ops.len(), 4); // Match, Substitute, Insert, Delete
    }

    #[test]
    fn test_to_operation_set_transposition() {
        let ops = Algorithm::Transposition.to_operation_set();
        assert_eq!(ops.len(), 5); // Standard + Transposition
    }

    #[test]
    fn test_to_operation_set_merge_split() {
        let ops = Algorithm::MergeAndSplit.to_operation_set();
        assert_eq!(ops.len(), 6); // Standard + Merge + Split
    }

    #[test]
    fn damerau_selector_round_trips_and_projects_its_local_repertoire() {
        let algorithm: Algorithm = "true-damerau".parse().unwrap();
        assert_eq!(algorithm, Algorithm::DamerauLevenshtein);
        assert_eq!(algorithm.to_string(), "damerau-levenshtein");
        assert!(algorithm.supports_transposition());
        assert!(algorithm.is_metric());
        assert_eq!(algorithm.to_operation_set().len(), 5);
    }

    #[test]
    fn test_from_algorithm_to_operation_set() {
        let ops: OperationSet = Algorithm::Standard.into();
        assert_eq!(ops.len(), 4);

        let ops: OperationSet = Algorithm::Transposition.into();
        assert_eq!(ops.len(), 5);

        let ops: OperationSet = Algorithm::MergeAndSplit.into();
        assert_eq!(ops.len(), 6);
    }

    #[test]
    fn capability_queries_match_declared_operation_sets() {
        for algorithm in [
            Algorithm::Standard,
            Algorithm::Transposition,
            Algorithm::MergeAndSplit,
            Algorithm::DamerauLevenshtein,
        ] {
            let operations = algorithm.to_operation_set();
            let has_transposition = operations
                .iter()
                .any(|operation| operation.consume_x() == 2 && operation.consume_y() == 2);
            let has_merge_or_split = operations.iter().any(|operation| {
                matches!(
                    (operation.consume_x(), operation.consume_y()),
                    (1, 2) | (2, 1)
                )
            });

            assert_eq!(algorithm.supports_transposition(), has_transposition);
            assert_eq!(algorithm.supports_merge_split(), has_merge_or_split);
        }
    }

    #[test]
    fn metric_classification_distinguishes_osa() {
        assert!(Algorithm::Standard.is_metric());
        assert!(!Algorithm::Transposition.is_metric());
        assert!(Algorithm::MergeAndSplit.is_metric());
        assert!(Algorithm::DamerauLevenshtein.is_metric());
    }
}
