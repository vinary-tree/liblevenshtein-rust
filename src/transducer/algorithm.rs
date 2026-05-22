//! Levenshtein distance algorithm variants.

use crate::transducer::OperationSet;

/// Levenshtein distance algorithm type.
///
/// Different algorithms support different edit operations and are
/// suited for different use cases.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
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

    /// Levenshtein distance with transposition.
    ///
    /// Extends Standard with:
    /// - Transpose: swap two adjacent characters
    ///
    /// Useful for catching common typos where adjacent letters are swapped.
    Transposition,

    /// Levenshtein distance with merge and split operations.
    ///
    /// Extends Standard with:
    /// - Merge: combine two characters into one
    /// - Split: expand one character into two
    ///
    /// Useful for OCR errors and other character-level transformations.
    MergeAndSplit,
}

impl Algorithm {
    /// Get a human-readable name for this algorithm
    pub fn name(&self) -> &'static str {
        match self {
            Algorithm::Standard => "standard",
            Algorithm::Transposition => "transposition",
            Algorithm::MergeAndSplit => "merge-and-split",
        }
    }

    /// Check if this algorithm supports transposition operations
    pub fn supports_transposition(&self) -> bool {
        matches!(self, Algorithm::Transposition | Algorithm::MergeAndSplit)
    }

    /// Check if this algorithm supports merge/split operations
    pub fn supports_merge_split(&self) -> bool {
        matches!(self, Algorithm::MergeAndSplit)
    }

    /// Convert this algorithm to an OperationSet
    ///
    /// Maps the enum variant to the corresponding operation set configuration.
    /// This enables backward compatibility with the generalized operations framework.
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
            _ => Err(format!(
                "Unknown algorithm: {}. Valid options: standard, transposition, merge-and-split",
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
    fn test_from_algorithm_to_operation_set() {
        let ops: OperationSet = Algorithm::Standard.into();
        assert_eq!(ops.len(), 4);

        let ops: OperationSet = Algorithm::Transposition.into();
        assert_eq!(ops.len(), 5);

        let ops: OperationSet = Algorithm::MergeAndSplit.into();
        assert_eq!(ops.len(), 6);
    }
}
