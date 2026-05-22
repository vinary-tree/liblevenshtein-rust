//! Collection of operation types for generalized Levenshtein distance.
//!
//! This module provides [`OperationSet`] and [`OperationSetBuilder`] for composing
//! multiple [`OperationType`]s into a complete edit distance metric.
//!
//! ## Overview
//!
//! An **operation set** is a collection of operation types that defines a complete
//! generalized Levenshtein distance metric. It replaces the hardcoded `Algorithm` enum
//! with a flexible, composable system.
//!
//! ## Examples
//!
//! ### Standard Levenshtein
//!
//! ```rust
//! # use liblevenshtein::transducer::{OperationSet, OperationSetBuilder};
//! let standard = OperationSetBuilder::new()
//!     .with_match()           // ⟨1, 1, 0.0⟩
//!     .with_substitution()    // ⟨1, 1, 1.0⟩
//!     .with_insertion()       // ⟨0, 1, 1.0⟩
//!     .with_deletion()        // ⟨1, 0, 1.0⟩
//!     .build();
//!
//! assert_eq!(standard.operations().len(), 4);
//! ```
//!
//! ### With Transposition
//!
//! ```rust
//! # use liblevenshtein::transducer::{OperationSet, OperationSetBuilder};
//! let transposition = OperationSetBuilder::new()
//!     .with_standard_ops()    // match, subst, insert, delete
//!     .with_transposition()   // ⟨2, 2, 1.0⟩
//!     .build();
//!
//! assert_eq!(transposition.operations().len(), 5);
//! ```
//!
//! ### Phonetic Corrections
//!
//! ```rust
//! # use liblevenshtein::transducer::{OperationSet, OperationSetBuilder, OperationType, SubstitutionSet};
//! let mut phonetic = SubstitutionSet::new();
//! phonetic.allow_str("ph", "f");
//! phonetic.allow_str("ch", "k");
//!
//! let ops = OperationSetBuilder::new()
//!     .with_match()
//!     .with_operation(OperationType::with_restriction(
//!         2, 1, 0.15,
//!         phonetic,
//!         "consonant_digraphs"
//!     ))
//!     .with_standard_ops()
//!     .build();
//! ```

use crate::transducer::OperationType;

/// A collection of operation types defining a generalized Levenshtein distance metric.
///
/// # Examples
///
/// ```rust
/// # use liblevenshtein::transducer::{OperationSet, OperationSetBuilder};
/// // Create using builder
/// let ops = OperationSetBuilder::new()
///     .with_standard_ops()
///     .build();
///
/// // Access operations
/// for op in ops.operations() {
///     println!("{}", op);
/// }
/// ```
#[derive(Clone, Debug)]
pub struct OperationSet {
    /// Vector of operation types.
    /// Order matters for iteration, but typically not for correctness.
    operations: Vec<OperationType>,
}

impl OperationSet {
    /// Create an empty operation set.
    ///
    /// This is rarely useful directly - prefer [`OperationSetBuilder`].
    ///
    /// # Example
    ///
    /// ```rust
    /// # use liblevenshtein::transducer::OperationSet;
    /// let ops = OperationSet::new();
    /// assert_eq!(ops.operations().len(), 0);
    /// ```
    #[inline]
    pub fn new() -> Self {
        Self {
            operations: Vec::new(),
        }
    }

    /// Create an operation set with expected capacity.
    ///
    /// Pre-allocates space for `capacity` operations to avoid reallocations.
    ///
    /// # Example
    ///
    /// ```rust
    /// # use liblevenshtein::transducer::OperationSet;
    /// let ops = OperationSet::with_capacity(10);
    /// assert_eq!(ops.operations().len(), 0);
    /// ```
    #[inline]
    pub fn with_capacity(capacity: usize) -> Self {
        Self {
            operations: Vec::with_capacity(capacity),
        }
    }

    /// Add an operation to the set.
    ///
    /// # Example
    ///
    /// ```rust
    /// # use liblevenshtein::transducer::{OperationSet, OperationType};
    /// let mut ops = OperationSet::new();
    /// ops.add(OperationType::new(1, 1, 0.0, "match"));
    /// assert_eq!(ops.operations().len(), 1);
    /// ```
    pub fn add(&mut self, op: OperationType) {
        self.operations.push(op);
    }

    /// Get a slice of all operations.
    ///
    /// # Example
    ///
    /// ```rust
    /// # use liblevenshtein::transducer::{OperationSet, OperationSetBuilder};
    /// let ops = OperationSetBuilder::new().with_standard_ops().build();
    /// assert_eq!(ops.operations().len(), 4);
    /// ```
    #[inline]
    pub fn operations(&self) -> &[OperationType] {
        &self.operations
    }

    /// Get the number of operations in the set.
    ///
    /// # Example
    ///
    /// ```rust
    /// # use liblevenshtein::transducer::{OperationSet, OperationSetBuilder};
    /// let ops = OperationSetBuilder::new().with_match().build();
    /// assert_eq!(ops.len(), 1);
    /// ```
    #[inline]
    pub fn len(&self) -> usize {
        self.operations.len()
    }

    /// Check if the operation set is empty.
    ///
    /// # Example
    ///
    /// ```rust
    /// # use liblevenshtein::transducer::OperationSet;
    /// let ops = OperationSet::new();
    /// assert!(ops.is_empty());
    /// ```
    #[inline]
    pub fn is_empty(&self) -> bool {
        self.operations.is_empty()
    }

    /// Iterate over operations.
    ///
    /// # Example
    ///
    /// ```rust
    /// # use liblevenshtein::transducer::{OperationSet, OperationSetBuilder};
    /// let ops = OperationSetBuilder::new().with_standard_ops().build();
    /// for op in ops.iter() {
    ///     println!("{}", op);
    /// }
    /// ```
    #[inline]
    pub fn iter(&self) -> impl Iterator<Item = &OperationType> {
        self.operations.iter()
    }

    /// Create a standard Levenshtein operation set.
    ///
    /// Includes: match, substitution, insertion, deletion.
    ///
    /// # Example
    ///
    /// ```rust
    /// # use liblevenshtein::transducer::OperationSet;
    /// let ops = OperationSet::standard();
    /// assert_eq!(ops.len(), 4);
    /// ```
    pub fn standard() -> Self {
        OperationSetBuilder::new().with_standard_ops().build()
    }

    /// Create a Levenshtein operation set with transposition.
    ///
    /// Includes: match, substitution, insertion, deletion, transposition.
    ///
    /// # Example
    ///
    /// ```rust
    /// # use liblevenshtein::transducer::OperationSet;
    /// let ops = OperationSet::with_transposition();
    /// assert_eq!(ops.len(), 5);
    /// ```
    pub fn with_transposition() -> Self {
        OperationSetBuilder::new()
            .with_standard_ops()
            .with_transposition()
            .build()
    }

    /// Create a Levenshtein operation set with merge and split.
    ///
    /// Includes: match, substitution, insertion, deletion, merge, split.
    ///
    /// # Example
    ///
    /// ```rust
    /// # use liblevenshtein::transducer::OperationSet;
    /// let ops = OperationSet::with_merge_split();
    /// assert_eq!(ops.len(), 6);
    /// ```
    pub fn with_merge_split() -> Self {
        OperationSetBuilder::new()
            .with_standard_ops()
            .with_merge()
            .with_split()
            .build()
    }
}

impl Default for OperationSet {
    /// Create a standard Levenshtein operation set (default).
    #[inline]
    fn default() -> Self {
        Self::standard()
    }
}

/// Builder for composing operation sets with a fluent API.
///
/// # Examples
///
/// ## Standard Levenshtein
///
/// ```rust
/// # use liblevenshtein::transducer::OperationSetBuilder;
/// let ops = OperationSetBuilder::new()
///     .with_match()
///     .with_substitution()
///     .with_insertion()
///     .with_deletion()
///     .build();
/// ```
///
/// ## Weighted Operations
///
/// ```rust
/// # use liblevenshtein::transducer::{OperationSetBuilder, OperationType};
/// let ops = OperationSetBuilder::new()
///     .with_match()
///     .with_operation(OperationType::new(1, 1, 0.5, "cheap_subst"))
///     .with_insertion()
///     .with_deletion()
///     .build();
/// ```
///
/// ## Phonetic Corrections
///
/// ```rust
/// # use liblevenshtein::transducer::{OperationSetBuilder, OperationType, SubstitutionSet};
/// let mut phonetic = SubstitutionSet::new();
/// phonetic.allow_str("ph", "f");
///
/// let ops = OperationSetBuilder::new()
///     .with_match()
///     .with_operation(OperationType::with_restriction(
///         2, 1, 0.15,
///         phonetic,
///         "ph_to_f"
///     ))
///     .with_standard_ops()
///     .build();
/// ```
#[derive(Clone, Debug)]
pub struct OperationSetBuilder {
    /// Operations being built.
    operations: Vec<OperationType>,
}

impl OperationSetBuilder {
    /// Create a new builder with no operations.
    ///
    /// # Example
    ///
    /// ```rust
    /// # use liblevenshtein::transducer::OperationSetBuilder;
    /// let builder = OperationSetBuilder::new();
    /// ```
    #[inline]
    pub fn new() -> Self {
        Self {
            operations: Vec::new(),
        }
    }

    /// Add a custom operation.
    ///
    /// # Example
    ///
    /// ```rust
    /// # use liblevenshtein::transducer::{OperationSetBuilder, OperationType};
    /// let ops = OperationSetBuilder::new()
    ///     .with_operation(OperationType::new(1, 1, 0.5, "custom"))
    ///     .build();
    /// ```
    pub fn with_operation(mut self, op: OperationType) -> Self {
        self.operations.push(op);
        self
    }

    /// Add match operation: ⟨1, 1, 0.0⟩.
    ///
    /// Matches a single character with zero cost.
    ///
    /// # Example
    ///
    /// ```rust
    /// # use liblevenshtein::transducer::OperationSetBuilder;
    /// let ops = OperationSetBuilder::new()
    ///     .with_match()
    ///     .build();
    /// ```
    pub fn with_match(self) -> Self {
        self.with_operation(OperationType::new(1, 1, 0.0, "match"))
    }

    /// Add substitution operation: ⟨1, 1, 1.0⟩.
    ///
    /// Substitutes one character for another with cost 1.
    ///
    /// # Example
    ///
    /// ```rust
    /// # use liblevenshtein::transducer::OperationSetBuilder;
    /// let ops = OperationSetBuilder::new()
    ///     .with_substitution()
    ///     .build();
    /// ```
    pub fn with_substitution(self) -> Self {
        self.with_operation(OperationType::new(1, 1, 1.0, "substitute"))
    }

    /// Add insertion operation: ⟨0, 1, 1.0⟩.
    ///
    /// Inserts a character from the query with cost 1.
    ///
    /// # Example
    ///
    /// ```rust
    /// # use liblevenshtein::transducer::OperationSetBuilder;
    /// let ops = OperationSetBuilder::new()
    ///     .with_insertion()
    ///     .build();
    /// ```
    pub fn with_insertion(self) -> Self {
        self.with_operation(OperationType::new(0, 1, 1.0, "insert"))
    }

    /// Add deletion operation: ⟨1, 0, 1.0⟩.
    ///
    /// Deletes a character from the dictionary with cost 1.
    ///
    /// # Example
    ///
    /// ```rust
    /// # use liblevenshtein::transducer::OperationSetBuilder;
    /// let ops = OperationSetBuilder::new()
    ///     .with_deletion()
    ///     .build();
    /// ```
    pub fn with_deletion(self) -> Self {
        self.with_operation(OperationType::new(1, 0, 1.0, "delete"))
    }

    /// Add transposition operation: ⟨2, 2, 1.0⟩.
    ///
    /// Swaps two adjacent characters with cost 1.
    ///
    /// # Example
    ///
    /// ```rust
    /// # use liblevenshtein::transducer::OperationSetBuilder;
    /// let ops = OperationSetBuilder::new()
    ///     .with_transposition()
    ///     .build();
    /// ```
    pub fn with_transposition(self) -> Self {
        self.with_operation(OperationType::new(2, 2, 1.0, "transpose"))
    }

    /// Add merge operation: ⟨1, 2, 1.0⟩.
    ///
    /// Merges two characters in the query into one in the dictionary.
    /// Example: "every one" → "everyone"
    ///
    /// # Example
    ///
    /// ```rust
    /// # use liblevenshtein::transducer::OperationSetBuilder;
    /// let ops = OperationSetBuilder::new()
    ///     .with_merge()
    ///     .build();
    /// ```
    pub fn with_merge(self) -> Self {
        self.with_operation(OperationType::new(1, 2, 1.0, "merge"))
    }

    /// Add split operation: ⟨2, 1, 1.0⟩.
    ///
    /// Splits one character in the dictionary into two in the query.
    /// Example: "everyone" → "every one"
    ///
    /// # Example
    ///
    /// ```rust
    /// # use liblevenshtein::transducer::OperationSetBuilder;
    /// let ops = OperationSetBuilder::new()
    ///     .with_split()
    ///     .build();
    /// ```
    pub fn with_split(self) -> Self {
        self.with_operation(OperationType::new(2, 1, 1.0, "split"))
    }

    /// Add all standard operations: match, substitution, insertion, deletion.
    ///
    /// This is a convenience method equivalent to calling:
    /// - `with_match()`
    /// - `with_substitution()`
    /// - `with_insertion()`
    /// - `with_deletion()`
    ///
    /// # Example
    ///
    /// ```rust
    /// # use liblevenshtein::transducer::OperationSetBuilder;
    /// let ops = OperationSetBuilder::new()
    ///     .with_standard_ops()
    ///     .build();
    ///
    /// assert_eq!(ops.len(), 4);
    /// ```
    pub fn with_standard_ops(self) -> Self {
        self.with_match()
            .with_substitution()
            .with_insertion()
            .with_deletion()
    }

    /// Build the final operation set.
    ///
    /// Consumes the builder and returns the constructed [`OperationSet`].
    ///
    /// # Example
    ///
    /// ```rust
    /// # use liblevenshtein::transducer::OperationSetBuilder;
    /// let ops = OperationSetBuilder::new()
    ///     .with_standard_ops()
    ///     .build();
    /// ```
    #[inline]
    pub fn build(self) -> OperationSet {
        OperationSet {
            operations: self.operations,
        }
    }
}

impl Default for OperationSetBuilder {
    /// Create a new empty builder (default).
    #[inline]
    fn default() -> Self {
        Self::new()
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::transducer::SubstitutionSet;

    #[test]
    fn test_empty_operation_set() {
        let ops = OperationSet::new();
        assert_eq!(ops.len(), 0);
        assert!(ops.is_empty());
    }

    #[test]
    fn test_standard_operations() {
        let ops = OperationSet::standard();
        assert_eq!(ops.len(), 4);
        assert!(!ops.is_empty());

        let names: Vec<_> = ops.iter().map(|op| op.name()).collect();
        assert!(names.contains(&"match"));
        assert!(names.contains(&"substitute"));
        assert!(names.contains(&"insert"));
        assert!(names.contains(&"delete"));
    }

    #[test]
    fn test_with_transposition() {
        let ops = OperationSet::with_transposition();
        assert_eq!(ops.len(), 5);

        let names: Vec<_> = ops.iter().map(|op| op.name()).collect();
        assert!(names.contains(&"transpose"));
    }

    #[test]
    fn test_with_merge_split() {
        let ops = OperationSet::with_merge_split();
        assert_eq!(ops.len(), 6);

        let names: Vec<_> = ops.iter().map(|op| op.name()).collect();
        assert!(names.contains(&"merge"));
        assert!(names.contains(&"split"));
    }

    #[test]
    fn test_builder_standard() {
        let ops = OperationSetBuilder::new().with_standard_ops().build();

        assert_eq!(ops.len(), 4);
    }

    #[test]
    fn test_builder_custom() {
        let ops = OperationSetBuilder::new()
            .with_match()
            .with_operation(OperationType::new(1, 1, 0.5, "cheap_subst"))
            .with_insertion()
            .build();

        assert_eq!(ops.len(), 3);

        let custom_op = ops
            .operations()
            .iter()
            .find(|op| op.name() == "cheap_subst")
            .expect("Custom operation should exist");

        assert_eq!(custom_op.weight(), 0.5);
    }

    #[test]
    fn test_builder_phonetic() {
        let mut phonetic = SubstitutionSet::new();
        phonetic.allow_str("ph", "f");

        let ops = OperationSetBuilder::new()
            .with_match()
            .with_operation(OperationType::with_restriction(
                2, 1, 0.15, phonetic, "ph_to_f",
            ))
            .with_standard_ops()
            .build();

        // Builder allows duplicates: match (explicit) + ph_to_f + standard_ops (match + substitute + insert + delete)
        assert_eq!(ops.len(), 6);

        let ph_op = ops
            .operations()
            .iter()
            .find(|op| op.name() == "ph_to_f")
            .expect("Phonetic operation should exist");

        assert!(ph_op.is_restricted());
        assert_eq!(ph_op.consume_x(), 2);
        assert_eq!(ph_op.consume_y(), 1);
    }

    #[test]
    fn test_default_is_standard() {
        let ops = OperationSet::default();
        assert_eq!(ops.len(), 4);
    }

    #[test]
    fn test_iteration() {
        let ops = OperationSet::standard();
        let count = ops.iter().count();
        assert_eq!(count, 4);
    }

    #[test]
    fn test_add_operation() {
        let mut ops = OperationSet::new();
        ops.add(OperationType::new(1, 1, 0.0, "match"));
        assert_eq!(ops.len(), 1);
    }
}
