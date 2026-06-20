//! Trait for polymorphic query iterator results.
//!
//! This module provides the `QueryResult` trait that allows `QueryIterator`
//! to return different result types (just the term, or term + distance) without
//! code duplication or performance overhead.
//!
//! This design mirrors the C++ type-specialization approach and Java's
//! factory pattern, providing zero-cost abstraction through Rust's
//! monomorphization.

use super::query::Candidate;

/// Trait for converting a match (term + distance) into a result type.
///
/// This enables `QueryIterator<N, R>` to be generic over the result type,
/// allowing it to return either:
/// - Just the term (`String`)
/// - Term with distance (`Candidate`)
/// - Custom user-defined types
///
/// The distance is computed once during automaton traversal, then converted
/// to the appropriate result type via this trait.
///
/// # Examples
///
/// ```no_run
/// use liblevenshtein::prelude::*;
/// use liblevenshtein::transducer::{QueryIterator, Candidate};
///
/// let dict = DoubleArrayTrie::from_terms(vec!["test", "testing"]);
/// let root = dict.root();
///
/// // Iterator that returns just strings
/// let iter: QueryIterator<_, String> = QueryIterator::new(
///     root.clone(),
///     "tset".to_string(),
///     2,
///     Algorithm::Standard
/// );
/// for term in iter {
///     println!("{}", term);
/// }
///
/// // Iterator that returns Candidate (term + distance)
/// let iter: QueryIterator<_, Candidate> = QueryIterator::new(
///     root,
///     "tset".to_string(),
///     2,
///     Algorithm::Standard
/// );
/// for candidate in iter {
///     println!("{}: distance {}", candidate.term, candidate.distance);
/// }
/// ```
pub trait QueryResult: Sized {
    /// Convert a match into the result type.
    ///
    /// # Parameters
    /// - `term`: The matched dictionary term
    /// - `distance`: The edit distance from the query
    ///
    /// # Returns
    /// The result in the appropriate format (String, Candidate, etc.)
    fn from_match(term: String, distance: usize) -> Self;
}

/// Implementation for String: returns just the term, ignoring distance.
impl QueryResult for String {
    #[inline]
    fn from_match(term: String, _distance: usize) -> Self {
        term
    }
}

/// Implementation for Candidate: returns both term and distance.
impl QueryResult for Candidate {
    #[inline]
    fn from_match(term: String, distance: usize) -> Self {
        Candidate { term, distance }
    }
}
