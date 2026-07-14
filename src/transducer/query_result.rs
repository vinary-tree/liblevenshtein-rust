//! Trait for polymorphic query iterator results.
//!
//! This module provides the `QueryResult` trait that allows `QueryIterator`
//! to return different result types (just the term, or term + distance) without
//! code duplication or performance overhead.
//!
//! This design mirrors the C++ type-specialization approach and Java's
//! factory pattern, providing zero-cost abstraction through Rust's
//! monomorphization.

use super::query::{Candidate, UnitCandidate};
use libdictenstein::CharUnit;

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
pub trait QueryResult<U: CharUnit>: Sized {
    /// Convert a match into the result type.
    ///
    /// # Parameters
    /// - `units`: The matched dictionary term as its raw unit sequence
    /// - `distance`: The edit distance from the query
    ///
    /// # Returns
    /// The result in the appropriate format. `String`/`Candidate` reconstruct the
    /// term text via [`CharUnit::to_string`]; `Vec<U>`/`UnitCandidate` keep the units
    /// verbatim (lossless for `u64` token sequences, whose `to_string` is a lossy
    /// byte-unpack).
    fn from_match(units: &[U], distance: usize) -> Self;
}

/// Implementation for `String`: reconstructs the term text, ignoring distance.
///
/// For a `u64` (token-sequence) dictionary this is a **lossy** byte-unpack; prefer a
/// `Vec<u64>` / [`UnitCandidate`] result there.
impl<U: CharUnit> QueryResult<U> for String {
    #[inline]
    fn from_match(units: &[U], _distance: usize) -> Self {
        U::to_string(units)
    }
}

/// Implementation for `Candidate`: reconstructs the term text plus its distance.
impl<U: CharUnit> QueryResult<U> for Candidate {
    #[inline]
    fn from_match(units: &[U], distance: usize) -> Self {
        Candidate {
            term: U::to_string(units),
            distance,
        }
    }
}

/// Implementation for `Vec<U>`: the matched term as its raw unit sequence — the
/// units-native result (e.g. a `Vec<u64>` token-id sequence), with no `String`
/// round-trip. Lossless for every alphabet.
impl<U: CharUnit> QueryResult<U> for Vec<U> {
    #[inline]
    fn from_match(units: &[U], _distance: usize) -> Self {
        units.to_vec()
    }
}

/// Implementation for [`UnitCandidate`]: units-native term plus its distance.
impl<U: CharUnit> QueryResult<U> for UnitCandidate<U> {
    #[inline]
    fn from_match(units: &[U], distance: usize) -> Self {
        UnitCandidate {
            term: units.to_vec(),
            distance,
        }
    }
}
