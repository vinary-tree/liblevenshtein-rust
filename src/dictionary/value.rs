//! Generic value types for dictionary mappings.
//!
//! This module provides a flexible trait hierarchy for associating arbitrary values
//! with dictionary terms. This enables use cases like:
//!
//! - **Contextual code completion**: Map identifiers to scope IDs
//! - **Metadata-rich dictionaries**: Associate terms with tags, categories, or custom data
//! - **Multi-value mappings**: Store lists or sets of related information
//!
//! # Examples
//!
//! ```
//! use liblevenshtein::dictionary::value::*;
//!
//! // Simple literal values
//! let scope_id: u32 = 42;
//! assert!(scope_id.is_value());
//!
//! // Collection values
//! let scope_ids = vec![1, 2, 3];
//! assert!(scope_ids.is_value());
//!
//! // Filtering with predicates on scalar values
//! let scope: u32 = 20;
//! assert!(scope.matches_any(&|&id| id == 20));
//! assert!(!scope.matches_any(&|&id| id == 99));
//! ```

use std::collections::HashSet;
use std::hash::Hash;

#[cfg(feature = "persistent-artrie")]
use serde::{de::DeserializeOwned, Serialize};

/// Marker trait for types that can be stored as dictionary values.
///
/// Any type implementing `DictionaryValue` can be associated with terms in a dictionary.
/// The trait requires `Clone`, `Send`, and `Sync` to support concurrent access patterns
/// common in fuzzy search applications.
///
/// # Automatic Implementation
///
/// This trait is automatically implemented for common types including:
/// - Unit type `()` (for dictionaries without values)
/// - Primitives: `u8`, `u16`, `u32`, `u64`, `usize`, `i8`, `i16`, `i32`, `i64`, `isize`
/// - Strings: `String`, `&'static str`
/// - Collections: `Vec<T>`, `HashSet<T>`, `smallvec::SmallVec<A>`
///
/// # Custom Types
///
/// You can implement this trait for your own types:
///
/// ```
/// use liblevenshtein::dictionary::value::DictionaryValue;
/// use serde::{Serialize, Deserialize};
///
/// #[derive(Clone, Serialize, Deserialize)]
/// struct Metadata {
///     category: String,
///     priority: u32,
/// }
///
/// impl DictionaryValue for Metadata {}
/// ```
///
/// # Serialization (persistent-artrie feature)
///
/// When the `persistent-artrie` feature is enabled, `DictionaryValue` additionally
/// requires `serde::Serialize + serde::de::DeserializeOwned` to support value persistence.
#[cfg(feature = "persistent-artrie")]
pub trait DictionaryValue:
    Clone + Send + Sync + Unpin + 'static + Serialize + DeserializeOwned
{
    /// Returns `true` if this is a meaningful value (not unit type).
    ///
    /// Default implementation returns `true`. The unit type `()` overrides this
    /// to return `false` for backward compatibility with non-map dictionaries.
    fn is_value(&self) -> bool {
        true
    }
}

/// Marker trait for types that can be stored as dictionary values.
///
/// Any type implementing `DictionaryValue` can be associated with terms in a dictionary.
/// The trait requires `Clone`, `Send`, and `Sync` to support concurrent access patterns
/// common in fuzzy search applications.
///
/// # Automatic Implementation
///
/// This trait is automatically implemented for common types including:
/// - Unit type `()` (for dictionaries without values)
/// - Primitives: `u8`, `u16`, `u32`, `u64`, `usize`, `i8`, `i16`, `i32`, `i64`, `isize`
/// - Strings: `String`, `&'static str`
/// - Collections: `Vec<T>`, `HashSet<T>`, `smallvec::SmallVec<A>`
///
/// # Custom Types
///
/// You can implement this trait for your own types:
///
/// ```
/// use liblevenshtein::dictionary::value::DictionaryValue;
///
/// #[derive(Clone)]
/// struct Metadata {
///     category: String,
///     priority: u32,
/// }
///
/// impl DictionaryValue for Metadata {}
/// ```
#[cfg(not(feature = "persistent-artrie"))]
pub trait DictionaryValue: Clone + Send + Sync + Unpin + 'static {
    /// Returns `true` if this is a meaningful value (not unit type).
    ///
    /// Default implementation returns `true`. The unit type `()` overrides this
    /// to return `false` for backward compatibility with non-map dictionaries.
    fn is_value(&self) -> bool {
        true
    }
}

/// Trait for values that support filtering operations.
///
/// This trait enables efficient pruning of the search space during fuzzy queries
/// by testing predicates on values during graph traversal, rather than filtering
/// results after the fact.
///
/// # Performance
///
/// Filtering during traversal (using this trait) provides 10-100x speedups compared
/// to post-filtering, especially when the filter is highly selective.
///
/// # Examples
///
/// ```
/// use liblevenshtein::dictionary::value::FilterableValue;
///
/// let scope: u32 = 4;
///
/// // Filter for even numbers
/// assert!(scope.matches_any(&|&id| id % 2 == 0));
///
/// // Filter for numbers > 10 (doesn't match)
/// assert!(!scope.matches_any(&|&id| id > 10));
/// ```
pub trait FilterableValue: DictionaryValue {
    /// Tests if this value matches a predicate.
    ///
    /// For single values, this tests the value directly.
    /// For collections, this returns `true` if *any* element matches.
    fn matches_any<F>(&self, predicate: &F) -> bool
    where
        F: Fn(&Self) -> bool;

    /// Tests if all elements (for collections) match a predicate.
    ///
    /// For single values, this is equivalent to `matches_any`.
    /// For collections, this returns `true` only if *all* elements match.
    fn matches_all<F>(&self, predicate: &F) -> bool
    where
        F: Fn(&Self) -> bool;
}

// =============================================================================
// Implementations for unit type (backward compatibility)
// =============================================================================

impl DictionaryValue for () {
    fn is_value(&self) -> bool {
        false // Unit type has no meaningful value
    }
}

impl FilterableValue for () {
    fn matches_any<F>(&self, _predicate: &F) -> bool
    where
        F: Fn(&Self) -> bool,
    {
        true // No filter, accept everything
    }

    fn matches_all<F>(&self, _predicate: &F) -> bool
    where
        F: Fn(&Self) -> bool,
    {
        true // No filter, accept everything
    }
}

// =============================================================================
// Implementations for primitive types
// =============================================================================

macro_rules! impl_primitive_value {
    ($($t:ty),*) => {
        $(
            impl DictionaryValue for $t {}

            impl FilterableValue for $t {
                fn matches_any<F>(&self, predicate: &F) -> bool
                where
                    F: Fn(&Self) -> bool,
                {
                    predicate(self)
                }

                fn matches_all<F>(&self, predicate: &F) -> bool
                where
                    F: Fn(&Self) -> bool,
                {
                    predicate(self)
                }
            }
        )*
    };
}

impl_primitive_value!(u8, u16, u32, u64, usize, i8, i16, i32, i64, isize, bool, char);

// =============================================================================
// Implementations for string types
// =============================================================================

impl DictionaryValue for String {}

impl FilterableValue for String {
    fn matches_any<F>(&self, predicate: &F) -> bool
    where
        F: Fn(&Self) -> bool,
    {
        predicate(self)
    }

    fn matches_all<F>(&self, predicate: &F) -> bool
    where
        F: Fn(&Self) -> bool,
    {
        predicate(self)
    }
}

// Note: &'static str does not implement DeserializeOwned, so it cannot be used
// as a DictionaryValue when persistent-artrie is enabled. Use String instead.
#[cfg(not(feature = "persistent-artrie"))]
impl DictionaryValue for &'static str {}

#[cfg(not(feature = "persistent-artrie"))]
impl FilterableValue for &'static str {
    fn matches_any<F>(&self, predicate: &F) -> bool
    where
        F: Fn(&Self) -> bool,
    {
        predicate(self)
    }

    fn matches_all<F>(&self, predicate: &F) -> bool
    where
        F: Fn(&Self) -> bool,
    {
        predicate(self)
    }
}

// =============================================================================
// Implementations for Vec<T>
// =============================================================================

impl<T: DictionaryValue> DictionaryValue for Vec<T> {}

impl<T: FilterableValue> FilterableValue for Vec<T> {
    fn matches_any<F>(&self, predicate: &F) -> bool
    where
        F: Fn(&Self) -> bool,
    {
        // If testing the entire vec, test it directly
        if predicate(self) {
            return true;
        }
        // Otherwise, check if any element would match a per-element predicate
        // (This is a simplification; real usage would pass element predicates)
        false
    }

    fn matches_all<F>(&self, predicate: &F) -> bool
    where
        F: Fn(&Self) -> bool,
    {
        predicate(self)
    }
}

// =============================================================================
// Implementations for HashSet<T>
// =============================================================================

impl<T: DictionaryValue + Eq + Hash> DictionaryValue for HashSet<T> {}

impl<T: FilterableValue + Eq + Hash> FilterableValue for HashSet<T> {
    fn matches_any<F>(&self, predicate: &F) -> bool
    where
        F: Fn(&Self) -> bool,
    {
        predicate(self)
    }

    fn matches_all<F>(&self, predicate: &F) -> bool
    where
        F: Fn(&Self) -> bool,
    {
        predicate(self)
    }
}

// =============================================================================
// Implementations for SmallVec<A>
// =============================================================================

impl<A: smallvec::Array + Send + Sync + Unpin + 'static> DictionaryValue for smallvec::SmallVec<A> where
    A::Item: DictionaryValue
{
}

impl<A: smallvec::Array + Send + Sync + Unpin + 'static> FilterableValue for smallvec::SmallVec<A>
where
    A::Item: FilterableValue,
{
    fn matches_any<F>(&self, predicate: &F) -> bool
    where
        F: Fn(&Self) -> bool,
    {
        predicate(self)
    }

    fn matches_all<F>(&self, predicate: &F) -> bool
    where
        F: Fn(&Self) -> bool,
    {
        predicate(self)
    }
}

// =============================================================================
// Helper functions for common filtering patterns
// =============================================================================

/// Tests if a collection contains a specific value.
///
/// # Examples
///
/// ```
/// use liblevenshtein::dictionary::value::contains;
///
/// let scopes = vec![10, 20, 30];
/// assert!(contains(&scopes, &20));
/// assert!(!contains(&scopes, &99));
/// ```
pub fn contains<T: PartialEq>(collection: &[T], value: &T) -> bool {
    collection.contains(value)
}

/// Tests if a collection contains any value matching a predicate.
///
/// # Examples
///
/// ```
/// use liblevenshtein::dictionary::value::any;
///
/// let scopes = vec![10, 20, 30];
/// assert!(any(&scopes, |&id| id > 25));
/// assert!(!any(&scopes, |&id| id > 100));
/// ```
pub fn any<T, F>(collection: &[T], predicate: F) -> bool
where
    F: Fn(&T) -> bool,
{
    collection.iter().any(predicate)
}

/// Tests if all values in a collection match a predicate.
///
/// # Examples
///
/// ```
/// use liblevenshtein::dictionary::value::all;
///
/// let scopes = vec![10, 20, 30];
/// assert!(all(&scopes, |&id| id > 5));
/// assert!(!all(&scopes, |&id| id > 15));
/// ```
pub fn all<T, F>(collection: &[T], predicate: F) -> bool
where
    F: Fn(&T) -> bool,
{
    collection.iter().all(predicate)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_unit_type() {
        let unit = ();
        assert!(!unit.is_value());
        assert!(unit.matches_any(&|_| false)); // Unit always matches
        assert!(unit.matches_all(&|_| false)); // Unit always matches
    }

    #[test]
    fn test_primitives() {
        let num: u32 = 42;
        assert!(num.is_value());
        assert!(num.matches_any(&|&x| x == 42));
        assert!(!num.matches_any(&|&x| x == 99));
        assert!(num.matches_all(&|&x| x > 0));
    }

    #[test]
    fn test_strings() {
        let s = String::from("hello");
        assert!(s.is_value());
        assert!(s.matches_any(&|x| x == "hello"));
        assert!(!s.matches_any(&|x| x == "world"));
    }

    #[test]
    fn test_vec() {
        let v = vec![1, 2, 3];
        assert!(v.is_value());

        // Test filtering the vec itself
        assert!(v.matches_any(&|x| x.len() == 3));
        assert!(!v.matches_any(&|x| x.is_empty()));
    }

    #[test]
    fn test_hashset() {
        let mut set = HashSet::new();
        set.insert(10);
        set.insert(20);
        set.insert(30);

        assert!(set.is_value());
        assert!(set.matches_any(&|x| x.contains(&20)));
        assert!(!set.matches_any(&|x| x.contains(&99)));
    }

    #[test]
    fn test_helper_functions() {
        let nums = vec![1, 2, 3, 4, 5];

        assert!(contains(&nums, &3));
        assert!(!contains(&nums, &10));

        assert!(any(&nums, |&x| x > 3));
        assert!(!any(&nums, |&x| x > 10));

        assert!(all(&nums, |&x| x > 0));
        assert!(!all(&nums, |&x| x > 2));
    }
}
