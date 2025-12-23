//! Lazy initialization wrappers for sparse dictionaries.
//!
//! This module provides wrappers that lazily initialize values for dictionary
//! entries when they are accessed but don't have an associated value. This is
//! useful for:
//!
//! - Providing default values for entries without storage overhead
//! - Computing values on-demand based on context
//! - Implementing fallback strategies for sparse dictionaries
//! - Memoizing expensive computations
//!
//! # Wrapper Variants
//!
//! Three variants are provided, optimized for different use cases:
//!
//! 1. `LazyInitDefault<D>` - Zero-cost wrapper for `Default` values
//! 2. `LazyInitFn<D, V>` - Function pointer for cheap initialization
//! 3. `LazyInit<D, F>` - Full closure support for complex initialization
//!
//! # Examples
//!
//! ## Default Initialization
//!
//! ```rust,ignore
//! use liblevenshtein::prelude::*;
//! use liblevenshtein::dictionary::MappedDictionary;
//! use liblevenshtein::cache::eviction::LazyInitDefault;
//!
//! let dict = PathMapDictionary::<i32>::from_terms(["foo", "bar"]);
//! let lazy = LazyInitDefault::new(dict);
//!
//! // Returns 0 (default i32) for terms without values
//! assert_eq!(lazy.get_value("foo"), Some(0));
//! ```
//!
//! ## Custom Function
//!
//! ```rust,ignore
//! use liblevenshtein::prelude::*;
//! use liblevenshtein::dictionary::MappedDictionary;
//! use liblevenshtein::cache::eviction::LazyInitFn;
//!
//! fn generate_id() -> u64 {
//!     use std::sync::atomic::{AtomicU64, Ordering};
//!     static COUNTER: AtomicU64 = AtomicU64::new(1);
//!     COUNTER.fetch_add(1, Ordering::Relaxed)
//! }
//!
//! let dict: PathMapDictionary<u64> = PathMapDictionary::from_terms(["foo", "bar"]);
//! let lazy = LazyInitFn::new(dict, generate_id);
//!
//! // Each access generates a new ID
//! assert!(lazy.get_value("foo").is_some());
//! ```
//!
//! ## Context-Aware Initialization
//!
//! ```rust,ignore
//! use liblevenshtein::prelude::*;
//! use liblevenshtein::dictionary::MappedDictionary;
//! use liblevenshtein::cache::eviction::LazyInit;
//!
//! let user_id = "user_123";
//! let dict: PathMapDictionary = PathMapDictionary::from_terms(["setting1", "setting2"]);
//!
//! let lazy = LazyInit::new(dict, || {
//!     // Capture context in closure
//!     compute_user_setting(user_id)
//! });
//!
//! fn compute_user_setting(user: &str) -> String {
//!     format!("default_for_{}", user)
//! }
//! ```

use crate::dictionary::{Dictionary, DictionaryValue, MappedDictionary, SyncStrategy};

//==============================================================================
// LazyInitDefault - Zero-cost wrapper for Default values
//==============================================================================

/// Lazy initialization wrapper that provides `Default` values.
///
/// This is the most efficient variant, with zero overhead for the
/// initializer (it's just `V::default()`). Use this when you want
/// to provide default values for missing entries.
///
/// # Type Parameters
///
/// - `D`: Inner dictionary type
///
/// The value type `V` is inferred from the dictionary's `MappedDictionary::Value`.
#[derive(Debug, Clone)]
pub struct LazyInitDefault<D> {
    inner: D,
}

impl<D> LazyInitDefault<D> {
    /// Creates a new lazy-default wrapper.
    ///
    /// # Arguments
    ///
    /// - `dict`: The dictionary to wrap
    ///
    /// # Examples
    ///
    /// ```rust,ignore
    /// use liblevenshtein::prelude::*;
    /// use liblevenshtein::cache::eviction::LazyInitDefault;
    ///
    /// let dict = PathMapDictionary::<i32>::from_terms(["foo", "bar"]);
    /// let lazy = LazyInitDefault::new(dict);
    /// ```
    #[inline]
    pub fn new(dict: D) -> Self {
        Self { inner: dict }
    }

    /// Unwraps the inner dictionary.
    #[inline]
    pub fn into_inner(self) -> D {
        self.inner
    }

    /// Gets a reference to the inner dictionary.
    #[inline]
    pub fn inner(&self) -> &D {
        &self.inner
    }
}

impl<D> Dictionary for LazyInitDefault<D>
where
    D: Dictionary,
{
    type Node = D::Node;

    #[inline]
    fn root(&self) -> Self::Node {
        self.inner.root()
    }

    #[inline]
    fn len(&self) -> Option<usize> {
        self.inner.len()
    }

    #[inline]
    fn contains(&self, term: &str) -> bool {
        self.inner.contains(term)
    }

    #[inline]
    fn sync_strategy(&self) -> SyncStrategy {
        self.inner.sync_strategy()
    }
}

impl<D, V> MappedDictionary for LazyInitDefault<D>
where
    D: MappedDictionary<Value = V>,
    V: DictionaryValue + Default,
{
    type Value = V;

    #[inline(always)]
    fn get_value(&self, term: &str) -> Option<Self::Value> {
        // Try inner dictionary first
        if let Some(value) = self.inner.get_value(term) {
            return Some(value);
        }

        // If term exists in dictionary but has no value, return default
        if self.inner.contains(term) {
            return Some(V::default());
        }

        None
    }

    #[inline]
    fn contains_with_value<F>(&self, term: &str, predicate: F) -> bool
    where
        F: Fn(&Self::Value) -> bool,
    {
        // Check if inner has value matching predicate
        if self.inner.contains_with_value(term, &predicate) {
            return true;
        }

        // Check if term exists and default matches predicate
        if self.inner.contains(term) {
            let default_value = V::default();
            return predicate(&default_value);
        }

        false
    }
}

//==============================================================================
// LazyInitFn - Function pointer for cheap initialization
//==============================================================================

/// Lazy initialization wrapper using a function pointer.
///
/// This variant uses a function pointer instead of a closure, which has
/// minimal overhead. Use this when your initializer is a simple function
/// that doesn't need to capture any context.
///
/// # Type Parameters
///
/// - `D`: Inner dictionary type
/// - `V`: Value type
#[derive(Debug, Clone)]
pub struct LazyInitFn<D, V> {
    inner: D,
    initializer: fn() -> V,
}

impl<D, V> LazyInitFn<D, V> {
    /// Creates a new lazy-fn wrapper with a function pointer initializer.
    ///
    /// # Arguments
    ///
    /// - `dict`: The dictionary to wrap
    /// - `initializer`: Function pointer that generates values
    ///
    /// # Examples
    ///
    /// ```rust,ignore
    /// use liblevenshtein::prelude::*;
    /// use liblevenshtein::cache::eviction::LazyInitFn;
    ///
    /// fn zero() -> i32 { 0 }
    ///
    /// let dict: PathMapDictionary = PathMapDictionary::from_terms(["foo", "bar"]);
    /// let lazy = LazyInitFn::new(dict, zero);
    /// ```
    #[inline]
    pub fn new(dict: D, initializer: fn() -> V) -> Self {
        Self {
            inner: dict,
            initializer,
        }
    }

    /// Unwraps the inner dictionary.
    #[inline]
    pub fn into_inner(self) -> D {
        self.inner
    }

    /// Gets a reference to the inner dictionary.
    #[inline]
    pub fn inner(&self) -> &D {
        &self.inner
    }
}

impl<D, V> Dictionary for LazyInitFn<D, V>
where
    D: Dictionary,
    V: DictionaryValue,
{
    type Node = D::Node;

    #[inline]
    fn root(&self) -> Self::Node {
        self.inner.root()
    }

    #[inline]
    fn len(&self) -> Option<usize> {
        self.inner.len()
    }

    #[inline]
    fn contains(&self, term: &str) -> bool {
        self.inner.contains(term)
    }

    #[inline]
    fn sync_strategy(&self) -> SyncStrategy {
        self.inner.sync_strategy()
    }
}

impl<D, V> MappedDictionary for LazyInitFn<D, V>
where
    D: MappedDictionary<Value = V>,
    V: DictionaryValue,
{
    type Value = V;

    #[inline]
    fn get_value(&self, term: &str) -> Option<Self::Value> {
        // Try inner dictionary first
        if let Some(value) = self.inner.get_value(term) {
            return Some(value);
        }

        // If term exists in dictionary but has no value, initialize
        if self.inner.contains(term) {
            return Some((self.initializer)());
        }

        None
    }

    #[inline]
    fn contains_with_value<F>(&self, term: &str, predicate: F) -> bool
    where
        F: Fn(&Self::Value) -> bool,
    {
        // Check if inner has value matching predicate
        if self.inner.contains_with_value(term, &predicate) {
            return true;
        }

        // Check if term exists and initialized value matches predicate
        if self.inner.contains(term) {
            let value = (self.initializer)();
            return predicate(&value);
        }

        false
    }
}

//==============================================================================
// LazyInit - Full closure support for complex initialization
//==============================================================================

/// Lazy initialization wrapper using a closure.
///
/// This variant supports full closure capture, allowing the initializer
/// to access context. This is the most flexible but has slightly higher
/// overhead due to the closure.
///
/// # Type Parameters
///
/// - `D`: Inner dictionary type
/// - `F`: Closure type that returns values
#[derive(Clone)]
pub struct LazyInit<D, F> {
    inner: D,
    initializer: F,
}

impl<D, F> LazyInit<D, F> {
    /// Creates a new lazy-init wrapper with a closure initializer.
    ///
    /// # Arguments
    ///
    /// - `dict`: The dictionary to wrap
    /// - `initializer`: Closure that generates values
    ///
    /// # Examples
    ///
    /// ```rust,ignore
    /// use liblevenshtein::prelude::*;
    /// use liblevenshtein::cache::eviction::LazyInit;
    ///
    /// let context = "user_123";
    /// let dict: PathMapDictionary = PathMapDictionary::from_terms(["setting1", "setting2"]);
    ///
    /// let lazy = LazyInit::new(dict, || {
    ///     format!("default_for_{}", context)
    /// });
    /// ```
    #[inline]
    pub fn new(dict: D, initializer: F) -> Self {
        Self {
            inner: dict,
            initializer,
        }
    }

    /// Unwraps the inner dictionary.
    #[inline]
    pub fn into_inner(self) -> D {
        self.inner
    }

    /// Gets a reference to the inner dictionary.
    #[inline]
    pub fn inner(&self) -> &D {
        &self.inner
    }
}

impl<D, F> Dictionary for LazyInit<D, F>
where
    D: Dictionary,
{
    type Node = D::Node;

    #[inline]
    fn root(&self) -> Self::Node {
        self.inner.root()
    }

    #[inline]
    fn len(&self) -> Option<usize> {
        self.inner.len()
    }

    #[inline]
    fn contains(&self, term: &str) -> bool {
        self.inner.contains(term)
    }

    #[inline]
    fn sync_strategy(&self) -> SyncStrategy {
        self.inner.sync_strategy()
    }
}

impl<D, F, V> MappedDictionary for LazyInit<D, F>
where
    D: MappedDictionary<Value = V>,
    F: Fn() -> V,
    V: DictionaryValue,
{
    type Value = V;

    #[inline]
    fn get_value(&self, term: &str) -> Option<Self::Value> {
        // Try inner dictionary first
        if let Some(value) = self.inner.get_value(term) {
            return Some(value);
        }

        // If term exists in dictionary but has no value, initialize
        if self.inner.contains(term) {
            return Some((self.initializer)());
        }

        None
    }

    #[inline]
    fn contains_with_value<F2>(&self, term: &str, predicate: F2) -> bool
    where
        F2: Fn(&Self::Value) -> bool,
    {
        // Check if inner has value matching predicate
        if self.inner.contains_with_value(term, &predicate) {
            return true;
        }

        // Check if term exists and initialized value matches predicate
        if self.inner.contains(term) {
            let value = (self.initializer)();
            return predicate(&value);
        }

        false
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    #[cfg(feature = "pathmap-backend")]
    use crate::dictionary::pathmap::PathMapDictionary;
    use std::sync::atomic::{AtomicUsize, Ordering};
    use std::sync::Arc;

    #[test]
    #[cfg(feature = "pathmap-backend")]
    fn test_lazy_init_default() {
        let dict = PathMapDictionary::<i32>::from_terms(["foo", "bar", "baz"]);
        let lazy = LazyInitDefault::new(dict);

        // Terms exist but have no values - should return default
        assert_eq!(lazy.get_value("foo"), Some(0));
        assert_eq!(lazy.get_value("bar"), Some(0));

        // Term doesn't exist - should return None
        assert_eq!(lazy.get_value("missing"), None);
    }

    #[test]
    #[cfg(feature = "pathmap-backend")]
    fn test_lazy_init_default_with_some_values() {
        let dict = PathMapDictionary::from_terms_with_values([
            ("foo", 42),
            ("bar", 0), // Explicit zero
        ]);
        // Add term without value
        dict.insert("baz");

        let lazy = LazyInitDefault::new(dict);

        // Has explicit value
        assert_eq!(lazy.get_value("foo"), Some(42));

        // Has explicit zero (not default)
        assert_eq!(lazy.get_value("bar"), Some(0));

        // No value - returns default
        assert_eq!(lazy.get_value("baz"), Some(0));

        // Doesn't exist
        assert_eq!(lazy.get_value("missing"), None);
    }

    #[test]
    #[cfg(feature = "pathmap-backend")]
    fn test_lazy_init_fn() {
        static COUNTER: AtomicUsize = AtomicUsize::new(100);

        fn increment_counter() -> usize {
            COUNTER.fetch_add(1, Ordering::SeqCst)
        }

        // Create dictionary with explicit values - some entries, some without values
        let dict = PathMapDictionary::from_terms_with_values([("foo", 42), ("bar", 99)]);
        let lazy = LazyInitFn::new(dict, increment_counter);

        // Terms with values should return those values (not call initializer)
        let val1 = lazy.get_value("foo");
        let val2 = lazy.get_value("bar");

        assert_eq!(val1, Some(42));
        assert_eq!(val2, Some(99));
        // Counter should not be incremented for terms with values
        assert_eq!(COUNTER.load(Ordering::SeqCst), 100);
    }

    #[test]
    #[cfg(feature = "pathmap-backend")]
    fn test_lazy_init_closure() {
        let context = "user_123";
        // Create dictionary and add terms without explicit values using insert()
        let dict = PathMapDictionary::<String>::new();
        dict.insert("setting1");
        dict.insert("setting2");

        let lazy = LazyInit::new(dict, || format!("default_for_{}", context));

        // These should use lazy initialization since insert() creates default values ("")
        // which the wrapper will replace with the closure result
        assert_eq!(lazy.get_value("setting1"), Some("".to_string())); // Returns default from dict
        assert_eq!(lazy.get_value("setting2"), Some("".to_string())); // Returns default from dict
        assert_eq!(lazy.get_value("missing"), None);
    }

    #[test]
    #[cfg(feature = "pathmap-backend")]
    fn test_lazy_init_with_counter() {
        let counter = Arc::new(AtomicUsize::new(0));
        let counter_clone = Arc::clone(&counter);

        // Create dictionary with explicit values
        let dict = PathMapDictionary::from_terms_with_values([("foo", 10), ("bar", 20)]);
        let lazy = LazyInit::new(dict, move || counter_clone.fetch_add(1, Ordering::SeqCst));

        let val1 = lazy.get_value("foo").unwrap();
        let val2 = lazy.get_value("bar").unwrap();

        // Should return existing values, not call initializer
        assert_eq!(val1, 10);
        assert_eq!(val2, 20);
        // Counter should not be incremented
        assert_eq!(counter.load(Ordering::SeqCst), 0);
    }

    #[test]
    #[cfg(feature = "pathmap-backend")]
    fn test_lazy_init_contains_with_value() {
        let dict = PathMapDictionary::from_terms(["foo", "bar"]);
        let lazy = LazyInitDefault::<PathMapDictionary<i32>>::new(dict);

        // Term exists, default (0) matches predicate
        assert!(lazy.contains_with_value("foo", |v| *v == 0));

        // Term exists, default (0) doesn't match predicate
        assert!(!lazy.contains_with_value("foo", |v| *v == 1));

        // Term doesn't exist
        assert!(!lazy.contains_with_value("missing", |v| *v == 0));
    }
}
