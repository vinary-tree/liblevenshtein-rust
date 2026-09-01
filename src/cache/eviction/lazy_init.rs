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
//! ```rust
//! # #[cfg(feature = "pathmap-backend")]
//! # {
//! use liblevenshtein::prelude::*;
//! use liblevenshtein::dictionary::MappedDictionary;
//! use liblevenshtein::cache::eviction::LazyInitDefault;
//!
//! let dict = PathMapDictionary::<i32>::from_terms(["foo", "bar"]);
//! let lazy = LazyInitDefault::new(dict);
//!
//! // Returns 0 (default i32) for terms without values
//! assert_eq!(lazy.get_value("foo"), Some(0));
//! # }
//! ```
//!
//! ## Custom Function
//!
//! ```rust
//! # #[cfg(feature = "pathmap-backend")]
//! # {
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
//! # }
//! ```
//!
//! ## Context-Aware Initialization
//!
//! ```rust
//! # #[cfg(feature = "pathmap-backend")]
//! # {
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
//! # }
//! ```

use crate::dictionary::node_adapter::{
    impl_semantic_dictionary_node_generic, map_transparent_traversal_root,
};
use libdictenstein::{
    Dictionary, DictionaryNode, DictionaryTraversalRoot, DictionaryValue, MappedDictionary,
    MappedDictionaryNode, SnapshotTraversalCursor, SnapshotTraversalGraph, SyncStrategy,
};
use std::sync::Arc;

/// Statically dispatched policy for synthesizing a missing final-node value.
#[doc(hidden)]
pub trait MissingValueInitializer<V: DictionaryValue>: Clone + Send + Sync {
    fn initialize(&self) -> V;
}

/// `Default`-based missing-value policy.
#[doc(hidden)]
#[derive(Clone, Copy, Debug, Default)]
pub struct DefaultValueInitializer;

impl<V> MissingValueInitializer<V> for DefaultValueInitializer
where
    V: DictionaryValue + Default,
{
    #[inline]
    fn initialize(&self) -> V {
        V::default()
    }
}

/// Function-pointer missing-value policy.
#[doc(hidden)]
pub struct FunctionValueInitializer<V: DictionaryValue>(fn() -> V);

impl<V: DictionaryValue> Clone for FunctionValueInitializer<V> {
    fn clone(&self) -> Self {
        *self
    }
}

impl<V: DictionaryValue> Copy for FunctionValueInitializer<V> {}

impl<V: DictionaryValue> MissingValueInitializer<V> for FunctionValueInitializer<V> {
    #[inline]
    fn initialize(&self) -> V {
        (self.0)()
    }
}

/// Closure-backed missing-value policy. The closure is allocated once when
/// the dictionary wrapper is constructed and shared by every retained node.
#[doc(hidden)]
pub struct ClosureValueInitializer<F>(Arc<F>);

impl<F> Clone for ClosureValueInitializer<F> {
    fn clone(&self) -> Self {
        Self(Arc::clone(&self.0))
    }
}

impl<F, V> MissingValueInitializer<V> for ClosureValueInitializer<F>
where
    F: Fn() -> V + Send + Sync,
    V: DictionaryValue,
{
    #[inline]
    fn initialize(&self) -> V {
        (self.0)()
    }
}

/// Node decorator shared by every lazy-initialization strategy.
#[doc(hidden)]
pub struct LazyInitNode<N, I> {
    inner: N,
    initializer: I,
}

impl<N: Clone, I: Clone> Clone for LazyInitNode<N, I> {
    fn clone(&self) -> Self {
        Self {
            inner: self.inner.clone(),
            initializer: self.initializer.clone(),
        }
    }
}

impl<N, I> LazyInitNode<N, I> {
    #[inline]
    fn new(inner: N, initializer: I) -> Self {
        Self { inner, initializer }
    }
}

impl_semantic_dictionary_node_generic!(
    [N, I],
    LazyInitNode<N, I>,
    [N: DictionaryNode, I: Clone + Send + Sync],
    |owner, child| LazyInitNode::new(child, owner.initializer.clone()),
    |owner| owner.inner.requires_final_units(),
    |owner, units| owner.inner.accepts_final_units(units)
);

impl<N, I, V> MappedDictionaryNode for LazyInitNode<N, I>
where
    N: MappedDictionaryNode<Value = V>,
    I: MissingValueInitializer<V>,
    V: DictionaryValue,
{
    type Value = V;

    #[inline]
    fn value(&self) -> Option<Self::Value> {
        self.inner.value().or_else(|| {
            (!self.inner.requires_final_units() && self.inner.is_final())
                .then(|| self.initializer.initialize())
        })
    }

    #[inline]
    fn value_at_final(&self) -> Option<Self::Value> {
        self.inner
            .value_at_final()
            .or_else(|| (!self.inner.requires_final_units()).then(|| self.initializer.initialize()))
    }

    #[inline]
    fn value_at_final_with_units(&self, units: &[Self::Unit]) -> Option<Self::Value> {
        if !self.inner.accepts_final_units(units) {
            return None;
        }
        self.inner
            .value_at_final_with_units(units)
            .or_else(|| Some(self.initializer.initialize()))
    }

    #[inline]
    fn supports_snapshot_cursor_values(&self) -> bool {
        self.inner.supports_snapshot_cursor_values()
    }

    #[inline]
    fn supports_snapshot_graph_values(&self) -> bool {
        self.inner.supports_snapshot_graph_values()
    }

    #[inline]
    fn snapshot_traversal_graph(
        &self,
    ) -> Option<Arc<SnapshotTraversalGraph<Self::Unit, Self::SnapshotGraphValueHandle>>> {
        self.inner.snapshot_traversal_graph()
    }

    #[inline]
    unsafe fn snapshot_cursor_value(
        &self,
        cursor: Self::SnapshotCursor,
    ) -> Option<Option<Self::Value>> {
        // SAFETY: this decorator retains the exact inner cursor owner.
        unsafe { self.inner.snapshot_cursor_value(cursor) }.map(|value| {
            value.or_else(|| {
                (!self.inner.requires_final_units()).then(|| self.initializer.initialize())
            })
        })
    }

    #[inline]
    unsafe fn snapshot_cursor_value_with_units(
        &self,
        cursor: Self::SnapshotCursor,
        units: &[Self::Unit],
    ) -> Option<Option<Self::Value>> {
        if !self.inner.accepts_final_units(units) {
            return Some(None);
        }
        // SAFETY: cursor and root-relative units name the retained inner node.
        unsafe { self.inner.snapshot_cursor_value_with_units(cursor, units) }
            .map(|value| Some(value.unwrap_or_else(|| self.initializer.initialize())))
    }

    #[inline]
    unsafe fn snapshot_graph_cursor_value(
        &self,
        graph: &SnapshotTraversalGraph<Self::Unit, Self::SnapshotGraphValueHandle>,
        cursor: SnapshotTraversalCursor,
    ) -> Option<Option<Self::Value>> {
        // SAFETY: this decorator retains the exact inner graph/cursor owner.
        unsafe { self.inner.snapshot_graph_cursor_value(graph, cursor) }.map(|value| {
            value.or_else(|| {
                (!self.inner.requires_final_units()).then(|| self.initializer.initialize())
            })
        })
    }

    #[inline]
    unsafe fn snapshot_graph_cursor_value_with_units(
        &self,
        graph: &SnapshotTraversalGraph<Self::Unit, Self::SnapshotGraphValueHandle>,
        cursor: SnapshotTraversalCursor,
        units: &[Self::Unit],
    ) -> Option<Option<Self::Value>> {
        if !self.inner.accepts_final_units(units) {
            return Some(None);
        }
        // SAFETY: graph, cursor, and units name the retained inner node.
        unsafe {
            self.inner
                .snapshot_graph_cursor_value_with_units(graph, cursor, units)
        }
        .map(|value| Some(value.unwrap_or_else(|| self.initializer.initialize())))
    }
}

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
    /// ```rust
    /// # #[cfg(feature = "pathmap-backend")]
    /// # {
    /// use liblevenshtein::prelude::*;
    /// use liblevenshtein::cache::eviction::LazyInitDefault;
    ///
    /// let dict = PathMapDictionary::<i32>::from_terms(["foo", "bar"]);
    /// let lazy = LazyInitDefault::new(dict);
    /// # }
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

impl<D, V> Dictionary for LazyInitDefault<D>
where
    D: MappedDictionary<Value = V>,
    D::Node: MappedDictionaryNode<Value = V>,
    V: DictionaryValue + Default,
{
    type Node = LazyInitNode<D::Node, DefaultValueInitializer>;

    #[inline]
    fn root(&self) -> Self::Node {
        LazyInitNode::new(self.inner.root(), DefaultValueInitializer)
    }

    #[inline]
    fn traversal_root(&self) -> DictionaryTraversalRoot<Self::Node> {
        map_transparent_traversal_root(self.inner.traversal_root(), |node| {
            LazyInitNode::new(node, DefaultValueInitializer)
        })
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

    #[inline]
    fn is_suffix_based(&self) -> bool {
        self.inner.is_suffix_based()
    }
}

impl<D, V> MappedDictionary for LazyInitDefault<D>
where
    D: MappedDictionary<Value = V>,
    D::Node: MappedDictionaryNode<Value = V>,
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
    /// ```rust
    /// # #[cfg(feature = "pathmap-backend")]
    /// # {
    /// use liblevenshtein::prelude::*;
    /// use liblevenshtein::cache::eviction::LazyInitFn;
    ///
    /// fn zero() -> i32 { 0 }
    ///
    /// let dict: PathMapDictionary = PathMapDictionary::from_terms(["foo", "bar"]);
    /// let lazy = LazyInitFn::new(dict, zero);
    /// # }
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
    D: MappedDictionary<Value = V>,
    D::Node: MappedDictionaryNode<Value = V>,
    V: DictionaryValue,
{
    type Node = LazyInitNode<D::Node, FunctionValueInitializer<V>>;

    #[inline]
    fn root(&self) -> Self::Node {
        LazyInitNode::new(
            self.inner.root(),
            FunctionValueInitializer(self.initializer),
        )
    }

    #[inline]
    fn traversal_root(&self) -> DictionaryTraversalRoot<Self::Node> {
        let initializer = FunctionValueInitializer(self.initializer);
        map_transparent_traversal_root(self.inner.traversal_root(), |node| {
            LazyInitNode::new(node, initializer)
        })
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

    #[inline]
    fn is_suffix_based(&self) -> bool {
        self.inner.is_suffix_based()
    }
}

impl<D, V> MappedDictionary for LazyInitFn<D, V>
where
    D: MappedDictionary<Value = V>,
    D::Node: MappedDictionaryNode<Value = V>,
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
pub struct LazyInit<D, F> {
    inner: D,
    initializer: Arc<F>,
}

impl<D: Clone, F> Clone for LazyInit<D, F> {
    fn clone(&self) -> Self {
        Self {
            inner: self.inner.clone(),
            initializer: Arc::clone(&self.initializer),
        }
    }
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
    /// ```rust
    /// # #[cfg(feature = "pathmap-backend")]
    /// # {
    /// use liblevenshtein::prelude::*;
    /// use liblevenshtein::cache::eviction::LazyInit;
    ///
    /// let context = "user_123";
    /// let dict: PathMapDictionary = PathMapDictionary::from_terms(["setting1", "setting2"]);
    ///
    /// let lazy = LazyInit::new(dict, || {
    ///     format!("default_for_{}", context)
    /// });
    /// # }
    /// ```
    #[inline]
    pub fn new(dict: D, initializer: F) -> Self {
        Self {
            inner: dict,
            initializer: Arc::new(initializer),
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

impl<D, F, V> Dictionary for LazyInit<D, F>
where
    D: MappedDictionary<Value = V>,
    D::Node: MappedDictionaryNode<Value = V>,
    F: Fn() -> V + Send + Sync,
    V: DictionaryValue,
{
    type Node = LazyInitNode<D::Node, ClosureValueInitializer<F>>;

    #[inline]
    fn root(&self) -> Self::Node {
        LazyInitNode::new(
            self.inner.root(),
            ClosureValueInitializer(Arc::clone(&self.initializer)),
        )
    }

    #[inline]
    fn traversal_root(&self) -> DictionaryTraversalRoot<Self::Node> {
        let initializer = ClosureValueInitializer(Arc::clone(&self.initializer));
        map_transparent_traversal_root(self.inner.traversal_root(), |node| {
            LazyInitNode::new(node, initializer)
        })
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

    #[inline]
    fn is_suffix_based(&self) -> bool {
        self.inner.is_suffix_based()
    }
}

impl<D, F, V> MappedDictionary for LazyInit<D, F>
where
    D: MappedDictionary<Value = V>,
    D::Node: MappedDictionaryNode<Value = V>,
    F: Fn() -> V + Send + Sync,
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

#[cfg(all(test, feature = "pathmap-backend"))]
mod tests {
    use super::*;
    #[cfg(feature = "pathmap-backend")]
    use libdictenstein::pathmap::PathMapDictionary;
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

        let val1 = lazy.get_value("foo").expect("expected Some value in test");
        let val2 = lazy.get_value("bar").expect("expected Some value in test");

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

#[cfg(test)]
mod semantic_traversal_tests {
    use super::*;
    use crate::cache::eviction::Ttl;
    use crate::transducer::{Algorithm, Transducer};
    use libdictenstein::dynamic_dawg::DynamicDawg;
    use std::collections::HashSet;
    use std::sync::atomic::{AtomicUsize, Ordering};
    use std::time::Duration;

    fn sparse_dictionary() -> DynamicDawg<u64> {
        let dictionary = DynamicDawg::new();
        dictionary.insert("cat");
        dictionary.insert_with_value("dog", 9);
        dictionary
    }

    fn eleven() -> u64 {
        11
    }

    fn assert_native_graph<D>(dictionary: &D)
    where
        D: Dictionary,
        D::Node: MappedDictionaryNode<Value = u64>,
    {
        let (graph, owner) = dictionary
            .traversal_root()
            .into_parts()
            .into_projection_and_root();
        assert!(graph.is_some());
        assert!(owner.supports_snapshot_graph_values());
    }

    #[test]
    fn every_lazy_policy_initializes_missing_values_during_native_traversal() {
        let default = LazyInitDefault::new(sparse_dictionary());
        assert_native_graph(&default);
        assert_eq!(
            Transducer::new(default, Algorithm::Standard)
                .query_values("cat", 0)
                .collect::<Vec<_>>(),
            [("cat".to_owned(), 0, 0)]
        );

        let function = LazyInitFn::new(sparse_dictionary(), eleven);
        assert_native_graph(&function);
        assert_eq!(
            Transducer::new(function, Algorithm::Standard)
                .query_values("cat", 0)
                .collect::<Vec<_>>(),
            [("cat".to_owned(), 0, 11)]
        );

        let calls = Arc::new(AtomicUsize::new(0));
        let observed_calls = Arc::clone(&calls);
        let closure = LazyInit::new(sparse_dictionary(), move || {
            observed_calls.fetch_add(1, Ordering::Relaxed);
            13
        });
        assert_native_graph(&closure);
        let transducer = Transducer::new(closure, Algorithm::Standard);

        assert_eq!(
            transducer.query_values("cat", 0).collect::<Vec<_>>(),
            [("cat".to_owned(), 0, 13)]
        );
        assert_eq!(calls.load(Ordering::Relaxed), 1);

        assert_eq!(
            transducer.query_values("dog", 0).collect::<Vec<_>>(),
            [("dog".to_owned(), 0, 9)]
        );
        assert!(transducer.query_values("owl", 0).next().is_none());
        assert_eq!(calls.load(Ordering::Relaxed), 1);
    }

    #[test]
    fn lazy_semantics_cover_filtered_set_and_ranked_value_queries() {
        let dictionary = LazyInitFn::new(sparse_dictionary(), eleven);
        let transducer = Transducer::new(dictionary, Algorithm::Standard);

        assert_eq!(
            transducer
                .query_filtered("cat", 0, |value| *value == 11)
                .map(|candidate| candidate.term)
                .collect::<Vec<_>>(),
            ["cat"]
        );

        let values = HashSet::from([11_u64]);
        assert_eq!(
            transducer
                .query_by_value_set("cat", 0, &values)
                .map(|candidate| candidate.term)
                .collect::<Vec<_>>(),
            ["cat"]
        );

        let suggestions: Vec<_> = transducer
            .query_suggestions("cat", 0, |_term: &str, _distance, value: &u64| {
                *value as f64
            })
            .collect();
        assert_eq!(suggestions.len(), 1);
        assert_eq!(suggestions[0].term, "cat");
        assert_eq!(suggestions[0].value, 11);
    }

    #[test]
    fn lazy_and_ttl_compose_without_resurrecting_expired_missing_values() {
        let ttl_outside = Ttl::new(
            LazyInitDefault::new(sparse_dictionary()),
            Duration::from_secs(1),
        );
        assert_eq!(ttl_outside.get_value("cat"), Some(0));
        ttl_outside.set_entry_age_for_test("cat", Duration::from_secs(2));
        assert!(Transducer::new(ttl_outside, Algorithm::Standard)
            .query_values("cat", 0)
            .next()
            .is_none());

        let ttl_inside = Ttl::new(sparse_dictionary(), Duration::from_secs(1));
        ttl_inside.set_entry_age_for_test("cat", Duration::from_secs(2));
        let lazy_outside = LazyInitDefault::new(ttl_inside);
        assert!(Transducer::new(lazy_outside, Algorithm::Standard)
            .query_values("cat", 0)
            .next()
            .is_none());
    }
}
