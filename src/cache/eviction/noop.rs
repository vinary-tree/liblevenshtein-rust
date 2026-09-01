//! Identity wrapper that provides no eviction behavior.
//!
//! The `Noop` wrapper is a pass-through wrapper that forwards all operations
//! to the inner dictionary without any modifications. It's useful for:
//!
//! - Testing and benchmarking (measuring wrapper overhead)
//! - Uniform API when eviction is conditionally needed
//! - Placeholder in generic code
//!
//! # Examples
//!
//! ```rust
//! # #[cfg(feature = "pathmap-backend")]
//! # {
//! use liblevenshtein::prelude::*;
//! use liblevenshtein::dictionary::MappedDictionary;
//! use liblevenshtein::cache::eviction::Noop;
//!
//! let dict = PathMapDictionary::from_terms_with_values([
//!     ("hello", 1),
//!     ("world", 2),
//! ]);
//!
//! let wrapped = Noop::new(dict);
//!
//! // Behaves exactly like the inner dictionary
//! assert_eq!(wrapped.get_value("hello"), Some(1));
//! assert!(wrapped.contains("world"));
//! # }
//! ```

use crate::dictionary::node_adapter::{
    impl_transparent_dictionary_node, impl_transparent_mapped_dictionary_node,
    map_transparent_traversal_root,
};
use libdictenstein::{
    Dictionary, DictionaryTraversalRoot, DictionaryValue, MappedDictionary, SyncStrategy,
};

/// Identity wrapper that provides no eviction behavior.
///
/// This is a zero-cost wrapper that simply forwards all operations to the
/// inner dictionary. It implements both `Dictionary` and `MappedDictionary`
/// traits.
#[derive(Debug, Clone)]
pub struct Noop<D> {
    inner: D,
}

impl<D> Noop<D> {
    /// Creates a new identity wrapper around the given dictionary.
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
    /// use liblevenshtein::cache::eviction::Noop;
    ///
    /// let dict: PathMapDictionary = PathMapDictionary::from_terms(["hello", "world"]);
    /// let wrapped = Noop::new(dict);
    /// # }
    /// ```
    #[inline]
    pub fn new(dict: D) -> Self {
        Self { inner: dict }
    }

    /// Unwraps the inner dictionary.
    ///
    /// # Examples
    ///
    /// ```rust
    /// # #[cfg(feature = "pathmap-backend")]
    /// # {
    /// use liblevenshtein::prelude::*;
    /// use liblevenshtein::cache::eviction::Noop;
    ///
    /// let dict: PathMapDictionary = PathMapDictionary::from_terms(["hello", "world"]);
    /// let wrapped = Noop::new(dict);
    /// let original = wrapped.into_inner();
    /// # }
    /// ```
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

/// Node wrapper for Noop dictionary.
///
/// This is a simple wrapper around the inner dictionary's node type.
#[derive(Debug, Clone)]
pub struct NoopNode<N> {
    inner: N,
}

impl<N> NoopNode<N> {
    #[inline]
    fn new(inner: N) -> Self {
        Self { inner }
    }
}

impl<D> Dictionary for Noop<D>
where
    D: Dictionary,
{
    type Node = NoopNode<D::Node>;

    #[inline]
    fn root(&self) -> Self::Node {
        NoopNode::new(self.inner.root())
    }

    #[inline]
    fn traversal_root(&self) -> DictionaryTraversalRoot<Self::Node> {
        map_transparent_traversal_root(self.inner.traversal_root(), NoopNode::new)
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

impl<D, V> MappedDictionary for Noop<D>
where
    D: MappedDictionary<Value = V>,
    V: DictionaryValue,
{
    type Value = V;

    #[inline]
    fn get_value(&self, term: &str) -> Option<Self::Value> {
        self.inner.get_value(term)
    }

    #[inline]
    fn contains_with_value<F>(&self, term: &str, predicate: F) -> bool
    where
        F: Fn(&Self::Value) -> bool,
    {
        self.inner.contains_with_value(term, predicate)
    }
}

impl_transparent_dictionary_node!(NoopNode, |_owner, child| NoopNode::new(child));
impl_transparent_mapped_dictionary_node!(NoopNode);

#[cfg(all(test, feature = "pathmap-backend"))]
mod tests {
    use super::*;
    #[cfg(feature = "pathmap-backend")]
    use libdictenstein::pathmap::PathMapDictionary;
    use libdictenstein::DictionaryNode;

    #[test]
    #[cfg(feature = "pathmap-backend")]
    fn test_noop_wrapper() {
        let dict = PathMapDictionary::<()>::from_terms(["hello", "world", "test"]);
        let wrapped = Noop::new(dict);

        assert_eq!(wrapped.len(), Some(3));
        assert!(wrapped.contains("hello"));
        assert!(wrapped.contains("world"));
        assert!(!wrapped.contains("missing"));
    }

    #[test]
    #[cfg(feature = "pathmap-backend")]
    fn test_noop_with_values() {
        let dict =
            PathMapDictionary::from_terms_with_values([("hello", 1), ("world", 2), ("test", 3)]);
        let wrapped = Noop::new(dict);

        assert_eq!(wrapped.get_value("hello"), Some(1));
        assert_eq!(wrapped.get_value("world"), Some(2));
        assert_eq!(wrapped.get_value("missing"), None);
    }

    #[test]
    #[cfg(feature = "pathmap-backend")]
    fn test_noop_node_traversal() {
        let dict = PathMapDictionary::<()>::from_terms(["hello", "help"]);
        let wrapped = Noop::new(dict);

        let root = wrapped.root();
        assert!(!root.is_final());

        // Traverse 'h'
        let h_node = root
            .transition(b'h')
            .expect("expected Some transition h in test");
        assert!(!h_node.is_final());

        // Traverse 'e'
        let e_node = h_node
            .transition(b'e')
            .expect("expected Some transition e in test");
        assert!(!e_node.is_final());

        // Traverse 'l'
        let l_node = e_node
            .transition(b'l')
            .expect("expected Some transition l in test");
        assert!(!l_node.is_final());

        // Traverse 'p' -> "help"
        let p_node = l_node
            .transition(b'p')
            .expect("expected Some transition p in test");
        assert!(p_node.is_final());

        // Traverse 'l' -> "hell" (not final, part of "hello")
        let l2_node = l_node
            .transition(b'l')
            .expect("expected Some transition l2 in test");
        assert!(!l2_node.is_final());

        // Traverse 'o' -> "hello"
        let o_node = l2_node
            .transition(b'o')
            .expect("expected Some transition o in test");
        assert!(o_node.is_final());
    }

    #[test]
    #[cfg(feature = "pathmap-backend")]
    fn test_noop_into_inner() {
        let dict = PathMapDictionary::<()>::from_terms(["hello"]);
        let wrapped = Noop::new(dict);
        let original = wrapped.into_inner();

        assert_eq!(original.len(), Some(1));
        assert!(original.contains("hello"));
    }

    #[test]
    #[cfg(feature = "pathmap-backend")]
    fn test_noop_inner_ref() {
        let dict = PathMapDictionary::<()>::from_terms(["hello"]);
        let wrapped = Noop::new(dict);
        let inner_ref = wrapped.inner();

        assert_eq!(inner_ref.len(), Some(1));
        assert!(inner_ref.contains("hello"));
    }
}
