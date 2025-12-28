//! Dictionary factory for creating different backend implementations.
//!
//! This module provides a unified interface for creating dictionary instances
//! with different backend implementations (PathMap, DAWG, DynamicDawg).
//!
//! # Example
//!
//! ```rust,ignore
//! use liblevenshtein::dictionary::factory::{DictionaryFactory, DictionaryBackend};
//!
//! // Create a PathMap dictionary
//! let dict = DictionaryFactory::create(
//!     DictionaryBackend::PathMap,
//!     vec!["test", "testing", "tested"]
//! );
//!
//! // Create a DynamicDawg dictionary
//! let dict = DictionaryFactory::create(
//!     DictionaryBackend::DynamicDawg,
//!     vec!["test", "testing", "tested"]
//! );
//! ```

use super::double_array_trie::DoubleArrayTrie;
use super::dynamic_dawg::DynamicDawg;
#[cfg(feature = "pathmap-backend")]
use super::pathmap::PathMapDictionary;
use super::suffix_automaton::SuffixAutomaton;
use super::Dictionary;

/// Dictionary backend types
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum DictionaryBackend {
    /// PathMap-based trie dictionary (fastest for queries, highest memory)
    #[cfg(feature = "pathmap-backend")]
    PathMap,
    /// Double-Array Trie (O(1) transitions, excellent cache, supports updates)
    DoubleArrayTrie,
    /// Dynamic DAWG dictionary (space-efficient, supports modifications)
    DynamicDawg,
    /// Suffix automaton dictionary (substring matching, dynamic)
    SuffixAutomaton,
}

impl std::fmt::Display for DictionaryBackend {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            #[cfg(feature = "pathmap-backend")]
            DictionaryBackend::PathMap => write!(f, "PathMap"),
            DictionaryBackend::DoubleArrayTrie => write!(f, "DoubleArrayTrie"),
            DictionaryBackend::DynamicDawg => write!(f, "DynamicDAWG"),
            DictionaryBackend::SuffixAutomaton => write!(f, "SuffixAutomaton"),
        }
    }
}

/// Unified dictionary container that can hold any backend type
#[derive(Debug)]
pub enum DictionaryContainer {
    /// PathMap-based trie dictionary
    #[cfg(feature = "pathmap-backend")]
    PathMap(PathMapDictionary),
    /// Double-Array Trie dictionary
    DoubleArrayTrie(DoubleArrayTrie),
    /// Dynamic DAWG dictionary
    DynamicDawg(DynamicDawg),
    /// Suffix automaton dictionary
    SuffixAutomaton(SuffixAutomaton),
}

impl DictionaryContainer {
    /// Get the backend type of this container
    pub fn backend(&self) -> DictionaryBackend {
        match self {
            #[cfg(feature = "pathmap-backend")]
            DictionaryContainer::PathMap(_) => DictionaryBackend::PathMap,
            DictionaryContainer::DoubleArrayTrie(_) => DictionaryBackend::DoubleArrayTrie,
            DictionaryContainer::DynamicDawg(_) => DictionaryBackend::DynamicDawg,
            DictionaryContainer::SuffixAutomaton(_) => DictionaryBackend::SuffixAutomaton,
        }
    }

    /// Get the number of terms in the dictionary
    pub fn len(&self) -> Option<usize> {
        match self {
            #[cfg(feature = "pathmap-backend")]
            DictionaryContainer::PathMap(d) => d.len(),
            DictionaryContainer::DoubleArrayTrie(d) => d.len(),
            DictionaryContainer::DynamicDawg(d) => d.len(),
            DictionaryContainer::SuffixAutomaton(d) => d.len(),
        }
    }

    /// Check if the dictionary is empty
    pub fn is_empty(&self) -> bool {
        self.len() == Some(0)
    }

    /// Check if a term exists in the dictionary
    pub fn contains(&self, term: &str) -> bool {
        match self {
            #[cfg(feature = "pathmap-backend")]
            DictionaryContainer::PathMap(d) => d.contains(term),
            DictionaryContainer::DoubleArrayTrie(d) => d.contains(term),
            DictionaryContainer::DynamicDawg(d) => d.contains(term),
            DictionaryContainer::SuffixAutomaton(d) => d.contains(term),
        }
    }
}

/// Factory for creating dictionaries with different backends
pub struct DictionaryFactory;

impl DictionaryFactory {
    /// Create a dictionary with the specified backend
    ///
    /// # Arguments
    ///
    /// * `backend` - The backend implementation to use
    /// * `terms` - Iterator of terms to insert into the dictionary
    ///
    /// # Returns
    ///
    /// A `DictionaryContainer` holding the created dictionary
    ///
    /// # Example
    ///
    /// ```rust,ignore
    /// use liblevenshtein::dictionary::factory::{DictionaryFactory, DictionaryBackend};
    ///
    /// let dict = DictionaryFactory::create(
    ///     DictionaryBackend::DynamicDawg,
    ///     vec!["hello", "world"]
    /// );
    ///
    /// assert!(dict.contains("hello"));
    /// assert_eq!(dict.len(), Some(2));
    /// ```
    pub fn create<I, S>(backend: DictionaryBackend, terms: I) -> DictionaryContainer
    where
        I: IntoIterator<Item = S>,
        S: AsRef<str>,
    {
        match backend {
            #[cfg(feature = "pathmap-backend")]
            DictionaryBackend::PathMap => {
                DictionaryContainer::PathMap(PathMapDictionary::from_terms(terms))
            }
            DictionaryBackend::DoubleArrayTrie => {
                DictionaryContainer::DoubleArrayTrie(DoubleArrayTrie::from_terms(terms))
            }
            DictionaryBackend::DynamicDawg => {
                DictionaryContainer::DynamicDawg(DynamicDawg::from_terms(terms))
            }
            DictionaryBackend::SuffixAutomaton => {
                DictionaryContainer::SuffixAutomaton(SuffixAutomaton::from_texts(terms))
            }
        }
    }

    /// Create an empty dictionary with the specified backend
    ///
    /// # Arguments
    ///
    /// * `backend` - The backend implementation to use
    ///
    /// # Returns
    ///
    /// An empty `DictionaryContainer`
    ///
    /// # Example
    ///
    /// ```rust,ignore
    /// use liblevenshtein::dictionary::factory::{DictionaryFactory, DictionaryBackend};
    ///
    /// let dict = DictionaryFactory::empty(DictionaryBackend::PathMap);
    /// assert_eq!(dict.len(), Some(0));
    /// ```
    pub fn empty(backend: DictionaryBackend) -> DictionaryContainer {
        match backend {
            #[cfg(feature = "pathmap-backend")]
            DictionaryBackend::PathMap => DictionaryContainer::PathMap(PathMapDictionary::new()),
            DictionaryBackend::DoubleArrayTrie => {
                DictionaryContainer::DoubleArrayTrie(DoubleArrayTrie::new())
            }
            DictionaryBackend::DynamicDawg => DictionaryContainer::DynamicDawg(DynamicDawg::new()),
            DictionaryBackend::SuffixAutomaton => {
                DictionaryContainer::SuffixAutomaton(SuffixAutomaton::new())
            }
        }
    }

    /// Get a list of all available backends
    pub fn available_backends() -> Vec<DictionaryBackend> {
        vec![
            #[cfg(feature = "pathmap-backend")]
            DictionaryBackend::PathMap,
            DictionaryBackend::DoubleArrayTrie,
            DictionaryBackend::DynamicDawg,
            DictionaryBackend::SuffixAutomaton,
        ]
    }

    /// Get a description of a backend's characteristics
    pub fn backend_description(backend: DictionaryBackend) -> &'static str {
        match backend {
            #[cfg(feature = "pathmap-backend")]
            DictionaryBackend::PathMap => {
                "Fast queries with higher memory usage. Best for in-memory applications."
            }
            DictionaryBackend::DoubleArrayTrie => {
                "O(1) transitions with excellent cache locality. Best for memory-constrained environments."
            }
            DictionaryBackend::DynamicDawg => {
                "Space-efficient with modification support. Best for evolving dictionaries."
            }
            DictionaryBackend::SuffixAutomaton => {
                "Substring matching anywhere in text. Best for full-text search and code search."
            }
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    #[cfg(feature = "pathmap-backend")]
    fn test_factory_pathmap() {
        let dict = DictionaryFactory::create(
            DictionaryBackend::PathMap,
            vec!["test", "testing", "tested"],
        );

        assert_eq!(dict.backend(), DictionaryBackend::PathMap);
        assert_eq!(dict.len(), Some(3));
        assert!(dict.contains("test"));
        assert!(dict.contains("testing"));
        assert!(dict.contains("tested"));
        assert!(!dict.contains("tester"));
    }

    #[test]
    fn test_factory_dynamic_dawg() {
        let dict =
            DictionaryFactory::create(DictionaryBackend::DynamicDawg, vec!["foo", "bar", "baz"]);

        assert_eq!(dict.backend(), DictionaryBackend::DynamicDawg);
        assert_eq!(dict.len(), Some(3));
        assert!(dict.contains("foo"));
        assert!(dict.contains("bar"));
        assert!(dict.contains("baz"));
        assert!(!dict.contains("qux"));
    }

    #[test]
    fn test_factory_empty() {
        #[cfg(feature = "pathmap-backend")]
        {
            let pathmap = DictionaryFactory::empty(DictionaryBackend::PathMap);
            assert_eq!(pathmap.len(), Some(0));
            assert!(pathmap.is_empty());
        }

        let dynamic_dawg = DictionaryFactory::empty(DictionaryBackend::DynamicDawg);
        assert_eq!(dynamic_dawg.len(), Some(0));
        assert!(dynamic_dawg.is_empty());
    }

    #[test]
    fn test_backend_display() {
        #[cfg(feature = "pathmap-backend")]
        assert_eq!(DictionaryBackend::PathMap.to_string(), "PathMap");
        assert_eq!(DictionaryBackend::DynamicDawg.to_string(), "DynamicDAWG");
    }

    #[test]
    fn test_available_backends() {
        let backends = DictionaryFactory::available_backends();
        #[cfg(feature = "pathmap-backend")]
        assert_eq!(backends.len(), 4);
        #[cfg(not(feature = "pathmap-backend"))]
        assert_eq!(backends.len(), 3);
        #[cfg(feature = "pathmap-backend")]
        assert!(backends.contains(&DictionaryBackend::PathMap));
        assert!(backends.contains(&DictionaryBackend::DoubleArrayTrie));
        assert!(backends.contains(&DictionaryBackend::DynamicDawg));
        assert!(backends.contains(&DictionaryBackend::SuffixAutomaton));
    }

    #[test]
    fn test_backend_descriptions() {
        for backend in DictionaryFactory::available_backends() {
            let desc = DictionaryFactory::backend_description(backend);
            assert!(!desc.is_empty());
            println!("{}: {}", backend, desc);
        }
    }

    #[test]
    fn test_all_backends_work() {
        let terms = vec!["apple", "banana", "cherry"];

        for backend in DictionaryFactory::available_backends() {
            let dict = DictionaryFactory::create(backend, terms.clone());
            assert_eq!(dict.len(), Some(3));
            assert!(dict.contains("apple"));
            assert!(dict.contains("banana"));
            assert!(dict.contains("cherry"));
        }
    }
}
