//! Dictionary to LatticeBackend adapter.
//!
//! This module provides [`DictionaryBackend`], which adapts a liblevenshtein
//! dictionary to lling-llang's [`LatticeBackend`] trait.

use std::sync::Arc;

use lling_llang::backend::{LatticeBackend, VocabId};
use rustc_hash::FxHashMap;

use libdictenstein::Dictionary;

/// Adapter that exposes a liblevenshtein dictionary as a lling-llang `LatticeBackend`.
///
/// This allows liblevenshtein's efficient dictionary structures (DoubleArrayTrie,
/// DynamicDawg, etc.) to be used directly with lling-llang's lattice infrastructure.
///
/// # Vocabulary Management
///
/// The adapter maintains a bidirectional mapping between:
/// - lling-llang's `VocabId` (sequential u32 indices)
/// - Dictionary terms (strings in the dictionary)
///
/// Terms are interned lazily as they are accessed.
///
/// # Example
///
/// ```rust,ignore
/// use liblevenshtein::wfst::DictionaryBackend;
/// use liblevenshtein::dictionary::dynamic_dawg_char::DynamicDawgChar;
/// use lling_llang::prelude::*;
///
/// // Create a dictionary
/// let dict = DynamicDawgChar::from_terms(vec!["hello", "help", "world"]);
///
/// // Wrap as LatticeBackend
/// let backend = DictionaryBackend::new(dict);
///
/// // Use with lling-llang's LatticeBuilder
/// let mut builder = LatticeBuilder::<TropicalWeight, _>::new(backend);
/// ```
#[derive(Clone)]
pub struct DictionaryBackend<D>
where
    D: Dictionary + Clone + Send + Sync,
    D::Node: Send + Sync,
{
    /// The underlying dictionary
    dictionary: D,
    /// Forward mapping: word -> VocabId
    word_to_id: FxHashMap<Arc<str>, VocabId>,
    /// Reverse mapping: VocabId -> word
    id_to_word: Vec<Arc<str>>,
}

impl<D> DictionaryBackend<D>
where
    D: Dictionary + Clone + Send + Sync,
    D::Node: Send + Sync,
{
    /// Create a new dictionary backend from an existing dictionary.
    ///
    /// The vocabulary is initially empty and will be populated lazily
    /// as terms are interned.
    pub fn new(dictionary: D) -> Self {
        Self {
            dictionary,
            word_to_id: FxHashMap::default(),
            id_to_word: Vec::new(),
        }
    }

    /// Create a dictionary backend with pre-populated vocabulary.
    ///
    /// This iterates over all terms in the dictionary and interns them
    /// upfront. Useful when you need stable VocabIds from the start.
    ///
    /// # Note
    ///
    /// This requires iterating over the entire dictionary, which may be
    /// expensive for large dictionaries. Use `new()` for lazy population.
    pub fn with_vocabulary<I>(dictionary: D, terms: I) -> Self
    where
        I: IntoIterator<Item = String>,
    {
        let mut backend = Self::new(dictionary);
        for term in terms {
            backend.intern(&term);
        }
        backend
    }

    /// Get the underlying dictionary.
    pub fn dictionary(&self) -> &D {
        &self.dictionary
    }

    /// Get mutable access to the underlying dictionary.
    ///
    /// Note: Modifying the dictionary after vocabulary has been built
    /// may cause inconsistencies if terms are removed.
    pub fn dictionary_mut(&mut self) -> &mut D {
        &mut self.dictionary
    }

    /// Take ownership of the underlying dictionary.
    pub fn into_dictionary(self) -> D {
        self.dictionary
    }
}

impl<D> LatticeBackend for DictionaryBackend<D>
where
    D: Dictionary + Clone + Send + Sync,
    D::Node: Send + Sync,
{
    fn intern(&mut self, word: &str) -> VocabId {
        // Check if already interned
        if let Some(&id) = self.word_to_id.get(word) {
            return id;
        }

        // Allocate new ID
        let id = self.id_to_word.len() as VocabId;
        let word_arc: Arc<str> = word.into();

        self.word_to_id.insert(word_arc.clone(), id);
        self.id_to_word.push(word_arc);

        id
    }

    fn lookup(&self, id: VocabId) -> Option<&str> {
        self.id_to_word.get(id as usize).map(|s| s.as_ref())
    }

    fn vocab_size(&self) -> usize {
        self.id_to_word.len()
    }

    fn contains(&self, word: &str) -> bool {
        // Check both our cache and the dictionary
        self.word_to_id.contains_key(word) || self.dictionary.contains(word)
    }

    fn get_id(&self, word: &str) -> Option<VocabId> {
        self.word_to_id.get(word).copied()
    }

    fn iter(&self) -> impl Iterator<Item = (VocabId, &str)> {
        self.id_to_word
            .iter()
            .enumerate()
            .map(|(i, s)| (i as VocabId, s.as_ref()))
    }

    fn supports_sharing(&self) -> bool {
        // Dictionary backends can support structural sharing
        // depending on the underlying dictionary type
        false
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use libdictenstein::dynamic_dawg_char::DynamicDawgChar;

    #[test]
    fn test_dictionary_backend_new() {
        let dict = DynamicDawgChar::<()>::from_terms(vec!["hello", "world"]);
        let backend = DictionaryBackend::new(dict);

        assert_eq!(backend.vocab_size(), 0); // Lazy - nothing interned yet
    }

    #[test]
    fn test_dictionary_backend_intern() {
        let dict = DynamicDawgChar::<()>::from_terms(vec!["hello", "world"]);
        let mut backend = DictionaryBackend::new(dict);

        let id1 = backend.intern("hello");
        let id2 = backend.intern("world");
        let id3 = backend.intern("hello"); // duplicate

        assert_eq!(id1, id3); // Same word, same ID
        assert_ne!(id1, id2); // Different words, different IDs
        assert_eq!(backend.vocab_size(), 2);
    }

    #[test]
    fn test_dictionary_backend_lookup() {
        let dict = DynamicDawgChar::<()>::from_terms(vec!["hello", "world"]);
        let mut backend = DictionaryBackend::new(dict);

        let id = backend.intern("hello");
        assert_eq!(backend.lookup(id), Some("hello"));
        assert_eq!(backend.lookup(999), None);
    }

    #[test]
    fn test_dictionary_backend_contains() {
        let dict = DynamicDawgChar::<()>::from_terms(vec!["hello", "world"]);
        let backend = DictionaryBackend::new(dict);

        // Contains checks the underlying dictionary
        assert!(backend.contains("hello"));
        assert!(backend.contains("world"));
        assert!(!backend.contains("missing"));
    }

    #[test]
    fn test_dictionary_backend_get_id() {
        let dict = DynamicDawgChar::<()>::from_terms(vec!["hello", "world"]);
        let mut backend = DictionaryBackend::new(dict);

        // Not interned yet
        assert_eq!(backend.get_id("hello"), None);

        // After interning
        let id = backend.intern("hello");
        assert_eq!(backend.get_id("hello"), Some(id));
    }

    #[test]
    fn test_dictionary_backend_iter() {
        let dict = DynamicDawgChar::<()>::from_terms(vec!["hello", "world"]);
        let mut backend = DictionaryBackend::new(dict);

        backend.intern("hello");
        backend.intern("world");

        let entries: Vec<_> = backend.iter().collect();
        assert_eq!(entries.len(), 2);
    }

    #[test]
    fn test_dictionary_backend_with_vocabulary() {
        let dict = DynamicDawgChar::<()>::from_terms(vec!["hello", "world", "test"]);
        let terms = vec!["hello".to_string(), "world".to_string()];
        let backend = DictionaryBackend::with_vocabulary(dict, terms);

        assert_eq!(backend.vocab_size(), 2);
        assert!(backend.get_id("hello").is_some());
        assert!(backend.get_id("world").is_some());
        assert_eq!(backend.get_id("test"), None); // Not pre-populated
    }

    #[test]
    fn test_dictionary_backend_clone() {
        let dict = DynamicDawgChar::<()>::from_terms(vec!["hello", "world"]);
        let mut backend = DictionaryBackend::new(dict);
        backend.intern("hello");

        let cloned = backend.clone();
        assert_eq!(cloned.vocab_size(), 1);
        assert_eq!(cloned.get_id("hello"), backend.get_id("hello"));
    }
}
