//! Static contextual completion engine for read-only dictionaries.
//!
//! This module provides `StaticContextualCompletionEngine` that works with
//! pre-built, immutable dictionaries like DoubleArrayTrie. Finalized terms
//! are stored separately in a HashMap rather than mutating the dictionary.

use super::error::{ContextError, Result};
use super::{CheckpointStack, Completion, ContextId, ContextTree, DraftBuffer};
use libdictenstein::double_array_trie::DoubleArrayTrie;
use libdictenstein::double_array_trie_char::DoubleArrayTrieChar;
use crate::transducer::{Algorithm, Transducer};
use std::collections::HashMap;
use std::sync::{Arc, Mutex, RwLock};

/// Static engine for contextual completion with read-only dictionaries.
///
/// This variant is optimized for scenarios where you have a large pre-built
/// dictionary (e.g., programming language standard library) and only need to
/// track a small number of user-defined terms separately.
///
/// # Type Parameters
///
/// - `D`: Dictionary backend (must implement `MappedDictionary<Value = Vec<ContextId>>`)
///
/// # Architecture
///
/// - **Static dictionary**: Pre-built, immutable, optimized for fast queries
/// - **Finalized terms**: Stored separately in HashMap (for rare finalize() calls)
/// - **Drafts**: In-memory buffers, queried with naive Levenshtein
///
/// # Examples
///
/// ```
/// use liblevenshtein::contextual::StaticContextualCompletionEngine;
/// use liblevenshtein::dictionary::double_array_trie::{DoubleArrayTrie, DoubleArrayTrieBuilder};
/// use liblevenshtein::transducer::Algorithm;
///
/// // Build a static dictionary
/// let mut builder = DoubleArrayTrieBuilder::new();
/// builder.insert_with_value("std", Some(vec![0]));
/// builder.insert_with_value("String", Some(vec![0]));
/// let dict = builder.build();
///
/// // Create static engine
/// let engine = StaticContextualCompletionEngine::with_double_array_trie(
///     dict,
///     Algorithm::Standard
/// );
/// ```
pub struct StaticContextualCompletionEngine<D = DoubleArrayTrie<Vec<ContextId>>>
where
    D: crate::dictionary::MappedDictionary<Value = Vec<ContextId>> + Clone,
{
    /// Draft buffers per context (context_id -> buffer)
    drafts: Arc<Mutex<HashMap<ContextId, DraftBuffer>>>,

    /// Checkpoint stacks per context (context_id -> stack)
    checkpoints: Arc<Mutex<HashMap<ContextId, CheckpointStack>>>,

    /// Hierarchical context tree
    context_tree: Arc<RwLock<ContextTree>>,

    /// Transducer for fuzzy matching against static dictionary
    transducer: Arc<RwLock<Transducer<D>>>,

    /// Finalized terms that aren't in the static dictionary
    /// Maps term -> contexts where it's defined
    finalized_terms: Arc<RwLock<HashMap<String, Vec<ContextId>>>>,
}

// Convenience constructors for DoubleArrayTrie backend
impl StaticContextualCompletionEngine<DoubleArrayTrie<Vec<ContextId>>> {
    /// Create an engine with DoubleArrayTrie backend (byte-level, read-only).
    ///
    /// DoubleArrayTrie provides the fastest queries of any dictionary type.
    /// Best for large pre-built dictionaries with rare runtime additions.
    ///
    /// # Arguments
    ///
    /// * `dictionary` - Pre-built DoubleArrayTrie dictionary
    /// * `algorithm` - Levenshtein algorithm to use for fuzzy matching
    ///
    /// # Examples
    ///
    /// ```
    /// use liblevenshtein::contextual::StaticContextualCompletionEngine;
    /// use liblevenshtein::dictionary::double_array_trie::{DoubleArrayTrie, DoubleArrayTrieBuilder};
    /// use liblevenshtein::transducer::Algorithm;
    ///
    /// let mut builder = DoubleArrayTrieBuilder::new();
    /// builder.insert_with_value("test", Some(vec![0]));
    /// let dict = builder.build();
    ///
    /// let engine = StaticContextualCompletionEngine::with_double_array_trie(
    ///     dict,
    ///     Algorithm::Standard
    /// );
    /// ```
    pub fn with_double_array_trie(
        dictionary: DoubleArrayTrie<Vec<ContextId>>,
        algorithm: Algorithm,
    ) -> Self {
        Self::with_dictionary(dictionary, algorithm)
    }
}

// Convenience constructors for DoubleArrayTrieChar backend
impl StaticContextualCompletionEngine<DoubleArrayTrieChar<Vec<ContextId>>> {
    /// Create an engine with DoubleArrayTrieChar backend (character-level, read-only, Unicode).
    ///
    /// DoubleArrayTrieChar provides fast queries with correct Unicode handling.
    /// Ideal for large pre-built dictionaries containing emoji, CJK, or accented characters.
    ///
    /// # Arguments
    ///
    /// * `dictionary` - Pre-built DoubleArrayTrieChar dictionary
    /// * `algorithm` - Levenshtein algorithm to use for fuzzy matching
    ///
    /// # Examples
    ///
    /// ```
    /// use liblevenshtein::contextual::StaticContextualCompletionEngine;
    /// use liblevenshtein::dictionary::double_array_trie_char::DoubleArrayTrieChar;
    /// use liblevenshtein::transducer::Algorithm;
    ///
    /// let dict = DoubleArrayTrieChar::from_terms_with_values([
    ///     ("世界", vec![0]),
    /// ]);
    ///
    /// let engine = StaticContextualCompletionEngine::with_double_array_trie_char(
    ///     dict,
    ///     Algorithm::Standard
    /// );
    /// ```
    pub fn with_double_array_trie_char(
        dictionary: DoubleArrayTrieChar<Vec<ContextId>>,
        algorithm: Algorithm,
    ) -> Self {
        Self::with_dictionary(dictionary, algorithm)
    }
}

// Generic implementation for all dictionary backends
impl<D> StaticContextualCompletionEngine<D>
where
    D: crate::dictionary::MappedDictionary<Value = Vec<ContextId>> + Clone,
{
    /// Create an engine with a custom dictionary backend.
    ///
    /// This constructor works with any read-only dictionary type that implements
    /// `MappedDictionary<Value = Vec<ContextId>>`.
    ///
    /// # Arguments
    ///
    /// * `dictionary` - Pre-built dictionary instance
    /// * `algorithm` - Levenshtein algorithm variant
    pub fn with_dictionary(dictionary: D, algorithm: Algorithm) -> Self {
        let transducer = Transducer::new(dictionary, algorithm);

        Self {
            drafts: Arc::new(Mutex::new(HashMap::new())),
            checkpoints: Arc::new(Mutex::new(HashMap::new())),
            context_tree: Arc::new(RwLock::new(ContextTree::new())),
            transducer: Arc::new(RwLock::new(transducer)),
            finalized_terms: Arc::new(RwLock::new(HashMap::new())),
        }
    }

    /// Get a reference to the underlying transducer.
    ///
    /// The transducer is wrapped in `Arc<RwLock<>>` for thread-safe access.
    /// Use this to access the dictionary for operations like:
    /// - Cloning the dictionary for serialization
    /// - Querying the dictionary directly
    /// - Accessing transducer metadata (algorithm, etc.)
    ///
    /// # Examples
    ///
    /// ```
    /// use liblevenshtein::contextual::StaticContextualCompletionEngine;
    /// use liblevenshtein::dictionary::double_array_trie::{DoubleArrayTrie, DoubleArrayTrieBuilder};
    /// use liblevenshtein::transducer::Algorithm;
    ///
    /// let mut builder = DoubleArrayTrieBuilder::new();
    /// builder.insert_with_value("test", Some(vec![0]));
    /// let dict = builder.build();
    ///
    /// let engine = StaticContextualCompletionEngine::with_double_array_trie(dict, Algorithm::Standard);
    ///
    /// // Access the transducer
    /// let transducer_ref = engine.transducer();
    /// let transducer = transducer_ref.read().expect("static engine: transducer RwLock poisoned");
    ///
    /// // Clone the dictionary for serialization
    /// let dict = transducer.dictionary().clone();
    /// ```
    #[inline]
    pub fn transducer(&self) -> &Arc<RwLock<Transducer<D>>> {
        &self.transducer
    }

    /// Create a root context (top-level scope).
    pub fn create_root_context(&self, id: ContextId) -> Result<ContextId> {
        let mut tree = self.context_tree.write().expect("static engine: context_tree RwLock poisoned");
        tree.create_root(id);

        let mut drafts = self.drafts.lock().expect("static engine: drafts Mutex poisoned");
        drafts.insert(id, DraftBuffer::new());

        Ok(id)
    }

    /// Create a child context (nested scope).
    ///
    /// # Arguments
    ///
    /// * `id` - ID for the new child context
    /// * `parent_id` - ID of the existing parent context
    ///
    /// # Returns
    ///
    /// The child context ID on success, or error if parent doesn't exist
    pub fn create_child_context(&self, id: ContextId, parent_id: ContextId) -> Result<ContextId> {
        let mut tree = self.context_tree.write().expect("static engine: context_tree RwLock poisoned");
        tree.create_child(id, parent_id)
            .map_err(|_| ContextError::ContextNotFound(parent_id))?;

        let mut drafts = self.drafts.lock().expect("static engine: drafts Mutex poisoned");
        drafts.insert(id, DraftBuffer::new());

        Ok(id)
    }

    /// Insert a character into the draft buffer.
    pub fn insert_char(&self, context: ContextId, ch: char) -> Result<()> {
        let mut drafts = self.drafts.lock().expect("static engine: drafts Mutex poisoned");
        let buffer = drafts.entry(context).or_default();
        buffer.insert(ch);
        Ok(())
    }

    /// Insert a string into the draft buffer.
    pub fn insert_str(&self, context: ContextId, s: &str) -> Result<()> {
        let mut drafts = self.drafts.lock().expect("static engine: drafts Mutex poisoned");
        let buffer = drafts.entry(context).or_default();
        for ch in s.chars() {
            buffer.insert(ch);
        }
        Ok(())
    }

    /// Delete the last character from the draft buffer (backspace).
    pub fn delete_char(&self, context: ContextId) -> Result<()> {
        let mut drafts = self.drafts.lock().expect("static engine: drafts Mutex poisoned");
        if let Some(buffer) = drafts.get_mut(&context) {
            buffer.delete();
        }
        Ok(())
    }

    /// Clear the draft buffer for a context.
    pub fn clear_draft(&self, context: ContextId) -> Result<()> {
        let mut drafts = self.drafts.lock().expect("static engine: drafts Mutex poisoned");
        if let Some(buffer) = drafts.get_mut(&context) {
            buffer.clear();
        }
        Ok(())
    }

    /// Get the current draft text.
    pub fn get_draft(&self, context: ContextId) -> Result<String> {
        let drafts = self.drafts.lock().expect("static engine: drafts Mutex poisoned");
        Ok(drafts.get(&context).map(|b| b.as_str()).unwrap_or_default())
    }

    /// Finalize a draft (store in finalized_terms HashMap, not in dictionary).
    ///
    /// Unlike `DynamicContextualCompletionEngine`, this does NOT modify the
    /// static dictionary. Instead, finalized terms are stored separately and
    /// queried alongside the dictionary.
    pub fn finalize(&self, context: ContextId) -> Result<String> {
        let mut drafts = self.drafts.lock().expect("static engine: drafts Mutex poisoned");
        let buffer = drafts
            .get_mut(&context)
            .ok_or(ContextError::ContextNotFound(context))?;

        let term_owned = buffer.as_str();
        let term_clone = term_owned.clone();
        buffer.clear();

        // Store in finalized_terms instead of dictionary
        let mut finalized = self.finalized_terms.write().expect("static engine: finalized_terms RwLock poisoned");
        finalized
            .entry(term_clone.clone())
            .or_default()
            .push(context);

        Ok(term_clone)
    }

    /// Complete with query fusion (dictionary + finalized_terms + drafts).
    pub fn complete(
        &self,
        context: ContextId,
        query: &str,
        max_distance: usize,
    ) -> Result<Vec<Completion>> {
        let mut results = HashMap::new();

        // Query static dictionary (fast!)
        let finalized_dict = self.complete_dictionary(context, query, max_distance)?;
        for completion in finalized_dict {
            results.entry(completion.term.clone()).or_insert(completion);
        }

        // Query finalized terms HashMap (small, rare)
        let finalized_hash = self.complete_finalized_terms(context, query, max_distance)?;
        for completion in finalized_hash {
            results.entry(completion.term.clone()).or_insert(completion);
        }

        // Query drafts (in-memory, always fresh)
        let drafts_results = self.complete_drafts(context, query, max_distance)?;
        for completion in drafts_results {
            results.insert(completion.term.clone(), completion);
        }

        let mut final_results: Vec<Completion> = results.into_values().collect();
        final_results.sort_by(|a, b| {
            a.distance
                .cmp(&b.distance)
                .then_with(|| a.term.cmp(&b.term))
        });

        Ok(final_results)
    }

    /// Query the static dictionary only.
    fn complete_dictionary(
        &self,
        context: ContextId,
        query: &str,
        max_distance: usize,
    ) -> Result<Vec<Completion>> {
        let tree = self.context_tree.read().expect("static engine: context_tree RwLock poisoned");
        let visible = tree.visible_contexts(context);

        let transducer = self.transducer.read().expect("static engine: transducer RwLock poisoned");
        let candidates: Vec<_> = transducer
            .query_with_distance(query, max_distance)
            .collect();

        let mut results = Vec::new();
        for candidate in candidates {
            if let Some(contexts) = transducer.dictionary().get_value(&candidate.term) {
                let visible_contexts: Vec<_> = contexts
                    .iter()
                    .filter(|ctx| visible.contains(ctx))
                    .copied()
                    .collect();

                if !visible_contexts.is_empty() {
                    results.push(Completion {
                        term: candidate.term.clone(),
                        distance: candidate.distance,
                        contexts: visible_contexts,
                        is_draft: false,
                    });
                }
            }
        }

        Ok(results)
    }

    /// Query finalized_terms HashMap.
    fn complete_finalized_terms(
        &self,
        context: ContextId,
        query: &str,
        max_distance: usize,
    ) -> Result<Vec<Completion>> {
        let tree = self.context_tree.read().expect("static engine: context_tree RwLock poisoned");
        let visible = tree.visible_contexts(context);

        let finalized = self.finalized_terms.read().expect("static engine: finalized_terms RwLock poisoned");
        let mut results = Vec::new();

        for (term, contexts) in finalized.iter() {
            let distance = Self::levenshtein_distance(query, term);
            if distance <= max_distance {
                let visible_contexts: Vec<_> = contexts
                    .iter()
                    .filter(|ctx| visible.contains(ctx))
                    .copied()
                    .collect();

                if !visible_contexts.is_empty() {
                    results.push(Completion {
                        term: term.clone(),
                        distance,
                        contexts: visible_contexts,
                        is_draft: false,
                    });
                }
            }
        }

        Ok(results)
    }

    /// Query draft buffers with naive Levenshtein.
    fn complete_drafts(
        &self,
        context: ContextId,
        query: &str,
        max_distance: usize,
    ) -> Result<Vec<Completion>> {
        let tree = self.context_tree.read().expect("static engine: context_tree RwLock poisoned");
        let visible = tree.visible_contexts(context);

        let drafts = self.drafts.lock().expect("static engine: drafts Mutex poisoned");
        let mut results = Vec::new();

        for &ctx in &visible {
            if let Some(buffer) = drafts.get(&ctx) {
                let draft_text = buffer.as_str();
                if !draft_text.is_empty() {
                    let distance = Self::levenshtein_distance(query, &draft_text);
                    if distance <= max_distance {
                        results.push(Completion {
                            term: draft_text,
                            distance,
                            contexts: vec![ctx],
                            is_draft: true,
                        });
                    }
                }
            }
        }

        Ok(results)
    }

    /// Simple Levenshtein distance calculation.
    fn levenshtein_distance(s1: &str, s2: &str) -> usize {
        let len1 = s1.chars().count();
        let len2 = s2.chars().count();

        if len1 == 0 {
            return len2;
        }
        if len2 == 0 {
            return len1;
        }

        let mut matrix = vec![vec![0; len2 + 1]; len1 + 1];

        for (i, row) in matrix.iter_mut().enumerate() {
            row[0] = i;
        }
        for (j, cell) in matrix[0].iter_mut().enumerate() {
            *cell = j;
        }

        let s1_chars: Vec<char> = s1.chars().collect();
        let s2_chars: Vec<char> = s2.chars().collect();

        for i in 1..=len1 {
            for j in 1..=len2 {
                let cost = if s1_chars[i - 1] == s2_chars[j - 1] {
                    0
                } else {
                    1
                };

                matrix[i][j] = std::cmp::min(
                    std::cmp::min(matrix[i - 1][j] + 1, matrix[i][j - 1] + 1),
                    matrix[i - 1][j - 1] + cost,
                );
            }
        }

        matrix[len1][len2]
    }
}

impl<D> Clone for StaticContextualCompletionEngine<D>
where
    D: crate::dictionary::MappedDictionary<Value = Vec<ContextId>> + Clone,
{
    fn clone(&self) -> Self {
        Self {
            drafts: Arc::clone(&self.drafts),
            checkpoints: Arc::clone(&self.checkpoints),
            context_tree: Arc::clone(&self.context_tree),
            transducer: Arc::clone(&self.transducer),
            finalized_terms: Arc::clone(&self.finalized_terms),
        }
    }
}
