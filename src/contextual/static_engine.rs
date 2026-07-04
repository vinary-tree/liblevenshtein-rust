//! Static contextual completion engine for read-only dictionaries.
//!
//! This module provides `StaticContextualCompletionEngine` that works with
//! pre-built, immutable dictionaries like DoubleArrayTrie. Finalized terms
//! are stored separately in a HashMap rather than mutating the dictionary.

use super::error::{ContextError, Result};
use super::locking::{lock_mutex, read_lock, write_lock};
use super::{CheckpointStack, Completion, ContextId, ContextTree, DraftBuffer};
use crate::transducer::{Algorithm, Transducer};
use libdictenstein::double_array_trie::char::DoubleArrayTrieChar;
use libdictenstein::double_array_trie::DoubleArrayTrie;
use std::collections::{HashMap, HashSet};
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
    /// Prefer [`Self::with_transducer`] or [`Self::with_transducer_mut`] for
    /// poison-tolerant access.
    ///
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
    /// // Clone the dictionary for serialization
    /// let dict = engine.with_transducer(|transducer| transducer.dictionary().clone());
    /// ```
    #[inline]
    pub fn transducer(&self) -> &Arc<RwLock<Transducer<D>>> {
        &self.transducer
    }

    /// Read the underlying transducer through the engine's poison-tolerant lock path.
    ///
    /// This accessor is the preferred way to inspect the transducer because it
    /// preserves engine availability after another thread panics while holding
    /// the transducer lock.
    #[inline]
    pub fn with_transducer<R>(&self, f: impl FnOnce(&Transducer<D>) -> R) -> R {
        let transducer = read_lock(&self.transducer);
        f(&transducer)
    }

    /// Mutate the underlying transducer through the engine's poison-tolerant lock path.
    ///
    /// This accessor mirrors write access through [`Self::transducer`] while
    /// keeping poisoned locks recoverable for long-lived completion services.
    #[inline]
    pub fn with_transducer_mut<R>(&self, f: impl FnOnce(&mut Transducer<D>) -> R) -> R {
        let mut transducer = write_lock(&self.transducer);
        f(&mut transducer)
    }

    /// Create a root context (top-level scope).
    pub fn create_root_context(&self, id: ContextId) -> Result<ContextId> {
        {
            let mut tree = write_lock(&self.context_tree);
            tree.try_create_root(id)?;
        }

        let mut drafts = lock_mutex(&self.drafts);
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
        {
            let mut tree = write_lock(&self.context_tree);
            tree.create_child(id, parent_id)?;
        }

        let mut drafts = lock_mutex(&self.drafts);
        drafts.insert(id, DraftBuffer::new());

        Ok(id)
    }

    fn ensure_context_exists(&self, context: ContextId) -> Result<()> {
        if read_lock(&self.context_tree).contains(context) {
            Ok(())
        } else {
            Err(ContextError::ContextNotFound(context))
        }
    }

    /// Insert a character into the draft buffer.
    pub fn insert_char(&self, context: ContextId, ch: char) -> Result<()> {
        self.ensure_context_exists(context)?;

        let mut drafts = lock_mutex(&self.drafts);
        let buffer = drafts
            .get_mut(&context)
            .ok_or(ContextError::NoDraftBuffer(context))?;
        buffer.insert(ch);
        Ok(())
    }

    /// Insert a string into the draft buffer.
    pub fn insert_str(&self, context: ContextId, s: &str) -> Result<()> {
        self.ensure_context_exists(context)?;
        if s.is_empty() {
            return Ok(());
        }

        let mut drafts = lock_mutex(&self.drafts);
        let buffer = drafts
            .get_mut(&context)
            .ok_or(ContextError::NoDraftBuffer(context))?;
        buffer.insert_str(s);
        Ok(())
    }

    /// Delete the last character from the draft buffer (backspace).
    pub fn delete_char(&self, context: ContextId) -> Result<()> {
        self.ensure_context_exists(context)?;

        let mut drafts = lock_mutex(&self.drafts);
        let buffer = drafts
            .get_mut(&context)
            .ok_or(ContextError::NoDraftBuffer(context))?;
        buffer.delete();
        Ok(())
    }

    /// Clear the draft buffer for a context.
    pub fn clear_draft(&self, context: ContextId) -> Result<()> {
        self.ensure_context_exists(context)?;

        let mut drafts = lock_mutex(&self.drafts);
        let buffer = drafts
            .get_mut(&context)
            .ok_or(ContextError::NoDraftBuffer(context))?;
        buffer.clear();
        Ok(())
    }

    /// Get the current draft text.
    pub fn get_draft(&self, context: ContextId) -> Result<String> {
        self.ensure_context_exists(context)?;

        let drafts = lock_mutex(&self.drafts);
        drafts
            .get(&context)
            .map(|buffer| buffer.as_str())
            .ok_or(ContextError::NoDraftBuffer(context))
    }

    /// Finalize a draft (store in finalized_terms HashMap, not in dictionary).
    ///
    /// Unlike `DynamicContextualCompletionEngine`, this does NOT modify the
    /// static dictionary. Instead, finalized terms are stored separately and
    /// queried alongside the dictionary.
    pub fn finalize(&self, context: ContextId) -> Result<String> {
        self.ensure_context_exists(context)?;

        let mut drafts = lock_mutex(&self.drafts);
        let buffer = drafts
            .get_mut(&context)
            .ok_or(ContextError::NoDraftBuffer(context))?;

        if buffer.is_empty() {
            return Err(ContextError::EmptyDraft(context));
        }

        let term = buffer.as_slice().to_owned();
        buffer.clear();

        // Store in finalized_terms instead of dictionary
        let mut finalized = write_lock(&self.finalized_terms);
        let contexts = finalized.entry(term.clone()).or_default();
        if !contexts.contains(&context) {
            contexts.push(context);
        }

        Ok(term)
    }

    /// Complete with query fusion (dictionary + finalized_terms + drafts).
    pub fn complete(
        &self,
        context: ContextId,
        query: &str,
        max_distance: usize,
    ) -> Result<Vec<Completion>> {
        // Query static dictionary (fast!)
        let finalized_dict = self.complete_dictionary(context, query, max_distance)?;
        let finalized_hash = self.complete_finalized_terms(context, query, max_distance)?;
        let drafts_results = self.complete_drafts(context, query, max_distance)?;

        let mut results = HashMap::with_capacity(
            finalized_dict.len() + finalized_hash.len() + drafts_results.len(),
        );
        for completion in finalized_dict {
            results.entry(completion.term.clone()).or_insert(completion);
        }

        // Query finalized terms HashMap (small, rare)
        for completion in finalized_hash {
            results.entry(completion.term.clone()).or_insert(completion);
        }

        // Query drafts (in-memory, always fresh)
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
        let tree = read_lock(&self.context_tree);
        let visible = tree.visible_contexts(context);
        if visible.is_empty() {
            return Ok(Vec::new());
        }
        let visible_set: HashSet<ContextId> = visible.iter().copied().collect();

        let transducer = read_lock(&self.transducer);
        let mut results = Vec::new();
        for candidate in transducer.query_with_distance(query, max_distance) {
            if let Some(contexts) = transducer.dictionary().get_value(&candidate.term) {
                let mut visible_contexts = Vec::with_capacity(contexts.len().min(visible.len()));
                for ctx in contexts {
                    if visible_set.contains(&ctx) {
                        visible_contexts.push(ctx);
                    }
                }

                if !visible_contexts.is_empty() {
                    results.push(Completion {
                        term: candidate.term,
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
        let tree = read_lock(&self.context_tree);
        let visible = tree.visible_contexts(context);
        if visible.is_empty() {
            return Ok(Vec::new());
        }
        let visible_set: HashSet<ContextId> = visible.iter().copied().collect();

        let finalized = read_lock(&self.finalized_terms);
        let mut results = Vec::with_capacity(finalized.len().min(64));

        for (term, contexts) in finalized.iter() {
            let Some(distance) = Self::levenshtein_distance_within(query, term, max_distance)
            else {
                continue;
            };

            let mut visible_contexts = Vec::with_capacity(contexts.len().min(visible.len()));
            for &ctx in contexts {
                if visible_set.contains(&ctx) {
                    visible_contexts.push(ctx);
                }
            }

            if !visible_contexts.is_empty() {
                results.push(Completion {
                    term: term.clone(),
                    distance,
                    contexts: visible_contexts,
                    is_draft: false,
                });
            }
        }

        Ok(results)
    }

    /// Query draft buffers with threshold-bounded Levenshtein.
    fn complete_drafts(
        &self,
        context: ContextId,
        query: &str,
        max_distance: usize,
    ) -> Result<Vec<Completion>> {
        let tree = read_lock(&self.context_tree);
        let visible = tree.visible_contexts(context);

        let drafts = lock_mutex(&self.drafts);
        let mut results = Vec::with_capacity(visible.len());

        for &ctx in &visible {
            if let Some(buffer) = drafts.get(&ctx) {
                let draft_text = buffer.as_slice();
                if !draft_text.is_empty() {
                    if let Some(distance) =
                        Self::levenshtein_distance_within(query, draft_text, max_distance)
                    {
                        results.push(Completion {
                            term: draft_text.to_owned(),
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

    /// Threshold-bounded Levenshtein distance calculation.
    fn levenshtein_distance_within(s1: &str, s2: &str, max_distance: usize) -> Option<usize> {
        crate::distance::standard_distance_bounded(s1, s2, max_distance)
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

#[cfg(test)]
mod tests {
    use super::*;

    fn static_engine() -> StaticContextualCompletionEngine<DoubleArrayTrie<Vec<ContextId>>> {
        let dict: DoubleArrayTrie<Vec<ContextId>> =
            DoubleArrayTrie::from_terms_with_values([("hello", vec![0])]);
        StaticContextualCompletionEngine::with_double_array_trie(dict, Algorithm::Standard)
    }

    #[test]
    fn recovers_after_internal_locks_are_poisoned() {
        let engine = std::sync::Arc::new(static_engine());
        let ctx = engine
            .create_root_context(0)
            .expect("test fixture: create root context");

        let thread_engine = std::sync::Arc::clone(&engine);
        let result = std::thread::spawn(move || {
            let _guard = thread_engine.context_tree.write().unwrap();
            panic!("intentional static context tree poisoning for recovery test");
        })
        .join();
        assert!(result.is_err());
        assert!(engine.complete(ctx, "hello", 0).is_ok());

        let thread_engine = std::sync::Arc::clone(&engine);
        let result = std::thread::spawn(move || {
            let mut drafts = thread_engine.drafts.lock().unwrap();
            let buffer = drafts
                .get_mut(&ctx)
                .expect("test fixture: root context has an initialized draft buffer");
            buffer.insert_str("local");
            panic!("intentional static draft poisoning for recovery test");
        })
        .join();
        assert!(result.is_err());
        assert_eq!(
            engine.get_draft(ctx).expect("test fixture: get draft"),
            "local"
        );

        let thread_engine = std::sync::Arc::clone(&engine);
        let result = std::thread::spawn(move || {
            let mut finalized = thread_engine.finalized_terms.write().unwrap();
            finalized.insert("poisoned".to_string(), vec![ctx]);
            panic!("intentional static finalized_terms poisoning for recovery test");
        })
        .join();
        assert!(result.is_err());
        assert_eq!(
            engine.finalize(ctx).expect("test fixture: finalize draft"),
            "local"
        );

        let thread_engine = std::sync::Arc::clone(&engine);
        let result = std::thread::spawn(move || {
            let _guard = thread_engine.transducer.write().unwrap();
            panic!("intentional static transducer poisoning for recovery test");
        })
        .join();
        assert!(result.is_err());
        assert_eq!(
            engine.with_transducer(|transducer| transducer.algorithm()),
            Algorithm::Standard
        );
        assert_eq!(
            engine.with_transducer_mut(|transducer| transducer.algorithm()),
            Algorithm::Standard
        );
    }

    #[test]
    fn create_context_rejects_duplicate_ids_and_self_parent_cycle() {
        let engine = static_engine();
        let root = engine
            .create_root_context(0)
            .expect("test fixture: create root context");

        assert_eq!(
            engine.create_root_context(root),
            Err(ContextError::ContextAlreadyExists(root))
        );
        assert_eq!(
            engine.create_child_context(root, root),
            Err(ContextError::CircularHierarchy(root, root))
        );

        let child = engine
            .create_child_context(1, root)
            .expect("test fixture: create child context");
        assert_eq!(
            engine.create_child_context(child, root),
            Err(ContextError::ContextAlreadyExists(child))
        );
    }

    #[test]
    fn draft_operations_reject_unknown_contexts() {
        let engine = static_engine();

        assert_eq!(
            engine.insert_char(999, 'x'),
            Err(ContextError::ContextNotFound(999))
        );
        assert_eq!(
            engine.insert_str(999, "x"),
            Err(ContextError::ContextNotFound(999))
        );
        assert_eq!(
            engine.delete_char(999),
            Err(ContextError::ContextNotFound(999))
        );
        assert_eq!(
            engine.clear_draft(999),
            Err(ContextError::ContextNotFound(999))
        );
        assert_eq!(
            engine.get_draft(999),
            Err(ContextError::ContextNotFound(999))
        );
    }

    #[test]
    fn finalize_rejects_empty_drafts() {
        let engine = static_engine();
        let ctx = engine
            .create_root_context(0)
            .expect("test fixture: create root context");

        assert_eq!(engine.finalize(ctx), Err(ContextError::EmptyDraft(ctx)));
    }

    #[test]
    fn repeated_finalize_deduplicates_context_membership() {
        let engine = static_engine();
        let ctx = engine
            .create_root_context(0)
            .expect("test fixture: create root context");

        for _ in 0..2 {
            engine
                .insert_str(ctx, "local_symbol")
                .expect("test fixture: insert draft");
            assert_eq!(
                engine.finalize(ctx).expect("test fixture: finalize draft"),
                "local_symbol"
            );
        }

        let completions = engine
            .complete(ctx, "local_symbol", 0)
            .expect("test fixture: exact complete");
        let local = completions
            .iter()
            .find(|completion| completion.term == "local_symbol")
            .expect("test fixture: finalized completion is visible");

        assert_eq!(local.contexts, vec![ctx]);
    }
}
