//! Contextual completion engine for incremental fuzzy matching with hierarchical scopes.
//!
//! This module provides the main `DynamicContextualCompletionEngine` that combines:
//! - Draft buffer management per context
//! - Hierarchical context tree for visibility
//! - Checkpoint-based undo/redo
//! - Query fusion (drafts + finalized terms)
//! - Thread-safe concurrent access

use super::error::{ContextError, Result};
use super::locking::{lock_mutex, read_lock, write_lock};
use super::{CheckpointStack, Completion, ContextId, ContextTree, DraftBuffer};
use crate::transducer::{Algorithm, Transducer};
use libdictenstein::dynamic_dawg::char::DynamicDawgChar;
use libdictenstein::dynamic_dawg::DynamicDawg;
use libdictenstein::pathmap::PathMapDictionary;
use std::collections::{HashMap, HashSet};
use std::sync::{Arc, Mutex, RwLock};

/// Engine for contextual code completion with hierarchical scopes.
///
/// The engine manages:
/// - **Drafts**: In-progress text buffers per context
/// - **Contexts**: Hierarchical tree of lexical scopes
/// - **Checkpoints**: Undo history per draft
/// - **Dictionary**: Finalized terms with context associations
///
/// # Type Parameters
///
/// - `D`: Dictionary backend (must implement `MappedDictionary<Value = Vec<ContextId>>`)
///
/// # Thread Safety
///
/// The engine uses interior mutability with `Mutex` and `RwLock` for
/// thread-safe concurrent access. Multiple threads can query and modify
/// different contexts simultaneously.
///
/// # Examples
///
/// ```
/// use liblevenshtein::contextual::DynamicContextualCompletionEngine;
/// use liblevenshtein::dictionary::pathmap::PathMapDictionary;
/// use liblevenshtein::transducer::Algorithm;
///
/// // Create engine with default PathMapDictionary backend
/// let engine = DynamicContextualCompletionEngine::new();
///
/// // Or create with a specific dictionary
/// let dict = PathMapDictionary::new();
/// let engine2 = DynamicContextualCompletionEngine::with_dictionary(dict, Algorithm::Standard);
///
/// // Create global context
/// let global = engine.create_root_context(0);
///
/// // Insert draft text
/// engine.insert_str(global, "hello");
///
/// // Query completions
/// let completions = engine.complete(global, "helo", 1);
/// assert!(completions.iter().any(|c| c.term == "hello"));
/// ```
pub struct DynamicContextualCompletionEngine<D = PathMapDictionary<Vec<ContextId>>>
where
    D: crate::dictionary::MutableMappedDictionary<Value = Vec<ContextId>> + Clone,
{
    /// Draft buffers per context (context_id -> buffer)
    drafts: Arc<Mutex<HashMap<ContextId, DraftBuffer>>>,

    /// Checkpoint stacks per context (context_id -> stack)
    checkpoints: Arc<Mutex<HashMap<ContextId, CheckpointStack>>>,

    /// Hierarchical context tree
    context_tree: Arc<RwLock<ContextTree>>,

    /// Transducer for fuzzy matching against finalized dictionary
    /// Maps terms to the contexts where they're defined
    transducer: Arc<RwLock<Transducer<D>>>,
}

// Convenience constructors for default PathMapDictionary backend
impl DynamicContextualCompletionEngine<PathMapDictionary<Vec<ContextId>>> {
    /// Create a new engine with default configuration (PathMapDictionary + Standard algorithm).
    ///
    /// # Examples
    ///
    /// ```
    /// use liblevenshtein::contextual::DynamicContextualCompletionEngine;
    ///
    /// let engine = DynamicContextualCompletionEngine::new();
    /// ```
    pub fn new() -> Self {
        Self::with_algorithm(Algorithm::Standard)
    }

    /// Create an engine with a specific Levenshtein algorithm variant (using PathMapDictionary).
    ///
    /// # Arguments
    ///
    /// * `algorithm` - Levenshtein algorithm to use for fuzzy matching
    ///
    /// # Examples
    ///
    /// ```
    /// use liblevenshtein::contextual::DynamicContextualCompletionEngine;
    /// use liblevenshtein::transducer::Algorithm;
    ///
    /// let engine = DynamicContextualCompletionEngine::with_algorithm(Algorithm::Transposition);
    /// ```
    pub fn with_algorithm(algorithm: Algorithm) -> Self {
        let dictionary = PathMapDictionary::<Vec<ContextId>>::new();
        Self::with_dictionary(dictionary, algorithm)
    }
}

// Convenience constructors for DynamicDawg backend
impl DynamicContextualCompletionEngine<DynamicDawg<Vec<ContextId>>> {
    /// Create an engine with DynamicDawg backend (byte-level, supports insert/remove).
    ///
    /// DynamicDawg provides faster queries than PathMapDictionary while still supporting
    /// runtime modifications. Best for applications that need both performance and flexibility.
    ///
    /// # Arguments
    ///
    /// * `algorithm` - Levenshtein algorithm to use for fuzzy matching
    ///
    /// # Examples
    ///
    /// ```
    /// use liblevenshtein::contextual::DynamicContextualCompletionEngine;
    /// use liblevenshtein::transducer::Algorithm;
    ///
    /// let engine = DynamicContextualCompletionEngine::with_dynamic_dawg(Algorithm::Standard);
    /// ```
    pub fn with_dynamic_dawg(algorithm: Algorithm) -> Self {
        let dictionary = DynamicDawg::<Vec<ContextId>>::new();
        Self::with_dictionary(dictionary, algorithm)
    }
}

// Convenience constructors for DynamicDawgChar backend
impl DynamicContextualCompletionEngine<DynamicDawgChar<Vec<ContextId>>> {
    /// Create an engine with DynamicDawgChar backend (character-level, full Unicode support).
    ///
    /// DynamicDawgChar handles multi-byte UTF-8 characters correctly, making it suitable for
    /// applications working with emoji, CJK text, or other non-ASCII Unicode characters.
    /// Provides faster queries than PathMapDictionary with proper Unicode handling.
    ///
    /// # Arguments
    ///
    /// * `algorithm` - Levenshtein algorithm to use for fuzzy matching
    ///
    /// # Examples
    ///
    /// ```
    /// use liblevenshtein::contextual::DynamicContextualCompletionEngine;
    /// use liblevenshtein::transducer::Algorithm;
    ///
    /// let engine = DynamicContextualCompletionEngine::with_dynamic_dawg_char(Algorithm::Standard);
    /// ```
    pub fn with_dynamic_dawg_char(algorithm: Algorithm) -> Self {
        let dictionary = DynamicDawgChar::<Vec<ContextId>>::new();
        Self::with_dictionary(dictionary, algorithm)
    }
}

// Generic implementation for all dictionary backends
impl<D> DynamicContextualCompletionEngine<D>
where
    D: crate::dictionary::MutableMappedDictionary<Value = Vec<ContextId>> + Clone,
{
    /// Create an engine with a specific dictionary backend and algorithm.
    ///
    /// # Arguments
    ///
    /// * `dictionary` - Dictionary backend to use for storing finalized terms
    /// * `algorithm` - Levenshtein algorithm variant
    ///
    /// # Examples
    ///
    /// ```
    /// use liblevenshtein::contextual::DynamicContextualCompletionEngine;
    /// use liblevenshtein::dictionary::pathmap::PathMapDictionary;
    /// use liblevenshtein::transducer::Algorithm;
    ///
    /// let dict = PathMapDictionary::new();
    /// let engine = DynamicContextualCompletionEngine::with_dictionary(dict, Algorithm::Standard);
    /// ```
    pub fn with_dictionary(dictionary: D, algorithm: Algorithm) -> Self {
        let transducer = Transducer::new(dictionary, algorithm);

        Self {
            drafts: Arc::new(Mutex::new(HashMap::new())),
            checkpoints: Arc::new(Mutex::new(HashMap::new())),
            context_tree: Arc::new(RwLock::new(ContextTree::new())),
            transducer: Arc::new(RwLock::new(transducer)),
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
    /// use liblevenshtein::contextual::DynamicContextualCompletionEngine;
    ///
    /// let engine = DynamicContextualCompletionEngine::new();
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

    /// Create a root context.
    ///
    /// Root contexts have no parent and serve as top-level scopes
    /// (e.g., global scope, module scope).
    ///
    /// # Arguments
    ///
    /// * `id` - Unique identifier for the context
    ///
    /// # Returns
    ///
    /// The context ID on success.
    ///
    /// If the context ID is already in use, the existing context is left
    /// unchanged and the ID is returned. Use [`Self::try_create_root_context`]
    /// when duplicate IDs should be reported as errors.
    ///
    /// # Examples
    ///
    /// ```
    /// use liblevenshtein::contextual::DynamicContextualCompletionEngine;
    ///
    /// let engine = DynamicContextualCompletionEngine::new();
    /// let global = engine.create_root_context(0);
    /// assert_eq!(global, 0);
    /// ```
    pub fn create_root_context(&self, id: ContextId) -> ContextId {
        let created = {
            let mut tree = write_lock(&self.context_tree);
            if tree.contains(id) {
                false
            } else {
                tree.create_root(id);
                true
            }
        };

        if created {
            // Initialize empty draft and checkpoint stack
            let mut drafts = lock_mutex(&self.drafts);
            drafts.insert(id, DraftBuffer::new());

            let mut checkpoints = lock_mutex(&self.checkpoints);
            checkpoints.insert(id, CheckpointStack::new());
        }

        id
    }

    /// Try to create a root context.
    ///
    /// Returns [`ContextError::ContextAlreadyExists`] if the ID is already in
    /// use. This is the strict variant of [`Self::create_root_context`].
    pub fn try_create_root_context(&self, id: ContextId) -> Result<ContextId> {
        {
            let mut tree = write_lock(&self.context_tree);
            tree.try_create_root(id)?;
        }

        let mut drafts = lock_mutex(&self.drafts);
        drafts.insert(id, DraftBuffer::new());

        let mut checkpoints = lock_mutex(&self.checkpoints);
        checkpoints.insert(id, CheckpointStack::new());

        Ok(id)
    }

    /// Create a child context.
    ///
    /// Child contexts inherit visibility from their parent, forming
    /// a hierarchy of lexical scopes.
    ///
    /// # Arguments
    ///
    /// * `id` - Unique identifier for the new context
    /// * `parent_id` - Parent context ID
    ///
    /// # Returns
    ///
    /// `Ok(context_id)` on success, `Err(ContextError)` if parent doesn't exist
    /// or ID is already in use.
    ///
    /// # Examples
    ///
    /// ```
    /// use liblevenshtein::contextual::DynamicContextualCompletionEngine;
    ///
    /// let engine = DynamicContextualCompletionEngine::new();
    /// let global = engine.create_root_context(0);
    /// let func = engine.create_child_context(1, global).expect("test fixture: create_child_context with valid parent");
    /// assert_eq!(func, 1);
    /// ```
    pub fn create_child_context(&self, id: ContextId, parent_id: ContextId) -> Result<ContextId> {
        let mut tree = write_lock(&self.context_tree);
        tree.create_child(id, parent_id)?;

        // Initialize empty draft and checkpoint stack
        let mut drafts = lock_mutex(&self.drafts);
        drafts.insert(id, DraftBuffer::new());

        let mut checkpoints = lock_mutex(&self.checkpoints);
        checkpoints.insert(id, CheckpointStack::new());

        Ok(id)
    }

    /// Remove a context and all its descendants.
    ///
    /// This also cleans up associated drafts and checkpoints.
    ///
    /// # Arguments
    ///
    /// * `id` - Context ID to remove
    ///
    /// # Returns
    ///
    /// `true` if the context was removed, `false` if it didn't exist.
    ///
    /// # Examples
    ///
    /// ```
    /// use liblevenshtein::contextual::DynamicContextualCompletionEngine;
    ///
    /// let engine = DynamicContextualCompletionEngine::new();
    /// let global = engine.create_root_context(0);
    /// let func = engine.create_child_context(1, global).expect("test fixture: create_child_context with valid parent");
    ///
    /// assert!(engine.remove_context(func));
    /// assert!(!engine.remove_context(func)); // Already removed
    /// ```
    pub fn remove_context(&self, id: ContextId) -> bool {
        let mut tree = write_lock(&self.context_tree);
        let removed = tree.remove(id);

        if removed {
            // Clean up drafts and checkpoints for removed context
            let mut drafts = lock_mutex(&self.drafts);
            drafts.retain(|ctx_id, _| tree.contains(*ctx_id));

            let mut checkpoints = lock_mutex(&self.checkpoints);
            checkpoints.retain(|ctx_id, _| tree.contains(*ctx_id));
        }

        removed
    }

    /// Get all contexts visible from a given context (including itself).
    ///
    /// Returns contexts in order: self, parent, grandparent, ..., root.
    ///
    /// # Arguments
    ///
    /// * `id` - Context ID to query from
    ///
    /// # Returns
    ///
    /// Vector of visible context IDs (empty if context doesn't exist).
    ///
    /// # Examples
    ///
    /// ```
    /// use liblevenshtein::contextual::DynamicContextualCompletionEngine;
    ///
    /// let engine = DynamicContextualCompletionEngine::new();
    /// let global = engine.create_root_context(0);
    /// let module = engine.create_child_context(1, global).expect("test fixture: create_child_context with valid parent");
    /// let func = engine.create_child_context(2, module).expect("test fixture: create_child_context with valid parent");
    ///
    /// let visible = engine.get_visible_contexts(func);
    /// assert_eq!(visible, vec![func, module, global]);
    /// ```
    pub fn get_visible_contexts(&self, id: ContextId) -> Vec<ContextId> {
        let tree = read_lock(&self.context_tree);
        tree.visible_contexts(id)
    }

    /// Check if a context exists.
    ///
    /// # Arguments
    ///
    /// * `id` - Context ID to check
    ///
    /// # Examples
    ///
    /// ```
    /// use liblevenshtein::contextual::DynamicContextualCompletionEngine;
    ///
    /// let engine = DynamicContextualCompletionEngine::new();
    /// assert!(!engine.context_exists(0));
    ///
    /// let global = engine.create_root_context(0);
    /// assert!(engine.context_exists(0));
    /// ```
    pub fn context_exists(&self, id: ContextId) -> bool {
        let tree = read_lock(&self.context_tree);
        tree.depth(id).is_some()
    }

    /// Get the current draft text for a context.
    ///
    /// # Arguments
    ///
    /// * `context` - Context ID
    ///
    /// # Returns
    ///
    /// `Some(draft_text)` if context exists and has a draft, `None` otherwise.
    ///
    /// # Examples
    ///
    /// ```
    /// use liblevenshtein::contextual::DynamicContextualCompletionEngine;
    ///
    /// let engine = DynamicContextualCompletionEngine::new();
    /// let ctx = engine.create_root_context(0);
    ///
    /// assert_eq!(engine.get_draft(ctx), Some(String::new()));
    ///
    /// engine.insert_str(ctx, "hello");
    /// assert_eq!(engine.get_draft(ctx), Some("hello".to_string()));
    /// ```
    pub fn get_draft(&self, context: ContextId) -> Option<String> {
        let drafts = lock_mutex(&self.drafts);
        drafts.get(&context).map(|buf| buf.as_str())
    }

    /// Check if a context has any draft text.
    ///
    /// # Arguments
    ///
    /// * `context` - Context ID
    ///
    /// # Returns
    ///
    /// `true` if the context has non-empty draft text.
    ///
    /// # Examples
    ///
    /// ```
    /// use liblevenshtein::contextual::DynamicContextualCompletionEngine;
    ///
    /// let engine = DynamicContextualCompletionEngine::new();
    /// let ctx = engine.create_root_context(0);
    ///
    /// assert!(!engine.has_draft(ctx));
    ///
    /// engine.insert_str(ctx, "hello");
    /// assert!(engine.has_draft(ctx));
    /// ```
    pub fn has_draft(&self, context: ContextId) -> bool {
        let drafts = lock_mutex(&self.drafts);
        drafts
            .get(&context)
            .map(|buf| !buf.is_empty())
            .unwrap_or(false)
    }

    /// Insert a single character into the draft buffer for a context.
    ///
    /// # Arguments
    ///
    /// * `context` - Context ID
    /// * `ch` - Character to insert
    ///
    /// # Returns
    ///
    /// `Ok(())` on success, `Err(ContextError)` if context doesn't exist.
    ///
    /// # Examples
    ///
    /// ```
    /// use liblevenshtein::contextual::DynamicContextualCompletionEngine;
    ///
    /// let engine = DynamicContextualCompletionEngine::new();
    /// let ctx = engine.create_root_context(0);
    ///
    /// engine.insert_char(ctx, 'h').expect("test fixture: insert_char on existing context");
    /// engine.insert_char(ctx, 'i').expect("test fixture: insert_char on existing context");
    /// assert_eq!(engine.get_draft(ctx), Some("hi".to_string()));
    /// ```
    pub fn insert_char(&self, context: ContextId, ch: char) -> Result<()> {
        if !self.context_exists(context) {
            return Err(ContextError::ContextNotFound(context));
        }

        let mut drafts = lock_mutex(&self.drafts);
        if let Some(buffer) = drafts.get_mut(&context) {
            buffer.insert(ch);
            Ok(())
        } else {
            Err(ContextError::NoDraftBuffer(context))
        }
    }

    /// Insert a string into the draft buffer for a context.
    ///
    /// # Arguments
    ///
    /// * `context` - Context ID
    /// * `s` - String to insert
    ///
    /// # Returns
    ///
    /// `Ok(())` on success, `Err(ContextError)` if context doesn't exist.
    ///
    /// # Examples
    ///
    /// ```
    /// use liblevenshtein::contextual::DynamicContextualCompletionEngine;
    ///
    /// let engine = DynamicContextualCompletionEngine::new();
    /// let ctx = engine.create_root_context(0);
    ///
    /// engine.insert_str(ctx, "hello").expect("test fixture: insert_str on existing context");
    /// engine.insert_str(ctx, " world").expect("test fixture: insert_str on existing context");
    /// assert_eq!(engine.get_draft(ctx), Some("hello world".to_string()));
    /// ```
    pub fn insert_str(&self, context: ContextId, s: &str) -> Result<()> {
        if !self.context_exists(context) {
            return Err(ContextError::ContextNotFound(context));
        }
        if s.is_empty() {
            return Ok(());
        }

        let mut drafts = lock_mutex(&self.drafts);
        if let Some(buffer) = drafts.get_mut(&context) {
            buffer.insert_str(s);
            Ok(())
        } else {
            Err(ContextError::NoDraftBuffer(context))
        }
    }

    /// Delete the last character from the draft buffer.
    ///
    /// # Arguments
    ///
    /// * `context` - Context ID
    ///
    /// # Returns
    ///
    /// `Some(char)` if a character was deleted, `None` if buffer was empty
    /// or context doesn't exist.
    ///
    /// # Examples
    ///
    /// ```
    /// use liblevenshtein::contextual::DynamicContextualCompletionEngine;
    ///
    /// let engine = DynamicContextualCompletionEngine::new();
    /// let ctx = engine.create_root_context(0);
    ///
    /// engine.insert_str(ctx, "hello").expect("test fixture: insert_str on existing context");
    /// assert_eq!(engine.delete_char(ctx), Some('o'));
    /// assert_eq!(engine.get_draft(ctx), Some("hell".to_string()));
    /// ```
    pub fn delete_char(&self, context: ContextId) -> Option<char> {
        let mut drafts = lock_mutex(&self.drafts);
        drafts.get_mut(&context).and_then(|buf| buf.delete())
    }

    /// Clear the draft buffer for a context.
    ///
    /// # Arguments
    ///
    /// * `context` - Context ID
    ///
    /// # Returns
    ///
    /// `Ok(())` on success, `Err(ContextError)` if context doesn't exist.
    ///
    /// # Examples
    ///
    /// ```
    /// use liblevenshtein::contextual::DynamicContextualCompletionEngine;
    ///
    /// let engine = DynamicContextualCompletionEngine::new();
    /// let ctx = engine.create_root_context(0);
    ///
    /// engine.insert_str(ctx, "hello").expect("test fixture: insert_str on existing context");
    /// assert!(engine.has_draft(ctx));
    ///
    /// engine.clear_draft(ctx).expect("test fixture: clear_draft on existing context");
    /// assert!(!engine.has_draft(ctx));
    /// ```
    pub fn clear_draft(&self, context: ContextId) -> Result<()> {
        if !self.context_exists(context) {
            return Err(ContextError::ContextNotFound(context));
        }

        let mut drafts = lock_mutex(&self.drafts);
        if let Some(buffer) = drafts.get_mut(&context) {
            buffer.clear();
            Ok(())
        } else {
            Err(ContextError::NoDraftBuffer(context))
        }
    }

    /// Create a checkpoint of the current draft state.
    ///
    /// Checkpoints enable undo functionality by saving the buffer position.
    ///
    /// # Arguments
    ///
    /// * `context` - Context ID
    ///
    /// # Returns
    ///
    /// `Ok(())` on success, `Err(ContextError)` if context doesn't exist.
    ///
    /// # Examples
    ///
    /// ```
    /// use liblevenshtein::contextual::DynamicContextualCompletionEngine;
    ///
    /// let engine = DynamicContextualCompletionEngine::new();
    /// let ctx = engine.create_root_context(0);
    ///
    /// engine.insert_str(ctx, "hello").expect("test fixture: insert_str on existing context");
    /// engine.checkpoint(ctx).expect("test fixture: checkpoint on existing context");
    ///
    /// engine.insert_str(ctx, " world").expect("test fixture: insert_str on existing context");
    /// assert_eq!(engine.get_draft(ctx), Some("hello world".to_string()));
    ///
    /// // Undo to checkpoint
    /// engine.undo(ctx).expect("test fixture: undo with available checkpoint");
    /// assert_eq!(engine.get_draft(ctx), Some("hello".to_string()));
    /// ```
    pub fn checkpoint(&self, context: ContextId) -> Result<()> {
        if !self.context_exists(context) {
            return Err(ContextError::ContextNotFound(context));
        }

        let drafts = lock_mutex(&self.drafts);
        let buffer = drafts
            .get(&context)
            .ok_or(ContextError::NoDraftBuffer(context))?;

        let mut checkpoints = lock_mutex(&self.checkpoints);
        let stack = checkpoints
            .get_mut(&context)
            .ok_or(ContextError::NoCheckpointStack(context))?;

        stack.push_from_buffer(buffer);
        Ok(())
    }

    /// Undo to the last checkpoint.
    ///
    /// Pops the most recent checkpoint and restores the buffer to the
    /// previous checkpoint position.
    ///
    /// # Arguments
    ///
    /// * `context` - Context ID
    ///
    /// # Returns
    ///
    /// `Ok(())` on success, `Err(ContextError)` if context doesn't exist or
    /// no checkpoints available.
    ///
    /// # Examples
    ///
    /// ```
    /// use liblevenshtein::contextual::DynamicContextualCompletionEngine;
    ///
    /// let engine = DynamicContextualCompletionEngine::new();
    /// let ctx = engine.create_root_context(0);
    ///
    /// engine.checkpoint(ctx).expect("test fixture: checkpoint on existing context"); // Empty checkpoint
    /// engine.insert_str(ctx, "hello").expect("test fixture: insert_str on existing context");
    /// engine.checkpoint(ctx).expect("test fixture: checkpoint on existing context"); // "hello" checkpoint
    ///
    /// engine.insert_str(ctx, " world").expect("test fixture: insert_str on existing context");
    /// assert_eq!(engine.get_draft(ctx), Some("hello world".to_string()));
    ///
    /// engine.undo(ctx).expect("test fixture: undo with available checkpoint"); // Restore to "hello"
    /// assert_eq!(engine.get_draft(ctx), Some("hello".to_string()));
    /// ```
    pub fn undo(&self, context: ContextId) -> Result<()> {
        if !self.context_exists(context) {
            return Err(ContextError::ContextNotFound(context));
        }

        let checkpoint = {
            let checkpoints = lock_mutex(&self.checkpoints);
            let stack = checkpoints
                .get(&context)
                .ok_or(ContextError::NoCheckpointStack(context))?;

            *stack.peek().ok_or(ContextError::NoCheckpoints(context))?
        };

        // Restore buffer
        let mut drafts = lock_mutex(&self.drafts);
        let buffer = drafts
            .get_mut(&context)
            .ok_or(ContextError::NoDraftBuffer(context))?;

        checkpoint.restore(buffer);
        drop(drafts);

        // Now pop the checkpoint after successful restore, without holding the
        // drafts lock. If a concurrent operation changed the stack, leave it
        // intact rather than consuming a different checkpoint.
        let mut checkpoints = lock_mutex(&self.checkpoints);
        let stack = checkpoints
            .get_mut(&context)
            .ok_or(ContextError::NoCheckpointStack(context))?;
        if stack.peek() == Some(&checkpoint) {
            stack.pop();
        }

        Ok(())
    }

    /// Get the number of available checkpoints for a context.
    ///
    /// # Arguments
    ///
    /// * `context` - Context ID
    ///
    /// # Returns
    ///
    /// Number of checkpoints (0 if context doesn't exist).
    ///
    /// # Examples
    ///
    /// ```
    /// use liblevenshtein::contextual::DynamicContextualCompletionEngine;
    ///
    /// let engine = DynamicContextualCompletionEngine::new();
    /// let ctx = engine.create_root_context(0);
    ///
    /// assert_eq!(engine.checkpoint_count(ctx), 0);
    ///
    /// engine.checkpoint(ctx).expect("test fixture: checkpoint on existing context");
    /// assert_eq!(engine.checkpoint_count(ctx), 1);
    ///
    /// engine.checkpoint(ctx).expect("test fixture: checkpoint on existing context");
    /// assert_eq!(engine.checkpoint_count(ctx), 2);
    /// ```
    pub fn checkpoint_count(&self, context: ContextId) -> usize {
        let checkpoints = lock_mutex(&self.checkpoints);
        checkpoints.get(&context).map(|s| s.len()).unwrap_or(0)
    }

    /// Clear all checkpoints for a context.
    ///
    /// # Arguments
    ///
    /// * `context` - Context ID
    ///
    /// # Returns
    ///
    /// `Ok(())` on success, `Err(ContextError)` if context doesn't exist.
    ///
    /// # Examples
    ///
    /// ```
    /// use liblevenshtein::contextual::DynamicContextualCompletionEngine;
    ///
    /// let engine = DynamicContextualCompletionEngine::new();
    /// let ctx = engine.create_root_context(0);
    ///
    /// engine.checkpoint(ctx).expect("test fixture: checkpoint on existing context");
    /// engine.checkpoint(ctx).expect("test fixture: checkpoint on existing context");
    /// assert_eq!(engine.checkpoint_count(ctx), 2);
    ///
    /// engine.clear_checkpoints(ctx).expect("test fixture: clear_checkpoints on existing context");
    /// assert_eq!(engine.checkpoint_count(ctx), 0);
    /// ```
    pub fn clear_checkpoints(&self, context: ContextId) -> Result<()> {
        if !self.context_exists(context) {
            return Err(ContextError::ContextNotFound(context));
        }

        let mut checkpoints = lock_mutex(&self.checkpoints);
        if let Some(stack) = checkpoints.get_mut(&context) {
            stack.clear();
            Ok(())
        } else {
            Err(ContextError::NoCheckpointStack(context))
        }
    }

    /// Finalize the current draft into the dictionary.
    ///
    /// Moves the draft text from the buffer into the finalized dictionary,
    /// associating it with the current context. The draft buffer is then
    /// cleared, and all checkpoints are removed.
    ///
    /// # Arguments
    ///
    /// * `context` - Context ID
    ///
    /// # Returns
    ///
    /// `Ok(term)` with the finalized term on success, `Err(ContextError)` if
    /// context doesn't exist or has no draft text.
    ///
    /// # Examples
    ///
    /// ```
    /// use liblevenshtein::contextual::DynamicContextualCompletionEngine;
    ///
    /// let engine = DynamicContextualCompletionEngine::new();
    /// let ctx = engine.create_root_context(0);
    ///
    /// engine.insert_str(ctx, "hello").expect("test fixture: insert_str on existing context");
    /// let term = engine.finalize(ctx).expect("test fixture: finalize with non-empty draft");
    /// assert_eq!(term, "hello");
    ///
    /// // Draft is cleared after finalization
    /// assert!(!engine.has_draft(ctx));
    /// ```
    pub fn finalize(&self, context: ContextId) -> Result<String> {
        if !self.context_exists(context) {
            return Err(ContextError::ContextNotFound(context));
        }

        // Get and validate draft
        let mut drafts = lock_mutex(&self.drafts);
        let buffer = drafts
            .get_mut(&context)
            .ok_or(ContextError::NoDraftBuffer(context))?;

        if buffer.is_empty() {
            return Err(ContextError::EmptyDraft(context));
        }

        let term_owned = buffer.as_slice().to_owned();

        // Clear draft
        buffer.clear();
        drop(drafts);

        // Add to dictionary
        let transducer = read_lock(&self.transducer);
        let dictionary = transducer.dictionary();

        // Get existing contexts for this term (if any) and append the new context
        let mut contexts = dictionary.get_value(&term_owned).unwrap_or_default();
        if !contexts.contains(&context) {
            contexts.push(context);
        }
        dictionary.insert_with_value(&term_owned, contexts);
        drop(transducer);

        // Clear checkpoints
        let mut checkpoints = lock_mutex(&self.checkpoints);
        if let Some(stack) = checkpoints.get_mut(&context) {
            stack.clear();
        }

        Ok(term_owned)
    }

    /// Finalize a term directly into the dictionary without a draft.
    ///
    /// Inserts a term into the dictionary associated with a context,
    /// bypassing the draft buffer entirely.
    ///
    /// # Arguments
    ///
    /// * `context` - Context ID
    /// * `term` - Term to insert
    ///
    /// # Returns
    ///
    /// `Ok(())` on success, `Err(ContextError)` if context doesn't exist.
    ///
    /// # Examples
    ///
    /// ```
    /// use liblevenshtein::contextual::DynamicContextualCompletionEngine;
    ///
    /// let engine = DynamicContextualCompletionEngine::new();
    /// let ctx = engine.create_root_context(0);
    ///
    /// engine.finalize_direct(ctx, "function").expect("test fixture: finalize_direct with non-empty term");
    /// engine.finalize_direct(ctx, "variable").expect("test fixture: finalize_direct with non-empty term");
    /// ```
    pub fn finalize_direct(&self, context: ContextId, term: &str) -> Result<()> {
        if !self.context_exists(context) {
            return Err(ContextError::ContextNotFound(context));
        }

        if term.is_empty() {
            return Err(ContextError::EmptyTerm);
        }

        let transducer = read_lock(&self.transducer);
        let dictionary = transducer.dictionary();

        // Get existing contexts for this term (if any) and append the new context
        let mut contexts = dictionary.get_value(term).unwrap_or_default();
        if !contexts.contains(&context) {
            contexts.push(context);
        }
        dictionary.insert_with_value(term, contexts);

        Ok(())
    }

    /// Discard the current draft without finalizing.
    ///
    /// Clears the draft buffer and all checkpoints without adding the
    /// draft to the dictionary.
    ///
    /// # Arguments
    ///
    /// * `context` - Context ID
    ///
    /// # Returns
    ///
    /// `Ok(())` on success, `Err(ContextError)` if context doesn't exist.
    ///
    /// # Examples
    ///
    /// ```
    /// use liblevenshtein::contextual::DynamicContextualCompletionEngine;
    ///
    /// let engine = DynamicContextualCompletionEngine::new();
    /// let ctx = engine.create_root_context(0);
    ///
    /// engine.insert_str(ctx, "mistake").expect("test fixture: insert_str on existing context");
    /// engine.discard(ctx).expect("test fixture: discard on existing context");
    ///
    /// // Draft is cleared
    /// assert!(!engine.has_draft(ctx));
    /// ```
    pub fn discard(&self, context: ContextId) -> Result<()> {
        if !self.context_exists(context) {
            return Err(ContextError::ContextNotFound(context));
        }

        // Clear draft
        self.clear_draft(context)?;

        // Clear checkpoints
        self.clear_checkpoints(context)?;

        Ok(())
    }

    /// Check if a term exists in the dictionary.
    ///
    /// # Arguments
    ///
    /// * `term` - Term to check
    ///
    /// # Returns
    ///
    /// `true` if the term exists in the dictionary.
    ///
    /// # Examples
    ///
    /// ```
    /// use liblevenshtein::contextual::DynamicContextualCompletionEngine;
    ///
    /// let engine = DynamicContextualCompletionEngine::new();
    /// let ctx = engine.create_root_context(0);
    ///
    /// assert!(!engine.has_term("hello"));
    ///
    /// engine.finalize_direct(ctx, "hello").expect("test fixture: finalize_direct with non-empty term");
    /// assert!(engine.has_term("hello"));
    /// ```
    pub fn has_term(&self, term: &str) -> bool {
        let transducer = read_lock(&self.transducer);
        transducer.dictionary().contains(term)
    }

    /// Get all contexts where a term is defined.
    ///
    /// # Arguments
    ///
    /// * `term` - Term to look up
    ///
    /// # Returns
    ///
    /// Vector of context IDs where the term is defined (empty if not found).
    ///
    /// # Examples
    ///
    /// ```
    /// use liblevenshtein::contextual::DynamicContextualCompletionEngine;
    ///
    /// let engine = DynamicContextualCompletionEngine::new();
    /// let global = engine.create_root_context(0);
    /// let func = engine.create_child_context(1, global).expect("test fixture: create_child_context with valid parent");
    ///
    /// engine.finalize_direct(global, "global_var").expect("test fixture: finalize_direct with non-empty term");
    /// engine.finalize_direct(func, "local_var").expect("test fixture: finalize_direct with non-empty term");
    ///
    /// assert_eq!(engine.term_contexts("global_var"), vec![global]);
    /// assert_eq!(engine.term_contexts("local_var"), vec![func]);
    /// assert!(engine.term_contexts("unknown").is_empty());
    /// ```
    pub fn term_contexts(&self, term: &str) -> Vec<ContextId> {
        let transducer = read_lock(&self.transducer);
        transducer.dictionary().get_value(term).unwrap_or_default()
    }

    /// Query for completions from both drafts and finalized terms.
    ///
    /// Performs fuzzy matching against:
    /// 1. The current context's draft (if any)
    /// 2. All finalized terms visible from the current context
    ///
    /// Results are deduplicated (draft overrides finalized with same term)
    /// and sorted by distance.
    ///
    /// # Arguments
    ///
    /// * `context` - Context ID to query from
    /// * `query` - Query string
    /// * `max_distance` - Maximum edit distance threshold
    ///
    /// # Returns
    ///
    /// Vector of completions sorted by distance, draft status, and term.
    ///
    /// # Examples
    ///
    /// ```
    /// use liblevenshtein::contextual::DynamicContextualCompletionEngine;
    ///
    /// let engine = DynamicContextualCompletionEngine::new();
    /// let ctx = engine.create_root_context(0);
    ///
    /// // Add finalized terms
    /// engine.finalize_direct(ctx, "hello").expect("test fixture: finalize_direct with non-empty term");
    /// engine.finalize_direct(ctx, "help").expect("test fixture: finalize_direct with non-empty term");
    ///
    /// // Add draft
    /// engine.insert_str(ctx, "hero").expect("test fixture: insert_str on existing context");
    ///
    /// // Query
    /// let results = engine.complete(ctx, "hel", 2);
    /// assert!(!results.is_empty());
    /// ```
    pub fn complete(
        &self,
        context: ContextId,
        query: &str,
        max_distance: usize,
    ) -> Vec<Completion> {
        let draft_results = self.complete_drafts(context, query, max_distance);
        let finalized_results = self.complete_finalized(context, query, max_distance);

        let mut by_term = HashMap::with_capacity(draft_results.len() + finalized_results.len());
        for completion in draft_results {
            by_term.entry(completion.term.clone()).or_insert(completion);
        }
        for completion in finalized_results {
            by_term.entry(completion.term.clone()).or_insert(completion);
        }

        let mut results: Vec<Completion> = by_term.into_values().collect();
        results.sort();
        results
    }

    /// Query only draft terms.
    ///
    /// Returns completions from visible draft buffers only.
    ///
    /// # Arguments
    ///
    /// * `context` - Context ID to query from
    /// * `query` - Query string
    /// * `max_distance` - Maximum edit distance threshold
    ///
    /// # Returns
    ///
    /// Vector of draft completions.
    ///
    /// # Examples
    ///
    /// ```
    /// use liblevenshtein::contextual::DynamicContextualCompletionEngine;
    ///
    /// let engine = DynamicContextualCompletionEngine::new();
    /// let ctx = engine.create_root_context(0);
    ///
    /// engine.insert_str(ctx, "hello").expect("test fixture: insert_str on existing context");
    ///
    /// let results = engine.complete_drafts(ctx, "helo", 1);
    /// assert_eq!(results.len(), 1);
    /// assert!(results[0].is_draft);
    /// ```
    pub fn complete_drafts(
        &self,
        context: ContextId,
        query: &str,
        max_distance: usize,
    ) -> Vec<Completion> {
        // Get visible contexts
        let visible = self.get_visible_contexts(context);
        let mut results = Vec::with_capacity(visible.len());

        // Check each visible context's draft
        let drafts = lock_mutex(&self.drafts);
        for ctx_id in visible {
            if let Some(buffer) = drafts.get(&ctx_id) {
                let term = buffer.as_slice();
                if !term.is_empty() {
                    if let Some(distance) =
                        Self::levenshtein_distance_within(query, term, max_distance)
                    {
                        results.push(Completion::draft(term.to_owned(), distance, ctx_id));
                    }
                }
            }
        }

        results
    }

    /// Query only finalized terms.
    ///
    /// Returns completions from the dictionary, filtered by visibility.
    ///
    /// # Arguments
    ///
    /// * `context` - Context ID to query from
    /// * `query` - Query string
    /// * `max_distance` - Maximum edit distance threshold
    ///
    /// # Returns
    ///
    /// Vector of finalized completions.
    ///
    /// # Examples
    ///
    /// ```
    /// use liblevenshtein::contextual::DynamicContextualCompletionEngine;
    ///
    /// let engine = DynamicContextualCompletionEngine::new();
    /// let ctx = engine.create_root_context(0);
    ///
    /// engine.finalize_direct(ctx, "hello").expect("test fixture: finalize_direct with non-empty term");
    /// engine.finalize_direct(ctx, "help").expect("test fixture: finalize_direct with non-empty term");
    ///
    /// let results = engine.complete_finalized(ctx, "helo", 1);
    /// assert!(results.len() >= 2);
    /// assert!(!results[0].is_draft);
    /// ```
    pub fn complete_finalized(
        &self,
        context: ContextId,
        query: &str,
        max_distance: usize,
    ) -> Vec<Completion> {
        // Get visible contexts
        let visible = self.get_visible_contexts(context);
        if visible.is_empty() {
            return Vec::new();
        }
        let visible_set: HashSet<ContextId> = visible.iter().copied().collect();
        let mut results = Vec::with_capacity(visible.len());

        // Query dictionary using transducer
        let transducer = read_lock(&self.transducer);
        for candidate in transducer.query_with_distance(query, max_distance) {
            // Get the contexts for this term
            if let Some(contexts) = transducer.dictionary().get_value(&candidate.term) {
                // Filter to only visible contexts
                let mut visible_contexts = Vec::with_capacity(contexts.len().min(visible.len()));
                for ctx_id in contexts {
                    if visible_set.contains(&ctx_id) {
                        visible_contexts.push(ctx_id);
                    }
                }

                if !visible_contexts.is_empty() {
                    results.push(Completion::finalized(
                        candidate.term,
                        candidate.distance,
                        visible_contexts,
                    ));
                }
            }
        }

        results
    }

    /// Threshold-bounded Levenshtein distance calculation for draft matching.
    ///
    /// Returns `None` as soon as the distance is known to exceed `max_distance`.
    /// Finalized terms use the transducer for efficient automaton-based matching.
    fn levenshtein_distance_within(s1: &str, s2: &str, max_distance: usize) -> Option<usize> {
        crate::distance::standard_distance_bounded(s1, s2, max_distance)
    }

    /// Full Levenshtein distance calculation.
    #[cfg(test)]
    fn levenshtein_distance(s1: &str, s2: &str) -> usize {
        crate::distance::standard_distance(s1, s2)
    }
}

impl Default for DynamicContextualCompletionEngine<PathMapDictionary<Vec<ContextId>>> {
    fn default() -> Self {
        Self::new()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_new() {
        let engine = DynamicContextualCompletionEngine::new();
        assert!(!engine.context_exists(0));
    }

    #[test]
    fn test_with_algorithm() {
        let engine = DynamicContextualCompletionEngine::with_algorithm(Algorithm::Transposition);
        assert!(!engine.context_exists(0));
    }

    #[test]
    fn recovers_after_internal_locks_are_poisoned() {
        let engine = std::sync::Arc::new(DynamicContextualCompletionEngine::new());
        let ctx = engine.create_root_context(0);

        let thread_engine = std::sync::Arc::clone(&engine);
        let result = std::thread::spawn(move || {
            let _guard = thread_engine.context_tree.write().unwrap();
            panic!("intentional context tree poisoning for engine recovery test");
        })
        .join();
        assert!(result.is_err());
        assert!(engine.context_exists(ctx));

        let thread_engine = std::sync::Arc::clone(&engine);
        let result = std::thread::spawn(move || {
            let mut drafts = thread_engine.drafts.lock().unwrap();
            let buffer = drafts
                .get_mut(&ctx)
                .expect("test fixture: root context has an initialized draft buffer");
            buffer.insert_str("poisoned-draft");
            panic!("intentional draft poisoning for engine recovery test");
        })
        .join();
        assert!(result.is_err());
        assert_eq!(engine.get_draft(ctx), Some("poisoned-draft".to_string()));

        engine
            .checkpoint(ctx)
            .expect("test fixture: checkpoint on existing context");
        let thread_engine = std::sync::Arc::clone(&engine);
        let result = std::thread::spawn(move || {
            let mut checkpoints = thread_engine.checkpoints.lock().unwrap();
            let stack = checkpoints
                .get_mut(&ctx)
                .expect("test fixture: root context has an initialized checkpoint stack");
            stack.clear();
            panic!("intentional checkpoint poisoning for engine recovery test");
        })
        .join();
        assert!(result.is_err());
        assert_eq!(engine.checkpoint_count(ctx), 0);

        let thread_engine = std::sync::Arc::clone(&engine);
        let result = std::thread::spawn(move || {
            let _guard = thread_engine.transducer.write().unwrap();
            panic!("intentional transducer poisoning for engine recovery test");
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
    fn test_create_root_context() {
        let engine = DynamicContextualCompletionEngine::new();
        let ctx = engine.create_root_context(0);
        assert_eq!(ctx, 0);
        assert!(engine.context_exists(0));
    }

    #[test]
    fn test_try_create_root_context_rejects_duplicate_id() {
        let engine = DynamicContextualCompletionEngine::new();
        engine
            .try_create_root_context(0)
            .expect("test fixture: first root create succeeds");

        assert_eq!(
            engine.try_create_root_context(0),
            Err(ContextError::ContextAlreadyExists(0))
        );
    }

    #[test]
    fn test_duplicate_root_context_preserves_existing_state_and_parent() {
        let engine = DynamicContextualCompletionEngine::new();
        let root = engine.create_root_context(0);
        let child = engine
            .create_child_context(1, root)
            .expect("test fixture: create child");

        engine
            .insert_str(child, "local")
            .expect("test fixture: insert child draft");
        engine
            .checkpoint(child)
            .expect("test fixture: checkpoint child draft");

        assert_eq!(engine.create_root_context(child), child);

        assert_eq!(engine.get_visible_contexts(child), vec![child, root]);
        assert_eq!(engine.get_draft(child), Some("local".to_string()));
        assert_eq!(engine.checkpoint_count(child), 1);
    }

    #[test]
    fn test_create_child_context() {
        let engine = DynamicContextualCompletionEngine::new();
        let root = engine.create_root_context(0);
        let child = engine
            .create_child_context(1, root)
            .expect("test fixture: create_child_context with valid parent");

        assert_eq!(child, 1);
        assert!(engine.context_exists(1));
    }

    #[test]
    fn test_create_child_context_rejects_duplicate_id() {
        let engine = DynamicContextualCompletionEngine::new();
        let root = engine.create_root_context(0);
        let child = engine
            .create_child_context(1, root)
            .expect("test fixture: create child");

        assert_eq!(
            engine.create_child_context(child, root),
            Err(ContextError::ContextAlreadyExists(child))
        );
        assert_eq!(engine.get_visible_contexts(child), vec![child, root]);
    }

    #[test]
    fn test_create_child_context_rejects_self_parent_cycle() {
        let engine = DynamicContextualCompletionEngine::new();
        let root = engine.create_root_context(0);

        assert_eq!(
            engine.create_child_context(root, root),
            Err(ContextError::CircularHierarchy(root, root))
        );
        assert_eq!(engine.get_visible_contexts(root), vec![root]);
    }

    #[test]
    fn test_create_child_invalid_parent() {
        let engine = DynamicContextualCompletionEngine::new();
        let result = engine.create_child_context(1, 999);

        assert!(result.is_err());
        assert!(!engine.context_exists(1));
    }

    #[test]
    fn test_remove_context() {
        let engine = DynamicContextualCompletionEngine::new();
        let root = engine.create_root_context(0);
        let child = engine
            .create_child_context(1, root)
            .expect("test fixture: create_child_context with valid parent");

        assert!(engine.remove_context(child));
        assert!(!engine.context_exists(child));
        assert!(engine.context_exists(root));

        // Removing again returns false
        assert!(!engine.remove_context(child));
    }

    #[test]
    fn test_remove_context_with_descendants() {
        let engine = DynamicContextualCompletionEngine::new();
        let root = engine.create_root_context(0);
        let child1 = engine
            .create_child_context(1, root)
            .expect("test fixture: create_child_context with valid parent");
        let child2 = engine
            .create_child_context(2, child1)
            .expect("test fixture: create_child_context with valid parent");

        // Remove child1 (should also remove child2)
        assert!(engine.remove_context(child1));
        assert!(!engine.context_exists(child1));
        assert!(!engine.context_exists(child2));
        assert!(engine.context_exists(root));
    }

    #[test]
    fn test_get_visible_contexts() {
        let engine = DynamicContextualCompletionEngine::new();
        let global = engine.create_root_context(0);
        let module = engine
            .create_child_context(1, global)
            .expect("test fixture: create_child_context with valid parent");
        let func = engine
            .create_child_context(2, module)
            .expect("test fixture: create_child_context with valid parent");

        let visible = engine.get_visible_contexts(func);
        assert_eq!(visible, vec![func, module, global]);

        let visible_module = engine.get_visible_contexts(module);
        assert_eq!(visible_module, vec![module, global]);

        let visible_global = engine.get_visible_contexts(global);
        assert_eq!(visible_global, vec![global]);
    }

    #[test]
    fn test_get_draft_empty() {
        let engine = DynamicContextualCompletionEngine::new();
        let ctx = engine.create_root_context(0);

        assert_eq!(engine.get_draft(ctx), Some(String::new()));
        assert!(!engine.has_draft(ctx));
    }

    #[test]
    fn test_get_draft_nonexistent() {
        let engine = DynamicContextualCompletionEngine::new();
        assert_eq!(engine.get_draft(999), None);
        assert!(!engine.has_draft(999));
    }

    #[test]
    fn test_insert_char() {
        let engine = DynamicContextualCompletionEngine::new();
        let ctx = engine.create_root_context(0);

        engine
            .insert_char(ctx, 'h')
            .expect("test fixture: insert_char on existing context");
        engine
            .insert_char(ctx, 'i')
            .expect("test fixture: insert_char on existing context");
        assert_eq!(engine.get_draft(ctx), Some("hi".to_string()));
        assert!(engine.has_draft(ctx));
    }

    #[test]
    fn test_insert_char_nonexistent_context() {
        let engine = DynamicContextualCompletionEngine::new();
        let result = engine.insert_char(999, 'x');
        assert!(result.is_err());
    }

    #[test]
    fn test_insert_str() {
        let engine = DynamicContextualCompletionEngine::new();
        let ctx = engine.create_root_context(0);

        engine
            .insert_str(ctx, "hello")
            .expect("test fixture: insert_str on existing context");
        assert_eq!(engine.get_draft(ctx), Some("hello".to_string()));

        engine
            .insert_str(ctx, " world")
            .expect("test fixture: insert_str on existing context");
        assert_eq!(engine.get_draft(ctx), Some("hello world".to_string()));
    }

    #[test]
    fn test_insert_empty_str_is_noop() {
        let engine = DynamicContextualCompletionEngine::new();
        let ctx = engine.create_root_context(0);

        engine
            .insert_str(ctx, "")
            .expect("test fixture: empty insert_str on existing context");
        assert_eq!(engine.get_draft(ctx), Some(String::new()));
        assert!(!engine.has_draft(ctx));
    }

    #[test]
    fn test_insert_str_unicode() {
        let engine = DynamicContextualCompletionEngine::new();
        let ctx = engine.create_root_context(0);

        engine
            .insert_str(ctx, "Hello 世界")
            .expect("test fixture: insert_str on existing context");
        assert_eq!(engine.get_draft(ctx), Some("Hello 世界".to_string()));

        engine
            .insert_str(ctx, " 🌍")
            .expect("test fixture: insert_str on existing context");
        assert_eq!(engine.get_draft(ctx), Some("Hello 世界 🌍".to_string()));
    }

    #[test]
    fn test_delete_char() {
        let engine = DynamicContextualCompletionEngine::new();
        let ctx = engine.create_root_context(0);

        engine
            .insert_str(ctx, "hello")
            .expect("test fixture: insert_str on existing context");
        assert_eq!(engine.delete_char(ctx), Some('o'));
        assert_eq!(engine.get_draft(ctx), Some("hell".to_string()));

        assert_eq!(engine.delete_char(ctx), Some('l'));
        assert_eq!(engine.get_draft(ctx), Some("hel".to_string()));
    }

    #[test]
    fn test_delete_char_empty() {
        let engine = DynamicContextualCompletionEngine::new();
        let ctx = engine.create_root_context(0);

        assert_eq!(engine.delete_char(ctx), None);
    }

    #[test]
    fn test_delete_char_nonexistent_context() {
        let engine = DynamicContextualCompletionEngine::new();
        assert_eq!(engine.delete_char(999), None);
    }

    #[test]
    fn test_clear_draft() {
        let engine = DynamicContextualCompletionEngine::new();
        let ctx = engine.create_root_context(0);

        engine
            .insert_str(ctx, "hello")
            .expect("test fixture: insert_str on existing context");
        assert!(engine.has_draft(ctx));

        engine
            .clear_draft(ctx)
            .expect("test fixture: clear_draft on existing context");
        assert!(!engine.has_draft(ctx));
        assert_eq!(engine.get_draft(ctx), Some(String::new()));
    }

    #[test]
    fn test_clear_draft_nonexistent_context() {
        let engine = DynamicContextualCompletionEngine::new();
        let result = engine.clear_draft(999);
        assert!(result.is_err());
    }

    #[test]
    fn test_checkpoint_and_undo() {
        let engine = DynamicContextualCompletionEngine::new();
        let ctx = engine.create_root_context(0);

        // Create initial checkpoint (empty)
        engine
            .checkpoint(ctx)
            .expect("test fixture: checkpoint on existing context");
        assert_eq!(engine.checkpoint_count(ctx), 1);

        // Type "hello"
        engine
            .insert_str(ctx, "hello")
            .expect("test fixture: insert_str on existing context");
        engine
            .checkpoint(ctx)
            .expect("test fixture: checkpoint on existing context");
        assert_eq!(engine.checkpoint_count(ctx), 2);

        // Type " world"
        engine
            .insert_str(ctx, " world")
            .expect("test fixture: insert_str on existing context");
        assert_eq!(engine.get_draft(ctx), Some("hello world".to_string()));

        // Undo to "hello" (restore to stack top, then pop)
        engine
            .undo(ctx)
            .expect("test fixture: undo with available checkpoint");
        assert_eq!(engine.get_draft(ctx), Some("hello".to_string()));
        assert_eq!(engine.checkpoint_count(ctx), 1); // empty still on stack

        // Undo to empty
        engine
            .undo(ctx)
            .expect("test fixture: undo with available checkpoint");
        assert_eq!(engine.get_draft(ctx), Some(String::new()));
        assert_eq!(engine.checkpoint_count(ctx), 0); // stack now empty
    }

    #[test]
    fn test_undo_no_checkpoints() {
        let engine = DynamicContextualCompletionEngine::new();
        let ctx = engine.create_root_context(0);

        let result = engine.undo(ctx);
        assert!(result.is_err());
    }

    #[test]
    fn test_checkpoint_nonexistent_context() {
        let engine = DynamicContextualCompletionEngine::new();
        let result = engine.checkpoint(999);
        assert!(result.is_err());
    }

    #[test]
    fn test_undo_nonexistent_context() {
        let engine = DynamicContextualCompletionEngine::new();
        let result = engine.undo(999);
        assert!(result.is_err());
    }

    #[test]
    fn test_checkpoint_count() {
        let engine = DynamicContextualCompletionEngine::new();
        let ctx = engine.create_root_context(0);

        assert_eq!(engine.checkpoint_count(ctx), 0);

        engine
            .checkpoint(ctx)
            .expect("test fixture: checkpoint on existing context");
        assert_eq!(engine.checkpoint_count(ctx), 1);

        engine
            .checkpoint(ctx)
            .expect("test fixture: checkpoint on existing context");
        assert_eq!(engine.checkpoint_count(ctx), 2);

        engine
            .checkpoint(ctx)
            .expect("test fixture: checkpoint on existing context");
        assert_eq!(engine.checkpoint_count(ctx), 3);
    }

    #[test]
    fn test_checkpoint_count_nonexistent() {
        let engine = DynamicContextualCompletionEngine::new();
        assert_eq!(engine.checkpoint_count(999), 0);
    }

    #[test]
    fn test_clear_checkpoints() {
        let engine = DynamicContextualCompletionEngine::new();
        let ctx = engine.create_root_context(0);

        engine
            .checkpoint(ctx)
            .expect("test fixture: checkpoint on existing context");
        engine
            .checkpoint(ctx)
            .expect("test fixture: checkpoint on existing context");
        assert_eq!(engine.checkpoint_count(ctx), 2);

        engine
            .clear_checkpoints(ctx)
            .expect("test fixture: clear_checkpoints on existing context");
        assert_eq!(engine.checkpoint_count(ctx), 0);
    }

    #[test]
    fn test_clear_checkpoints_nonexistent_context() {
        let engine = DynamicContextualCompletionEngine::new();
        let result = engine.clear_checkpoints(999);
        assert!(result.is_err());
    }

    #[test]
    fn test_multiple_undo_steps() {
        let engine = DynamicContextualCompletionEngine::new();
        let ctx = engine.create_root_context(0);

        // Build up checkpoints
        engine
            .checkpoint(ctx)
            .expect("test fixture: checkpoint on existing context"); // ""
        engine
            .insert_char(ctx, 'a')
            .expect("test fixture: insert_char on existing context");
        engine
            .checkpoint(ctx)
            .expect("test fixture: checkpoint on existing context"); // "a"
        engine
            .insert_char(ctx, 'b')
            .expect("test fixture: insert_char on existing context");
        engine
            .checkpoint(ctx)
            .expect("test fixture: checkpoint on existing context"); // "ab"
        engine
            .insert_char(ctx, 'c')
            .expect("test fixture: insert_char on existing context");
        // Don't checkpoint after 'c', so undo will go back to "ab"

        assert_eq!(engine.get_draft(ctx), Some("abc".to_string()));
        assert_eq!(engine.checkpoint_count(ctx), 3);

        // Undo step by step - restores to stack top, then pops
        engine
            .undo(ctx)
            .expect("test fixture: undo with available checkpoint"); // Restore to "ab", pop
        assert_eq!(engine.get_draft(ctx), Some("ab".to_string()));
        assert_eq!(engine.checkpoint_count(ctx), 2);

        engine
            .undo(ctx)
            .expect("test fixture: undo with available checkpoint"); // Restore to "a", pop
        assert_eq!(engine.get_draft(ctx), Some("a".to_string()));
        assert_eq!(engine.checkpoint_count(ctx), 1);

        engine
            .undo(ctx)
            .expect("test fixture: undo with available checkpoint"); // Restore to empty, pop
        assert_eq!(engine.get_draft(ctx), Some(String::new()));
        assert_eq!(engine.checkpoint_count(ctx), 0);
    }

    #[test]
    fn test_finalize() {
        let engine = DynamicContextualCompletionEngine::new();
        let ctx = engine.create_root_context(0);

        engine
            .insert_str(ctx, "hello")
            .expect("test fixture: insert_str on existing context");
        engine
            .checkpoint(ctx)
            .expect("test fixture: checkpoint on existing context");

        let term = engine
            .finalize(ctx)
            .expect("test fixture: finalize with non-empty draft");
        assert_eq!(term, "hello");

        // Draft cleared
        assert!(!engine.has_draft(ctx));

        // Checkpoints cleared
        assert_eq!(engine.checkpoint_count(ctx), 0);

        // Term in dictionary
        assert!(engine.has_term("hello"));
        assert_eq!(engine.term_contexts("hello"), vec![ctx]);
    }

    #[test]
    fn test_finalize_empty_draft() {
        let engine = DynamicContextualCompletionEngine::new();
        let ctx = engine.create_root_context(0);

        let result = engine.finalize(ctx);
        assert!(result.is_err());
    }

    #[test]
    fn test_finalize_nonexistent_context() {
        let engine = DynamicContextualCompletionEngine::new();
        let result = engine.finalize(999);
        assert!(result.is_err());
    }

    #[test]
    fn test_finalize_direct() {
        let engine = DynamicContextualCompletionEngine::new();
        let ctx = engine.create_root_context(0);

        engine
            .finalize_direct(ctx, "function")
            .expect("test fixture: finalize_direct with non-empty term");
        engine
            .finalize_direct(ctx, "variable")
            .expect("test fixture: finalize_direct with non-empty term");

        assert!(engine.has_term("function"));
        assert!(engine.has_term("variable"));
        assert_eq!(engine.term_contexts("function"), vec![ctx]);
    }

    #[test]
    fn test_finalize_direct_empty_term() {
        let engine = DynamicContextualCompletionEngine::new();
        let ctx = engine.create_root_context(0);

        let result = engine.finalize_direct(ctx, "");
        assert!(result.is_err());
    }

    #[test]
    fn test_finalize_direct_nonexistent_context() {
        let engine = DynamicContextualCompletionEngine::new();
        let result = engine.finalize_direct(999, "test");
        assert!(result.is_err());
    }

    #[test]
    fn test_discard() {
        let engine = DynamicContextualCompletionEngine::new();
        let ctx = engine.create_root_context(0);

        engine
            .insert_str(ctx, "mistake")
            .expect("test fixture: insert_str on existing context");
        engine
            .checkpoint(ctx)
            .expect("test fixture: checkpoint on existing context");
        assert!(engine.has_draft(ctx));
        assert_eq!(engine.checkpoint_count(ctx), 1);

        engine
            .discard(ctx)
            .expect("test fixture: discard on existing context");

        // Draft and checkpoints cleared
        assert!(!engine.has_draft(ctx));
        assert_eq!(engine.checkpoint_count(ctx), 0);

        // Not in dictionary
        assert!(!engine.has_term("mistake"));
    }

    #[test]
    fn test_discard_nonexistent_context() {
        let engine = DynamicContextualCompletionEngine::new();
        let result = engine.discard(999);
        assert!(result.is_err());
    }

    #[test]
    fn test_has_term() {
        let engine = DynamicContextualCompletionEngine::new();
        let ctx = engine.create_root_context(0);

        assert!(!engine.has_term("test"));

        engine
            .finalize_direct(ctx, "test")
            .expect("test fixture: finalize_direct with non-empty term");
        assert!(engine.has_term("test"));
    }

    #[test]
    fn test_term_contexts() {
        let engine = DynamicContextualCompletionEngine::new();
        let global = engine.create_root_context(0);
        let func = engine
            .create_child_context(1, global)
            .expect("test fixture: create_child_context with valid parent");

        engine
            .finalize_direct(global, "global_var")
            .expect("test fixture: finalize_direct with non-empty term");
        engine
            .finalize_direct(func, "local_var")
            .expect("test fixture: finalize_direct with non-empty term");

        // Same term in multiple contexts
        engine
            .finalize_direct(func, "shared")
            .expect("test fixture: finalize_direct with non-empty term");
        engine
            .finalize_direct(global, "shared")
            .expect("test fixture: finalize_direct with non-empty term");

        assert_eq!(engine.term_contexts("global_var"), vec![global]);
        assert_eq!(engine.term_contexts("local_var"), vec![func]);
        assert_eq!(engine.term_contexts("shared"), vec![func, global]);
        assert!(engine.term_contexts("unknown").is_empty());
    }

    #[test]
    fn test_term_contexts_unknown() {
        let engine = DynamicContextualCompletionEngine::new();
        assert!(engine.term_contexts("unknown").is_empty());
    }

    #[test]
    fn test_complete_drafts() {
        let engine = DynamicContextualCompletionEngine::new();
        let ctx = engine.create_root_context(0);

        engine
            .insert_str(ctx, "hello")
            .expect("test fixture: insert_str on existing context");

        let results = engine.complete_drafts(ctx, "hel", 2);
        assert_eq!(results.len(), 1);
        assert_eq!(results[0].term, "hello");
        assert!(results[0].is_draft);
        assert_eq!(results[0].distance, 2); // "hel" -> "hello" = 2 insertions
    }

    #[test]
    fn test_complete_finalized() {
        let engine = DynamicContextualCompletionEngine::new();
        let ctx = engine.create_root_context(0);

        engine
            .finalize_direct(ctx, "hello")
            .expect("test fixture: finalize_direct with non-empty term");
        engine
            .finalize_direct(ctx, "help")
            .expect("test fixture: finalize_direct with non-empty term");

        let results = engine.complete_finalized(ctx, "hel", 2);
        assert_eq!(results.len(), 2);
        assert!(results.iter().any(|c| c.term == "hello"));
        assert!(results.iter().any(|c| c.term == "help"));
        assert!(results.iter().all(|c| !c.is_draft));
    }

    #[test]
    fn test_complete_fusion() {
        let engine = DynamicContextualCompletionEngine::new();
        let ctx = engine.create_root_context(0);

        // Finalized terms
        engine
            .finalize_direct(ctx, "hello")
            .expect("test fixture: finalize_direct with non-empty term");
        engine
            .finalize_direct(ctx, "help")
            .expect("test fixture: finalize_direct with non-empty term");

        // Draft
        engine
            .insert_str(ctx, "hero")
            .expect("test fixture: insert_str on existing context");

        let results = engine.complete(ctx, "hel", 2);

        // Should have hello, help, hero
        assert!(results.len() >= 3);
        assert!(results.iter().any(|c| c.term == "hello"));
        assert!(results.iter().any(|c| c.term == "help"));
        assert!(results.iter().any(|c| c.term == "hero"));
    }

    #[test]
    fn test_complete_deduplication() {
        let engine = DynamicContextualCompletionEngine::new();
        let ctx = engine.create_root_context(0);

        // Finalized term
        engine
            .finalize_direct(ctx, "test")
            .expect("test fixture: finalize_direct with non-empty term");

        // Draft with same term (should override)
        engine
            .insert_str(ctx, "test")
            .expect("test fixture: insert_str on existing context");

        let results = engine.complete(ctx, "test", 0);

        // Should only have one "test" (draft version)
        let test_results: Vec<_> = results.iter().filter(|c| c.term == "test").collect();
        assert_eq!(test_results.len(), 1);
        assert!(test_results[0].is_draft);
    }

    #[test]
    fn test_complete_hierarchical_visibility() {
        let engine = DynamicContextualCompletionEngine::new();
        let global = engine.create_root_context(0);
        let func = engine
            .create_child_context(1, global)
            .expect("test fixture: create_child_context with valid parent");

        // Global term
        engine
            .finalize_direct(global, "global_var")
            .expect("test fixture: finalize_direct with non-empty term");

        // Local term
        engine
            .finalize_direct(func, "local_var")
            .expect("test fixture: finalize_direct with non-empty term");

        // Query from func - should see both
        let results = engine.complete_finalized(func, "var", 10);
        assert!(results.iter().any(|c| c.term == "global_var"));
        assert!(results.iter().any(|c| c.term == "local_var"));

        // Query from global - should NOT see local_var
        let results = engine.complete_finalized(global, "var", 10);
        assert!(results.iter().any(|c| c.term == "global_var"));
        assert!(!results.iter().any(|c| c.term == "local_var"));
    }

    #[test]
    fn test_complete_sorting() {
        let engine = DynamicContextualCompletionEngine::new();
        let ctx = engine.create_root_context(0);

        engine
            .finalize_direct(ctx, "test")
            .expect("test fixture: finalize_direct with non-empty term"); // distance 0
        engine
            .finalize_direct(ctx, "text")
            .expect("test fixture: finalize_direct with non-empty term"); // distance 1
        engine
            .finalize_direct(ctx, "best")
            .expect("test fixture: finalize_direct with non-empty term"); // distance 1

        let mut results = engine.complete_finalized(ctx, "test", 1);
        results.sort();

        // Should be sorted by distance, then term
        assert!(results.len() >= 2);
        assert!(results[0].distance <= results[1].distance);
        if results.len() >= 2 && results[0].distance == results[1].distance {
            assert!(results[0].term <= results[1].term);
        }
    }

    #[test]
    fn test_complete_empty_query() {
        let engine = DynamicContextualCompletionEngine::new();
        let ctx = engine.create_root_context(0);

        engine
            .finalize_direct(ctx, "test")
            .expect("test fixture: finalize_direct with non-empty term");

        // Empty query should match with distance = term length
        let results = engine.complete_finalized(ctx, "", 10);
        assert!(!results.is_empty());
    }

    #[test]
    fn test_complete_no_matches() {
        let engine = DynamicContextualCompletionEngine::new();
        let ctx = engine.create_root_context(0);

        engine
            .finalize_direct(ctx, "hello")
            .expect("test fixture: finalize_direct with non-empty term");

        // Query too far away
        let results = engine.complete_finalized(ctx, "xyz", 1);
        assert!(results.is_empty());
    }

    #[test]
    fn test_levenshtein_distance() {
        type Engine = DynamicContextualCompletionEngine<PathMapDictionary<Vec<ContextId>>>;

        assert_eq!(Engine::levenshtein_distance("", ""), 0);
        assert_eq!(Engine::levenshtein_distance("abc", ""), 3);
        assert_eq!(Engine::levenshtein_distance("", "abc"), 3);
        assert_eq!(Engine::levenshtein_distance("abc", "abc"), 0);
        assert_eq!(Engine::levenshtein_distance("abc", "abd"), 1);
        assert_eq!(Engine::levenshtein_distance("abc", "abcd"), 1);
        assert_eq!(Engine::levenshtein_distance("kitten", "sitting"), 3);
    }
}
