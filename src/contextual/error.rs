//! Error types for contextual completion operations.

use thiserror::Error;

/// Errors that can occur during contextual completion operations.
#[derive(Debug, Error, Clone, PartialEq, Eq)]
pub enum ContextError {
    /// The specified context ID is already in use.
    ///
    /// Context IDs are stable keys into the context tree, draft buffers, and
    /// checkpoint stacks. Reusing an ID would overwrite existing state or
    /// reparent an existing scope, so creation APIs reject duplicates.
    #[error("Context {0} already exists")]
    ContextAlreadyExists(crate::contextual::ContextId),

    /// The specified context does not exist.
    ///
    /// This error occurs when trying to access a context that hasn't been created
    /// or has been removed.
    #[error("Context {0} does not exist")]
    ContextNotFound(crate::contextual::ContextId),

    /// The context does not have a draft buffer.
    ///
    /// This error occurs when trying to access draft operations on a context
    /// that doesn't have an initialized draft buffer.
    #[error("Context {0} does not have a draft buffer")]
    NoDraftBuffer(crate::contextual::ContextId),

    /// The context does not have a checkpoint stack.
    ///
    /// This error occurs when trying to access checkpoint operations on a context
    /// that doesn't have an initialized checkpoint stack.
    #[error("Context {0} does not have a checkpoint stack")]
    NoCheckpointStack(crate::contextual::ContextId),

    /// No checkpoints available to restore.
    ///
    /// This error occurs when calling `undo()` on a context that has no
    /// checkpoints saved.
    #[error("No checkpoints to restore to for context {0}")]
    NoCheckpoints(crate::contextual::ContextId),

    /// The draft is empty.
    ///
    /// This error occurs when trying to finalize an empty draft.
    #[error("Draft is empty for context {0}")]
    EmptyDraft(crate::contextual::ContextId),

    /// The term is empty.
    ///
    /// This error occurs when trying to finalize or insert an empty term.
    #[error("Term is empty")]
    EmptyTerm,

    /// A circular context hierarchy was detected.
    ///
    /// This error occurs when trying to create a parent-child relationship
    /// that would create a cycle in the context tree.
    #[error("Circular context hierarchy detected: {0} -> {1}")]
    CircularHierarchy(crate::contextual::ContextId, crate::contextual::ContextId),
}

/// A specialized `Result` type for contextual completion operations.
pub type Result<T> = std::result::Result<T, ContextError>;
