//! Core command definitions and types shared between CLI and REPL

use crate::transducer::Algorithm;
use std::path::PathBuf;

#[cfg(feature = "cli")]
use crate::cli::args::SerializationFormat;
#[cfg(feature = "cli")]
use crate::repl::state::DictionaryBackend;

/// Query parameters used by both CLI and REPL
#[derive(Debug, Clone)]
pub struct QueryParams {
    /// The term to search for
    pub term: String,
    /// Maximum edit distance
    pub max_distance: usize,
    /// Levenshtein algorithm to use
    pub algorithm: Algorithm,
    /// Enable prefix matching mode
    pub prefix: bool,
    /// Show distances in results
    pub show_distances: bool,
    /// Limit number of results
    pub limit: Option<usize>,
}

/// Dictionary modification operations
#[derive(Debug, Clone)]
pub enum ModifyOp {
    /// Insert terms into the dictionary
    Insert {
        /// Terms to insert
        terms: Vec<String>,
    },
    /// Delete terms from the dictionary
    Delete {
        /// Terms to delete
        terms: Vec<String>,
    },
    /// Clear all terms from the dictionary
    Clear,
}

/// Dictionary I/O operations
#[derive(Debug, Clone)]
pub enum IoOp {
    /// Load dictionary from file
    Load {
        /// Path to dictionary file
        path: PathBuf,
    },
    /// Save dictionary to file
    Save {
        /// Path to save to
        path: PathBuf,
    },
    /// Display dictionary information
    Info {
        /// Optional path (use current dict if None)
        path: Option<PathBuf>,
    },
}

/// Result of command execution
#[derive(Debug)]
pub struct CommandResult {
    /// Output message to display
    pub output: String,
    /// Whether the dictionary was modified
    pub modified: bool,
    /// Whether to exit (for REPL)
    pub should_exit: bool,
}

impl CommandResult {
    /// Create a successful result with output
    pub fn success(output: impl Into<String>) -> Self {
        Self {
            output: output.into(),
            modified: false,
            should_exit: false,
        }
    }

    /// Create a result indicating modification
    pub fn modified(output: impl Into<String>) -> Self {
        Self {
            output: output.into(),
            modified: true,
            should_exit: false,
        }
    }

    /// Create a result that signals exit
    pub fn exit(output: impl Into<String>) -> Self {
        Self {
            output: output.into(),
            modified: false,
            should_exit: true,
        }
    }
}

// ============================================================================
// I/O Operation Types (requires CLI feature for format/backend types)
// ============================================================================

/// Parameters for serializing (saving) a dictionary
#[cfg(feature = "cli")]
#[derive(Debug, Clone)]
pub struct SerializeParams {
    /// Path to save the dictionary to
    pub path: PathBuf,
    /// Serialization format to use
    pub format: SerializationFormat,
    /// Whether to overwrite existing files
    pub overwrite: bool,
}

/// Parameters for deserializing (loading) a dictionary
#[cfg(feature = "cli")]
#[derive(Debug, Clone)]
pub struct DeserializeParams {
    /// Path to load the dictionary from
    pub path: PathBuf,
    /// Optional backend hint (auto-detected if None)
    pub backend: Option<DictionaryBackend>,
    /// Optional format hint (auto-detected if None)
    pub format: Option<SerializationFormat>,
}

/// Result of a serialization operation
#[cfg(feature = "cli")]
#[derive(Debug, Clone)]
pub struct SerializeResult {
    /// Number of terms serialized
    pub term_count: usize,
    /// Size of the serialized data in bytes
    pub byte_size: u64,
    /// Format used for serialization
    pub format: SerializationFormat,
}

/// Result of a deserialization operation
#[cfg(feature = "cli")]
#[derive(Debug, Clone)]
pub struct DeserializeResult {
    /// Number of terms loaded
    pub term_count: usize,
    /// Backend used for the dictionary
    pub backend: DictionaryBackend,
    /// Format detected/used for deserialization
    pub format: SerializationFormat,
}

/// Information about a dictionary file
#[cfg(feature = "cli")]
#[derive(Debug, Clone)]
pub struct DictInfo {
    /// File path
    pub path: PathBuf,
    /// Number of terms in the dictionary
    pub term_count: usize,
    /// Backend type
    pub backend: DictionaryBackend,
    /// Serialization format
    pub format: SerializationFormat,
    /// File size in bytes
    pub file_size: u64,
}
