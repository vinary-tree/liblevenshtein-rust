//! CLI argument definitions using flag-based conventions

use crate::repl::state::DictionaryBackend;
use crate::transducer::Algorithm;
use clap::{ArgGroup, Parser, ValueEnum};
use std::path::PathBuf;

/// Command-line interface for liblevenshtein
///
/// Uses traditional Unix flag-based conventions with mutually exclusive operation flags.
#[derive(Parser)]
#[command(name = "liblevenshtein")]
#[command(about = "Fast fuzzy string matching with Levenshtein automata")]
#[command(version)]
#[command(group(
    ArgGroup::new("operation")
        .required(true)
        .args(["compile", "phonetic", "query", "repl", "info", "convert",
               "insert", "delete", "clear", "minimize", "settings", "config_mgmt",
               "compile_regex", "match_regex", "grep"])
))]
pub struct Cli {
    // ========================================================================
    // Global options
    // ========================================================================

    /// Custom configuration file path
    #[arg(short = 'c', long = "config")]
    pub config_file: Option<PathBuf>,

    // ========================================================================
    // Operation flags (mutually exclusive)
    // ========================================================================

    /// Compile phonetic rules from .llev to binary format
    #[arg(short = 'C', long)]
    pub compile: bool,

    /// Apply phonetic rules to transform text
    #[arg(short = 'P', long)]
    pub phonetic: bool,

    /// Query dictionary for fuzzy matches
    #[arg(short = 'Q', long)]
    pub query: bool,

    /// Launch interactive REPL
    #[arg(short = 'R', long)]
    pub repl: bool,

    /// Display dictionary information
    #[arg(short = 'I', long)]
    pub info: bool,

    /// Convert dictionary between formats
    #[arg(long)]
    pub convert: bool,

    /// Insert terms into dictionary
    #[arg(long)]
    pub insert: bool,

    /// Delete terms from dictionary
    #[arg(short = 'D', long)]
    pub delete: bool,

    /// Clear all terms from dictionary
    #[arg(long)]
    pub clear: bool,

    /// Minimize/compact dictionary (for DynamicDawg)
    #[arg(short = 'M', long)]
    pub minimize: bool,

    /// Show or update user settings
    #[arg(short = 'S', long)]
    pub settings: bool,

    /// Manage config file location
    #[arg(long)]
    pub config_mgmt: bool,

    /// Compile a .llre regex file to binary format
    #[arg(long = "compile-regex")]
    pub compile_regex: bool,

    /// Match text against a .llre regex pattern
    #[arg(long = "match-regex")]
    pub match_regex: bool,

    /// Search files for fuzzy phonetic matches (grep-style)
    #[arg(short = 'G', long)]
    pub grep: bool,

    // ========================================================================
    // Input/output arguments (shared across operations)
    // ========================================================================

    /// Input file (for --compile, --convert)
    #[arg(long)]
    pub input: Option<PathBuf>,

    /// Output file (for --compile, --convert)
    #[arg(short = 'o', long)]
    pub output: Option<PathBuf>,

    /// Dictionary file
    #[arg(short = 'd', long)]
    pub dict: Option<PathBuf>,

    /// Rules file (.llev source or .llev.bin compiled) for --phonetic
    #[arg(short = 'r', long)]
    pub rules: Option<PathBuf>,

    /// Text to transform (for --phonetic) or query term (for --query)
    #[arg(long)]
    pub text: Option<String>,

    /// Terms to insert or delete (positional arguments)
    #[arg(value_name = "TERMS")]
    pub terms: Vec<String>,

    // ========================================================================
    // Compile-specific options
    // ========================================================================

    /// Use character-level (Unicode) rules (default: true)
    #[arg(long, default_value = "true")]
    pub unicode: bool,

    /// Verify compilation by loading result
    #[arg(long)]
    pub verify: bool,

    /// Use compiled binary format (auto-detected from extension)
    #[arg(long)]
    pub compiled: bool,

    // ========================================================================
    // Regex options (for --compile-regex, --match-regex)
    // ========================================================================

    /// Enable multiline mode for regex (^ and $ match line boundaries)
    #[arg(long)]
    pub multiline: bool,

    /// Enable dotall mode for regex (. matches newlines)
    #[arg(long)]
    pub dotall: bool,

    // ========================================================================
    // Grep options (for --grep)
    // ========================================================================

    /// Pattern to search for (regex or literal) in grep mode
    #[arg(long)]
    pub pattern: Option<String>,

    /// Files to search (for --grep)
    #[arg(long)]
    pub files: Vec<PathBuf>,

    /// Print filename for each match
    #[arg(long = "with-filename")]
    pub with_filename: bool,

    /// Suppress filename output
    #[arg(long = "no-filename")]
    pub no_filename: bool,

    /// Print line number for each match
    #[arg(short = 'n', long = "line-number")]
    pub line_number: bool,

    /// Print column number for each match
    #[arg(long = "column")]
    pub column: bool,

    /// Count matches only
    #[arg(long = "count")]
    pub count: bool,

    /// Disable colored output
    #[arg(long = "no-color")]
    pub no_color: bool,

    /// Case insensitive matching
    #[arg(short = 'i', long = "ignore-case")]
    pub ignore_case: bool,

    /// Only show matching part of line
    #[arg(short = 'O', long = "only-matching")]
    pub only_matching: bool,

    /// Disable automatic decompression (treat compressed files as plain text)
    #[arg(long = "no-decompress")]
    pub no_decompress: bool,

    /// Filter pattern for archive entries (glob syntax, e.g. "*.log", "src/**/*.rs")
    #[arg(short = 'A', long = "archive-filter")]
    pub archive_filter: Option<String>,

    /// Maximum file size to process in bytes (default: 100MB)
    #[arg(long = "max-file-size")]
    pub max_file_size: Option<u64>,

    /// Include hidden files (starting with .)
    #[arg(long = "hidden")]
    pub include_hidden: bool,

    /// Extract text from document files (PDF, DOCX, XLSX, EPUB, ODT)
    #[arg(long = "extract-documents")]
    pub extract_documents: bool,

    /// Enable OCR for image-based PDFs (requires Tesseract)
    #[arg(long = "ocr")]
    pub enable_ocr: bool,

    /// OCR language code (default: eng)
    #[arg(long = "ocr-lang", default_value = "eng")]
    pub ocr_language: String,

    // ========================================================================
    // Dictionary options
    // ========================================================================

    /// Dictionary backend (auto-detected if not specified)
    #[arg(short = 'b', long)]
    pub backend: Option<DictionaryBackend>,

    /// Serialization format (auto-detected if not specified)
    #[arg(short = 'f', long)]
    pub format: Option<SerializationFormat>,

    /// Levenshtein algorithm
    #[arg(short = 'a', long, default_value = "standard")]
    pub algorithm: Algorithm,

    /// Maximum edit distance
    #[arg(short = 'm', long, default_value = "2")]
    pub max_distance: usize,

    /// Enable prefix matching mode
    #[arg(short = 'p', long)]
    pub prefix: bool,

    /// Show distances in query results
    #[arg(short = 's', long)]
    pub show_distances: bool,

    /// Result limit
    #[arg(short = 'l', long)]
    pub limit: Option<usize>,

    /// Enable auto-sync (save after every modification)
    #[arg(long)]
    pub auto_sync: bool,

    // ========================================================================
    // Convert-specific options
    // ========================================================================

    /// Input backend (auto-detected if not specified)
    #[arg(long)]
    pub from_backend: Option<DictionaryBackend>,

    /// Output backend
    #[arg(long, default_value = "path-map")]
    pub to_backend: DictionaryBackend,

    /// Input format (auto-detected if not specified)
    #[arg(long)]
    pub from_format: Option<SerializationFormat>,

    /// Output format
    #[arg(long, default_value = "bincode")]
    pub to_format: SerializationFormat,

    // ========================================================================
    // Settings-specific options
    // ========================================================================

    /// Set default dictionary path
    #[arg(long)]
    pub set_dict: Option<PathBuf>,

    /// Set default backend
    #[arg(long)]
    pub set_backend: Option<DictionaryBackend>,

    /// Set default format
    #[arg(long)]
    pub set_format: Option<SerializationFormat>,

    /// Set default algorithm
    #[arg(long)]
    pub set_algorithm: Option<Algorithm>,

    /// Set default max distance
    #[arg(long)]
    pub set_max_distance: Option<usize>,

    /// Reset configuration to defaults
    #[arg(long)]
    pub reset: bool,

    // ========================================================================
    // Config management options
    // ========================================================================

    /// Switch to a different config file (updates app config)
    #[arg(long)]
    pub switch: Option<PathBuf>,

    /// Show current config file path
    #[arg(long)]
    pub show: bool,
}

/// Serialization format for dictionary storage
#[derive(Debug, Clone, Copy, PartialEq, Eq, ValueEnum, serde::Serialize, serde::Deserialize)]
pub enum SerializationFormat {
    /// Plain text (one term per line)
    Text,
    /// Bincode binary format
    Bincode,
    /// JSON format
    Json,
    /// Protocol Buffers format
    #[cfg(feature = "protobuf")]
    Protobuf,
    /// Gzip-compressed Bincode format
    #[cfg(feature = "compression")]
    #[value(name = "bincode-gz")]
    BincodeGzip,
    /// Gzip-compressed JSON format
    #[cfg(feature = "compression")]
    #[value(name = "json-gz")]
    JsonGzip,
    /// Gzip-compressed Protocol Buffers format
    #[cfg(all(feature = "protobuf", feature = "compression"))]
    #[value(name = "protobuf-gz")]
    ProtobufGzip,
    /// PathMap's native .paths compressed format (PathMap backend only)
    #[value(name = "paths")]
    PathsNative,
}

impl std::fmt::Display for SerializationFormat {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            Self::Text => write!(f, "text"),
            Self::Bincode => write!(f, "bincode"),
            Self::Json => write!(f, "json"),
            #[cfg(feature = "protobuf")]
            Self::Protobuf => write!(f, "protobuf"),
            #[cfg(feature = "compression")]
            Self::BincodeGzip => write!(f, "bincode-gz"),
            #[cfg(feature = "compression")]
            Self::JsonGzip => write!(f, "json-gz"),
            #[cfg(all(feature = "protobuf", feature = "compression"))]
            Self::ProtobufGzip => write!(f, "protobuf-gz"),
            Self::PathsNative => write!(f, "paths"),
        }
    }
}
