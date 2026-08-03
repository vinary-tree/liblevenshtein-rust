//! Command-line interface for liblevenshtein.
//!
//! Provides the `liblevenshtein` binary's command-line utilities for building, loading,
//! saving, and querying dictionaries. The interface is a thin front-end: argument parsing
//! lives in [`args`](crate::cli::args), format detection in
//! [`detect`](crate::cli::detect), paths/config in [`paths`](crate::cli::paths), and the
//! actual work in [`commands`](crate::cli::commands) — the same backend-agnostic primitives the [`crate::repl`]
//! reuses, so both surfaces behave identically.
//!
//! ## Submodules
//!
//! - [`args`](crate::cli::args) — the `clap` argument and subcommand model.
//! - [`commands`](crate::cli::commands) — the shared load / save / query primitives (reused by the REPL).
//! - [`detect`](crate::cli::detect) — serialization-format auto-detection (magic bytes → extension → content).
//! - [`paths`](crate::cli::paths) — default paths, persistent configuration, and path validation.

pub mod args;
pub mod commands;
pub mod detect;
pub mod paths;

pub use args::{Cli, SerializationFormat};
pub use detect::{detect_format, DictFormat, FormatDetection};
pub use paths::{config_dir, default_dict_path};
