//! Command-line interface for liblevenshtein.
//!
//! Provides the `liblevenshtein` binary's command-line utilities for building, loading,
//! saving, and querying dictionaries. The interface is a thin front-end: argument parsing
//! lives in [`args`], format detection in [`detect`], paths/config in [`paths`], and the
//! actual work in [`commands`] — the same backend-agnostic primitives the [`crate::repl`]
//! reuses, so both surfaces behave identically.
//!
//! ## Submodules
//!
//! - [`args`] — the `clap` argument and subcommand model.
//! - [`commands`] — the shared load / save / query primitives (reused by the REPL).
//! - [`detect`] — serialization-format auto-detection (magic bytes → extension → content).
//! - [`paths`] — default paths, persistent configuration, and path validation.

pub mod args;
pub mod commands;
pub mod detect;
pub mod paths;

pub use args::{Cli, SerializationFormat};
pub use detect::{detect_format, DictFormat, FormatDetection};
pub use paths::{config_dir, default_dict_path};
