//! CLI interface for liblevenshtein
//!
//! Provides command-line utilities for dictionary operations and queries.

pub mod args;
pub mod commands;
pub mod detect;
pub mod paths;

pub use args::{Cli, SerializationFormat};
pub use detect::{detect_format, DictFormat, FormatDetection};
pub use paths::{config_dir, default_dict_path};
