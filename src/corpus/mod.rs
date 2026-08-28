//! Test corpus utilities for validation and benchmarking.
//!
//! This module provides parsers and generators for standard spelling correction
//! test corpora, including:
//!
//! - **Norvig's big.txt**: Natural language word frequency distribution
//! - **Birkbeck corpus**: Real-world spelling errors from native speakers
//! - **Holbrook corpus**: Secondary school spelling errors
//! - **Aspell corpus**: Technical and domain-specific term errors
//! - **Wikipedia corpus**: Common web-scale misspellings
//!
//! ## Usage
//!
//! This module is only available in test and benchmark builds:
//!
//! ```rust,no_run
//! use liblevenshtein::corpus::{BigTxtCorpus, MittonCorpus};
//!
//! # fn main() -> std::io::Result<()> {
//! // Load Norvig's big.txt for dictionary construction
//! let big_txt = BigTxtCorpus::load("data/corpora/big.txt")?;
//! let unique_word_count = big_txt.unique_words(); // ~32K words
//! assert!(unique_word_count > 0);
//!
//! // Load Holbrook corpus for error validation
//! let holbrook = MittonCorpus::load("data/corpora/holbrook.dat")?;
//! for (correct, misspellings) in &holbrook.errors {
//!     for (misspelling, frequency) in misspellings {
//!         // Validate spell correction...
//!     }
//! }
//! # Ok(())
//! # }
//! ```
//!
//! ## Corpus Formats
//!
//! ### Norvig's big.txt
//!
//! Plain text, one token per line (preserving frequency):
//!
//! ```text
//! the
//! the
//! the
//! ...
//! ```
//!
//! ### Mitton `.dat` Format
//!
//! Used by Holbrook, Aspell, and Wikipedia corpora:
//!
//! ```text
//! $correct_word
//! misspelling1 frequency1
//! misspelling2 frequency2
//! ...
//! ```
//!
//! Each correct word is preceded by `$` and followed by its misspellings
//! with optional frequency counts (default: 1).

pub mod generator;
pub mod parser;

pub use generator::{QueryWorkload, TypoGenerator};
pub use parser::{BigTxtCorpus, MittonCorpus};
