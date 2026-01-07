//! Re-exports serialization from libdictenstein for backwards compatibility.
//!
//! This module re-exports all serialization functionality from the libdictenstein crate.
//! All dictionary serialization has been moved to libdictenstein to improve modularity.
//!
//! # Migration
//!
//! Replace imports from `liblevenshtein::serialization` with `libdictenstein::serialization`:
//!
//! ```rust,ignore
//! // Old (deprecated):
//! use liblevenshtein::serialization::{BincodeSerializer, DictionarySerializer};
//!
//! // New (recommended):
//! use libdictenstein::serialization::{BincodeSerializer, DictionarySerializer};
//! ```
//!
//! # Example
//!
//! ```rust,ignore
//! use libdictenstein::prelude::*;
//! use libdictenstein::serialization::{BincodeSerializer, DictionarySerializer};
//! use std::fs::File;
//!
//! // Create and populate dictionary
//! let dict = DoubleArrayTrie::from_terms(vec!["test", "testing"]);
//!
//! // Serialize to file
//! let file = File::create("dict.bin")?;
//! BincodeSerializer::serialize(&dict, file)?;
//!
//! // Deserialize from file
//! let file = File::open("dict.bin")?;
//! let loaded_dict: DoubleArrayTrie = BincodeSerializer::deserialize(file)?;
//! ```

// Re-export everything from libdictenstein::serialization
#[cfg(feature = "serialization")]
pub use libdictenstein::serialization::*;
