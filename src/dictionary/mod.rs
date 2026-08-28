//! Dictionary abstractions for pluggable backends.
//!
//! **DEPRECATION NOTICE**: This module re-exports types from the [`libdictenstein`] crate.
//! Direct imports from `liblevenshtein::dictionary` are deprecated and will be removed
//! in the next major version. Please update your imports to use `libdictenstein` directly.
//!
//! # Migration Guide
//!
//! Replace imports like:
//! ```rust,ignore
//! use liblevenshtein::dictionary::{Dictionary, DictionaryNode, DoubleArrayTrie};
//! ```
//!
//! With:
//! ```rust
//! use libdictenstein::{Dictionary, DictionaryNode};
//! use libdictenstein::double_array_trie::DoubleArrayTrie;
//! ```
//!
//! Or use the prelude:
//! ```rust
//! use libdictenstein::prelude::*;
//! ```
//!
//! # Choosing a Dictionary Backend
//!
//! The library provides multiple dictionary backends optimized for different use cases.
//! See the [`libdictenstein`] crate documentation for the full comparison table.
//!
//! ## Quick Start
//!
//! - **General use**: [`DoubleArrayTrie`](libdictenstein::double_array_trie::DoubleArrayTrie)
//! - **Unicode text**: [`DoubleArrayTrieChar`](libdictenstein::double_array_trie::char::DoubleArrayTrieChar)
//! - **Insert + Remove**: [`DynamicDawg`](libdictenstein::dynamic_dawg::DynamicDawg)
//! - **Substring search**: [`SuffixAutomaton`](libdictenstein::suffix_automaton::SuffixAutomaton)

pub(crate) mod node_adapter;

// Keep phonetic_normalized locally since it depends on liblevenshtein's phonetic NFAs
#[cfg(feature = "phonetic-rules")]
pub mod phonetic_normalized;

// ============================================================================
// DEPRECATED RE-EXPORTS FROM LIBDICTENSTEIN
//
// All dictionary functionality has been moved to the libdictenstein crate.
// These re-exports are provided for backwards compatibility and will be
// removed in the next major version.
// ============================================================================

// Re-export submodules with deprecation warnings
#[deprecated(
    since = "0.9.0",
    note = "Use libdictenstein::char_unit directly. This re-export will be removed in version 1.0."
)]
pub use libdictenstein::char_unit;

#[deprecated(
    since = "0.9.0",
    note = "Use libdictenstein::double_array_trie directly. This re-export will be removed in version 1.0."
)]
pub use libdictenstein::double_array_trie;

#[deprecated(
    since = "0.9.0",
    note = "Use libdictenstein::double_array_trie::char directly. This re-export will be removed in version 1.0."
)]
pub use libdictenstein::double_array_trie::char as double_array_trie_char;

#[deprecated(
    since = "0.9.0",
    note = "Use libdictenstein::double_array_trie::char_zipper directly. This re-export will be removed in version 1.0."
)]
pub use libdictenstein::double_array_trie::char_zipper as double_array_trie_char_zipper;

#[deprecated(
    since = "0.9.0",
    note = "Use libdictenstein::double_array_trie::zipper directly. This re-export will be removed in version 1.0."
)]
pub use libdictenstein::double_array_trie::zipper as double_array_trie_zipper;

#[deprecated(
    since = "0.9.0",
    note = "Use libdictenstein::dynamic_dawg directly. This re-export will be removed in version 1.0."
)]
pub use libdictenstein::dynamic_dawg;

#[deprecated(
    since = "0.9.0",
    note = "Use libdictenstein::dynamic_dawg::char directly. This re-export will be removed in version 1.0."
)]
pub use libdictenstein::dynamic_dawg::char as dynamic_dawg_char;

#[deprecated(
    since = "0.9.0",
    note = "Use libdictenstein::dynamic_dawg::char_zipper directly. This re-export will be removed in version 1.0."
)]
pub use libdictenstein::dynamic_dawg::char_zipper as dynamic_dawg_char_zipper;

#[deprecated(
    since = "0.9.0",
    note = "Use libdictenstein::dynamic_dawg::u64 directly. This re-export will be removed in version 1.0."
)]
pub use libdictenstein::dynamic_dawg::u64 as dynamic_dawg_u64;

#[deprecated(
    since = "0.9.0",
    note = "Use libdictenstein::dynamic_dawg::u64_zipper directly. This re-export will be removed in version 1.0."
)]
pub use libdictenstein::dynamic_dawg::u64_zipper as dynamic_dawg_u64_zipper;

#[deprecated(
    since = "0.9.0",
    note = "Use libdictenstein::dynamic_dawg::zipper directly. This re-export will be removed in version 1.0."
)]
pub use libdictenstein::dynamic_dawg::zipper as dynamic_dawg_zipper;

#[deprecated(
    since = "0.9.0",
    note = "Use libdictenstein::factory directly. This re-export will be removed in version 1.0."
)]
pub use libdictenstein::factory;

#[deprecated(
    since = "0.9.0",
    note = "Use libdictenstein::iterator directly. This re-export will be removed in version 1.0."
)]
pub use libdictenstein::iterator;

#[cfg(feature = "pathmap-backend")]
#[deprecated(
    since = "0.9.0",
    note = "Use libdictenstein::pathmap directly. This re-export will be removed in version 1.0."
)]
pub use libdictenstein::pathmap;

#[cfg(feature = "pathmap-backend")]
#[deprecated(
    since = "0.9.0",
    note = "Use libdictenstein::pathmap::char directly. This re-export will be removed in version 1.0."
)]
pub use libdictenstein::pathmap::char as pathmap_char;

#[cfg(feature = "pathmap-backend")]
#[deprecated(
    since = "0.9.0",
    note = "Use libdictenstein::pathmap::zipper directly. This re-export will be removed in version 1.0."
)]
pub use libdictenstein::pathmap::zipper as pathmap_zipper;

#[cfg(feature = "persistent-artrie")]
#[deprecated(
    since = "0.9.0",
    note = "Use libdictenstein::persistent_artrie directly. This re-export will be removed in version 1.0."
)]
pub use libdictenstein::persistent_artrie;

#[cfg(feature = "persistent-artrie")]
#[deprecated(
    since = "0.9.0",
    note = "Use libdictenstein::persistent_artrie::char directly. This re-export will be removed in version 1.0."
)]
pub use libdictenstein::persistent_artrie::char as persistent_artrie_char;

#[deprecated(
    since = "0.9.0",
    note = "Use libdictenstein::prefix_zipper directly. This re-export will be removed in version 1.0."
)]
pub use libdictenstein::prefix_zipper;

#[deprecated(
    since = "0.9.0",
    note = "Use libdictenstein::scdawg directly. This re-export will be removed in version 1.0."
)]
pub use libdictenstein::scdawg;

#[deprecated(
    since = "0.9.0",
    note = "Use libdictenstein::scdawg::char directly. This re-export will be removed in version 1.0."
)]
pub use libdictenstein::scdawg::char as scdawg_char;

#[deprecated(
    since = "0.9.0",
    note = "Use libdictenstein::substring directly. This re-export will be removed in version 1.0."
)]
pub use libdictenstein::substring;

#[deprecated(
    since = "0.9.0",
    note = "Use libdictenstein::suffix_automaton directly. This re-export will be removed in version 1.0."
)]
pub use libdictenstein::suffix_automaton;

#[deprecated(
    since = "0.9.0",
    note = "Use libdictenstein::suffix_automaton::char directly. This re-export will be removed in version 1.0."
)]
pub use libdictenstein::suffix_automaton::char as suffix_automaton_char;

#[deprecated(
    since = "0.9.0",
    note = "Use libdictenstein::suffix_automaton::char_zipper directly. This re-export will be removed in version 1.0."
)]
pub use libdictenstein::suffix_automaton::char_zipper as suffix_automaton_char_zipper;

#[deprecated(
    since = "0.9.0",
    note = "Use libdictenstein::suffix_automaton::zipper directly. This re-export will be removed in version 1.0."
)]
pub use libdictenstein::suffix_automaton::zipper as suffix_automaton_zipper;

#[deprecated(
    since = "0.9.0",
    note = "Use libdictenstein::value directly. This re-export will be removed in version 1.0."
)]
pub use libdictenstein::value;

#[deprecated(
    since = "0.9.0",
    note = "Use libdictenstein::zipper directly. This re-export will be removed in version 1.0."
)]
pub use libdictenstein::zipper;

// Re-export commonly used types at module root for convenience
// These are the types most users import directly

#[deprecated(
    since = "0.9.0",
    note = "Use libdictenstein::CharUnit directly. This re-export will be removed in version 1.0."
)]
pub use libdictenstein::CharUnit;

#[deprecated(
    since = "0.9.0",
    note = "Use libdictenstein::DictionaryIterator directly. This re-export will be removed in version 1.0."
)]
pub use libdictenstein::DictionaryIterator;

#[deprecated(
    since = "0.9.0",
    note = "Use libdictenstein::DictionaryTermIterator directly. This re-export will be removed in version 1.0."
)]
pub use libdictenstein::DictionaryTermIterator;

#[cfg(feature = "persistent-artrie")]
#[deprecated(
    since = "0.9.0",
    note = "Use libdictenstein::PersistentARTrie directly. This re-export will be removed in version 1.0."
)]
pub use libdictenstein::PersistentARTrie;

#[cfg(feature = "persistent-artrie")]
#[deprecated(
    since = "0.9.0",
    note = "Use libdictenstein::PersistentARTrieZipper directly. This re-export will be removed in version 1.0."
)]
pub use libdictenstein::PersistentARTrieZipper;

#[cfg(feature = "persistent-artrie")]
#[deprecated(
    since = "0.9.0",
    note = "Use libdictenstein::PersistentARTrieChar directly. This re-export will be removed in version 1.0."
)]
pub use libdictenstein::PersistentARTrieChar;

#[cfg(feature = "persistent-artrie")]
#[deprecated(
    since = "0.9.0",
    note = "Use libdictenstein::PersistentARTrieCharNode directly. This re-export will be removed in version 1.0."
)]
pub use libdictenstein::PersistentARTrieCharNode;

#[cfg(feature = "persistent-artrie")]
#[deprecated(
    since = "0.9.0",
    note = "Use libdictenstein::PersistentARTrieCharZipper directly. This re-export will be removed in version 1.0."
)]
pub use libdictenstein::PersistentARTrieCharZipper;

#[deprecated(
    since = "0.9.0",
    note = "Use libdictenstein::BidirectionalDictionaryNode directly. This re-export will be removed in version 1.0."
)]
pub use libdictenstein::BidirectionalDictionaryNode;

#[deprecated(
    since = "0.9.0",
    note = "Use libdictenstein::ExtensionResult directly. This re-export will be removed in version 1.0."
)]
pub use libdictenstein::ExtensionResult;

#[deprecated(
    since = "0.9.0",
    note = "Use libdictenstein::SubstringDictionary directly. This re-export will be removed in version 1.0."
)]
pub use libdictenstein::SubstringDictionary;

#[deprecated(
    since = "0.9.0",
    note = "Use libdictenstein::SubstringMatch directly. This re-export will be removed in version 1.0."
)]
pub use libdictenstein::SubstringMatch;

#[deprecated(
    since = "0.9.0",
    note = "Use libdictenstein::DictionaryValue directly. This re-export will be removed in version 1.0."
)]
pub use libdictenstein::DictionaryValue;

#[deprecated(
    since = "0.9.0",
    note = "Use libdictenstein::DictZipper directly. This re-export will be removed in version 1.0."
)]
pub use libdictenstein::DictZipper;

#[deprecated(
    since = "0.9.0",
    note = "Use libdictenstein::ValuedDictZipper directly. This re-export will be removed in version 1.0."
)]
pub use libdictenstein::ValuedDictZipper;

// Core traits - these are the most important re-exports

#[deprecated(
    since = "0.9.0",
    note = "Use libdictenstein::SyncStrategy directly. This re-export will be removed in version 1.0."
)]
pub use libdictenstein::SyncStrategy;

#[deprecated(
    since = "0.9.0",
    note = "Use libdictenstein::Dictionary directly. This re-export will be removed in version 1.0."
)]
pub use libdictenstein::Dictionary;

#[deprecated(
    since = "0.9.0",
    note = "Use libdictenstein::DictionaryNode directly. This re-export will be removed in version 1.0."
)]
pub use libdictenstein::DictionaryNode;

#[deprecated(
    since = "0.9.0",
    note = "Use libdictenstein::MappedDictionary directly. This re-export will be removed in version 1.0."
)]
pub use libdictenstein::MappedDictionary;

#[deprecated(
    since = "0.9.0",
    note = "Use libdictenstein::MappedDictionaryNode directly. This re-export will be removed in version 1.0."
)]
pub use libdictenstein::MappedDictionaryNode;

#[deprecated(
    since = "0.9.0",
    note = "Use libdictenstein::MutableMappedDictionary directly. This re-export will be removed in version 1.0."
)]
pub use libdictenstein::MutableMappedDictionary;
