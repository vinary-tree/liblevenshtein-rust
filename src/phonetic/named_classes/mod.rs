//! Built-in named character classes for phonetic grammars.
//!
//! This module provides standard POSIX classes and phonetic-specific classes
//! for use in LLev grammar files and regex patterns.
//!
//! # Class Categories
//!
//! - **POSIX classes**: `[:alpha:]`, `[:digit:]`, `[:space:]`, etc.
//! - **Phonetic classes**: `[:vowel:]`, `[:consonant:]`, `[:fricative:]`, etc.
//!
//! # Syntax
//!
//! - Standalone: `[:vowel:]` - matches any vowel
//! - Mixed: `[[:vowel:]y]` - matches vowels plus 'y'
//!
//! # Examples
//!
//! ```text
//! # LLev grammar rules
//! c -> s / _[:front_vowel:];    # c -> s before front vowel
//! [:voiced:] -> [:voiceless:];  # devoicing rule
//! [:nasal:] -> m / _[:stop:];   # nasal assimilation
//! ```

mod algebra;
mod lookup;
mod phone_pattern;
mod registry;

#[cfg(test)]
mod tests;

pub use algebra::{get_all_phonetic_chars, intersect_char_sets, negate_char_set};
pub use lookup::{
    all_builtin_class_names, get_chars_only, get_digraphs_only, get_named_class,
    get_sequences_only, get_tetragraphs_only, get_trigraphs_only, is_builtin_class,
};
pub use phone_pattern::PhonePattern;
pub use registry::{NamedClass, NAMED_CLASSES};
