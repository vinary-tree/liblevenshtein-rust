//! Transducer-specific proptest strategies.

use crate::ascii_strategies::ascii_word_strategy;
use proptest::prelude::*;

/// Generate medium dictionaries of lowercase ASCII words.
pub fn medium_dict_strategy() -> impl Strategy<Value = Vec<String>> {
    prop::collection::vec(ascii_word_strategy(), 10..=50)
}

/// Generate Unicode words including Greek letters.
pub fn unicode_word_strategy() -> impl Strategy<Value = String> {
    "[a-zA-Z0-9αβγδεζηθικλμνξοπρστυφχψω]{1,8}"
}

/// Generate dictionaries with Unicode words.
pub fn unicode_dict_strategy() -> impl Strategy<Value = Vec<String>> {
    prop::collection::vec(unicode_word_strategy(), 1..=10)
}
