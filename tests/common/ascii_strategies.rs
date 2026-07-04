//! Shared ASCII word proptest strategies.

use proptest::prelude::*;

/// Generate lowercase ASCII words.
pub fn ascii_word_strategy() -> impl Strategy<Value = String> {
    "[a-z]{1,10}"
}

/// Generate small dictionaries of lowercase ASCII words.
pub fn small_dict_strategy() -> impl Strategy<Value = Vec<String>> {
    prop::collection::vec(ascii_word_strategy(), 1..=10)
}
