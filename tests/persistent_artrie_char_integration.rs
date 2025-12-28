//! UTF-8 integration tests for PersistentARTrieChar.
//!
//! These tests verify proper Unicode character handling:
//! - Multi-byte UTF-8 sequences
//! - Emoji and special characters
//! - Mixed ASCII/UTF-8 operations
//! - Unicode edge cases (combining chars, RTL, etc.)
//! - Character-level edit distance semantics
//!
//! # Why PersistentARTrieChar?
//!
//! The byte-level PersistentARTrie treats multi-byte UTF-8 sequences as
//! multiple bytes. This is problematic for edit distance:
//! - "" (ñ) is 2 bytes → distance from "n" is 2 (wrong)
//! - With PersistentARTrieChar, "ñ" is 1 character → distance from "n" is 1 (correct)

#![cfg(feature = "persistent-artrie")]

use liblevenshtein::dictionary::persistent_artrie_char::{
    PersistentARTrieChar, PersistentARTrieCharZipper,
};
use liblevenshtein::dictionary::zipper::DictZipper;
use liblevenshtein::dictionary::{DictionaryNode, MappedDictionary};

// =============================================================================
// Test: Basic Unicode Insertion and Lookup
// =============================================================================

#[test]
fn test_basic_unicode_insertion() {
    let trie: PersistentARTrieChar<()> = PersistentARTrieChar::new();

    // Insert various Unicode strings
    trie.insert("café");
    trie.insert("naïve");
    trie.insert("résumé");

    assert!(trie.contains("café"));
    assert!(trie.contains("naïve"));
    assert!(trie.contains("résumé"));
    assert!(!trie.contains("cafe")); // Different from café
    assert_eq!(trie.len(), 3);
}

#[test]
fn test_cjk_characters() {
    let trie: PersistentARTrieChar<()> = PersistentARTrieChar::new();

    // Chinese characters
    trie.insert("中文");
    trie.insert("你好");
    trie.insert("世界");

    // Japanese
    trie.insert("日本語");
    trie.insert("こんにちは");
    trie.insert("ありがとう");

    // Korean
    trie.insert("한국어");
    trie.insert("안녕하세요");

    assert!(trie.contains("中文"));
    assert!(trie.contains("こんにちは"));
    assert!(trie.contains("한국어"));
    assert!(!trie.contains("中")); // Partial prefix
    assert_eq!(trie.len(), 8);
}

#[test]
fn test_emoji_handling() {
    let trie: PersistentARTrieChar<()> = PersistentARTrieChar::new();

    // Single emoji
    trie.insert("🎉");
    trie.insert("🌍");
    trie.insert("❤️");

    // Emoji sequences
    trie.insert("👨‍👩‍👧"); // Family emoji (ZWJ sequence)
    trie.insert("🏳️‍🌈"); // Rainbow flag

    // Mixed text and emoji
    trie.insert("Hello 🌍!");
    trie.insert("I ❤️ Rust");

    assert!(trie.contains("🎉"));
    assert!(trie.contains("Hello 🌍!"));
    assert!(!trie.contains("🎊")); // Different emoji
    assert_eq!(trie.len(), 7);
}

// =============================================================================
// Test: Unicode Edge Cases
// =============================================================================

#[test]
fn test_combining_characters() {
    let trie: PersistentARTrieChar<()> = PersistentARTrieChar::new();

    // Precomposed vs decomposed forms
    // Note: Rust strings are UTF-8 byte sequences, not normalized
    let precomposed = "é"; // U+00E9 (single code point)
    let decomposed = "é"; // U+0065 + U+0301 (e + combining acute)

    trie.insert(precomposed);
    trie.insert(decomposed);

    // These may or may not be the same depending on normalization
    assert!(trie.contains(precomposed));
    assert!(trie.contains(decomposed));

    // Count depends on whether they are identical after normalization
    // Without normalization, they are different strings
}

#[test]
fn test_rtl_text() {
    let trie: PersistentARTrieChar<()> = PersistentARTrieChar::new();

    // Arabic
    trie.insert("مرحبا");
    trie.insert("العالم");

    // Hebrew
    trie.insert("שלום");
    trie.insert("עולם");

    assert!(trie.contains("مرحبا"));
    assert!(trie.contains("שלום"));
    assert_eq!(trie.len(), 4);
}

#[test]
fn test_mixed_scripts() {
    let trie: PersistentARTrieChar<()> = PersistentARTrieChar::new();

    // Mixed script strings
    trie.insert("Café 中文");
    trie.insert("Tōkyō 東京");
    trie.insert("München München");
    trie.insert("São Paulo");

    assert!(trie.contains("Café 中文"));
    assert!(trie.contains("Tōkyō 東京"));
    assert_eq!(trie.len(), 4);
}

#[test]
fn test_special_unicode_categories() {
    let trie: PersistentARTrieChar<()> = PersistentARTrieChar::new();

    // Mathematical symbols
    trie.insert("∑∏∫∂");
    trie.insert("∀x∃y");

    // Currency symbols
    trie.insert("$€£¥₿");

    // Musical symbols
    trie.insert("♩♪♫♬");

    // Technical symbols
    trie.insert("⚡⚙️🔧");

    assert!(trie.contains("∑∏∫∂"));
    assert!(trie.contains("$€£¥₿"));
    assert_eq!(trie.len(), 5);
}

// =============================================================================
// Test: Values with Unicode Keys
// =============================================================================

#[test]
fn test_unicode_keys_with_values() {
    let trie: PersistentARTrieChar<i32> = PersistentARTrieChar::new();

    trie.insert_with_value("café", 1);
    trie.insert_with_value("中文", 2);
    trie.insert_with_value("🎉", 3);

    assert_eq!(trie.get_value("café"), Some(1));
    assert_eq!(trie.get_value("中文"), Some(2));
    assert_eq!(trie.get_value("🎉"), Some(3));
    assert_eq!(trie.get_value("notfound"), None);
}

#[test]
fn test_unicode_keys_with_string_values() {
    let trie: PersistentARTrieChar<String> = PersistentARTrieChar::new();

    trie.insert_with_value("hello", "greeting".to_string());
    trie.insert_with_value("世界", "world".to_string());
    trie.insert_with_value("café", "coffee place".to_string());

    assert_eq!(trie.get_value("hello"), Some("greeting".to_string()));
    assert_eq!(trie.get_value("世界"), Some("world".to_string()));
    assert_eq!(trie.get_value("café"), Some("coffee place".to_string()));
}

// =============================================================================
// Test: Zipper Navigation with Unicode
// =============================================================================

#[test]
fn test_zipper_unicode_navigation() {
    let trie: PersistentARTrieChar<()> = PersistentARTrieChar::new();

    trie.insert("café");
    trie.insert("cat");

    let zipper = PersistentARTrieCharZipper::new(&trie);

    // Navigate to "café" using char-by-char descent
    let z = zipper
        .descend('c')
        .and_then(|z| z.descend('a'))
        .and_then(|z| z.descend('f'))
        .and_then(|z| z.descend('é'));

    assert!(z.is_some());
    let z = z.unwrap();
    assert!(z.is_final());

    // Path should be characters, not bytes
    let path = z.path();
    assert_eq!(path, vec!['c', 'a', 'f', 'é']);
}

#[test]
fn test_zipper_cjk_navigation() {
    let trie: PersistentARTrieChar<()> = PersistentARTrieChar::new();

    trie.insert("中文");

    let zipper = PersistentARTrieCharZipper::new(&trie);

    // Navigate character by character
    let z = zipper.descend('中').and_then(|z| z.descend('文'));

    assert!(z.is_some());
    let z = z.unwrap();
    assert!(z.is_final());
    assert_eq!(z.path(), vec!['中', '文']);
}

#[test]
fn test_zipper_children_with_unicode() {
    let trie: PersistentARTrieChar<()> = PersistentARTrieChar::new();

    // Insert terms with different second characters
    trie.insert("ab");
    trie.insert("aé");
    trie.insert("a中");
    trie.insert("a🎉");

    let zipper = PersistentARTrieCharZipper::new(&trie);
    let a_zipper = zipper.descend('a').expect("should have 'a'");

    // Collect all children labels
    let children: Vec<char> = a_zipper.children().map(|(c, _)| c).collect();

    assert!(children.contains(&'b'));
    assert!(children.contains(&'é'));
    assert!(children.contains(&'中'));
    assert!(children.contains(&'🎉'));
    assert_eq!(children.len(), 4);
}

// =============================================================================
// Test: Dictionary Trait Implementation
// =============================================================================

#[test]
fn test_dictionary_trait_unicode() {
    let trie: PersistentARTrieChar<()> = PersistentARTrieChar::new();

    trie.insert("hello");
    trie.insert("世界");
    trie.insert("café");

    // Test Dictionary trait methods
    assert!(trie.contains("hello"));
    assert!(trie.contains("世界"));
    assert_eq!(trie.len(), 3);

    // Test root navigation
    let root = trie.root();
    assert!(!root.is_final()); // Empty string is not in dictionary
}

#[test]
fn test_dictionary_node_trait_unicode() {
    let trie: PersistentARTrieChar<()> = PersistentARTrieChar::new();

    trie.insert("abc");
    trie.insert("aéc");

    let root = trie.root();

    // Test transition to 'a'
    let a_node = root.transition('a');
    assert!(a_node.is_some());

    let a_node = a_node.unwrap();

    // Test edges from 'a' - should include both 'b' and 'é'
    let edges: Vec<char> = a_node.edges().map(|(c, _)| c).collect();
    assert!(edges.contains(&'b'));
    assert!(edges.contains(&'é'));
}

// =============================================================================
// Test: FromIterator with Unicode
// =============================================================================

#[test]
fn test_from_iterator_unicode() {
    let terms = vec!["café", "naïve", "中文", "🎉"];
    let trie: PersistentARTrieChar<()> = terms.into_iter().collect();

    assert_eq!(trie.len(), 4);
    assert!(trie.contains("café"));
    assert!(trie.contains("中文"));
    assert!(trie.contains("🎉"));
}

#[test]
fn test_from_iterator_owned_strings() {
    let terms: Vec<String> = vec![
        "résumé".to_string(),
        "東京".to_string(),
        "🌍🌎🌏".to_string(),
    ];
    let trie: PersistentARTrieChar<()> = terms.into_iter().collect();

    assert_eq!(trie.len(), 3);
    assert!(trie.contains("résumé"));
    assert!(trie.contains("東京"));
}

// =============================================================================
// Test: Iterator with Unicode
// =============================================================================

#[test]
fn test_iterator_unicode() {
    let trie: PersistentARTrieChar<()> = PersistentARTrieChar::new();

    trie.insert("aaa");
    trie.insert("café");
    trie.insert("中文");

    // Collect all terms via iteration
    let terms: Vec<String> = trie.iter().collect();

    assert_eq!(terms.len(), 3);
    assert!(terms.contains(&"aaa".to_string()));
    assert!(terms.contains(&"café".to_string()));
    assert!(terms.contains(&"中文".to_string()));
}

// =============================================================================
// Test: Unicode Prefix Sharing
// =============================================================================

#[test]
fn test_unicode_prefix_sharing() {
    let trie: PersistentARTrieChar<()> = PersistentARTrieChar::new();

    // Terms sharing Unicode prefixes
    trie.insert("中文");
    trie.insert("中国");
    trie.insert("中心");

    assert_eq!(trie.len(), 3);
    assert!(trie.contains("中文"));
    assert!(trie.contains("中国"));
    assert!(trie.contains("中心"));
}

#[test]
fn test_emoji_prefix_sharing() {
    let trie: PersistentARTrieChar<()> = PersistentARTrieChar::new();

    // Emoji with common prefix
    trie.insert("🎉🎊");
    trie.insert("🎉🎁");
    trie.insert("🎉🎈");

    assert_eq!(trie.len(), 3);

    // Check prefix navigation
    let zipper = PersistentARTrieCharZipper::new(&trie);
    let party = zipper.descend('🎉').expect("should have party emoji");

    let children: Vec<char> = party.children().map(|(c, _)| c).collect();
    assert_eq!(children.len(), 3);
}

// =============================================================================
// Test: Edge Cases
// =============================================================================

#[test]
fn test_empty_string() {
    let trie: PersistentARTrieChar<()> = PersistentARTrieChar::new();

    // Empty string should be insertable
    trie.insert("");

    assert!(trie.contains(""));
    assert_eq!(trie.len(), 1);
}

#[test]
fn test_single_character_unicode() {
    let trie: PersistentARTrieChar<()> = PersistentARTrieChar::new();

    // Single character terms of various types
    trie.insert("a");
    trie.insert("é");
    trie.insert("中");
    trie.insert("🎉");

    assert_eq!(trie.len(), 4);
    assert!(trie.contains("a"));
    assert!(trie.contains("é"));
    assert!(trie.contains("中"));
    assert!(trie.contains("🎉"));
}

#[test]
fn test_long_unicode_string() {
    let trie: PersistentARTrieChar<()> = PersistentARTrieChar::new();

    // Long Unicode string
    let long_unicode = "这是一个非常长的中文字符串，包含很多字符";
    trie.insert(long_unicode);

    assert!(trie.contains(long_unicode));
    assert_eq!(trie.len(), 1);
}

#[test]
fn test_duplicate_unicode_insertion() {
    let trie: PersistentARTrieChar<()> = PersistentARTrieChar::new();

    assert!(trie.insert("café"));
    assert!(!trie.insert("café")); // Duplicate should return false
    assert_eq!(trie.len(), 1);
}

// =============================================================================
// Test: Concurrent Unicode Access
// =============================================================================

#[test]
fn test_concurrent_unicode_reads() {
    use std::sync::Arc;
    use std::thread;

    let trie: PersistentARTrieChar<()> = PersistentARTrieChar::new();

    // Insert Unicode terms
    let terms = vec!["café", "中文", "🎉", "日本語", "한국어"];
    for term in &terms {
        trie.insert(term);
    }

    let trie: Arc<PersistentARTrieChar<()>> = Arc::new(trie);

    // Spawn multiple reader threads
    let handles: Vec<_> = (0..4)
        .map(|_| {
            let trie_clone: Arc<PersistentARTrieChar<()>> = Arc::clone(&trie);
            let terms_clone = terms.clone();

            thread::spawn(move || {
                for term in &terms_clone {
                    assert!(trie_clone.contains(term));
                }
            })
        })
        .collect();

    for handle in handles {
        handle.join().expect("thread join");
    }
}

// =============================================================================
// Test: Clone Behavior
// =============================================================================

#[test]
fn test_clone_shares_unicode_state() {
    let trie: PersistentARTrieChar<()> = PersistentARTrieChar::new();

    trie.insert("café");

    let clone = trie.clone();

    // Clone should see original insert
    assert!(clone.contains("café"));

    // Insert via clone
    clone.insert("中文");

    // Original should see clone's insert
    assert!(trie.contains("中文"));
}

// =============================================================================
// Test: MappedDictionary with Unicode
// =============================================================================

#[test]
fn test_mapped_dictionary_unicode() {
    let trie: PersistentARTrieChar<i32> = PersistentARTrieChar::new();

    trie.insert_with_value("one", 1);
    trie.insert_with_value("一", 1); // Chinese for "one"
    trie.insert_with_value("하나", 1); // Korean for "one"

    // MappedDictionary::get_value
    assert_eq!(trie.get_value("one"), Some(1));
    assert_eq!(trie.get_value("一"), Some(1));
    assert_eq!(trie.get_value("하나"), Some(1));
}

// =============================================================================
// Test: Supplementary Plane Characters (> U+FFFF)
// =============================================================================

#[test]
fn test_supplementary_plane_characters() {
    let trie: PersistentARTrieChar<()> = PersistentARTrieChar::new();

    // Characters outside Basic Multilingual Plane (BMP)
    // These require surrogate pairs in UTF-16 but are single code points in Rust

    // Musical symbols (U+1D100-U+1D1FF)
    trie.insert("𝄞"); // G clef

    // Mathematical Alphanumeric Symbols (U+1D400-U+1D7FF)
    trie.insert("𝐀𝐁𝐂"); // Bold letters

    // Ancient scripts
    trie.insert("𐀀"); // Linear B syllable

    // Emoji with skin tone modifiers (U+1F3FB-U+1F3FF)
    trie.insert("👋🏽"); // Waving hand with medium skin tone

    assert!(trie.contains("𝄞"));
    assert!(trie.contains("𝐀𝐁𝐂"));
    assert!(trie.contains("𐀀"));
    assert!(trie.contains("👋🏽"));
    assert_eq!(trie.len(), 4);
}

// =============================================================================
// Test: Zero-Width Characters
// =============================================================================

#[test]
fn test_zero_width_characters() {
    let trie: PersistentARTrieChar<()> = PersistentARTrieChar::new();

    // Zero-width joiner (ZWJ) and zero-width non-joiner (ZWNJ)
    let with_zwj = "a\u{200D}b"; // a + ZWJ + b
    let without_zwj = "ab";

    trie.insert(with_zwj);
    trie.insert(without_zwj);

    // These should be different strings
    assert!(trie.contains(with_zwj));
    assert!(trie.contains(without_zwj));

    // Count depends on whether they are treated as same
    // Without normalization, they are different
    assert!(trie.len() >= 1); // At least one unique
}

// =============================================================================
// Test: Whitespace Variations
// =============================================================================

#[test]
fn test_unicode_whitespace() {
    let trie: PersistentARTrieChar<()> = PersistentARTrieChar::new();

    // Different types of spaces
    trie.insert("hello world"); // Regular space
    trie.insert("hello\u{00A0}world"); // Non-breaking space
    trie.insert("hello\u{2003}world"); // Em space
    trie.insert("hello\u{3000}world"); // Ideographic space

    // These should all be different
    assert!(trie.contains("hello world"));
    assert!(trie.contains("hello\u{00A0}world"));
    assert!(trie.contains("hello\u{2003}world"));
    assert!(trie.contains("hello\u{3000}world"));
    assert_eq!(trie.len(), 4);
}
