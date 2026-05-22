//! On-the-fly phonetic grep with substring matching.
//!
//! This module provides streaming phonetic pattern matching without
//! requiring dictionary preprocessing. It uses the product automaton
//! (NFA × Levenshtein) for fuzzy regex matching.
//!
//! # No Preprocessing Required
//!
//! Unlike the transducer approach which preprocesses a dictionary,
//! `PhoneticGrep` evaluates each candidate independently using the
//! product automaton's stateless `min_distance()` method.
//!
//! # Examples
//!
//! ```ignore
//! use liblevenshtein::phonetic::grep::PhoneticGrep;
//!
//! // Search for fuzzy matches
//! let grep = PhoneticGrep::from_pattern("phone", 1)?;
//! for line_match in grep.grep_file(file_content) {
//!     for m in &line_match.matches {
//!         println!("{}:{}: {} (distance {})",
//!             line_match.line_number,
//!             m.start_column,
//!             m.matched_text,
//!             m.distance);
//!     }
//! }
//! ```

use std::path::Path;

use crate::phonetic::nfa::{compile_with_flags, NFAChar, ProductAutomatonChar};
use crate::phonetic::regex::ast::UnicodeNormalization;
use crate::phonetic::regex::transform::normalize_input;
use crate::phonetic::regex::{parse, ParseError};
use crate::transducer::Algorithm;

/// Error type for grep operations.
#[derive(Debug)]
pub enum GrepError {
    /// Pattern parsing failed
    Parse(ParseError),
    /// NFA compilation failed
    Compile(String),
    /// Rule loading failed
    RuleLoad(String),
    /// IO error
    Io(std::io::Error),
}

impl std::fmt::Display for GrepError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            GrepError::Parse(e) => write!(f, "pattern parse error: {}", e),
            GrepError::Compile(e) => write!(f, "NFA compile error: {}", e),
            GrepError::RuleLoad(e) => write!(f, "rule load error: {}", e),
            GrepError::Io(e) => write!(f, "IO error: {}", e),
        }
    }
}

impl std::error::Error for GrepError {}

impl From<ParseError> for GrepError {
    fn from(e: ParseError) -> Self {
        GrepError::Parse(e)
    }
}

impl From<std::io::Error> for GrepError {
    fn from(e: std::io::Error) -> Self {
        GrepError::Io(e)
    }
}

/// A match found within a line.
#[derive(Debug, Clone)]
pub struct GrepMatch {
    /// The matched text (substring)
    pub matched_text: String,
    /// Start column (1-indexed, in bytes)
    pub start_column: usize,
    /// End column (1-indexed, in bytes)
    pub end_column: usize,
    /// Edit distance from pattern
    pub distance: u8,
}

/// A line with all its matches.
#[derive(Debug)]
pub struct LineMatch {
    /// Line number (1-indexed)
    pub line_number: usize,
    /// The full line text
    pub line: String,
    /// All matches within this line
    pub matches: Vec<GrepMatch>,
}

/// Phonetic grep matcher with substring support.
///
/// Provides on-the-fly fuzzy pattern matching without dictionary preprocessing.
/// Uses the product automaton (NFA × Levenshtein) for efficient fuzzy matching.
///
/// # Regex Flags
///
/// The pattern can include inline flags that affect matching:
///
/// - `(?i:pattern)` - Case-insensitive matching (transformed at compile time)
/// - `(?a:pattern)` - Accent-insensitive matching (transformed at compile time)
/// - `(?u:NFC:pattern)` - Unicode normalization applied to input at runtime
/// - `(?ia:pattern)` - Combined case and accent insensitive
///
/// # Examples
///
/// ```ignore
/// // Case-insensitive via pattern flag
/// let grep = PhoneticGrep::from_pattern("(?i:hello)", 0)?;
/// assert!(grep.matches("HELLO").is_some());
///
/// // Accent-insensitive
/// let grep = PhoneticGrep::from_pattern("(?a:cafe)", 0)?;
/// assert!(grep.matches("café").is_some());
/// ```
pub struct PhoneticGrep {
    /// The compiled NFA pattern
    nfa: NFAChar,
    /// Maximum edit distance (from CLI or constructor)
    max_distance: u8,
    /// Local distance override from pattern (e.g., `(?;N:pattern)`)
    /// When set, this takes precedence over `max_distance`.
    local_distance: Option<u8>,
    /// Levenshtein algorithm variant
    algorithm: Algorithm,
    /// Optional phonetic rules for normalization
    rules: Option<Vec<crate::phonetic::RewriteRuleChar>>,
    /// Case-insensitive matching (applied at runtime, in addition to (?i) flag)
    case_insensitive: bool,
    /// Unicode normalization to apply to input (extracted from (?u:NFC) etc.)
    unicode_normalization: Option<UnicodeNormalization>,
}

impl PhoneticGrep {
    /// Create a grep matcher from a regex pattern.
    ///
    /// # Arguments
    ///
    /// * `pattern` - Regex pattern to match
    /// * `max_distance` - Maximum edit distance (0 = exact match)
    ///
    /// # Examples
    ///
    /// ```ignore
    /// // Basic pattern
    /// let grep = PhoneticGrep::from_pattern("phone", 1)?;
    ///
    /// // Case-insensitive via (?i) flag
    /// let grep = PhoneticGrep::from_pattern("(?i:phone)", 1)?;
    /// assert!(grep.matches("PHONE").is_some());
    ///
    /// // Accent-insensitive via (?a) flag
    /// let grep = PhoneticGrep::from_pattern("(?a:cafe)", 0)?;
    /// assert!(grep.matches("café").is_some());
    /// ```
    pub fn from_pattern(pattern: &str, max_distance: u8) -> Result<Self, GrepError> {
        let regex = parse(pattern)?;
        let result = compile_with_flags(&regex).map_err(|e| GrepError::Compile(e.to_string()))?;
        Ok(Self {
            nfa: result.nfa,
            max_distance,
            local_distance: result.local_distance,
            algorithm: Algorithm::Standard,
            rules: None,
            case_insensitive: false,
            unicode_normalization: result.unicode_normalization,
        })
    }

    /// Create a grep matcher from a regex pattern with a specific algorithm.
    ///
    /// # Arguments
    ///
    /// * `pattern` - Regex pattern to match
    /// * `max_distance` - Maximum edit distance (0 = exact match)
    /// * `algorithm` - Levenshtein algorithm variant
    ///
    /// # Examples
    ///
    /// ```ignore
    /// use liblevenshtein::transducer::Algorithm;
    /// let grep = PhoneticGrep::from_pattern_with_algorithm("phone", 1, Algorithm::Transposition)?;
    /// ```
    pub fn from_pattern_with_algorithm(
        pattern: &str,
        max_distance: u8,
        algorithm: Algorithm,
    ) -> Result<Self, GrepError> {
        let regex = parse(pattern)?;
        let result = compile_with_flags(&regex).map_err(|e| GrepError::Compile(e.to_string()))?;
        Ok(Self {
            nfa: result.nfa,
            max_distance,
            local_distance: result.local_distance,
            algorithm,
            rules: None,
            case_insensitive: false,
            unicode_normalization: result.unicode_normalization,
        })
    }

    /// Create a grep matcher with phonetic rules for normalization.
    ///
    /// # Arguments
    ///
    /// * `pattern` - Regex pattern to match
    /// * `rules_path` - Path to .llev rules file
    /// * `max_distance` - Maximum edit distance
    pub fn with_rules(
        pattern: &str,
        rules_path: &Path,
        max_distance: u8,
    ) -> Result<Self, GrepError> {
        let regex = parse(pattern)?;
        let result = compile_with_flags(&regex).map_err(|e| GrepError::Compile(e.to_string()))?;

        // Load rules
        let llev_file = crate::phonetic::llev::load_file(rules_path)
            .map_err(|e| GrepError::RuleLoad(e.to_string()))?;
        let ruleset = crate::phonetic::RuleSetChar::from_llev(&llev_file)
            .map_err(|e| GrepError::RuleLoad(e.to_string()))?;

        Ok(Self {
            nfa: result.nfa,
            max_distance,
            local_distance: result.local_distance,
            algorithm: Algorithm::Standard,
            rules: Some(ruleset.rules),
            case_insensitive: false,
            unicode_normalization: result.unicode_normalization,
        })
    }

    /// Create a grep matcher with pre-loaded rules.
    ///
    /// # Arguments
    ///
    /// * `pattern` - Regex pattern to match
    /// * `rules` - Pre-loaded phonetic rules
    /// * `max_distance` - Maximum edit distance
    pub fn with_loaded_rules(
        pattern: &str,
        rules: Vec<crate::phonetic::RewriteRuleChar>,
        max_distance: u8,
    ) -> Result<Self, GrepError> {
        let regex = parse(pattern)?;
        let result = compile_with_flags(&regex).map_err(|e| GrepError::Compile(e.to_string()))?;
        Ok(Self {
            nfa: result.nfa,
            max_distance,
            local_distance: result.local_distance,
            algorithm: Algorithm::Standard,
            rules: Some(rules),
            case_insensitive: false,
            unicode_normalization: result.unicode_normalization,
        })
    }

    /// Enable case-insensitive matching.
    ///
    /// # Examples
    ///
    /// ```ignore
    /// let grep = PhoneticGrep::from_pattern("hello", 0)?
    ///     .case_insensitive(true);
    /// assert!(grep.find_in_line("HELLO World").len() == 1);
    /// ```
    pub fn case_insensitive(mut self, yes: bool) -> Self {
        self.case_insensitive = yes;
        self
    }

    /// Set the Levenshtein algorithm variant.
    ///
    /// # Examples
    ///
    /// ```ignore
    /// use liblevenshtein::transducer::Algorithm;
    /// let grep = PhoneticGrep::from_pattern("hello", 1)?
    ///     .algorithm(Algorithm::Transposition);
    /// ```
    pub fn algorithm(mut self, algorithm: Algorithm) -> Self {
        self.algorithm = algorithm;
        self
    }

    /// Get the current algorithm variant.
    pub fn get_algorithm(&self) -> Algorithm {
        self.algorithm
    }

    /// Get the maximum edit distance (from constructor).
    pub fn max_distance(&self) -> u8 {
        self.max_distance
    }

    /// Get the local distance override from pattern (if any).
    ///
    /// When a pattern includes `(?;N:...)` syntax, this returns `Some(N)`.
    pub fn local_distance(&self) -> Option<u8> {
        self.local_distance
    }

    /// Get the effective edit distance for matching.
    ///
    /// Returns `local_distance` if set (from pattern `(?;N:...)`),
    /// otherwise returns `max_distance` (from constructor/CLI).
    pub fn effective_distance(&self) -> u8 {
        self.local_distance.unwrap_or(self.max_distance)
    }

    /// Find all matches within a line.
    ///
    /// Returns all non-overlapping matches with their positions.
    ///
    /// # Arguments
    ///
    /// * `line` - Line of text to search
    ///
    /// # Returns
    ///
    /// Vector of matches found within the line.
    pub fn find_in_line(&self, line: &str) -> Vec<GrepMatch> {
        let mut matches = Vec::new();
        let product = ProductAutomatonChar::with_algorithm(
            self.nfa.clone(),
            self.effective_distance(),
            self.algorithm,
        );

        // Scan through all words in the line
        for (start_byte, word, end_byte) in WordBoundaryIterator::new(line) {
            let candidate = if self.case_insensitive {
                word.to_lowercase()
            } else {
                word.to_string()
            };

            let normalized = self.normalize(&candidate);
            if let Some(distance) = product.min_distance(&normalized) {
                matches.push(GrepMatch {
                    matched_text: word.to_string(),
                    start_column: start_byte + 1, // 1-indexed
                    end_column: end_byte,
                    distance,
                });
            }
        }

        matches
    }

    /// Check if a single candidate matches.
    ///
    /// # Arguments
    ///
    /// * `candidate` - Word to check
    ///
    /// # Returns
    ///
    /// Edit distance if within max_distance, None otherwise.
    pub fn matches(&self, candidate: &str) -> Option<u8> {
        let candidate = if self.case_insensitive {
            candidate.to_lowercase()
        } else {
            candidate.to_string()
        };
        let normalized = self.normalize(&candidate);
        let product = ProductAutomatonChar::with_algorithm(
            self.nfa.clone(),
            self.effective_distance(),
            self.algorithm,
        );
        product.min_distance(&normalized)
    }

    /// Search a file, returning matches for each matching line.
    ///
    /// # Arguments
    ///
    /// * `content` - File content to search
    ///
    /// # Returns
    ///
    /// Iterator over lines that contain matches.
    pub fn grep_file<'a>(&'a self, content: &'a str) -> impl Iterator<Item = LineMatch> + 'a {
        content.lines().enumerate().filter_map(move |(idx, line)| {
            let matches = self.find_in_line(line);
            if matches.is_empty() {
                None
            } else {
                Some(LineMatch {
                    line_number: idx + 1,
                    line: line.to_string(),
                    matches,
                })
            }
        })
    }

    /// Search content word by word.
    ///
    /// # Arguments
    ///
    /// * `content` - Content to search
    ///
    /// # Returns
    ///
    /// Iterator over matching words with their distances.
    pub fn grep_words<'a>(&'a self, content: &'a str) -> impl Iterator<Item = (&'a str, u8)> + 'a {
        WordBoundaryIterator::new(content)
            .filter_map(move |(_, word, _)| self.matches(word).map(|d| (word, d)))
    }

    /// Normalize text using unicode normalization and phonetic rules if available.
    fn normalize(&self, text: &str) -> String {
        // First, apply unicode normalization if configured via (?u:NFC) etc.
        let text = match self.unicode_normalization {
            Some(form) => normalize_input(text, form),
            None => text.to_string(),
        };

        // Then apply phonetic rules if available
        match &self.rules {
            Some(rules) => {
                // Convert string to Vec<PhoneChar>
                let vowels = ['a', 'e', 'i', 'o', 'u', 'A', 'E', 'I', 'O', 'U'];
                let input_phones: Vec<crate::phonetic::PhoneChar> = text
                    .chars()
                    .map(|c| {
                        if vowels.contains(&c) {
                            crate::phonetic::PhoneChar::Vowel(c)
                        } else {
                            crate::phonetic::PhoneChar::Consonant(c)
                        }
                    })
                    .collect();

                // Apply rules with fuel (max 100 iterations for normalization)
                let result = crate::phonetic::apply_rules_seq_char(rules, &input_phones, 100);

                // Convert result back to string
                match result {
                    Some(phones) => {
                        let mut s = String::new();
                        for p in phones.iter() {
                            match p {
                                crate::phonetic::PhoneChar::Vowel(c)
                                | crate::phonetic::PhoneChar::Consonant(c) => s.push(*c),
                                crate::phonetic::PhoneChar::Digraph(c1, c2) => {
                                    s.push(*c1);
                                    s.push(*c2);
                                }
                                crate::phonetic::PhoneChar::Trigraph(c1, c2, c3) => {
                                    s.push(*c1);
                                    s.push(*c2);
                                    s.push(*c3);
                                }
                                crate::phonetic::PhoneChar::Tetragraph(c1, c2, c3, c4) => {
                                    s.push(*c1);
                                    s.push(*c2);
                                    s.push(*c3);
                                    s.push(*c4);
                                }
                                crate::phonetic::PhoneChar::Pentagraph(c1, c2, c3, c4, c5) => {
                                    s.push(*c1);
                                    s.push(*c2);
                                    s.push(*c3);
                                    s.push(*c4);
                                    s.push(*c5);
                                }
                                crate::phonetic::PhoneChar::Hexagraph(c1, c2, c3, c4, c5, c6) => {
                                    s.push(*c1);
                                    s.push(*c2);
                                    s.push(*c3);
                                    s.push(*c4);
                                    s.push(*c5);
                                    s.push(*c6);
                                }
                                crate::phonetic::PhoneChar::Heptagraph(
                                    c1,
                                    c2,
                                    c3,
                                    c4,
                                    c5,
                                    c6,
                                    c7,
                                ) => {
                                    s.push(*c1);
                                    s.push(*c2);
                                    s.push(*c3);
                                    s.push(*c4);
                                    s.push(*c5);
                                    s.push(*c6);
                                    s.push(*c7);
                                }
                                crate::phonetic::PhoneChar::Sequence(seq) => {
                                    for c in seq {
                                        s.push(*c);
                                    }
                                }
                                crate::phonetic::PhoneChar::Silent => {}
                            }
                        }
                        s
                    }
                    None => text,
                }
            }
            None => text,
        }
    }
}

/// Iterator over word boundaries in a string.
/// Yields (start_byte, word, end_byte) for each word.
pub struct WordBoundaryIterator<'a> {
    text: &'a str,
    pos: usize,
}

impl<'a> WordBoundaryIterator<'a> {
    /// Create a new word boundary iterator.
    pub fn new(text: &'a str) -> Self {
        Self { text, pos: 0 }
    }
}

impl<'a> Iterator for WordBoundaryIterator<'a> {
    type Item = (usize, &'a str, usize);

    fn next(&mut self) -> Option<Self::Item> {
        let bytes = self.text.as_bytes();

        // Skip non-word characters
        while self.pos < bytes.len() && !is_word_char(bytes[self.pos]) {
            self.pos += 1;
        }

        if self.pos >= bytes.len() {
            return None;
        }

        let start = self.pos;

        // Consume word characters
        while self.pos < bytes.len() && is_word_char(bytes[self.pos]) {
            self.pos += 1;
        }

        let end = self.pos;
        Some((start, &self.text[start..end], end))
    }
}

/// Check if a byte is a word character.
fn is_word_char(b: u8) -> bool {
    b.is_ascii_alphanumeric() || b == b'_' || b == b'\''
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_find_in_line_exact() {
        let grep = PhoneticGrep::from_pattern("phone", 0).expect("pattern should parse");
        let matches = grep.find_in_line("My phone is ringing");
        assert_eq!(matches.len(), 1);
        assert_eq!(matches[0].matched_text, "phone");
        assert_eq!(matches[0].start_column, 4); // 1-indexed
        assert_eq!(matches[0].distance, 0);
    }

    #[test]
    fn test_find_in_line_fuzzy() {
        let grep = PhoneticGrep::from_pattern("phone", 1).expect("pattern should parse");
        // "phon" is distance 1 from "phone" (deletion of 'e')
        // "fone" is distance 2 from "phone" (requires 2 edits: 'p'→'f' and delete 'h', or similar)
        let matches = grep.find_in_line("fone and phon are similar");
        assert_eq!(matches.len(), 1);
        assert_eq!(matches[0].matched_text, "phon");
        assert_eq!(matches[0].distance, 1); // deletion of 'e'

        // Test with distance 2 to include "fone"
        let grep2 = PhoneticGrep::from_pattern("phone", 2).expect("pattern should parse");
        let matches2 = grep2.find_in_line("fone and phon are similar");
        assert_eq!(matches2.len(), 2);
        assert_eq!(matches2[0].matched_text, "fone");
        assert_eq!(matches2[0].distance, 2); // 'ph' → 'f' requires 2 edits
        assert_eq!(matches2[1].matched_text, "phon");
        assert_eq!(matches2[1].distance, 1); // deletion of 'e'
    }

    #[test]
    fn test_grep_file_with_line_numbers() {
        let grep = PhoneticGrep::from_pattern("hello", 1).expect("pattern should parse");
        let content = "First line\nhelo world\nAnother line\nhello there";
        let results: Vec<_> = grep.grep_file(content).collect();
        assert_eq!(results.len(), 2);
        assert_eq!(results[0].line_number, 2);
        assert_eq!(results[0].matches[0].matched_text, "helo");
        assert_eq!(results[1].line_number, 4);
        assert_eq!(results[1].matches[0].matched_text, "hello");
    }

    #[test]
    fn test_case_insensitive() {
        let grep = PhoneticGrep::from_pattern("hello", 0)
            .expect("pattern should parse")
            .case_insensitive(true);
        let matches = grep.find_in_line("HELLO World");
        assert_eq!(matches.len(), 1);
        assert_eq!(matches[0].matched_text, "HELLO");
    }

    #[test]
    fn test_multiple_matches_per_line() {
        let grep = PhoneticGrep::from_pattern("test", 0).expect("pattern should parse");
        let matches = grep.find_in_line("test the test case with test data");
        assert_eq!(matches.len(), 3);
        assert_eq!(matches[0].start_column, 1);
        assert_eq!(matches[1].start_column, 10);
        assert_eq!(matches[2].start_column, 25);
    }

    #[test]
    fn test_word_boundary_iterator() {
        let text = "hello world, test!";
        let words: Vec<_> = WordBoundaryIterator::new(text).collect();
        assert_eq!(words.len(), 3);
        assert_eq!(words[0].1, "hello");
        assert_eq!(words[1].1, "world");
        assert_eq!(words[2].1, "test");
    }

    #[test]
    fn test_no_matches() {
        let grep = PhoneticGrep::from_pattern("xyz", 0).expect("pattern should parse");
        let matches = grep.find_in_line("hello world");
        assert!(matches.is_empty());
    }

    #[test]
    fn test_grep_words() {
        let grep = PhoneticGrep::from_pattern("test", 1).expect("pattern should parse");
        let content = "test best rest fest";
        let matches: Vec<_> = grep.grep_words(content).collect();
        assert_eq!(matches.len(), 4); // all within distance 1
    }

    // ========== Regex Flag Tests ==========

    #[test]
    fn test_case_insensitive_flag() {
        // Use (?i:...) flag for case-insensitive matching
        let grep = PhoneticGrep::from_pattern("(?i:hello)", 0).expect("pattern should parse");
        let matches = grep.find_in_line("HELLO World");
        assert_eq!(matches.len(), 1, "(?i:...) flag should match uppercase");
        assert_eq!(matches[0].matched_text, "HELLO");
        assert_eq!(matches[0].distance, 0);
    }

    #[test]
    fn test_case_insensitive_flag_mixed_case() {
        let grep = PhoneticGrep::from_pattern("(?i:HeLLo)", 0).expect("pattern should parse");

        // Should match all case variants
        assert!(grep.matches("hello").is_some());
        assert!(grep.matches("HELLO").is_some());
        assert!(grep.matches("Hello").is_some());
        assert!(grep.matches("HeLLo").is_some());
    }

    #[test]
    fn test_accent_insensitive_flag() {
        // Use (?a:...) flag for accent-insensitive matching
        let grep = PhoneticGrep::from_pattern("(?a:cafe)", 0).expect("pattern should parse");

        // Should match with and without accents
        assert!(grep.matches("cafe").is_some(), "should match base word");
        assert!(
            grep.matches("café").is_some(),
            "should match accented variant"
        );
    }

    #[test]
    fn test_combined_case_accent_flags() {
        // Use (?ia:...) for both case and accent insensitive
        let grep = PhoneticGrep::from_pattern("(?ia:cafe)", 0).expect("pattern should parse");

        // Should match all variants
        assert!(grep.matches("cafe").is_some());
        assert!(grep.matches("CAFE").is_some());
        assert!(grep.matches("café").is_some());
        assert!(grep.matches("CAFÉ").is_some());
        assert!(grep.matches("Cafe").is_some());
    }

    #[test]
    fn test_case_flag_in_alternation() {
        // Flags should apply to entire pattern
        let grep = PhoneticGrep::from_pattern("(?i:phone|fone)", 0).expect("pattern should parse");

        assert!(grep.matches("phone").is_some());
        assert!(grep.matches("PHONE").is_some());
        assert!(grep.matches("fone").is_some());
        assert!(grep.matches("FONE").is_some());
    }

    #[test]
    fn test_case_flag_with_fuzzy_matching() {
        // Case flag + fuzzy matching
        let grep = PhoneticGrep::from_pattern("(?i:hello)", 1).expect("pattern should parse");

        // Exact matches in any case
        assert!(grep.matches("hello").is_some());
        assert!(grep.matches("HELLO").is_some());

        // Fuzzy matches in any case (distance 1)
        assert!(grep.matches("helo").is_some());
        assert!(grep.matches("HELO").is_some());
    }

    #[test]
    fn test_case_flag_vs_runtime_case_insensitive() {
        // (?i:...) should work without .case_insensitive(true)
        let grep_flag = PhoneticGrep::from_pattern("(?i:test)", 0).expect("pattern should parse");
        let grep_runtime = PhoneticGrep::from_pattern("test", 0)
            .expect("pattern should parse")
            .case_insensitive(true);

        // Both should match uppercase
        assert!(grep_flag.matches("TEST").is_some());
        assert!(grep_runtime.matches("TEST").is_some());
    }

    // ========== Local Distance Override Tests ==========

    #[test]
    fn test_local_distance_override_exact() {
        // Pattern with (?;0:...) should override CLI max_distance
        // Even though CLI says max_distance=5, the pattern says 0
        let grep = PhoneticGrep::from_pattern("(?;0:test)", 5).expect("pattern should parse");

        // Verify local_distance is set
        assert_eq!(grep.local_distance(), Some(0), "local_distance should be 0");
        assert_eq!(
            grep.effective_distance(),
            0,
            "effective_distance should be 0"
        );

        // Only exact matches should work
        assert!(grep.matches("test").is_some(), "exact match should work");
        assert!(
            grep.matches("tset").is_none(),
            "1-edit should NOT match with ;0"
        );
        assert!(
            grep.matches("tes").is_none(),
            "1-edit should NOT match with ;0"
        );
    }

    #[test]
    fn test_local_distance_override_fuzzy() {
        // Pattern with (?;2:...) allows 2 edits even if CLI says 0
        let grep = PhoneticGrep::from_pattern("(?;2:test)", 0).expect("pattern should parse");

        // Verify local_distance overrides CLI
        assert_eq!(grep.local_distance(), Some(2), "local_distance should be 2");
        assert_eq!(
            grep.effective_distance(),
            2,
            "effective_distance should be 2"
        );

        // Should allow up to 2 edits
        assert!(grep.matches("test").is_some(), "exact match");
        assert!(grep.matches("tes").is_some(), "1-edit match");
        assert!(grep.matches("te").is_some(), "2-edit match");
        assert!(grep.matches("t").is_none(), "3-edit should NOT match");
    }

    #[test]
    fn test_no_local_distance_uses_max_distance() {
        // Pattern without (?;N:...) should use CLI max_distance
        let grep = PhoneticGrep::from_pattern("test", 1).expect("pattern should parse");

        // Verify local_distance is not set
        assert_eq!(grep.local_distance(), None, "local_distance should be None");
        assert_eq!(
            grep.effective_distance(),
            1,
            "effective_distance should use max_distance"
        );

        // Should use max_distance = 1
        assert!(grep.matches("test").is_some(), "exact match");
        assert!(grep.matches("tes").is_some(), "1-edit match");
        assert!(
            grep.matches("te").is_none(),
            "2-edit should NOT match with max_distance=1"
        );
    }

    #[test]
    fn test_local_distance_with_case_insensitive() {
        // Combined case-insensitive flag with local distance
        let grep = PhoneticGrep::from_pattern("(?i;0:test)", 5).expect("pattern should parse");

        // Verify both flags work
        assert_eq!(grep.local_distance(), Some(0), "local_distance should be 0");
        assert_eq!(
            grep.effective_distance(),
            0,
            "effective_distance should be 0"
        );

        // Case-insensitive exact matches should work
        assert!(grep.matches("test").is_some(), "lowercase exact match");
        assert!(grep.matches("TEST").is_some(), "uppercase exact match");
        assert!(grep.matches("Test").is_some(), "mixed case exact match");

        // But fuzzy matches should NOT work
        assert!(
            grep.matches("tset").is_none(),
            "fuzzy match should NOT work with ;0"
        );
        assert!(
            grep.matches("TSET").is_none(),
            "uppercase fuzzy should NOT work"
        );
    }

    #[test]
    fn test_local_distance_in_grep_file() {
        // Verify local_distance works through grep_file
        let grep = PhoneticGrep::from_pattern("(?;0:hello)", 5).expect("pattern should parse");

        let content = "hello world helo there";
        let results: Vec<_> = grep.grep_file(content).collect();

        // Only "hello" should match (exact), not "helo" (1 edit away)
        assert_eq!(results.len(), 1, "should have 1 matching line");
        assert_eq!(results[0].matches.len(), 1, "should have 1 match in line");
        assert_eq!(
            results[0].matches[0].matched_text, "hello",
            "only exact match"
        );
    }
}
