//! Incremental-input phonetic scanning with exact source spans.
//!
//! This module accepts a document one character at a time, preserves the
//! original UTF-8 text across chunk boundaries, and uses the optimized buffered
//! engine in [`PhoneticGrepOnline`] when the caller finishes the stream.
//! Deferring normalization until `finish` gives the
//! rewrite transducer enough right context to map every normalized output back
//! to the correct source span.
//!
//! # Example
//!
//! ```rust
//! use liblevenshtein::phonetic::online_scanner::OnlinePhoneticScannerChar;
//! use liblevenshtein::phonetic::{ContextChar, PhoneChar, RewriteRuleChar};
//!
//! let ph_to_f = RewriteRuleChar {
//!     rule_id: 1,
//!     rule_name: "ph to f".into(),
//!     pattern: vec![PhoneChar::Consonant('p'), PhoneChar::Consonant('h')],
//!     replacement: vec![PhoneChar::Consonant('f')],
//!     context: ContextChar::Anywhere,
//!     weight: 1.0,
//!     syllable_condition: None,
//! };
//! let mut scanner = OnlinePhoneticScannerChar::new("phone", &[ph_to_f], 0);
//!
//! for ch in "Call my phone now".chars() {
//!     scanner.feed(ch, ch.len_utf8());
//! }
//! scanner.finish();
//! let matches = scanner.scan("");
//!
//! assert_eq!(matches.len(), 1);
//! assert_eq!(matches[0].original_text, "phone");
//! assert_eq!(matches[0].normalized_text, "fone");
//! assert_eq!(matches[0].distance, 0);
//! ```

use super::grep_online::PhoneticGrepOnline;
use super::types::RewriteRuleChar;

/// Result of a successful match during scanning.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct ScanMatch {
    /// Byte range in the original document (start, end exclusive).
    pub byte_range: (usize, usize),
    /// Character range in the original document (start, end exclusive).
    pub char_range: (usize, usize),
    /// The original text that matched.
    pub original_text: String,
    /// The normalized text (after phonetic transformation).
    pub normalized_text: String,
    /// Edit distance between normalized text and normalized query.
    pub distance: u8,
}

/// Phonetic scanner for documents supplied incrementally.
///
/// Input is accumulated as UTF-8 and evaluated by the same bounded-window
/// engine as [`PhoneticGrepOnline::scan`]. This deliberately uses memory linear
/// in the supplied document: phonetic rules can require right context, and
/// retaining the source is what makes [`ScanMatch::byte_range`],
/// [`ScanMatch::char_range`], and [`ScanMatch::original_text`] exact even when a
/// rewrite spans several source characters or input chunks.
#[derive(Debug, Clone)]
pub struct OnlinePhoneticScannerChar {
    matcher: PhoneticGrepOnline,
    normalized_query: String,
    document: String,
    completed_matches: Vec<ScanMatch>,
    reported_matches: Vec<ScanMatch>,
    current_char: usize,
    dirty: bool,
}

impl OnlinePhoneticScannerChar {
    /// Create a scanner for a query, rewrite-rule set, and edit-distance bound.
    pub fn new(query: &str, rules: &[RewriteRuleChar], max_distance: u8) -> Self {
        Self::from_matcher(PhoneticGrepOnline::with_rules(
            query,
            rules.to_vec(),
            max_distance,
        ))
    }

    pub(crate) fn from_matcher(matcher: PhoneticGrepOnline) -> Self {
        let normalized_query = matcher.normalized_query();
        Self {
            matcher,
            normalized_query,
            document: String::new(),
            completed_matches: Vec::new(),
            reported_matches: Vec::new(),
            current_char: 0,
            dirty: false,
        }
    }

    /// Append a document and return matches not reported by an earlier scan.
    ///
    /// Calling this method repeatedly appends to the same logical stream. Use
    /// [`Self::reset`] before scanning an independent document.
    pub fn scan(&mut self, document: &str) -> Vec<ScanMatch> {
        for ch in document.chars() {
            self.feed(ch, ch.len_utf8());
        }
        self.finish();
        std::mem::take(&mut self.completed_matches)
    }

    /// Feed one Unicode scalar from the document.
    ///
    /// `byte_len` is retained in the signature for source compatibility. The
    /// scanner derives byte positions from the accumulated UTF-8 string so an
    /// incorrect hint cannot corrupt returned ranges.
    pub fn feed(&mut self, c: char, _byte_len: usize) {
        self.document.push(c);
        self.current_char += 1;
        self.dirty = true;
    }

    /// Evaluate all input supplied since construction or [`Self::reset`].
    ///
    /// Repeated calls without additional input are idempotent. When more input
    /// is appended after a prior finish, only newly observed matches are made
    /// available to the next [`Self::scan`] call.
    pub fn finish(&mut self) {
        if !self.dirty {
            return;
        }

        for scan_match in self.matcher.scan(&self.document) {
            if !self.reported_matches.contains(&scan_match) {
                self.reported_matches.push(scan_match.clone());
                self.completed_matches.push(scan_match);
            }
        }
        self.dirty = false;
    }

    /// Get the normalized query string.
    pub fn normalized_query(&self) -> &str {
        &self.normalized_query
    }

    /// Get progress statistics for the current stream.
    pub fn stats(&self) -> ScannerStats {
        ScannerStats {
            chars_scanned: self.current_char,
            bytes_scanned: self.document.len(),
            matches_found: self.reported_matches.len(),
            active_matches: 0,
        }
    }

    /// Reset the scanner for an independent document.
    pub fn reset(&mut self) {
        self.document.clear();
        self.completed_matches.clear();
        self.reported_matches.clear();
        self.current_char = 0;
        self.dirty = false;
    }
}

/// Statistics from an incremental scan.
#[derive(Debug, Clone, Copy)]
pub struct ScannerStats {
    /// Number of Unicode scalar values supplied.
    pub chars_scanned: usize,
    /// Number of UTF-8 bytes supplied.
    pub bytes_scanned: usize,
    /// Number of distinct matches reported so far.
    pub matches_found: usize,
    /// Number of live product states.
    ///
    /// The exact-span implementation performs product traversal at
    /// [`OnlinePhoneticScannerChar::finish`], so this value is always zero.
    pub active_matches: usize,
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::phonetic::types::{ContextChar, PhoneChar};

    fn make_rule(pattern: &str, replacement: &str, context: ContextChar) -> RewriteRuleChar {
        fn char_to_phone(c: char) -> PhoneChar {
            if "aeiou".contains(c.to_ascii_lowercase()) {
                PhoneChar::Vowel(c)
            } else {
                PhoneChar::Consonant(c)
            }
        }

        RewriteRuleChar {
            rule_id: 0,
            rule_name: format!("{pattern} -> {replacement}"),
            pattern: pattern.chars().map(char_to_phone).collect(),
            replacement: replacement.chars().map(char_to_phone).collect(),
            context,
            weight: 1.0,
            syllable_condition: None,
        }
    }

    #[test]
    fn empty_document_has_no_match() {
        let mut scanner = OnlinePhoneticScannerChar::new("phone", &[], 0);
        assert!(scanner.scan("").is_empty());
    }

    #[test]
    fn exact_and_phonetic_matches_preserve_source_spans() {
        let rules = [make_rule("ph", "f", ContextChar::Anywhere)];
        let mut scanner = OnlinePhoneticScannerChar::new("phone", &rules, 0);
        let matches = scanner.scan("Call my phone and fone now");

        assert_eq!(matches.len(), 2);
        assert_eq!(matches[0].original_text, "phone");
        assert_eq!(matches[0].byte_range, (8, 13));
        assert_eq!(matches[0].char_range, (8, 13));
        assert_eq!(matches[0].normalized_text, "fone");
        assert_eq!(matches[1].original_text, "fone");
        assert!(matches.iter().all(|m| m.distance == 0));
    }

    #[test]
    fn match_survives_trailing_input_and_chunk_boundaries() {
        let rules = [make_rule("ph", "f", ContextChar::Anywhere)];
        let mut scanner = OnlinePhoneticScannerChar::new("phone", &rules, 0);

        for chunk in ["Call my pho", "ne; keep talking"] {
            for ch in chunk.chars() {
                scanner.feed(ch, ch.len_utf8());
            }
        }
        scanner.finish();
        let matches = scanner.scan("");

        assert_eq!(matches.len(), 1);
        assert_eq!(matches[0].original_text, "phone");
        assert_eq!(matches[0].byte_range, (8, 13));
        assert_eq!(matches[0].distance, 0);
    }

    #[test]
    fn final_context_and_fuzzy_deletion_are_supported() {
        let rules = [
            make_rule("oo", "u", ContextChar::Anywhere),
            make_rule("e", "", ContextChar::Final),
        ];
        let mut scanner = OnlinePhoneticScannerChar::new("fude", &rules, 0);
        let matches = scanner.scan("food");
        assert_eq!(matches[0].distance, 0);

        let mut fuzzy = OnlinePhoneticScannerChar::new("phone", &[], 1);
        let matches = fuzzy.scan("phon");
        assert_eq!(matches[0].original_text, "phon");
        assert_eq!(matches[0].distance, 1);
    }

    #[test]
    fn normalized_query_stats_and_reset_are_stable() {
        let rules = [make_rule("ph", "f", ContextChar::Anywhere)];
        let mut scanner = OnlinePhoneticScannerChar::new("phone", &rules, 0);
        assert_eq!(scanner.normalized_query(), "fone");

        let matches = scanner.scan("phone");
        assert_eq!(matches.len(), 1);
        assert_eq!(scanner.stats().chars_scanned, 5);
        assert_eq!(scanner.stats().bytes_scanned, 5);
        assert_eq!(scanner.stats().matches_found, 1);
        assert_eq!(scanner.stats().active_matches, 0);

        scanner.reset();
        assert_eq!(scanner.stats().chars_scanned, 0);
        assert_eq!(scanner.stats().matches_found, 0);
        assert_eq!(scanner.scan("fone").len(), 1);
    }
}
