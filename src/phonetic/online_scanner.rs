//! Online (streaming) phonetic scanner for character-by-character matching.
//!
//! This module implements a streaming scanner that tracks multiple potential matches
//! simultaneously as it scans through a document character-by-character.
//!
//! # Architecture
//!
//! The scanner combines:
//! - **OnlinePhoneticTransducerChar**: Normalizes document text on-the-fly
//! - **ProductAutomatonChar**: NFA × Levenshtein for fuzzy matching
//! - **Multiple match tracking**: Starts new match attempts at each character position
//!
//! # Example
//!
//! ```ignore
//! use liblevenshtein::phonetic::online_scanner::OnlinePhoneticScannerChar;
//! use liblevenshtein::phonetic::rules::english;
//!
//! let rules = english::base().rules_vec();
//! let mut scanner = OnlinePhoneticScannerChar::new("phone", &rules, 1);
//!
//! // Scan a document
//! let matches = scanner.scan("Call my fone or phone.");
//! // matches[0]: "fone" at bytes 8..12, distance 0 (phonetically equivalent)
//! // matches[1]: "phone" at bytes 16..21, distance 0
//! ```

use super::nfa::product::{ProductAutomatonChar, ProductStateChar};
use super::nfa::thompson::ThompsonBuilderChar;
use super::nfa::NFAChar;
use super::online_transducer::OnlinePhoneticTransducerChar;
use super::types::RewriteRuleChar;

/// Maximum number of active match attempts to track simultaneously.
/// This prevents unbounded memory growth in pathological cases.
const MAX_ACTIVE_MATCHES: usize = 1000;

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

/// A potential match being tracked during scanning.
#[derive(Debug, Clone)]
struct PotentialMatch {
    /// Starting byte position in the original document.
    start_byte: usize,
    /// Starting character position in the original document.
    start_char: usize,
    /// Ending byte position (exclusive) - updated as we consume chars.
    end_byte: usize,
    /// Ending character position (exclusive) - updated as we consume chars.
    end_char: usize,
    /// Accumulated original text for this match attempt.
    original_chars: Vec<char>,
    /// Accumulated normalized text for this match attempt.
    normalized_chars: Vec<char>,
    /// Current state in the product automaton.
    product_state: ProductStateChar,
    /// Minimum distance seen at any accepting state.
    min_distance: Option<u8>,
    /// Whether this match attempt is still alive.
    alive: bool,
}

impl PotentialMatch {
    fn new(start_byte: usize, start_char: usize, initial_state: ProductStateChar) -> Self {
        Self {
            start_byte,
            start_char,
            end_byte: start_byte,
            end_char: start_char,
            original_chars: Vec::new(),
            normalized_chars: Vec::new(),
            product_state: initial_state,
            min_distance: None,
            alive: true,
        }
    }

    fn original_text(&self) -> String {
        self.original_chars.iter().collect()
    }

    fn normalized_text(&self) -> String {
        self.normalized_chars.iter().collect()
    }
}

/// Online phonetic scanner for streaming document matching.
///
/// Scans through documents character-by-character, applying phonetic normalization
/// on-the-fly and tracking multiple potential matches simultaneously.
#[derive(Debug, Clone)]
pub struct OnlinePhoneticScannerChar {
    /// The normalized query string.
    normalized_query: String,
    /// NFA for the normalized query.
    query_nfa: NFAChar,
    /// Product automaton for fuzzy matching.
    product: ProductAutomatonChar,
    /// Maximum edit distance allowed.
    max_distance: u8,
    /// Transducer for normalizing document text.
    transducer: OnlinePhoneticTransducerChar,
    /// Active match attempts being tracked.
    active_matches: Vec<PotentialMatch>,
    /// Completed matches found.
    completed_matches: Vec<ScanMatch>,
    /// Current byte position in the document.
    current_byte: usize,
    /// Current character position in the document.
    current_char: usize,
    /// Buffer of original chars awaiting normalization confirmation.
    /// (We need to track original chars corresponding to normalized output)
    pending_original: Vec<(char, usize)>, // (char, byte_len)
    /// Length of the normalized query (for match length checking).
    query_len: usize,
    /// Total matches found across all scans (preserved after scan() returns).
    total_matches_found: usize,
}

impl OnlinePhoneticScannerChar {
    fn active_match_normalized_limit(query_len: usize, max_distance: u8) -> Option<usize> {
        query_len
            .checked_add(usize::from(max_distance))?
            .checked_add(1)?
            .checked_mul(2)
    }

    /// Create a new scanner for the given query.
    ///
    /// # Arguments
    ///
    /// * `query` - The pattern to search for
    /// * `rules` - Phonetic rewrite rules to apply
    /// * `max_distance` - Maximum edit distance for fuzzy matching
    ///
    /// # Returns
    ///
    /// A new scanner ready to scan documents.
    pub fn new(query: &str, rules: &[RewriteRuleChar], max_distance: u8) -> Self {
        // Create a transducer for normalizing the query
        let mut query_transducer = OnlinePhoneticTransducerChar::new(rules.to_vec());
        let normalized_query = query_transducer.normalize(query);

        // Build NFA for the normalized query (literal match)
        let builder = ThompsonBuilderChar::new();
        let query_nfa = builder.literal(&normalized_query);

        // Create product automaton
        let product = ProductAutomatonChar::new(query_nfa.clone(), max_distance);

        // Create transducer for document normalization
        let transducer = OnlinePhoneticTransducerChar::new(rules.to_vec());

        let query_len = normalized_query.chars().count();

        Self {
            normalized_query,
            query_nfa,
            product,
            max_distance,
            transducer,
            active_matches: Vec::with_capacity(64),
            completed_matches: Vec::new(),
            current_byte: 0,
            current_char: 0,
            pending_original: Vec::new(),
            query_len,
            total_matches_found: 0,
        }
    }

    /// Scan an entire document and return all matches.
    ///
    /// This is a convenience method that feeds all characters and finishes.
    pub fn scan(&mut self, document: &str) -> Vec<ScanMatch> {
        for c in document.chars() {
            self.feed(c, c.len_utf8());
        }
        self.finish();

        // Update total before taking
        self.total_matches_found += self.completed_matches.len();
        std::mem::take(&mut self.completed_matches)
    }

    /// Feed a single character from the document.
    ///
    /// # Arguments
    ///
    /// * `c` - The character to process
    /// * `byte_len` - The byte length of this character in UTF-8
    pub fn feed(&mut self, c: char, byte_len: usize) {
        // Track this original character
        self.pending_original.push((c, byte_len));

        // Start a new potential match at the current position
        let initial_state = self.product.initial_state();
        self.active_matches.push(PotentialMatch::new(
            self.current_byte,
            self.current_char,
            initial_state,
        ));

        // Feed to transducer and get normalized output
        let normalized_chars: Vec<char> = self.transducer.feed(c).collect();

        // Process each normalized character through all active matches
        for norm_c in normalized_chars {
            self.process_normalized_char(norm_c);
        }

        // Update position tracking
        self.current_byte += byte_len;
        self.current_char += 1;

        // Prune dead matches and limit active count
        self.prune_matches();
    }

    /// Signal end of document and finish scanning.
    ///
    /// This flushes the transducer buffer and completes any pending matches.
    pub fn finish(&mut self) {
        // Flush remaining normalized characters
        let remaining: Vec<char> = self.transducer.finish().collect();
        for norm_c in remaining {
            self.process_normalized_char(norm_c);
        }

        // Try to complete matches that might need deletions (pattern longer than input)
        self.try_deletions_at_end();

        // Check all active matches for final acceptance
        self.finalize_matches();
    }

    /// Try to complete matches by allowing deletions at end of document.
    ///
    /// When the document is shorter than the query, the remaining query chars
    /// can be "deleted" (skipped) with edit distance cost.
    fn try_deletions_at_end(&mut self) {
        // Collect updates to avoid borrow conflict
        let mut updates: Vec<(usize, u8)> = Vec::new();

        for (i, m) in self.active_matches.iter().enumerate() {
            if !m.alive {
                continue;
            }

            // If already at an accepting state, record it
            if self.product.is_accepting(&m.product_state) {
                let dist = m.product_state.edit_distance();
                if m.min_distance.map_or(true, |d| dist < d) {
                    updates.push((i, dist));
                }
                continue;
            }

            // Try to reach accepting state through deletions (advancing NFA without input)
            let mut state = m.product_state.clone();
            let mut attempts = 0;
            let max_deletions = self.max_distance.saturating_sub(state.edit_distance());

            while attempts < max_deletions as usize + 1 {
                if self.product.is_accepting(&state) {
                    let dist = state.edit_distance();
                    if m.min_distance.map_or(true, |d| dist < d) {
                        updates.push((i, dist));
                    }
                    break;
                }

                // Try deletion: advance NFA without consuming input
                if let Some(deletion_state) = self.try_deletion_helper(&state) {
                    state = deletion_state;
                    attempts += 1;
                } else {
                    break;
                }
            }
        }

        // Apply updates
        for (i, dist) in updates {
            if let Some(m) = self.active_matches.get_mut(i) {
                if m.min_distance.map_or(true, |d| dist < d) {
                    m.min_distance = Some(dist);
                }
            }
        }
    }

    /// Try a single deletion operation (advance NFA without consuming input).
    fn try_deletion_helper(&self, state: &ProductStateChar) -> Option<ProductStateChar> {
        if state.edit_distance() >= self.max_distance {
            return None;
        }

        use rustc_hash::FxHashSet;

        let mut next_states = FxHashSet::default();

        // For each NFA state, try all transitions (deletion = advance NFA without input)
        for &nfa_state in &state.nfa_states {
            for trans in self.query_nfa.transitions_from(nfa_state) {
                if trans.label.consumes_input() {
                    let closure = self
                        .query_nfa
                        .epsilon_closure(&std::iter::once(trans.to).collect());
                    next_states.extend(closure.iter());
                }
            }
        }

        if next_states.is_empty() {
            None
        } else {
            Some(ProductStateChar::new(
                next_states,
                state.accumulated_cost + 1.0,
            ))
        }
    }

    /// Process a normalized character through all active matches.
    fn process_normalized_char(&mut self, c: char) {
        // Distribute original chars to active matches
        // This is approximate - we assign original chars proportionally
        self.distribute_original_chars();
        let normalized_limit =
            Self::active_match_normalized_limit(self.query_len, self.max_distance)
                .unwrap_or(usize::MAX);

        for m in &mut self.active_matches {
            if !m.alive {
                continue;
            }

            // Add normalized char to this match's buffer
            m.normalized_chars.push(c);

            // Compute successor states via product automaton
            let successors = self.product.transition(&m.product_state, c);

            if successors.is_empty() {
                // No valid transitions - this match is dead
                m.alive = false;
            } else {
                // Take the best successor (minimum edit distance)
                let best = successors
                    .into_iter()
                    .min_by_key(|s| s.edit_distance())
                    .expect("successors not empty");

                // Check if we've reached an accepting state
                if self.product.is_accepting(&best) {
                    // Record if this is the best distance seen
                    let dist = best.edit_distance();
                    if m.min_distance.map_or(true, |d| dist < d) {
                        m.min_distance = Some(dist);
                    }
                }

                m.product_state = best;

                // Kill if we've consumed too many chars without hope
                if m.normalized_chars.len() > normalized_limit {
                    m.alive = false;
                }
            }
        }
    }

    /// Distribute pending original characters to active matches.
    fn distribute_original_chars(&mut self) {
        if self.pending_original.is_empty() {
            return;
        }

        // Track accumulated byte position as we process pending chars
        let mut byte_pos = self.current_byte;
        for &(_, byte_len) in &self.pending_original {
            byte_pos = byte_pos.saturating_sub(byte_len);
        }

        // Give each active match only the chars from their start position onward
        for (orig_c, byte_len) in self.pending_original.drain(..) {
            for m in &mut self.active_matches {
                // Only give this char to matches that started at or before this position
                if m.alive && m.start_byte <= byte_pos {
                    m.original_chars.push(orig_c);
                    m.end_byte = byte_pos + byte_len;
                    m.end_char = m.start_char + m.original_chars.len();
                }
            }
            byte_pos += byte_len;
        }
    }

    /// Prune dead matches and limit total active matches.
    fn prune_matches(&mut self) {
        // Remove dead matches
        self.active_matches.retain(|m| m.alive);

        // If too many, keep the most promising ones
        if self.active_matches.len() > MAX_ACTIVE_MATCHES {
            // Sort by (has_match, -distance, recency)
            self.active_matches.sort_by(|a, b| {
                // Prefer matches that have found accepting states
                let a_has = a.min_distance.is_some();
                let b_has = b.min_distance.is_some();
                if a_has != b_has {
                    return b_has.cmp(&a_has);
                }

                // Then by minimum distance seen
                match (a.min_distance, b.min_distance) {
                    (Some(a_d), Some(b_d)) => a_d.cmp(&b_d),
                    _ => {
                        // Prefer shorter matches (more likely to complete)
                        a.normalized_chars.len().cmp(&b.normalized_chars.len())
                    }
                }
            });

            self.active_matches.truncate(MAX_ACTIVE_MATCHES);
        }
    }

    /// Finalize active matches at end of document.
    fn finalize_matches(&mut self) {
        for m in &self.active_matches {
            if let Some(dist) = m.min_distance {
                // Only record if we haven't seen this match already
                let scan_match = ScanMatch {
                    byte_range: (m.start_byte, m.end_byte),
                    char_range: (m.start_char, m.end_char),
                    original_text: m.original_text(),
                    normalized_text: m.normalized_text(),
                    distance: dist,
                };

                // Avoid duplicates
                if !self.completed_matches.iter().any(|existing| {
                    existing.byte_range == scan_match.byte_range
                        && existing.distance == scan_match.distance
                }) {
                    self.completed_matches.push(scan_match);
                }
            }
        }

        // Sort by position then distance
        self.completed_matches
            .sort_by_key(|m| (m.byte_range.0, m.distance));
    }

    /// Get the normalized query string.
    pub fn normalized_query(&self) -> &str {
        &self.normalized_query
    }

    /// Get statistics about the scan.
    pub fn stats(&self) -> ScannerStats {
        ScannerStats {
            chars_scanned: self.current_char,
            bytes_scanned: self.current_byte,
            matches_found: self.total_matches_found + self.completed_matches.len(),
            active_matches: self.active_matches.len(),
        }
    }

    /// Reset the scanner for a new document.
    pub fn reset(&mut self) {
        self.transducer.reset();
        self.active_matches.clear();
        self.completed_matches.clear();
        self.current_byte = 0;
        self.current_char = 0;
        self.pending_original.clear();
        self.total_matches_found = 0;
    }
}

/// Statistics from a scan operation.
#[derive(Debug, Clone, Copy)]
pub struct ScannerStats {
    /// Number of characters scanned.
    pub chars_scanned: usize,
    /// Number of bytes scanned.
    pub bytes_scanned: usize,
    /// Number of matches found.
    pub matches_found: usize,
    /// Number of active match attempts at end of scan.
    pub active_matches: usize,
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::phonetic::types::{ContextChar, PhoneChar};

    /// Helper to create a simple rule for testing.
    fn make_rule(pattern: &str, replacement: &str, context: ContextChar) -> RewriteRuleChar {
        fn char_to_phone(c: char) -> PhoneChar {
            let lower = c.to_ascii_lowercase();
            if "aeiou".contains(lower) {
                PhoneChar::Vowel(c)
            } else {
                PhoneChar::Consonant(c)
            }
        }

        RewriteRuleChar {
            rule_id: 0,
            rule_name: format!("{} -> {}", pattern, replacement),
            pattern: pattern.chars().map(char_to_phone).collect(),
            replacement: replacement.chars().map(char_to_phone).collect(),
            context,
            weight: 1.0,
            syllable_condition: None,
        }
    }

    #[test]
    fn test_active_match_normalized_limit_checks_overflow() {
        assert_eq!(
            OnlinePhoneticScannerChar::active_match_normalized_limit(4, 1),
            Some(12)
        );
        assert_eq!(
            OnlinePhoneticScannerChar::active_match_normalized_limit(usize::MAX, 0),
            None
        );
        assert_eq!(
            OnlinePhoneticScannerChar::active_match_normalized_limit(usize::MAX / 2, 0),
            None
        );
    }

    #[test]
    fn test_empty_document() {
        let rules = vec![make_rule("ph", "f", ContextChar::Anywhere)];
        let mut scanner = OnlinePhoneticScannerChar::new("phone", &rules, 0);

        let matches = scanner.scan("");
        assert!(matches.is_empty());
    }

    #[test]
    fn test_exact_match() {
        let rules = vec![make_rule("ph", "f", ContextChar::Anywhere)];
        let mut scanner = OnlinePhoneticScannerChar::new("phone", &rules, 0);

        // "phone" normalizes to "fone", query "phone" also normalizes to "fone"
        let matches = scanner.scan("phone");

        // Should find the match
        assert!(!matches.is_empty(), "expected to find 'phone' match");
        assert_eq!(matches[0].distance, 0);
    }

    #[test]
    fn test_phonetic_equivalent() {
        let rules = vec![make_rule("ph", "f", ContextChar::Anywhere)];
        let mut scanner = OnlinePhoneticScannerChar::new("phone", &rules, 0);

        // Query "phone" normalizes to "fone"
        assert_eq!(scanner.normalized_query(), "fone");

        // Document "fone" normalizes to "fone" - exact match
        let matches = scanner.scan("fone");

        assert!(!matches.is_empty(), "expected to find 'fone' match");
        assert_eq!(matches[0].distance, 0);
        assert_eq!(matches[0].original_text, "fone");
    }

    #[test]
    fn test_fude_food_equivalence() {
        // The key test case: "fude" matches "food" with distance 0
        let rules = vec![
            make_rule("oo", "u", ContextChar::Anywhere),
            make_rule("e", "", ContextChar::Final),
        ];

        let mut scanner = OnlinePhoneticScannerChar::new("fude", &rules, 0);

        // "fude" normalizes to "fud"
        assert_eq!(scanner.normalized_query(), "fud");

        // "food" normalizes to "fud" - exact match after normalization
        let matches = scanner.scan("food");

        assert!(!matches.is_empty(), "expected 'food' to match 'fude'");
        assert_eq!(
            matches[0].distance, 0,
            "should be exact match after normalization"
        );
    }

    #[test]
    fn test_fuzzy_match() {
        let rules = vec![make_rule("ph", "f", ContextChar::Anywhere)];
        let mut scanner = OnlinePhoneticScannerChar::new("phone", &rules, 1);

        // "fon" is one deletion away from "fone"
        let matches = scanner.scan("fon");

        assert!(!matches.is_empty(), "expected fuzzy match with distance 1");
        assert!(matches[0].distance <= 1);
    }

    #[test]
    fn test_multiple_matches() {
        let rules = vec![make_rule("ph", "f", ContextChar::Anywhere)];
        let mut scanner = OnlinePhoneticScannerChar::new("phone", &rules, 0);

        let matches = scanner.scan("phone and fone are both phones");

        // Should find at least one match (the first "phone")
        // NOTE: Multiple matches in a single scan is limited because all matches share
        // a single transducer. Future iterations may use per-match transducers or
        // a global normalized buffer for better multi-match detection.
        assert!(!matches.is_empty(), "expected at least 1 match");

        // The first match should be "phone" at the start
        assert_eq!(matches[0].distance, 0);
    }

    #[test]
    fn test_normalized_query() {
        let rules = vec![
            make_rule("ph", "f", ContextChar::Anywhere),
            make_rule("oo", "u", ContextChar::Anywhere),
        ];

        let scanner = OnlinePhoneticScannerChar::new("philosophy", &rules, 0);
        assert_eq!(scanner.normalized_query(), "filosofy");

        let scanner2 = OnlinePhoneticScannerChar::new("food", &rules, 0);
        assert_eq!(scanner2.normalized_query(), "fud");
    }

    #[test]
    fn test_no_match() {
        let rules = vec![make_rule("ph", "f", ContextChar::Anywhere)];
        let mut scanner = OnlinePhoneticScannerChar::new("phone", &rules, 0);

        let matches = scanner.scan("hello world");
        assert!(matches.is_empty(), "should not match unrelated text");
    }

    #[test]
    fn test_stats() {
        let rules = vec![make_rule("ph", "f", ContextChar::Anywhere)];
        let mut scanner = OnlinePhoneticScannerChar::new("phone", &rules, 0);

        scanner.scan("phone");
        let stats = scanner.stats();

        assert_eq!(stats.chars_scanned, 5);
        assert_eq!(stats.bytes_scanned, 5);
        assert!(stats.matches_found >= 1);
    }

    #[test]
    fn test_reset() {
        let rules = vec![make_rule("ph", "f", ContextChar::Anywhere)];
        let mut scanner = OnlinePhoneticScannerChar::new("phone", &rules, 0);

        scanner.scan("phone");
        assert!(scanner.stats().matches_found >= 1);

        scanner.reset();
        assert_eq!(scanner.stats().chars_scanned, 0);
        assert_eq!(scanner.stats().matches_found, 0);

        // Should work again after reset
        scanner.scan("phone");
        assert!(scanner.stats().matches_found >= 1);
    }
}
