//! Online (streaming) phonetic grep without word boundary constraints.
//!
//! Unlike [`PhoneticGrep`] which operates on word boundaries, `PhoneticGrepOnline`
//! performs character-by-character scanning, allowing it to match substrings and
//! handle text without clear word boundaries.
//!
//! # Architecture
//!
//! The online grep combines:
//! - **Pre-normalized query**: Query is normalized once using phonetic rules
//! - **Streaming transducer**: Document text is normalized character-by-character
//! - **Product automaton**: NFA × Levenshtein for fuzzy matching
//! - **Multi-match tracking**: Concurrent match attempts at every position
//!
//! # Comparison with PhoneticGrep
//!
//! | Feature | PhoneticGrep | PhoneticGrepOnline |
//! |---------|--------------|-------------------|
//! | Matching | Word boundaries | Character-by-character |
//! | Normalization | Per-word | Streaming |
//! | Memory | O(1) per word | O(active_matches) |
//! | Use case | Dictionary lookup | Document scanning |
//!
//! # Examples
//!
//! ```ignore
//! use liblevenshtein::phonetic::grep_online::PhoneticGrepOnline;
//! use liblevenshtein::phonetic::rules::english;
//!
//! // "fude" and "food" both normalize to "fud", so distance is 0
//! let grep = PhoneticGrepOnline::with_rules("fude", english::base().rules_vec(), 0);
//!
//! let matches = grep.scan("The food was delicious.");
//! assert_eq!(matches.len(), 1);
//! assert_eq!(matches[0].original_text, "food");
//! assert_eq!(matches[0].distance, 0);  // phonetically equivalent!
//! ```
//!
//! # Streaming API
//!
//! For processing large documents or streams:
//!
//! ```ignore
//! let grep = PhoneticGrepOnline::with_rules("phone", rules, 1);
//! let mut stream = grep.streaming();
//!
//! // Feed chunks as they arrive
//! stream.feed("My pho");
//! stream.feed("ne is ringing");
//!
//! // Get all matches when done
//! let matches = stream.finish();
//! ```

use std::borrow::Cow;
use std::path::Path;

#[cfg(feature = "parallel-grep")]
use std::sync::Arc;

#[cfg(feature = "parallel-grep")]
use rayon::prelude::*;

use super::application::can_apply_at_char;
use super::grep::GrepError;
use super::llev::{load_file, RuleSetChar};
use super::nfa::product::ProductAutomatonChar;
use super::nfa::thompson::ThompsonBuilderChar;
use super::online_scanner::{OnlinePhoneticScannerChar, ScanMatch, ScannerStats};
use super::types::{compare_rule_weight_desc, PhoneChar, RewriteRuleChar};

#[derive(Clone, Copy, Debug)]
struct SourceSpan {
    byte_start: usize,
    byte_end: usize,
    char_start: usize,
    char_end: usize,
}

#[derive(Clone, Copy, Debug)]
struct SourceChar {
    ch: char,
    span: SourceSpan,
}

#[derive(Debug)]
struct NormalizedDocument {
    chars: Vec<char>,
    spans: Vec<SourceSpan>,
    doc_byte_len: usize,
    doc_char_len: usize,
}

#[derive(Debug, Clone)]
struct CompiledQuery {
    normalized: String,
    len: usize,
    product: ProductAutomatonChar,
}

struct BufferedCandidate<'a> {
    start: usize,
    byte_start: usize,
    char_start: usize,
    chars: &'a [char],
    spans: &'a [SourceSpan],
    product: &'a ProductAutomatonChar,
    query_len: usize,
    original_doc: &'a str,
    doc_byte_len: usize,
    doc_char_len: usize,
}

// ============================================================================
// Parallel Scanning Data Structures (requires parallel-grep feature)
// ============================================================================

/// Shared normalized document output for zero-copy parallel access.
///
/// After Phase 1 normalization, this structure provides immutable access
/// to the normalized characters from multiple Rayon threads.
#[cfg(feature = "parallel-grep")]
#[derive(Debug, Clone)]
struct SharedNormalized {
    /// Normalized characters (shared across all threads).
    chars: Arc<[char]>,
    /// Maps normalized char index to its source span.
    spans: Arc<[SourceSpan]>,
}

#[cfg(feature = "parallel-grep")]
struct ParallelCandidate<'a> {
    start: usize,
    byte_start: usize,
    char_start: usize,
    shared: &'a SharedNormalized,
    product: &'a ProductAutomatonChar,
    query_len: usize,
    original_doc: &'a str,
    doc_byte_len: usize,
    doc_char_len: usize,
}

#[cfg(feature = "parallel-grep")]
impl SharedNormalized {
    /// Get the byte offset for a character position.
    fn byte_offset(&self, char_pos: usize) -> Option<usize> {
        self.spans.get(char_pos).map(|span| span.byte_start)
    }

    /// Get the original document character offset for a normalized position.
    fn source_char_offset(&self, char_pos: usize) -> Option<usize> {
        self.spans.get(char_pos).map(|span| span.char_start)
    }

    fn byte_end(&self, end_char: usize, doc_byte_len: usize) -> usize {
        if end_char == 0 {
            0
        } else if end_char <= self.spans.len() {
            self.spans[end_char - 1].byte_end
        } else {
            doc_byte_len
        }
    }

    fn char_end(&self, end_char: usize, doc_char_len: usize) -> usize {
        if end_char == 0 {
            0
        } else if end_char <= self.spans.len() {
            self.spans[end_char - 1].char_end
        } else {
            doc_char_len
        }
    }
}

/// Online phonetic grep for streaming document matching.
///
/// Provides character-by-character scanning with on-the-fly phonetic normalization.
/// Unlike `PhoneticGrep`, this does not require word boundaries and can match
/// arbitrary substrings.
///
/// # Example
///
/// ```ignore
/// use liblevenshtein::phonetic::grep_online::PhoneticGrepOnline;
/// use liblevenshtein::phonetic::rules::english;
///
/// // Find phonetic equivalents
/// let grep = PhoneticGrepOnline::with_rules("phone", english::base().rules_vec(), 0);
/// let matches = grep.scan("Call my fone!"); // "fone" normalizes to same as "phone"
/// ```
#[derive(Debug, Clone)]
pub struct PhoneticGrepOnline {
    /// Phonetic rewrite rules for normalization.
    rules: Vec<RewriteRuleChar>,
    /// The query pattern (before normalization).
    pattern: String,
    compiled: CompiledQuery,
    /// Maximum edit distance for fuzzy matching.
    max_distance: u8,
    /// Case-insensitive matching.
    case_insensitive: bool,
}

impl PhoneticGrepOnline {
    /// Create an online grep matcher with phonetic rules.
    ///
    /// # Arguments
    ///
    /// * `pattern` - Pattern to search for (will be phonetically normalized)
    /// * `rules` - Phonetic rewrite rules for normalization
    /// * `max_distance` - Maximum edit distance (0 = exact after normalization)
    ///
    /// # Example
    ///
    /// ```ignore
    /// use liblevenshtein::phonetic::rules::english;
    ///
    /// let grep = PhoneticGrepOnline::with_rules(
    ///     "phone",
    ///     english::base().rules_vec(),
    ///     1, // Allow 1 edit after normalization
    /// );
    /// ```
    pub fn with_rules(pattern: &str, mut rules: Vec<RewriteRuleChar>, max_distance: u8) -> Self {
        rules.sort_by(compare_rule_weight_desc);
        let pattern = pattern.to_string();
        let compiled = Self::compile_query(&pattern, &rules, false, max_distance);
        Self {
            rules,
            pattern,
            compiled,
            max_distance,
            case_insensitive: false,
        }
    }

    /// Create an online grep matcher loading rules from a file.
    ///
    /// # Arguments
    ///
    /// * `pattern` - Pattern to search for
    /// * `rules_path` - Path to `.llev` rules file
    /// * `max_distance` - Maximum edit distance
    ///
    /// # Errors
    ///
    /// Returns `GrepError` if the rules file cannot be loaded.
    pub fn from_rules_file(
        pattern: &str,
        rules_path: &Path,
        max_distance: u8,
    ) -> Result<Self, GrepError> {
        let llev_file = load_file(rules_path).map_err(|e| GrepError::RuleLoad(e.to_string()))?;
        let ruleset =
            RuleSetChar::from_llev(&llev_file).map_err(|e| GrepError::RuleLoad(e.to_string()))?;

        Ok(Self::with_rules(pattern, ruleset.rules, max_distance))
    }

    /// Create an online grep matcher without phonetic rules.
    ///
    /// This performs pure Levenshtein fuzzy matching without phonetic normalization.
    /// Use this when you want substring fuzzy matching but don't need phonetic
    /// equivalence.
    ///
    /// # Arguments
    ///
    /// * `pattern` - Pattern to search for
    /// * `max_distance` - Maximum edit distance
    pub fn without_rules(pattern: &str, max_distance: u8) -> Self {
        Self::with_rules(pattern, Vec::new(), max_distance)
    }

    /// Enable case-insensitive matching.
    ///
    /// When enabled, both the pattern and document are case-folded before
    /// matching. Note that the `original_text` in matches still contains the
    /// original case.
    ///
    /// # Example
    ///
    /// ```ignore
    /// let grep = PhoneticGrepOnline::without_rules("hello", 0)
    ///     .case_insensitive(true);
    /// let matches = grep.scan("HELLO World");
    /// assert_eq!(matches[0].original_text, "HELLO");
    /// ```
    pub fn case_insensitive(mut self, yes: bool) -> Self {
        if self.case_insensitive != yes {
            self.case_insensitive = yes;
            self.compiled = Self::compile_query(
                &self.pattern,
                &self.rules,
                self.case_insensitive,
                self.max_distance,
            );
        }
        self
    }

    /// Get the pattern (before normalization).
    pub fn pattern(&self) -> &str {
        &self.pattern
    }

    /// Get the maximum edit distance.
    pub fn max_distance(&self) -> u8 {
        self.max_distance
    }

    /// Get the phonetic rules.
    pub fn rules(&self) -> &[RewriteRuleChar] {
        &self.rules
    }

    /// Scan a document and return all matches.
    ///
    /// This is the primary method for finding matches. It normalizes the
    /// document once, verifies bounded candidate windows against the product
    /// automaton, and returns all non-overlapping matches found.
    ///
    /// # Arguments
    ///
    /// * `document` - Text to search
    ///
    /// # Returns
    ///
    /// Vector of matches, sorted by position then distance.
    ///
    /// # Example
    ///
    /// ```ignore
    /// let grep = PhoneticGrepOnline::with_rules("phone", rules, 1);
    /// let matches = grep.scan("My fone and phone are both matched.");
    ///
    /// for m in matches {
    ///     println!("{} at {}-{} (distance {})",
    ///         m.original_text,
    ///         m.byte_range.0,
    ///         m.byte_range.1,
    ///         m.distance);
    /// }
    /// ```
    pub fn scan(&self, document: &str) -> Vec<ScanMatch> {
        self.scan_buffered(document)
    }

    /// Scan a document with the original active-state streaming engine.
    ///
    /// This preserves the prior implementation for streaming parity tests and
    /// scientific A/B evaluation. For ordinary whole-document scanning, prefer
    /// [`Self::scan`], which uses the buffered bounded-window engine.
    pub fn scan_stateful(&self, document: &str) -> Vec<ScanMatch> {
        let pattern = self.prepare_pattern();
        let doc = self.prepare_document(document);

        let mut scanner = OnlinePhoneticScannerChar::new(&pattern, &self.rules, self.max_distance);
        scanner.scan(doc.as_ref())
    }

    /// Scan a document and return matches with statistics.
    ///
    /// Like `scan()`, but also returns statistics about the scanning process.
    ///
    /// # Arguments
    ///
    /// * `document` - Text to search
    ///
    /// # Returns
    ///
    /// Tuple of (matches, stats).
    pub fn scan_with_stats(&self, document: &str) -> (Vec<ScanMatch>, ScannerStats) {
        let matches = self.scan_buffered(document);
        let stats = ScannerStats {
            chars_scanned: document.chars().count(),
            bytes_scanned: document.len(),
            matches_found: matches.len(),
            active_matches: 0,
        };
        (matches, stats)
    }

    /// Get the normalized form of the query pattern.
    ///
    /// This shows what the pattern looks like after phonetic normalization.
    /// Useful for debugging and understanding why certain matches occur.
    ///
    /// # Example
    ///
    /// ```ignore
    /// let grep = PhoneticGrepOnline::with_rules("phone", english_rules, 0);
    /// assert_eq!(grep.normalized_query(), "fone"); // ph → f
    /// ```
    pub fn normalized_query(&self) -> String {
        self.compiled.normalized.clone()
    }

    /// Create a streaming scanner for incremental feeding.
    ///
    /// Use this for large documents or when data arrives in chunks.
    /// The scanner maintains state between `feed()` calls.
    ///
    /// # Example
    ///
    /// ```ignore
    /// let grep = PhoneticGrepOnline::with_rules("hello", rules, 1);
    /// let mut stream = grep.streaming();
    ///
    /// // Feed chunks as they arrive
    /// for chunk in reader.chunks() {
    ///     stream.feed(&chunk);
    /// }
    ///
    /// // Get all matches
    /// let matches = stream.finish();
    /// ```
    pub fn streaming(&self) -> StreamingScanner {
        let pattern = self.prepare_pattern();
        let scanner = OnlinePhoneticScannerChar::new(&pattern, &self.rules, self.max_distance);
        StreamingScanner {
            inner: scanner,
            case_insensitive: self.case_insensitive,
        }
    }

    /// Prepare the pattern for matching (apply case transformation if needed).
    fn prepare_pattern(&self) -> String {
        if self.case_insensitive {
            self.pattern.to_lowercase()
        } else {
            self.pattern.clone()
        }
    }

    /// Prepare the document for matching (apply case transformation if needed).
    fn prepare_document<'a>(&self, document: &'a str) -> Cow<'a, str> {
        if self.case_insensitive {
            Cow::Owned(document.to_lowercase())
        } else {
            Cow::Borrowed(document)
        }
    }

    /// Scan a document using a normalized buffer and bounded candidate windows.
    ///
    /// This path trades `O(n)` normalized-buffer storage for predictable
    /// `O(n * (|query| + d))` candidate verification and avoids maintaining an
    /// active match object for every live start position.
    pub fn scan_buffered(&self, document: &str) -> Vec<ScanMatch> {
        let compiled = &self.compiled;
        let query_len = compiled.len;

        if query_len == 0 || document.is_empty() {
            return Vec::new();
        }

        let normalized_doc = self.normalize_document_with_positions(document);
        if normalized_doc.chars.is_empty() {
            return Vec::new();
        }

        let candidate_starts =
            Self::candidate_start_count(normalized_doc.chars.len(), query_len, self.max_distance);
        if candidate_starts == 0 {
            return Vec::new();
        }

        let mut matches = Vec::with_capacity(candidate_starts.min(64));
        for start_char in 0..candidate_starts {
            let Some(span) = normalized_doc.spans.get(start_char).copied() else {
                continue;
            };
            if let Some(scan_match) = self.verify_candidate_buffered(BufferedCandidate {
                start: start_char,
                byte_start: span.byte_start,
                char_start: span.char_start,
                chars: &normalized_doc.chars,
                spans: &normalized_doc.spans,
                product: &compiled.product,
                query_len,
                original_doc: document,
                doc_byte_len: normalized_doc.doc_byte_len,
                doc_char_len: normalized_doc.doc_char_len,
            }) {
                matches.push(scan_match);
            }
        }

        matches.sort_by(|a, b| {
            a.byte_range
                .0
                .cmp(&b.byte_range.0)
                .then(a.distance.cmp(&b.distance))
        });
        self.deduplicate_matches(matches)
    }

    fn best_match_from_frontier(
        &self,
        product: &ProductAutomatonChar,
        query_len: usize,
        chars: &[char],
        start: usize,
    ) -> Option<(u8, usize, String)> {
        let min_len = Self::min_candidate_len(query_len, self.max_distance);
        let max_len = query_len.saturating_add(self.max_distance as usize);
        let max_end = chars.len().min(start.saturating_add(max_len));

        let mut frontier = product.initial_frontier();
        let mut best_match: Option<(u8, usize)> = None;

        for end in start + 1..=max_end {
            let ch = chars[end - 1];
            frontier = product.transition_frontier(&frontier, ch);
            if frontier.is_empty() {
                break;
            }

            let len = end - start;
            if len < min_len {
                continue;
            }

            if let Some(distance) = product.min_accepting_distance(&frontier) {
                match &best_match {
                    None => best_match = Some((distance, end)),
                    Some((best_dist, _)) if distance < *best_dist => {
                        best_match = Some((distance, end));
                    }
                    Some((best_dist, best_end)) if distance == *best_dist && end > *best_end => {
                        best_match = Some((distance, end));
                    }
                    _ => {}
                }

                if distance == 0 {
                    break;
                }
            }
        }

        best_match.map(|(distance, end)| {
            let normalized_text = Self::collect_chars_exact(&chars[start..end]);
            (distance, end, normalized_text)
        })
    }

    fn min_candidate_len(query_len: usize, max_distance: u8) -> usize {
        query_len.saturating_sub(max_distance as usize).max(1)
    }

    fn candidate_start_count(normalized_len: usize, query_len: usize, max_distance: u8) -> usize {
        let min_len = Self::min_candidate_len(query_len, max_distance);
        normalized_len
            .checked_sub(min_len)
            .and_then(Self::source_index_end)
            .unwrap_or(0)
    }

    fn source_index_end(start: usize) -> Option<usize> {
        start.checked_add(1)
    }

    fn collect_chars_exact(chars: &[char]) -> String {
        let byte_len = chars.iter().map(|ch| ch.len_utf8()).sum();
        let mut text = String::with_capacity(byte_len);
        for ch in chars {
            text.push(*ch);
        }
        text
    }

    /// Normalize document and track source spans for each normalized char.
    fn normalize_document_with_positions(&self, document: &str) -> NormalizedDocument {
        Self::normalize_text_with_positions(document, &self.rules, self.case_insensitive)
    }

    fn normalize_text_with_positions(
        document: &str,
        rules: &[RewriteRuleChar],
        case_insensitive: bool,
    ) -> NormalizedDocument {
        let doc_byte_len = document.len();
        if document.is_empty() {
            return NormalizedDocument {
                chars: Vec::new(),
                spans: Vec::new(),
                doc_byte_len,
                doc_char_len: 0,
            };
        }

        if rules.is_empty() {
            return Self::normalize_without_rules(document, case_insensitive);
        }

        let (chars, doc_char_len) = Self::source_chars_with_positions(document, case_insensitive);
        if chars.is_empty() {
            return NormalizedDocument {
                chars: Vec::new(),
                spans: Vec::new(),
                doc_byte_len,
                doc_char_len,
            };
        }

        let phones: Vec<PhoneChar> = chars
            .iter()
            .map(|source| Self::char_to_phone(source.ch))
            .collect();
        let mut normalized = Vec::with_capacity(chars.len());
        let mut spans = Vec::with_capacity(chars.len());
        let mut pos = 0;

        while pos < chars.len() {
            if let Some(rule) = rules.iter().find(|rule| {
                pos + rule.pattern.len() <= phones.len() && can_apply_at_char(rule, &phones, pos)
            }) {
                let end_pos = pos + rule.pattern.len();
                let span = SourceSpan {
                    byte_start: chars[pos].span.byte_start,
                    byte_end: chars[end_pos - 1].span.byte_end,
                    char_start: chars[pos].span.char_start,
                    char_end: chars[end_pos - 1].span.char_end,
                };
                for phone in &rule.replacement {
                    for ch in phone.chars() {
                        normalized.push(ch);
                        spans.push(span);
                    }
                }
                pos = end_pos;
            } else {
                let source = chars[pos];
                normalized.push(source.ch);
                spans.push(source.span);
                pos += 1;
            }
        }

        NormalizedDocument {
            chars: normalized,
            spans,
            doc_byte_len,
            doc_char_len,
        }
    }

    fn estimated_source_char_count(document: &str) -> usize {
        if document.is_ascii() {
            document.len()
        } else {
            document.len().saturating_div(2).max(1)
        }
    }

    fn normalize_without_rules(document: &str, case_insensitive: bool) -> NormalizedDocument {
        let estimated_chars = Self::estimated_source_char_count(document);
        let mut normalized = Vec::with_capacity(estimated_chars);
        let mut spans = Vec::with_capacity(estimated_chars);
        let mut doc_char_len = 0;

        for (char_start, (byte_start, ch)) in document.char_indices().enumerate() {
            let Some(char_end) = Self::source_index_end(char_start) else {
                break;
            };
            let byte_end = byte_start
                .checked_add(ch.len_utf8())
                .unwrap_or(document.len());
            doc_char_len = char_end;
            let span = SourceSpan {
                byte_start,
                byte_end,
                char_start,
                char_end,
            };
            if case_insensitive {
                for folded in ch.to_lowercase() {
                    normalized.push(folded);
                    spans.push(span);
                }
            } else {
                normalized.push(ch);
                spans.push(span);
            }
        }

        NormalizedDocument {
            chars: normalized,
            spans,
            doc_byte_len: document.len(),
            doc_char_len,
        }
    }

    fn source_chars_with_positions(
        document: &str,
        case_insensitive: bool,
    ) -> (Vec<SourceChar>, usize) {
        let mut chars = Vec::with_capacity(Self::estimated_source_char_count(document));
        let mut doc_char_len = 0;

        for (char_start, (byte_start, ch)) in document.char_indices().enumerate() {
            let Some(char_end) = Self::source_index_end(char_start) else {
                break;
            };
            let byte_end = byte_start
                .checked_add(ch.len_utf8())
                .unwrap_or(document.len());
            doc_char_len = char_end;
            let span = SourceSpan {
                byte_start,
                byte_end,
                char_start,
                char_end,
            };
            if case_insensitive {
                chars.extend(ch.to_lowercase().map(|ch| SourceChar { ch, span }));
            } else {
                chars.push(SourceChar { ch, span });
            }
        }

        (chars, doc_char_len)
    }

    fn compile_query(
        pattern: &str,
        rules: &[RewriteRuleChar],
        case_insensitive: bool,
        max_distance: u8,
    ) -> CompiledQuery {
        let normalized: String =
            Self::normalize_text_with_positions(pattern, rules, case_insensitive)
                .chars
                .into_iter()
                .collect();
        let len = normalized.chars().count();
        let builder = ThompsonBuilderChar::new();
        let query_nfa = builder.literal(&normalized);
        let product = ProductAutomatonChar::new(query_nfa, max_distance);

        CompiledQuery {
            normalized,
            len,
            product,
        }
    }

    fn char_to_phone(ch: char) -> PhoneChar {
        let lower = ch.to_ascii_lowercase();
        if "aeiou".contains(lower) {
            PhoneChar::Vowel(ch)
        } else {
            PhoneChar::Consonant(ch)
        }
    }

    fn verify_candidate_buffered(&self, candidate_ctx: BufferedCandidate<'_>) -> Option<ScanMatch> {
        let BufferedCandidate {
            start,
            byte_start,
            char_start,
            chars,
            spans,
            product,
            query_len,
            original_doc,
            doc_byte_len,
            doc_char_len,
        } = candidate_ctx;
        let best_match = self.best_match_from_frontier(product, query_len, chars, start);

        best_match.map(|(distance, end, normalized_text)| {
            let (byte_end, char_end) = if end > 0 && end <= spans.len() {
                (spans[end - 1].byte_end, spans[end - 1].char_end)
            } else {
                (doc_byte_len, doc_char_len)
            };
            let original_text = original_doc
                .get(byte_start..byte_end)
                .unwrap_or("")
                .to_string();

            ScanMatch {
                byte_range: (byte_start, byte_end),
                char_range: (char_start, char_end),
                original_text,
                normalized_text,
                distance,
            }
        })
    }

    // ========================================================================
    // Parallel Scanning (Rayon) - requires parallel-grep feature
    // ========================================================================

    /// Scan a document using parallel verification with Rayon.
    ///
    /// **Requires the `parallel-grep` feature.**
    ///
    /// This method uses a two-phase approach:
    /// 1. **Phase 1 (Sequential)**: Normalize the document using the phonetic transducer
    /// 2. **Phase 2 (Parallel)**: Verify candidate matches in parallel using Rayon
    ///
    /// # Performance
    ///
    /// Parallel scanning provides speedup for large documents with many potential
    /// match candidates. The sequential transducer phase limits scalability
    /// (Amdahl's Law), but verification parallelism helps for documents with
    /// many candidate positions.
    ///
    /// | Document Size | Expected Speedup (8 cores) |
    /// |--------------|---------------------------|
    /// | < 10 KB      | 1.0x - 1.25x              |
    /// | 100 KB       | 2x - 3x                   |
    /// | 1 MB         | 4x - 5x                   |
    /// | 10 MB        | 5x - 6x                   |
    ///
    /// # Arguments
    ///
    /// * `document` - Text to search
    ///
    /// # Returns
    ///
    /// Vector of matches, sorted by position then distance.
    ///
    /// # Example
    ///
    /// ```ignore
    /// let grep = PhoneticGrepOnline::with_rules("phone", rules, 1);
    ///
    /// // For large documents, parallel scanning may be faster
    /// let matches = grep.scan_parallel(&large_document);
    /// ```
    #[cfg(feature = "parallel-grep")]
    pub fn scan_parallel(&self, document: &str) -> Vec<ScanMatch> {
        let compiled = &self.compiled;
        let query_len = compiled.len;

        if query_len == 0 || document.is_empty() {
            return Vec::new();
        }

        // Phase 1: Sequential normalization
        let normalized_doc = self.normalize_document_with_positions(document);

        if normalized_doc.chars.is_empty() {
            return Vec::new();
        }

        // Create shared normalized data
        let normalized_len = normalized_doc.chars.len();
        let candidate_starts =
            Self::candidate_start_count(normalized_len, query_len, self.max_distance);
        if candidate_starts == 0 {
            return Vec::new();
        }

        let doc_byte_len = normalized_doc.doc_byte_len;
        let doc_char_len = normalized_doc.doc_char_len;
        let shared = SharedNormalized {
            chars: Arc::from(normalized_doc.chars.into_boxed_slice()),
            spans: Arc::from(normalized_doc.spans.into_boxed_slice()),
        };

        // Phase 2: Parallel verification
        let mut matches: Vec<ScanMatch> = (0..candidate_starts)
            .into_par_iter()
            .filter_map(|start| {
                self.verify_candidate_parallel(ParallelCandidate {
                    start,
                    byte_start: shared.byte_offset(start).unwrap_or(0),
                    char_start: shared.source_char_offset(start).unwrap_or(0),
                    shared: &shared,
                    product: &compiled.product,
                    query_len,
                    original_doc: document,
                    doc_byte_len,
                    doc_char_len,
                })
            })
            .collect();

        // Sort by position then distance
        matches.sort_by(|a, b| {
            a.byte_range
                .0
                .cmp(&b.byte_range.0)
                .then(a.distance.cmp(&b.distance))
        });

        // Deduplicate overlapping matches (keep best)
        self.deduplicate_matches(matches)
    }

    /// Verify a single candidate match in parallel.
    #[cfg(feature = "parallel-grep")]
    fn verify_candidate_parallel(&self, candidate_ctx: ParallelCandidate<'_>) -> Option<ScanMatch> {
        let ParallelCandidate {
            start,
            byte_start,
            char_start,
            shared,
            product,
            query_len,
            original_doc,
            doc_byte_len,
            doc_char_len,
        } = candidate_ctx;
        let best_match = self.best_match_from_frontier(product, query_len, &shared.chars, start);

        // Convert best match to ScanMatch
        best_match.map(|(distance, end, normalized_text)| {
            let byte_end = shared.byte_end(end, doc_byte_len);
            let char_end = shared.char_end(end, doc_char_len);

            // Safely extract original text
            let original_text = if byte_start <= byte_end && byte_end <= original_doc.len() {
                original_doc
                    .get(byte_start..byte_end)
                    .unwrap_or("")
                    .to_string()
            } else {
                String::new()
            };

            ScanMatch {
                byte_range: (byte_start, byte_end),
                char_range: (char_start, char_end),
                original_text,
                normalized_text,
                distance,
            }
        })
    }

    /// Remove overlapping matches, keeping the best (lowest distance, then longest).
    fn deduplicate_matches(&self, matches: Vec<ScanMatch>) -> Vec<ScanMatch> {
        if matches.len() <= 1 {
            return matches;
        }

        let mut result: Vec<ScanMatch> = Vec::with_capacity(matches.len());
        let mut group_best: Option<ScanMatch> = None;
        let mut group_end = 0usize;

        for m in matches {
            match group_best.as_mut() {
                Some(best) if m.byte_range.0 < group_end => {
                    group_end = group_end.max(m.byte_range.1);
                    if Self::is_better_match(&m, best) {
                        *best = m;
                    }
                }
                Some(_) => {
                    if let Some(best) = group_best.take() {
                        result.push(best);
                    }
                    group_end = m.byte_range.1;
                    group_best = Some(m);
                }
                None => {
                    group_end = m.byte_range.1;
                    group_best = Some(m);
                }
            }
        }

        if let Some(best) = group_best {
            result.push(best);
        }

        result
    }

    fn is_better_match(candidate: &ScanMatch, incumbent: &ScanMatch) -> bool {
        candidate.distance < incumbent.distance
            || (candidate.distance == incumbent.distance
                && Self::match_byte_len(candidate) > Self::match_byte_len(incumbent))
    }

    fn match_byte_len(scan_match: &ScanMatch) -> usize {
        scan_match
            .byte_range
            .1
            .saturating_sub(scan_match.byte_range.0)
    }

    // ========================================================================
    // Inter-Document Parallelism (Rayon) - requires parallel-grep feature
    // ========================================================================

    /// Scan multiple documents in parallel using Rayon.
    ///
    /// **Requires the `parallel-grep` feature.**
    ///
    /// This method uses Rayon to process multiple documents concurrently,
    /// providing significant speedup when scanning many files.
    ///
    /// # Arguments
    ///
    /// * `documents` - Iterator of (document_id, document_text) pairs
    ///
    /// # Returns
    ///
    /// Vector of (document_id, matches) pairs, preserving document order.
    ///
    /// # Performance
    ///
    /// Inter-document parallelism scales linearly with CPU cores for
    /// independent documents. This is the recommended approach for
    /// scanning many files.
    ///
    /// # Example
    ///
    /// ```ignore
    /// let grep = PhoneticGrepOnline::with_rules("phone", rules, 1);
    ///
    /// let documents = vec![
    ///     ("file1.txt", "My phone is ringing"),
    ///     ("file2.txt", "Call my fone"),
    ///     ("file3.txt", "No match here"),
    /// ];
    ///
    /// let results = grep.scan_documents_parallel(documents);
    /// for (doc_id, matches) in results {
    ///     println!("{}: {} matches", doc_id, matches.len());
    /// }
    /// ```
    #[cfg(feature = "parallel-grep")]
    pub fn scan_documents_parallel<'a, I, D>(&self, documents: I) -> Vec<(D, Vec<ScanMatch>)>
    where
        I: IntoIterator<Item = (D, &'a str)>,
        D: Send,
    {
        let docs: Vec<(D, &str)> = documents.into_iter().collect();

        docs.into_par_iter()
            .map(|(doc_id, text)| {
                let matches = self.scan(text);
                (doc_id, matches)
            })
            .collect()
    }

    /// Scan multiple documents in parallel, using parallel verification within each.
    ///
    /// **Requires the `parallel-grep` feature.**
    ///
    /// This method combines inter-document parallelism (multiple documents
    /// processed concurrently) with intra-document parallelism (parallel
    /// verification within each document).
    ///
    /// # When to Use
    ///
    /// - **Few large documents**: Use `scan_documents_parallel_nested` for
    ///   maximum parallelism within each document.
    /// - **Many small documents**: Use `scan_documents_parallel` which uses
    ///   sequential scanning per document (less overhead).
    ///
    /// # Arguments
    ///
    /// * `documents` - Iterator of (document_id, document_text) pairs
    ///
    /// # Returns
    ///
    /// Vector of (document_id, matches) pairs.
    ///
    /// # Example
    ///
    /// ```ignore
    /// let grep = PhoneticGrepOnline::with_rules("phone", rules, 1);
    ///
    /// // Large documents benefit from nested parallelism
    /// let large_docs = vec![
    ///     ("book1.txt", &large_text_1),
    ///     ("book2.txt", &large_text_2),
    /// ];
    ///
    /// let results = grep.scan_documents_parallel_nested(large_docs);
    /// ```
    #[cfg(feature = "parallel-grep")]
    pub fn scan_documents_parallel_nested<'a, I, D>(&self, documents: I) -> Vec<(D, Vec<ScanMatch>)>
    where
        I: IntoIterator<Item = (D, &'a str)>,
        D: Send,
    {
        let docs: Vec<(D, &str)> = documents.into_iter().collect();

        docs.into_par_iter()
            .map(|(doc_id, text)| {
                let matches = self.scan_parallel(text);
                (doc_id, matches)
            })
            .collect()
    }

    /// Filter documents to only those with matches.
    ///
    /// **Requires the `parallel-grep` feature.**
    ///
    /// Scans documents in parallel and returns only those that contain
    /// at least one match. Useful for filtering large document sets.
    ///
    /// # Arguments
    ///
    /// * `documents` - Iterator of (document_id, document_text) pairs
    ///
    /// # Returns
    ///
    /// Vector of (document_id, matches) pairs for documents with matches.
    ///
    /// # Example
    ///
    /// ```ignore
    /// let grep = PhoneticGrepOnline::with_rules("error", rules, 1);
    ///
    /// let log_files: Vec<_> = logs.iter()
    ///     .map(|log| (log.path.clone(), log.content.as_str()))
    ///     .collect();
    ///
    /// // Get only logs containing "error" (phonetically)
    /// let matching_logs = grep.filter_documents_parallel(log_files);
    /// ```
    #[cfg(feature = "parallel-grep")]
    pub fn filter_documents_parallel<'a, I, D>(&self, documents: I) -> Vec<(D, Vec<ScanMatch>)>
    where
        I: IntoIterator<Item = (D, &'a str)>,
        D: Send,
    {
        let docs: Vec<(D, &str)> = documents.into_iter().collect();

        docs.into_par_iter()
            .filter_map(|(doc_id, text)| {
                let matches = self.scan(text);
                if matches.is_empty() {
                    None
                } else {
                    Some((doc_id, matches))
                }
            })
            .collect()
    }

    /// Count matches across multiple documents in parallel.
    ///
    /// **Requires the `parallel-grep` feature.**
    ///
    /// A lightweight alternative to `scan_documents_parallel` when you only
    /// need match counts, not the matches themselves.
    ///
    /// # Arguments
    ///
    /// * `documents` - Iterator of (document_id, document_text) pairs
    ///
    /// # Returns
    ///
    /// Vector of (document_id, match_count) pairs.
    ///
    /// # Example
    ///
    /// ```ignore
    /// let grep = PhoneticGrepOnline::with_rules("needle", rules, 0);
    ///
    /// let source_files: Vec<_> = files.iter()
    ///     .map(|f| (f.path(), f.content()))
    ///     .collect();
    ///
    /// let match_counts = grep.count_documents_parallel(source_files);
    /// let total: usize = match_counts.iter().map(|(_, c)| c).sum();
    /// println!("Total matches: {}", total);
    /// ```
    #[cfg(feature = "parallel-grep")]
    pub fn count_documents_parallel<'a, I, D>(&self, documents: I) -> Vec<(D, usize)>
    where
        I: IntoIterator<Item = (D, &'a str)>,
        D: Send,
    {
        let docs: Vec<(D, &str)> = documents.into_iter().collect();

        docs.into_par_iter()
            .map(|(doc_id, text)| {
                let matches = self.scan(text);
                (doc_id, matches.len())
            })
            .collect()
    }
}

/// Streaming scanner for incremental document feeding.
///
/// Created by [`PhoneticGrepOnline::streaming()`]. Use this for large documents
/// or streaming data where you want to feed chunks incrementally.
///
/// # Example
///
/// ```ignore
/// let grep = PhoneticGrepOnline::with_rules("pattern", rules, 1);
/// let mut stream = grep.streaming();
///
/// stream.feed("first chunk ");
/// stream.feed("second chunk");
///
/// let matches = stream.finish();
/// ```
pub struct StreamingScanner {
    inner: OnlinePhoneticScannerChar,
    case_insensitive: bool,
}

impl StreamingScanner {
    /// Feed a chunk of text to the scanner.
    ///
    /// Characters are processed immediately, but matches may not be complete
    /// until more context is available or `finish()` is called.
    ///
    /// # Arguments
    ///
    /// * `chunk` - Text chunk to process
    pub fn feed(&mut self, chunk: &str) {
        if self.case_insensitive {
            for c in chunk.to_lowercase().chars() {
                self.inner.feed(c, c.len_utf8());
            }
        } else {
            for c in chunk.chars() {
                self.inner.feed(c, c.len_utf8());
            }
        }
    }

    /// Signal end of input and return all matches.
    ///
    /// This flushes any pending state in the phonetic transducer and
    /// finalizes all match attempts.
    ///
    /// # Returns
    ///
    /// Vector of all matches found during scanning.
    pub fn finish(mut self) -> Vec<ScanMatch> {
        // scan("") is a no-op for input but calls finish() and returns matches
        self.inner.scan("")
    }

    /// Get current statistics without finishing.
    ///
    /// Useful for progress reporting during long scans.
    pub fn stats(&self) -> ScannerStats {
        self.inner.stats()
    }

    /// Get the normalized query string.
    pub fn normalized_query(&self) -> &str {
        self.inner.normalized_query()
    }
}

// ScanMatch is already re-exported via use statement at top

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

    fn scan_match(byte_range: (usize, usize), distance: u8) -> ScanMatch {
        ScanMatch {
            byte_range,
            char_range: byte_range,
            original_text: String::new(),
            normalized_text: String::new(),
            distance,
        }
    }

    #[test]
    fn test_source_index_end_checks_overflow() {
        assert_eq!(PhoneticGrepOnline::source_index_end(0), Some(1));
        assert_eq!(PhoneticGrepOnline::source_index_end(usize::MAX), None);
    }

    #[test]
    fn test_candidate_start_count_uses_checked_successor() {
        assert_eq!(PhoneticGrepOnline::candidate_start_count(5, 3, 0), 3);
        assert_eq!(PhoneticGrepOnline::candidate_start_count(0, 3, 0), 0);
    }

    #[test]
    fn test_basic_match() {
        let rules = vec![make_rule("ph", "f", ContextChar::Anywhere)];
        let grep = PhoneticGrepOnline::with_rules("phone", rules, 0);

        let matches = grep.scan("phone");
        assert!(!matches.is_empty(), "should match 'phone'");
        assert_eq!(matches[0].distance, 0);
    }

    #[test]
    fn test_phonetic_equivalence() {
        let rules = vec![make_rule("ph", "f", ContextChar::Anywhere)];
        let grep = PhoneticGrepOnline::with_rules("phone", rules, 0);

        // "fone" should match "phone" with distance 0 after normalization
        let matches = grep.scan("fone");
        assert!(!matches.is_empty(), "'fone' should match 'phone'");
        assert_eq!(matches[0].distance, 0, "phonetically equivalent");
    }

    #[test]
    fn test_fude_food_equivalence() {
        // Key test case from the plan: "fude" matches "food" with distance 0
        let rules = vec![
            make_rule("oo", "u", ContextChar::Anywhere),
            make_rule("e", "", ContextChar::Final),
        ];

        let grep = PhoneticGrepOnline::with_rules("fude", rules, 0);

        // Verify normalization
        assert_eq!(grep.normalized_query(), "fud");

        // "food" should match "fude" with distance 0
        let matches = grep.scan("food");
        assert!(!matches.is_empty(), "'food' should match 'fude'");
        assert_eq!(
            matches[0].distance, 0,
            "should be exact after normalization"
        );
    }

    #[test]
    fn test_fuzzy_with_phonetic() {
        let rules = vec![make_rule("ph", "f", ContextChar::Anywhere)];
        let grep = PhoneticGrepOnline::with_rules("phone", rules, 1);

        // "fon" is distance 1 from "fone" (which is normalized "phone")
        let matches = grep.scan("fon");
        assert!(!matches.is_empty(), "'fon' should fuzzy match 'phone'");
        assert!(matches[0].distance <= 1);
    }

    #[test]
    fn test_without_rules() {
        // Without phonetic rules, the scanner still works but is best
        // suited for exact or near-exact matches of the document
        let grep = PhoneticGrepOnline::without_rules("hello", 0);

        // Exact match works
        let matches = grep.scan("hello");
        assert!(!matches.is_empty(), "should match 'hello'");
        assert_eq!(matches[0].distance, 0);
    }

    #[test]
    fn test_case_insensitive() {
        // Case insensitive works by folding both pattern and document for matching.
        let grep = PhoneticGrepOnline::without_rules("hello", 0).case_insensitive(true);

        // Match just the lowercased pattern.
        let matches = grep.scan("HELLO");
        assert!(!matches.is_empty(), "should match case-insensitively");
    }

    #[test]
    fn test_case_insensitive_rebuilds_cached_query() {
        let grep = PhoneticGrepOnline::without_rules("HELLO", 0);
        assert_eq!(grep.normalized_query(), "HELLO");

        let grep = grep.case_insensitive(true);

        assert_eq!(grep.normalized_query(), "hello");
        assert_eq!(grep.scan("hello").len(), 1);
    }

    #[test]
    fn test_case_insensitive_preserves_original_text_and_stats() {
        let grep = PhoneticGrepOnline::without_rules("hello", 0).case_insensitive(true);
        let doc = "Say HELLO World";

        let (matches, stats) = grep.scan_with_stats(doc);

        assert_eq!(matches.len(), 1);
        assert_eq!(matches[0].original_text, "HELLO");
        assert_eq!(matches[0].byte_range, (4, 9));
        assert_eq!(matches[0].char_range, (4, 9));
        assert_eq!(stats.chars_scanned, doc.chars().count());
        assert_eq!(stats.bytes_scanned, doc.len());
        assert_eq!(stats.matches_found, 1);
    }

    #[test]
    fn test_case_insensitive_unicode_expansion_preserves_source_span() {
        let grep = PhoneticGrepOnline::without_rules("i\u{307}", 0).case_insensitive(true);

        let matches = grep.scan("İ");

        assert_eq!(matches.len(), 1);
        assert_eq!(matches[0].original_text, "İ");
        assert_eq!(matches[0].normalized_text, "i\u{307}");
        assert_eq!(matches[0].byte_range, (0, "İ".len()));
        assert_eq!(matches[0].char_range, (0, 1));
    }

    #[test]
    fn test_no_rule_normalization_tracks_case_fold_expansion_spans() {
        let normalized = PhoneticGrepOnline::normalize_text_with_positions("aİb", &[], true);

        assert_eq!(normalized.chars, vec!['a', 'i', '\u{307}', 'b']);
        assert_eq!(normalized.doc_byte_len, "aİb".len());
        assert_eq!(normalized.doc_char_len, 3);
        assert_eq!(normalized.spans[1].byte_start, 1);
        assert_eq!(normalized.spans[1].byte_end, 1 + "İ".len());
        assert_eq!(normalized.spans[1].char_start, 1);
        assert_eq!(normalized.spans[1].char_end, 2);
        assert_eq!(
            normalized.spans[2].byte_start,
            normalized.spans[1].byte_start
        );
        assert_eq!(normalized.spans[2].byte_end, normalized.spans[1].byte_end);
        assert_eq!(
            normalized.spans[2].char_start,
            normalized.spans[1].char_start
        );
        assert_eq!(normalized.spans[2].char_end, normalized.spans[1].char_end);
    }

    #[test]
    fn test_streaming() {
        let rules = vec![make_rule("ph", "f", ContextChar::Anywhere)];
        let grep = PhoneticGrepOnline::with_rules("phone", rules, 0);

        let mut stream = grep.streaming();
        stream.feed("my pho");
        stream.feed("ne");

        let matches = stream.finish();
        assert!(!matches.is_empty(), "streaming should find match");
    }

    #[test]
    fn test_normalized_query() {
        let rules = vec![
            make_rule("ph", "f", ContextChar::Anywhere),
            make_rule("oo", "u", ContextChar::Anywhere),
        ];

        let grep = PhoneticGrepOnline::with_rules("philosophy", rules.clone(), 0);
        assert_eq!(grep.normalized_query(), "filosofy");

        let grep2 = PhoneticGrepOnline::with_rules("food", rules, 0);
        assert_eq!(grep2.normalized_query(), "fud");
    }

    #[test]
    fn test_no_match() {
        let rules = vec![make_rule("ph", "f", ContextChar::Anywhere)];
        let grep = PhoneticGrepOnline::with_rules("phone", rules, 0);

        let matches = grep.scan("hello world");
        assert!(matches.is_empty(), "should not match unrelated text");
    }

    #[test]
    fn test_scan_with_stats() {
        let rules = vec![make_rule("ph", "f", ContextChar::Anywhere)];
        let grep = PhoneticGrepOnline::with_rules("phone", rules, 0);

        let (matches, stats) = grep.scan_with_stats("phone");
        assert!(!matches.is_empty());
        assert_eq!(stats.chars_scanned, 5);
        assert_eq!(stats.bytes_scanned, 5);
        assert!(stats.matches_found >= 1);
    }

    #[test]
    fn test_in_sentence() {
        // Test scanning for "phone" (normalizes to "fone") with phonetic rules
        // The grep API wraps the underlying scanner
        let rules = vec![make_rule("ph", "f", ContextChar::Anywhere)];

        // Test 1: Exact document = exact pattern match
        let grep = PhoneticGrepOnline::with_rules("phone", rules.clone(), 0);
        let matches = grep.scan("phone");
        assert!(!matches.is_empty(), "should find 'phone' exactly");

        // Test 2: Phonetically equivalent "fone"
        let grep2 = PhoneticGrepOnline::with_rules("phone", rules.clone(), 0);
        let matches2 = grep2.scan("fone");
        assert!(!matches2.is_empty(), "'fone' should match 'phone'");
    }

    #[test]
    fn test_buffered_scan_finds_proper_substring() {
        let grep = PhoneticGrepOnline::without_rules("phone", 0);
        let matches = grep.scan("call phone now");

        assert_eq!(matches.len(), 1);
        assert_eq!(matches[0].original_text, "phone");
        assert_eq!(matches[0].byte_range, (5, 10));
        assert_eq!(matches[0].distance, 0);
    }

    #[test]
    fn test_buffered_scan_checks_last_exact_candidate_start() {
        let grep = PhoneticGrepOnline::without_rules("phone", 0);
        let matches = grep.scan("xxphone");

        assert_eq!(matches.len(), 1);
        assert_eq!(matches[0].original_text, "phone");
        assert_eq!(matches[0].byte_range, (2, 7));
        assert_eq!(matches[0].distance, 0);
    }

    #[test]
    fn test_buffered_scan_checks_last_fuzzy_candidate_start() {
        let grep = PhoneticGrepOnline::without_rules("phone", 1);
        let matches = grep.scan("xxphon");

        assert_eq!(matches.len(), 1);
        assert_eq!(matches[0].original_text, "phon");
        assert_eq!(matches[0].byte_range, (2, 6));
        assert_eq!(matches[0].distance, 1);
    }

    #[test]
    fn test_buffered_scan_reports_original_char_range_after_rewrite() {
        let rules = vec![make_rule("ph", "f", ContextChar::Anywhere)];
        let grep = PhoneticGrepOnline::with_rules("phone", rules, 0);

        let matches = grep.scan("call phone now");

        assert_eq!(matches.len(), 1);
        assert_eq!(matches[0].original_text, "phone");
        assert_eq!(matches[0].byte_range, (5, 10));
        assert_eq!(matches[0].char_range, (5, 10));
        assert_eq!(matches[0].normalized_text, "fone");
    }

    #[test]
    fn test_buffered_scan_finds_repeated_phonetic_matches() {
        let rules = vec![make_rule("ph", "f", ContextChar::Anywhere)];
        let grep = PhoneticGrepOnline::with_rules("phone", rules, 0);
        let matches = grep.scan("phone and fone are both phones");

        let matched_texts: Vec<_> = matches.iter().map(|m| m.original_text.as_str()).collect();
        assert_eq!(matched_texts, vec!["phone", "fone", "phone"]);
        assert!(matches.iter().all(|m| m.distance == 0));
    }

    #[test]
    fn test_deduplicate_matches_uses_connected_overlap_groups() {
        let grep = PhoneticGrepOnline::without_rules("x", 0);
        let matches = vec![
            scan_match((0, 10), 2),
            scan_match((2, 4), 0),
            scan_match((5, 8), 0),
            scan_match((12, 15), 1),
        ];

        let deduplicated = grep.deduplicate_matches(matches);

        assert_eq!(deduplicated.len(), 2);
        assert_eq!(deduplicated[0].byte_range, (5, 8));
        assert_eq!(deduplicated[0].distance, 0);
        assert_eq!(deduplicated[1].byte_range, (12, 15));
    }

    #[test]
    fn test_incremental_frontier_distances_match_exact_min_distance() {
        let grep = PhoneticGrepOnline::without_rules("phone", 2);
        let normalized_query = grep.normalized_query();
        let builder = ThompsonBuilderChar::new();
        let query_nfa = builder.literal(&normalized_query);
        let product = ProductAutomatonChar::new(query_nfa, grep.max_distance());
        let chars: Vec<char> = "xxphone phon fone phones".chars().collect();
        let max_len = normalized_query.chars().count() + grep.max_distance() as usize;

        for start in 0..chars.len() {
            let mut candidate = String::with_capacity(max_len);
            let mut frontier = product.initial_frontier();
            let max_end = chars.len().min(start + max_len);

            for end in start + 1..=max_end {
                let ch = chars[end - 1];
                candidate.push(ch);
                frontier = product.transition_frontier(&frontier, ch);

                assert_eq!(
                    product.min_accepting_distance(&frontier),
                    product.min_distance(&candidate),
                    "distance mismatch for candidate {candidate:?} at {start}..{end}"
                );
            }
        }
    }

    // ========================================================================
    // Parallel Scanning Tests (requires parallel-grep feature)
    // ========================================================================

    #[cfg(feature = "parallel-grep")]
    #[test]
    fn test_parallel_basic_match() {
        let rules = vec![make_rule("ph", "f", ContextChar::Anywhere)];
        let grep = PhoneticGrepOnline::with_rules("phone", rules, 0);

        let matches = grep.scan_parallel("phone");
        assert!(!matches.is_empty(), "parallel should match 'phone'");
        assert_eq!(matches[0].distance, 0);
    }

    #[cfg(feature = "parallel-grep")]
    #[test]
    fn test_parallel_phonetic_equivalence() {
        let rules = vec![make_rule("ph", "f", ContextChar::Anywhere)];
        let grep = PhoneticGrepOnline::with_rules("phone", rules, 0);

        // "fone" should match "phone" with distance 0 after normalization
        let matches = grep.scan_parallel("fone");
        assert!(!matches.is_empty(), "parallel: 'fone' should match 'phone'");
        assert_eq!(matches[0].distance, 0, "phonetically equivalent");
    }

    #[cfg(feature = "parallel-grep")]
    #[test]
    fn test_parallel_fude_food_equivalence() {
        // Key test case from the plan: "fude" matches "food" with distance 0
        let rules = vec![
            make_rule("oo", "u", ContextChar::Anywhere),
            make_rule("e", "", ContextChar::Final),
        ];

        let grep = PhoneticGrepOnline::with_rules("fude", rules, 0);

        // Verify normalization
        assert_eq!(grep.normalized_query(), "fud");

        // "food" should match "fude" with distance 0
        let matches = grep.scan_parallel("food");
        assert!(!matches.is_empty(), "parallel: 'food' should match 'fude'");
        assert_eq!(
            matches[0].distance, 0,
            "should be exact after normalization"
        );
    }

    #[cfg(feature = "parallel-grep")]
    #[test]
    fn test_parallel_fuzzy_match() {
        let rules = vec![make_rule("ph", "f", ContextChar::Anywhere)];
        let grep = PhoneticGrepOnline::with_rules("phone", rules, 1);

        // "fon" is distance 1 from "fone" (which is normalized "phone")
        let matches = grep.scan_parallel("fon");
        assert!(
            !matches.is_empty(),
            "parallel: 'fon' should fuzzy match 'phone'"
        );
        assert!(matches[0].distance <= 1);
    }

    #[cfg(feature = "parallel-grep")]
    #[test]
    fn test_parallel_no_match() {
        let rules = vec![make_rule("ph", "f", ContextChar::Anywhere)];
        let grep = PhoneticGrepOnline::with_rules("phone", rules, 0);

        let matches = grep.scan_parallel("xyz");
        assert!(
            matches.is_empty(),
            "parallel: should not match unrelated text"
        );
    }

    #[cfg(feature = "parallel-grep")]
    #[test]
    fn test_parallel_empty_document() {
        let rules = vec![make_rule("ph", "f", ContextChar::Anywhere)];
        let grep = PhoneticGrepOnline::with_rules("phone", rules, 0);

        let matches = grep.scan_parallel("");
        assert!(
            matches.is_empty(),
            "parallel: empty document has no matches"
        );
    }

    #[cfg(feature = "parallel-grep")]
    #[test]
    fn test_parallel_without_rules() {
        let grep = PhoneticGrepOnline::without_rules("hello", 0);

        let matches = grep.scan_parallel("hello");
        assert!(!matches.is_empty(), "parallel: should match 'hello'");
        assert_eq!(matches[0].distance, 0);
    }

    #[cfg(feature = "parallel-grep")]
    #[test]
    fn test_parallel_case_insensitive_preserves_original_text() {
        let grep = PhoneticGrepOnline::without_rules("hello", 0).case_insensitive(true);

        let matches = grep.scan_parallel("Say HELLO World");

        assert_eq!(matches.len(), 1);
        assert_eq!(matches[0].original_text, "HELLO");
        assert_eq!(matches[0].byte_range, (4, 9));
        assert_eq!(matches[0].char_range, (4, 9));
    }

    #[cfg(feature = "parallel-grep")]
    #[test]
    fn test_parallel_matches_sequential() {
        // Verify parallel and sequential produce same results
        let rules = vec![
            make_rule("ph", "f", ContextChar::Anywhere),
            make_rule("oo", "u", ContextChar::Anywhere),
        ];
        let grep = PhoneticGrepOnline::with_rules("phone", rules, 1);

        let seq_matches = grep.scan("fone");
        let par_matches = grep.scan_parallel("fone");

        // Both should find the match
        assert!(!seq_matches.is_empty(), "sequential should find match");
        assert!(!par_matches.is_empty(), "parallel should find match");

        // Distances should be the same
        assert_eq!(
            seq_matches[0].distance, par_matches[0].distance,
            "distances should match"
        );
    }

    #[cfg(feature = "parallel-grep")]
    #[test]
    fn test_parallel_multiple_candidates() {
        // Test with a longer document to exercise parallel verification
        let rules = vec![make_rule("ph", "f", ContextChar::Anywhere)];
        let grep = PhoneticGrepOnline::with_rules("fone", rules, 0);

        // Create a document with the pattern at the start
        let doc = "fone world";
        let matches = grep.scan_parallel(doc);

        assert!(!matches.is_empty(), "should find at least one match");
        // The match should be at position 0
        assert_eq!(matches[0].byte_range.0, 0, "match should be at start");
    }

    #[cfg(feature = "parallel-grep")]
    #[test]
    fn test_parallel_checks_last_fuzzy_candidate_start() {
        let grep = PhoneticGrepOnline::without_rules("phone", 1);
        let matches = grep.scan_parallel("xxphon");

        assert_eq!(matches.len(), 1);
        assert_eq!(matches[0].original_text, "phon");
        assert_eq!(matches[0].byte_range, (2, 6));
        assert_eq!(matches[0].distance, 1);
    }

    // ========================================================================
    // Inter-Document Parallelism Tests
    // ========================================================================

    #[cfg(feature = "parallel-grep")]
    #[test]
    fn test_scan_documents_parallel() {
        let rules = vec![make_rule("ph", "f", ContextChar::Anywhere)];
        let grep = PhoneticGrepOnline::with_rules("phone", rules, 0);

        // Use exact matches since the scanner matches entire documents
        let documents = vec![("doc1", "phone"), ("doc2", "fone"), ("doc3", "hello")];

        let results = grep.scan_documents_parallel(documents);

        assert_eq!(results.len(), 3, "should return results for all documents");

        // Find results by document ID
        let doc1_matches = results.iter().find(|(id, _)| *id == "doc1").map(|(_, m)| m);
        let doc2_matches = results.iter().find(|(id, _)| *id == "doc2").map(|(_, m)| m);
        let doc3_matches = results.iter().find(|(id, _)| *id == "doc3").map(|(_, m)| m);

        assert!(
            doc1_matches.is_some_and(|m| !m.is_empty()),
            "doc1 should have matches"
        );
        assert!(
            doc2_matches.is_some_and(|m| !m.is_empty()),
            "doc2 should have matches"
        );
        assert!(
            doc3_matches.is_some_and(|m| m.is_empty()),
            "doc3 should have no matches"
        );
    }

    #[cfg(feature = "parallel-grep")]
    #[test]
    fn test_scan_documents_parallel_accepts_move_only_ids() {
        #[derive(Debug, PartialEq, Eq)]
        struct DocumentId(&'static str);

        let grep = PhoneticGrepOnline::without_rules("phone", 0);
        let documents = vec![(DocumentId("doc1"), "phone"), (DocumentId("doc2"), "hello")];

        let results = grep.scan_documents_parallel(documents);

        assert_eq!(results.len(), 2);
        let matched = results
            .iter()
            .find(|(id, _)| id == &DocumentId("doc1"))
            .map(|(_, matches)| matches);
        assert!(matched.is_some_and(|matches| !matches.is_empty()));
    }

    #[cfg(feature = "parallel-grep")]
    #[test]
    fn test_scan_documents_parallel_nested() {
        let rules = vec![make_rule("ph", "f", ContextChar::Anywhere)];
        let grep = PhoneticGrepOnline::with_rules("phone", rules, 0);

        let documents = vec![("doc1", "phone"), ("doc2", "fone")];

        let results = grep.scan_documents_parallel_nested(documents);

        assert_eq!(results.len(), 2, "should return results for all documents");

        for (doc_id, matches) in &results {
            assert!(!matches.is_empty(), "{} should have matches", doc_id);
            assert_eq!(
                matches[0].distance, 0,
                "{} should match with distance 0",
                doc_id
            );
        }
    }

    #[cfg(feature = "parallel-grep")]
    #[test]
    fn test_filter_documents_parallel() {
        let rules = vec![make_rule("ph", "f", ContextChar::Anywhere)];
        let grep = PhoneticGrepOnline::with_rules("phone", rules, 0);

        // Use exact matches since the scanner matches entire documents
        let documents = vec![
            ("match1", "phone"),
            ("nomatch", "hello"),
            ("match2", "fone"),
        ];

        let results = grep.filter_documents_parallel(documents);

        assert_eq!(
            results.len(),
            2,
            "should only return documents with matches"
        );

        let ids: Vec<_> = results.iter().map(|(id, _)| *id).collect();
        assert!(ids.contains(&"match1"), "should contain match1");
        assert!(ids.contains(&"match2"), "should contain match2");
        assert!(!ids.contains(&"nomatch"), "should not contain nomatch");
    }

    #[cfg(feature = "parallel-grep")]
    #[test]
    fn test_count_documents_parallel() {
        let rules = vec![make_rule("ph", "f", ContextChar::Anywhere)];
        let grep = PhoneticGrepOnline::with_rules("phone", rules, 0);

        let documents = vec![("doc1", "phone"), ("doc2", "no match"), ("doc3", "fone")];

        let results = grep.count_documents_parallel(documents);

        assert_eq!(results.len(), 3, "should return counts for all documents");

        let doc1_count = results
            .iter()
            .find(|(id, _)| *id == "doc1")
            .map(|(_, c)| *c);
        let doc2_count = results
            .iter()
            .find(|(id, _)| *id == "doc2")
            .map(|(_, c)| *c);
        let doc3_count = results
            .iter()
            .find(|(id, _)| *id == "doc3")
            .map(|(_, c)| *c);

        assert!(
            doc1_count.is_some_and(|c| c >= 1),
            "doc1 should have >= 1 match"
        );
        assert_eq!(doc2_count, Some(0), "doc2 should have 0 matches");
        assert!(
            doc3_count.is_some_and(|c| c >= 1),
            "doc3 should have >= 1 match"
        );
    }

    #[cfg(feature = "parallel-grep")]
    #[test]
    fn test_scan_documents_parallel_empty() {
        let rules = vec![make_rule("ph", "f", ContextChar::Anywhere)];
        let grep = PhoneticGrepOnline::with_rules("phone", rules, 0);

        let documents: Vec<(&str, &str)> = vec![];
        let results = grep.scan_documents_parallel(documents);

        assert!(
            results.is_empty(),
            "empty input should produce empty output"
        );
    }

    #[cfg(feature = "parallel-grep")]
    #[test]
    fn test_scan_documents_parallel_with_string_ids() {
        let rules = vec![make_rule("ph", "f", ContextChar::Anywhere)];
        let grep = PhoneticGrepOnline::with_rules("phone", rules, 0);

        // Test with owned String IDs
        let documents: Vec<(String, &str)> = vec![
            ("file1.txt".to_string(), "phone"),
            ("file2.txt".to_string(), "fone"),
        ];

        let results = grep.scan_documents_parallel(documents);

        assert_eq!(results.len(), 2);
        for (path, matches) in results {
            assert!(!matches.is_empty(), "{} should have matches", path);
        }
    }

    // ========================================================================
    // Non-ASCII regression tests (Phase 4 / item 1781)
    //
    // The online scanner reports BOTH a `char_range` and a `byte_range`. For
    // multi-byte UTF-8 these two must diverge, `byte_range` must land on char
    // boundaries, and slicing the document by `byte_range` must recover
    // `original_text` exactly.
    // ========================================================================

    #[test]
    fn test_scan_online_non_ascii_two_byte_char_range_and_byte_range() {
        // `é` is a 2-byte UTF-8 scalar, so char and byte offsets diverge.
        let grep = PhoneticGrepOnline::without_rules("café", 0);
        let document = "un café serré";

        let matches = grep.scan(document);
        assert_eq!(matches.len(), 1, "expected exactly one 'café' match");

        let m = &matches[0];
        assert_eq!(m.original_text, "café");
        assert_eq!(m.char_range, (3, 7));
        assert_eq!(m.byte_range, (3, 3 + "café".len()));
        // The 2-byte `é` forces the byte range to differ from the char range.
        assert_ne!(m.byte_range, m.char_range);
        assert!(document.is_char_boundary(m.byte_range.0));
        assert!(document.is_char_boundary(m.byte_range.1));
        assert_eq!(
            document.get(m.byte_range.0..m.byte_range.1),
            Some(m.original_text.as_str())
        );
    }

    #[test]
    fn test_scan_online_non_ascii_cjk_three_byte() {
        // Each CJK ideograph is 3 bytes; byte offsets are 3x the char offsets here.
        let grep = PhoneticGrepOnline::without_rules("中文", 0);
        let document = "你好中文世界";

        let matches = grep.scan(document);
        assert_eq!(matches.len(), 1, "expected exactly one '中文' match");

        let m = &matches[0];
        assert_eq!(m.original_text, "中文");
        assert_eq!(m.char_range, (2, 4));
        assert_eq!(m.byte_range, (6, 12));
        assert_ne!(m.byte_range, m.char_range);
        assert!(document.is_char_boundary(m.byte_range.0));
        assert!(document.is_char_boundary(m.byte_range.1));
        assert_eq!(
            document.get(m.byte_range.0..m.byte_range.1),
            Some(m.original_text.as_str())
        );
    }

    #[test]
    fn test_scan_online_non_ascii_emoji_four_byte() {
        // `🌍` is a 4-byte (astral-plane) UTF-8 scalar occupying a single char.
        let grep = PhoneticGrepOnline::without_rules("🌍", 0);
        let document = "hi 🌍!";

        let matches = grep.scan(document);
        assert_eq!(matches.len(), 1, "expected exactly one '🌍' match");

        let m = &matches[0];
        assert_eq!(m.original_text, "🌍");
        assert_eq!(m.char_range, (3, 4));
        assert_eq!(m.byte_range, (3, 7));
        assert_ne!(m.byte_range, m.char_range);
        assert!(document.is_char_boundary(m.byte_range.0));
        assert!(document.is_char_boundary(m.byte_range.1));
        assert_eq!(
            document.get(m.byte_range.0..m.byte_range.1),
            Some(m.original_text.as_str())
        );
    }
}
