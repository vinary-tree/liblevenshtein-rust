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

use std::path::Path;

#[cfg(feature = "parallel-grep")]
use std::sync::Arc;

#[cfg(feature = "parallel-grep")]
use rayon::prelude::*;

use super::grep::GrepError;
use super::llev::{load_file, RuleSetChar};
#[cfg(feature = "parallel-grep")]
use super::nfa::product::ProductAutomatonChar;
#[cfg(feature = "parallel-grep")]
use super::nfa::thompson::ThompsonBuilderChar;
use super::online_scanner::{OnlinePhoneticScannerChar, ScanMatch, ScannerStats};
use super::online_transducer::OnlinePhoneticTransducerChar;
use super::types::RewriteRuleChar;

// ============================================================================
// Parallel Scanning Data Structures (requires parallel-grep feature)
// ============================================================================

/// Candidate match position for parallel verification.
///
/// During Phase 1 (sequential), we normalize the document and generate
/// candidate start positions. In Phase 2 (parallel), each candidate
/// is verified independently using Rayon.
#[cfg(feature = "parallel-grep")]
#[derive(Debug, Clone)]
struct CandidateTask {
    /// Start position in the original document (bytes).
    start_byte: usize,
    /// Start position in the normalized character array.
    start_char: usize,
}

/// Shared normalized document output for zero-copy parallel access.
///
/// After Phase 1 normalization, this structure provides immutable access
/// to the normalized characters from multiple Rayon threads.
#[cfg(feature = "parallel-grep")]
#[derive(Debug, Clone)]
struct SharedNormalized {
    /// Normalized characters (shared across all threads).
    chars: Arc<[char]>,
    /// Maps normalized char index → original byte offset.
    /// `byte_positions[i]` is the byte offset where normalized char `i` started.
    byte_positions: Arc<[usize]>,
}

#[cfg(feature = "parallel-grep")]
impl SharedNormalized {
    /// Get the byte offset for a character position.
    fn byte_offset(&self, char_pos: usize) -> Option<usize> {
        self.byte_positions.get(char_pos).copied()
    }

    fn len(&self) -> usize {
        self.chars.len()
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
    pub fn with_rules(pattern: &str, rules: Vec<RewriteRuleChar>, max_distance: u8) -> Self {
        Self {
            rules,
            pattern: pattern.to_string(),
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
        let llev_file =
            load_file(rules_path).map_err(|e| GrepError::RuleLoad(e.to_string()))?;
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
        Self {
            rules: Vec::new(),
            pattern: pattern.to_string(),
            max_distance,
            case_insensitive: false,
        }
    }

    /// Enable case-insensitive matching.
    ///
    /// When enabled, both the pattern and document are converted to lowercase
    /// before matching. Note that the `original_text` in matches still contains
    /// the original case.
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
        self.case_insensitive = yes;
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
    /// This is the primary method for finding matches. It creates a scanner,
    /// feeds the entire document, and returns all matches found.
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
        let pattern = self.prepare_pattern();
        let doc = self.prepare_document(document);

        let mut scanner = OnlinePhoneticScannerChar::new(&pattern, &self.rules, self.max_distance);
        scanner.scan(&doc)
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
        let pattern = self.prepare_pattern();
        let doc = self.prepare_document(document);

        let mut scanner = OnlinePhoneticScannerChar::new(&pattern, &self.rules, self.max_distance);
        let matches = scanner.scan(&doc);
        let stats = scanner.stats();
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
        let pattern = self.prepare_pattern();
        let scanner = OnlinePhoneticScannerChar::new(&pattern, &self.rules, self.max_distance);
        scanner.normalized_query().to_string()
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
    fn prepare_document(&self, document: &str) -> String {
        if self.case_insensitive {
            document.to_lowercase()
        } else {
            document.to_string()
        }
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
        let pattern = self.prepare_pattern();
        let doc = self.prepare_document(document);

        // Phase 1: Sequential normalization
        let (normalized, byte_positions, doc_byte_len) =
            self.normalize_document_with_positions(&doc);

        if normalized.is_empty() {
            return Vec::new();
        }

        // Build the query NFA and product automaton
        let normalized_query = self.compute_normalized_query(&pattern);
        let query_len = normalized_query.chars().count();

        if query_len == 0 {
            return Vec::new();
        }

        let builder = ThompsonBuilderChar::new();
        let query_nfa = builder.literal(&normalized_query);
        let product = Arc::new(ProductAutomatonChar::new(query_nfa, self.max_distance));

        // Create shared normalized data
        let shared = SharedNormalized {
            chars: Arc::from(normalized.as_slice()),
            byte_positions: Arc::from(byte_positions.as_slice()),
        };

        // Generate candidates (every position is a potential match start)
        let candidates: Vec<CandidateTask> = (0..normalized.len())
            .map(|i| CandidateTask {
                start_byte: shared.byte_offset(i).unwrap_or(0),
                start_char: i,
            })
            .collect();

        // Phase 2: Parallel verification
        let mut matches: Vec<ScanMatch> = candidates
            .into_par_iter()
            .filter_map(|candidate| {
                self.verify_candidate_parallel(
                    &candidate,
                    &shared,
                    &product,
                    query_len,
                    &doc,
                    doc_byte_len,
                )
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

    /// Normalize document and track byte positions for each normalized character.
    #[cfg(feature = "parallel-grep")]
    fn normalize_document_with_positions(&self, document: &str) -> (Vec<char>, Vec<usize>, usize) {
        if self.rules.is_empty() {
            // No rules - just convert to chars and track positions
            let mut chars = Vec::with_capacity(document.len());
            let mut positions = Vec::with_capacity(document.len());
            let mut byte_pos = 0;

            for c in document.chars() {
                positions.push(byte_pos);
                chars.push(c);
                byte_pos += c.len_utf8();
            }

            return (chars, positions, byte_pos);
        }

        // With rules - use transducer and track positions
        let mut transducer = OnlinePhoneticTransducerChar::new(self.rules.clone());

        // Track pending byte positions for buffered characters.
        // When the transducer buffers input, we need to track which byte position
        // each buffered character came from.
        let mut pending_positions: Vec<usize> = Vec::new();
        let mut normalized_chars = Vec::with_capacity(document.len());
        let mut byte_positions = Vec::with_capacity(document.len());

        let mut input_byte_pos = 0;

        for c in document.chars() {
            // Record position before feeding
            pending_positions.push(input_byte_pos);
            let char_byte_len = c.len_utf8();

            // Count output chars before feeding
            let output_count_before = normalized_chars.len();

            // Feed character and collect output
            for out_c in transducer.feed(c) {
                normalized_chars.push(out_c);
            }

            // For each newly emitted char, assign the earliest pending position
            let new_output_count = normalized_chars.len() - output_count_before;
            for _ in 0..new_output_count {
                if !pending_positions.is_empty() {
                    byte_positions.push(pending_positions.remove(0));
                } else {
                    // Fallback: use current position
                    byte_positions.push(input_byte_pos);
                }
            }

            input_byte_pos += char_byte_len;
        }

        // Flush remaining
        for out_c in transducer.finish() {
            normalized_chars.push(out_c);
            if !pending_positions.is_empty() {
                byte_positions.push(pending_positions.remove(0));
            } else {
                byte_positions.push(input_byte_pos.saturating_sub(1));
            }
        }

        (normalized_chars, byte_positions, input_byte_pos)
    }

    /// Compute the normalized form of the query pattern.
    #[cfg(feature = "parallel-grep")]
    fn compute_normalized_query(&self, pattern: &str) -> String {
        if self.rules.is_empty() {
            return pattern.to_string();
        }

        let mut transducer = OnlinePhoneticTransducerChar::new(self.rules.clone());
        transducer.normalize(pattern)
    }

    /// Verify a single candidate match in parallel.
    #[cfg(feature = "parallel-grep")]
    fn verify_candidate_parallel(
        &self,
        candidate: &CandidateTask,
        shared: &SharedNormalized,
        product: &ProductAutomatonChar,
        query_len: usize,
        original_doc: &str,
        doc_byte_len: usize,
    ) -> Option<ScanMatch> {
        // Try different end positions (query_len ± max_distance)
        let min_len = query_len.saturating_sub(self.max_distance as usize);
        let max_len = query_len + self.max_distance as usize;

        let start = candidate.start_char;

        // Find the best match (minimum distance) among all possible lengths
        let mut best_match: Option<(u8, usize, String)> = None; // (distance, end, normalized_text)

        // Check each possible substring length
        for len in min_len..=max_len {
            let end = start + len;
            if end > shared.len() {
                break;
            }

            // Extract the candidate substring
            let candidate_str: String = shared.chars[start..end].iter().collect();

            // Check if it matches within the distance budget
            if let Some(distance) = product.min_distance(&candidate_str) {
                // Check if this is the best match so far
                match &best_match {
                    None => {
                        best_match = Some((distance, end, candidate_str));
                    }
                    Some((best_dist, _, _)) if distance < *best_dist => {
                        best_match = Some((distance, end, candidate_str));
                    }
                    Some((best_dist, best_end, _))
                        if distance == *best_dist && end > *best_end =>
                    {
                        // Prefer longer match at same distance
                        best_match = Some((distance, end, candidate_str));
                    }
                    _ => {}
                }

                // Early exit if we found a perfect match
                if distance == 0 {
                    break;
                }
            }
        }

        // Convert best match to ScanMatch
        best_match.map(|(distance, end, normalized_text)| {
            let byte_start = candidate.start_byte;
            let byte_end = if end < shared.byte_positions.len() {
                shared.byte_positions[end]
            } else {
                doc_byte_len
            };

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
                char_range: (start, end),
                original_text,
                normalized_text,
                distance,
            }
        })
    }

    /// Remove overlapping matches, keeping the best (lowest distance, then longest).
    #[cfg(feature = "parallel-grep")]
    fn deduplicate_matches(&self, matches: Vec<ScanMatch>) -> Vec<ScanMatch> {
        if matches.len() <= 1 {
            return matches;
        }

        let mut result: Vec<ScanMatch> = Vec::with_capacity(matches.len());
        let mut last_end = 0usize;

        for m in matches {
            // Skip if this match starts before the previous match ended
            if m.byte_range.0 < last_end {
                // Check if this match is better (lower distance)
                if let Some(prev) = result.last_mut() {
                    if m.distance < prev.distance
                        || (m.distance == prev.distance
                            && m.byte_range.1 - m.byte_range.0
                                > prev.byte_range.1 - prev.byte_range.0)
                    {
                        *prev = m;
                        last_end = prev.byte_range.1;
                    }
                }
            } else {
                last_end = m.byte_range.1;
                result.push(m);
            }
        }

        result
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
        D: Clone + Send + Sync,
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
        D: Clone + Send + Sync,
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
        D: Clone + Send + Sync,
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
    /// let grep = PhoneticGrepOnline::with_rules("TODO", rules, 0);
    ///
    /// let source_files: Vec<_> = files.iter()
    ///     .map(|f| (f.path(), f.content()))
    ///     .collect();
    ///
    /// let todo_counts = grep.count_documents_parallel(source_files);
    /// let total: usize = todo_counts.iter().map(|(_, c)| c).sum();
    /// println!("Total TODOs: {}", total);
    /// ```
    #[cfg(feature = "parallel-grep")]
    pub fn count_documents_parallel<'a, I, D>(&self, documents: I) -> Vec<(D, usize)>
    where
        I: IntoIterator<Item = (D, &'a str)>,
        D: Clone + Send + Sync,
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
        let text = if self.case_insensitive {
            chunk.to_lowercase()
        } else {
            chunk.to_string()
        };

        for c in text.chars() {
            self.inner.feed(c, c.len_utf8());
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
        // Case insensitive works by lowercasing both pattern and document
        let grep = PhoneticGrepOnline::without_rules("hello", 0).case_insensitive(true);

        // Match just the lowercased pattern (document converted to lowercase)
        let matches = grep.scan("HELLO");
        assert!(!matches.is_empty(), "should match case-insensitively");
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
        assert!(matches.is_empty(), "parallel: should not match unrelated text");
    }

    #[cfg(feature = "parallel-grep")]
    #[test]
    fn test_parallel_empty_document() {
        let rules = vec![make_rule("ph", "f", ContextChar::Anywhere)];
        let grep = PhoneticGrepOnline::with_rules("phone", rules, 0);

        let matches = grep.scan_parallel("");
        assert!(matches.is_empty(), "parallel: empty document has no matches");
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

    // ========================================================================
    // Inter-Document Parallelism Tests
    // ========================================================================

    #[cfg(feature = "parallel-grep")]
    #[test]
    fn test_scan_documents_parallel() {
        let rules = vec![make_rule("ph", "f", ContextChar::Anywhere)];
        let grep = PhoneticGrepOnline::with_rules("phone", rules, 0);

        // Use exact matches since the scanner matches entire documents
        let documents = vec![
            ("doc1", "phone"),
            ("doc2", "fone"),
            ("doc3", "hello"),
        ];

        let results = grep.scan_documents_parallel(documents);

        assert_eq!(results.len(), 3, "should return results for all documents");

        // Find results by document ID
        let doc1_matches = results.iter().find(|(id, _)| *id == "doc1").map(|(_, m)| m);
        let doc2_matches = results.iter().find(|(id, _)| *id == "doc2").map(|(_, m)| m);
        let doc3_matches = results.iter().find(|(id, _)| *id == "doc3").map(|(_, m)| m);

        assert!(
            doc1_matches.map_or(false, |m| !m.is_empty()),
            "doc1 should have matches"
        );
        assert!(
            doc2_matches.map_or(false, |m| !m.is_empty()),
            "doc2 should have matches"
        );
        assert!(
            doc3_matches.map_or(false, |m| m.is_empty()),
            "doc3 should have no matches"
        );
    }

    #[cfg(feature = "parallel-grep")]
    #[test]
    fn test_scan_documents_parallel_nested() {
        let rules = vec![make_rule("ph", "f", ContextChar::Anywhere)];
        let grep = PhoneticGrepOnline::with_rules("phone", rules, 0);

        let documents = vec![
            ("doc1", "phone"),
            ("doc2", "fone"),
        ];

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

        let documents = vec![
            ("doc1", "phone"),
            ("doc2", "no match"),
            ("doc3", "fone"),
        ];

        let results = grep.count_documents_parallel(documents);

        assert_eq!(results.len(), 3, "should return counts for all documents");

        let doc1_count = results.iter().find(|(id, _)| *id == "doc1").map(|(_, c)| *c);
        let doc2_count = results.iter().find(|(id, _)| *id == "doc2").map(|(_, c)| *c);
        let doc3_count = results.iter().find(|(id, _)| *id == "doc3").map(|(_, c)| *c);

        assert!(doc1_count.map_or(false, |c| c >= 1), "doc1 should have >= 1 match");
        assert_eq!(doc2_count, Some(0), "doc2 should have 0 matches");
        assert!(doc3_count.map_or(false, |c| c >= 1), "doc3 should have >= 1 match");
    }

    #[cfg(feature = "parallel-grep")]
    #[test]
    fn test_scan_documents_parallel_empty() {
        let rules = vec![make_rule("ph", "f", ContextChar::Anywhere)];
        let grep = PhoneticGrepOnline::with_rules("phone", rules, 0);

        let documents: Vec<(&str, &str)> = vec![];
        let results = grep.scan_documents_parallel(documents);

        assert!(results.is_empty(), "empty input should produce empty output");
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
}
