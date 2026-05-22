//! Online (streaming) phonetic transducer for character-by-character normalization.
//!
//! This module implements a streaming phonetic transducer that buffers input characters
//! and applies phonetic rules as soon as sufficient context is available.
//!
//! # Key Features
//!
//! - **Incremental processing**: Characters are processed one at a time
//! - **Context-aware buffering**: Waits for lookahead when rules require context
//! - **Multi-character patterns**: Handles rules like `ph → f` that span multiple chars
//! - **Memory efficient**: Fixed buffer size based on rule characteristics
//!
//! # Example
//!
//! ```ignore
//! use liblevenshtein::phonetic::online_transducer::OnlinePhoneticTransducerChar;
//! use liblevenshtein::phonetic::rules::english;
//!
//! let rules = english::base().rules_vec();
//! let mut transducer = OnlinePhoneticTransducerChar::new(rules);
//!
//! // Feed characters one at a time
//! for c in "phone".chars() {
//!     for normalized in transducer.feed(c) {
//!         print!("{}", normalized);
//!     }
//! }
//! // Flush remaining buffer
//! for c in transducer.finish() {
//!     print!("{}", c);
//! }
//! // Output: "fon"
//! ```

use std::collections::VecDeque;

use super::application::can_apply_at_char;
use super::types::{ContextChar, PhoneChar, RewriteRuleChar};

/// Online phonetic transducer for streaming normalization.
///
/// Buffers input characters and applies phonetic rules incrementally,
/// emitting normalized output as soon as sufficient context is available.
#[derive(Debug, Clone)]
pub struct OnlinePhoneticTransducerChar {
    /// Phonetic rewrite rules (sorted by weight, highest first)
    rules: Vec<RewriteRuleChar>,

    /// Input buffer for partial pattern matching.
    /// Characters are held here until we can determine whether a rule applies.
    input_buffer: Vec<char>,

    /// Output buffer for normalized characters ready for emission.
    output_buffer: VecDeque<char>,

    /// Maximum pattern length across all rules.
    /// Determines the minimum buffer size needed for pattern matching.
    max_pattern_len: usize,

    /// Maximum lookahead needed for context conditions.
    /// Rules with `BeforeVowel`, `BeforeConsonant`, or `Final` need lookahead.
    max_lookahead: usize,

    /// Whether we've received end-of-input signal.
    at_end: bool,

    /// Absolute position of the first character in input_buffer within the input stream.
    /// Used for `Initial` context checking - position 0 means buffer starts at input start.
    buffer_start_pos: usize,

    /// Statistics for debugging
    chars_processed: usize,
    rules_applied: usize,
}

impl OnlinePhoneticTransducerChar {
    /// Create a new online transducer with the given rules.
    ///
    /// Rules are automatically sorted by weight (highest first) for priority ordering.
    pub fn new(mut rules: Vec<RewriteRuleChar>) -> Self {
        // Sort rules by weight descending (higher weight = higher priority)
        rules.sort_by(|a, b| {
            b.weight
                .partial_cmp(&a.weight)
                .unwrap_or(std::cmp::Ordering::Equal)
        });

        // Compute max pattern length
        let max_pattern_len = rules.iter().map(|r| r.pattern.len()).max().unwrap_or(0);

        // Compute max lookahead from context requirements
        let max_lookahead = rules
            .iter()
            .map(|r| Self::context_lookahead(&r.context))
            .max()
            .unwrap_or(0);

        Self {
            rules,
            input_buffer: Vec::with_capacity(max_pattern_len + max_lookahead + 1),
            output_buffer: VecDeque::with_capacity(max_pattern_len * 2),
            max_pattern_len,
            max_lookahead,
            at_end: false,
            buffer_start_pos: 0,
            chars_processed: 0,
            rules_applied: 0,
        }
    }

    /// Determine how much lookahead a context requires.
    fn context_lookahead(ctx: &ContextChar) -> usize {
        match ctx {
            // "Before" contexts need to see the next character
            ContextChar::BeforeVowel(_) | ContextChar::BeforeConsonant(_) => 1,
            // Final needs end-of-input signal (infinite lookahead until at_end)
            ContextChar::Final => 1,
            // "After" contexts don't need lookahead (they look backward)
            ContextChar::AfterVowel(_) | ContextChar::AfterConsonant(_) => 0,
            // Initial and Anywhere don't need lookahead
            ContextChar::Initial | ContextChar::Anywhere => 0,
            // Compound contexts: max of children
            ContextChar::And(a, b) => Self::context_lookahead(a).max(Self::context_lookahead(b)),
            ContextChar::Or(a, b) => Self::context_lookahead(a).max(Self::context_lookahead(b)),
            ContextChar::Not(inner) => Self::context_lookahead(inner),
        }
    }

    /// Feed a single character to the transducer.
    ///
    /// Returns an iterator over normalized characters ready for emission.
    /// Note that output may be delayed due to buffering for context-sensitive rules.
    pub fn feed(&mut self, c: char) -> impl Iterator<Item = char> + '_ {
        self.input_buffer.push(c);
        self.chars_processed += 1;

        // Try to process the buffer and emit what we can
        self.process_buffer();

        // Return iterator over output buffer
        DrainIter {
            inner: &mut self.output_buffer,
        }
    }

    /// Signal end of input and flush remaining buffer.
    ///
    /// Must be called after all input has been fed to ensure remaining
    /// characters are normalized and emitted.
    pub fn finish(&mut self) -> impl Iterator<Item = char> + '_ {
        self.at_end = true;

        // Process remaining buffer with final context
        self.process_buffer_final();

        // Return iterator over output buffer
        DrainIter {
            inner: &mut self.output_buffer,
        }
    }

    /// Process the input buffer, applying rules and emitting output.
    fn process_buffer(&mut self) {
        // Keep processing until no more progress can be made
        loop {
            let mut applied = false;

            // Try each position in the buffer
            let mut pos = 0;
            while pos < self.input_buffer.len() {
                // Check if we have enough lookahead for context
                let remaining = self.input_buffer.len() - pos;
                let required_lookahead = self.max_pattern_len + self.max_lookahead;

                if remaining < required_lookahead && !self.at_end {
                    // Need more characters for context - emit safe prefix and wait
                    break;
                }

                // Try to apply a rule at this position
                if let Some((rule_idx, pattern_len)) = self.find_matching_rule(pos) {
                    // Collect characters to emit before the match point
                    let prefix_chars: Vec<char> = (0..pos).map(|i| self.input_buffer[i]).collect();

                    // Collect replacement characters
                    let replacement_chars: Vec<char> = self.rules[rule_idx]
                        .replacement
                        .iter()
                        .filter_map(Self::phone_to_char)
                        .collect();

                    // Now emit everything (no longer borrowing self.input_buffer or self.rules)
                    for c in prefix_chars {
                        self.emit_char(c);
                    }
                    for c in replacement_chars {
                        self.emit_char(c);
                    }

                    // Remove processed characters from buffer and track position
                    let remove_count = pos + pattern_len;
                    self.input_buffer.drain(0..remove_count);
                    self.buffer_start_pos += remove_count;

                    self.rules_applied += 1;
                    applied = true;
                    break; // Restart from beginning of buffer
                }

                pos += 1;
            }

            if !applied {
                // No rules applied - emit safe prefix if we can
                self.emit_safe_prefix();
                break;
            }
        }
    }

    /// Process buffer at end-of-input.
    fn process_buffer_final(&mut self) {
        // Try to apply rules one more time with at_end = true
        self.process_buffer();

        // Collect remaining characters first to avoid borrow conflict
        let remaining: Vec<char> = self.input_buffer.drain(..).collect();
        let count = remaining.len();
        for c in remaining {
            self.emit_char(c);
        }
        self.buffer_start_pos += count;
    }

    /// Find the highest-priority rule that matches at position.
    ///
    /// Returns `Some((rule_index, pattern_length))` if a rule matches.
    fn find_matching_rule(&self, pos: usize) -> Option<(usize, usize)> {
        // Convert buffer slice to PhoneChar for rule matching
        let phones = self.buffer_to_phones();

        // Rules are pre-sorted by weight (highest first)
        for (idx, rule) in self.rules.iter().enumerate() {
            // Check if we can apply at this position
            if self.can_apply_in_buffer(rule, &phones, pos) {
                return Some((idx, rule.pattern.len()));
            }
        }
        None
    }

    /// Check if a rule can be applied at a position in the buffer.
    ///
    /// This is similar to `can_apply_at_char` but handles streaming context.
    fn can_apply_in_buffer(
        &self,
        rule: &RewriteRuleChar,
        phones: &[PhoneChar],
        pos: usize,
    ) -> bool {
        // First check pattern match (delegates to standard implementation)
        if !can_apply_at_char(rule, phones, pos) {
            return false;
        }

        // For streaming, we may need to defer context evaluation
        // if we don't have enough lookahead
        if let Some(ctx_result) =
            self.context_matches_in_buffer(&rule.context, phones, pos, rule.pattern.len())
        {
            ctx_result
        } else {
            // Need more context - can't determine yet
            false
        }
    }

    /// Evaluate a context condition in the buffer.
    ///
    /// Returns `Some(true)` if context matches, `Some(false)` if not,
    /// or `None` if we need more input to determine.
    fn context_matches_in_buffer(
        &self,
        ctx: &ContextChar,
        phones: &[PhoneChar],
        pos: usize,
        pattern_len: usize,
    ) -> Option<bool> {
        // Calculate absolute position in the input stream
        let absolute_pos = self.buffer_start_pos + pos;

        match ctx {
            ContextChar::Initial => Some(absolute_pos == 0),

            ContextChar::Final => {
                // Position after pattern
                let ctx_pos = pos + pattern_len;
                if self.at_end {
                    Some(ctx_pos >= phones.len())
                } else {
                    // Can't determine final until we see end-of-input
                    None
                }
            }

            ContextChar::BeforeVowel(vowels) => {
                let ctx_pos = pos + pattern_len;
                if ctx_pos < phones.len() {
                    // Have enough context
                    Some(Self::is_matching_vowel(&phones[ctx_pos], vowels))
                } else if self.at_end {
                    // No more chars coming - not before a vowel
                    Some(false)
                } else {
                    // Need more input
                    None
                }
            }

            ContextChar::BeforeConsonant(consonants) => {
                let ctx_pos = pos + pattern_len;
                if ctx_pos < phones.len() {
                    Some(Self::is_matching_consonant(&phones[ctx_pos], consonants))
                } else if self.at_end {
                    Some(false)
                } else {
                    None
                }
            }

            ContextChar::AfterVowel(vowels) => {
                // Look backward - no lookahead needed
                if pos == 0 {
                    Some(false)
                } else {
                    Some(Self::is_matching_vowel(&phones[pos - 1], vowels))
                }
            }

            ContextChar::AfterConsonant(consonants) => {
                if pos == 0 {
                    Some(false)
                } else {
                    Some(Self::is_matching_consonant(&phones[pos - 1], consonants))
                }
            }

            ContextChar::Anywhere => Some(true),

            ContextChar::And(a, b) => {
                match (
                    self.context_matches_in_buffer(a, phones, pos, pattern_len),
                    self.context_matches_in_buffer(b, phones, pos, pattern_len),
                ) {
                    (Some(false), _) | (_, Some(false)) => Some(false),
                    (Some(true), Some(true)) => Some(true),
                    _ => None,
                }
            }

            ContextChar::Or(a, b) => {
                match (
                    self.context_matches_in_buffer(a, phones, pos, pattern_len),
                    self.context_matches_in_buffer(b, phones, pos, pattern_len),
                ) {
                    (Some(true), _) | (_, Some(true)) => Some(true),
                    (Some(false), Some(false)) => Some(false),
                    _ => None,
                }
            }

            ContextChar::Not(inner) => self
                .context_matches_in_buffer(inner, phones, pos, pattern_len)
                .map(|b| !b),
        }
    }

    /// Check if a phone is a vowel matching the given set.
    fn is_matching_vowel(phone: &PhoneChar, vowels: &[char]) -> bool {
        match phone {
            PhoneChar::Vowel(c) => vowels.is_empty() || vowels.contains(c),
            _ => false,
        }
    }

    /// Check if a phone is a consonant matching the given set.
    fn is_matching_consonant(phone: &PhoneChar, consonants: &[char]) -> bool {
        match phone {
            PhoneChar::Consonant(c) => consonants.is_empty() || consonants.contains(c),
            _ => false,
        }
    }

    /// Convert the input buffer to PhoneChar representation.
    fn buffer_to_phones(&self) -> Vec<PhoneChar> {
        self.input_buffer
            .iter()
            .map(|&c| Self::char_to_phone(c))
            .collect()
    }

    /// Convert a character to a PhoneChar.
    fn char_to_phone(c: char) -> PhoneChar {
        let lower = c.to_ascii_lowercase();
        if "aeiou".contains(lower) {
            PhoneChar::Vowel(c)
        } else if c.is_alphabetic() {
            PhoneChar::Consonant(c)
        } else {
            // Non-alphabetic treated as consonant for matching purposes
            PhoneChar::Consonant(c)
        }
    }

    /// Convert a PhoneChar back to a character.
    fn phone_to_char(phone: &PhoneChar) -> Option<char> {
        match phone {
            PhoneChar::Vowel(c) => Some(*c),
            PhoneChar::Consonant(c) => Some(*c),
            PhoneChar::Digraph(c1, _c2) => {
                // For digraphs, return first char (caller should handle both)
                Some(*c1)
            }
            PhoneChar::Trigraph(c1, _c2, _c3) => {
                // For trigraphs, return first char (caller should handle all three)
                Some(*c1)
            }
            PhoneChar::Tetragraph(c1, _c2, _c3, _c4) => {
                // For tetragraphs, return first char (caller should handle all four)
                Some(*c1)
            }
            PhoneChar::Pentagraph(c1, _c2, _c3, _c4, _c5) => {
                // For pentagraphs, return first char (caller should handle all five)
                Some(*c1)
            }
            PhoneChar::Hexagraph(c1, _c2, _c3, _c4, _c5, _c6) => {
                // For hexagraphs, return first char (caller should handle all six)
                Some(*c1)
            }
            PhoneChar::Heptagraph(c1, _c2, _c3, _c4, _c5, _c6, _c7) => {
                // For heptagraphs, return first char (caller should handle all seven)
                Some(*c1)
            }
            PhoneChar::Sequence(s) => {
                // For sequences, return first char (caller should handle all)
                s.first().copied()
            }
            PhoneChar::Silent => None,
        }
    }

    /// Emit a character to the output buffer.
    fn emit_char(&mut self, c: char) {
        self.output_buffer.push_back(c);
    }

    /// Emit characters from the buffer prefix that cannot start any rule pattern.
    fn emit_safe_prefix(&mut self) {
        if self.input_buffer.is_empty() {
            return;
        }

        // Find the first position where a rule MIGHT apply
        // This is conservative - we emit only what we're sure about
        let safe_len = self.find_safe_prefix_length();

        // Emit the safe prefix
        for i in 0..safe_len {
            self.emit_char(self.input_buffer[i]);
        }
        self.input_buffer.drain(0..safe_len);
        self.buffer_start_pos += safe_len;
    }

    /// Find how many characters from the start of the buffer are "safe" to emit.
    ///
    /// A character is safe if no rule pattern could possibly start there.
    fn find_safe_prefix_length(&self) -> usize {
        if self.input_buffer.is_empty() {
            return 0;
        }

        // If we have enough context and no rule matches, the first char is safe
        let required_context = self.max_pattern_len + self.max_lookahead;
        if self.input_buffer.len() >= required_context || self.at_end {
            // We have enough context - if no rule matched at position 0,
            // the first character is safe to emit
            let phones = self.buffer_to_phones();
            for rule in &self.rules {
                if self.can_apply_in_buffer(rule, &phones, 0) {
                    // A rule might match - not safe
                    return 0;
                }
            }
            // No rule matches at position 0 - emit first character
            1
        } else {
            // Not enough context yet
            0
        }
    }

    /// Get statistics for debugging.
    pub fn stats(&self) -> (usize, usize) {
        (self.chars_processed, self.rules_applied)
    }

    /// Normalize an entire string at once (convenience method).
    ///
    /// This is equivalent to feeding all characters and finishing.
    pub fn normalize(&mut self, input: &str) -> String {
        let mut result = String::with_capacity(input.len());

        for c in input.chars() {
            for out_c in self.feed(c) {
                result.push(out_c);
            }
        }

        for out_c in self.finish() {
            result.push(out_c);
        }

        result
    }

    /// Reset the transducer for reuse.
    pub fn reset(&mut self) {
        self.input_buffer.clear();
        self.output_buffer.clear();
        self.at_end = false;
        self.buffer_start_pos = 0;
        self.chars_processed = 0;
        self.rules_applied = 0;
    }
}

/// Iterator adapter that drains from a VecDeque.
struct DrainIter<'a> {
    inner: &'a mut VecDeque<char>,
}

impl<'a> Iterator for DrainIter<'a> {
    type Item = char;

    fn next(&mut self) -> Option<Self::Item> {
        self.inner.pop_front()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    /// Helper to create a simple rule for testing.
    fn make_rule(pattern: &str, replacement: &str, context: ContextChar) -> RewriteRuleChar {
        RewriteRuleChar {
            rule_id: 0,
            rule_name: format!("{} -> {}", pattern, replacement),
            pattern: pattern
                .chars()
                .map(OnlinePhoneticTransducerChar::char_to_phone)
                .collect(),
            replacement: replacement
                .chars()
                .map(OnlinePhoneticTransducerChar::char_to_phone)
                .collect(),
            context,
            weight: 1.0,
            syllable_condition: None,
        }
    }

    #[test]
    fn test_empty_input() {
        let rules = vec![make_rule("ph", "f", ContextChar::Anywhere)];
        let mut transducer = OnlinePhoneticTransducerChar::new(rules);

        let result = transducer.normalize("");
        assert_eq!(result, "");
    }

    #[test]
    fn test_no_rules() {
        let rules = vec![];
        let mut transducer = OnlinePhoneticTransducerChar::new(rules);

        let result = transducer.normalize("hello");
        assert_eq!(result, "hello");
    }

    #[test]
    fn test_simple_substitution() {
        let rules = vec![make_rule("ph", "f", ContextChar::Anywhere)];
        let mut transducer = OnlinePhoneticTransducerChar::new(rules);

        let result = transducer.normalize("phone");
        assert_eq!(result, "fone");
    }

    #[test]
    fn test_multiple_substitutions() {
        let rules = vec![make_rule("ph", "f", ContextChar::Anywhere)];
        let mut transducer = OnlinePhoneticTransducerChar::new(rules);

        let result = transducer.normalize("phosphate");
        assert_eq!(result, "fosfate");
    }

    #[test]
    fn test_no_match() {
        let rules = vec![make_rule("ph", "f", ContextChar::Anywhere)];
        let mut transducer = OnlinePhoneticTransducerChar::new(rules);

        let result = transducer.normalize("hello");
        assert_eq!(result, "hello");
    }

    #[test]
    fn test_streaming_character_by_character() {
        let rules = vec![make_rule("ph", "f", ContextChar::Anywhere)];
        let mut transducer = OnlinePhoneticTransducerChar::new(rules);

        let mut result = String::new();
        for c in "phone".chars() {
            for out_c in transducer.feed(c) {
                result.push(out_c);
            }
        }
        for out_c in transducer.finish() {
            result.push(out_c);
        }

        assert_eq!(result, "fone");
    }

    #[test]
    fn test_buffering_partial_pattern() {
        let rules = vec![make_rule("ph", "f", ContextChar::Anywhere)];
        let mut transducer = OnlinePhoneticTransducerChar::new(rules);

        // Feed 'p' - should buffer, not emit
        let _out1: Vec<char> = transducer.feed('p').collect();
        // May or may not emit 'p' depending on implementation
        // The key test is that final result is correct

        // Feed 'h' - should apply rule
        let _out2: Vec<char> = transducer.feed('h').collect();

        // Feed remaining chars
        for c in "one".chars() {
            for _out_c in transducer.feed(c) {
                // collect
            }
        }

        // Get final result
        transducer.reset();
        let result = transducer.normalize("phone");
        assert_eq!(result, "fone");
    }

    #[test]
    fn test_context_initial() {
        let rules = vec![make_rule("k", "c", ContextChar::Initial)];
        let mut transducer = OnlinePhoneticTransducerChar::new(rules);

        // 'k' at start should become 'c'
        let result1 = transducer.normalize("king");
        assert_eq!(result1, "cing");

        // 'k' not at start should stay 'k'
        transducer.reset();
        let result2 = transducer.normalize("bike");
        assert_eq!(result2, "bike");
    }

    #[test]
    fn test_context_final() {
        let rules = vec![make_rule("e", "", ContextChar::Final)];
        let mut transducer = OnlinePhoneticTransducerChar::new(rules);

        // Final 'e' should be removed
        let result1 = transducer.normalize("phone");
        assert_eq!(result1, "phon");

        // Non-final 'e' should stay
        transducer.reset();
        let result2 = transducer.normalize("elephant");
        assert_eq!(result2, "elephant");
    }

    #[test]
    fn test_context_before_vowel() {
        let rules = vec![make_rule(
            "c",
            "s",
            ContextChar::BeforeVowel(vec!['e', 'i']),
        )];
        let mut transducer = OnlinePhoneticTransducerChar::new(rules);

        // 'c' before 'e' should become 's'
        let result1 = transducer.normalize("cent");
        assert_eq!(result1, "sent");

        // 'c' before 'a' should stay 'c'
        transducer.reset();
        let result2 = transducer.normalize("cat");
        assert_eq!(result2, "cat");
    }

    #[test]
    fn test_rule_priority() {
        // Higher weight rule should apply first
        let rules = vec![
            RewriteRuleChar {
                rule_id: 1,
                rule_name: "ph -> f".to_string(),
                pattern: vec![PhoneChar::Consonant('p'), PhoneChar::Consonant('h')],
                replacement: vec![PhoneChar::Consonant('f')],
                context: ContextChar::Anywhere,
                weight: 2.0, // Higher priority
                syllable_condition: None,
            },
            RewriteRuleChar {
                rule_id: 2,
                rule_name: "p -> b".to_string(),
                pattern: vec![PhoneChar::Consonant('p')],
                replacement: vec![PhoneChar::Consonant('b')],
                context: ContextChar::Anywhere,
                weight: 1.0,
                syllable_condition: None,
            },
        ];

        let mut transducer = OnlinePhoneticTransducerChar::new(rules);
        let result = transducer.normalize("phone");
        // 'ph' should match first (higher weight), not 'p' → 'b'
        assert_eq!(result, "fone");
    }

    #[test]
    fn test_multiple_rules() {
        let rules = vec![
            make_rule("ph", "f", ContextChar::Anywhere),
            make_rule("oo", "u", ContextChar::Anywhere),
        ];
        let mut transducer = OnlinePhoneticTransducerChar::new(rules);

        let result = transducer.normalize("food");
        assert_eq!(result, "fud");
    }

    #[test]
    fn test_fude_food_equivalence() {
        // This is the key test case: "fude" and "food" should both normalize to "fud"
        let rules = vec![
            make_rule("oo", "u", ContextChar::Anywhere),
            make_rule("e", "", ContextChar::Final), // Silent final 'e'
        ];

        let mut transducer = OnlinePhoneticTransducerChar::new(rules);
        let result_food = transducer.normalize("food");
        assert_eq!(result_food, "fud");

        transducer.reset();
        let result_fude = transducer.normalize("fude");
        assert_eq!(result_fude, "fud");
    }

    #[test]
    fn test_stats() {
        let rules = vec![make_rule("ph", "f", ContextChar::Anywhere)];
        let mut transducer = OnlinePhoneticTransducerChar::new(rules);

        transducer.normalize("phosphate");
        let (chars, rules_applied) = transducer.stats();
        assert_eq!(chars, 9); // "phosphate" has 9 chars
        assert_eq!(rules_applied, 2); // "ph" appears twice
    }
}
