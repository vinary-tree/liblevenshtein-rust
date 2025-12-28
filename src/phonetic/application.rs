//! Rule application for phonetic rewrite systems.
//!
//! This module implements the rule application logic for phonetic rewrite rules,
//! directly translated from the Coq/Rocq verification in
//! `docs/verification/phonetic/rewrite_rules.v`.
//!
//! # Functions
//!
//! - [`apply_rule_at`] / [`apply_rule_at_char`] - Apply a rule at a specific position
//! - [`find_first_match`] / [`find_first_match_char`] - Find first matching position
//! - [`apply_rules_seq`] / [`apply_rules_seq_char`] - Sequential rule application
//!
//! # Constants
//!
//! - [`MAX_EXPANSION_FACTOR`] - Maximum string expansion (from Theorem 2)
//!
//! # Formal Guarantees
//!
//! These functions implement algorithms with proven properties:
//!
//! - **Bounded Expansion** (Theorem 2, `zompist_rules.v:425`):
//!   Output length ≤ input length + [`MAX_EXPANSION_FACTOR`]
//!
//! - **Termination** (Theorem 4, `zompist_rules.v:569`):
//!   Sequential application always terminates with sufficient fuel
//!
//! - **Idempotence** (Theorem 5, `zompist_rules.v:615`):
//!   Fixed points remain unchanged under further application

use super::matching::{context_matches, context_matches_char, pattern_matches_at, pattern_matches_at_char};
use super::syllable::evaluate_syllable_expr;
use super::types::{Context, ContextChar, Phone, PhoneChar, RewriteRule, RewriteRuleChar};
use std::collections::HashSet;

/// Convert a slice of Phone units to a string for syllable evaluation.
/// Extracts the characters from vowels, consonants, digraphs, trigraphs, tetragraphs,
/// pentagraphs, hexagraphs, heptagraphs, and sequences.
fn phones_to_string(phones: &[Phone]) -> String {
    let mut result = String::with_capacity(phones.len() * 7);
    for phone in phones {
        match phone {
            Phone::Vowel(c) | Phone::Consonant(c) => result.push(*c as char),
            Phone::Digraph(c1, c2) => {
                result.push(*c1 as char);
                result.push(*c2 as char);
            }
            Phone::Trigraph(c1, c2, c3) => {
                result.push(*c1 as char);
                result.push(*c2 as char);
                result.push(*c3 as char);
            }
            Phone::Tetragraph(c1, c2, c3, c4) => {
                result.push(*c1 as char);
                result.push(*c2 as char);
                result.push(*c3 as char);
                result.push(*c4 as char);
            }
            Phone::Pentagraph(c1, c2, c3, c4, c5) => {
                result.push(*c1 as char);
                result.push(*c2 as char);
                result.push(*c3 as char);
                result.push(*c4 as char);
                result.push(*c5 as char);
            }
            Phone::Hexagraph(c1, c2, c3, c4, c5, c6) => {
                result.push(*c1 as char);
                result.push(*c2 as char);
                result.push(*c3 as char);
                result.push(*c4 as char);
                result.push(*c5 as char);
                result.push(*c6 as char);
            }
            Phone::Heptagraph(c1, c2, c3, c4, c5, c6, c7) => {
                result.push(*c1 as char);
                result.push(*c2 as char);
                result.push(*c3 as char);
                result.push(*c4 as char);
                result.push(*c5 as char);
                result.push(*c6 as char);
                result.push(*c7 as char);
            }
            Phone::Sequence(s) => {
                for c in s {
                    result.push(*c as char);
                }
            }
            Phone::Silent => {}
        }
    }
    result
}

/// Convert a slice of PhoneChar units to a string for syllable evaluation.
/// Extracts the characters from vowels, consonants, digraphs, trigraphs, tetragraphs,
/// pentagraphs, hexagraphs, heptagraphs, and sequences.
fn phone_chars_to_string(phones: &[PhoneChar]) -> String {
    let mut result = String::with_capacity(phones.len() * 7);
    for phone in phones {
        match phone {
            PhoneChar::Vowel(c) | PhoneChar::Consonant(c) => result.push(*c),
            PhoneChar::Digraph(c1, c2) => {
                result.push(*c1);
                result.push(*c2);
            }
            PhoneChar::Trigraph(c1, c2, c3) => {
                result.push(*c1);
                result.push(*c2);
                result.push(*c3);
            }
            PhoneChar::Tetragraph(c1, c2, c3, c4) => {
                result.push(*c1);
                result.push(*c2);
                result.push(*c3);
                result.push(*c4);
            }
            PhoneChar::Pentagraph(c1, c2, c3, c4, c5) => {
                result.push(*c1);
                result.push(*c2);
                result.push(*c3);
                result.push(*c4);
                result.push(*c5);
            }
            PhoneChar::Hexagraph(c1, c2, c3, c4, c5, c6) => {
                result.push(*c1);
                result.push(*c2);
                result.push(*c3);
                result.push(*c4);
                result.push(*c5);
                result.push(*c6);
            }
            PhoneChar::Heptagraph(c1, c2, c3, c4, c5, c6, c7) => {
                result.push(*c1);
                result.push(*c2);
                result.push(*c3);
                result.push(*c4);
                result.push(*c5);
                result.push(*c6);
                result.push(*c7);
            }
            PhoneChar::Sequence(s) => {
                for c in s {
                    result.push(*c);
                }
            }
            PhoneChar::Silent => {}
        }
    }
    result
}

#[cfg(feature = "perf-instrumentation")]
use std::sync::atomic::{AtomicUsize, Ordering};

#[cfg(feature = "perf-instrumentation")]
static BYTES_COPIED: AtomicUsize = AtomicUsize::new(0);

#[cfg(feature = "perf-instrumentation")]
static ALLOCATIONS: AtomicUsize = AtomicUsize::new(0);

#[cfg(feature = "perf-instrumentation")]
pub fn get_perf_stats() -> (usize, usize) {
    (
        BYTES_COPIED.load(Ordering::Relaxed),
        ALLOCATIONS.load(Ordering::Relaxed),
    )
}

#[cfg(feature = "perf-instrumentation")]
pub fn reset_perf_stats() {
    BYTES_COPIED.store(0, Ordering::Relaxed);
    ALLOCATIONS.store(0, Ordering::Relaxed);
}

/// Maximum expansion factor for phonetic rewrite rules.
///
/// **Formal Specification**: Theorem 2, `docs/verification/phonetic/zompist_rules.v:425`
///
/// This constant bounds the maximum string growth from any single rule application.
/// For all well-formed rules in the zompist rule set:
///
/// ```text
/// length(output) ≤ length(input) + MAX_EXPANSION_FACTOR
/// ```
///
/// The value 20 is proven sufficient for all 56 zompist rules.
pub const MAX_EXPANSION_FACTOR: usize = 20;

/// Maximum total expansion allowed during normalization.
///
/// This is a defensive safeguard against runaway expansion caused by
/// pathological rule interactions (e.g., expansion rules that create
/// patterns matching other expansion rules).
///
/// If the result exceeds `input.len() + MAX_TOTAL_EXPANSION`, normalization
/// is aborted and the current state is returned with a warning.
pub const MAX_TOTAL_EXPANSION: usize = 100;

// ============================================================================
// Position-dependent context checks (for position skipping optimization)
// ============================================================================

/// Check if any rules have position-dependent contexts (byte-level).
///
/// **Formal Specification**: `docs/verification/phonetic/position_skipping_proof.v:3453`
///
/// Returns `true` if position skipping is UNSAFE (any rule uses `Context::Final`).
/// Returns `false` if position skipping is SAFE.
///
/// # Position Skipping Safety
///
/// Position skipping optimization starts the next rule search from the last match
/// position instead of position 0. This is SAFE when no rules use `Context::Final`
/// because:
///
/// - All other contexts (Initial, BeforeVowel, AfterConsonant, etc.) depend only
///   on local structure, not on the overall string length
/// - `Final` depends on `pos == s.len()`, which changes when the string is shortened
///
/// # Example
///
/// ```rust,ignore
/// use liblevenshtein::phonetic::{has_position_dependent_rules, orthography_rules, phonetic_rules};
///
/// // orthography_rules contains rule_silent_e_final with Context::Final
/// assert!(has_position_dependent_rules(&orthography_rules()));
///
/// // phonetic_rules has no Final context - safe for optimization
/// assert!(!has_position_dependent_rules(&phonetic_rules()));
/// ```
#[inline]
pub fn has_position_dependent_rules(rules: &[RewriteRule]) -> bool {
    rules.iter().any(|r| r.context.is_position_dependent())
}

/// Check if any rules have position-dependent contexts (character-level).
///
/// **Formal Specification**: `docs/verification/phonetic/position_skipping_proof.v:3453`
///
/// Character-level variant of [`has_position_dependent_rules`].
#[inline]
pub fn has_position_dependent_rules_char(rules: &[RewriteRuleChar]) -> bool {
    rules.iter().any(|r| r.context.is_position_dependent())
}

// ============================================================================
// Rule application (byte-level)
// ============================================================================

/// Check if a rewrite rule can be applied at a specific position (byte-level).
///
/// **Performance Optimization**: This function checks rule applicability without
/// allocating a result vector, making it much faster for position scanning.
///
/// # Arguments
///
/// - `rule` - The rewrite rule to check
/// - `s` - The phonetic string
/// - `pos` - The position to check
///
/// # Returns
///
/// - `true` if the rule can be applied at the position
/// - `false` otherwise
#[inline]
pub fn can_apply_at(rule: &RewriteRule, s: &[Phone], pos: usize) -> bool {
    // Check pattern matches first (quick rejection)
    if !pattern_matches_at(&rule.pattern, s, pos) {
        return false;
    }

    // Check context - context_matches now handles position computation internally
    // based on context type (Initial/After* check at start, Final/Before* check at end)
    if !context_matches(&rule.context, s, pos, rule.pattern.len()) {
        return false;
    }

    // Check syllable condition if present
    if let Some(ref syllable_expr) = rule.syllable_condition {
        // Convert Phone slice to string for syllable evaluation
        let word = phones_to_string(s);
        if !evaluate_syllable_expr(syllable_expr, &word, pos) {
            return false;
        }
    }

    true
}

/// Apply a rewrite rule at a specific position if possible (byte-level).
///
/// **Formal Specification**: `docs/verification/phonetic/rewrite_rules.v:177-187`
///
/// Attempts to apply a rule at the given position in the phonetic string.
/// Returns `Some(new_string)` if the rule applies, `None` otherwise.
///
/// # Algorithm
///
/// 1. Check if the context is satisfied at the position
/// 2. Check if the pattern matches at the position
/// 3. If both conditions hold, replace the pattern with the replacement
///
/// # Arguments
///
/// - `rule` - The rewrite rule to apply
/// - `s` - The phonetic string
/// - `pos` - The position to attempt application
///
/// # Returns
///
/// - `Some(new_string)` if the rule applies at the position
/// - `None` if the rule does not apply
///
/// # Examples
///
/// ```rust,ignore
/// use liblevenshtein::phonetic::{apply_rule_at, Phone, Context, RewriteRule};
///
/// let rule = RewriteRule {
///     rule_id: 1,
///     rule_name: "gh → f".to_string(),
///     pattern: vec![Phone::Consonant(b'g'), Phone::Consonant(b'h')],
///     replacement: vec![Phone::Consonant(b'f')],
///     context: Context::Anywhere,
///     weight: 0.15,
///     syllable_condition: None,
/// };
///
/// let s = vec![
///     Phone::Vowel(b'e'),
///     Phone::Consonant(b'g'),
///     Phone::Consonant(b'h'),
/// ];
///
/// let result = apply_rule_at(&rule, &s, 1);
/// assert_eq!(result, Some(vec![Phone::Vowel(b'e'), Phone::Consonant(b'f')]));
/// ```
pub fn apply_rule_at(rule: &RewriteRule, s: &[Phone], pos: usize) -> Option<Vec<Phone>> {
    // Check if rule can be applied (no allocation)
    if !can_apply_at(rule, s, pos) {
        return None;
    }

    // Build result: prefix + replacement + suffix
    let mut result = Vec::with_capacity(s.len() + MAX_EXPANSION_FACTOR);

    #[cfg(feature = "perf-instrumentation")]
    {
        ALLOCATIONS.fetch_add(1, Ordering::Relaxed);
        // Count bytes copied: prefix + replacement + suffix
        let bytes_copied = pos + rule.replacement.len() + (s.len() - pos - rule.pattern.len());
        BYTES_COPIED.fetch_add(bytes_copied, Ordering::Relaxed);
    }

    result.extend_from_slice(&s[..pos]);
    result.extend_from_slice(&rule.replacement);
    result.extend_from_slice(&s[(pos + rule.pattern.len())..]);

    Some(result)
}

/// Find the first position where a rule can be applied (byte-level).
///
/// **Formal Specification**: `docs/verification/phonetic/rewrite_rules.v:190-198`
///
/// Scans the string from left to right to find the first position where
/// the rule can be applied.
///
/// # Arguments
///
/// - `rule` - The rewrite rule to match
/// - `s` - The phonetic string
///
/// # Returns
///
/// - `Some(pos)` if the rule can be applied at position `pos`
/// - `None` if the rule cannot be applied anywhere
pub fn find_first_match(rule: &RewriteRule, s: &[Phone]) -> Option<usize> {
    find_first_match_from(rule, s, 0)
}

/// Find the first position where a rule can be applied, starting from a given position.
///
/// This is an optimization helper for sequential rule application.
///
/// # Arguments
///
/// - `rule` - The rewrite rule to match
/// - `s` - The phonetic string
/// - `start_pos` - The position to start scanning from (0-based)
///
/// # Returns
///
/// - `Some(pos)` if the rule can be applied at position `pos >= start_pos`
/// - `None` if the rule cannot be applied anywhere from `start_pos` onward
#[inline]
pub fn find_first_match_from(rule: &RewriteRule, s: &[Phone], start_pos: usize) -> Option<usize> {
    // Try each position from start_pos to s.len()
    // Optimization: use can_apply_at() to avoid allocating vectors during search
    for pos in start_pos..=s.len() {
        if can_apply_at(rule, s, pos) {
            return Some(pos);
        }
    }
    None
}

/// Apply a list of rules sequentially until fixed point or fuel exhausted (byte-level).
///
/// **Formal Specification**: `docs/verification/phonetic/rewrite_rules.v:203-227`
///
/// Applies rules in order, restarting from the first rule after each successful
/// application, until no rules can be applied or fuel is exhausted.
///
/// # Formal Guarantees
///
/// - **Termination** (Theorem 4, `zompist_rules.v:569`):
///   Always terminates with sufficient fuel
///
/// - **Idempotence** (Theorem 5, `zompist_rules.v:615`):
///   Result is a fixed point (applying rules again produces the same result)
///
/// # Algorithm
///
/// ```text
/// loop:
///   for each rule r in rules:
///     if r can be applied:
///       apply r
///       restart loop
///   no rules applied → return fixed point
/// ```
///
/// # Arguments
///
/// - `rules` - The list of rewrite rules to apply
/// - `s` - The phonetic string
/// - `fuel` - Maximum number of iterations (prevents infinite loops)
///
/// # Returns
///
/// - `Some(result)` with the transformed string
/// - `None` if fuel is exhausted (shouldn't happen with sufficient fuel)
///
/// # Fuel Calculation
///
/// Sufficient fuel is:
/// ```text
/// fuel >= length(s) * length(rules) * MAX_EXPANSION_FACTOR
/// ```
///
/// For practical use, `fuel = s.len() * rules.len() * 100` is recommended.
pub fn apply_rules_seq(rules: &[RewriteRule], s: &[Phone], fuel: usize) -> Option<Vec<Phone>> {
    let mut current = s.to_vec();
    let original_len = s.len();

    #[cfg(feature = "perf-instrumentation")]
    {
        ALLOCATIONS.fetch_add(1, Ordering::Relaxed);
        BYTES_COPIED.fetch_add(s.len(), Ordering::Relaxed);
    }

    let mut remaining_fuel = fuel;

    loop {
        if remaining_fuel == 0 {
            // Out of fuel - return current state
            return Some(current);
        }

        let mut applied = false;

        // Try each rule in order
        for rule in rules {
            if let Some(pos) = find_first_match(rule, &current) {
                if let Some(new_s) = apply_rule_at(rule, &current, pos) {
                    // Defensive safeguard: abort if expansion exceeds limit
                    if new_s.len() > original_len + MAX_TOTAL_EXPANSION {
                        eprintln!(
                            "[phonetic] Warning: Normalization exceeded expansion limit \
                             ({} > {} + {}). Returning current state to prevent runaway expansion. \
                             Consider revising rules to avoid pathological interactions.",
                            new_s.len(),
                            original_len,
                            MAX_TOTAL_EXPANSION
                        );
                        return Some(current);
                    }
                    current = new_s;
                    remaining_fuel -= 1;
                    applied = true;
                    break; // Restart from first rule
                }
            }
        }

        if !applied {
            // Fixed point reached - no rules can be applied
            return Some(current);
        }
    }
}

// ============================================================================
// Cycle-aware rule application with equivalence set recovery (byte-level)
// ============================================================================

/// Result of applying phonetic rules with cycle awareness.
///
/// This enum distinguishes between three outcomes:
/// - `FixedPoint`: No more rules can be applied (normal termination)
/// - `Cycle`: A cycle was detected, returning all equivalent forms
/// - `FuelExhausted`: Ran out of fuel before termination (shouldn't happen with sufficient fuel)
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum NormalizationResult {
    /// Reached a fixed point where no rules can be applied.
    FixedPoint(Vec<Phone>),

    /// Detected a cycle - all forms in the set are equivalent.
    /// The cycle may include forms that are not directly reachable from each other,
    /// but all appeared on the path to detecting the cycle.
    Cycle(HashSet<Vec<Phone>>),

    /// Fuel exhausted before reaching a fixed point or detecting a cycle.
    FuelExhausted(Vec<Phone>),
}

impl NormalizationResult {
    /// Get a canonical representative form.
    ///
    /// For fixed points and fuel exhaustion, returns the result.
    /// For cycles, returns the shortest form.
    pub fn canonical(&self) -> Vec<Phone> {
        match self {
            NormalizationResult::FixedPoint(s) => s.clone(),
            NormalizationResult::FuelExhausted(s) => s.clone(),
            NormalizationResult::Cycle(set) => {
                // Pick shortest form
                set.iter()
                    .min_by_key(|v| v.len())
                    .cloned()
                    .unwrap_or_default()
            }
        }
    }

    /// Get all equivalent forms.
    ///
    /// For fixed points and fuel exhaustion, returns a singleton set.
    /// For cycles, returns all forms that appeared in the cycle.
    pub fn all_forms(&self) -> HashSet<Vec<Phone>> {
        match self {
            NormalizationResult::FixedPoint(s) => {
                let mut set = HashSet::new();
                set.insert(s.clone());
                set
            }
            NormalizationResult::FuelExhausted(s) => {
                let mut set = HashSet::new();
                set.insert(s.clone());
                set
            }
            NormalizationResult::Cycle(set) => set.clone(),
        }
    }

    /// Returns `true` if a cycle was detected.
    pub fn is_cycle(&self) -> bool {
        matches!(self, NormalizationResult::Cycle(_))
    }

    /// Returns `true` if a fixed point was reached.
    pub fn is_fixed_point(&self) -> bool {
        matches!(self, NormalizationResult::FixedPoint(_))
    }
}

/// Apply rules with cycle detection and equivalence set recovery (byte-level).
///
/// This function tracks all intermediate forms and detects when a previously
/// seen form is reached again (indicating a cycle). When a cycle is detected,
/// all forms seen on the path are returned as an equivalence set.
///
/// # Algorithm
///
/// ```text
/// seen = {s}
/// current = s
/// loop:
///   for each rule r in rules:
///     if r can be applied:
///       new = apply r to current
///       if new in seen:
///         log warning and return Cycle(seen)
///       seen.insert(new)
///       current = new
///       restart loop
///   no rules applied → return FixedPoint(current)
/// ```
///
/// # Arguments
///
/// - `rules` - The list of rewrite rules to apply
/// - `s` - The phonetic string
/// - `fuel` - Maximum number of iterations (prevents infinite loops in degenerate cases)
///
/// # Returns
///
/// - `FixedPoint(result)` if no rules can be applied
/// - `Cycle(set)` if a previously seen form is reached (all forms in cycle)
/// - `FuelExhausted(current)` if fuel runs out (rare with proper fuel calculation)
///
/// # Example
///
/// ```rust,ignore
/// // With rules u -> you and you -> u:
/// // Input "u" will detect a cycle and return {u, you}
/// match apply_rules_with_cycle_detection(&rules, &phones, fuel) {
///     NormalizationResult::Cycle(forms) => {
///         // forms contains all equivalent normalizations
///         let canonical = forms.iter().min_by_key(|f| f.len()).unwrap();
///     }
///     NormalizationResult::FixedPoint(result) => {
///         // Normal case - no cycle
///     }
///     _ => {}
/// }
/// ```
pub fn apply_rules_with_cycle_detection(
    rules: &[RewriteRule],
    s: &[Phone],
    fuel: usize,
) -> NormalizationResult {
    let mut current = s.to_vec();
    let mut seen: HashSet<Vec<Phone>> = HashSet::new();
    let mut remaining_fuel = fuel;

    seen.insert(current.clone());

    loop {
        if remaining_fuel == 0 {
            return NormalizationResult::FuelExhausted(current);
        }

        let mut applied = false;

        for rule in rules {
            if let Some(pos) = find_first_match(rule, &current) {
                if let Some(new_s) = apply_rule_at(rule, &current, pos) {
                    // Check for cycle BEFORE updating current
                    if seen.contains(&new_s) {
                        // Cycle detected! Log warning and return all forms
                        eprintln!(
                            "[phonetic] Warning: Cycle detected in rule application. \
                             {} equivalent forms found and will all be indexed. \
                             Consider revising rules to avoid cycles.",
                            seen.len()
                        );
                        return NormalizationResult::Cycle(seen);
                    }

                    seen.insert(new_s.clone());
                    current = new_s;
                    remaining_fuel -= 1;
                    applied = true;
                    break; // Restart from first rule
                }
            }
        }

        if !applied {
            return NormalizationResult::FixedPoint(current);
        }
    }
}

/// Apply rules with cycle detection (character-level variant).
///
/// See [`apply_rules_with_cycle_detection`] for documentation.
pub fn apply_rules_with_cycle_detection_char(
    rules: &[RewriteRuleChar],
    s: &[PhoneChar],
    fuel: usize,
) -> NormalizationResultChar {
    let mut current = s.to_vec();
    let mut seen: HashSet<Vec<PhoneChar>> = HashSet::new();
    let mut remaining_fuel = fuel;

    seen.insert(current.clone());

    loop {
        if remaining_fuel == 0 {
            return NormalizationResultChar::FuelExhausted(current);
        }

        let mut applied = false;

        for rule in rules {
            if let Some(pos) = find_first_match_char(rule, &current) {
                if let Some(new_s) = apply_rule_at_char(rule, &current, pos) {
                    if seen.contains(&new_s) {
                        eprintln!(
                            "[phonetic] Warning: Cycle detected in rule application. \
                             {} equivalent forms found and will all be indexed. \
                             Consider revising rules to avoid cycles.",
                            seen.len()
                        );
                        return NormalizationResultChar::Cycle(seen);
                    }

                    seen.insert(new_s.clone());
                    current = new_s;
                    remaining_fuel -= 1;
                    applied = true;
                    break;
                }
            }
        }

        if !applied {
            return NormalizationResultChar::FixedPoint(current);
        }
    }
}

/// Result of applying phonetic rules with cycle awareness (character-level).
///
/// See [`NormalizationResult`] for documentation.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum NormalizationResultChar {
    /// Reached a fixed point where no rules can be applied.
    FixedPoint(Vec<PhoneChar>),

    /// Detected a cycle - all forms in the set are equivalent.
    Cycle(HashSet<Vec<PhoneChar>>),

    /// Fuel exhausted before reaching a fixed point or detecting a cycle.
    FuelExhausted(Vec<PhoneChar>),
}

impl NormalizationResultChar {
    /// Get a canonical representative form.
    pub fn canonical(&self) -> Vec<PhoneChar> {
        match self {
            NormalizationResultChar::FixedPoint(s) => s.clone(),
            NormalizationResultChar::FuelExhausted(s) => s.clone(),
            NormalizationResultChar::Cycle(set) => {
                set.iter()
                    .min_by_key(|v| v.len())
                    .cloned()
                    .unwrap_or_default()
            }
        }
    }

    /// Get all equivalent forms.
    pub fn all_forms(&self) -> HashSet<Vec<PhoneChar>> {
        match self {
            NormalizationResultChar::FixedPoint(s) => {
                let mut set = HashSet::new();
                set.insert(s.clone());
                set
            }
            NormalizationResultChar::FuelExhausted(s) => {
                let mut set = HashSet::new();
                set.insert(s.clone());
                set
            }
            NormalizationResultChar::Cycle(set) => set.clone(),
        }
    }

    /// Returns `true` if a cycle was detected.
    pub fn is_cycle(&self) -> bool {
        matches!(self, NormalizationResultChar::Cycle(_))
    }

    /// Returns `true` if a fixed point was reached.
    pub fn is_fixed_point(&self) -> bool {
        matches!(self, NormalizationResultChar::FixedPoint(_))
    }
}

// ============================================================================
// Optimized rule application with conditional position skipping (byte-level)
// ============================================================================

/// Apply rules with position skipping optimization enabled (byte-level).
///
/// **Formal Specification**: `docs/verification/phonetic/position_skipping_proof.v`
///
/// **SAFETY**: This function MUST only be called when no rules use `Context::Final`.
/// Verify with [`has_position_dependent_rules`] before calling this function directly.
///
/// # Algorithm
///
/// ```text
/// last_pos = 0
/// loop:
///   for each rule r in rules:
///     if r can be applied at pos >= last_pos:
///       apply r at pos
///       last_pos = pos  // Key optimization: skip positions [0, last_pos)
///       restart loop
///   no rules applied → return fixed point
/// ```
///
/// # Performance
///
/// Position skipping reduces redundant position checks. When rules repeatedly
/// apply near the same position, this can significantly reduce iterations.
///
/// # Arguments
///
/// - `rules` - The list of rewrite rules (MUST NOT contain `Context::Final`)
/// - `s` - The phonetic string
/// - `fuel` - Maximum number of iterations
///
/// # Returns
///
/// - `Some(result)` with the transformed string
/// - `None` if fuel is exhausted
pub fn apply_rules_seq_optimized(
    rules: &[RewriteRule],
    s: &[Phone],
    fuel: usize,
) -> Option<Vec<Phone>> {
    debug_assert!(
        !has_position_dependent_rules(rules),
        "apply_rules_seq_optimized requires no position-dependent rules (Context::Final)"
    );

    let mut current = s.to_vec();

    #[cfg(feature = "perf-instrumentation")]
    {
        ALLOCATIONS.fetch_add(1, Ordering::Relaxed);
        BYTES_COPIED.fetch_add(s.len(), Ordering::Relaxed);
    }

    let mut remaining_fuel = fuel;
    let mut last_pos: usize = 0; // Position skipping: start from here

    loop {
        if remaining_fuel == 0 {
            // Out of fuel - return current state
            return Some(current);
        }

        let mut applied = false;

        // Try each rule in order, starting from last_pos
        for rule in rules {
            // Key optimization: search from last_pos instead of 0
            if let Some(pos) = find_first_match_from(rule, &current, last_pos) {
                if let Some(new_s) = apply_rule_at(rule, &current, pos) {
                    current = new_s;
                    remaining_fuel -= 1;
                    last_pos = pos; // Next iteration starts from here
                    applied = true;
                    break; // Restart from first rule
                }
            }
        }

        if !applied {
            // Fixed point reached - no rules can be applied
            return Some(current);
        }
    }
}

// ============================================================================
// Rule application (character-level)
// ============================================================================

/// Check if a rewrite rule can be applied at a specific position (character-level).
///
/// **Performance Optimization**: This function checks rule applicability without
/// allocating a result vector, making it much faster for position scanning.
///
/// # Arguments
///
/// - `rule` - The rewrite rule to check
/// - `s` - The phonetic string
/// - `pos` - The position to check
///
/// # Returns
///
/// - `true` if the rule can be applied at the position
/// - `false` otherwise
#[inline]
pub fn can_apply_at_char(rule: &RewriteRuleChar, s: &[PhoneChar], pos: usize) -> bool {
    // Check pattern matches first (quick rejection)
    if !pattern_matches_at_char(&rule.pattern, s, pos) {
        return false;
    }

    // Check context - context_matches_char now handles position computation internally
    // based on context type (Initial/After* check at start, Final/Before* check at end)
    if !context_matches_char(&rule.context, s, pos, rule.pattern.len()) {
        return false;
    }

    // Check syllable condition if present
    if let Some(ref syllable_expr) = rule.syllable_condition {
        // Convert PhoneChar slice to string for syllable evaluation
        let word = phone_chars_to_string(s);
        if !evaluate_syllable_expr(syllable_expr, &word, pos) {
            return false;
        }
    }

    true
}

/// Apply a rewrite rule at a specific position if possible (character-level).
///
/// **Formal Specification**: `docs/verification/phonetic/rewrite_rules.v:177-187`
///
/// This is the character-level variant of [`apply_rule_at`].
pub fn apply_rule_at_char(
    rule: &RewriteRuleChar,
    s: &[PhoneChar],
    pos: usize,
) -> Option<Vec<PhoneChar>> {
    // Check if rule can be applied (no allocation)
    if !can_apply_at_char(rule, s, pos) {
        return None;
    }

    // Build result: prefix + replacement + suffix
    let mut result = Vec::with_capacity(s.len() + MAX_EXPANSION_FACTOR);
    result.extend_from_slice(&s[..pos]);
    result.extend_from_slice(&rule.replacement);
    result.extend_from_slice(&s[(pos + rule.pattern.len())..]);

    Some(result)
}

/// Find the first position where a rule can be applied (character-level).
///
/// **Formal Specification**: `docs/verification/phonetic/rewrite_rules.v:190-198`
///
/// This is the character-level variant of [`find_first_match`].
pub fn find_first_match_char(rule: &RewriteRuleChar, s: &[PhoneChar]) -> Option<usize> {
    find_first_match_from_char(rule, s, 0)
}

/// Find the first position where a rule can be applied, starting from a given position (character-level).
///
/// This is the character-level variant of [`find_first_match_from`].
///
/// # Arguments
///
/// - `rule` - The rewrite rule to match
/// - `s` - The phonetic string
/// - `start_pos` - The position to start scanning from (0-based)
///
/// # Returns
///
/// - `Some(pos)` if the rule can be applied at position `pos >= start_pos`
/// - `None` if the rule cannot be applied anywhere from `start_pos` onward
#[inline]
pub fn find_first_match_from_char(
    rule: &RewriteRuleChar,
    s: &[PhoneChar],
    start_pos: usize,
) -> Option<usize> {
    // Optimization: use can_apply_at_char() to avoid allocating vectors during search
    for pos in start_pos..=s.len() {
        if can_apply_at_char(rule, s, pos) {
            return Some(pos);
        }
    }
    None
}

/// Apply a list of rules sequentially until fixed point or fuel exhausted (character-level).
///
/// **Formal Specification**: `docs/verification/phonetic/rewrite_rules.v:203-227`
///
/// This is the character-level variant of [`apply_rules_seq`].
pub fn apply_rules_seq_char(
    rules: &[RewriteRuleChar],
    s: &[PhoneChar],
    fuel: usize,
) -> Option<Vec<PhoneChar>> {
    let mut current = s.to_vec();
    let original_len = s.len();
    let mut remaining_fuel = fuel;

    loop {
        if remaining_fuel == 0 {
            return Some(current);
        }

        let mut applied = false;

        for rule in rules {
            if let Some(pos) = find_first_match_char(rule, &current) {
                if let Some(new_s) = apply_rule_at_char(rule, &current, pos) {
                    // Defensive safeguard: abort if expansion exceeds limit
                    if new_s.len() > original_len + MAX_TOTAL_EXPANSION {
                        eprintln!(
                            "[phonetic] Warning: Normalization exceeded expansion limit \
                             ({} > {} + {}). Returning current state to prevent runaway expansion. \
                             Consider revising rules to avoid pathological interactions.",
                            new_s.len(),
                            original_len,
                            MAX_TOTAL_EXPANSION
                        );
                        return Some(current);
                    }
                    current = new_s;
                    remaining_fuel -= 1;
                    applied = true;
                    break;
                }
            }
        }

        if !applied {
            return Some(current);
        }
    }
}

// ============================================================================
// Optimized rule application with conditional position skipping (character-level)
// ============================================================================

/// Apply rules with position skipping optimization enabled (character-level).
///
/// **Formal Specification**: `docs/verification/phonetic/position_skipping_proof.v`
///
/// This is the character-level variant of [`apply_rules_seq_optimized`].
/// See that function for detailed documentation.
///
/// **SAFETY**: This function MUST only be called when no rules use `ContextChar::Final`.
pub fn apply_rules_seq_optimized_char(
    rules: &[RewriteRuleChar],
    s: &[PhoneChar],
    fuel: usize,
) -> Option<Vec<PhoneChar>> {
    debug_assert!(
        !has_position_dependent_rules_char(rules),
        "apply_rules_seq_optimized_char requires no position-dependent rules (ContextChar::Final)"
    );

    let mut current = s.to_vec();
    let mut remaining_fuel = fuel;
    let mut last_pos: usize = 0; // Position skipping: start from here

    loop {
        if remaining_fuel == 0 {
            // Out of fuel - return current state
            return Some(current);
        }

        let mut applied = false;

        // Try each rule in order, starting from last_pos
        for rule in rules {
            // Key optimization: search from last_pos instead of 0
            if let Some(pos) = find_first_match_from_char(rule, &current, last_pos) {
                if let Some(new_s) = apply_rule_at_char(rule, &current, pos) {
                    current = new_s;
                    remaining_fuel -= 1;
                    last_pos = pos; // Next iteration starts from here
                    applied = true;
                    break; // Restart from first rule
                }
            }
        }

        if !applied {
            // Fixed point reached - no rules can be applied
            return Some(current);
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::phonetic::types::{Context, ContextChar, Phone, PhoneChar};

    // ========================================================================
    // Byte-level tests
    // ========================================================================

    #[test]
    fn test_apply_rule_at_success() {
        let rule = RewriteRule {
            rule_id: 1,
            rule_name: "gh → f".to_string(),
            pattern: vec![Phone::Consonant(b'g'), Phone::Consonant(b'h')],
            replacement: vec![Phone::Consonant(b'f')],
            context: Context::Anywhere,
            weight: 0.15,
            syllable_condition: None,
        };

        let s = vec![
            Phone::Vowel(b'e'),
            Phone::Consonant(b'g'),
            Phone::Consonant(b'h'),
        ];

        let result = apply_rule_at(&rule, &s, 1);
        assert_eq!(result, Some(vec![Phone::Vowel(b'e'), Phone::Consonant(b'f')]));
    }

    #[test]
    fn test_apply_rule_at_no_match() {
        let rule = RewriteRule {
            rule_id: 1,
            rule_name: "gh → f".to_string(),
            pattern: vec![Phone::Consonant(b'g'), Phone::Consonant(b'h')],
            replacement: vec![Phone::Consonant(b'f')],
            context: Context::Anywhere,
            weight: 0.15,
            syllable_condition: None,
        };

        let s = vec![Phone::Vowel(b'e'), Phone::Consonant(b'k')];

        let result = apply_rule_at(&rule, &s, 0);
        assert_eq!(result, None);
    }

    #[test]
    fn test_apply_rule_at_context_fail() {
        let rule = RewriteRule {
            rule_id: 1,
            rule_name: "gh → f (initial only)".to_string(),
            pattern: vec![Phone::Consonant(b'g'), Phone::Consonant(b'h')],
            replacement: vec![Phone::Consonant(b'f')],
            context: Context::Initial,
            weight: 0.15,
            syllable_condition: None,
        };

        let s = vec![
            Phone::Vowel(b'e'),
            Phone::Consonant(b'g'),
            Phone::Consonant(b'h'),
        ];

        // Should fail at pos=1 because not initial
        let result = apply_rule_at(&rule, &s, 1);
        assert_eq!(result, None);

        // Should succeed at pos=0 if pattern were there
        let s2 = vec![Phone::Consonant(b'g'), Phone::Consonant(b'h')];
        let result2 = apply_rule_at(&rule, &s2, 0);
        assert_eq!(result2, Some(vec![Phone::Consonant(b'f')]));
    }

    #[test]
    fn test_find_first_match() {
        let rule = RewriteRule {
            rule_id: 1,
            rule_name: "gh → f".to_string(),
            pattern: vec![Phone::Consonant(b'g'), Phone::Consonant(b'h')],
            replacement: vec![Phone::Consonant(b'f')],
            context: Context::Anywhere,
            weight: 0.15,
            syllable_condition: None,
        };

        let s = vec![
            Phone::Vowel(b'e'),
            Phone::Consonant(b'g'),
            Phone::Consonant(b'h'),
            Phone::Vowel(b'o'),
        ];

        let pos = find_first_match(&rule, &s);
        assert_eq!(pos, Some(1));
    }

    #[test]
    fn test_find_first_match_none() {
        let rule = RewriteRule {
            rule_id: 1,
            rule_name: "gh → f".to_string(),
            pattern: vec![Phone::Consonant(b'g'), Phone::Consonant(b'h')],
            replacement: vec![Phone::Consonant(b'f')],
            context: Context::Anywhere,
            weight: 0.15,
            syllable_condition: None,
        };

        let s = vec![Phone::Vowel(b'e'), Phone::Consonant(b'k')];

        let pos = find_first_match(&rule, &s);
        assert_eq!(pos, None);
    }

    #[test]
    fn test_apply_rules_seq_single_rule() {
        let rule = RewriteRule {
            rule_id: 1,
            rule_name: "gh → f".to_string(),
            pattern: vec![Phone::Consonant(b'g'), Phone::Consonant(b'h')],
            replacement: vec![Phone::Consonant(b'f')],
            context: Context::Anywhere,
            weight: 0.15,
            syllable_condition: None,
        };

        let s = vec![
            Phone::Vowel(b'e'),
            Phone::Consonant(b'g'),
            Phone::Consonant(b'h'),
        ];

        let result = apply_rules_seq(&[rule], &s, 100);
        assert_eq!(result, Some(vec![Phone::Vowel(b'e'), Phone::Consonant(b'f')]));
    }

    #[test]
    fn test_apply_rules_seq_multiple_applications() {
        let rule = RewriteRule {
            rule_id: 1,
            rule_name: "gh → f".to_string(),
            pattern: vec![Phone::Consonant(b'g'), Phone::Consonant(b'h')],
            replacement: vec![Phone::Consonant(b'f')],
            context: Context::Anywhere,
            weight: 0.15,
            syllable_condition: None,
        };

        // "ghgh" → "fgh" → "ff"
        let s = vec![
            Phone::Consonant(b'g'),
            Phone::Consonant(b'h'),
            Phone::Consonant(b'g'),
            Phone::Consonant(b'h'),
        ];

        let result = apply_rules_seq(&[rule], &s, 100);
        assert_eq!(
            result,
            Some(vec![Phone::Consonant(b'f'), Phone::Consonant(b'f')])
        );
    }

    #[test]
    fn test_apply_rules_seq_fixed_point() {
        let rule = RewriteRule {
            rule_id: 1,
            rule_name: "gh → f".to_string(),
            pattern: vec![Phone::Consonant(b'g'), Phone::Consonant(b'h')],
            replacement: vec![Phone::Consonant(b'f')],
            context: Context::Anywhere,
            weight: 0.15,
            syllable_condition: None,
        };

        // Already a fixed point (no 'gh')
        let s = vec![Phone::Vowel(b'e'), Phone::Consonant(b'f')];

        let result = apply_rules_seq(&[rule], &s, 100);
        assert_eq!(result, Some(vec![Phone::Vowel(b'e'), Phone::Consonant(b'f')]));
    }

    // ========================================================================
    // Character-level tests
    // ========================================================================

    #[test]
    fn test_apply_rule_at_char_success() {
        let rule = RewriteRuleChar {
            rule_id: 1,
            rule_name: "gh → f".to_string(),
            pattern: vec![PhoneChar::Consonant('g'), PhoneChar::Consonant('h')],
            replacement: vec![PhoneChar::Consonant('f')],
            context: ContextChar::Anywhere,
            weight: 0.15,
            syllable_condition: None,
        };

        let s = vec![
            PhoneChar::Vowel('e'),
            PhoneChar::Consonant('g'),
            PhoneChar::Consonant('h'),
        ];

        let result = apply_rule_at_char(&rule, &s, 1);
        assert_eq!(
            result,
            Some(vec![PhoneChar::Vowel('e'), PhoneChar::Consonant('f')])
        );
    }

    #[test]
    fn test_apply_rules_seq_char() {
        let rule = RewriteRuleChar {
            rule_id: 1,
            rule_name: "gh → f".to_string(),
            pattern: vec![PhoneChar::Consonant('g'), PhoneChar::Consonant('h')],
            replacement: vec![PhoneChar::Consonant('f')],
            context: ContextChar::Anywhere,
            weight: 0.15,
            syllable_condition: None,
        };

        let s = vec![
            PhoneChar::Vowel('e'),
            PhoneChar::Consonant('g'),
            PhoneChar::Consonant('h'),
        ];

        let result = apply_rules_seq_char(&[rule], &s, 100);
        assert_eq!(
            result,
            Some(vec![PhoneChar::Vowel('e'), PhoneChar::Consonant('f')])
        );
    }

    // ========================================================================
    // Position skipping optimization tests (byte-level)
    // ========================================================================

    #[test]
    fn test_has_position_dependent_rules_empty() {
        let rules: Vec<RewriteRule> = vec![];
        assert!(!has_position_dependent_rules(&rules));
    }

    #[test]
    fn test_has_position_dependent_rules_no_final() {
        let rules = vec![
            RewriteRule {
                rule_id: 1,
                rule_name: "test".to_string(),
                pattern: vec![Phone::Consonant(b'g')],
                replacement: vec![Phone::Consonant(b'k')],
                context: Context::Anywhere,
                weight: 1.0,
            syllable_condition: None,
            },
            RewriteRule {
                rule_id: 2,
                rule_name: "test2".to_string(),
                pattern: vec![Phone::Consonant(b'c')],
                replacement: vec![Phone::Consonant(b's')],
                context: Context::Initial,
                weight: 1.0,
            syllable_condition: None,
            },
        ];
        assert!(!has_position_dependent_rules(&rules));
    }

    #[test]
    fn test_has_position_dependent_rules_with_final() {
        let rules = vec![
            RewriteRule {
                rule_id: 1,
                rule_name: "test".to_string(),
                pattern: vec![Phone::Consonant(b'g')],
                replacement: vec![Phone::Consonant(b'k')],
                context: Context::Anywhere,
                weight: 1.0,
            syllable_condition: None,
            },
            RewriteRule {
                rule_id: 2,
                rule_name: "final_rule".to_string(),
                pattern: vec![Phone::Vowel(b'e')],
                replacement: vec![],
                context: Context::Final,
                weight: 1.0,
            syllable_condition: None,
            },
        ];
        assert!(has_position_dependent_rules(&rules));
    }

    #[test]
    fn test_find_first_match_from_start() {
        let rule = RewriteRule {
            rule_id: 1,
            rule_name: "gh → f".to_string(),
            pattern: vec![Phone::Consonant(b'g'), Phone::Consonant(b'h')],
            replacement: vec![Phone::Consonant(b'f')],
            context: Context::Anywhere,
            weight: 0.15,
            syllable_condition: None,
        };

        let s = vec![
            Phone::Vowel(b'e'),
            Phone::Consonant(b'g'),
            Phone::Consonant(b'h'),
            Phone::Vowel(b'o'),
        ];

        // Search from start
        assert_eq!(find_first_match_from(&rule, &s, 0), Some(1));
        // Search from position 1 (should still find it)
        assert_eq!(find_first_match_from(&rule, &s, 1), Some(1));
        // Search from position 2 (should not find it)
        assert_eq!(find_first_match_from(&rule, &s, 2), None);
    }

    #[test]
    fn test_find_first_match_from_multiple_occurrences() {
        let rule = RewriteRule {
            rule_id: 1,
            rule_name: "gh → f".to_string(),
            pattern: vec![Phone::Consonant(b'g'), Phone::Consonant(b'h')],
            replacement: vec![Phone::Consonant(b'f')],
            context: Context::Anywhere,
            weight: 0.15,
            syllable_condition: None,
        };

        // "e gh o gh a"
        let s = vec![
            Phone::Vowel(b'e'),
            Phone::Consonant(b'g'),
            Phone::Consonant(b'h'),
            Phone::Vowel(b'o'),
            Phone::Consonant(b'g'),
            Phone::Consonant(b'h'),
            Phone::Vowel(b'a'),
        ];

        // Search from start finds first occurrence
        assert_eq!(find_first_match_from(&rule, &s, 0), Some(1));
        // Search from position 2 finds second occurrence
        assert_eq!(find_first_match_from(&rule, &s, 2), Some(4));
        // Search from position 5 finds nothing
        assert_eq!(find_first_match_from(&rule, &s, 5), None);
    }

    #[test]
    fn test_apply_rules_seq_optimized_produces_same_result() {
        // Verify optimized version produces same result as non-optimized
        let rule = RewriteRule {
            rule_id: 1,
            rule_name: "gh → f".to_string(),
            pattern: vec![Phone::Consonant(b'g'), Phone::Consonant(b'h')],
            replacement: vec![Phone::Consonant(b'f')],
            context: Context::Anywhere,
            weight: 0.15,
            syllable_condition: None,
        };

        let s = vec![
            Phone::Vowel(b'e'),
            Phone::Consonant(b'g'),
            Phone::Consonant(b'h'),
            Phone::Vowel(b'o'),
            Phone::Consonant(b'g'),
            Phone::Consonant(b'h'),
        ];

        let standard_result = apply_rules_seq(&[rule.clone()], &s, 100);
        let optimized_result = apply_rules_seq_optimized(&[rule], &s, 100);

        assert_eq!(standard_result, optimized_result);
    }

    #[test]
    fn test_apply_rules_seq_optimized_fixed_point() {
        let rule = RewriteRule {
            rule_id: 1,
            rule_name: "gh → f".to_string(),
            pattern: vec![Phone::Consonant(b'g'), Phone::Consonant(b'h')],
            replacement: vec![Phone::Consonant(b'f')],
            context: Context::Anywhere,
            weight: 0.15,
            syllable_condition: None,
        };

        // Already at fixed point
        let s = vec![Phone::Vowel(b'e'), Phone::Consonant(b'f')];

        let result = apply_rules_seq_optimized(&[rule], &s, 100);
        assert_eq!(result, Some(vec![Phone::Vowel(b'e'), Phone::Consonant(b'f')]));
    }

    // ========================================================================
    // Position skipping optimization tests (character-level)
    // ========================================================================

    #[test]
    fn test_has_position_dependent_rules_char_empty() {
        let rules: Vec<RewriteRuleChar> = vec![];
        assert!(!has_position_dependent_rules_char(&rules));
    }

    #[test]
    fn test_has_position_dependent_rules_char_no_final() {
        let rules = vec![RewriteRuleChar {
            rule_id: 1,
            rule_name: "test".to_string(),
            pattern: vec![PhoneChar::Consonant('g')],
            replacement: vec![PhoneChar::Consonant('k')],
            context: ContextChar::Anywhere,
            weight: 1.0,
            syllable_condition: None,
        }];
        assert!(!has_position_dependent_rules_char(&rules));
    }

    #[test]
    fn test_has_position_dependent_rules_char_with_final() {
        let rules = vec![RewriteRuleChar {
            rule_id: 1,
            rule_name: "final_rule".to_string(),
            pattern: vec![PhoneChar::Vowel('e')],
            replacement: vec![],
            context: ContextChar::Final,
            weight: 1.0,
            syllable_condition: None,
        }];
        assert!(has_position_dependent_rules_char(&rules));
    }

    #[test]
    fn test_find_first_match_from_char() {
        let rule = RewriteRuleChar {
            rule_id: 1,
            rule_name: "gh → f".to_string(),
            pattern: vec![PhoneChar::Consonant('g'), PhoneChar::Consonant('h')],
            replacement: vec![PhoneChar::Consonant('f')],
            context: ContextChar::Anywhere,
            weight: 0.15,
            syllable_condition: None,
        };

        let s = vec![
            PhoneChar::Vowel('e'),
            PhoneChar::Consonant('g'),
            PhoneChar::Consonant('h'),
            PhoneChar::Vowel('o'),
        ];

        assert_eq!(find_first_match_from_char(&rule, &s, 0), Some(1));
        assert_eq!(find_first_match_from_char(&rule, &s, 1), Some(1));
        assert_eq!(find_first_match_from_char(&rule, &s, 2), None);
    }

    #[test]
    fn test_apply_rules_seq_optimized_char_produces_same_result() {
        let rule = RewriteRuleChar {
            rule_id: 1,
            rule_name: "gh → f".to_string(),
            pattern: vec![PhoneChar::Consonant('g'), PhoneChar::Consonant('h')],
            replacement: vec![PhoneChar::Consonant('f')],
            context: ContextChar::Anywhere,
            weight: 0.15,
            syllable_condition: None,
        };

        let s = vec![
            PhoneChar::Vowel('e'),
            PhoneChar::Consonant('g'),
            PhoneChar::Consonant('h'),
            PhoneChar::Vowel('o'),
            PhoneChar::Consonant('g'),
            PhoneChar::Consonant('h'),
        ];

        let standard_result = apply_rules_seq_char(&[rule.clone()], &s, 100);
        let optimized_result = apply_rules_seq_optimized_char(&[rule], &s, 100);

        assert_eq!(standard_result, optimized_result);
    }

    // ========================================================================
    // Cycle detection tests
    // ========================================================================

    #[test]
    fn test_cycle_detection_simple_cycle() {
        // Rules: ab -> ba, ba -> ab (creates a true cycle)
        // Both patterns are same length, so no expansion happens
        let rule_ab_to_ba = RewriteRule {
            rule_id: 1,
            rule_name: "ab → ba".to_string(),
            pattern: vec![Phone::Vowel(b'a'), Phone::Consonant(b'b')],
            replacement: vec![Phone::Consonant(b'b'), Phone::Vowel(b'a')],
            context: Context::Anywhere,
            weight: 0.0,
            syllable_condition: None,
        };

        let rule_ba_to_ab = RewriteRule {
            rule_id: 2,
            rule_name: "ba → ab".to_string(),
            pattern: vec![Phone::Consonant(b'b'), Phone::Vowel(b'a')],
            replacement: vec![Phone::Vowel(b'a'), Phone::Consonant(b'b')],
            context: Context::Anywhere,
            weight: 0.0,
            syllable_condition: None,
        };

        let rules = vec![rule_ab_to_ba, rule_ba_to_ab];
        let input = vec![Phone::Vowel(b'a'), Phone::Consonant(b'b')];

        let result = apply_rules_with_cycle_detection(&rules, &input, 100);

        // Should detect cycle
        assert!(result.is_cycle(), "Expected cycle detection, got {:?}", result);

        if let NormalizationResult::Cycle(forms) = &result {
            // Should have both forms
            assert!(forms.contains(&vec![Phone::Vowel(b'a'), Phone::Consonant(b'b')]));
            assert!(forms.contains(&vec![Phone::Consonant(b'b'), Phone::Vowel(b'a')]));
            assert_eq!(forms.len(), 2, "Expected exactly 2 forms in cycle");
        }
    }

    #[test]
    fn test_cycle_detection_fixed_point() {
        // Rule that doesn't create a cycle: ph -> f
        let rule = RewriteRule {
            rule_id: 1,
            rule_name: "ph → f".to_string(),
            pattern: vec![Phone::Consonant(b'p'), Phone::Consonant(b'h')],
            replacement: vec![Phone::Consonant(b'f')],
            context: Context::Anywhere,
            weight: 0.15,
            syllable_condition: None,
        };

        let input = vec![
            Phone::Consonant(b'p'),
            Phone::Consonant(b'h'),
            Phone::Vowel(b'o'),
            Phone::Consonant(b'n'),
            Phone::Vowel(b'e'),
        ];

        let result = apply_rules_with_cycle_detection(&[rule], &input, 100);

        // Should reach fixed point, not cycle
        assert!(result.is_fixed_point(), "Expected fixed point, got {:?}", result);

        if let NormalizationResult::FixedPoint(form) = &result {
            assert_eq!(
                form,
                &vec![
                    Phone::Consonant(b'f'),
                    Phone::Vowel(b'o'),
                    Phone::Consonant(b'n'),
                    Phone::Vowel(b'e'),
                ]
            );
        }
    }

    #[test]
    fn test_cycle_detection_fuel_exhausted() {
        // Rule that keeps expanding (never terminates without fuel)
        let rule = RewriteRule {
            rule_id: 1,
            rule_name: "a → aa".to_string(),
            pattern: vec![Phone::Vowel(b'a')],
            replacement: vec![Phone::Vowel(b'a'), Phone::Vowel(b'a')],
            context: Context::Anywhere,
            weight: 0.0,
            syllable_condition: None,
        };

        let input = vec![Phone::Vowel(b'a')];

        // With very limited fuel, should exhaust before cycle (never cycles)
        let result = apply_rules_with_cycle_detection(&[rule], &input, 3);

        // Should exhaust fuel (expansion doesn't cycle, just grows)
        assert!(
            matches!(result, NormalizationResult::FuelExhausted(_)),
            "Expected fuel exhaustion, got {:?}",
            result
        );
    }

    #[test]
    fn test_cycle_detection_all_forms() {
        // Three-way cycle: a -> b -> c -> a
        let rule_a_to_b = RewriteRule {
            rule_id: 1,
            rule_name: "a → b".to_string(),
            pattern: vec![Phone::Vowel(b'a')],
            replacement: vec![Phone::Consonant(b'b')],
            context: Context::Anywhere,
            weight: 0.0,
            syllable_condition: None,
        };

        let rule_b_to_c = RewriteRule {
            rule_id: 2,
            rule_name: "b → c".to_string(),
            pattern: vec![Phone::Consonant(b'b')],
            replacement: vec![Phone::Consonant(b'c')],
            context: Context::Anywhere,
            weight: 0.0,
            syllable_condition: None,
        };

        let rule_c_to_a = RewriteRule {
            rule_id: 3,
            rule_name: "c → a".to_string(),
            pattern: vec![Phone::Consonant(b'c')],
            replacement: vec![Phone::Vowel(b'a')],
            context: Context::Anywhere,
            weight: 0.0,
            syllable_condition: None,
        };

        let rules = vec![rule_a_to_b, rule_b_to_c, rule_c_to_a];
        let input = vec![Phone::Vowel(b'a')];

        let result = apply_rules_with_cycle_detection(&rules, &input, 100);

        assert!(result.is_cycle());

        let all_forms = result.all_forms();
        assert_eq!(all_forms.len(), 3, "Expected 3 forms in cycle");
        assert!(all_forms.contains(&vec![Phone::Vowel(b'a')]));
        assert!(all_forms.contains(&vec![Phone::Consonant(b'b')]));
        assert!(all_forms.contains(&vec![Phone::Consonant(b'c')]));
    }

    #[test]
    fn test_normalization_result_canonical_shortest() {
        // Test that canonical() picks the shortest form
        let mut forms = HashSet::new();
        forms.insert(vec![Phone::Vowel(b'a'), Phone::Vowel(b'a')]);
        forms.insert(vec![Phone::Vowel(b'b')]);
        forms.insert(vec![Phone::Vowel(b'c'), Phone::Vowel(b'c'), Phone::Vowel(b'c')]);

        let result = NormalizationResult::Cycle(forms);

        // Canonical should be the shortest: [b]
        assert_eq!(result.canonical(), vec![Phone::Vowel(b'b')]);
    }
}
