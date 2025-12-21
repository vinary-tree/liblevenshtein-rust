//! Core spellcheck logic shared between native and WASM builds.
//!
//! This module contains the spellchecker implementation without any I/O
//! dependencies, making it suitable for WebAssembly.

use liblevenshtein::phonetic::{apply_rules_seq, Phone, RewriteRule};
use liblevenshtein::prelude::*;
use liblevenshtein::transducer::Algorithm;
use serde::{Deserialize, Serialize};
use std::collections::HashMap;

use crate::embedded::{phones_to_string, string_to_phones};

/// Maximum edit distance for fuzzy matching
pub const MAX_DISTANCE: usize = 2;

/// Maximum number of results to return
pub const MAX_RESULTS: usize = 20;

/// Default cache size
pub const DEFAULT_CACHE_SIZE: usize = 1000;

/// Spellchecker configuration
#[derive(Clone, Debug)]
pub struct SpellcheckerConfig {
    pub max_distance: usize,
    pub max_results: usize,
}

impl Default for SpellcheckerConfig {
    fn default() -> Self {
        Self {
            max_distance: MAX_DISTANCE,
            max_results: MAX_RESULTS,
        }
    }
}

/// A spelling match with the matched word and edit distance
#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct SpellMatch {
    pub word: String,
    pub distance: usize,
}

/// Result of a spelling query
#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct QueryResult {
    pub original: String,
    pub normalized: String,
    pub matches: Vec<SpellMatch>,
    pub warning: Option<String>,
    pub from_cache: bool,
}

/// Core phonetic spellchecker (no I/O dependencies)
///
/// Uses DynamicDawg for fast lookups and a Levenshtein transducer
/// for fuzzy matching with transposition support.
pub struct PhoneticSpellchecker {
    /// Levenshtein transducer over normalized dictionary terms
    transducer: Transducer<DynamicDawg>,
    /// Maps normalized phonetic forms back to original dictionary words
    normalized_to_original: DynamicDawg<Vec<String>>,
    /// Phonetic normalization rules
    rules: Vec<RewriteRule>,
    /// Configuration
    config: SpellcheckerConfig,
    /// Query result cache
    cache: HashMap<String, Vec<SpellMatch>>,
    /// Maximum cache size
    cache_size: usize,
}

impl PhoneticSpellchecker {
    /// Create a new spellchecker from dictionary terms and rules
    pub fn new(
        dictionary: &[String],
        rules: Vec<RewriteRule>,
        config: SpellcheckerConfig,
    ) -> Self {
        // Build normalized index: map each dictionary word to its phonetic form
        let mut entries: HashMap<String, Vec<String>> = HashMap::new();
        for term in dictionary {
            let normalized = normalize_fast(term, &rules);
            entries.entry(normalized).or_default().push(term.clone());
        }

        // Build DynamicDawg for normalized -> original mappings
        let normalized_to_original: DynamicDawg<Vec<String>> = DynamicDawg::new();
        for (normalized, originals) in &entries {
            normalized_to_original.insert_with_value(normalized, originals.clone());
        }

        // Build transducer from normalized terms
        let normalized_terms: Vec<&str> = entries.keys().map(|s| s.as_str()).collect();
        let dict = DynamicDawg::from_terms(normalized_terms.into_iter());
        let transducer = Transducer::new(dict, Algorithm::Transposition);

        Self {
            transducer,
            normalized_to_original,
            rules,
            config,
            cache: HashMap::with_capacity(DEFAULT_CACHE_SIZE),
            cache_size: DEFAULT_CACHE_SIZE,
        }
    }

    /// Query for spelling suggestions
    pub fn query(&mut self, word: &str) -> QueryResult {
        let (normalized, warning) = self.normalize(word);

        // Check cache first
        if let Some(cached) = self.cache.get(&normalized) {
            return QueryResult {
                original: word.to_string(),
                normalized,
                matches: cached.clone(),
                warning,
                from_cache: true,
            };
        }

        // Compute matches using the transducer
        let candidates: Vec<_> = self
            .transducer
            .query_with_distance(&normalized, self.config.max_distance)
            .collect();

        // Map normalized matches back to original terms
        let mut seen = std::collections::HashSet::new();
        let mut matches: Vec<SpellMatch> = Vec::new();

        for candidate in &candidates {
            if let Some(originals) = self.normalized_to_original.get_value(&candidate.term) {
                for original in originals {
                    if seen.insert(original.clone()) {
                        matches.push(SpellMatch {
                            word: original.clone(),
                            distance: candidate.distance,
                        });
                    }
                }
            }
        }

        // Sort by distance, then alphabetically
        matches.sort_by(|a, b| a.distance.cmp(&b.distance).then_with(|| a.word.cmp(&b.word)));
        matches.truncate(self.config.max_results);

        // Cache result (simple eviction: clear if full)
        if self.cache.len() >= self.cache_size {
            self.cache.clear();
        }
        self.cache.insert(normalized.clone(), matches.clone());

        QueryResult {
            original: word.to_string(),
            normalized,
            matches,
            warning,
            from_cache: false,
        }
    }

    /// Normalize a string using phonetic rules
    fn normalize(&self, text: &str) -> (String, Option<String>) {
        let phones = string_to_phones(text);
        // Use a practical fuel limit - most words converge in < 10 applications
        let fuel = 50;

        match apply_rules_seq(&self.rules, &phones, fuel) {
            Some(result) => (phones_to_string(&result), None),
            None => (text.to_string(), Some("Fuel exhausted".to_string())),
        }
    }

    /// Clear the query cache
    pub fn clear_cache(&mut self) {
        self.cache.clear();
    }

    /// Get number of cached queries
    pub fn cache_size(&self) -> usize {
        self.cache.len()
    }
}

/// Fast normalization for dictionary preprocessing
///
/// Uses a smaller fuel limit since dictionary words are unlikely
/// to trigger pathological rule applications.
fn normalize_fast(text: &str, rules: &[RewriteRule]) -> String {
    let phones = string_to_phones(text);
    let fuel = 50;

    match apply_rules_seq(rules, &phones, fuel) {
        Some(result) => phones_to_string(&result),
        None => text.to_string(),
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::embedded;

    #[test]
    fn test_spellchecker_creation() {
        let dict = embedded::dictionary();
        let rules = embedded::rules().clone();
        let config = SpellcheckerConfig::default();

        let checker = PhoneticSpellchecker::new(dict, rules, config);
        assert!(checker.cache.is_empty());
    }

    #[test]
    fn test_spellchecker_query() {
        let dict = embedded::dictionary();
        let rules = embedded::rules().clone();
        let config = SpellcheckerConfig::default();

        let mut checker = PhoneticSpellchecker::new(dict, rules, config);
        let result = checker.query("fone");

        // Should find "phone" as a match
        assert!(
            result.matches.iter().any(|m| m.word == "phone"),
            "Expected 'phone' in matches for 'fone'"
        );
    }

    #[test]
    fn test_cache_hit() {
        let dict = embedded::dictionary();
        let rules = embedded::rules().clone();
        let config = SpellcheckerConfig::default();

        let mut checker = PhoneticSpellchecker::new(dict, rules, config);

        // First query - cache miss
        let result1 = checker.query("fone");
        assert!(!result1.from_cache);

        // Second query - cache hit
        let result2 = checker.query("fone");
        assert!(result2.from_cache);
    }
}
