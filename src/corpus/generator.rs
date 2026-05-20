//! Generators for synthetic errors and realistic query workloads.
//!
//! This module provides utilities for generating:
//! - Synthetic typing errors (insertions, deletions, substitutions, transpositions)
//! - Realistic query workloads with frequency-stratified sampling
//!
//! **Note**: This module requires the optional `rand` dependency.

#[cfg(feature = "rand")]
use rand::rngs::StdRng;
#[cfg(feature = "rand")]
use rand::seq::SliceRandom;
#[cfg(feature = "rand")]
use rand::{Rng, SeedableRng};

use std::collections::HashMap;

/// Generates synthetic typing errors for testing.
///
/// Creates realistic misspellings using standard edit operations:
/// - Deletion: "hello" → "helo"
/// - Insertion: "hello" → "helllo"
/// - Substitution: "hello" → "hallo"
/// - Transposition: "hello" → "helol"
///
/// # Example
///
/// ```rust,ignore
/// use liblevenshtein::corpus::TypoGenerator;
///
/// let mut gen = TypoGenerator::new(42);
/// let typos = gen.generate_typos("hello", 1, 5);
/// // ["helo", "hllo", "hallo", "helol", "hemlo"]
/// ```
pub struct TypoGenerator {
    rng: StdRng,
    alphabet: Vec<char>,
}

impl TypoGenerator {
    /// Create new generator with seed.
    ///
    /// # Arguments
    ///
    /// * `seed` - Random seed for reproducibility
    pub fn new(seed: u64) -> Self {
        Self {
            rng: StdRng::seed_from_u64(seed),
            alphabet: "abcdefghijklmnopqrstuvwxyz".chars().collect(),
        }
    }

    /// Generate typos at specified edit distance.
    ///
    /// # Arguments
    ///
    /// * `word` - Original word
    /// * `distance` - Maximum edit distance (1-3)
    /// * `count` - Number of typos to generate
    ///
    /// # Returns
    ///
    /// Vector of generated typos (may contain duplicates)
    pub fn generate_typos(&mut self, word: &str, distance: usize, count: usize) -> Vec<String> {
        let mut typos = Vec::with_capacity(count);

        for _ in 0..count {
            let typo = self.generate_single_typo(word, distance);
            typos.push(typo);
        }

        typos
    }

    /// Generate all possible typos at distance 1.
    ///
    /// # Arguments
    ///
    /// * `word` - Original word
    ///
    /// # Returns
    ///
    /// Vector of all possible single-edit typos
    pub fn all_distance_1(&self, word: &str) -> Vec<String> {
        let chars: Vec<char> = word.chars().collect();
        let mut typos = Vec::new();

        // Deletions
        for i in 0..chars.len() {
            let mut deleted = chars.clone();
            deleted.remove(i);
            typos.push(deleted.iter().collect());
        }

        // Insertions
        for i in 0..=chars.len() {
            for &c in &self.alphabet {
                let mut inserted = chars.clone();
                inserted.insert(i, c);
                typos.push(inserted.iter().collect());
            }
        }

        // Substitutions
        for i in 0..chars.len() {
            for &c in &self.alphabet {
                if c != chars[i] {
                    let mut substituted = chars.clone();
                    substituted[i] = c;
                    typos.push(substituted.iter().collect());
                }
            }
        }

        // Transpositions
        for i in 0..chars.len().saturating_sub(1) {
            let mut transposed = chars.clone();
            transposed.swap(i, i + 1);
            typos.push(transposed.iter().collect());
        }

        typos
    }

    fn generate_single_typo(&mut self, word: &str, distance: usize) -> String {
        let mut result = word.to_string();

        for _ in 0..distance {
            result = self.apply_random_edit(&result);
        }

        result
    }

    fn apply_random_edit(&mut self, word: &str) -> String {
        if word.is_empty() {
            return self.alphabet[self.rng.gen_range(0..self.alphabet.len())]
                .to_string();
        }

        let chars: Vec<char> = word.chars().collect();
        let edit_type = self.rng.gen_range(0..4);

        match edit_type {
            0 => self.apply_deletion(&chars),
            1 => self.apply_insertion(&chars),
            2 => self.apply_substitution(&chars),
            _ => self.apply_transposition(&chars),
        }
    }

    fn apply_deletion(&mut self, chars: &[char]) -> String {
        if chars.is_empty() {
            return String::new();
        }

        let pos = self.rng.gen_range(0..chars.len());
        let mut result = chars.to_vec();
        result.remove(pos);
        result.iter().collect()
    }

    fn apply_insertion(&mut self, chars: &[char]) -> String {
        let pos = self.rng.gen_range(0..=chars.len());
        let new_char = self.alphabet[self.rng.gen_range(0..self.alphabet.len())];

        let mut result = chars.to_vec();
        result.insert(pos, new_char);
        result.iter().collect()
    }

    fn apply_substitution(&mut self, chars: &[char]) -> String {
        if chars.is_empty() {
            return String::new();
        }

        let pos = self.rng.gen_range(0..chars.len());
        let new_char = self.alphabet[self.rng.gen_range(0..self.alphabet.len())];

        let mut result = chars.to_vec();
        result[pos] = new_char;
        result.iter().collect()
    }

    fn apply_transposition(&mut self, chars: &[char]) -> String {
        if chars.len() < 2 {
            return chars.iter().collect();
        }

        let pos = self.rng.gen_range(0..chars.len() - 1);
        let mut result = chars.to_vec();
        result.swap(pos, pos + 1);
        result.iter().collect()
    }
}

/// Generates realistic query workloads with frequency-stratified sampling.
///
/// Samples queries from a corpus using its frequency distribution (Zipfian),
/// ensuring that common words appear more frequently in the workload.
///
/// # Example
///
/// ```rust,ignore
/// use liblevenshtein::corpus::{BigTxtCorpus, QueryWorkload};
///
/// let corpus = BigTxtCorpus::load("data/corpora/big.txt")?;
/// let workload = QueryWorkload::from_corpus(&corpus, 1000, 42);
///
/// for (query, expected_freq) in &workload.queries {
///     // Test with realistic query distribution...
/// }
/// ```
#[derive(Debug, Clone)]
pub struct QueryWorkload {
    /// Query words with their expected frequencies in the distribution
    pub queries: Vec<(String, usize)>,
}

impl QueryWorkload {
    /// Create workload from corpus with frequency-stratified sampling.
    ///
    /// # Arguments
    ///
    /// * `frequencies` - Word frequency map from corpus
    /// * `total_tokens` - Total number of tokens in corpus
    /// * `num_queries` - Number of queries to generate
    /// * `seed` - Random seed for reproducibility
    ///
    /// # Returns
    ///
    /// Query workload with realistic frequency distribution
    pub fn from_frequencies(
        frequencies: &HashMap<String, usize>,
        total_tokens: usize,
        num_queries: usize,
        seed: u64,
    ) -> Self {
        let mut rng = StdRng::seed_from_u64(seed);

        // Build cumulative distribution for sampling
        let mut words: Vec<_> = frequencies.iter().collect();
        words.sort_unstable_by(|a, b| b.1.cmp(a.1)); // Sort by frequency descending

        let mut cumulative = Vec::with_capacity(words.len());
        let mut sum = 0;

        for (word, &freq) in &words {
            sum += freq;
            cumulative.push((word.as_str(), sum));
        }

        // Sample queries according to frequency distribution
        let mut queries = Vec::with_capacity(num_queries);

        for _ in 0..num_queries {
            let sample = rng.gen_range(0..total_tokens);

            // Binary search for word
            let idx = cumulative
                .binary_search_by(|&(_, cum)| cum.cmp(&sample))
                .unwrap_or_else(|i| i);

            let word = cumulative[idx.min(cumulative.len() - 1)].0;
            let freq = frequencies[word];

            queries.push((word.to_string(), freq));
        }

        Self { queries }
    }

    /// Create workload with uniform distribution (all words equally likely).
    ///
    /// # Arguments
    ///
    /// * `words` - List of words to sample from
    /// * `num_queries` - Number of queries to generate
    /// * `seed` - Random seed for reproducibility
    ///
    /// # Returns
    ///
    /// Query workload with uniform distribution
    pub fn uniform(words: &[String], num_queries: usize, seed: u64) -> Self {
        let mut rng = StdRng::seed_from_u64(seed);
        let mut queries = Vec::with_capacity(num_queries);

        for _ in 0..num_queries {
            let word = words.choose(&mut rng).expect("uniform query gen requires non-empty word list");
            queries.push((word.clone(), 1));
        }

        Self { queries }
    }

    /// Get queries as a simple vector of strings.
    pub fn query_strings(&self) -> Vec<&str> {
        self.queries.iter().map(|(s, _)| s.as_str()).collect()
    }

    /// Get unique queries (deduplicated).
    pub fn unique_queries(&self) -> Vec<&str> {
        let mut unique: Vec<_> = self.queries.iter().map(|(s, _)| s.as_str()).collect();
        unique.sort_unstable();
        unique.dedup();
        unique
    }

    /// Get statistics about the workload.
    pub fn stats(&self) -> WorkloadStats {
        let unique = self.unique_queries().len();
        let total = self.queries.len();

        let frequencies: Vec<_> = self.queries.iter().map(|(_, f)| *f).collect();
        let min_freq = *frequencies.iter().min().unwrap_or(&0);
        let max_freq = *frequencies.iter().max().unwrap_or(&0);
        let avg_freq = if !frequencies.is_empty() {
            frequencies.iter().sum::<usize>() as f64 / frequencies.len() as f64
        } else {
            0.0
        };

        WorkloadStats {
            total_queries: total,
            unique_queries: unique,
            min_frequency: min_freq,
            max_frequency: max_freq,
            avg_frequency: avg_freq,
        }
    }
}

/// Statistics about a query workload.
#[derive(Debug, Clone)]
pub struct WorkloadStats {
    /// Total number of queries
    pub total_queries: usize,
    /// Number of unique queries
    pub unique_queries: usize,
    /// Minimum word frequency
    pub min_frequency: usize,
    /// Maximum word frequency
    pub max_frequency: usize,
    /// Average word frequency
    pub avg_frequency: f64,
}

#[cfg(all(test, feature = "rand"))]
mod tests {
    use super::*;

    #[test]
    fn test_typo_generator_distance_1() {
        let mut gen = TypoGenerator::new(42);
        let typos = gen.generate_typos("test", 1, 10);

        assert_eq!(typos.len(), 10);

        // All typos should be at distance 1
        for typo in &typos {
            assert!(typo.len() >= 3 && typo.len() <= 5);
        }
    }

    #[test]
    fn test_typo_generator_all_distance_1() {
        let gen = TypoGenerator::new(42);
        let typos = gen.all_distance_1("ab");

        // Deletions: 2
        // Insertions: (2+1) * 26 = 78
        // Substitutions: 2 * 25 = 50 (excluding same letter)
        // Transpositions: 1
        // Total: 2 + 78 + 50 + 1 = 131
        assert_eq!(typos.len(), 131);

        // Check some specific typos
        assert!(typos.contains(&"a".to_string())); // Deletion
        assert!(typos.contains(&"b".to_string())); // Deletion
        assert!(typos.contains(&"ba".to_string())); // Transposition
        assert!(typos.contains(&"aab".to_string())); // Insertion
        assert!(typos.contains(&"xb".to_string())); // Substitution
    }

    #[test]
    fn test_query_workload_uniform() {
        let words = vec![
            "hello".to_string(),
            "world".to_string(),
            "test".to_string(),
        ];

        let workload = QueryWorkload::uniform(&words, 100, 42);

        assert_eq!(workload.queries.len(), 100);

        let unique = workload.unique_queries();
        assert!(unique.len() <= 3);
    }

    #[test]
    fn test_query_workload_from_frequencies() {
        let mut frequencies = HashMap::new();
        frequencies.insert("the".to_string(), 100);
        frequencies.insert("quick".to_string(), 10);
        frequencies.insert("fox".to_string(), 1);

        let total = 111;
        let workload = QueryWorkload::from_frequencies(&frequencies, total, 1000, 42);

        assert_eq!(workload.queries.len(), 1000);

        // "the" should appear much more frequently than "fox"
        let the_count = workload.queries.iter().filter(|(w, _)| w == "the").count();
        let fox_count = workload.queries.iter().filter(|(w, _)| w == "fox").count();

        assert!(the_count > fox_count * 10);
    }

    #[test]
    fn test_workload_stats() {
        let queries = vec![
            ("the".to_string(), 100),
            ("the".to_string(), 100),
            ("quick".to_string(), 10),
        ];

        let workload = QueryWorkload { queries };
        let stats = workload.stats();

        assert_eq!(stats.total_queries, 3);
        assert_eq!(stats.unique_queries, 2);
        assert_eq!(stats.min_frequency, 10);
        assert_eq!(stats.max_frequency, 100);
    }
}
