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
use rand::seq::IndexedRandom;
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
/// ```rust
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
        let char_len = chars.len();
        let insertion_count = char_len
            .saturating_add(1)
            .saturating_mul(self.alphabet.len());
        let substitution_count = chars.iter().fold(0usize, |count, &source| {
            count.saturating_add(self.alphabet.iter().filter(|&&c| c != source).count())
        });
        let typo_count = char_len
            .saturating_add(insertion_count)
            .saturating_add(substitution_count)
            .saturating_add(char_len.saturating_sub(1));
        let mut typos = Vec::with_capacity(typo_count);

        // Deletions
        for i in 0..char_len {
            let mut deleted = String::with_capacity(word.len().saturating_sub(chars[i].len_utf8()));
            for (idx, &ch) in chars.iter().enumerate() {
                if idx != i {
                    deleted.push(ch);
                }
            }
            typos.push(deleted);
        }

        // Insertions
        for i in 0..=char_len {
            for &c in &self.alphabet {
                let mut inserted = String::with_capacity(word.len().saturating_add(c.len_utf8()));
                for &ch in &chars[..i] {
                    inserted.push(ch);
                }
                inserted.push(c);
                for &ch in &chars[i..] {
                    inserted.push(ch);
                }
                typos.push(inserted);
            }
        }

        // Substitutions
        for i in 0..char_len {
            for &c in &self.alphabet {
                if c != chars[i] {
                    let mut substituted = String::with_capacity(
                        word.len()
                            .saturating_sub(chars[i].len_utf8())
                            .saturating_add(c.len_utf8()),
                    );
                    for (idx, &ch) in chars.iter().enumerate() {
                        substituted.push(if idx == i { c } else { ch });
                    }
                    typos.push(substituted);
                }
            }
        }

        // Transpositions
        for i in 0..char_len.saturating_sub(1) {
            let mut transposed = String::with_capacity(word.len());
            for idx in 0..char_len {
                if idx == i {
                    transposed.push(chars[i + 1]);
                } else if idx == i + 1 {
                    transposed.push(chars[i]);
                } else {
                    transposed.push(chars[idx]);
                }
            }
            typos.push(transposed);
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
            return self.alphabet[self.rng.random_range(0..self.alphabet.len())].to_string();
        }

        let chars: Vec<char> = word.chars().collect();
        let edit_type = self.rng.random_range(0..4);

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

        let pos = self.rng.random_range(0..chars.len());
        let mut result = chars.to_vec();
        result.remove(pos);
        result.iter().collect()
    }

    fn apply_insertion(&mut self, chars: &[char]) -> String {
        let pos = self.rng.random_range(0..=chars.len());
        let new_char = self.alphabet[self.rng.random_range(0..self.alphabet.len())];

        let mut result = chars.to_vec();
        result.insert(pos, new_char);
        result.iter().collect()
    }

    fn apply_substitution(&mut self, chars: &[char]) -> String {
        if chars.is_empty() {
            return String::new();
        }

        let pos = self.rng.random_range(0..chars.len());
        let new_char = self.alphabet[self.rng.random_range(0..self.alphabet.len())];

        let mut result = chars.to_vec();
        result[pos] = new_char;
        result.iter().collect()
    }

    fn apply_transposition(&mut self, chars: &[char]) -> String {
        if chars.len() < 2 {
            return chars.iter().collect();
        }

        let pos = self.rng.random_range(0..chars.len() - 1);
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
/// ```rust,no_run
/// use liblevenshtein::corpus::{BigTxtCorpus, QueryWorkload};
///
/// # fn main() -> std::io::Result<()> {
/// let corpus = BigTxtCorpus::load("data/corpora/big.txt")?;
/// let workload = QueryWorkload::from_frequencies(
///     &corpus.frequencies,
///     corpus.total_tokens(),
///     1000,
///     42,
/// );
///
/// for (query, expected_freq) in &workload.queries {
///     // Test with realistic query distribution...
/// }
/// # Ok(())
/// # }
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
    /// Query workload with realistic frequency distribution. Empty or
    /// all-zero frequency maps produce an empty workload.
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
        let sampling_total = if total_tokens == sum {
            total_tokens
        } else {
            sum
        };
        if sampling_total == 0 {
            return Self {
                queries: Vec::new(),
            };
        }

        // Sample queries according to frequency distribution
        let mut queries = Vec::with_capacity(num_queries);

        for _ in 0..num_queries {
            let sample = rng.random_range(0..sampling_total);
            let idx = cumulative.partition_point(|&(_, cum)| cum <= sample);

            let word = cumulative[idx].0;
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
    /// Query workload with uniform distribution. If `words` is empty, the
    /// workload is empty.
    pub fn uniform(words: &[String], num_queries: usize, seed: u64) -> Self {
        let mut rng = StdRng::seed_from_u64(seed);
        let mut queries = Vec::with_capacity(num_queries);

        for _ in 0..num_queries {
            if let Some(word) = words.choose(&mut rng) {
                queries.push((word.clone(), 1));
            } else {
                break;
            }
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
    fn test_typo_generator_all_distance_1_unicode() {
        let gen = TypoGenerator::new(42);
        let typos = gen.all_distance_1("é");

        // Deletions: 1, insertions: 2 * 26, substitutions: 26.
        assert_eq!(typos.len(), 79);
        assert!(typos.contains(&String::new()));
        assert!(typos.contains(&"aé".to_string()));
        assert!(typos.contains(&"éa".to_string()));
        assert!(typos.contains(&"a".to_string()));
    }

    #[test]
    fn test_query_workload_uniform() {
        let words = vec!["hello".to_string(), "world".to_string(), "test".to_string()];

        let workload = QueryWorkload::uniform(&words, 100, 42);

        assert_eq!(workload.queries.len(), 100);

        let unique = workload.unique_queries();
        assert!(unique.len() <= 3);
    }

    #[test]
    fn test_query_workload_uniform_empty_words_is_empty() {
        let workload = QueryWorkload::uniform(&[], 100, 42);

        assert!(workload.queries.is_empty());
        assert!(workload.query_strings().is_empty());
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
    fn test_query_workload_from_frequencies_empty_input_is_empty() {
        let frequencies = HashMap::new();
        let workload = QueryWorkload::from_frequencies(&frequencies, 0, 100, 42);

        assert!(workload.queries.is_empty());
    }

    #[test]
    fn test_query_workload_from_frequencies_skips_zero_frequency_entries() {
        let mut frequencies = HashMap::new();
        frequencies.insert("zero".to_string(), 0);
        frequencies.insert("one".to_string(), 1);

        let workload = QueryWorkload::from_frequencies(&frequencies, 1, 100, 42);

        assert_eq!(workload.queries.len(), 100);
        assert!(workload
            .queries
            .iter()
            .all(|(word, freq)| { word == "one" && *freq == 1 }));
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
