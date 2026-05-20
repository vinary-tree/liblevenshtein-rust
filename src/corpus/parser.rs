//! Corpus parsers for loading test data.
//!
//! This module provides parsers for standard spelling correction test corpora.

use std::collections::HashMap;
use std::fs::File;
use std::io::{BufRead, BufReader, Result};
use std::path::Path;

/// Parser for Norvig's big.txt corpus.
///
/// This corpus contains ~230K words (32,192 unique) from public domain literature,
/// widely used for benchmarking spelling correction algorithms.
///
/// # Format
///
/// Plain text, one token per line (frequency preserved):
///
/// ```text
/// the
/// the
/// the
/// ...
/// ```
///
/// # Example
///
/// ```rust,ignore
/// use liblevenshtein::corpus::BigTxtCorpus;
///
/// let corpus = BigTxtCorpus::load("data/corpora/big.txt")?;
/// let unique_words = corpus.unique_words();
/// let total_tokens = corpus.total_tokens();
/// let frequency = corpus.frequency("the"); // Number of occurrences
/// ```
#[derive(Debug, Clone)]
pub struct BigTxtCorpus {
    /// Word frequencies: word -> count
    pub frequencies: HashMap<String, usize>,
    /// Total number of tokens
    pub total: usize,
}

impl BigTxtCorpus {
    /// Load corpus from file.
    ///
    /// # Arguments
    ///
    /// * `path` - Path to big.txt file
    ///
    /// # Returns
    ///
    /// Parsed corpus with word frequencies
    ///
    /// # Errors
    ///
    /// Returns error if file cannot be read or parsed
    pub fn load<P: AsRef<Path>>(path: P) -> Result<Self> {
        let file = File::open(path)?;
        let reader = BufReader::new(file);

        let mut frequencies = HashMap::new();
        let mut total = 0;

        for line in reader.lines() {
            let line = line?;
            let word = line.trim().to_lowercase();

            if !word.is_empty() {
                *frequencies.entry(word).or_insert(0) += 1;
                total += 1;
            }
        }

        Ok(Self { frequencies, total })
    }

    /// Get number of unique words in corpus.
    #[inline]
    pub fn unique_words(&self) -> usize {
        self.frequencies.len()
    }

    /// Get total number of tokens in corpus.
    #[inline]
    pub fn total_tokens(&self) -> usize {
        self.total
    }

    /// Get frequency of a word in corpus.
    ///
    /// # Arguments
    ///
    /// * `word` - Word to query
    ///
    /// # Returns
    ///
    /// Number of occurrences (0 if word not in corpus)
    #[inline]
    pub fn frequency(&self, word: &str) -> usize {
        self.frequencies.get(word).copied().unwrap_or(0)
    }

    /// Get all words sorted by frequency (descending).
    ///
    /// # Returns
    ///
    /// Vector of (word, frequency) pairs sorted by frequency
    pub fn words_by_frequency(&self) -> Vec<(&str, usize)> {
        let mut words: Vec<_> = self
            .frequencies
            .iter()
            .map(|(w, &f)| (w.as_str(), f))
            .collect();

        words.sort_unstable_by(|a, b| b.1.cmp(&a.1));
        words
    }

    /// Get words as a sorted vector (lexicographic order).
    ///
    /// Useful for dictionary construction.
    ///
    /// # Returns
    ///
    /// Vector of unique words in lexicographic order
    pub fn words_sorted(&self) -> Vec<&str> {
        let mut words: Vec<_> = self.frequencies.keys().map(|s| s.as_str()).collect();
        words.sort_unstable();
        words
    }
}

/// Parser for Mitton-format corpora (Holbrook, Aspell, Wikipedia).
///
/// These corpora use the format developed by Roger Mitton (Birkbeck College)
/// for spelling error datasets.
///
/// # Format
///
/// ```text
/// $correct_word
/// misspelling1 frequency1
/// misspelling2 frequency2
/// ...
/// $next_correct_word
/// ...
/// ```
///
/// Each correct word is preceded by `$` and followed by its misspellings
/// with optional frequency counts (default: 1).
///
/// # Example
///
/// ```rust,ignore
/// use liblevenshtein::corpus::MittonCorpus;
///
/// let holbrook = MittonCorpus::load("data/corpora/holbrook.dat")?;
///
/// for (correct, misspellings) in &holbrook.errors {
///     println!("Correct: {}", correct);
///     for (misspelling, frequency) in misspellings {
///         println!("  {} (×{})", misspelling, frequency);
///     }
/// }
/// ```
#[derive(Debug, Clone)]
pub struct MittonCorpus {
    /// Spelling errors: correct_word -> [(misspelling, frequency)]
    pub errors: HashMap<String, Vec<(String, usize)>>,
}

impl MittonCorpus {
    /// Load corpus from file.
    ///
    /// # Arguments
    ///
    /// * `path` - Path to .dat file (Holbrook, Aspell, Wikipedia format)
    ///
    /// # Returns
    ///
    /// Parsed corpus with error mappings
    ///
    /// # Errors
    ///
    /// Returns error if file cannot be read or parsed
    pub fn load<P: AsRef<Path>>(path: P) -> Result<Self> {
        let file = File::open(path)?;
        let reader = BufReader::new(file);

        let mut errors: HashMap<String, Vec<(String, usize)>> = HashMap::new();
        let mut current_correct: Option<String> = None;

        for line in reader.lines() {
            let line = line?;
            let trimmed = line.trim();

            if trimmed.is_empty() {
                continue;
            }

            if let Some(correct) = trimmed.strip_prefix('$') {
                // New correct word
                current_correct = Some(correct.to_string());
                errors.entry(correct.to_string()).or_default();
            } else if let Some(correct) = &current_correct {
                // Misspelling line: "word" or "word frequency"
                let parts: Vec<&str> = trimmed.split_whitespace().collect();

                if parts.is_empty() {
                    continue;
                }

                let misspelling = parts[0].to_string();
                let frequency = if parts.len() > 1 {
                    parts[1].parse().unwrap_or(1)
                } else {
                    1
                };

                errors
                    .entry(correct.clone())
                    .or_default()
                    .push((misspelling, frequency));
            }
        }

        Ok(Self { errors })
    }

    /// Get number of correct words in corpus.
    #[inline]
    pub fn num_correct_words(&self) -> usize {
        self.errors.len()
    }

    /// Get total number of misspelling instances (with frequency).
    pub fn total_misspellings(&self) -> usize {
        self.errors
            .values()
            .flat_map(|v| v.iter().map(|(_, freq)| freq))
            .sum()
    }

    /// Get total number of unique misspellings (ignoring frequency).
    pub fn unique_misspellings(&self) -> usize {
        self.errors.values().map(|v| v.len()).sum()
    }

    /// Get all correct words as a sorted vector.
    pub fn correct_words_sorted(&self) -> Vec<&str> {
        let mut words: Vec<_> = self.errors.keys().map(|s| s.as_str()).collect();
        words.sort_unstable();
        words
    }

    /// Get all (misspelling, correct, frequency) triples.
    ///
    /// Useful for validation testing.
    pub fn all_errors(&self) -> Vec<(&str, &str, usize)> {
        self.errors
            .iter()
            .flat_map(|(correct, misspellings)| {
                misspellings
                    .iter()
                    .map(move |(misspelling, freq)| (misspelling.as_str(), correct.as_str(), *freq))
            })
            .collect()
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::io::Write;
    use tempfile::NamedTempFile;

    #[test]
    fn test_big_txt_corpus() {
        let mut file = NamedTempFile::new().expect("test fixture: tempfile must be Ok");
        writeln!(file, "the").expect("test fixture: write must succeed");
        writeln!(file, "the").expect("test fixture: write must succeed");
        writeln!(file, "the").expect("test fixture: write must succeed");
        writeln!(file, "quick").expect("test fixture: write must succeed");
        writeln!(file, "brown").expect("test fixture: write must succeed");
        writeln!(file, "").expect("test fixture: write must succeed"); // Empty line
        file.flush().expect("test fixture: flush must succeed");

        let corpus = BigTxtCorpus::load(file.path()).expect("test fixture: load must be Ok");

        assert_eq!(corpus.unique_words(), 3);
        assert_eq!(corpus.total_tokens(), 5);
        assert_eq!(corpus.frequency("the"), 3);
        assert_eq!(corpus.frequency("quick"), 1);
        assert_eq!(corpus.frequency("missing"), 0);

        let by_freq = corpus.words_by_frequency();
        assert_eq!(by_freq[0], ("the", 3));
    }

    #[test]
    fn test_mitton_corpus() {
        let mut file = NamedTempFile::new().expect("test fixture: tempfile must be Ok");
        writeln!(file, "$hello").expect("test fixture: write must succeed");
        writeln!(file, "helo 2").expect("test fixture: write must succeed");
        writeln!(file, "hllo 1").expect("test fixture: write must succeed");
        writeln!(file, "").expect("test fixture: write must succeed"); // Empty line
        writeln!(file, "$world").expect("test fixture: write must succeed");
        writeln!(file, "wrld").expect("test fixture: write must succeed"); // No frequency (default 1)
        file.flush().expect("test fixture: flush must succeed");

        let corpus = MittonCorpus::load(file.path()).expect("test fixture: load must be Ok");

        assert_eq!(corpus.num_correct_words(), 2);
        assert_eq!(corpus.unique_misspellings(), 3);
        assert_eq!(corpus.total_misspellings(), 4); // 2 + 1 + 1

        let hello_errors = &corpus.errors["hello"];
        assert_eq!(hello_errors.len(), 2);
        assert!(hello_errors.contains(&("helo".to_string(), 2)));
        assert!(hello_errors.contains(&("hllo".to_string(), 1)));

        let world_errors = &corpus.errors["world"];
        assert_eq!(world_errors.len(), 1);
        assert!(world_errors.contains(&("wrld".to_string(), 1)));

        let all = corpus.all_errors();
        assert_eq!(all.len(), 3);
    }

    #[test]
    fn test_mitton_corpus_missing_frequency() {
        let mut file = NamedTempFile::new().expect("test fixture: tempfile must be Ok");
        writeln!(file, "$test").expect("test fixture: write must succeed");
        writeln!(file, "tset").expect("test fixture: write must succeed");
        file.flush().expect("test fixture: flush must succeed");

        let corpus = MittonCorpus::load(file.path()).expect("test fixture: load must be Ok");

        let errors = &corpus.errors["test"];
        assert_eq!(errors.len(), 1);
        assert_eq!(errors[0], ("tset".to_string(), 1));
    }
}
