//! Corpus parsers for loading test data.
//!
//! This module provides parsers for standard spelling correction test corpora.

use std::collections::HashMap;
use std::fs::File;
#[cfg(feature = "grep-archives")]
use std::io::Read;
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

        words.sort_unstable_by_key(|entry| std::cmp::Reverse(entry.1));
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

        Self::load_reader(reader)
    }

    fn load_reader<R: BufRead>(reader: R) -> Result<Self> {
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

    /// Load the original Birkbeck `.643` spelling-error files from their ZIP
    /// archive.
    ///
    /// The original archive is heterogeneous rather than one interchange
    /// format. This loader recognizes every member whose records carry an
    /// explicit correction: fixed-column pairs, comma-separated pair lists,
    /// dollar-prefixed correction sections, and pair-plus-context records.
    /// Source-text and numeric-answer-sheet members whose intended corrections
    /// are not encoded in the member are deliberately ignored; guessing a
    /// correction from prose would corrupt corpus evidence.
    ///
    /// Documentation members are ignored. The archive is bounded to 10,000
    /// members and 64 MiB of declared uncompressed data so an untrusted ZIP
    /// cannot silently expand without limit. Every explicit source pair counts
    /// as one observation even when the archive also records a frequency.
    #[cfg(feature = "grep-archives")]
    pub fn load_birkbeck_zip<P: AsRef<Path>>(path: P) -> Result<Self> {
        const MAX_MEMBERS: usize = 10_000;
        const MAX_UNCOMPRESSED_BYTES: u64 = 64 * 1024 * 1024;

        let file = File::open(path)?;
        let mut archive = zip::ZipArchive::new(file).map_err(std::io::Error::other)?;
        if archive.len() > MAX_MEMBERS {
            return Err(std::io::Error::new(
                std::io::ErrorKind::InvalidData,
                "Birkbeck archive contains too many members",
            ));
        }

        let mut declared_bytes = 0u64;
        let mut errors: HashMap<String, Vec<(String, usize)>> = HashMap::new();
        for index in 0..archive.len() {
            let entry = archive.by_index(index).map_err(std::io::Error::other)?;
            let name = entry.name().to_ascii_uppercase();
            if !name.ends_with("DAT.643") || name.contains("DOC.643") {
                continue;
            }

            declared_bytes = declared_bytes.checked_add(entry.size()).ok_or_else(|| {
                std::io::Error::new(
                    std::io::ErrorKind::InvalidData,
                    "Birkbeck archive size overflow",
                )
            })?;
            if declared_bytes > MAX_UNCOMPRESSED_BYTES {
                return Err(std::io::Error::new(
                    std::io::ErrorKind::InvalidData,
                    "Birkbeck archive exceeds the uncompressed-size limit",
                ));
            }

            let mut text = String::new();
            BufReader::new(entry).read_to_string(&mut text)?;
            parse_birkbeck_member(&name, &text, &mut errors);
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

#[cfg(feature = "grep-archives")]
#[derive(Clone, Copy)]
enum PairDirection {
    CorrectThenMisspelling,
    MisspellingThenCorrect,
}

#[cfg(feature = "grep-archives")]
fn normalize_birkbeck_token(token: &str) -> Option<String> {
    let token = token
        .trim_matches(|character: char| matches!(character, ',' | '.' | ';' | ':' | '(' | ')'));
    if token.is_empty()
        || !token
            .chars()
            .any(|character| character.is_ascii_alphabetic())
        || !token.chars().all(|character| {
            character.is_ascii_alphabetic() || matches!(character, '\'' | '-' | '_')
        })
    {
        return None;
    }
    Some(token.to_ascii_lowercase())
}

#[cfg(feature = "grep-archives")]
fn add_birkbeck_pair(
    errors: &mut HashMap<String, Vec<(String, usize)>>,
    first: &str,
    second: &str,
    direction: PairDirection,
) {
    let (correct, misspelling) = match direction {
        PairDirection::CorrectThenMisspelling => (first, second),
        PairDirection::MisspellingThenCorrect => (second, first),
    };
    let (Some(correct), Some(misspelling)) = (
        normalize_birkbeck_token(correct),
        normalize_birkbeck_token(misspelling),
    ) else {
        return;
    };
    if correct == misspelling {
        return;
    }
    errors.entry(correct).or_default().push((misspelling, 1));
}

#[cfg(feature = "grep-archives")]
fn parse_whitespace_pair(
    line: &str,
    direction: PairDirection,
    errors: &mut HashMap<String, Vec<(String, usize)>>,
) {
    let mut fields = line.split_whitespace();
    let (Some(first), Some(second)) = (fields.next(), fields.next()) else {
        return;
    };
    add_birkbeck_pair(errors, first, second, direction);
}

#[cfg(feature = "grep-archives")]
fn parse_comma_pairs(
    line: &str,
    direction: PairDirection,
    errors: &mut HashMap<String, Vec<(String, usize)>>,
) {
    for record in line.split(',') {
        let mut fields = record
            .trim()
            .trim_start_matches(['+', '$'])
            .split_whitespace();
        let (Some(first), Some(second)) = (fields.next(), fields.next()) else {
            continue;
        };
        add_birkbeck_pair(errors, first, second, direction);
    }
}

#[cfg(feature = "grep-archives")]
fn parse_birkbeck_member(
    name: &str,
    text: &str,
    errors: &mut HashMap<String, Vec<(String, usize)>>,
) {
    match name {
        "ABODAT.643" | "SUOMIDAT.643" => {
            for line in text
                .lines()
                .filter(|line| !line.trim_start().starts_with('$'))
            {
                parse_comma_pairs(line, PairDirection::MisspellingThenCorrect, errors);
            }
        }
        "APPLING1DAT.643" | "APPLING2DAT.643" | "EXAMSDAT.643" | "PETERS1ADAT.643"
        | "PETERS2DAT.643" | "TELEMARKDAT.643" => {
            for line in text
                .lines()
                .filter(|line| !line.trim_start().starts_with('$'))
            {
                parse_whitespace_pair(line, PairDirection::MisspellingThenCorrect, errors);
            }
        }
        "FAWTHROP1DAT.643" | "FAWTHROP2DAT.643" | "SHEFFIELDDAT.643" | "UPWARDDAT.643" => {
            for line in text.lines() {
                parse_whitespace_pair(line, PairDirection::CorrectThenMisspelling, errors);
            }
        }
        "BLOORDAT.643" => {
            for line in text.lines() {
                let fields: Vec<_> = line.split_whitespace().collect();
                if fields.len() < 3 || fields[1].parse::<usize>().is_err() {
                    continue;
                }
                for misspelling in fields[2..]
                    .iter()
                    .take_while(|field| !field.starts_with('['))
                {
                    add_birkbeck_pair(
                        errors,
                        fields[0],
                        misspelling,
                        PairDirection::CorrectThenMisspelling,
                    );
                }
            }
        }
        "GATESDAT.643" => {
            for line in text.lines() {
                let mut fields = line.split_whitespace();
                let Some(correct) = fields.next() else {
                    continue;
                };
                for field in fields {
                    if let Some(misspelling) = field.strip_prefix('*') {
                        add_birkbeck_pair(
                            errors,
                            correct,
                            misspelling,
                            PairDirection::CorrectThenMisspelling,
                        );
                    }
                }
            }
        }
        "MASTERSDAT.643" => {
            let mut correct = None;
            for line in text.lines() {
                let mut fields = line.split_whitespace();
                let Some(first) = fields.next() else {
                    continue;
                };
                if let Some(section) = first.strip_prefix('$') {
                    correct = normalize_birkbeck_token(section);
                } else if let Some(correct) = correct.as_deref() {
                    add_birkbeck_pair(
                        errors,
                        correct,
                        first,
                        PairDirection::CorrectThenMisspelling,
                    );
                }
            }
        }
        "PERIN1DAT.643" => {
            let mut explicit_pair_list = false;
            for line in text.lines() {
                let trimmed = line.trim_start();
                if trimmed.starts_with("$ ") {
                    explicit_pair_list = true;
                    parse_comma_pairs(trimmed, PairDirection::MisspellingThenCorrect, errors);
                } else if explicit_pair_list && trimmed.starts_with('+') {
                    parse_comma_pairs(trimmed, PairDirection::MisspellingThenCorrect, errors);
                } else if trimmed.contains('!') {
                    explicit_pair_list = false;
                }
            }
        }
        "PERIN2DAT.643" => {
            for line in text.lines() {
                let trimmed = line.trim_start();
                if !trimmed.starts_with('$') {
                    parse_comma_pairs(trimmed, PairDirection::CorrectThenMisspelling, errors);
                }
            }
        }
        "TESDELLDAT.643" => {
            for line in text.lines() {
                let trimmed = line.trim_start();
                let records = if trimmed.starts_with('+') {
                    trimmed.trim_start_matches('+').trim_start().to_owned()
                } else {
                    trimmed
                        .split_whitespace()
                        .skip(4)
                        .collect::<Vec<_>>()
                        .join(" ")
                };
                parse_comma_pairs(&records, PairDirection::MisspellingThenCorrect, errors);
            }
        }
        "WINGDAT.643" => {
            for line in text
                .lines()
                .filter(|line| !line.trim_start().starts_with('$'))
            {
                parse_whitespace_pair(line, PairDirection::MisspellingThenCorrect, errors);
            }
        }
        // These members contain raw prose or numeric answer-sheet codes without
        // an explicit correction in the same member. Treating adjacent words as
        // a pair would invent evidence.
        "ASHFORDDAT.643" | "CHESDAT.643" | "HOLBROOKDAT.643" | "NFER1DAT.643" | "NFER2DAT.643"
        | "PERIN3DAT.643" | "PETERS1DAT.643" | "SAMPLESDAT.643" => {}
        _ => {}
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
        writeln!(file).expect("test fixture: write must succeed"); // Empty line
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
        writeln!(file).expect("test fixture: write must succeed"); // Empty line
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

    #[cfg(feature = "grep-archives")]
    #[test]
    fn birkbeck_member_parsers_respect_documented_directions_and_shapes() {
        let mut errors = HashMap::new();

        parse_birkbeck_member("FAWTHROP2DAT.643", "ABILITY ABLITY 12\n", &mut errors);
        parse_birkbeck_member(
            "ABODAT.643",
            "$group\ncaugt caught, choped chopped 7,\n",
            &mut errors,
        );
        parse_birkbeck_member(
            "MASTERSDAT.643",
            "$absurd 1 2 3\nabserd 9 3 2\n",
            &mut errors,
        );
        parse_birkbeck_member(
            "GATESDAT.643",
            "ability $abilIty 61 *abbility 23 *abilaty 42\n",
            &mut errors,
        );

        assert_eq!(errors["caught"], vec![("caugt".to_owned(), 1)]);
        assert_eq!(errors["chopped"], vec![("choped".to_owned(), 1)]);
        assert_eq!(errors["absurd"], vec![("abserd".to_owned(), 1)]);
        assert_eq!(
            errors["ability"],
            vec![
                ("ablity".to_owned(), 1),
                ("abbility".to_owned(), 1),
                ("abilaty".to_owned(), 1),
            ]
        );
    }

    #[cfg(feature = "grep-archives")]
    #[test]
    fn birkbeck_raw_prose_and_encoded_answer_sheets_do_not_invent_pairs() {
        let mut errors = HashMap::new();
        parse_birkbeck_member(
            "HOLBROOKDAT.643",
            "I have four in my Family Dad Mum and siter.\n",
            &mut errors,
        );
        parse_birkbeck_member(
            "NFER1DAT.643",
            "21003 10 oar 11 too 12 suns !\n",
            &mut errors,
        );
        assert!(errors.is_empty());
    }
}
