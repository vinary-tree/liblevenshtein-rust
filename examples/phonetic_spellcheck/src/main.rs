//! Phonetic Spellcheck Demo
//!
//! An interactive spell checker combining phonetic normalization with
//! Damerau-Levenshtein (transposition) distance for robust fuzzy matching.
//!
//! # Features
//!
//! - **Phonetic Normalization**: Uses 13 Zompist rules from `.llev` file
//! - **Transposition Support**: Catches adjacent character swaps ("teh" → "the")
//! - **Max Edit Distance 2**: Balances accuracy with result relevance
//! - **Interactive REPL**: Type queries and see matches in real-time
//!
//! # Usage
//!
//! ```bash
//! cd examples/phonetic_spellcheck
//! make run
//! ```
//!
//! # Algorithm
//!
//! 1. Load dictionary from `data/english_words.txt`
//! 2. Parse phonetic rules from `rules/zompist.llev`
//! 3. Normalize each dictionary term with phonetic rules
//! 4. Build transducer with `Algorithm::Transposition`
//! 5. For each query:
//!    a. Normalize query with same rules
//!    b. Search transducer with max distance 2
//!    c. Display original terms (not normalized) with distances

use std::collections::HashMap;
use std::error::Error;
use std::fs;
use std::io::{self, BufRead, Write};
use std::path::PathBuf;

use liblevenshtein::prelude::*;
use liblevenshtein::phonetic::{
    apply_rules_seq, parse_str, Phone, RewriteRule, RuleSet, MAX_EXPANSION_FACTOR,
};
use std::path::Path;
use liblevenshtein::transducer::Algorithm;

/// Result type for this example
type Result<T> = std::result::Result<T, Box<dyn Error>>;

/// Maximum edit distance for fuzzy matching
const MAX_DISTANCE: usize = 2;

/// Maximum number of results to display
const MAX_RESULTS: usize = 20;

fn main() {
    if let Err(e) = run() {
        eprintln!("Error: {}", e);
        std::process::exit(1);
    }
}

fn run() -> Result<()> {
    // Find the example directory (supports running from various locations)
    let example_dir = find_example_dir()?;

    print_banner();

    // Load dictionary
    let dict_path = example_dir.join("data/english_words.txt");
    println!("Loading dictionary from {}...", dict_path.display());
    let dictionary = load_dictionary(&dict_path)?;
    println!("  Loaded {} terms", dictionary.len());

    // Load phonetic rules from multiple files
    let rules_dir = example_dir.join("rules");
    println!("Loading phonetic rules from {}...", rules_dir.display());
    let rules = load_rules(&rules_dir)?;
    println!("  Loaded {} rules total", rules.len());

    // Normalize dictionary and build index
    println!("Normalizing dictionary with phonetic rules...");
    let (normalized_to_original, transducer) = build_index(&dictionary, &rules)?;
    println!("  Built transducer with {} normalized forms", normalized_to_original.len());
    println!();

    println!("Using Damerau-Levenshtein (transposition) with max distance {}", MAX_DISTANCE);
    println!();
    println!("Type a misspelled word to find matches. Commands:");
    println!("  exit, quit, q - Exit the program");
    println!("  help, ?       - Show this help");
    println!();

    // Interactive query loop
    let stdin = io::stdin();
    let mut stdout = io::stdout();

    loop {
        print!("query> ");
        stdout.flush()?;

        let mut input = String::new();
        if stdin.lock().read_line(&mut input)? == 0 {
            // EOF
            println!();
            break;
        }

        let query = input.trim();
        if query.is_empty() {
            continue;
        }

        // Handle commands
        match query.to_lowercase().as_str() {
            "exit" | "quit" | "q" => {
                println!("Goodbye!");
                break;
            }
            "help" | "?" => {
                print_help();
                continue;
            }
            _ => {}
        }

        // Normalize the query
        let normalized_query = normalize(query, &rules);

        println!();
        println!(
            "Matches for \"{}\" (normalized: \"{}\"):",
            query, normalized_query
        );

        // Query the transducer with distance information
        let candidates: Vec<_> = transducer
            .query_with_distance(&normalized_query, MAX_DISTANCE)
            .collect();

        if candidates.is_empty() {
            println!("  No matches found within distance {}", MAX_DISTANCE);
        } else {
            // Map normalized matches back to original terms and deduplicate
            let mut seen = std::collections::HashSet::new();
            let mut results: Vec<(String, usize)> = Vec::new();

            for candidate in &candidates {
                if let Some(originals) = normalized_to_original.get(&candidate.term) {
                    for original in originals {
                        if seen.insert(original.clone()) {
                            results.push((original.clone(), candidate.distance));
                        }
                    }
                }
            }

            // Sort by distance, then alphabetically
            results.sort_by(|a, b| a.1.cmp(&b.1).then_with(|| a.0.cmp(&b.0)));
            results.truncate(MAX_RESULTS);

            for (i, (term, distance)) in results.iter().enumerate() {
                let note = if distance == &0 {
                    " (exact phonetic match)"
                } else {
                    ""
                };
                println!("  {}. {} (distance: {}){}", i + 1, term, distance, note);
            }
        }
        println!();
    }

    Ok(())
}

/// Find the example directory by checking various locations
fn find_example_dir() -> Result<PathBuf> {
    let candidates = [
        PathBuf::from("."),
        PathBuf::from("examples/phonetic_spellcheck"),
        PathBuf::from("../phonetic_spellcheck"),
    ];

    for candidate in &candidates {
        let rules_path = candidate.join("rules/zompist.llev");
        if rules_path.exists() {
            return Ok(candidate.clone());
        }
    }

    // Try to find from CARGO_MANIFEST_DIR
    if let Ok(manifest_dir) = std::env::var("CARGO_MANIFEST_DIR") {
        let path = PathBuf::from(manifest_dir).join("examples/phonetic_spellcheck");
        if path.join("rules/zompist.llev").exists() {
            return Ok(path);
        }
    }

    Err("Could not find example directory. Please run from examples/phonetic_spellcheck or project root.".into())
}

/// Load dictionary from a text file (one term per line)
fn load_dictionary(path: &PathBuf) -> Result<Vec<String>> {
    let content = fs::read_to_string(path)?;
    let terms: Vec<String> = content
        .lines()
        .filter(|line| !line.is_empty())
        .map(|line| line.trim().to_lowercase())
        .filter(|line| line.chars().all(|c| c.is_ascii_alphabetic()))
        .collect();
    Ok(terms)
}

/// Load phonetic rules from multiple .llev files and merge them
fn load_rules(rules_dir: &Path) -> Result<Vec<RewriteRule>> {
    // Load slang first so text-speak abbreviations have priority
    let rule_files = ["slang.llev", "zompist.llev"];
    let mut combined = RuleSet::default();

    for filename in &rule_files {
        let path = rules_dir.join(filename);
        if path.exists() {
            let content = fs::read_to_string(&path)?;
            let llev_file = parse_str(&content)?;
            let ruleset = RuleSet::from_llev(&llev_file)?;
            println!("    {} ({} rules)", filename, ruleset.len());
            combined.merge(ruleset);
        }
    }

    Ok(combined.rules)
}

/// Build the search index: normalize terms and create transducer
fn build_index(
    dictionary: &[String],
    rules: &[RewriteRule],
) -> Result<(HashMap<String, Vec<String>>, Transducer<DynamicDawg>)> {
    // Normalize each term and build reverse mapping
    let mut normalized_to_original: HashMap<String, Vec<String>> = HashMap::new();

    for term in dictionary {
        let normalized = normalize(term, rules);
        normalized_to_original
            .entry(normalized)
            .or_default()
            .push(term.clone());
    }

    // Build transducer on normalized forms
    // Note: DynamicDawg handles large dictionaries better than DoubleArrayTrie
    let normalized_terms: Vec<String> = normalized_to_original.keys().cloned().collect();
    let mut dict = DynamicDawg::new();
    for term in &normalized_terms {
        dict.insert(term);
    }
    let transducer = Transducer::new(dict, Algorithm::Transposition);

    Ok((normalized_to_original, transducer))
}

/// Normalize a string using phonetic rules
fn normalize(text: &str, rules: &[RewriteRule]) -> String {
    let phones = string_to_phones(text);
    let fuel = phones.len().max(1) * rules.len().max(1) * MAX_EXPANSION_FACTOR;

    match apply_rules_seq(rules, &phones, fuel) {
        Some(result) => phones_to_string(&result),
        None => text.to_string(), // Fallback if fuel exhausted
    }
}

/// Convert a string to a vector of Phones
fn string_to_phones(s: &str) -> Vec<Phone> {
    s.bytes()
        .map(|b| {
            let lower = b.to_ascii_lowercase();
            if matches!(lower, b'a' | b'e' | b'i' | b'o' | b'u') {
                Phone::Vowel(lower)
            } else if b.is_ascii_alphabetic() {
                Phone::Consonant(lower)
            } else {
                Phone::Consonant(b)
            }
        })
        .collect()
}

/// Convert a vector of Phones back to a string
fn phones_to_string(phones: &[Phone]) -> String {
    phones
        .iter()
        .filter_map(|p| match p {
            Phone::Vowel(c) | Phone::Consonant(c) => Some(*c as char),
            Phone::Digraph(c1, _c2) => {
                // Return first character for digraphs (simplified)
                Some(*c1 as char)
            }
            Phone::Silent => None,
        })
        .collect()
}

/// Simple Levenshtein distance calculation
fn levenshtein_distance(a: &str, b: &str) -> usize {
    let a_chars: Vec<char> = a.chars().collect();
    let b_chars: Vec<char> = b.chars().collect();
    let m = a_chars.len();
    let n = b_chars.len();

    if m == 0 {
        return n;
    }
    if n == 0 {
        return m;
    }

    let mut dp = vec![vec![0usize; n + 1]; m + 1];

    for i in 0..=m {
        dp[i][0] = i;
    }
    for j in 0..=n {
        dp[0][j] = j;
    }

    for i in 1..=m {
        for j in 1..=n {
            let cost = if a_chars[i - 1] == b_chars[j - 1] { 0 } else { 1 };
            dp[i][j] = (dp[i - 1][j] + 1)
                .min(dp[i][j - 1] + 1)
                .min(dp[i - 1][j - 1] + cost);
        }
    }

    dp[m][n]
}

fn print_banner() {
    println!();
    println!("╔══════════════════════════════════════════════════════════════════╗");
    println!("║           Phonetic Spellcheck Demo                               ║");
    println!("║   Combining phonetic normalization with Levenshtein distance     ║");
    println!("╚══════════════════════════════════════════════════════════════════╝");
    println!();
}

fn print_help() {
    println!();
    println!("Commands:");
    println!("  <word>        - Search for matches to the given word");
    println!("  exit, quit, q - Exit the program");
    println!("  help, ?       - Show this help");
    println!();
    println!("The spellchecker normalizes both your query and the dictionary");
    println!("using 13 Zompist phonetic rules, then finds matches within");
    println!("edit distance {} using Damerau-Levenshtein (supports transpositions).", MAX_DISTANCE);
    println!();
    println!("Examples:");
    println!("  fone       -> phone (ph->f normalization)");
    println!("  teh        -> the (transposition: e<->h)");
    println!("  filosofy   -> philosophy (ph->f + silent e)");
    println!("  enuf       -> enough (gh->silent, final e->silent)");
    println!();
}
