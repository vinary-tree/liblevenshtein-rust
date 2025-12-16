//! Phonetic Spellcheck Demo
//!
//! An interactive spell checker combining phonetic normalization with
//! Damerau-Levenshtein (transposition) distance for robust fuzzy matching.
//!
//! # Features
//!
//! - **Phonetic Normalization**: Uses Zompist rules from `.llev` files
//! - **Transposition Support**: Catches adjacent character swaps ("teh" → "the")
//! - **AOT Compilation**: Pre-compile rules for instant startup
//! - **Memoized Matching**: Cache repeated queries for instant responses
//! - **Cycle Detection**: Safe handling of rule pathologies
//! - **Interactive REPL**: Type queries and see matches in real-time
//!
//! # Usage
//!
//! ```bash
//! cd examples/phonetic_spellcheck
//! make run              # Normal mode
//! make aot              # Pre-compile rules
//! make run-fast         # Run with cached rules
//! ```

use clap::Parser;
use serde::{Deserialize, Serialize};
use std::collections::HashMap;
use std::error::Error;
use std::fs;
use std::hash::{Hash, Hasher};
use std::io::{self, BufRead, Write};
use std::path::{Path, PathBuf};
use std::time::{Duration, Instant};

// DynamicDawg is imported via prelude
use liblevenshtein::phonetic::{
    apply_rules_seq, apply_rules_with_cycle_detection, parse_str, NormalizationResult,
    Phone, RewriteRule, RuleSet, MAX_EXPANSION_FACTOR,
};
use liblevenshtein::phonetic::llev::{load as load_ruleset, save as save_ruleset, LLevError};
use liblevenshtein::prelude::*;
use liblevenshtein::transducer::Algorithm;

/// Result type for this example
type Result<T> = std::result::Result<T, Box<dyn Error>>;

/// Maximum edit distance for fuzzy matching
const MAX_DISTANCE: usize = 2;

/// Maximum number of results to display
const MAX_RESULTS: usize = 20;

/// Default LRU cache size
const DEFAULT_CACHE_SIZE: usize = 1000;

#[derive(Parser, Debug)]
#[command(name = "phonetic_spellcheck")]
#[command(about = "Interactive phonetic spell checker with Levenshtein distance")]
#[command(version)]
struct Args {
    /// Pre-compile rules to binary cache for faster startup
    #[arg(long)]
    aot: bool,

    /// Show normalization steps for each query
    #[arg(short, long)]
    verbose: bool,

    /// Show performance statistics on exit
    #[arg(long)]
    stats: bool,

    /// LRU cache size for memoized queries
    #[arg(long, default_value_t = DEFAULT_CACHE_SIZE)]
    cache_size: usize,

    /// Use pre-compiled binary cache if available
    #[arg(long)]
    use_cache: bool,

    /// Use cycle detection when applying rules
    #[arg(long)]
    detect_cycles: bool,
}

/// Performance statistics
#[derive(Default)]
struct Stats {
    rule_loading_time: Duration,
    dictionary_loading_time: Duration,
    normalization_time: Duration,
    transducer_build_time: Duration,
    query_count: usize,
    cache_hits: usize,
    cache_misses: usize,
    cycles_detected: usize,
}

impl Stats {
    fn total_startup_time(&self) -> Duration {
        self.rule_loading_time
            + self.dictionary_loading_time
            + self.normalization_time
            + self.transducer_build_time
    }

    fn hit_rate(&self) -> f64 {
        if self.query_count == 0 {
            0.0
        } else {
            self.cache_hits as f64 / self.query_count as f64 * 100.0
        }
    }

    fn print(&self) {
        println!();
        println!("╔════════════════════════════════════════╗");
        println!("║         Performance Statistics         ║");
        println!("╠════════════════════════════════════════╣");
        println!("║ Startup Timing:                        ║");
        println!("║   Rule loading:      {:>10.2?}        ║", self.rule_loading_time);
        println!("║   Dictionary load:   {:>10.2?}        ║", self.dictionary_loading_time);
        println!("║   Normalization:     {:>10.2?}        ║", self.normalization_time);
        println!("║   Transducer build:  {:>10.2?}        ║", self.transducer_build_time);
        println!("║   ─────────────────────────────        ║");
        println!("║   Total:             {:>10.2?}        ║", self.total_startup_time());
        println!("╠════════════════════════════════════════╣");
        println!("║ Query Statistics:                      ║");
        println!("║   Total queries:     {:>10}        ║", self.query_count);
        println!("║   Cache hits:        {:>10}        ║", self.cache_hits);
        println!("║   Cache misses:      {:>10}        ║", self.cache_misses);
        println!("║   Hit rate:          {:>9.1}%        ║", self.hit_rate());
        if self.cycles_detected > 0 {
            println!("║   Cycles detected:   {:>10}        ║", self.cycles_detected);
        }
        println!("╚════════════════════════════════════════╝");
    }
}

/// Cached normalized dictionary index for fast startup.
///
/// This struct serializes the normalized dictionary mapping so that subsequent
/// runs can skip the expensive normalization step (~1-2s → ~50ms).
#[derive(Serialize, Deserialize)]
struct NormalizedIndex {
    /// Map from normalized phonetic form to original dictionary words.
    /// Stored as Vec for serialization; converted to DynamicDawg on load.
    entries: Vec<(String, Vec<String>)>,
    /// Checksum of the source dictionary file for cache invalidation.
    dictionary_checksum: u64,
    /// Checksum of the rules files for cache invalidation.
    rules_checksum: u64,
    /// Cache format version for compatibility checking.
    version: u32,
}

impl NormalizedIndex {
    /// Current cache format version. Increment when the format changes.
    const CURRENT_VERSION: u32 = 2;  // Bumped for DynamicDawg change

    /// Create a new index from a HashMap of entries.
    fn from_entries(
        entries: HashMap<String, Vec<String>>,
        dictionary_checksum: u64,
        rules_checksum: u64,
    ) -> Self {
        Self {
            entries: entries.into_iter().collect(),
            dictionary_checksum,
            rules_checksum,
            version: Self::CURRENT_VERSION,
        }
    }

    /// Convert the cached entries into a DynamicDawg for fast lookups.
    /// DynamicDawg provides O(k) lookup where k = key length, plus
    /// native serialization support for fast cache loading.
    fn into_dawg(self) -> DynamicDawg<Vec<String>> {
        let dawg = DynamicDawg::new();
        for (normalized, originals) in self.entries {
            dawg.insert_with_value(&normalized, originals);
        }
        dawg
    }

    /// Check if this cache is still valid given current checksums.
    fn is_valid(&self, current_dict_checksum: u64, current_rules_checksum: u64) -> bool {
        self.version == Self::CURRENT_VERSION
            && self.dictionary_checksum == current_dict_checksum
            && self.rules_checksum == current_rules_checksum
    }

    /// Save the index to a binary file.
    fn save(&self, path: &Path) -> Result<()> {
        let file = fs::File::create(path)?;
        let writer = io::BufWriter::new(file);
        bincode::serialize_into(writer, self)?;
        Ok(())
    }

    /// Load an index from a binary file.
    fn load(path: &Path) -> Result<Self> {
        let file = fs::File::open(path)?;
        let reader = io::BufReader::new(file);
        let index: Self = bincode::deserialize_from(reader)?;
        Ok(index)
    }
}

/// Compute a checksum for a file's contents.
fn compute_file_checksum(path: &Path) -> io::Result<u64> {
    let content = fs::read(path)?;
    let mut hasher = std::collections::hash_map::DefaultHasher::new();
    content.hash(&mut hasher);
    Ok(hasher.finish())
}

/// Compute a combined checksum for multiple files in a directory.
fn compute_rules_checksum(rules_dir: &Path) -> io::Result<u64> {
    let rule_files = ["text_speak.llev", "homophones.llev", "zompist.llev"];
    let mut hasher = std::collections::hash_map::DefaultHasher::new();

    for filename in &rule_files {
        let path = rules_dir.join(filename);
        if path.exists() {
            let content = fs::read(&path)?;
            content.hash(&mut hasher);
        }
    }

    Ok(hasher.finish())
}

/// Memoized spell checker with LRU-style cache
///
/// Uses DynamicDawg for the normalized-to-original mapping for fast cache loading
/// and O(k) lookups where k = key length.
struct MemoizedSpellChecker {
    transducer: Transducer<DynamicDawg>,
    /// Maps normalized phonetic forms back to original dictionary words.
    normalized_to_original: DynamicDawg<Vec<String>>,
    /// Query result cache with simple LRU-style eviction (clear on full).
    cache: HashMap<String, Vec<(String, usize)>>,
    cache_size: usize,
}

impl MemoizedSpellChecker {
    fn new(
        transducer: Transducer<DynamicDawg>,
        normalized_to_original: DynamicDawg<Vec<String>>,
        cache_size: usize,
    ) -> Self {
        Self {
            transducer,
            normalized_to_original,
            cache: HashMap::with_capacity(cache_size),
            cache_size,
        }
    }

    fn query(&mut self, normalized_query: &str, stats: &mut Stats) -> Vec<(String, usize)> {
        stats.query_count += 1;

        // Check cache first
        if let Some(cached) = self.cache.get(normalized_query) {
            stats.cache_hits += 1;
            return cached.clone();
        }

        stats.cache_misses += 1;

        // Compute result
        let candidates: Vec<_> = self.transducer
            .query_with_distance(normalized_query, MAX_DISTANCE)
            .collect();

        // Map normalized matches back to original terms using DynamicDawg
        let mut seen = std::collections::HashSet::new();
        let mut results: Vec<(String, usize)> = Vec::new();

        for candidate in &candidates {
            if let Some(originals) = self.normalized_to_original.get_value(&candidate.term) {
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

        // Cache result (simple eviction: clear if full)
        if self.cache.len() >= self.cache_size {
            self.cache.clear();
        }
        self.cache.insert(normalized_query.to_string(), results.clone());

        results
    }
}

fn main() {
    let args = Args::parse();

    if let Err(e) = run(args) {
        eprintln!("Error: {}", e);
        std::process::exit(1);
    }
}

fn run(args: Args) -> Result<()> {
    let example_dir = find_example_dir()?;
    let cache_dir = example_dir.join("cache");

    // AOT mode: just compile and save, then exit
    if args.aot {
        return run_aot_compilation(&example_dir, &cache_dir);
    }

    let mut stats = Stats::default();

    print_banner();

    // Load phonetic rules
    let rules = load_rules(&example_dir, &cache_dir, args.use_cache, &mut stats)?;
    println!("  Loaded {} rules", rules.len());

    // Paths for cache validation
    let dict_path = example_dir.join("data/english_words.txt");
    let rules_dir = example_dir.join("rules");
    let index_cache_path = cache_dir.join("index.bin");

    // Try to load cached index if requested
    let (normalized_to_original, transducer) = if args.use_cache && index_cache_path.exists() {
        // Compute current checksums for validation
        let current_dict_checksum = compute_file_checksum(&dict_path).unwrap_or(0);
        let current_rules_checksum = compute_rules_checksum(&rules_dir).unwrap_or(0);

        println!("Loading cached normalized index...");
        let start = Instant::now();

        match NormalizedIndex::load(&index_cache_path) {
            Ok(cached_index) if cached_index.is_valid(current_dict_checksum, current_rules_checksum) => {
                // Cache is valid - use it!
                let entry_count = cached_index.entries.len();
                let load_time = start.elapsed();
                println!("  Loaded {} entries from cache in {:?}", entry_count, load_time);

                // Build transducer directly from cached entries (faster than rebuilding trie first)
                let start = Instant::now();
                let normalized_terms: Vec<&str> = cached_index.entries.iter()
                    .map(|(k, _)| k.as_str())
                    .collect();
                let dict = DynamicDawg::from_terms(normalized_terms.iter().copied());
                let transducer = Transducer::new(dict, Algorithm::Transposition);
                stats.transducer_build_time = start.elapsed();
                println!("  Built transducer in {:?}", stats.transducer_build_time);

                // Build DynamicDawg from cached entries
                let start = Instant::now();
                let normalized_to_original = cached_index.into_dawg();
                stats.normalization_time = start.elapsed();
                println!("  Built lookup dawg in {:?}", stats.normalization_time);

                (normalized_to_original, transducer)
            }
            Ok(_) => {
                // Cache is stale - rebuild
                println!("  Cache is stale (source files changed), rebuilding...");
                build_index_fresh(&dict_path, &rules, args.verbose, &mut stats)?
            }
            Err(e) => {
                // Cache is corrupted - rebuild
                eprintln!("  Warning: Could not load cache: {}", e);
                println!("  Rebuilding from source...");
                build_index_fresh(&dict_path, &rules, args.verbose, &mut stats)?
            }
        }
    } else {
        // No cache requested or available - build fresh
        build_index_fresh(&dict_path, &rules, args.verbose, &mut stats)?
    };

    println!();

    // Create memoized spell checker
    let mut checker = MemoizedSpellChecker::new(
        transducer,
        normalized_to_original,
        args.cache_size,
    );

    println!("Using Damerau-Levenshtein (transposition) with max distance {}", MAX_DISTANCE);
    println!("Query cache size: {}", args.cache_size);
    if args.detect_cycles {
        println!("Cycle detection: enabled");
    }
    if args.verbose {
        println!("Verbose mode: enabled");
    }
    println!();
    println!("Type a misspelled word to find matches. Commands:");
    println!("  exit, quit, q - Exit the program");
    println!("  help, ?       - Show this help");
    println!("  stats         - Show performance statistics");
    println!("  clear         - Clear the query cache");
    println!();

    // Interactive query loop
    let stdin = io::stdin();
    let mut stdout = io::stdout();

    loop {
        print!("query> ");
        stdout.flush()?;

        let mut input = String::new();
        if stdin.lock().read_line(&mut input)? == 0 {
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
                if args.stats {
                    stats.print();
                }
                println!("Goodbye!");
                break;
            }
            "help" | "?" => {
                print_help();
                continue;
            }
            "stats" => {
                stats.print();
                continue;
            }
            "clear" => {
                checker.cache.clear();
                println!("Query cache cleared.");
                println!();
                continue;
            }
            _ => {}
        }

        // Normalize the query
        let (normalized_query, cycle_warning) = normalize(
            query,
            &rules,
            args.detect_cycles,
            args.verbose,
            &mut stats,
        );

        println!();
        println!(
            "Matches for \"{}\" (normalized: \"{}\"):",
            query, normalized_query
        );

        if let Some(warning) = cycle_warning {
            println!("  Warning: {}", warning);
        }

        // Query with memoization
        let results = checker.query(&normalized_query, &mut stats);

        if results.is_empty() {
            println!("  No matches found within distance {}", MAX_DISTANCE);
        } else {
            for (i, (term, distance)) in results.iter().enumerate() {
                let note = if *distance == 0 {
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

/// Run AOT compilation mode
///
/// Pre-compiles both the phonetic rules AND the normalized dictionary index.
/// This provides ~30x faster startup by skipping normalization on subsequent runs.
fn run_aot_compilation(example_dir: &Path, cache_dir: &Path) -> Result<()> {
    println!("╔══════════════════════════════════════════════════════════════════╗");
    println!("║              AOT Compilation Mode                                ║");
    println!("╚══════════════════════════════════════════════════════════════════╝");
    println!();

    // Create cache directory
    fs::create_dir_all(cache_dir)?;

    // Compute checksums for cache invalidation
    let dict_path = example_dir.join("data/english_words.txt");
    let rules_dir = example_dir.join("rules");

    let dict_checksum = compute_file_checksum(&dict_path)
        .unwrap_or_else(|_| {
            eprintln!("  Warning: Could not compute dictionary checksum");
            0
        });
    let rules_checksum = compute_rules_checksum(&rules_dir)
        .unwrap_or_else(|_| {
            eprintln!("  Warning: Could not compute rules checksum");
            0
        });

    // === Step 1: Compile rules ===
    println!("Step 1: Compiling phonetic rules from {}...", rules_dir.display());

    let start = Instant::now();
    let rules = load_rules_from_llev(&rules_dir)?;
    let parse_time = start.elapsed();
    println!("  Parsed {} rules in {:?}", rules.len(), parse_time);

    // Save compiled rules
    let rules_cache_path = cache_dir.join("rules.bin");
    let start = Instant::now();
    let ruleset = RuleSet {
        rules: rules.clone(),
        name: Some("phonetic_spellcheck combined rules".to_string()),
        version: Some("1.0.0".to_string()),
    };
    save_ruleset(&ruleset, &rules_cache_path)?;
    let save_time = start.elapsed();
    println!("  Saved to {} in {:?}", rules_cache_path.display(), save_time);

    // Verify rules by loading
    let start = Instant::now();
    let loaded = load_ruleset(&rules_cache_path)?;
    let rules_load_time = start.elapsed();
    println!("  Verified: loaded {} rules in {:?}", loaded.rules.len(), rules_load_time);
    println!("  Rules speedup: {:.1}x faster loading", parse_time.as_secs_f64() / rules_load_time.as_secs_f64());

    // === Step 2: Normalize dictionary and save index ===
    println!();
    println!("Step 2: Normalizing dictionary with phonetic rules...");

    // Load dictionary
    let start = Instant::now();
    println!("  Loading dictionary from {}...", dict_path.display());
    let dictionary = load_dictionary(&dict_path)?;
    let dict_load_time = start.elapsed();
    println!("  Loaded {} terms in {:?}", dictionary.len(), dict_load_time);

    // Normalize dictionary (fast path, no cycle detection)
    let start = Instant::now();
    let mut entries: HashMap<String, Vec<String>> = HashMap::new();
    for term in &dictionary {
        let normalized = normalize_fast(term, &rules);
        entries.entry(normalized).or_default().push(term.clone());
    }
    let normalize_time = start.elapsed();
    println!("  Normalized {} terms → {} unique forms in {:?}",
             dictionary.len(), entries.len(), normalize_time);

    // Save normalized index
    let index = NormalizedIndex::from_entries(entries, dict_checksum, rules_checksum);
    let index_cache_path = cache_dir.join("index.bin");
    let start = Instant::now();
    index.save(&index_cache_path)?;
    let index_save_time = start.elapsed();
    println!("  Saved index to {} in {:?}", index_cache_path.display(), index_save_time);

    // Verify index by loading
    let start = Instant::now();
    let loaded_index = NormalizedIndex::load(&index_cache_path)?;
    let index_load_time = start.elapsed();
    println!("  Verified: loaded {} entries in {:?}", loaded_index.entries.len(), index_load_time);

    // === Summary ===
    println!();
    println!("╔════════════════════════════════════════╗");
    println!("║       AOT Compilation Complete!        ║");
    println!("╠════════════════════════════════════════╣");
    println!("║ Cache files created:                   ║");
    println!("║   - cache/rules.bin                    ║");
    println!("║   - cache/index.bin                    ║");
    println!("╠════════════════════════════════════════╣");
    println!("║ Performance improvements:              ║");
    let total_cold = parse_time + dict_load_time + normalize_time;
    let total_warm = rules_load_time + index_load_time;
    println!("║   Cold start: {:>10.2?}              ║", total_cold);
    println!("║   Warm start: {:>10.2?}              ║", total_warm);
    println!("║   Speedup:    {:>10.1}x              ║", total_cold.as_secs_f64() / total_warm.as_secs_f64());
    println!("╚════════════════════════════════════════╝");
    println!();
    println!("Run with --use-cache to use the compiled cache:");
    println!("  cargo run --release -- --use-cache");

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

/// Load phonetic rules (from cache or .llev files)
fn load_rules(
    example_dir: &Path,
    cache_dir: &Path,
    use_cache: bool,
    stats: &mut Stats,
) -> Result<Vec<RewriteRule>> {
    let cache_path = cache_dir.join("rules.bin");

    let start = Instant::now();

    if use_cache && cache_path.exists() {
        println!("Loading pre-compiled rules from {}...", cache_path.display());
        let ruleset = load_ruleset(&cache_path)?;
        stats.rule_loading_time = start.elapsed();
        println!("  Loaded in {:?} (cached)", stats.rule_loading_time);
        return Ok(ruleset.rules);
    }

    let rules_dir = example_dir.join("rules");
    println!("Loading phonetic rules from {}...", rules_dir.display());
    let rules = load_rules_from_llev(&rules_dir)?;
    stats.rule_loading_time = start.elapsed();

    Ok(rules)
}

/// Load phonetic rules from .llev files
fn load_rules_from_llev(rules_dir: &Path) -> Result<Vec<RewriteRule>> {
    let rule_files = ["text_speak.llev", "homophones.llev", "zompist.llev"];
    let mut combined = RuleSet::default();

    for filename in &rule_files {
        let path = rules_dir.join(filename);
        if path.exists() {
            let content = fs::read_to_string(&path)?;
            // Parse with filename attached to errors for better diagnostics
            let llev_file = parse_str(&content)
                .map_err(|e: LLevError| e.in_file(path.clone()))?;
            let ruleset = RuleSet::from_llev(&llev_file)?;
            println!("    {} ({} rules)", filename, ruleset.len());
            combined.merge(ruleset);
        }
    }

    Ok(combined.rules)
}

/// Build the search index from fresh data: load dictionary, normalize, create transducer.
///
/// This is the cold-start path when no cache is available or cache is stale.
fn build_index_fresh(
    dict_path: &Path,
    rules: &[RewriteRule],
    verbose: bool,
    stats: &mut Stats,
) -> Result<(DynamicDawg<Vec<String>>, Transducer<DynamicDawg>)> {
    // Load dictionary
    let start = Instant::now();
    println!("Loading dictionary from {}...", dict_path.display());
    let dictionary = load_dictionary(&dict_path.to_path_buf())?;
    stats.dictionary_loading_time = start.elapsed();
    println!("  Loaded {} terms in {:?}", dictionary.len(), stats.dictionary_loading_time);

    // Normalize dictionary
    println!("Normalizing dictionary with phonetic rules...");
    let start = Instant::now();
    let (normalized_to_original, transducer) = build_index(&dictionary, rules, verbose, stats)?;
    stats.normalization_time = start.elapsed();
    println!(
        "  Built transducer with {} normalized forms in {:?}",
        normalized_to_original.len().unwrap_or(0),
        stats.normalization_time
    );

    Ok((normalized_to_original, transducer))
}

/// Build the search index: normalize terms and create transducer
///
/// NOTE: Dictionary normalization ALWAYS uses the fast path (apply_rules_seq) without
/// cycle detection. The --detect-cycles flag only affects individual user queries.
/// This is critical for performance: cycle detection has O(n²) overhead per word,
/// which causes ~124k words × HashSet operations = 5+ minute hang.
fn build_index(
    dictionary: &[String],
    rules: &[RewriteRule],
    verbose: bool,
    stats: &mut Stats,
) -> Result<(DynamicDawg<Vec<String>>, Transducer<DynamicDawg>)> {
    // First pass: collect all (normalized -> original) mappings into a HashMap
    // ALWAYS use fast path for dictionary normalization - cycle detection is only
    // useful for individual user queries where the overhead is negligible
    let mut entries: HashMap<String, Vec<String>> = HashMap::new();

    for term in dictionary {
        let normalized = normalize_fast(term, rules);
        if verbose {
            eprintln!("  '{}' → '{}'", term, normalized);
        }
        entries
            .entry(normalized)
            .or_default()
            .push(term.clone());
    }

    // Build DynamicDawg from collected entries
    // DynamicDawg provides O(k) lookup where k = key length, plus serialization support
    let normalized_to_original: DynamicDawg<Vec<String>> = DynamicDawg::new();
    for (normalized, originals) in entries {
        normalized_to_original.insert_with_value(&normalized, originals);
    }

    // Build transducer from dawg keys
    // DynamicDawg iterator yields (Vec<u8>, V), so we convert bytes to String
    let start = Instant::now();
    let normalized_terms: Vec<String> = (&normalized_to_original)
        .into_iter()
        .map(|(k, _)| String::from_utf8_lossy(&k).into_owned())
        .collect();
    let dict = DynamicDawg::from_terms(normalized_terms.iter().map(|s| s.as_str()));
    let transducer = Transducer::new(dict, Algorithm::Transposition);
    stats.transducer_build_time = start.elapsed();

    Ok((normalized_to_original, transducer))
}

/// Fast normalization without cycle detection overhead.
///
/// This function is used for bulk dictionary normalization where performance is
/// critical. Cycle detection is skipped because:
/// 1. It requires O(n²) HashSet operations per word
/// 2. Dictionary words are unlikely to trigger pathological cycles
/// 3. The fuel limit provides sufficient protection against infinite loops
///
/// Performance: Uses a smaller fuel limit than the theoretical maximum to improve
/// throughput. Most words converge in just a few rule applications.
fn normalize_fast(text: &str, rules: &[RewriteRule]) -> String {
    let phones = string_to_phones(text);
    // Use a smaller fuel limit for dictionary normalization:
    // Most words need < 10 rule applications. Use 50 as a practical limit
    // that catches any pathological cases while maintaining fast throughput.
    let fuel = 50;

    match apply_rules_seq(rules, &phones, fuel) {
        Some(result) => phones_to_string(&result),
        None => text.to_string(), // Fuel exhausted - return original
    }
}

/// Normalize a string using phonetic rules
fn normalize(
    text: &str,
    rules: &[RewriteRule],
    detect_cycles: bool,
    verbose: bool,
    stats: &mut Stats,
) -> (String, Option<String>) {
    let phones = string_to_phones(text);
    let fuel = phones.len().max(1) * rules.len().max(1) * MAX_EXPANSION_FACTOR;

    if detect_cycles {
        match apply_rules_with_cycle_detection(rules, &phones, fuel) {
            NormalizationResult::FixedPoint(result) => {
                let normalized = phones_to_string(&result);
                if verbose {
                    println!("  Normalization: \"{}\" → \"{}\" (fixed point)", text, normalized);
                }
                (normalized, None)
            }
            NormalizationResult::Cycle(equivalence_set) => {
                stats.cycles_detected += 1;
                let canonical = equivalence_set
                    .iter()
                    .min_by_key(|v| v.len())
                    .cloned()
                    .unwrap_or_default();
                let normalized = phones_to_string(&canonical);
                if verbose {
                    println!(
                        "  Normalization: \"{}\" → \"{}\" (cycle: {} forms)",
                        text,
                        normalized,
                        equivalence_set.len()
                    );
                }
                (
                    normalized,
                    Some(format!("Cycle detected ({} equivalent forms)", equivalence_set.len())),
                )
            }
            NormalizationResult::FuelExhausted(result) => {
                let normalized = phones_to_string(&result);
                if verbose {
                    println!("  Normalization: \"{}\" → \"{}\" (fuel exhausted)", text, normalized);
                }
                (normalized, Some("Fuel exhausted".to_string()))
            }
        }
    } else {
        match apply_rules_seq(rules, &phones, fuel) {
            Some(result) => {
                let normalized = phones_to_string(&result);
                if verbose {
                    println!("  Normalization: \"{}\" → \"{}\"", text, normalized);
                }
                (normalized, None)
            }
            None => {
                if verbose {
                    println!("  Normalization: \"{}\" → \"{}\" (fuel exhausted)", text, text);
                }
                (text.to_string(), Some("Fuel exhausted".to_string()))
            }
        }
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
            Phone::Digraph(c1, _c2) => Some(*c1 as char),
            Phone::Silent => None,
        })
        .collect()
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
    println!("  stats         - Show performance statistics");
    println!("  clear         - Clear the query cache");
    println!();
    println!("The spellchecker normalizes both your query and the dictionary");
    println!("using phonetic rules, then finds matches within edit distance {}.", MAX_DISTANCE);
    println!();
    println!("Examples:");
    println!("  fone       -> phone (ph->f normalization)");
    println!("  teh        -> the (transposition: e<->h)");
    println!("  filosofy   -> philosophy (ph->f + silent e)");
    println!("  enuf       -> enough (gh->silent, final e->silent)");
    println!();
    println!("CLI Options:");
    println!("  --aot          Pre-compile rules to binary cache");
    println!("  --use-cache    Use pre-compiled rules (faster startup)");
    println!("  --verbose, -v  Show normalization steps");
    println!("  --stats        Show statistics on exit");
    println!("  --detect-cycles Enable cycle detection in rule application");
    println!("  --cache-size N LRU cache size (default: 1000)");
    println!();
}
