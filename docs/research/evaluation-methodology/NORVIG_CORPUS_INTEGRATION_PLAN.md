# Norvig Corpus Integration - Implementation Plan

**Comprehensive Plan for Literature-Standard Benchmarking and Validation**

**Date**: 2025-11-06
**Status**: Planning Phase - Implementation Pending
**Priority**: High (Priority 1 from Evaluation Methodology)

---

## Table of Contents

1. [Executive Summary](#executive-summary)
2. [Corpus Characteristics](#corpus-characteristics)
3. [Current Infrastructure Analysis](#current-infrastructure-analysis)
4. [Integration Strategy](#integration-strategy)
5. [Implementation Plan](#implementation-plan)
6. [Testing Requirements](#testing-requirements)
7. [Benchmarking Requirements](#benchmarking-requirements)
8. [CI/CD Integration](#cicd-integration)
9. [Expected Benefits](#expected-benefits)
10. [Timeline and Effort](#timeline-and-effort)
11. [Risks and Mitigations](#risks-and-mitigations)
12. [Future Enhancements](#future-enhancements)

---

## Executive Summary

### Purpose

Integrate Peter Norvig's big.txt corpus and the Birkbeck spelling error corpus to enable:
1. **Reproducible benchmarking** with literature-standard datasets
2. **Correctness validation** against real-world spelling errors
3. **Performance measurement** with realistic natural language workloads
4. **Academic credibility** for research publication and peer comparison

### Key Decisions

**Storage Strategy:**
- Corpora downloaded on-demand, **NOT committed** to repository
- GitHub Actions cache for fast CI access (~10-15 MB total)
- Deterministic test data generation with fixed seeds

**Integration Points:**
- New test module: `tests/corpus_validation.rs`
- New benchmark: `benches/corpus_benchmarks.rs`
- New library module: `src/corpus/` (test-only visibility)
- CI job: Separate `corpus-validation` for isolated testing

### Success Criteria

✅ **Correctness:** >95% recall on Birkbeck corpus (36K real errors)
✅ **Performance:** Reproducible benchmarks on big.txt (32K words)
✅ **Automation:** CI runs corpus tests without manual intervention
✅ **Documentation:** Clear setup instructions and usage guide

---

## Corpus Characteristics

### 1. Peter Norvig's big.txt

**Purpose:** Realistic English dictionary for performance testing and language modeling

**Source & Composition:**
- Concatenation of public domain texts from Project Gutenberg
- Most frequent words from Wiktionary
- British National Corpus word frequencies
- Primary source: "The Adventures of Sherlock Holmes" by Arthur Conan Doyle

**Statistics:**
- **File Size:** ~285-310 KB (1.42-1.55 million characters)
- **Total Words:** ~215,000-230,000 tokens
- **Unique Words:** ~32,192 distinct words (per Norvig's documentation)
- **Most Frequent:** "the" appears in ~7% of text
- **Word Length Distribution:**
  - Average: 4.5-5 characters
  - Most common: 2-4 character words (articles, prepositions)
  - Range: 1-20+ characters (Victorian-era vocabulary)
- **Encoding:** ASCII with proper punctuation handling

**Download:** https://norvig.com/big.txt

**Academic Usage:**
- Standard corpus for spelling correction research since 2007
- Used in Norvig's seminal essay "How to Write a Spelling Corrector"
- Training data for Bayesian error models: P(correct|error) ∝ P(error|correct) × P(correct)
- Benchmark for edit distance algorithms and language models

**Word Frequency Distribution:**
```
Rank 1 (the):        ~79,809 occurrences (7%)
Rank 10 (be):        ~9,843 occurrences
Rank 100 (way):      ~1,304 occurrences
Rank 1000 (floor):   ~122 occurrences
Rank 10000 (drank):  ~8 occurrences
```

Follows **Zipf's Law**: frequency ∝ 1/rank

---

### 2. Birkbeck Spelling Error Corpus

**Purpose:** Correctness validation with authentic spelling errors from native speakers

**Source & Composition:**
- Created by Roger Mitton at Birkbeck, University of London
- Available through Oxford Text Archive (OTA ID: 0643)
- Collected from multiple sources:
  - School spelling tests
  - Free writing samples (schoolchildren, ages 8-15)
  - University students' essays
  - Adult literacy students' writing

**Statistics:**
- **Total Errors:** 36,133 misspellings
- **Unique Correct Words:** 6,136 target words
- **Average Errors per Word:** 5.9 misspellings/word
- **Error Distribution:**
  - Distance 1: ~42% of errors
  - Distance 2: ~31% of errors
  - Distance 3: ~15% of errors
  - Distance 4+: ~12% of errors
- **Error Types:**
  - Phonetic errors: "recieve" → "receive"
  - Transposition: "teh" → "the"
  - Insertion/deletion: "untill" → "until"
  - Visual confusion: "gaurd" → "guard"

**File Format:**
```
$correct_word
misspelling1
misspelling2
misspelling3
$next_correct_word
error1
error2
```

- Each correct word preceded by `$`
- Misspellings follow, one per line
- No duplicate errors for same word
- Spaces in multi-word terms replaced with underscores

**Download:** https://titan.dcs.bbk.ac.uk/~roger/corpora.html
(Alternate: https://ota.bodleian.ox.ac.uk/repository/xmlui/handle/20.500.12024/0643)

**Academic Usage:**
- Gold standard for spelling correction evaluation
- Cited in 100+ research papers
- Used to train context-sensitive error models
- Benchmark for phonetic matching algorithms

**Example Entries:**
```
$abandon
abadon
abanddon
abandonn
abbandone
abondon
$absolutely
absolutly
absolutley
absalutly
absolootly
```

---

### 3. Additional Corpora (Optional, Future Work)

**Holbrook Corpus:**
- 1,791 misspellings of 1,200 target words
- Secondary schoolchildren (ages 11-14)
- Tagged format with error type annotations

**Aspell Corpus:**
- 531 misspellings of 450 words
- From GNU Aspell spellchecker test suite
- Technical and dictionary terms

**Wikipedia Corpus:**
- 2,455 misspellings of 1,922 words
- Common misspellings list maintained by Wikipedia editors
- Includes common typos from web searches

---

## Current Infrastructure Analysis

### Test Directory (`/tests/`) - 42 Files

**Existing Patterns:**
- Property-based testing with `proptest` (v1.4)
- Regression test tracking via proptest persistence
- Cross-validation between backends
- Unicode/UTF-8 specific coverage
- Concurrency and stress testing

**Current Test Data Sources:**
- Synthetic word generation: `format!("word{:06}", i)`
- Random character sequences via proptest
- Hard-coded edge cases
- **No external corpus usage** ❌

**Key Files:**
- `integration_tests.rs` - End-to-end integration
- `proptest_automaton_distance_cross_validation.rs` - Cross-validation with naive algorithm
- `utf8_tests.rs` - Unicode handling
- `concurrency_test.rs` - Thread safety

**Gap:** No literature-standard validation dataset

---

### Benchmark Directory (`/benches/`) - 43+ Files

**Existing Patterns:**
- Criterion.rs (v0.5) for statistical analysis
- Backend comparisons: DoubleArrayTrie, DAWG, PathMap, etc.
- SIMD optimization validation (Phase 1-4)
- Distance stratification (d=0, 1, 2, 3, 4)
- Query length stratification (2, 4, 7, 11, 17+ chars)

**Current Data Sources:**
- `/usr/share/dict/words` (varies by OS, non-reproducible)
- Synthetic dictionaries: `word000001`, `word000002`, ...
- Hard-coded query sets

**Key Files:**
- `backend_comparison.rs` - 6 backend comparison
- `comprehensive_profiling.rs` - End-to-end scenarios
- `real_world_profiling.rs` - 30-second stress test
- `batch1-4_simd_benchmarks.rs` - SIMD validation

**Gap:** No standardized, reproducible corpus for cross-project comparison

---

### Data Directory (`/data/`)

**Current Contents:**
- `english_words.txt` - 123,985 lines, ~1.17 MB
- Used by: `real_world_benchmark.rs`, `dawg_query_comparison.rs`, `ordered_query_benchmark.rs`

**Characteristics:**
- Source: Unknown (appears to be system dictionary)
- Format: One word per line
- No frequency information
- No error pairs

**Planned Structure:**
```
data/
├── english_words.txt              # Existing
├── corpora/                       # NEW
│   ├── README.md                  # Documentation
│   ├── big.txt                    # Downloaded (not committed)
│   └── birkbeck.txt               # Downloaded (not committed)
└── generated/                     # NEW (test data, not committed)
    ├── typos_distance_1.txt       # Generated from big.txt
    ├── typos_distance_2.txt
    └── query_workload.txt         # Benchmark queries
```

---

### CI/CD Infrastructure (`.github/workflows/ci.yml`)

**Current Jobs:**
1. `test` - Unit and integration tests (Linux/macOS, stable/nightly)
2. `lint` - Clippy and rustfmt
3. `benchmarks` - Performance benchmarks on master
4. `test-report` - Aggregate test results

**Current Caching:**
- Cargo registry and target directories
- Cargo.lock dependencies

**Planned Addition:**
- `corpus-validation` job (new, isolated)
- Corpus file caching (~10-15 MB)
- Validation result artifacts (7-day retention)

**Gap:** No corpus download/caching infrastructure

---

### Dev Dependencies (`Cargo.toml`)

**Current:**
```toml
[dev-dependencies]
criterion = "0.5"      # Statistical benchmarking
tempfile = "3.8"       # Temporary file handling
proptest = "1.4"       # Property-based testing
rayon = "1.11"         # Parallel iteration
num_cpus = "1.16"      # CPU detection
```

**Needs:** No new dependencies required! ✅
- File I/O: stdlib
- HTTP download: `curl` via shell script
- Random generation: `rand` (already in main dependencies)

---

## Integration Strategy

### Storage Strategy

**Decision: Download On-Demand, Do Not Commit**

**Rationale:**
1. **Repository Size:** Keep repo lightweight (~10-15 MB saved)
2. **Licensing:** Clear attribution, not redistributing
3. **Updates:** Easy to update corpus versions
4. **Flexibility:** Users can opt-in to corpus tests

**Implementation:**
- Add to `.gitignore`: `data/corpora/*.txt`, `data/generated/*.txt`
- Download script: `scripts/download_corpora.sh`
- CI caching: GitHub Actions cache (~10-15 MB, well under 10GB limit)
- Checksum verification: SHA256 hashes for integrity

**Alternatives Considered:**
- ❌ Commit to repository: Bloats repo, licensing concerns
- ❌ Git LFS: Adds complexity, quota limits on free tier
- ✅ **On-demand download with caching: Best balance**

---

### Test Strategy

**Decision: Opt-In Tests with `#[ignore]` Attribute**

**Rationale:**
1. **Local Development:** Fast feedback loop (skip corpus tests by default)
2. **CI Execution:** Explicit corpus validation job
3. **Flexibility:** Developers can run `cargo test -- --ignored` when needed

**Implementation:**
```rust
#[test]
#[ignore] // Run explicitly with: cargo test -- --ignored
fn test_birkbeck_recall() {
    // Test implementation
}
```

**Alternatives Considered:**
- ❌ Always run: Slow local development
- ❌ Feature flag: Complicates build, easy to forget
- ✅ **`#[ignore]` attribute: Standard Rust pattern, explicit opt-in**

---

### Module Visibility Strategy

**Decision: Test-Only Module (`#[cfg(test)]`)**

**Rationale:**
1. **Corpus utilities not part of public API** (testing/benchmarking only)
2. **Avoid binary bloat** (exclude from release builds)
3. **Clear separation** (test infrastructure vs production code)

**Implementation:**
```rust
// src/lib.rs
#[cfg(test)]
pub mod corpus;
```

**Alternatives Considered:**
- ❌ Public module: Exposes internal testing utilities
- ❌ Separate crate: Overkill for internal tooling
- ✅ **`#[cfg(test)]` module: Standard Rust pattern**

---

### Data Generation Strategy

**Decision: Deterministic Generation with Seeded RNG**

**Rationale:**
1. **Reproducibility:** Same queries across test runs
2. **Debuggability:** Can replay specific test case
3. **Consistency:** Same workload across machines/CI

**Implementation:**
```rust
use rand::{SeedableRng, rngs::StdRng};

let mut rng = StdRng::seed_from_u64(42);  // Fixed seed
let typo = generate_typo(&word, &mut rng);
```

**Alternatives Considered:**
- ❌ True randomness: Non-reproducible failures
- ❌ Pre-generated files: Requires committing large files
- ✅ **Seeded generation: Reproducible + flexible**

---

## Implementation Plan

### Phase 1: Infrastructure Setup (Week 1, 12-16 hours)

#### File 1: `.gitignore` (Update)

**Purpose:** Exclude downloaded corpora and generated test data

**Changes:**
```gitignore
# Corpus data (download on demand)
data/corpora/*.txt
data/generated/*.txt

# Keep documentation
!data/corpora/README.md
!data/generated/README.md
```

**Rationale:** Prevents accidental commits of large corpus files

---

#### File 2: `data/corpora/README.md` (NEW, ~100 lines)

**Purpose:** Document corpus sources, attribution, and usage

**Contents:**
1. **Overview** - What corpora are available
2. **Download Instructions** - How to run `download_corpora.sh`
3. **Corpus Descriptions:**
   - big.txt: Size, source, word count, usage
   - birkbeck.txt: Error count, format, academic citation
4. **Attribution** - Proper citations for both corpora
5. **Usage Examples:**
   - Running validation tests
   - Running corpus benchmarks
   - Generating custom test data
6. **Licensing** - Public domain (big.txt), academic use (Birkbeck)

**Template:**
```markdown
# Corpus Data for Testing and Benchmarking

## Quick Start

Download corpora:
```bash
./scripts/download_corpora.sh
```

## Available Corpora

### big.txt (Peter Norvig)
- **Size:** ~300 KB
- **Words:** ~32,192 unique, ~230,000 total
- **Source:** Project Gutenberg + Wiktionary
- **Usage:** Performance benchmarking, frequency analysis
- **URL:** https://norvig.com/big.txt
- **License:** Public domain

### birkbeck.txt (Roger Mitton)
- **Size:** ~500 KB
- **Errors:** 36,133 misspellings of 6,136 words
- **Source:** School tests + literacy students
- **Usage:** Correctness validation, recall testing
- **URL:** https://titan.dcs.bbk.ac.uk/~roger/corpora.html
- **License:** Academic use, attribution required

## Usage

### Validation Tests
```bash
cargo test --test corpus_validation -- --ignored
```

### Benchmarks
```bash
cargo bench --bench corpus_benchmarks
```

## Attribution

When publishing results using these corpora:

- **big.txt:** Norvig, P. (2007). "How to Write a Spelling Corrector"
- **Birkbeck:** Mitton, R. (1985). "Birkbeck spelling error corpus", Oxford Text Archive
```

---

#### File 3: `scripts/download_corpora.sh` (NEW, ~150 lines)

**Purpose:** Automated corpus download with verification

**Features:**
1. **Idempotent:** Skip if already downloaded (unless `--force`)
2. **Checksum Verification:** SHA256 hashes for integrity
3. **Error Handling:** Retry logic, clear error messages
4. **Progress Reporting:** User-friendly output

**Pseudocode:**
```bash
#!/bin/bash
set -euo pipefail

CORPORA_DIR="data/corpora"
mkdir -p "$CORPORA_DIR"

# SHA256 checksums (computed from known-good downloads)
BIG_TXT_SHA256="..."
BIRKBECK_SHA256="..."

download_big_txt() {
    if [ ! -f "$CORPORA_DIR/big.txt" ] || [ "$1" == "--force" ]; then
        echo "Downloading big.txt from norvig.com..."
        curl -f -o "$CORPORA_DIR/big.txt" https://norvig.com/big.txt
        verify_checksum "$CORPORA_DIR/big.txt" "$BIG_TXT_SHA256"
        echo "✓ big.txt downloaded and verified"
    else
        echo "✓ big.txt already exists (use --force to re-download)"
    fi
}

download_birkbeck() {
    if [ ! -f "$CORPORA_DIR/birkbeck.txt" ] || [ "$1" == "--force" ]; then
        echo "Downloading Birkbeck corpus..."

        # Download from Oxford Text Archive
        curl -L -o "$CORPORA_DIR/birkbeck.zip" \
            "https://ota.bodleian.ox.ac.uk/repository/xmlui/bitstream/handle/20.500.12024/0643/missp.dat"

        # Extract
        unzip -o "$CORPORA_DIR/birkbeck.zip" -d "$CORPORA_DIR/"
        mv "$CORPORA_DIR/missp.dat" "$CORPORA_DIR/birkbeck.txt"
        rm "$CORPORA_DIR/birkbeck.zip"

        verify_checksum "$CORPORA_DIR/birkbeck.txt" "$BIRKBECK_SHA256"
        echo "✓ Birkbeck corpus downloaded and verified"
    else
        echo "✓ Birkbeck corpus already exists"
    fi
}

verify_checksum() {
    local file=$1
    local expected=$2
    local actual=$(sha256sum "$file" | awk '{print $1}')

    if [ "$actual" != "$expected" ]; then
        echo "ERROR: Checksum mismatch for $file"
        echo "  Expected: $expected"
        echo "  Actual:   $actual"
        exit 1
    fi
}

main() {
    local force_flag=${1:-""}
    download_big_txt "$force_flag"
    download_birkbeck "$force_flag"
    echo ""
    echo "All corpora downloaded successfully!"
    echo "Run 'cargo test --test corpus_validation -- --ignored' to validate"
}

main "$@"
```

**Error Handling:**
- `set -euo pipefail`: Exit on any error
- Checksum verification prevents corrupted downloads
- Clear error messages with manual download instructions as fallback

---

#### File 4: `.github/workflows/ci.yml` (Update)

**Purpose:** Integrate corpus tests into CI pipeline

**Changes:**

**Step 1: Add Corpus Cache** (after existing cache steps)
```yaml
- name: Cache corpora
  id: cache-corpora
  uses: actions/cache@v4
  with:
    path: data/corpora
    key: corpora-v1-${{ hashFiles('scripts/download_corpora.sh') }}
    restore-keys: |
      corpora-v1-
```

**Rationale:** Cache key includes script hash, invalidates when download logic changes

**Step 2: Download Corpora**
```yaml
- name: Download corpora
  if: steps.cache-corpora.outputs.cache-hit != 'true'
  run: |
    chmod +x scripts/download_corpora.sh
    ./scripts/download_corpora.sh
```

**Rationale:** Only download if cache miss (first run or script change)

**Step 3: New Job - Corpus Validation**
```yaml
corpus-validation:
  name: Corpus Validation Tests
  runs-on: ubuntu-latest
  env:
    RUSTFLAGS: "-C target-cpu=native"

  steps:
    - name: Checkout repository
      uses: actions/checkout@v4

    - name: Install Rust toolchain
      uses: actions-rust-lang/setup-rust-toolchain@v1
      with:
        toolchain: stable

    - name: Cache corpora
      id: cache-corpora
      uses: actions/cache@v4
      with:
        path: data/corpora
        key: corpora-v1-${{ hashFiles('scripts/download_corpora.sh') }}
        restore-keys: corpora-v1-

    - name: Download corpora
      if: steps.cache-corpora.outputs.cache-hit != 'true'
      run: |
        chmod +x scripts/download_corpora.sh
        ./scripts/download_corpora.sh

    - name: Run corpus validation tests
      run: |
        echo "::group::Birkbeck Recall Test"
        cargo test --test corpus_validation test_birkbeck_recall -- --ignored --nocapture
        echo "::endgroup::"

        echo "::group::Algorithm Consistency Test"
        cargo test --test corpus_validation test_algorithm_consistency -- --ignored --nocapture
        echo "::endgroup::"

        echo "::group::Distance Accuracy Test"
        cargo test --test corpus_validation test_distance_accuracy -- --ignored --nocapture
        echo "::endgroup::"

    - name: Upload validation results
      if: always()
      uses: actions/upload-artifact@v4
      with:
        name: corpus-validation-results
        path: |
          test-results/
          *.log
        retention-days: 7
```

**Step 4: Update Aggregate Job**
```yaml
test-report:
  name: Test Report Summary
  runs-on: ubuntu-latest
  needs: [test, lint, benchmarks, corpus-validation]  # Add corpus-validation
  if: always()
```

---

#### File 5: `src/corpus/mod.rs` (NEW, ~30 lines)

**Purpose:** Module structure for corpus utilities

**Contents:**
```rust
//! Corpus utilities for testing and benchmarking
//!
//! This module provides parsers and generators for working with
//! literature-standard corpora:
//!
//! - **Peter Norvig's big.txt**: Natural language word frequencies
//! - **Birkbeck corpus**: Real-world spelling errors
//!
//! # Examples
//!
//! ```rust,ignore
//! use liblevenshtein::corpus::parser::BigTxtCorpus;
//!
//! let corpus = BigTxtCorpus::load("data/corpora/big.txt")?;
//! let top_words = corpus.most_frequent(1000);
//! ```

pub mod parser;
pub mod generator;
pub mod stats;

pub use parser::{BirkbeckCorpus, BigTxtCorpus};
pub use generator::{TypoGenerator, QueryWorkload};
pub use stats::CorpusStats;
```

**Design:**
- Public exports for common types
- Sub-modules for organization
- Documentation with examples

---

#### File 6: `src/corpus/parser.rs` (NEW, ~250 lines)

**Purpose:** Parse Birkbeck and big.txt corpora

**Structure:**

**Type 1: `BirkbeckCorpus`**
```rust
pub struct BirkbeckCorpus {
    /// Map: correct_word -> vec![misspelling1, ...]
    pub errors: HashMap<String, Vec<String>>,
}

impl BirkbeckCorpus {
    /// Load from file path
    pub fn load<P: AsRef<Path>>(path: P) -> io::Result<Self>;

    /// Total number of misspellings
    pub fn total_errors(&self) -> usize;

    /// Number of unique correct words
    pub fn unique_words(&self) -> usize;

    /// Get errors for specific word
    pub fn errors_for(&self, word: &str) -> Option<&Vec<String>>;

    /// Sample random error pairs (for testing)
    pub fn sample(&self, count: usize, seed: u64) -> Vec<(String, String)>;
}
```

**Parsing Logic:**
```
Read line-by-line:
  If starts with '$': Set current_word = line[1..]
  Else: Add line to errors[current_word]
```

**Type 2: `BigTxtCorpus`**
```rust
pub struct BigTxtCorpus {
    /// Word -> frequency count
    pub frequencies: HashMap<String, usize>,
    /// Total word count
    pub total_words: usize,
}

impl BigTxtCorpus {
    /// Load and tokenize big.txt
    pub fn load<P: AsRef<Path>>(path: P) -> io::Result<Self>;

    /// Number of unique words
    pub fn unique_words(&self) -> usize;

    /// Top N most frequent words
    pub fn most_frequent(&self, n: usize) -> Vec<(&String, &usize)>;

    /// Probability of word: P(word) = count / total
    pub fn probability(&self, word: &str) -> f64;

    /// Sample word by frequency (Zipfian distribution)
    pub fn sample_by_frequency(&self, rng: &mut impl Rng) -> String;
}
```

**Tokenization:**
```
Read entire file to string
Split on whitespace
For each token:
  - Convert to lowercase
  - Strip non-alphabetic characters
  - Count frequency
```

**Tests:**
```rust
#[cfg(test)]
mod tests {
    #[test]
    #[ignore]
    fn test_load_birkbeck() {
        let corpus = BirkbeckCorpus::load("data/corpora/birkbeck.txt").unwrap();
        assert!(corpus.unique_words() > 6000);
        assert!(corpus.total_errors() > 35000);
    }

    #[test]
    #[ignore]
    fn test_load_big_txt() {
        let corpus = BigTxtCorpus::load("data/corpora/big.txt").unwrap();
        assert!(corpus.unique_words() > 30000);

        // "the" should be most frequent
        let top = corpus.most_frequent(1);
        assert_eq!(top[0].0, "the");
    }
}
```

---

#### File 7: `src/corpus/generator.rs` (NEW, ~200 lines)

**Purpose:** Generate realistic typos and query workloads

**Type 1: `TypoGenerator`**
```rust
pub struct TypoGenerator {
    rng: StdRng,  // Seeded for reproducibility
}

impl TypoGenerator {
    /// Create with fixed seed
    pub fn new(seed: u64) -> Self;

    /// Generate all possible edits at distance 1
    /// (Norvig's algorithm)
    pub fn edits_distance_1(&self, word: &str) -> HashSet<String>;

    /// Generate random typo at specified distance
    pub fn random_typo(&mut self, word: &str, distance: usize) -> String;

    /// Apply single random edit (internal)
    fn apply_random_edit(&mut self, word: &str) -> String;
}
```

**Edit Operations:**
```
Deletions:   "test" -> "tst", "est", "tet", "tes"
Transpositions: "test" -> "tset", "tets"
Replacements: "test" -> "aest", "best", ... "zest", "tast", ...
Insertions:  "test" -> "atest", "taest", "tesat", "testa", ...
```

**Type 2: `QueryWorkload`**
```rust
pub struct QueryWorkload {
    pub queries: Vec<Query>,
}

pub struct Query {
    pub word: String,           // Typo
    pub max_distance: usize,    // Search distance
    pub expected_result: Option<String>,  // Correct word
}

impl QueryWorkload {
    /// Generate workload from corpus
    pub fn generate_from_corpus(
        corpus_words: &[String],
        corpus_frequencies: &HashMap<String, usize>,
        count: usize,
        seed: u64,
    ) -> Self;

    /// Sample word by frequency (Zipfian)
    fn sample_by_frequency(...) -> String;

    /// Save to file
    pub fn save<P: AsRef<Path>>(&self, path: P) -> io::Result<()>;

    /// Load from file
    pub fn load<P: AsRef<Path>>(path: P) -> io::Result<Self>;
}
```

**Stratified Sampling:**
```
40% high-frequency (rank 1-1000):   Most common queries
40% medium-frequency (rank 1000-10000): Moderate vocabulary
20% low-frequency (rank 10000+):     Rare words
```

**Tests:**
```rust
#[test]
fn test_edits_distance_1() {
    let gen = TypoGenerator::new(42);
    let edits = gen.edits_distance_1("test");

    assert!(edits.contains("est"));   // deletion
    assert!(edits.contains("tset"));  // transposition
    assert!(edits.contains("best"));  // replacement
    assert!(edits.contains("atest")); // insertion

    // Should generate ~54n + 25 edits for n-letter word
    assert!(edits.len() > 200);
}

#[test]
fn test_deterministic_generation() {
    let mut gen1 = TypoGenerator::new(42);
    let mut gen2 = TypoGenerator::new(42);

    assert_eq!(gen1.random_typo("test", 2), gen2.random_typo("test", 2));
}
```

---

#### File 8: `src/lib.rs` (Update, 1 line)

**Purpose:** Expose corpus module for tests/benchmarks only

**Change:**
```rust
// At top level, after other modules

#[cfg(test)]
pub mod corpus;
```

**Rationale:**
- `#[cfg(test)]`: Only compiled for test/bench builds
- `pub`: Accessible from `tests/` and `benches/` directories
- Avoids bloating release binaries

---

### Phase 2: Validation Tests (Week 2, 16-20 hours)

#### File 9: `tests/corpus_validation.rs` (NEW, ~400 lines)

**Purpose:** Validate correctness against literature-standard errors

**Test 1: Birkbeck Recall Test**

**Goal:** Verify >95% recall on real spelling errors

**Methodology:**
```
For each (correct_word, misspelling) pair in Birkbeck:
  1. Calculate actual Levenshtein distance
  2. Query automaton with max_distance = min(actual_distance, 3)
  3. Check if correct_word in results
  4. Accumulate recall statistics
```

**Pseudocode:**
```rust
#[test]
#[ignore]
fn test_birkbeck_recall() {
    let corpus = BirkbeckCorpus::load("data/corpora/birkbeck.txt")?;
    let dict = build_dictionary(corpus.errors.keys());
    let transducer = Transducer::new(dict, Algorithm::Standard);

    let mut stats = RecallStats::new();

    for (correct, misspellings) in &corpus.errors {
        for misspelling in misspellings {
            let actual_distance = naive_levenshtein(misspelling, correct);
            let max_distance = actual_distance.min(3);

            let results: Vec<_> = transducer.query(misspelling, max_distance).collect();
            let found = results.contains(correct);

            stats.record(actual_distance, found);
        }
    }

    // Print detailed breakdown
    println!("\n=== Birkbeck Recall Results ===");
    println!("Total: {} / {} ({:.2}%)",
        stats.found, stats.total, stats.recall() * 100.0);

    for d in 1..=10 {
        if let Some(rate) = stats.recall_at_distance(d) {
            println!("  d={}: {:.2}%", d, rate * 100.0);
        }
    }

    // Assert minimum threshold
    assert!(stats.recall() >= 0.95,
        "Recall {:.2}% below 95% threshold", stats.recall() * 100.0);
}

struct RecallStats {
    total: usize,
    found: usize,
    by_distance: HashMap<usize, (usize, usize)>,  // (total, found)
}
```

**Expected Output:**
```
=== Birkbeck Recall Results ===
Total: 34,527 / 36,133 (95.56%)
  d=1: 98.21%
  d=2: 96.45%
  d=3: 92.33%
  d=4: 84.12%
  d=5: 71.56%
```

**Failure Modes:**
- Recall < 95%: Algorithm bug or subsumption issue
- High variance across distances: Specific operation handling bug
- Specific word patterns failing: Edge case in transition logic

---

**Test 2: Algorithm Consistency Test**

**Goal:** Verify all algorithm variants produce consistent results

**Methodology:**
```
For sample of 1000 error pairs:
  Run Standard, Transposition, MergeAndSplit algorithms
  Verify all find (or all miss) the correct word
  (Different algorithms may find different additional matches,
   but should agree on the target word)
```

**Pseudocode:**
```rust
#[test]
#[ignore]
fn test_algorithm_consistency() {
    let corpus = BirkbeckCorpus::load("data/corpora/birkbeck.txt")?;
    let sample = corpus.sample(1000, 42);  // Seeded sampling

    let dict = build_dictionary(corpus.errors.keys());
    let algorithms = [Algorithm::Standard, Algorithm::Transposition, Algorithm::MergeAndSplit];

    let mut inconsistencies = 0;

    for (correct, misspelling) in sample {
        let mut found_by_algo = Vec::new();

        for algo in &algorithms {
            let transducer = Transducer::new(dict.clone(), *algo);
            let results = transducer.query(&misspelling, 2).collect::<Vec<_>>();
            let found = results.contains(&correct);
            found_by_algo.push(found);
        }

        // Check consistency (all found or all not found)
        let all_found = found_by_algo.iter().all(|&f| f);
        let none_found = found_by_algo.iter().all(|&f| !f);

        if !all_found && !none_found {
            eprintln!("INCONSISTENCY: '{}' -> '{}'", misspelling, correct);
            eprintln!("  Standard: {}, Transposition: {}, MergeAndSplit: {}",
                found_by_algo[0], found_by_algo[1], found_by_algo[2]);
            inconsistencies += 1;
        }
    }

    assert_eq!(inconsistencies, 0,
        "{} inconsistencies found across algorithms", inconsistencies);
}
```

**Expected:** 0 inconsistencies (all algorithms agree)

**Rationale:** While different algorithms may find different nearby words (e.g., Transposition finds more transposition errors), they should agree on whether the *correct* word is within distance n.

---

**Test 3: Distance Accuracy Test**

**Goal:** Verify reported distances match ground truth

**Methodology:**
```
For sample of error pairs:
  Calculate naive Levenshtein distance (ground truth)
  Query automaton with query_with_distance()
  Verify reported distance == ground truth
```

**Pseudocode:**
```rust
#[test]
#[ignore]
fn test_distance_accuracy() {
    let corpus = BirkbeckCorpus::load("data/corpora/birkbeck.txt")?;
    let sample = corpus.sample(500, 42);

    let dict = build_dictionary(corpus.errors.keys());
    let transducer = Transducer::new(dict, Algorithm::Standard);

    let mut mismatches = 0;

    for (correct, misspelling) in sample {
        let actual = naive_levenshtein(&misspelling, &correct);

        if actual <= 3 {
            let results = transducer
                .query_with_distance(&misspelling, actual)
                .collect::<Vec<_>>();

            if let Some(candidate) = results.iter().find(|c| c.term == correct) {
                if candidate.distance != actual {
                    eprintln!("MISMATCH: '{}' -> '{}' (expected {}, got {})",
                        misspelling, correct, actual, candidate.distance);
                    mismatches += 1;
                }
            }
        }
    }

    assert_eq!(mismatches, 0, "{} distance mismatches", mismatches);
}

/// Naive Levenshtein for ground truth
fn naive_levenshtein(s1: &str, s2: &str) -> usize {
    let len1 = s1.chars().count();
    let len2 = s2.chars().count();

    // Wagner-Fischer DP algorithm
    let mut dp = vec![vec![0; len2 + 1]; len1 + 1];

    for i in 0..=len1 { dp[i][0] = i; }
    for j in 0..=len2 { dp[0][j] = j; }

    let s1_chars: Vec<char> = s1.chars().collect();
    let s2_chars: Vec<char> = s2.chars().collect();

    for i in 1..=len1 {
        for j in 1..=len2 {
            let cost = if s1_chars[i-1] == s2_chars[j-1] { 0 } else { 1 };
            dp[i][j] = min3(
                dp[i-1][j] + 1,      // deletion
                dp[i][j-1] + 1,      // insertion
                dp[i-1][j-1] + cost, // substitution
            );
        }
    }

    dp[len1][len2]
}
```

---

### Phase 3: Benchmark Suite (Week 2-3, 16-20 hours)

#### File 10: `benches/corpus_benchmarks.rs` (NEW, ~500 lines)

**Purpose:** Reproducible performance benchmarks on standard corpus

**Benchmark 1: Dictionary Construction**

**Goal:** Measure construction time and memory for 32K words

**Implementation:**
```rust
fn bench_construction(c: &mut Criterion) {
    let corpus = BigTxtCorpus::load("data/corpora/big.txt").unwrap();
    let words: Vec<String> = corpus.frequencies.keys().cloned().collect();

    let mut group = c.benchmark_group("construction_big_txt");
    group.throughput(Throughput::Elements(words.len() as u64));

    // DoubleArrayTrie
    group.bench_function("DoubleArrayTrie", |b| {
        b.iter(|| {
            let dict = DoubleArrayTrie::from_terms(words.clone());
            black_box(dict)
        })
    });

    // DAWG
    group.bench_function("DawgDictionary", |b| {
        b.iter(|| {
            let dict = DawgDictionary::from_iter(words.iter().map(|s| s.as_str()));
            black_box(dict)
        })
    });

    // DynamicDawg
    group.bench_function("DynamicDawg", |b| {
        b.iter(|| {
            let dict = DynamicDawg::from_terms(words.clone());
            black_box(dict)
        })
    });

    // PathMap
    #[cfg(feature = "pathmap-backend")]
    group.bench_function("PathMapDictionary", |b| {
        b.iter(|| {
            let dict = PathMapDictionary::from_terms(words.clone());
            black_box(dict)
        })
    });

    group.finish();
}
```

**Expected Results:**
```
construction_big_txt/DoubleArrayTrie    time: [3.2 ms 3.3 ms 3.4 ms]
                                        thrpt: [9.6 K elem/s 9.8 K elem/s 10.0 K elem/s]
construction_big_txt/DawgDictionary     time: [6.0 ms 6.2 ms 6.4 ms]
construction_big_txt/DynamicDawg        time: [3.9 ms 4.0 ms 4.1 ms]
construction_big_txt/PathMapDictionary  time: [3.0 ms 3.1 ms 3.2 ms]
```

---

**Benchmark 2: Realistic Query Workload**

**Goal:** Measure query performance with frequency-weighted queries

**Implementation:**
```rust
fn bench_query_realistic_workload(c: &mut Criterion) {
    let corpus = BigTxtCorpus::load("data/corpora/big.txt").unwrap();
    let words: Vec<String> = corpus.frequencies.keys().cloned().collect();

    // Generate 1000 queries with realistic typos
    let workload = QueryWorkload::generate_from_corpus(
        &words,
        &corpus.frequencies,
        1000,
        42,  // Fixed seed for reproducibility
    );

    let dict = DoubleArrayTrie::from_terms(words);
    let transducer = Transducer::new(dict, Algorithm::Standard);

    let mut group = c.benchmark_group("query_realistic_workload");
    group.throughput(Throughput::Elements(workload.queries.len() as u64));

    for distance in [1, 2, 3] {
        let queries_at_d: Vec<_> = workload.queries.iter()
            .filter(|q| q.max_distance == distance)
            .collect();

        group.bench_with_input(
            BenchmarkId::new("distance", distance),
            &queries_at_d,
            |b, queries| {
                b.iter(|| {
                    for query in queries.iter() {
                        let results: Vec<_> = transducer
                            .query(&query.word, black_box(distance))
                            .collect();
                        black_box(results);
                    }
                })
            },
        );
    }

    group.finish();
}
```

**Expected Results:**
```
query_realistic_workload/distance/1     time: [8.2 µs 8.4 µs 8.6 µs]
                                        thrpt: [119 K queries/s 122 K queries/s 125 K queries/s]
query_realistic_workload/distance/2     time: [12.9 µs 13.1 µs 13.3 µs]
                                        thrpt: [75 K queries/s 76 K queries/s 78 K queries/s]
query_realistic_workload/distance/3     time: [18.5 µs 18.8 µs 19.1 µs]
                                        thrpt: [52 K queries/s 53 K queries/s 54 K queries/s]
```

---

**Benchmark 3: Frequency-Stratified Queries**

**Goal:** Compare performance for high vs medium frequency words

**Implementation:**
```rust
fn bench_frequency_stratified_queries(c: &mut Criterion) {
    let corpus = BigTxtCorpus::load("data/corpora/big.txt").unwrap();
    let words: Vec<String> = corpus.frequencies.keys().cloned().collect();

    // High-frequency: top 1000
    let high_freq: Vec<String> = corpus.most_frequent(1000)
        .into_iter()
        .map(|(w, _)| w.clone())
        .collect();

    // Medium-frequency: rank 1000-5000
    let mut sorted: Vec<_> = corpus.frequencies.iter().collect();
    sorted.sort_by(|a, b| b.1.cmp(a.1));
    let medium_freq: Vec<String> = sorted[1000..5000]
        .iter()
        .map(|(w, _)| (*w).clone())
        .collect();

    let dict = DoubleArrayTrie::from_terms(words);
    let transducer = Transducer::new(dict, Algorithm::Standard);

    let mut group = c.benchmark_group("query_by_frequency");

    group.bench_function("high_frequency", |b| {
        b.iter(|| {
            for word in high_freq.iter().take(100) {
                let results: Vec<_> = transducer.query(word, 2).collect();
                black_box(results);
            }
        })
    });

    group.bench_function("medium_frequency", |b| {
        b.iter(|| {
            for word in medium_freq.iter().take(100) {
                let results: Vec<_> = transducer.query(word, 2).collect();
                black_box(results);
            }
        })
    });

    group.finish();
}
```

**Expected Insight:**
- High-frequency words may have more results (shorter words, more ambiguity)
- Medium-frequency words may be faster (less ambiguity, pruning)

---

**Benchmark 4: Algorithm Comparison**

**Goal:** Measure overhead of extended algorithms on standard corpus

**Implementation:**
```rust
fn bench_algorithm_comparison(c: &mut Criterion) {
    let corpus = BigTxtCorpus::load("data/corpora/big.txt").unwrap();
    let words: Vec<String> = corpus.frequencies.keys().cloned().collect();
    let dict = DoubleArrayTrie::from_terms(words);

    let workload = QueryWorkload::generate_from_corpus(
        &corpus.frequencies.keys().cloned().collect::<Vec<_>>(),
        &corpus.frequencies,
        100,
        42,
    );

    let algorithms = vec![
        ("Standard", Algorithm::Standard),
        ("Transposition", Algorithm::Transposition),
        ("MergeAndSplit", Algorithm::MergeAndSplit),
    ];

    let mut group = c.benchmark_group("algorithm_comparison_corpus");

    for (name, algo) in algorithms {
        let transducer = Transducer::new(dict.clone(), algo);

        group.bench_function(name, |b| {
            b.iter(|| {
                for query in &workload.queries {
                    let results: Vec<_> = transducer
                        .query(&query.word, query.max_distance)
                        .collect();
                    black_box(results);
                }
            })
        });
    }

    group.finish();
}
```

**Expected Results:**
```
algorithm_comparison_corpus/Standard       time: [12.8 µs 13.0 µs 13.2 µs]
algorithm_comparison_corpus/Transposition  time: [14.8 µs 15.1 µs 15.4 µs]  (+16%)
algorithm_comparison_corpus/MergeAndSplit  time: [15.9 µs 16.2 µs 16.5 µs]  (+25%)
```

---

#### File 11: `Cargo.toml` (Update)

**Purpose:** Register new benchmark

**Changes:**
```toml
[[bench]]
name = "corpus_benchmarks"
harness = false
```

**Location:** Add after existing `[[bench]]` entries

---

### Phase 4: Documentation (Week 3, 4-5 hours)

#### File 12: `docs/benchmarks/CORPUS_BENCHMARKING.md` (NEW, ~600 lines)

**Purpose:** Comprehensive guide to corpus-based benchmarking

**Contents:**

**Section 1: Overview** (~50 lines)
- Purpose and benefits of corpus-based benchmarking
- What corpora are used and why
- How this enables reproducible research

**Section 2: Corpus Descriptions** (~150 lines)
- big.txt: Statistics, source, academic usage, citations
- Birkbeck: Error patterns, format, academic usage, citations
- When to use each corpus

**Section 3: Setup Instructions** (~100 lines)
- Download script usage
- Verification (checksums)
- Troubleshooting download failures
- Manual download instructions (if automated fails)

**Section 4: Running Validation Tests** (~100 lines)
- Command examples for each test
- Expected output and interpretation
- What to do if tests fail
- How to debug recall issues

**Section 5: Running Benchmarks** (~100 lines)
- Command examples for each benchmark
- How to interpret Criterion output
- Statistical significance
- Regression detection

**Section 6: CI Integration** (~50 lines)
- How corpus tests run in CI
- Caching strategy
- Artifact retention
- How to view historical results

**Section 7: Expected Results** (~50 lines)
- Baseline metrics from current implementation
- How to compare results
- When to update baselines

**Section 8: References** (~50 lines)
- Academic citations for both corpora
- Related research papers
- Links to evaluation methodology docs

**Template:**
```markdown
# Corpus-Based Benchmarking Guide

## Quick Start

1. Download corpora:
```bash
./scripts/download_corpora.sh
```

2. Run validation tests:
```bash
cargo test --test corpus_validation -- --ignored
```

3. Run benchmarks:
```bash
cargo bench --bench corpus_benchmarks
```

## Corpora

### big.txt (Peter Norvig, 2007)

**Purpose:** Realistic English word distribution for performance testing

**Statistics:**
- 32,192 unique words
- 230,000 total tokens
- Zipf distribution (frequency ∝ 1/rank)

**Usage:**
- Dictionary construction benchmarks
- Query performance with natural language
- Frequency-weighted workload generation

**Citation:**
```
Norvig, P. (2007). How to Write a Spelling Corrector.
Retrieved from https://norvig.com/spell-correct.html
```

### Birkbeck (Roger Mitton, 1985)

**Purpose:** Real-world spelling errors for correctness validation

**Statistics:**
- 36,133 misspellings
- 6,136 correct words
- Distance distribution: 42% d=1, 31% d=2, 15% d=3

**Usage:**
- Recall testing (target: >95%)
- Algorithm consistency validation
- Distance accuracy verification

**Citation:**
```
Mitton, R. (1985). Birkbeck spelling error corpus.
Oxford Text Archive. http://ota.ahds.ac.uk/
```

[... continues with all sections ...]
```

---

## CI/CD Integration

### Cache Strategy

**Key:** `corpora-v1-${{ hashFiles('scripts/download_corpora.sh') }}`

**Rationale:**
- Version prefix (`corpora-v1`): Manual cache invalidation if needed
- Script hash: Automatic invalidation when download logic changes
- Path: `data/corpora/`
- Size: ~10-15 MB (well within GitHub's 10 GB limit)

**Behavior:**
- First run: Download corpora, cache them (~30s download)
- Subsequent runs: Restore from cache (~5s restore)
- Script change: Re-download and re-cache
- Manual invalidation: Increment version (`corpora-v2`)

---

### Job Isolation

**Separate `corpus-validation` Job:**

**Benefits:**
1. **Parallel Execution:** Runs alongside other tests
2. **Independent Failure:** Corpus tests can fail without blocking other checks
3. **Clear Reporting:** Dedicated status check in GitHub UI
4. **Resource Allocation:** Can configure different runners if needed

**Dependencies:**
```yaml
test-report:
  needs: [test, lint, benchmarks, corpus-validation]
```

**Result:** Overall success requires all jobs passing

---

### Artifact Retention

**Validation Results:** 7 days
```yaml
- name: Upload validation results
  uses: actions/upload-artifact@v4
  with:
    name: corpus-validation-results
    retention-days: 7
```

**Rationale:**
- Short-term debugging (recent failures)
- Balance storage costs vs utility
- Benchmarks have separate 30-day retention

**Benchmark Results:** 30 days (existing configuration)

**Rationale:**
- Historical comparison
- Regression detection
- Performance trend analysis

---

## Expected Benefits

### 1. Correctness Validation (Quantitative)

**Metrics:**
- **Recall:** % of misspellings where correct word found
  - Target: >95% overall
  - Breakdown: >98% d=1, >96% d=2, >92% d=3
- **Precision:** % of results that are relevant
  - Calculated for top-k results (k=5, 10, 20)
- **F1 Score:** Harmonic mean of precision and recall
- **Algorithm Consistency:** 100% agreement across variants

**Benefits:**
- Confidence in production deployments
- Detection of edge cases missed by synthetic tests
- Validation against established benchmark (36K real errors)

---

### 2. Performance Benchmarking (Reproducible)

**Metrics:**
- **Construction:** ms to build 32K-word dictionary
- **Query Latency:** Median, p50, p95, p99 for realistic workload
- **Throughput:** Queries/sec with natural language distribution
- **Memory:** Bytes/word for each backend
- **Scaling:** Verify O(|W|) construction, O(|D|) query

**Benefits:**
- Reproducible results across machines/CI
- Standardized comparison across commits
- Academic publication-ready metrics
- Historical trend tracking

---

### 3. Academic Credibility

**Benefits:**
- **Literature-Standard Benchmarks:** Enables peer comparison
- **Citation-Worthy Results:** Norvig + Birkbeck = established baselines
- **Transparent Methodology:** Fully documented, reproducible
- **Cross-Project Comparison:** "Our recall on Birkbeck: 96.2%"

**Research Impact:**
- Publishable in academic venues
- Comparable to prior work (Norvig 2007, Mitton 1985)
- Establishes liblevenshtein-rust as reference implementation

---

### 4. Development Workflow

**CI Benefits:**
- **Automated Validation:** Every commit tested against corpora
- **Fast Feedback:** Cached downloads (~5s overhead)
- **Regression Detection:** Historical comparison via Criterion
- **Clear Acceptance Criteria:** >95% recall required to merge

**Developer Benefits:**
- **Opt-In Local Testing:** `#[ignore]` attribute prevents slow tests by default
- **Explicit Validation:** `cargo test -- --ignored` when needed
- **Confidence in Refactoring:** Real-world data guards correctness
- **Performance Awareness:** Benchmark results show impact

---

## Timeline and Effort

### Week 1: Infrastructure (12-16 hours)

**Days 1-2 (8 hours):**
- Download script with checksums (3h)
- .gitignore updates (1h)
- CI configuration with caching (4h)

**Days 3-4 (8 hours):**
- Corpus parsing library (6h)
  - BirkbeckCorpus: 3h
  - BigTxtCorpus: 3h
- Initial tests for parsers (2h)

**Deliverables:**
- ✅ Corpora download automatically in CI
- ✅ Parsing library complete with tests
- ✅ Documentation scaffolding

---

### Week 2: Testing & Validation (16-20 hours)

**Days 1-2 (10 hours):**
- Birkbeck recall test (5h)
- Algorithm consistency test (3h)
- Distance accuracy test (2h)

**Days 3-4 (8 hours):**
- Typo generator (4h)
- Query workload generator (4h)

**Deliverables:**
- ✅ Validation tests complete (>95% recall target)
- ✅ Typo/workload generators with deterministic output
- ✅ Test documentation

---

### Week 3: Benchmarking & Documentation (16-20 hours)

**Days 1-3 (12 hours):**
- Construction benchmark (3h)
- Query workload benchmark (4h)
- Frequency-stratified benchmark (2h)
- Algorithm comparison benchmark (3h)

**Days 4-5 (8 hours):**
- CORPUS_BENCHMARKING.md documentation (5h)
- README updates (2h)
- Final integration testing (1h)

**Deliverables:**
- ✅ Complete benchmark suite
- ✅ Comprehensive documentation
- ✅ CI integration verified
- ✅ Historical baseline established

---

**Total Effort:** 44-56 hours (~1-1.5 weeks full-time)

**Breakdown:**
- Infrastructure: 28% (12-16h)
- Testing: 36% (16-20h)
- Benchmarking & Docs: 36% (16-20h)

---

## Risks and Mitigations

### Risk 1: Download Failures

**Probability:** Medium (external dependencies)
**Impact:** High (blocks CI)

**Mitigation:**
1. **Retry Logic:** Download script retries 3× with exponential backoff
2. **CI Cache:** Provides redundancy (corpus cached after first successful download)
3. **Checksum Verification:** Detects corrupted downloads
4. **Manual Fallback:** Clear instructions for manual download
5. **Archive Backup:** Store corpora in GitHub Releases as last resort

**Monitoring:** CI job failure alerts

---

### Risk 2: Corpus Availability

**Probability:** Low (established, archived corpora)
**Impact:** Medium (breaks reproducibility)

**Mitigation:**
1. **Multiple Sources:**
   - big.txt: norvig.com + Internet Archive Wayback Machine
   - Birkbeck: Oxford Text Archive + Birkbeck mirror
2. **Version Pinning:** Checksums prevent unexpected changes
3. **GitHub Release Archive:** Backup copy in project releases
4. **Documentation:** Alternative download methods documented

**Long-term:** Consider hosting mirrors

---

### Risk 3: Test Flakiness

**Probability:** Low (deterministic generation)
**Impact:** Medium (false failures)

**Mitigation:**
1. **Seeded RNG:** All random generation uses fixed seeds
2. **Clear Thresholds:** Recall >95% based on empirical measurement
3. **Per-Distance Breakdown:** Identify specific issues (e.g., d=3 problems)
4. **Proptest Integration:** Property-based validation complements corpus tests
5. **Statistical Analysis:** Criterion provides confidence intervals

**Monitoring:** Track recall variance across CI runs

---

### Risk 4: CI Performance Impact

**Probability:** Medium (corpus tests are slower)
**Impact:** Low (separate job)

**Mitigation:**
1. **Job Isolation:** `corpus-validation` doesn't block other checks
2. **Efficient Caching:** ~5s cache restore vs ~30s download
3. **Sample Size Tuning:** Can reduce test sample for faster CI
4. **Parallel Execution:** Corpus tests run concurrently with other jobs
5. **Benchmark Throttling:** Full benchmarks only on master branch

**Typical Impact:** +1-2 minutes total CI time (amortized across parallel jobs)

---

### Risk 5: Recall Threshold Too High/Low

**Probability:** Medium (unknown ground truth)
**Impact:** Medium (false positives or negatives)

**Mitigation:**
1. **Empirical Baseline:** Run tests, measure actual recall, set threshold
2. **Per-Distance Targets:** Different thresholds for d=1 (>98%), d=2 (>96%), d=3 (>92%)
3. **Gradual Rollout:** Start with warning, convert to error after stability
4. **Manual Review:** Investigate failures to determine if bug or unrealistic expectation
5. **Literature Comparison:** Compare to other implementations' reported recall

**Process:** Measure → Analyze → Set threshold → Monitor → Adjust

---

## Future Enhancements

### Phase 2: Additional Corpora (Optional)

**Holbrook Corpus:**
- **Effort:** 4-6 hours (parser + tests)
- **Benefit:** Secondary school errors, different demographics
- **When:** If Birkbeck alone insufficient

**Aspell Corpus:**
- **Effort:** 3-4 hours (smaller corpus)
- **Benefit:** Technical terms, different error patterns
- **When:** For specialized vocabulary testing

**Wikipedia Corpus:**
- **Effort:** 4-6 hours
- **Benefit:** Common web-scale errors
- **When:** For production web applications

---

### Phase 3: Advanced Metrics (Optional)

**Mean Reciprocal Rank (MRR):**
- **Effort:** 6-8 hours (test + benchmark)
- **Benefit:** Evaluate ranking quality for weighted costs
- **When:** After implementing weight learning (WEIGHT_LEARNING.md)

**Normalized Discounted Cumulative Gain (NDCG):**
- **Effort:** 8-10 hours
- **Benefit:** Information retrieval metric
- **When:** For production search applications

**Per-Language Performance:**
- **Effort:** 2-4 weeks (multi-language corpora + tests)
- **Benefit:** International validation
- **When:** For multi-language support

---

### Phase 4: Cross-Library Comparison (Optional)

**Benchmark Suite:**
- **Effort:** 1-2 weeks
- **Libraries:** SymSpell, BK-Tree, other Rust/C++/Python implementations
- **Benefit:** Marketing ("10× faster than X")
- **When:** For competitive analysis, academic publication

**Challenges:**
- Apples-to-apples comparison (same algorithm vs different approaches)
- Fair benchmarking (avoid biasing toward our implementation)
- Maintenance (keep other libraries updated)

---

### Phase 5: Visualization & Dashboard (Optional)

**Performance Dashboard:**
- **Effort:** 1-2 weeks (web app + GitHub Pages)
- **Features:** Historical trends, regression alerts, comparison charts
- **Benefit:** Visual performance tracking
- **When:** For public-facing performance claims

**Coverage Heatmaps:**
- **Effort:** 1 week
- **Features:** Visualize which error types found/missed
- **Benefit:** Identify algorithm weaknesses
- **When:** For algorithm development

---

## Summary

### What This Plan Delivers

1. **Infrastructure** - Download automation, CI integration, corpus utilities
2. **Validation** - >95% recall on 36K real errors, algorithm consistency
3. **Benchmarks** - Reproducible performance on 32K-word natural language corpus
4. **Documentation** - Comprehensive guides for setup, usage, interpretation
5. **Foundation** - Basis for future enhancements (additional corpora, metrics, comparisons)

### Why This Matters

**Academic:**
- Literature-standard benchmarks enable peer comparison
- Citation-worthy results for research papers
- Establishes liblevenshtein-rust as reference implementation

**Production:**
- Confidence in correctness (validated against real errors)
- Performance guarantees (reproducible benchmarks)
- Regression detection (automated CI testing)

**Development:**
- Clear acceptance criteria (>95% recall)
- Fast feedback loop (cached downloads)
- Historical tracking (Criterion + artifacts)

### Next Steps

1. **Review Plan:** Stakeholder approval
2. **Pilot Phase:** Week 1 infrastructure (validate approach)
3. **Full Implementation:** Weeks 2-3 (tests + benchmarks)
4. **Validation:** Verify >95% recall, reproducible benchmarks
5. **Documentation:** Publish results, update README
6. **Maintenance:** Monitor CI, update baselines as algorithms improve

---

**Last Updated:** 2025-11-06
**Status:** Planning Complete - Awaiting Implementation Approval
**Estimated Delivery:** 1-1.5 weeks full-time effort
**Risk Level:** Low (modular design, established corpora, clear deliverables)

---

**Related Documentation:**
- [Evaluation Methodology](../evaluation-methodology/README.md) - Overall evaluation framework
- [Weight Learning](../weighted-levenshtein-automata/WEIGHT_LEARNING.md) - Optimal cost derivation
- [Algorithm Documentation](../../algorithms/README.md) - Implementation details
