# Phase A: FuzzySource Implementation Guide

**Last Updated**: 2025-12-23
**Version**: 0.9.1 (PhoneticNormalizedDictionary API)
**Status**: Implementation Guide

This document provides detailed implementation guidance for MORK integration: creating FuzzySource and FuzzyPhoneticSource that enable approximate string matching in MeTTa queries.

## Overview

**Goal**: Enable MORK queries to perform fuzzy symbol matching using liblevenshtein's `PhoneticNormalizedDictionary`.

**Result**: MeTTa queries support both standard and phonetic-aware fuzzy matching:
- `!(match &space (fuzzy "colr" 2 $result) $result)` - standard Levenshtein
- `!(match &space (fuzzy-phonetic "fone" 2 $result) $result)` - phonetic-aware matching

---

## PhoneticNormalizedDictionary API

The FuzzySource adapters use `PhoneticNormalizedDictionary` for phonetic-aware fuzzy matching.

### Architecture Overview

```
PhoneticNormalizedDictionary<V, D>
├── originals: D                           # Backend dictionary (DynamicDawgChar)
├── normalized_multimap: FuzzyMultiMap     # normalized → {originals}
│   └── Uses Levenshtein automaton for O(k log n) fuzzy queries
├── rules: Vec<RewriteRuleChar>            # Phonetic transformation rules
└── fuel: usize                            # Prevents infinite rule loops
```

**Key Optimizations:**
- **Exact match fast path (d=0)**: Direct trie lookup is **100-300× faster** than automaton traversal
- **FuzzyMultiMap**: $`\mathcal{O}(k \log n)`$ fuzzy queries via Levenshtein automaton pruning
- **Thread-local NormalizeBuffers (H3)**: Reuses buffers to reduce allocations
- **$`\mathcal{O}(1)`$ vowel classification**: Bitmask lookup instead of linear array search

### Building a Dictionary

```rust
use liblevenshtein::dictionary::phonetic_normalized::{
    PhoneticNormalizedDictionary, PhoneticNormalizedCandidate
};
use liblevenshtein::phonetic::rules::english;

// Build with combined English rules (base + homophones + text_speak)
let combined_rules = english::combined();
let dict = PhoneticNormalizedDictionary::<()>::from_terms_with_rules(&words, combined_rules);

// Or use specific rule sets
let dict_base = PhoneticNormalizedDictionary::<()>::from_terms_with_rules(
    &words,
    english::base().rules
);
```

### Fuzzy Queries

```rust
// Query returns Vec<PhoneticNormalizedCandidate>
let results = dict.query("fone", 2);  // max distance = 2

for candidate in results {
    println!("{}: distance={}, normalized='{}'",
        candidate.term, candidate.distance, candidate.normalized_form);
}
// Output:
// phone: distance=0, normalized='fon'
// phon: distance=1, normalized='fon'

// PhoneticNormalizedCandidate structure:
// - term: String           # Original term from dictionary
// - distance: usize        # Edit distance in normalized space
// - normalized_form: String # The normalized form that matched
```

### Pre-Compiled English Rules

```rust
use liblevenshtein::phonetic::rules::english;

let base = english::base();              // 62 orthographic rules
let homophones = english::homophones();  // Homophone pairs
let text_speak = english::text_speak();  // Text-speak expansions

// Combined rule set (recommended for most use cases)
let combined = english::combined();
```

---

## Architecture

### Data Flow

```
MeTTa Query: (fuzzy-phonetic "fone" 2 $result)
    |
    v
MORK Query Parser
    |
    | Recognizes (fuzzy ...) or (fuzzy-phonetic ...) pattern
    v
FuzzySource or FuzzyPhoneticSource::new(expr)
    |
    | Parses: max_distance, pattern
    v
FuzzyPhoneticSource::request()
    |
    | Requests BTM (PathMap) access
    v
FuzzyPhoneticSource::source()
    |
    | Builds FuzzyDictionaryView from PathMap
    | Uses PhoneticNormalizedDictionary.query()
    | Returns FuzzyZipper over PhoneticNormalizedCandidate
    v
ProductZipper (combines with other sources)
    |
    v
Unification with query pattern
    |
    v
Results (with normalized forms for debugging/ranking)
```

### Component Relationships

```
┌─────────────────────────────────────────────────────────────┐
│ MORK/kernel/src/                                            │
│                                                             │
│  ┌─────────────────────┐     ┌─────────────────────────┐   │
│  │ sources.rs          │     │ fuzzy_source.rs         │   │
│  │                     │     │                         │   │
│  │ ASource enum        │◄────│ FuzzySource             │   │
│  │   BTMSource         │     │ FuzzyPhoneticSource     │   │
│  │   ACTSource         │     │ FuzzyConfig             │   │
│  │   FuzzySource       │     │ FuzzyDictionaryView     │   │
│  │   FuzzyPhoneticSrc  │     └───────────┬─────────────┘   │
│  └──────────┬──────────┘                 │                 │
│             │                            │                 │
│             v                            v                 │
│  ┌─────────────────────┐     ┌─────────────────────────┐   │
│  │ AFactor enum        │     │ fuzzy_zipper.rs         │   │
│  │   PosSource         │     │                         │   │
│  │   ACTSource         │◄────│ FuzzyZipper (Candidate) │   │
│  │   FuzzySource       │     │                         │   │
│  │   FuzzyPhoneticSrc  │     └───────────┬─────────────┘   │
│  └─────────────────────┘                 │                 │
│                                          │                 │
└──────────────────────────────────────────│─────────────────┘
                                           │
                                           v
┌─────────────────────────────────────────────────────────────┐
│ liblevenshtein-rust/src/                                    │
│                                                             │
│  ┌─────────────────────────────────────────────────────┐   │
│  │ dictionary/phonetic_normalized/                      │   │
│  │   PhoneticNormalizedDictionary                       │   │
│  │   PhoneticNormalizedCandidate                        │   │
│  │                                                      │   │
│  │   .query() → d=0: trie lookup (100-300× faster)     │   │
│  │           → d≥1: FuzzyMultiMap automaton pruning     │   │
│  └─────────────────────────────────────────────────────┘   │
│                                                             │
└─────────────────────────────────────────────────────────────┘
```

---

## Implementation

### Step 1: Add Dependency to MORK

**File**: `MORK/kernel/Cargo.toml`

```toml
[dependencies]
# Existing dependencies...
pathmap = { path = "../../PathMap" }

# Add liblevenshtein with pathmap backend
liblevenshtein = { path = "../../../liblevenshtein-rust", features = ["pathmap-backend"] }

[features]
default = ["grounding"]
grounding = []
fuzzy = []  # NEW: Enable fuzzy matching support
```

### Step 2: Create FuzzyConfig

**File**: `MORK/kernel/src/fuzzy_source.rs`

```rust
//! Fuzzy pattern matching source using liblevenshtein transducer.
//!
//! This module provides approximate string matching capabilities for MORK queries.
//! It wraps liblevenshtein's transducer to enable fuzzy symbol lookup in PathMap-backed
//! knowledge graphs.

use liblevenshtein::transducer::{Algorithm, Candidate, Transducer};
use libdictenstein::pathmap::PathMapDictionary;
use mork_expr::{Expr, ExprEnv, item_byte, Tag};
use pathmap::PathMap;
use pathmap::zipper::{ReadZipperUntracked, Zipper, ZipperMoving, ZipperIteration};

/// Configuration for fuzzy matching behavior.
///
/// # Example
/// ```rust
/// let config = FuzzyConfig {
///     max_distance: 2,
///     algorithm: Algorithm::Transposition,  // OSA / restricted Damerau
///     include_exact: true,
/// };
/// ```
#[derive(Clone, Debug)]
pub struct FuzzyConfig {
    /// Maximum edit distance for matches (typically 1-3)
    pub max_distance: usize,

    /// Algorithm to use for distance calculation
    /// - `Standard`: Insert, delete, substitute
    /// - `Transposition`: Adds adjacent swaps under optimal string alignment
    /// - `MergeAndSplit`: OCR-optimized two-char ↔ one-char operations
    pub algorithm: Algorithm,

    /// Whether to include exact matches (distance 0) in results
    pub include_exact: bool,
}

impl Default for FuzzyConfig {
    fn default() -> Self {
        Self {
            max_distance: 2,
            algorithm: Algorithm::Standard,
            include_exact: true,
        }
    }
}

/// Result from fuzzy matching with metadata.
#[derive(Clone, Debug)]
pub struct FuzzyResult {
    /// The matched term from the dictionary
    pub term: String,

    /// Edit distance from query to this term
    pub distance: usize,

    /// Path in PathMap where this term was found
    pub path: Vec<u8>,
}
```

### Step 3: Create FuzzyDictionaryView

```rust
use liblevenshtein::dictionary::phonetic_normalized::{
    PhoneticNormalizedDictionary, PhoneticNormalizedCandidate
};
use liblevenshtein::phonetic::rules::english;
use pathmap::PathMap;
use pathmap::zipper::{ReadZipperUntracked, ZipperIteration};

/// Wrapper providing phonetic-aware fuzzy lookup over a PathMap subtrie.
///
/// This struct builds a PhoneticNormalizedDictionary from PathMap symbols
/// and provides phonetic-aware fuzzy query capabilities.
pub struct FuzzyDictionaryView {
    dict: PhoneticNormalizedDictionary<()>,
}

impl FuzzyDictionaryView {
    /// Build a fuzzy dictionary view from a PathMap at a given prefix.
    ///
    /// # Arguments
    /// * `map` - The PathMap to extract symbols from
    /// * `prefix` - Path prefix to scope the dictionary
    ///
    /// # Example
    /// ```rust
    /// let view = FuzzyDictionaryView::new(&space.btm, b"symbols/");
    /// for candidate in view.fuzzy_lookup("color", 2) {
    ///     println!("{}: distance {}, normalized='{}'",
    ///         candidate.term, candidate.distance, candidate.normalized_form);
    /// }
    /// ```
    pub fn new(map: &PathMap<()>, prefix: &[u8]) -> Self {
        let terms = Self::extract_terms(map, prefix);
        let combined = english::combined();
        let dict = PhoneticNormalizedDictionary::<()>::from_terms_with_rules(terms, combined);
        Self { dict }
    }

    /// Extract all terminal symbols under the given prefix as dictionary terms.
    fn extract_terms(map: &PathMap<()>, prefix: &[u8]) -> Vec<String> {
        let mut terms = Vec::new();
        let mut rz = map.read_zipper();

        if !rz.descend_to(prefix) {
            return terms; // Prefix not found
        }

        while rz.to_next_val() {
            let path = rz.path();
            if path.len() > prefix.len() {
                let symbol = &path[prefix.len()..];
                if let Ok(term) = std::str::from_utf8(symbol) {
                    terms.push(term.to_string());
                }
            }
        }

        terms
    }

    /// Perform phonetic-aware fuzzy lookup.
    ///
    /// # Arguments
    /// * `query` - The query term
    /// * `max_distance` - Maximum edit distance for matches
    ///
    /// # Returns
    /// Vector of `PhoneticNormalizedCandidate` with term, distance, and normalized form.
    pub fn fuzzy_lookup(&self, query: &str, max_distance: usize)
        -> Vec<PhoneticNormalizedCandidate>
    {
        self.dict.query(query, max_distance)
    }

    /// Get the normalized form of a query string.
    pub fn normalize(&self, query: &str) -> String {
        self.dict.normalize(query)
    }
}
```

### Step 4: Create FuzzyPhoneticSource

```rust
use liblevenshtein::dictionary::phonetic_normalized::{
    PhoneticNormalizedDictionary, PhoneticNormalizedCandidate
};
use liblevenshtein::phonetic::rules::english;
use crate::sources::{Source, ResourceRequest, Resource, AFactor};
use mork_expr::{Expr, Tag, item_byte};

/// Phonetic-aware fuzzy source using PhoneticNormalizedDictionary.
///
/// FuzzyPhoneticSource matches expressions of the form:
/// ```metta
/// (FUZZY-PHONETIC max_distance pattern)
/// ```
///
/// Where:
/// - `max_distance` is an integer (1-255)
/// - `pattern` is the symbol to match with phonetic awareness
///
/// # Example
/// ```metta
/// !(match &space (fuzzy-phonetic "fone" 2 $result) $result)
/// ; Returns: phone, phon, fawn, etc. (phonetically similar terms)
/// ```
pub struct FuzzyPhoneticSource {
    /// Original expression for error messages
    e: Expr,

    /// Maximum edit distance
    max_distance: usize,

    /// The pattern symbol to match
    pattern_symbol: Vec<u8>,
}

impl FuzzyPhoneticSource {
    /// Create a new FuzzyPhoneticSource from a MORK expression.
    ///
    /// # Expression Format
    /// ```
    /// (FUZZY-PHONETIC max_dist pattern)
    ///  ^               ^        ^
    ///  |               |        └── Symbol to match
    ///  |               └── Maximum edit distance (u8)
    ///  └── Arity 3 with "FUZZY-PHONETIC" head
    /// ```
    pub fn new(e: Expr) -> Self {
        let (max_distance, pattern_symbol) = Self::parse_expr(e);
        Self { e, max_distance, pattern_symbol }
    }

    fn parse_expr(e: Expr) -> (usize, Vec<u8>) {
        unsafe {
            let ptr = e.ptr;
            // Skip arity tag and "FUZZY-PHONETIC" symbol
            // Format: (FUZZY-PHONETIC max_dist pattern)
            let dist_ptr = ptr.add(1 + 1 + 14); // Arity(3) + SymbolSize(14) + "FUZZY-PHONETIC"
            let max_distance = (*dist_ptr & 0x3F) as usize;

            let pattern_ptr = dist_ptr.add(1);
            let pattern_len = (*pattern_ptr & 0x3F) as usize;
            let pattern_data = std::slice::from_raw_parts(pattern_ptr.add(1), pattern_len);

            (max_distance, pattern_data.to_vec())
        }
    }
}

/// Standard Levenshtein fuzzy source (non-phonetic).
///
/// FuzzySource matches expressions of the form:
/// ```metta
/// (FUZZY max_distance pattern)
/// ```
///
/// # Example
/// ```metta
/// !(match &space (fuzzy "colr" 2 $result) $result)
/// ; Returns: color, colour, collar (edit distance only, no phonetic rules)
/// ```
pub struct FuzzySource {
    e: Expr,
    config: FuzzyConfig,
    pattern_symbol: Vec<u8>,
}

impl FuzzySource {
    pub fn new(e: Expr) -> Self {
        let (config, pattern_symbol) = Self::parse_fuzzy_expr(e);
        Self { e, config, pattern_symbol }
    }

    fn parse_fuzzy_expr(e: Expr) -> (FuzzyConfig, Vec<u8>) {
        unsafe {
            let ptr = e.ptr;
            // Skip arity tag and "FUZZY" symbol
            // Arity(3) + SymbolSize(5) + "FUZZY" = 1 + 1 + 5 = 7 bytes
            let dist_ptr = ptr.add(7);
            let max_distance = (*dist_ptr & 0x3F) as usize;

            let pattern_ptr = dist_ptr.add(1);
            let pattern_len = (*pattern_ptr & 0x3F) as usize;
            let pattern_data = std::slice::from_raw_parts(pattern_ptr.add(1), pattern_len);

            let config = FuzzyConfig {
                max_distance,
                algorithm: Algorithm::Standard,
                include_exact: true,
            };

            (config, pattern_data.to_vec())
        }
    }
}
```

### Step 5: Implement Source Trait

```rust
use crate::sources::{Source, ResourceRequest, Resource, AFactor};

/// Source implementation for phonetic-aware fuzzy matching.
impl Source for FuzzyPhoneticSource {
    fn new(e: Expr) -> Self {
        FuzzyPhoneticSource::new(e)
    }

    fn request(&self) -> impl Iterator<Item = ResourceRequest> {
        // Request access to the main BTM store for dictionary construction
        std::iter::once(ResourceRequest::BTM(&[]))
    }

    fn source<'trie, 'path, It>(
        &self,
        mut it: It,
        _path: &[u8],
    ) -> AFactor<'trie, ()>
    where
        It: Iterator<Item = Resource<'trie, 'path>>,
        'path: 'trie,
    {
        // Get the BTM resource
        let btm = match it.next() {
            Some(Resource::BTM(map)) => map,
            _ => panic!("FuzzyPhoneticSource requires BTM resource"),
        };

        // Build PhoneticNormalizedDictionary from PathMap
        let view = FuzzyDictionaryView::new(btm, &[]);

        // Query with phonetic normalization
        let query_str = String::from_utf8_lossy(&self.pattern_symbol);
        let candidates = view.fuzzy_lookup(&query_str, self.max_distance);

        // Return as FuzzyZipper wrapped in AFactor
        AFactor::FuzzyPhoneticSource(FuzzyZipper::new(candidates, &[]))
    }
}

/// Source implementation for standard Levenshtein fuzzy matching (non-phonetic).
impl Source for FuzzySource {
    fn new(e: Expr) -> Self {
        FuzzySource::new(e)
    }

    fn request(&self) -> impl Iterator<Item = ResourceRequest> {
        std::iter::once(ResourceRequest::BTM(&[]))
    }

    fn source<'trie, 'path, It>(
        &self,
        mut it: It,
        _path: &[u8],
    ) -> AFactor<'trie, ()>
    where
        It: Iterator<Item = Resource<'trie, 'path>>,
        'path: 'trie,
    {
        let btm = match it.next() {
            Some(Resource::BTM(map)) => map,
            _ => panic!("FuzzySource requires BTM resource"),
        };

        // For standard fuzzy, build without phonetic rules
        // Uses standard Levenshtein transducer
        let view = FuzzyDictionaryView::new(btm, &[]);
        let query_str = String::from_utf8_lossy(&self.pattern_symbol);
        let candidates = view.fuzzy_lookup(&query_str, self.config.max_distance);

        AFactor::FuzzySource(FuzzyZipper::new(candidates, &[]))
    }
}
```

### Step 6: Create FuzzyZipper

**File**: `MORK/kernel/src/fuzzy_zipper.rs`

```rust
//! Zipper adapter that presents PhoneticNormalizedCandidate results as a virtual trie.
//!
//! FuzzyZipper wraps a vector of candidates and presents them as a navigable
//! path structure compatible with MORK's ProductZipper.

use liblevenshtein::dictionary::phonetic_normalized::PhoneticNormalizedCandidate;
use pathmap::zipper::{Zipper, ZipperIteration, ZipperAbsolutePath};

/// A zipper that iterates over phonetic fuzzy match candidates.
///
/// Presents PhoneticNormalizedDictionary results as navigable paths for
/// integration with MORK's query pipeline.
pub struct FuzzyZipper {
    /// Candidates from PhoneticNormalizedDictionary
    candidates: std::vec::IntoIter<PhoneticNormalizedCandidate>,

    /// Current candidate (if any)
    current: Option<PhoneticNormalizedCandidate>,

    /// Buffer for constructing path representation
    path_buffer: Vec<u8>,

    /// Prefix for all result paths
    prefix: Vec<u8>,
}

impl FuzzyZipper {
    /// Create a new FuzzyZipper from a vector of candidates.
    ///
    /// # Arguments
    /// * `candidates` - Vector of PhoneticNormalizedCandidate from query
    /// * `prefix` - Path prefix for result paths
    pub fn new(candidates: Vec<PhoneticNormalizedCandidate>, prefix: &[u8]) -> Self {
        let mut zipper = Self {
            candidates: candidates.into_iter(),
            current: None,
            path_buffer: Vec::with_capacity(256),
            prefix: prefix.to_vec(),
        };
        zipper.advance();
        zipper
    }

    /// Advance to the next candidate.
    fn advance(&mut self) {
        self.current = self.candidates.next();
        self.rebuild_path();
    }

    /// Rebuild the path buffer from current candidate.
    fn rebuild_path(&mut self) {
        self.path_buffer.clear();
        self.path_buffer.extend_from_slice(&self.prefix);
        if let Some(ref c) = self.current {
            self.path_buffer.extend_from_slice(c.term.as_bytes());
        }
    }

    /// Get the edit distance of the current candidate.
    pub fn current_distance(&self) -> Option<usize> {
        self.current.as_ref().map(|c| c.distance)
    }

    /// Get the term of the current candidate.
    pub fn current_term(&self) -> Option<&str> {
        self.current.as_ref().map(|c| c.term.as_str())
    }

    /// Get the normalized form of the current candidate.
    ///
    /// Useful for debugging and understanding why a match occurred.
    pub fn current_normalized(&self) -> Option<&str> {
        self.current.as_ref().map(|c| c.normalized_form.as_str())
    }
}

impl ZipperIteration for FuzzyZipper {
    fn to_next_val(&mut self) -> bool {
        if self.current.is_some() {
            self.advance();
            self.current.is_some()
        } else {
            false
        }
    }

    fn is_val(&self) -> bool {
        self.current.is_some()
    }

    fn to_next_step(&mut self) -> bool {
        self.to_next_val()
    }
}

impl ZipperAbsolutePath for FuzzyZipper {
    fn path(&self) -> &[u8] {
        &self.path_buffer
    }

    fn origin_path(&self) -> &[u8] {
        &self.path_buffer
    }
}

impl Zipper for FuzzyZipper {
    type Value = ();

    fn val(&self) -> Option<&Self::Value> {
        if self.current.is_some() { Some(&()) } else { None }
    }
}
```

### Step 7: Extend ASource Enum

**File**: `MORK/kernel/src/sources.rs` (modifications)

```rust
// Add to imports
use crate::fuzzy_source::{FuzzySource, FuzzyPhoneticSource};
use crate::fuzzy_zipper::FuzzyZipper;

// Add variants to ASource enum
pub enum ASource {
    PosSource(BTMSource),
    ACTSource(ACTSource),
    CmpSource(CmpSource),
    FuzzySource(FuzzySource),           // Standard Levenshtein
    FuzzyPhoneticSource(FuzzyPhoneticSource),  // Phonetic-aware
}

impl Source for ASource {
    fn new(e: Expr) -> Self {
        unsafe {
            // Check for FUZZY-PHONETIC pattern first (longer symbol)
            // (FUZZY-PHONETIC max_dist pattern)
            if *e.ptr == item_byte(Tag::Arity(3)) {
                let second = e.ptr.add(1);
                if *second == item_byte(Tag::SymbolSize(14)) {
                    let sym = std::slice::from_raw_parts(second.add(1), 14);
                    if sym == b"FUZZY-PHONETIC" {
                        return ASource::FuzzyPhoneticSource(FuzzyPhoneticSource::new(e));
                    }
                }
                // Check for FUZZY pattern: (FUZZY max_dist pattern)
                if *second == item_byte(Tag::SymbolSize(5)) {
                    let sym = std::slice::from_raw_parts(second.add(1), 5);
                    if sym == b"FUZZY" {
                        return ASource::FuzzySource(FuzzySource::new(e));
                    }
                }
            }
        }

        // ... existing pattern matching for other sources ...

        // Default to BTMSource
        ASource::PosSource(BTMSource::new(e))
    }

    // ... other trait methods ...
}

// Add variants to AFactor enum
#[derive(PolyZipper)]
pub enum AFactor<'trie, V: Clone + Send + Sync + Unpin + 'static = ()> {
    PosSource(PrefixZipper<'trie, ReadZipperUntracked<'trie, 'trie, V>>),
    ACTSource(PrefixZipper<'trie, ACTMmapZipper<'trie, V>>),
    CmpSource(/* ... */),
    FuzzySource(FuzzyZipper),         // Standard Levenshtein results
    FuzzyPhoneticSource(FuzzyZipper), // Phonetic-aware results
}
```

---

## Usage Examples

This section provides comprehensive examples of FuzzySource usage in MeTTa queries. For additional examples and architectural context, see [README.md](./README.md#metta-query-examples).

### Knowledge Base Setup

The examples below assume the following knowledge base:

```metta
; Entity definitions with names and types
(= (entity-name "e1") "color")
(= (entity-name "e2") "colour")
(= (entity-name "e3") "collar")
(= (entity-name "e4") "blue")
(= (entity-type "e1") "property")
(= (entity-type "e2") "property")
(= (entity-type "e3") "object")
(= (entity-type "e4") "property")

; Person records
(person john (age 30) (dept engineering))
(person jon (age 25) (dept sales))
(person jane (age 28) (dept engineering))
(person joan (age 35) (dept marketing))
```

### Basic Fuzzy Query

```metta
; Find symbols within edit distance 2 of "colr" (a typo)
!(match &space (fuzzy "colr" 2 $result) $result)

; Returns: "color" "colour" "collar"
; Explanation:
;   "color"  → distance 1 (insert 'o')
;   "colour" → distance 2 (insert 'o', insert 'u')
;   "collar" → distance 2 (substitute 'o'→'a', substitute 'r'→'l', substitute end)
;   "blue"   → distance 4 (not returned, exceeds max_distance)
```

### Variable Binding with Distance

```metta
; Get matched terms with the fuzzy binding variable
!(match &space
    (= (entity-name $entity) (fuzzy "colr" 2 $matched-name))
    ($entity $matched-name))

; Returns: ("e1" "color") ("e2" "colour") ("e3" "collar")
```

### Combining Fuzzy with Exact Constraints

The key advantage of FuzzySource in MORK is composing fuzzy matching with additional pattern constraints in a single unified query:

```metta
; Find entities with names similar to "colr" AND type = "property"
!(match &space
    (= (entity-name $entity) (fuzzy "colr" 2 $name))
    (= (entity-type $entity) "property")
    ($entity $name))

; Returns: ("e1" "color") ("e2" "colour")
; Note: "collar" excluded because entity-type is "object", not "property"
```

How this works internally:
1. FuzzySource yields candidates: `color`, `colour`, `collar`
2. For each candidate, MORK checks the `(= (entity-type $entity) "property")` constraint
3. Only entities satisfying BOTH constraints are returned in a single pass

### Fuzzy Query on Nested Structures

```metta
; Find persons with fuzzy name matching in structured records
!(match &space
    (person (fuzzy "john" 1 $name) (age $age) (dept $dept))
    ($name $age $dept))

; Returns:
; ("john" 30 engineering)  ; exact match, distance 0
; ("jon" 25 sales)         ; distance 1 (delete 'h')
; ("joan" 35 marketing)    ; distance 1 (substitute 'h'→'a')
; Note: "jane" not returned (distance 2 > max_distance of 1)
```

### Multiple Fuzzy Constraints

```metta
; Query with multiple fuzzy fields
!(match &space
    (= (entity-name $e1) (fuzzy "colr" 2 $name1))
    (= (entity-type $e1) "property")
    (= (entity-name $e2) (fuzzy "blu" 2 $name2))
    (= (entity-type $e2) "property")
    (($e1 $name1) ($e2 $name2)))

; Returns all pairs of (color-like property, blue-like property)
; (("e1" "color") ("e4" "blue"))
; (("e2" "colour") ("e4" "blue"))
```

### Algorithm Selection (Extended Syntax)

```metta
; Standard Levenshtein (insert, delete, substitute)
!(match &space (fuzzy "teh" 1 standard $result) $result)
; Returns: "the" (distance 1 via substitute)

; Transposition (optimal string alignment / restricted Damerau)
!(match &space (fuzzy "teh" 1 transposition $result) $result)
; Returns: "the" (distance 1 via swap 'e'↔'h')

; MergeAndSplit (OCR-optimized, handles character splitting)
!(match &space (fuzzy "rn" 1 merge-and-split $result) $result)
; Returns: "m" (distance 1, recognizes "rn" ≈ "m" in OCR errors)
```

### Error Handling

```metta
; Query with typo in type constraint (demonstrates fuzzy doesn't affect exact matches)
!(match &space
    (= (entity-name $entity) (fuzzy "colr" 2 $name))
    (= (entity-type $entity) "proprety")  ; Typo! Should be "property"
    ($entity $name))

; Returns: () (empty - no entities have type "proprety")
; The fuzzy specifier only applies to its direct argument, not to other constraints
```

```metta
; Distance 0 requires exact match
!(match &space (fuzzy "color" 0 $result) $result)

; Returns: "color" only (exact match required)
```

### Ranking Integration (Phase B Preview)

With Phase B lattice infrastructure, results include distance for ranking:

```metta
; Get top 5 matches with edit distances
!(match &space
    (= (entity-name $entity) (fuzzy-ranked "colr" 2 5 $name $distance))
    ($entity $name $distance))

; Returns (ordered by distance):
; ("e1" "color" 1)
; ("e2" "colour" 2)
; ("e3" "collar" 2)
```

### Phonetic Matching [CURRENT]

Phonetic-aware matching is fully available via `fuzzy-phonetic` and `PhoneticNormalizedDictionary`.

```metta
; Find names that sound like "Steven" using phonetic rules
!(match &space
    (person (fuzzy-phonetic "Steven" 2 $name) $attrs)
    ($name $attrs))

; Returns: ("Stephen" ...) ("Stefan" ...) ("Stephan" ...) ("Steve" ...)
; Phonetic rules: "ph" ≈ "f", "v" ≈ "ph", vowel variations
; Expected output:
; [("Stephen", (age 45, dept research)),
;  ("Stefan", (age 32, dept engineering)),
;  ("Steve", (age 28, dept marketing))]
```

```metta
; Find words that sound like "fone" (phonetic misspelling of "phone")
!(match &space (fuzzy-phonetic "fone" 2 $result) $result)

; Returns: "phone", "phon", etc.
; Expected output: ["phone", "phon"]
; Explanation: "fone" normalizes to "fon", which matches "phone" (→ "fon") at distance 0
```

The underlying implementation uses `PhoneticNormalizedDictionary`:

```rust
use liblevenshtein::dictionary::phonetic_normalized::PhoneticNormalizedDictionary;
use liblevenshtein::phonetic::rules::english;

// Build dictionary with combined English phonetic rules
let dict = PhoneticNormalizedDictionary::<()>::from_terms_with_rules(
    &words,
    english::combined()
);

// Query with phonetic normalization
let results = dict.query("fone", 2);
// results contains PhoneticNormalizedCandidate:
//   - term: "phone"
//   - distance: 0  (in normalized space)
//   - normalized_form: "fon"
```

---

## Testing

### Unit Tests

**File**: `MORK/kernel/src/fuzzy_source.rs` (add at bottom)

```rust
#[cfg(test)]
mod tests {
    use super::*;
    use liblevenshtein::dictionary::phonetic_normalized::PhoneticNormalizedCandidate;

    #[test]
    fn test_fuzzy_config_default() {
        let config = FuzzyConfig::default();
        assert_eq!(config.max_distance, 2);
        assert!(config.include_exact);
    }

    #[test]
    fn test_fuzzy_dictionary_view_empty() {
        let map = PathMap::new();
        let view = FuzzyDictionaryView::new(&map, b"");

        let results = view.fuzzy_lookup("test", 2);
        assert!(results.is_empty());
    }

    #[test]
    fn test_fuzzy_dictionary_view_matches() {
        let mut map = PathMap::new();
        map.set_val_at(b"color", ());
        map.set_val_at(b"colour", ());
        map.set_val_at(b"collar", ());
        map.set_val_at(b"zebra", ());

        let view = FuzzyDictionaryView::new(&map, b"");

        let results = view.fuzzy_lookup("color", 2);

        assert!(results.iter().any(|c| c.term == "color" && c.distance == 0));
        assert!(results.iter().any(|c| c.term == "colour"));
        assert!(results.iter().any(|c| c.term == "collar"));
        assert!(!results.iter().any(|c| c.term == "zebra"));
    }

    #[test]
    fn test_fuzzy_dictionary_view_phonetic() {
        let mut map = PathMap::new();
        map.set_val_at(b"phone", ());
        map.set_val_at(b"phon", ());
        map.set_val_at(b"fone", ());
        map.set_val_at(b"zebra", ());

        let view = FuzzyDictionaryView::new(&map, b"");

        // "fone" should match "phone" due to phonetic normalization
        let results = view.fuzzy_lookup("fone", 2);

        // Check that we get phonetically similar matches
        assert!(results.iter().any(|c| c.term == "phone"));
        assert!(!results.iter().any(|c| c.term == "zebra"));
    }

    #[test]
    fn test_fuzzy_zipper_iteration() {
        let candidates = vec![
            PhoneticNormalizedCandidate {
                term: "color".to_string(),
                distance: 0,
                normalized_form: "color".to_string(),
            },
            PhoneticNormalizedCandidate {
                term: "colour".to_string(),
                distance: 1,
                normalized_form: "colour".to_string(),
            },
        ];

        let mut zipper = FuzzyZipper::new(candidates, b"prefix/");

        assert!(zipper.is_val());
        assert_eq!(zipper.current_term(), Some("color"));
        assert_eq!(zipper.current_distance(), Some(0));
        assert_eq!(zipper.current_normalized(), Some("color"));

        assert!(zipper.to_next_val());
        assert_eq!(zipper.current_term(), Some("colour"));
        assert_eq!(zipper.current_distance(), Some(1));

        assert!(!zipper.to_next_val());
        assert!(!zipper.is_val());
    }

    #[test]
    fn test_fuzzy_zipper_phonetic_normalized() {
        let candidates = vec![
            PhoneticNormalizedCandidate {
                term: "phone".to_string(),
                distance: 0,
                normalized_form: "fon".to_string(),
            },
        ];

        let zipper = FuzzyZipper::new(candidates, b"");

        assert_eq!(zipper.current_term(), Some("phone"));
        assert_eq!(zipper.current_normalized(), Some("fon"));
    }
}
```

### Integration Tests

**File**: `MORK/kernel/tests/fuzzy_integration.rs`

```rust
//! Integration tests for FuzzySource and FuzzyPhoneticSource in MORK query pipeline.

use mork_kernel::prelude::*;

#[test]
fn test_fuzzy_query_basic() {
    let mut space = Space::new();

    // Insert test data
    space.insert_sexpr("(word color)");
    space.insert_sexpr("(word colour)");
    space.insert_sexpr("(word collar)");
    space.insert_sexpr("(word zebra)");

    // Query with standard fuzzy matching (Levenshtein only)
    let results = space.query_sexpr("(word (fuzzy \"color\" 2 $result))");

    assert!(results.contains("color"));
    assert!(results.contains("colour"));
    assert!(results.contains("collar"));
    assert!(!results.contains("zebra"));
}

#[test]
fn test_fuzzy_phonetic_query() {
    let mut space = Space::new();

    // Insert test data
    space.insert_sexpr("(word phone)");
    space.insert_sexpr("(word phon)");
    space.insert_sexpr("(word fone)");
    space.insert_sexpr("(word zebra)");

    // Query with phonetic-aware fuzzy matching
    let results = space.query_sexpr("(word (fuzzy-phonetic \"fone\" 2 $result))");

    // "fone" should match "phone" due to phonetic normalization (ph → f)
    assert!(results.contains("phone"));
    assert!(results.contains("phon"));
    assert!(results.contains("fone"));
    assert!(!results.contains("zebra"));
}

#[test]
fn test_fuzzy_query_with_unification() {
    let mut space = Space::new();

    space.insert_sexpr("(person john (age 30))");
    space.insert_sexpr("(person jon (age 25))");
    space.insert_sexpr("(person jane (age 28))");

    let results = space.query_sexpr("(person (fuzzy \"john\" 1 $name) $age)");

    // Should match "john" (distance 0) and "jon" (distance 1)
    assert_eq!(results.len(), 2);
}

#[test]
fn test_fuzzy_phonetic_names() {
    let mut space = Space::new();

    space.insert_sexpr("(person Stephen (age 45))");
    space.insert_sexpr("(person Stefan (age 32))");
    space.insert_sexpr("(person Steve (age 28))");
    space.insert_sexpr("(person Alice (age 30))");

    // Phonetic query for "Steven" should find phonetically similar names
    let results = space.query_sexpr("(person (fuzzy-phonetic \"Steven\" 2 $name) $attrs)");

    // Stephen, Stefan, Steve are phonetically similar to Steven
    assert!(results.iter().any(|r| r.contains("Stephen")));
    assert!(results.iter().any(|r| r.contains("Stefan")));
    assert!(results.iter().any(|r| r.contains("Steve")));
    assert!(!results.iter().any(|r| r.contains("Alice")));
}
```

---

## Performance Considerations

### Dictionary Building

The `FuzzyDictionaryView::new()` function iterates over all values under a prefix. For large PathMaps, consider:

1. **Caching**: Cache the dictionary view for repeated queries
2. **Prefix scoping**: Use specific prefixes to limit dictionary size
3. **Lazy building**: Build dictionary incrementally during query

### Query Latency

Target: <10ms for typical queries (1-3 terms, distance 1-2, dictionary size <100K).

Factors affecting latency:
- Dictionary size: $`\mathcal{O}(n)`$ iteration to find all candidates
- Max distance: Higher distance = more candidates to check
- Algorithm: Transposition slightly slower than Standard

### Memory Usage

- FuzzyDictionaryView: Temporary dictionary copy
- FuzzyZipper: Minimal (iterator state + path buffer)
- Candidate collection: One string per match

---

## Future Enhancements

1. **Weighted results**: Return candidates with combined phonetic + edit distance scores
2. **Algorithm selection**: Parse algorithm from expression (Standard, Transposition, MergeAndSplit)
3. **Streaming iteration**: Avoid collecting all candidates for large result sets
4. **Caching**: Cache PhoneticNormalizedDictionary views across queries
5. **Language-specific rules**: Support for non-English phonetic rules (German, French, etc.)
6. **Custom rule sets**: Allow users to define domain-specific phonetic rules

---

## References

- [MORK Source Trait](../../../../MORK/kernel/src/sources.rs)
- [PhoneticNormalizedDictionary](../../../src/dictionary/phonetic_normalized.rs)
- [PhoneticNormalizedCandidate](../../../src/dictionary/phonetic_normalized.rs)
- [English Phonetic Rules](../../../src/phonetic/rules/english.rs)
- PathMap Dictionary Backend
- [PathMap Zipper Traits](../../../../PathMap/src/zipper.rs)
