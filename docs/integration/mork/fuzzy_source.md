# Phase A: FuzzySource Implementation Guide

**Last Updated**: 2025-12-21
**Version**: v0.8.0
**Status**: Partially Implemented

This document provides detailed implementation guidance for Phase A of the MORK integration: creating a FuzzySource that enables approximate string matching in MeTTa queries.

> **Note**: This document describes the proposed FuzzySource adapter for MORK. liblevenshtein v0.8.0 provides the underlying APIs (`Transducer`, `ProductAutomatonChar`, `english::*` rules) that this adapter would wrap.

## Overview

**Goal**: Enable MORK queries to perform fuzzy symbol matching using liblevenshtein's transducer.

**Result**: MeTTa queries like `!(match &space (fuzzy "colr" 2 $result) $result)` return approximate matches within edit distance 2.

---

## v0.8.0 liblevenshtein APIs

FuzzySource can leverage these v0.8.0 APIs from liblevenshtein:

### Basic Transducer (Levenshtein)

```rust
use liblevenshtein::transducer::{Algorithm, Transducer};
use liblevenshtein::dictionary::DynamicDawgChar;

let dict = DynamicDawgChar::from_iter(["color", "colour", "collar", "blue"]);
let transducer = Transducer::new(&dict, Algorithm::Transposition);

for candidate in transducer.query("colr", 2) {
    println!("{}: distance {}", candidate.term, candidate.distance);
}
```

### ProductAutomatonChar (Phonetic + Levenshtein)

For phonetic-aware matching, use `ProductAutomatonChar`:

```rust
use liblevenshtein::phonetic::nfa::ProductAutomatonChar;
use liblevenshtein::phonetic::verified::rules_to_nfa_char;
use liblevenshtein::phonetic::rules::english;

// Pre-compiled English rules (62 Zompist orthographic rules)
let rules = english::zompist();
let nfa = rules_to_nfa_char(&rules.rules);
let product = ProductAutomatonChar::new(nfa, 2);

// Check acceptance (considers both phonetic rules and edit distance)
assert!(product.accepts("phone"));  // Matches via ph→f rule
```

### Compile-Time Pattern Embedding

```rust
use liblevenshtein::{llre, llev};

// LLRE: Compile-time regex → NFA (embedded in binary)
let phone_pattern = llre!(r"(ph|f)one");

// LLEV: Compile-time phonetic rules
let rules = llev!(r#"
    ph -> f;
    gh -> / [:vowel:]_;
"#);
```

### Pre-Compiled English Rule Sets

```rust
use liblevenshtein::phonetic::rules::english;

let zompist = english::zompist();        // 62 orthographic rules
let homophones = english::homophones();  // Homophone pairs
let text_speak = english::text_speak();  // Text-speak expansions

// Combine rule sets
use liblevenshtein::phonetic::RuleSetChar;
let combined = RuleSetChar::new().merge(&zompist).merge(&homophones);
```

---

## Architecture

### Data Flow

```
MeTTa Query
    |
    v
MORK Query Parser
    |
    | Recognizes (fuzzy ...) pattern
    v
FuzzySource::new(expr)
    |
    | Parses: max_distance, algorithm, pattern
    v
FuzzySource::request()
    |
    | Requests BTM (PathMap) access
    v
FuzzySource::source()
    |
    | Builds FuzzyDictionaryView from PathMap
    | Creates Transducer<PathMapDictionary>
    | Returns FuzzyZipper over candidates
    v
ProductZipper (combines with other sources)
    |
    v
Unification with query pattern
    |
    v
Results
```

### Component Relationships

```
┌─────────────────────────────────────────────────────────────┐
│ MORK/kernel/src/                                            │
│                                                             │
│  ┌─────────────────┐     ┌─────────────────┐               │
│  │ sources.rs      │     │ fuzzy_source.rs │ (NEW)         │
│  │                 │     │                 │               │
│  │ ASource enum    │◄────│ FuzzySource     │               │
│  │   BTMSource     │     │ FuzzyConfig     │               │
│  │   ACTSource     │     │ FuzzyResult     │               │
│  │   FuzzySource   │     └────────┬────────┘               │
│  └────────┬────────┘              │                        │
│           │                       │                        │
│           v                       v                        │
│  ┌─────────────────┐     ┌─────────────────┐               │
│  │ AFactor enum    │     │ fuzzy_zipper.rs │ (NEW)         │
│  │   PosSource     │     │                 │               │
│  │   ACTSource     │◄────│ FuzzyZipper     │               │
│  │   FuzzySource   │     │                 │               │
│  └─────────────────┘     └────────┬────────┘               │
│                                   │                        │
└───────────────────────────────────│────────────────────────┘
                                    │
                                    v
┌─────────────────────────────────────────────────────────────┐
│ liblevenshtein-rust/src/                                    │
│                                                             │
│  ┌─────────────────┐     ┌─────────────────┐               │
│  │ transducer/     │     │ dictionary/     │               │
│  │   mod.rs        │     │   pathmap.rs    │               │
│  │   Transducer<D> │◄────│ PathMapDict     │               │
│  │   Candidate     │     │                 │               │
│  └─────────────────┘     └─────────────────┘               │
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
use liblevenshtein::dictionary::pathmap::PathMapDictionary;
use mork_expr::{Expr, ExprEnv, item_byte, Tag};
use pathmap::PathMap;
use pathmap::zipper::{ReadZipperUntracked, Zipper, ZipperMoving, ZipperIteration};

/// Configuration for fuzzy matching behavior.
///
/// # Example
/// ```rust
/// let config = FuzzyConfig {
///     max_distance: 2,
///     algorithm: Algorithm::Transposition,  // Damerau-Levenshtein
///     include_exact: true,
/// };
/// ```
#[derive(Clone, Debug)]
pub struct FuzzyConfig {
    /// Maximum edit distance for matches (typically 1-3)
    pub max_distance: usize,

    /// Algorithm to use for distance calculation
    /// - `Standard`: Insert, delete, substitute
    /// - `Transposition`: Adds adjacent character swaps (Damerau-Levenshtein)
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
/// Wrapper providing fuzzy lookup over a PathMap subtrie.
///
/// This struct builds a temporary dictionary from PathMap symbols
/// and provides fuzzy query capabilities via liblevenshtein.
pub struct FuzzyDictionaryView {
    transducer: Transducer<PathMapDictionary<()>>,
}

impl FuzzyDictionaryView {
    /// Build a fuzzy dictionary view from a PathMap at a given prefix.
    ///
    /// # Arguments
    /// * `map` - The PathMap to extract symbols from
    /// * `prefix` - Path prefix to scope the dictionary
    /// * `config` - Fuzzy matching configuration
    ///
    /// # Example
    /// ```rust
    /// let view = FuzzyDictionaryView::new(&space.btm, b"symbols/", &config);
    /// for candidate in view.fuzzy_lookup(b"color", 2) {
    ///     println!("{}: distance {}", candidate.term, candidate.distance);
    /// }
    /// ```
    pub fn new(map: &PathMap<()>, prefix: &[u8], config: &FuzzyConfig) -> Self {
        let dict = Self::build_dictionary_from_prefix(map, prefix);
        let transducer = Transducer::new(dict, config.algorithm);
        Self { transducer }
    }

    /// Extract all terminal symbols under the given prefix as dictionary terms.
    fn build_dictionary_from_prefix(map: &PathMap<()>, prefix: &[u8]) -> PathMapDictionary<()> {
        let mut dict = PathMapDictionary::new();

        // Navigate to prefix in PathMap
        let mut rz = map.read_zipper();
        if !rz.descend_to(prefix) {
            return dict;  // Prefix not found, return empty dictionary
        }

        // Collect all terminal symbols (values) under this prefix
        while rz.to_next_val() {
            let path = rz.path();
            // Extract the symbol portion after the prefix
            if path.len() > prefix.len() {
                let symbol = &path[prefix.len()..];
                // Convert to string for dictionary
                if let Ok(term) = std::str::from_utf8(symbol) {
                    dict.insert(term);
                }
            }
        }

        dict
    }

    /// Perform fuzzy lookup and return matching candidates.
    ///
    /// # Arguments
    /// * `query` - The query term as bytes
    /// * `max_distance` - Maximum edit distance for matches
    ///
    /// # Returns
    /// Iterator over `Candidate` structs containing matched terms and distances.
    pub fn fuzzy_lookup<'a>(
        &'a self,
        query: &[u8],
        max_distance: usize,
    ) -> impl Iterator<Item = Candidate> + 'a {
        let query_str = String::from_utf8_lossy(query);
        self.transducer.query_candidates(&query_str, max_distance)
    }
}
```

### Step 4: Create FuzzySource

```rust
/// Fuzzy source implementing the MORK Source trait pattern.
///
/// FuzzySource matches expressions of the form:
/// ```metta
/// (FUZZY max_distance pattern)
/// ```
///
/// Where:
/// - `max_distance` is an integer (1-255)
/// - `pattern` is the symbol to match fuzzily
///
/// # Example
/// ```metta
/// !(match &space (FUZZY 2 "color") $result)
/// ; Returns: color, colour, collar, etc.
/// ```
pub struct FuzzySource {
    /// Original expression for error messages
    e: Expr,

    /// Parsed configuration
    config: FuzzyConfig,

    /// The pattern symbol to match
    pattern_symbol: Vec<u8>,
}

impl FuzzySource {
    /// Create a new FuzzySource from a MORK expression.
    ///
    /// # Expression Format
    /// ```
    /// (FUZZY max_dist pattern)
    ///  ^      ^        ^
    ///  |      |        └── Symbol to match
    ///  |      └── Maximum edit distance (u8)
    ///  └── Arity 3 with "FUZZY" head
    /// ```
    pub fn new(e: Expr) -> Self {
        let (config, pattern_symbol) = Self::parse_fuzzy_expr(e);
        Self { e, config, pattern_symbol }
    }

    /// Parse a FUZZY expression into config and pattern.
    fn parse_fuzzy_expr(e: Expr) -> (FuzzyConfig, Vec<u8>) {
        // Safety: Expression structure validated by MORK parser
        unsafe {
            let ptr = e.ptr;

            // Skip arity tag and "FUZZY" symbol
            // Arity(3) + SymbolSize(5) + "FUZZY" = 1 + 1 + 5 = 7 bytes
            let dist_ptr = ptr.add(7);

            // Parse max_distance (assume small integer encoding)
            let max_distance = Self::parse_int_at(dist_ptr);

            // Parse pattern symbol (after distance)
            let pattern_ptr = dist_ptr.add(Self::int_size(dist_ptr));
            let pattern_symbol = Self::parse_symbol_at(pattern_ptr);

            let config = FuzzyConfig {
                max_distance,
                algorithm: Algorithm::Standard,  // Default; could parse from expression
                include_exact: true,
            };

            (config, pattern_symbol)
        }
    }

    /// Parse an integer from MORK expression encoding.
    unsafe fn parse_int_at(ptr: *const u8) -> usize {
        // MORK encodes small integers; implementation depends on encoding
        // Simplified: assume direct byte value for distance 0-63
        let tag = *ptr;
        if tag < 64 {
            tag as usize
        } else {
            2  // Default fallback
        }
    }

    /// Get size of integer encoding at pointer.
    unsafe fn int_size(_ptr: *const u8) -> usize {
        1  // Simplified; depends on actual encoding
    }

    /// Parse a symbol from MORK expression encoding.
    unsafe fn parse_symbol_at(ptr: *const u8) -> Vec<u8> {
        let tag = *ptr;
        let size = (tag & 0x3F) as usize;  // SymbolSize(n) encodes size in low bits
        let data = std::slice::from_raw_parts(ptr.add(1), size);
        data.to_vec()
    }
}
```

### Step 5: Implement Source Trait

```rust
use crate::sources::{Source, ResourceRequest, Resource, AFactor};

impl Source for FuzzySource {
    fn new(e: Expr) -> Self {
        FuzzySource::new(e)
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
            _ => panic!("FuzzySource requires BTM resource"),
        };

        // Build fuzzy dictionary view
        let view = FuzzyDictionaryView::new(btm, &[], &self.config);

        // Get fuzzy matches
        let candidates: Vec<_> = view
            .fuzzy_lookup(&self.pattern_symbol, self.config.max_distance)
            .collect();

        // Return as FuzzyZipper wrapped in AFactor
        AFactor::FuzzySource(FuzzyZipper::new(candidates.into_iter(), &[]))
    }
}
```

### Step 6: Create FuzzyZipper

**File**: `MORK/kernel/src/fuzzy_zipper.rs`

```rust
//! Zipper adapter that presents transducer results as a virtual trie.
//!
//! FuzzyZipper wraps an iterator over fuzzy match candidates and presents
//! them as a navigable path structure compatible with MORK's ProductZipper.

use liblevenshtein::transducer::Candidate;
use pathmap::zipper::{Zipper, ZipperIteration, ZipperAbsolutePath};

/// A zipper that iterates over fuzzy match candidates.
///
/// Presents transducer results as navigable paths for integration with
/// MORK's query pipeline.
pub struct FuzzyZipper<I: Iterator<Item = Candidate>> {
    /// Underlying candidate iterator
    candidates: I,

    /// Current candidate (if any)
    current_candidate: Option<Candidate>,

    /// Buffer for constructing path representation
    path_buffer: Vec<u8>,

    /// Prefix for all result paths
    prefix: Vec<u8>,
}

impl<I: Iterator<Item = Candidate>> FuzzyZipper<I> {
    /// Create a new FuzzyZipper from an iterator of candidates.
    ///
    /// # Arguments
    /// * `candidates` - Iterator over fuzzy match candidates
    /// * `prefix` - Path prefix for result paths
    pub fn new(candidates: I, prefix: &[u8]) -> Self {
        let mut zipper = Self {
            candidates,
            current_candidate: None,
            path_buffer: Vec::with_capacity(256),
            prefix: prefix.to_vec(),
        };
        // Initialize to first candidate
        zipper.advance();
        zipper
    }

    /// Advance to the next candidate.
    fn advance(&mut self) {
        self.current_candidate = self.candidates.next();
        self.rebuild_path_buffer();
    }

    /// Rebuild the path buffer from current candidate.
    fn rebuild_path_buffer(&mut self) {
        self.path_buffer.clear();
        self.path_buffer.extend_from_slice(&self.prefix);
        if let Some(ref candidate) = self.current_candidate {
            self.path_buffer.extend_from_slice(candidate.term.as_bytes());
        }
    }

    /// Get the edit distance of the current candidate.
    ///
    /// Returns `None` if no current candidate.
    pub fn current_distance(&self) -> Option<usize> {
        self.current_candidate.as_ref().map(|c| c.distance)
    }

    /// Get the term of the current candidate.
    pub fn current_term(&self) -> Option<&str> {
        self.current_candidate.as_ref().map(|c| c.term.as_str())
    }
}

impl<I: Iterator<Item = Candidate>> ZipperIteration for FuzzyZipper<I> {
    fn to_next_val(&mut self) -> bool {
        if self.current_candidate.is_some() {
            self.advance();
            self.current_candidate.is_some()
        } else {
            false
        }
    }

    fn is_val(&self) -> bool {
        self.current_candidate.is_some()
    }

    fn to_next_step(&mut self) -> bool {
        // FuzzyZipper only has values, no intermediate steps
        self.to_next_val()
    }
}

impl<I: Iterator<Item = Candidate>> ZipperAbsolutePath for FuzzyZipper<I> {
    fn path(&self) -> &[u8] {
        &self.path_buffer
    }

    fn origin_path(&self) -> &[u8] {
        &self.path_buffer
    }
}

// Placeholder implementations for other zipper traits
impl<I: Iterator<Item = Candidate>> Zipper for FuzzyZipper<I> {
    type Value = ();

    fn val(&self) -> Option<&Self::Value> {
        if self.current_candidate.is_some() {
            Some(&())
        } else {
            None
        }
    }
}
```

### Step 7: Extend ASource Enum

**File**: `MORK/kernel/src/sources.rs` (modifications)

```rust
// Add to imports
use crate::fuzzy_source::FuzzySource;
use crate::fuzzy_zipper::FuzzyZipper;

// Add variant to ASource enum
pub enum ASource {
    PosSource(BTMSource),
    ACTSource(ACTSource),
    CmpSource(CmpSource),
    FuzzySource(FuzzySource),  // NEW
}

impl Source for ASource {
    fn new(e: Expr) -> Self {
        // Check for FUZZY pattern: (FUZZY max_dist pattern)
        // Arity(3) + SymbolSize(5) + "FUZZY"
        unsafe {
            if *e.ptr == item_byte(Tag::Arity(3)) {
                let second = e.ptr.add(1);
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

// Add variant to AFactor enum
#[derive(PolyZipper)]
pub enum AFactor<'trie, V: Clone + Send + Sync + Unpin + 'static = ()> {
    PosSource(PrefixZipper<'trie, ReadZipperUntracked<'trie, 'trie, V>>),
    ACTSource(PrefixZipper<'trie, ACTMmapZipper<'trie, V>>),
    CmpSource(/* ... */),
    FuzzySource(FuzzyZipper<std::vec::IntoIter<Candidate>>),  // NEW
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

; Transposition (Damerau-Levenshtein, includes adjacent swaps)
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

### Phonetic Matching (Phase C Preview)

> **v0.8.0 Note**: Phonetic matching is available via `ProductAutomatonChar` and `english::*` rule sets.
> See [v0.8.0 liblevenshtein APIs](#v080-liblevenshtein-apis) above.

```metta
; Find names that sound like "Steven" using phonetic rules
!(match &space
    (person (fuzzy-phonetic "Steven" english $name) $attrs)
    ($name $attrs))

; Returns: ("Stephen" ...) ("Stefan" ...) ("Stephan" ...) ("Steve" ...)
; Phonetic rules: "ph" ≈ "f", "v" ≈ "ph", vowel variations
```

The underlying implementation would use:

```rust
use liblevenshtein::phonetic::nfa::ProductAutomatonChar;
use liblevenshtein::phonetic::rules::english;
use liblevenshtein::phonetic::verified::rules_to_nfa_char;

// Use pre-compiled English phonetic rules
let rules = english::zompist();
let nfa = rules_to_nfa_char(&rules.rules);
let product = ProductAutomatonChar::new(nfa, 2);

// ProductAutomatonChar handles phonetic + edit distance
```

---

## Testing

### Unit Tests

**File**: `MORK/kernel/src/fuzzy_source.rs` (add at bottom)

```rust
#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_fuzzy_config_default() {
        let config = FuzzyConfig::default();
        assert_eq!(config.max_distance, 2);
        assert!(config.include_exact);
    }

    #[test]
    fn test_fuzzy_dictionary_view_empty() {
        let map = PathMap::new();
        let config = FuzzyConfig::default();
        let view = FuzzyDictionaryView::new(&map, b"", &config);

        let results: Vec<_> = view.fuzzy_lookup(b"test", 2).collect();
        assert!(results.is_empty());
    }

    #[test]
    fn test_fuzzy_dictionary_view_matches() {
        let mut map = PathMap::new();
        map.set_val_at(b"color", ());
        map.set_val_at(b"colour", ());
        map.set_val_at(b"collar", ());
        map.set_val_at(b"zebra", ());

        let config = FuzzyConfig::default();
        let view = FuzzyDictionaryView::new(&map, b"", &config);

        let results: Vec<_> = view.fuzzy_lookup(b"color", 2).collect();

        assert!(results.iter().any(|c| c.term == "color" && c.distance == 0));
        assert!(results.iter().any(|c| c.term == "colour" && c.distance == 1));
        assert!(results.iter().any(|c| c.term == "collar" && c.distance == 2));
        assert!(!results.iter().any(|c| c.term == "zebra"));
    }

    #[test]
    fn test_fuzzy_zipper_iteration() {
        let candidates = vec![
            Candidate { term: "color".to_string(), distance: 0 },
            Candidate { term: "colour".to_string(), distance: 1 },
        ];

        let mut zipper = FuzzyZipper::new(candidates.into_iter(), b"prefix/");

        assert!(zipper.is_val());
        assert_eq!(zipper.current_term(), Some("color"));
        assert_eq!(zipper.current_distance(), Some(0));

        assert!(zipper.to_next_val());
        assert_eq!(zipper.current_term(), Some("colour"));
        assert_eq!(zipper.current_distance(), Some(1));

        assert!(!zipper.to_next_val());
        assert!(!zipper.is_val());
    }
}
```

### Integration Tests

**File**: `MORK/kernel/tests/fuzzy_integration.rs`

```rust
//! Integration tests for FuzzySource in MORK query pipeline.

use mork_kernel::prelude::*;

#[test]
fn test_fuzzy_query_basic() {
    let mut space = Space::new();

    // Insert test data
    space.insert_sexpr("(word color)");
    space.insert_sexpr("(word colour)");
    space.insert_sexpr("(word collar)");
    space.insert_sexpr("(word zebra)");

    // Query with fuzzy matching
    let results = space.query_sexpr("(word (FUZZY 2 \"color\"))");

    assert!(results.contains("color"));
    assert!(results.contains("colour"));
    assert!(results.contains("collar"));
    assert!(!results.contains("zebra"));
}

#[test]
fn test_fuzzy_query_with_unification() {
    let mut space = Space::new();

    space.insert_sexpr("(person john (age 30))");
    space.insert_sexpr("(person jon (age 25))");
    space.insert_sexpr("(person jane (age 28))");

    let results = space.query_sexpr("(person (FUZZY 1 \"john\") $age)");

    // Should match "john" (distance 0) and "jon" (distance 1)
    assert_eq!(results.len(), 2);
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
- Dictionary size: O(n) iteration to find all candidates
- Max distance: Higher distance = more candidates to check
- Algorithm: Transposition slightly slower than Standard

### Memory Usage

- FuzzyDictionaryView: Temporary dictionary copy
- FuzzyZipper: Minimal (iterator state + path buffer)
- Candidate collection: One string per match

---

## Future Enhancements (Phase B+)

1. **Weighted results**: Return `ScoredCandidate` with phonetic costs
2. **Algorithm selection**: Parse algorithm from expression
3. **Streaming iteration**: Avoid collecting all candidates
4. **Caching**: Cache dictionary views across queries
5. **Phonetic integration**: Combine with phonetic rules for better matching

---

## References

- [MORK Source Trait](../../../MORK/kernel/src/sources.rs)
- [liblevenshtein Transducer](../../src/transducer/mod.rs)
- [PathMap Dictionary Backend](../../src/dictionary/pathmap.rs)
- [PathMap Zipper Traits](../../../PathMap/src/zipper.rs)
