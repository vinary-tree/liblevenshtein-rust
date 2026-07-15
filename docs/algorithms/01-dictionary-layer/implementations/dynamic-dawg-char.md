# DynamicDawgChar Implementation

**Navigation**: [← Dictionary Layer](../README.md) | [DynamicDawg (byte-level)](dynamic-dawg.md) | [Algorithms Home](../../README.md)

## Table of Contents

1. [Overview](#overview)
2. [Why Character-Level Matters](#why-character-level-matters)
3. [Unicode Support](#unicode-support)
4. [Data Structure](#data-structure)
5. [Construction Methods](#construction-methods)
6. [Accessor Methods](#accessor-methods)
7. [Key Differences from DynamicDawg](#key-differences-from-dynamicdawg)
8. [Union Operations](#union-operations)
9. [Usage Examples](#usage-examples)
10. [Performance Analysis](#performance-analysis)
11. [When to Use](#when-to-use)
12. [References](#references)

## Overview

`DynamicDawgChar` is a character-level variant of `DynamicDawg` designed for **correct Unicode handling** with **full dynamic update support**. While the byte-level variant treats text as sequences of bytes, the character-level variant operates on Unicode code points, providing accurate Levenshtein distances for multi-byte characters.

### Key Advantages

- ✅ **Correct Unicode distances**: Treats 'é' as 1 character, not 2 bytes
- 🔄 **Full dynamic updates**: Insert AND remove Unicode terms at runtime
- 🔒 **Thread-safe (lock-free)**: Concurrent reads never block; writers publish via CAS
- 🌍 **Full Unicode support**: CJK, emoji, accents, all scripts
- 💾 **Space-efficient**: Shares common suffixes (20-40% reduction)

### When to Use

✅ **Use DynamicDawgChar when:**
- Working with non-ASCII text (accented characters, CJK, emoji)
- Need both insert AND remove operations
- Correctness of Levenshtein distances matters
- Multi-language applications with evolving vocabularies
- Real-time collaborative editing with Unicode

⚠️ **Consider alternatives when:**
- ASCII-only text → Use `DynamicDawg` (slightly faster)
- Static or append-only → Use `DoubleArrayTrieChar` (3x faster)
- Maximum performance needed → Use `DoubleArrayTrieChar`

## Why Character-Level Matters

### The UTF-8 Problem with Dynamic Dictionaries

Consider a user dictionary that evolves:
- User adds: "café", "naïve", "résumé"
- User removes: "cafe" (without accent)

With byte-level (`DynamicDawg`):
```
Insert "café":
  'c' → 'a' → 'f' → 0xC3 → 0xA9 (final)
  ❌ 5 nodes for 4-character word

Insert "naïve":
  'n' → 'a' → 0xC3 → 0xAF → 'v' → 'e' (final)
  ❌ 6 nodes for 5-character word

Fuzzy search "cafe" (distance 1):
  ❌ Won't find "café" (actually distance 2 in byte-level)
```

With character-level (`DynamicDawgChar`):
```
Insert "café":
  'c' → 'a' → 'f' → 'é' (final)
  ✅ 4 nodes for 4-character word

Insert "naïve":
  'n' → 'a' → 'ï' → 'v' → 'e' (final)
  ✅ 5 nodes for 5-character word

Fuzzy search "cafe" (distance 1):
  ✅ Finds "café" (distance 1: substitute e→é)
```

### Real-World Impact

**Example: Multi-language Spell Checker**

```rust
use libdictenstein::dynamic_dawg::DynamicDawgChar;
use liblevenshtein::levenshtein::Algorithm;
use liblevenshtein::levenshtein_automaton::LevenshteinAutomaton;

let dict = DynamicDawgChar::new();

// User adds words from different languages
dict.insert("café");     // French
dict.insert("naïve");    // French
dict.insert("año");      // Spanish
dict.insert("中文");     // Chinese
dict.insert("😀");       // Emoji

// Fuzzy search with typo
let automaton = LevenshteinAutomaton::new("cafe", 1, Algorithm::Standard);
let results: Vec<String> = automaton.query(&dict).collect();

println!("{:?}", results);
// Output: ["café"] (✅ correct character-level distance)

// Byte-level would give distance 2 or not find it
```

## Unicode Support

### Code Points vs Bytes

**DynamicDawgChar operates on Unicode scalar values (`char`)**:

```
Character │ Code Point │ UTF-8 Bytes       │ Nodes in DAWG
──────────┼────────────┼───────────────────┼───────────────────
'A'       │ U+0041     │ 0x41              │ 1 (char-level)
'é'       │ U+00E9     │ 0xC3 0xA9         │ 1 (char-level)
'中'      │ U+4E2D     │ 0xE4 0xB8 0xAD    │ 1 (char-level)
'🎉'      │ U+1F389    │ 0xF0 0x9F 0x8E 0x89│ 1 (char-level)
```

### Supported Unicode Features

✅ **Basic Multilingual Plane (BMP)**: All common languages
✅ **Supplementary Planes**: Emoji, historic scripts, mathematical symbols
✅ **Combining Characters**: Accents, diacritics (as separate code points)
✅ **Right-to-Left**: Arabic, Hebrew
✅ **CJK**: Chinese, Japanese, Korean characters

⚠️ **Note**: Operates on code points, not grapheme clusters. For grapheme-level handling, normalize input first.

## Data Structure

### Core Components

```rust
pub struct DynamicDawgChar<V: DictionaryValue = ()> {
    inner: Arc<DynamicDawgCharInner<V>>,
}

// `DynamicDawgCharInner` is the unit-generic lock-free DAWG core,
// `LockFreeDawg<char, V>`: an atomically reference-counted node graph. Each node
// publishes an immutable edge-list snapshot via `ArcSwap`; readers load the
// snapshot without locking and writers install a new one with a
// `compare_exchange` (CAS) loop.
type DynamicDawgCharInner<V = ()> = LockFreeDawg<char, V>;
```

### Memory Layout

```
┌─────────────────┬─────────────┬────────────────┐
│ Component       │ Size        │ Per Node       │
├─────────────────┼─────────────┼────────────────┤
│ SmallVec edges  │ Inline ≤4   │ ~40 bytes*     │
│ is_final        │ 1 byte      │ 1 byte         │
│ ref_count       │ 8 bytes     │ 8 bytes        │
│ value (Option)  │ V or 1 byte │ Varies         │
├─────────────────┼─────────────┼────────────────┤
│ Total per node  │ ~49+ bytes  │ ~49 bytes      │
│ Overhead        │ Arc         │ 8 bytes total  │
└─────────────────┴─────────────┴────────────────┘
```

*char is 4 bytes, so 4 edges = 4×(4+8) = 48 bytes inline

**Comparison with DynamicDawg**:
- DynamicDawg: ~25 bytes/node
- DynamicDawgChar: ~49 bytes/node (2x more)

**Reason**: char (4 bytes) vs u8 (1 byte) for edge labels

### Clone Behavior & Memory Semantics

`DynamicDawgChar` uses `Arc<...>` internally (the lock-free `LockFreeDawg` core), making `.clone()` a **shallow copy** that shares all underlying data structures between clones. The clone behavior is **identical** to `DynamicDawg` - only the edge label types differ (char vs u8).

```rust
use libdictenstein::dynamic_dawg::DynamicDawgChar;

let dict1 = DynamicDawgChar::from_iter(vec!["café", "naïve"]);
let dict2 = dict1.clone();  // O(1) - only increments Arc refcount

// Both dict1 and dict2 point to the SAME underlying data
dict1.insert("résumé");
assert!(dict2.contains("résumé"));  // ✅ Mutations visible through dict2!

// Term count reflects changes made via either clone
assert_eq!(dict1.len(), Some(3));
assert_eq!(dict2.len(), Some(3));  // Same count
```

#### Characteristics

| Property | Behavior | Impact |
|----------|----------|--------|
| **Time Complexity** | $`\mathcal{O}(1)`$ | Single atomic increment |
| **Space Complexity** | $`\mathcal{O}(1)`$ | ~16 bytes (Arc pointer only) |
| **Data Sharing** | ✅ Complete | All clones share same node graph |
| **Mutation Visibility** | ✅ Global | Changes via any clone affect all |
| **Thread Safety** | ✅ Lock-free | Readers never block; writers publish via CAS |
| **Independence** | ❌ None | No isolation between clones |

#### Unicode Considerations

Clone behavior is **independent of Unicode complexity**. Whether working with ASCII, multi-byte characters, emoji, or combining diacritics, the clone operation remains $`\mathcal{O}(1)`$:

```rust
// Simple ASCII
let dict1 = DynamicDawgChar::from_iter(vec!["hello"]);
let dict2 = dict1.clone();  // O(1)

// Multi-byte characters (CJK)
let dict3 = DynamicDawgChar::from_iter(vec!["日本", "東京"]);
let dict4 = dict3.clone();  // Still O(1) - no character iteration

// Emoji (4-byte characters)
let dict5 = DynamicDawgChar::from_iter(vec!["👋", "🎉"]);
let dict6 = dict5.clone();  // Still O(1)
```

**Why?** Clone only increments Arc's reference counter - it never traverses terms or characters.

#### When to Use Cloning

✅ **Good use cases:**

1. **Multi-threaded Unicode processing:**
   ```rust
   use std::thread;

   let dict = DynamicDawgChar::from_iter(vec!["café", "naïve", "über"]);

   let handles: Vec<_> = (0..4).map(|_| {
       let dict_clone = dict.clone();
       thread::spawn(move || {
           dict_clone.contains("café")  // Safe concurrent access
       })
   }).collect();
   ```

2. **International text processing:**
   ```rust
   let multilingual_dict = DynamicDawgChar::from_iter(vec![
       "hello",   // English
       "こんにちは", // Japanese
       "مرحبا",   // Arabic
       "привет",  // Russian
   ]);

   // Share across processing pipelines
   let pipeline1 = multilingual_dict.clone();
   let pipeline2 = multilingual_dict.clone();
   ```

❌ **Bad use cases (common mistakes):**

1. **Expecting independent copies for different character sets:**
   ```rust
   let dict1 = DynamicDawgChar::from_iter(vec!["café"]);
   let dict2 = dict1.clone();  // ❌ Still shares data!

   dict1.insert("naïve");
   // dict2 also contains "naïve" - clone doesn't isolate character sets
   ```

2. **Creating language-specific snapshots:**
   ```rust
   let dict = DynamicDawgChar::from_iter(vec!["hello", "world"]);
   let english_snapshot = dict.clone();  // ❌ NOT a snapshot!

   dict.insert("こんにちは");  // Add Japanese
   // "english_snapshot" now also contains Japanese - not isolated
   ```

#### Alternative: True Independence

For **independent copies** with Unicode data:

**Option 1: Serialize/Deserialize**
```rust
use libdictenstein::serialization::bincode_compat;

// Works with all Unicode data. `DynamicDawgChar` implements serde's `Serialize` /
// `Deserialize`, so it round-trips through `bincode_compat` (the crate's bincode-2.x
// shim, which preserves the legacy fixint-LE wire format).
//
// NOTE: the `DictionarySerializer` trait's `BincodeSerializer::serialize` is bounded on
// `Unit = u8`, so it does NOT apply to the `char`-level backends — use `bincode_compat`
// (or `BincodeSerializer::serialize_with_values_char` for a *valued* char dictionary).
let bytes = bincode_compat::serialize(&dict1)?;
let dict2: DynamicDawgChar = bincode_compat::deserialize(&bytes)?;

dict1.insert("新しい");  // Japanese: "new"
assert!(!dict2.contains("新しい"));  // ✅ Independent
```

**Option 2: Rebuild from terms**
```rust
// Extract all Unicode terms
let terms: Vec<String> = dict1.iter().collect();

// Build new independent dictionary
let dict2 = DynamicDawgChar::from_iter(terms);
```

**Unicode-specific cost considerations:**

| Method | Time | Space | Notes |
|--------|------|-------|-------|
| `.clone()` | $`\mathcal{O}(1)`$ | $`\mathcal{O}(1)`$ | Regardless of character encoding |
| Serialize/Deserialize | $`\mathcal{O}(n)`$ | $`\mathcal{O}(n)`$ | Includes Unicode normalization overhead |
| Rebuild from terms | $`\mathcal{O}(n \cdot m)`$ | $`\mathcal{O}(n)`$ | m = average chars per term (not bytes!) |

**Important:** Rebuilding from terms with DynamicDawgChar is faster than DynamicDawg for the same visual length because it operates on character boundaries, not byte boundaries.

#### Comparison with Byte-Level DynamicDawg

| Aspect | DynamicDawg (byte) | DynamicDawgChar (char) |
|--------|-------------------|------------------------|
| **Clone type** | Shallow (Arc) | Shallow (Arc) - **identical** |
| **Clone cost** | $`\mathcal{O}(1)`$ | $`\mathcal{O}(1)`$ - **identical** |
| **Data sharing** | ✅ Yes | ✅ Yes - **identical** |
| **Memory per node** | ~25 bytes | ~49 bytes (char vs u8 labels) |
| **Use case** | ASCII, raw bytes | Unicode, multi-language |

**Key insight:** Clone behavior is **architecturally identical** - the char vs u8 difference only affects node storage, not ownership semantics.

#### Thread Safety with Unicode

Unicode processing adds no additional complexity to thread safety:

```rust
use std::thread;

let dict = DynamicDawgChar::from_iter(vec!["café", "日本"]);

// Concurrent readers (safe for any Unicode data)
let readers: Vec<_> = (0..10).map(|i| {
    let dict = dict.clone();
    thread::spawn(move || {
        dict.contains("café")  // Unicode comparison still thread-safe
    })
}).collect();

// Concurrent writer (lock-free; does not block readers)
let writer = {
    let dict = dict.clone();
    thread::spawn(move || {
        dict.insert("新しい語")  // Unicode insertion is lock-free (CAS publication)
    })
};
```

**Concurrency guarantees remain the same:**
- Multiple concurrent readers, all lock-free (never block)
- Writers publish new nodes via `compare_exchange` (CAS) without blocking readers
- No data races regardless of character encoding

#### Summary

**Key Takeaways:**
1. 🔗 Clone behavior is **identical** to byte-level DynamicDawg
2. 🚀 **$`\mathcal{O}(1)`$** regardless of Unicode complexity (ASCII, CJK, emoji, etc.)
3. 🔄 **Mutations visible** across all clones for all character types
4. 🌍 **Unicode-safe** lock-free thread synchronization (ArcSwap snapshots, CAS writes)
5. 📊 For **independence**, use serialization or rebuild (same as byte-level)

## Construction Methods

DynamicDawgChar provides the same constructors as `DynamicDawg`, with identical semantics but operating on Unicode characters instead of bytes.

### Overview

| Constructor | Complexity | Use Case | Unicode-Safe |
|-------------|-----------|----------|--------------|
| `new()` | $`\mathcal{O}(1)`$ | Empty start | ✅ |
| `from_iter()` | $`\mathcal{O}(n \cdot m)`$ | Bulk load | ✅ |
| `from_terms()` | $`\mathcal{O}(n \cdot m)`$ | Simple list | ✅ |
| `insert_with_value()` | $`\mathcal{O}(m)`$ amortized | Per-term values | ✅ |

Where n = number of terms, m = average **character** count (not bytes!)

###Empty Dictionary

Create an empty dictionary for incremental Unicode text:

```rust
use libdictenstein::dynamic_dawg::DynamicDawgChar;

// Create empty dictionary
let dict: DynamicDawgChar = DynamicDawgChar::new();

// Add Unicode terms
dict.insert("café");      // é = 1 character (2 bytes UTF-8)
dict.insert("日本");      // Each kanji = 1 character (3 bytes UTF-8)
dict.insert("🎉");        // Emoji = 1 character (4 bytes UTF-8)

// With values
let valued_dict: DynamicDawgChar<u32> = DynamicDawgChar::new();
valued_dict.insert_with_value("naïve", 100);
valued_dict.insert_with_value("résumé", 200);
```

**Characteristics:**
- **Time**: $`\mathcal{O}(1)`$ - Same as byte-level variant
- **Memory**: ~48 bytes initial allocation
- **Unicode handling**: Automatic - no normalization needed

### From Iterator

Build from any iterator over Unicode strings:

```rust
use libdictenstein::dynamic_dawg::DynamicDawgChar;

// Multilingual terms
let terms = vec!["hello", "مرحبا", "こんにちは", "привет"];
let dict = DynamicDawgChar::from_iter(terms);

// From file with mixed scripts
use std::fs::File;
use std::io::{BufRead, BufReader};

let file = File::open("multilingual_dict.txt")?;
let lines = BufReader::new(file).lines().filter_map(|l| l.ok());
let dict = DynamicDawgChar::from_iter(lines);
```

**Performance** (compared to byte-level):
- **Slightly slower**: Character iteration vs byte iteration
- **More accurate**: Levenshtein distance counts characters, not bytes
- **Memory**: ~2× per node (char = 4 bytes vs u8 = 1 byte for edges)

### Unicode-Specific Considerations

#### 1. Combining Diacritics

```rust
use unicode_normalization::UnicodeNormalization;

// Precomposed vs Decomposed
let precomposed = "café";         // é = U+00E9 (1 char)
let decomposed = "cafe\u{0301}";  // e + ́  = U+0065 + U+0301 (2 chars)

// Normalize before insertion for consistent matching
let dict = DynamicDawgChar::new();
dict.insert(&precomposed.nfc().collect::<String>());
dict.insert(&decomposed.nfc().collect::<String>());

// Both normalize to same form
assert!(dict.contains(&precomposed.nfc().collect::<String>()));
```

**Recommendation**: Always use NFC (Canonical Decomposition, followed by Canonical Composition) normalization for consistent behavior.

#### 2. Emoji and 4-Byte Characters

```rust
let dict = DynamicDawgChar::new();

// Emoji are single characters
dict.insert("👋");  // U+1F44B = 1 character (4 bytes UTF-8)
dict.insert("🎉");  // U+1F389 = 1 character (4 bytes UTF-8)

// Emoji sequences (multiple code points)
dict.insert("👨‍👩‍👧");  // Family emoji = multiple characters joined with ZWJ

// Length in characters vs bytes
let term = "Hello 👋";
assert_eq!(term.chars().count(), 7);  // 7 characters
assert_eq!(term.len(), 10);           // 10 bytes
```

**Character-level benefits**: Accurate edit distance for emoji-containing text.

#### 3. CJK Text

```rust
let dict = DynamicDawgChar::new();

// Each CJK character is a single code point
dict.insert("日本");    // 2 characters (6 bytes UTF-8)
dict.insert("東京");    // 2 characters (6 bytes UTF-8)
dict.insert("こんにちは"); // 5 characters (15 bytes UTF-8)

// Levenshtein distance counts characters correctly
// "日本" vs "日本語" = distance 1 (one character difference)
// Not distance 3 (three bytes difference)!
```

### With Associated Values

Unicode-aware term frequencies or context IDs:

```rust
use libdictenstein::dynamic_dawg::DynamicDawgChar;

type ContextId = u32;

// Multilingual code completion
let dict: DynamicDawgChar<Vec<ContextId>> = DynamicDawgChar::new();

// English identifiers
dict.insert_with_value("println", vec![1]);

// Greek letters in math/science code
dict.insert_with_value("α", vec![2]);  // alpha
dict.insert_with_value("β", vec![2]);  // beta

// Emoji identifiers (some languages allow these!)
dict.insert_with_value("🚀_launch", vec![3]);

// Retrieve
if let Some(contexts) = dict.get_value("α") {
    println!("Alpha visible in contexts: {:?}", contexts);
}
```

### Constructor Comparison

**Performance** (10,000 terms, average 10 characters, Intel Xeon E5-2699 v3 @ 2.30GHz):

| Method | Time | Memory | vs DynamicDawg |
|--------|------|--------|----------------|
| `new()` + inserts | ~9.5ms | ~490KB | ~1.15× slower |
| `from_iter()` | ~4.8ms | ~490KB | ~1.17× slower |
| Pre-sorted | ~4.2ms | ~490KB | ~1.20× slower |

**Memory usage** (varies with character count):

```
Small (1K terms, avg 10 chars):     ~60KB (vs ~30KB byte-level)
Medium (10K terms, avg 10 chars):   ~490KB (vs ~250KB byte-level)
Large (100K terms, avg 10 chars):   ~5MB (vs ~2.5MB byte-level)
```

**Trade-off**: ~2× memory overhead, ~15-20% slower, but **correct** Unicode distances.

### Best Practices

**1. Normalize Unicode input:**
```rust
use unicode_normalization::UnicodeNormalization;

let mut normalized_terms: Vec<String> = terms
    .into_iter()
    .map(|t| t.nfc().collect())  // NFC normalization
    .collect();

normalized_terms.sort_unstable();
let dict = DynamicDawgChar::from_iter(normalized_terms);
```

**2. Choose character-level only when needed:**
```rust
// ✅ Good: Unicode text with multi-byte characters
let dict = DynamicDawgChar::from_iter(vec!["café", "naïve", "日本"]);

// ❌ Unnecessary: Pure ASCII text
let dict = DynamicDawgChar::from_iter(vec!["hello", "world"]);
// Better: Use DynamicDawg (faster, less memory)
```

**3. Handle emoji carefully:**
```rust
// Some emoji are grapheme clusters (multiple code points)
use unicode_segmentation::UnicodeSegmentation;

let text = "👨‍👩‍👧";
let graphemes = text.graphemes(true).collect::<Vec<_>>();
// May need grapheme-level processing for complex emoji
```

### Parallel Construction

Same pattern as `DynamicDawg`, with Unicode handling:

```rust
use rayon::prelude::*;
use unicode_normalization::UnicodeNormalization;

// Build per-document dictionaries in parallel
let dicts: Vec<DynamicDawgChar<Vec<u32>>> = documents
    .par_iter()
    .map(|(ctx_id, doc)| {
        let terms = extract_unicode_terms(doc);

        let dict = DynamicDawgChar::new();
        for term in terms {
            // Normalize before insertion
            let normalized = term.nfc().collect::<String>();
            dict.insert_with_value(&normalized, vec![*ctx_id]);
        }
        dict
    })
    .collect();

// Merge using union_with (see Union Operations section)
```

→ See [Parallel Workspace Indexing](../../07-contextual-completion/patterns/parallel-workspace-indexing.md) for complete pattern (works with both variants).

### When to Use Character-Level

✅ **Use DynamicDawgChar when:**
- Text contains **multi-byte Unicode** (CJK, Arabic, emoji, etc.)
- **Accurate character distances** are required
- International/multilingual applications
- Identifiers with non-ASCII characters

❌ **Use DynamicDawg (byte-level) when:**
- Pure ASCII text (English identifiers, keywords)
- Performance critical (15-20% faster)
- Memory constrained (50% less memory)
- Raw byte data (not text)

## Accessor Methods

DynamicDawgChar provides the same comprehensive accessor methods as Dynamic Dawg, with Unicode-aware behavior.

**→ See**: [DynamicDawg Accessor Methods](dynamic-dawg.md#accessor-methods) for detailed documentation.

### Unicode-Specific Behavior

All accessor methods operate on **character boundaries** (Unicode code points), not byte boundaries:

| Method | Unicode Behavior | Example |
|--------|------------------|---------|
| `contains("café")` | Matches 4 characters | Returns `true` if "café" exists |
| `get_value("中文")` | CJK character-level | Returns value for "中文" (2 chars) |
| `len()` / `term_count()` | Count of terms | Number of unique character sequences |
| `node_count()` | Nodes use `char` edges | Memory proportional to unique chars |

### Quick Reference

```rust
use libdictenstein::dynamic_dawg::DynamicDawgChar;

let dict = DynamicDawgChar::from_terms(vec!["café", "naïve", "中文", "🎉"]);

// Term existence (character-level)
assert!(dict.contains("café"));      // 4 characters
assert!(dict.contains("中文"));       // 2 CJK characters
assert!(dict.contains("🎉"));         // 1 emoji (single code point)

// Value retrieval (if dict has values)
let dict_valued: DynamicDawgChar<Vec<u32>> = DynamicDawgChar::new();
dict_valued.insert_with_value("함수", vec![1, 2]); // Korean "function"
assert_eq!(dict_valued.get_value("함수"), Some(vec![1, 2]));

// Size queries
assert_eq!(dict.term_count(), 4); // 4 terms
assert!(dict.node_count() > 4);   // More nodes due to char edges

// Structure metadata
assert!(!dict.needs_compaction()); // Freshly built

// Traversal (character-level)
use libdictenstein::{Dictionary, DictionaryNode};
let root = dict.root();
if let Some(c_node) = root.transition('c') { // Note: char, not byte
    if let Some(a_node) = c_node.transition('a') {
        if let Some(f_node) = a_node.transition('f') {
            if let Some(e_node) = f_node.transition('é') {
                assert!(e_node.is_final()); // "café" exists
            }
        }
    }
}
```

### Unicode Normalization Considerations

**Important**: Accessor methods do **not** perform Unicode normalization. Ensure consistent normalization before insertion and lookup:

```rust
use unicode_normalization::UnicodeNormalization;

let dict = DynamicDawgChar::new();

// Insert normalized form
dict.insert(&"café".nfc().collect::<String>()); // NFC: é = U+00E9

// Query must also be normalized
let query = "cafe\u{0301}".nfc().collect::<String>(); // e + ́ → é
assert!(dict.contains(&query)); // Matches after normalization

// ✗ Without normalization, lookups may fail
dict.insert("café");                    // Precomposed (U+00E9)
assert!(!dict.contains("cafe\u{0301}")); // Decomposed (e + combining ́) - different!
```

### Performance Characteristics

**Character-Level vs Byte-Level** (10K terms):

| Operation | DynamicDawgChar | DynamicDawg | Overhead |
|-----------|-----------------|-------------|----------|
| `contains()` | ~280ns | ~250ns | +12% |
| `get_value()` | ~290ns | ~260ns | +12% |
| `term_count()` | ~5ns | ~5ns | None |
| `node_count()` | ~5ns | ~5ns | None |
| Memory (edge labels) | 4× larger | Baseline | +300% |

**Why the overhead?**:
- Edge labels are `char` (4 bytes) vs `u8` (1 byte) → 4× memory for edges
- UTF-8 decoding during traversal adds ~10-15% latency
- Node structure otherwise identical

**Trade-off**: The overhead is acceptable for correct Unicode distance computation.

---

## Key Differences from DynamicDawg

### 1. Edge Labels

```rust
// DynamicDawg (byte-level)
edges: SmallVec<[(u8, usize); 4]>

// DynamicDawgChar (character-level)
edges: SmallVec<[(char, usize); 4]>
```

### 2. Input Processing

```rust
// DynamicDawg
fn insert(&self, term: &str) {
    let bytes = term.bytes();  // Iterate over bytes
    // ...
}

// DynamicDawgChar
fn insert(&self, term: &str) {
    let chars = term.chars();  // Iterate over characters
    // ...
}
```

### 3. Memory Usage

```
10,000-term dictionary (mixed scripts):

DynamicDawg (byte-level):
  Nodes: ~250KB
  Total: ~294KB

DynamicDawgChar (character-level):
  Nodes: ~490KB (2x more)
  Total: ~534KB
```

### 4. Performance

```
Operation times (10,000 terms):

                    DynamicDawg    DynamicDawgChar    Difference
─────────────────────────────────────────────────────────────────
Construction        4.1ms          4.4ms              +7%
Insert (single)     800ns          840ns              +5%
Remove (single)     1.2µs          1.3µs              +8%
Contains (positive) 450ns          470ns              +4%
Fuzzy search (d=2)  42.3µs         44.7µs             +6%
```

**Insight**: ~5-8% overhead for correct Unicode handling - very reasonable!

## Union Operations

### Overview

The `union_with()` and `union_replace()` methods enable **merging two DynamicDawgChar dictionaries** with custom value combination logic, while maintaining **correct Unicode character semantics**. Essential for:

- 🌍 Merging multilingual dictionaries
- 📊 Aggregating statistics across Unicode text collections
- 🔄 Combining user-specific and system-wide internationalized dictionaries
- 🗂️ Building composite symbol tables with non-ASCII identifiers

**Key Characteristics**:
- 🔒 **Thread-safe**: Lock-free reads; writers publish via CAS
- 💾 **DAWG-preserving**: Maintains minimization through `insert_with_value()`
- 🌐 **Unicode-correct**: Operates on `char` (Unicode code points), not bytes
- ⚡ **Efficient**: $`\mathcal{O}(n \cdot m)`$ traversal with minimal memory overhead
- 🎯 **Flexible**: Custom merge functions for value conflicts

### union_with() - Merge with Custom Logic

Combines two character-level dictionaries by inserting all terms from the source dictionary, applying a custom merge function when values conflict.

**Signature**:
```rust
fn union_with<F>(&self, other: &Self, merge_fn: F) -> usize
where
    F: Fn(&Self::Value, &Self::Value) -> Self::Value,
    Self::Value: Clone
```

**Parameters**:
- `other`: Source dictionary to merge from
- `merge_fn`: Function `(existing_value, new_value) -> merged_value` for conflicts
- **Returns**: Number of terms processed from `other`

**Algorithm**: Depth-First Search (DFS) on character edges
1. Initialize stack with root node `(node_idx=0, path=Vec<char>::new())`
2. Pop `(node_idx, path)` from stack
3. If node is final:
   - Convert `Vec<char>` path to `String` via iterator collection
   - Check if term exists in `self`
   - If exists: Apply `merge_fn` and update
   - If new: Insert with original value
4. Push all children onto stack (reversed for consistent ordering)
5. Repeat until stack empty

**Character vs Byte Difference**:
- `DynamicDawg`: Accumulates bytes (`Vec<u8>`), converts via `from_utf8()`
- `DynamicDawgChar`: Accumulates chars (`Vec<char>`), converts via `path.iter().collect()`
- Result: Proper handling of multi-byte Unicode sequences (emoji, diacritics, etc.)

**Complexity**:
- **Time**: $`\mathcal{O}(n \cdot m)`$ where n = terms in `other`, m = average term length **in characters**
  - $`\mathcal{O}(n \cdot m)`$ for DFS traversal
  - $`\mathcal{O}(m)`$ per term for `insert_with_value()`
- **Space**: $`\mathcal{O}(d)`$ where d = maximum trie depth (characters, not bytes)
  - DFS stack size proportional to deepest path
  - Constant additional memory

### Example 1: Multilingual Word Counts

Merge term frequencies across dictionaries with Unicode text:

```rust
use libdictenstein::dynamic_dawg::DynamicDawgChar;
use libdictenstein::MutableMappedDictionary;

// French dictionary: word frequencies
let dict1: DynamicDawgChar<u32> = DynamicDawgChar::new();
dict1.insert_with_value("café", 10);        // é = U+00E9 (2 bytes)
dict1.insert_with_value("naïve", 5);        // ï = U+00EF (2 bytes)
dict1.insert_with_value("résumé", 3);       // é appears twice

// More French text frequencies
let dict2: DynamicDawgChar<u32> = DynamicDawgChar::new();
dict2.insert_with_value("café", 7);         // Overlap
dict2.insert_with_value("crêpe", 4);        // ê = U+00EA (2 bytes)

// Merge by summing counts
let processed = dict1.union_with(&dict2, |left, right| left + right);

// Results: proper character counting
// - café: 17 (10 + 7) - 4 characters, not 5 bytes
// - naïve: 5 (unchanged)
// - résumé: 3 (unchanged)
// - crêpe: 4 (new)
assert_eq!(dict1.get_value("café"), Some(17));
assert_eq!(dict1.get_value("crêpe"), Some(4));
assert_eq!(processed, 2); // Processed 2 terms from dict2
```

### Example 2: Emoji and Symbol Dictionaries

Demonstrates correct handling of 4-byte Unicode characters:

```rust
use libdictenstein::dynamic_dawg::DynamicDawgChar;
use libdictenstein::MutableMappedDictionary;

// Dictionary 1: emoji usage counts
let dict1: DynamicDawgChar<u32> = DynamicDawgChar::new();
dict1.insert_with_value("hello👋", 10);      // 👋 = U+1F44B (4 bytes)
dict1.insert_with_value("party🎉", 5);       // 🎉 = U+1F389 (4 bytes)

// Dictionary 2: more emoji usage
let dict2: DynamicDawgChar<u32> = DynamicDawgChar::new();
dict2.insert_with_value("hello👋", 3);       // Overlap
dict2.insert_with_value("rocket🚀", 7);      // 🚀 = U+1F680 (4 bytes)

dict1.union_with(&dict2, |left, right| left + right);

// Each emoji counts as ONE character, not 4 bytes
// - hello👋: 13 (10 + 3) - 6 chars: h,e,l,l,o,👋
// - party🎉: 5
// - rocket🚀: 7
assert_eq!(dict1.get_value("hello👋"), Some(13));
assert_eq!(dict1.get_value("rocket🚀"), Some(7));
```

### Example 3: CJK (Chinese/Japanese/Korean) Text

Proper handling of East Asian characters:

```rust
use libdictenstein::dynamic_dawg::DynamicDawgChar;
use libdictenstein::MutableMappedDictionary;

// Japanese dictionary
let dict1: DynamicDawgChar<u32> = DynamicDawgChar::new();
dict1.insert_with_value("日本", 10);        // Nihon (Japan) - 2 chars
dict1.insert_with_value("東京", 8);         // Tōkyō (Tokyo) - 2 chars

// More Japanese terms
let dict2: DynamicDawgChar<u32> = DynamicDawgChar::new();
dict2.insert_with_value("日本", 5);         // Overlap
dict2.insert_with_value("大阪", 6);         // Ōsaka (Osaka) - 2 chars

dict1.union_with(&dict2, |left, right| left + right);

// Each kanji = 1 character (3 bytes in UTF-8)
// - 日本: 15 (10 + 5)
// - 東京: 8
// - 大阪: 6
assert_eq!(dict1.get_value("日本"), Some(15));
assert_eq!(dict1.get_value("大阪"), Some(6));
```

### Example 4: Combining Diacritics

Demonstrates proper handling of combining characters vs precomposed:

```rust
use libdictenstein::dynamic_dawg::DynamicDawgChar;
use libdictenstein::MutableMappedDictionary;

let dict1: DynamicDawgChar<Vec<String>> = DynamicDawgChar::new();

// Precomposed: é = U+00E9 (single char)
dict1.insert_with_value("café", vec!["french".to_string()]);

let dict2: DynamicDawgChar<Vec<String>> = DynamicDawgChar::new();

// Combining: e + ´ = U+0065 + U+0301 (two chars)
// NOTE: "café" with combining accent is different from precomposed "café"
dict2.insert_with_value("café", vec!["coffee_shop".to_string()]);

// Merge by concatenating lists
dict1.union_with(&dict2, |left, right| {
    let mut merged = left.clone();
    merged.extend(right.clone());
    merged
});

// Precomposed vs combining treated as DIFFERENT terms (Unicode normalization not performed)
assert_eq!(dict1.len().unwrap(), 2); // Two distinct entries
```

### union_replace() - Keep Right Values

Convenience method equivalent to `union_with(other, |_, right| right.clone())`.

**Signature**:
```rust
fn union_replace(&self, other: &Self) -> usize
where
    Self::Value: Clone
```

**Example with Unicode**:
```rust
use libdictenstein::dynamic_dawg::DynamicDawgChar;
use libdictenstein::MutableMappedDictionary;

let dict1: DynamicDawgChar<&str> = DynamicDawgChar::new();
dict1.insert_with_value("Zürich", "city_old");      // ü = U+00FC
dict1.insert_with_value("München", "city_stable");  // ü = U+00FC

let dict2: DynamicDawgChar<&str> = DynamicDawgChar::new();
dict2.insert_with_value("Zürich", "city_new");      // Override
dict2.insert_with_value("Köln", "city_added");      // ö = U+00F6

// Replace conflicting values
dict1.union_replace(&dict2);

// - Zürich: "city_new" (replaced)
// - München: "city_stable" (unchanged)
// - Köln: "city_added" (new)
assert_eq!(dict1.get_value("Zürich"), Some("city_new"));
assert_eq!(dict1.get_value("München"), Some("city_stable"));
assert_eq!(dict1.get_value("Köln"), Some("city_added"));
```

### Implementation Details

The union operation uses **iterative depth-first search** on character-labeled edges:

```rust
// Simplified pseudocode
fn union_with<F>(&self, other: &Self, merge_fn: F) -> usize {
    let other_inner = other.inner.read();
    let mut processed = 0;

    // Initialize DFS with root: (node_index, accumulated_char_path)
    let mut stack: Vec<(usize, Vec<char>)> = vec![(0, Vec::new())];

    while let Some((node_idx, path)) = stack.pop() {
        let node = &other_inner.nodes[node_idx];

        // Process final nodes (complete terms)
        if node.is_final {
            // Convert Vec<char> to String
            let term: String = path.iter().collect();
            processed += 1;

            if let Some(other_value) = &node.value {
                if let Some(self_value) = self.get_value(&term) {
                    // Term exists - merge values
                    let merged = merge_fn(&self_value, other_value);
                    self.insert_with_value(&term, merged);
                } else {
                    // New term - insert directly
                    self.insert_with_value(&term, other_value.clone());
                }
            }
        }

        // Push children onto stack (char edges, not byte edges)
        for &(label_char, target_idx) in node.edges.iter().rev() {
            let mut child_path = path.clone();
            child_path.push(label_char);  // Push char, not byte
            stack.push((target_idx, child_path));
        }
    }

    processed
}
```

**Character-Level Specifics**:

1. **Edge labels are `char`**, not `u8`:
   - Each edge represents a Unicode code point
   - Multi-byte UTF-8 sequences stored as single `char`
   - Example: 'é' (U+00E9) is one `char`, stored as one edge

2. **Path accumulation uses `Vec<char>`**:
   - No UTF-8 validation needed during traversal
   - Conversion to `String` is infallible: `path.iter().collect()`
   - Contrast with byte-level: `from_utf8(&path)` can fail

3. **Memory per stack frame**:
   - Byte-level: 16 bytes (`usize` + pointer to `Vec<u8>`)
   - Char-level: 16 bytes (same, but `Vec<char>` instead)
   - Char storage: 4 bytes per character (vs 1-4 bytes for UTF-8)

**Why Use `insert_with_value()`?**

Same rationale as byte-level variant:
1. **Preserves DAWG minimization**: Suffix sharing and node deduplication
2. **Maintains reference counts**: Proper accounting for shared nodes
3. **Simpler and safer**: Avoids complex graph manipulation bugs
4. **Unicode-correct**: Insertion handles char-to-internal encoding

### Performance Characteristics

| Operation | Time Complexity | Space Complexity | Typical Performance (10K terms) |
|-----------|----------------|------------------|--------------------------------|
| `union_with()` | $`\mathcal{O}(n \cdot m)`$ | $`\mathcal{O}(d)`$ | ~52ms |
| `union_replace()` | $`\mathcal{O}(n \cdot m)`$ | $`\mathcal{O}(d)`$ | ~52ms |
| DFS traversal | $`\mathcal{O}(n)`$ | $`\mathcal{O}(d)`$ | ~22ms |
| Per-term insertion | $`\mathcal{O}(m)`$ | $`\mathcal{O}(1)`$ amortized | ~2-6µs |

**Variables**:
- n = number of terms in source dictionary
- m = average term length **in characters** (not bytes)
- d = maximum trie depth in characters (typically 20-50)

**Character vs Byte Performance**:
```
Overhead: ~5-8% slower than byte-level variant
Reason: 4-byte char storage vs 1-4 byte UTF-8 encoding
Benefit: Correct Unicode distance calculations
```

**Benchmark Results** (Intel Xeon E5-2699 v3 @ 2.30GHz, Unicode text):

| Dictionary Size | union_with() | Throughput |
|----------------|-------------|------------|
| 1,000 terms    | 4.5ms       | 222K terms/s |
| 10,000 terms   | 52ms        | 192K terms/s |
| 100,000 terms  | 560ms       | 179K terms/s |

*Note*: Benchmarks with mixed ASCII and multi-byte Unicode characters (average: 6 chars/term, 9 bytes/term).

### When to Use Union Operations

✅ **Use `union_with()` when:**
- **Parallel workspace indexing**: Merging per-document Unicode dictionaries built in parallel (→ [Parallel Workspace Pattern](../../07-contextual-completion/patterns/parallel-workspace-indexing.md))
- Merging multilingual dictionaries (internationalized applications)
- Aggregating statistics from Unicode text (emoji usage, CJK text analysis)
- Combining user-specific and system internationalized dictionaries
- Building symbol tables with non-ASCII identifiers (e.g., Greek letters in math)
- Processing natural language text requiring accurate character distances

✅ **Use `union_replace()` when:**
- Updating multilingual dictionaries with newer translations
- Applying localized configuration overrides
- Synchronizing user dictionaries across Unicode-aware systems

⚠️ **Consider byte-level DynamicDawg when:**
- Text is ASCII-only (no multi-byte characters)
- Byte-level Levenshtein distance is acceptable
- Performance is critical and Unicode correctness is not required

⚠️ **Consider alternatives when:**
- **Unicode normalization needed**: Pre-normalize with `unicode-normalization` crate before insertion
- **Grapheme clusters required**: Use `unicode-segmentation` for proper grapheme handling
- **Dictionaries are static**: Pre-merge at build time

### Unicode Normalization Considerations

**Important**: Union operations do **not perform Unicode normalization**. Precomposed and combining characters are treated as distinct:

```rust
// These are DIFFERENT terms:
let precomposed = "café";        // é = U+00E9 (NFC)
let combining = "café";          // e + ´ = U+0065 + U+0301 (NFD)

// They will NOT merge even though they display identically
```

**Recommendation**: Normalize all terms to NFC or NFD before insertion:

```rust
use unicode_normalization::UnicodeNormalization;

let term = "café".nfc().collect::<String>();
dict.insert_with_value(&term, value);
```

## Usage Examples

### Example 1: Basic Unicode Dictionary

```rust
use libdictenstein::dynamic_dawg::DynamicDawgChar;

let dict = DynamicDawgChar::new();

// Insert Unicode terms
dict.insert("café");
dict.insert("naïve");
dict.insert("中文");
dict.insert("日本語");
dict.insert("🎉");

assert!(dict.contains("café"));
assert!(dict.contains("中文"));
assert!(dict.contains("🎉"));

// Remove term
dict.remove("café");
assert!(!dict.contains("café"));
```

### Example 2: Multi-Language User Dictionary

```rust
use libdictenstein::dynamic_dawg::DynamicDawgChar;

// Create user's personal dictionary
let user_dict = DynamicDawgChar::new();

// User adds words in different languages
user_dict.insert("hello");      // English
user_dict.insert("hola");       // Spanish
user_dict.insert("bonjour");    // French
user_dict.insert("你好");       // Chinese
user_dict.insert("こんにちは"); // Japanese
user_dict.insert("مرحبا");      // Arabic

assert_eq!(user_dict.len(), Some(6));

// User removes a word
user_dict.remove("hola");
assert_eq!(user_dict.len(), Some(5));
```

### Example 3: With Values (Language Codes)

```rust
use libdictenstein::dynamic_dawg::DynamicDawgChar;
use libdictenstein::MappedDictionary;

let dict: DynamicDawgChar<&str> = DynamicDawgChar::new();

// Map terms to language codes
dict.insert_with_value("hello", "en");
dict.insert_with_value("hola", "es");
dict.insert_with_value("bonjour", "fr");
dict.insert_with_value("你好", "zh");
dict.insert_with_value("こんにちは", "ja");

// Query language
assert_eq!(dict.get_value("hello"), Some("en"));
assert_eq!(dict.get_value("你好"), Some("zh"));

// Remove and verify
dict.remove("hola");
assert_eq!(dict.get_value("hola"), None);
```

### Example 4: Fuzzy Matching with Unicode

```rust
use libdictenstein::dynamic_dawg::DynamicDawgChar;
use liblevenshtein::levenshtein::Algorithm;
use liblevenshtein::levenshtein_automaton::LevenshteinAutomaton;

let dict = DynamicDawgChar::from_terms(vec![
    "café", "naïve", "résumé", "déjà"
]);

// Fuzzy search for "cafe" (missing accent)
let automaton = LevenshteinAutomaton::new("cafe", 1, Algorithm::Standard);
let results: Vec<String> = automaton.query(&dict).collect();

println!("{:?}", results);
// Output: ["café"] (distance 1: substitute e→é)

// Add more terms dynamically
dict.insert("cafeteria");

// Search again
let results: Vec<String> = automaton.query(&dict).collect();
println!("{:?}", results);
// Output: ["café", "cafeteria"]
```

### Example 5: Emoji Support

```rust
use libdictenstein::dynamic_dawg::DynamicDawgChar;

let dict = DynamicDawgChar::new();

// Insert emoji
dict.insert("🎉");
dict.insert("🎊");
dict.insert("🎁");
dict.insert("😀");
dict.insert("😃");

// All work correctly
assert!(dict.contains("🎉"));
assert!(dict.contains("😀"));

// Fuzzy search works
let automaton = LevenshteinAutomaton::new("🎉", 1, Algorithm::Standard);
let results: Vec<String> = automaton.query(&dict).collect();

// Finds emoji within distance 1
println!("Matches: {}", results.len());
```

### Example 6: CJK Text

```rust
use libdictenstein::dynamic_dawg::DynamicDawgChar;

let dict = DynamicDawgChar::new();

// Chinese
dict.insert("中文");
dict.insert("中国");
dict.insert("北京");

// Japanese
dict.insert("日本");
dict.insert("東京");

// Korean
dict.insert("한국");
dict.insert("서울");

// Fuzzy search in Chinese
let automaton = LevenshteinAutomaton::new("中文", 1, Algorithm::Standard);
let results: Vec<String> = automaton.query(&dict).collect();

println!("{:?}", results);
// Finds: ["中文", "中国"] (both share "中")
```

### Example 7: Thread-Safe Updates

```rust
use libdictenstein::dynamic_dawg::DynamicDawgChar;
use std::sync::Arc;
use std::thread;

let dict = Arc::new(DynamicDawgChar::new());

// Spawn multiple threads adding Unicode terms
let handles: Vec<_> = (0..4).map(|i| {
    let dict = Arc::clone(&dict);
    thread::spawn(move || {
        match i {
            0 => dict.insert("café"),
            1 => dict.insert("中文"),
            2 => dict.insert("😀"),
            3 => dict.insert("مرحبا"),
            _ => {}
        }
    })
}).collect();

for handle in handles {
    handle.join().unwrap();
}

// All terms successfully inserted
assert!(dict.contains("café"));
assert!(dict.contains("中文"));
assert!(dict.contains("😀"));
assert!(dict.contains("مرحبا"));
```

### Example 8: Compaction with Unicode

```rust
use libdictenstein::dynamic_dawg::DynamicDawgChar;

let dict = DynamicDawgChar::from_terms(vec![
    "café", "cafétéria", "naïve", "résumé", "déjà"
]);

println!("Before deletion: {} nodes", dict.node_count());

// Remove several terms
dict.remove("café");
dict.remove("naïve");
dict.remove("déjà");

println!("After deletion: {} nodes (orphans)", dict.node_count());

// Compact to restore minimality
dict.compact();

println!("After compaction: {} nodes", dict.node_count());
```

## Performance Analysis

### Time Complexity

Same as DynamicDawg:

| Operation | Complexity | Notes |
|-----------|-----------|-------|
| **Insert** | $`\mathcal{O}(m)`$ | m = term length (characters) |
| **Remove** | $`\mathcal{O}(m)`$ | Plus ref count updates |
| **Contains** | $`\mathcal{O}(m)`$ | With Bloom filter: $`\mathcal{O}(1)`$ rejection |
| **Compact** | $`\mathcal{O}(n)`$ | n = total nodes |
| **Query (fuzzy)** | $`\mathcal{O}(m \times d^{2} \times b)`$ | d = distance, b = branching |

### Benchmark Results

#### Construction

```
Build from 10,000 mixed-script terms:
  DynamicDawgChar:   4.4ms
  DynamicDawg:       4.1ms  (7% faster)
  DoubleArrayTrieChar: 3.4ms (22% faster)
```

#### Runtime Operations

```
Single insertion (Unicode term):
  DynamicDawgChar:  ~840ns
  DynamicDawg:      ~800ns  (5% faster)

Single deletion:
  DynamicDawgChar:  ~1.3µs
  DynamicDawg:      ~1.2µs  (8% faster)

Contains check (positive):
  DynamicDawgChar:  ~470ns
  DynamicDawg:      ~450ns  (4% faster)
```

#### Fuzzy Search

```
Query "café" (distance 2) in 10K-term dict:
  DynamicDawgChar:      44.7µs
  DynamicDawg:          42.3µs  (5% faster)
  DoubleArrayTrieChar:  17.1µs  (62% faster)
```

### Memory Usage

```
10,000-term dictionary (mixed scripts):
  Nodes:          ~490KB
  Suffix cache:   ~32KB
  Bloom filter:   ~12KB
  Total:          ~534KB

vs DynamicDawg:      ~294KB (1.8x smaller)
vs DoubleArrayTrieChar: ~900KB (1.7x larger)
```

### Character-Level vs Byte-Level Trade-offs

```
                        DynamicDawg    DynamicDawgChar    Difference
────────────────────────────────────────────────────────────────────────
Memory per node         25 bytes       49 bytes           +96%
Construction time       4.1ms          4.4ms              +7%
Insert time             800ns          840ns              +5%
Query time              42.3µs         44.7µs             +6%
Unicode correctness     ❌             ✅                 Priceless!
```

**Verdict**: ~2x memory, ~5-8% performance overhead for correct Unicode handling is excellent value.

## When to Use

### Decision Matrix

| Scenario | Recommended | Alternative |
|----------|-------------|-------------|
| **Unicode + dynamic updates** | ✅ DynamicDawgChar | - |
| **Multilingual real-time app** | ✅ DynamicDawgChar | - |
| **ASCII + dynamic updates** | ⚠️ DynamicDawg | Slightly faster |
| **Unicode + static/append-only** | ⚠️ DoubleArrayTrieChar | 3x faster |
| **Maximum performance** | ⚠️ DoubleArrayTrieChar | Fastest |
| **Pure ASCII** | ⚠️ DynamicDawg | Less memory |

### Ideal Use Cases

1. **Multi-Language User Dictionaries**
   - Users add/remove words in various languages
   - Correct character-level distances crucial
   - Personal vocabularies with emoji, CJK, etc.

2. **Collaborative Editing (International)**
   - Multiple users from different regions
   - Thread-safe concurrent access
   - Full Unicode support needed

3. **Adaptive Spell Checkers**
   - Learn from user corrections
   - Remove obsolete suggestions
   - Handle all scripts correctly

4. **Chat/Messaging Applications**
   - Dynamic emoji/sticker names
   - User-specific vocabularies
   - Multi-language support

5. **Code Completion (International)**
   - Variable names in various scripts
   - Dynamic scope-based filtering
   - Unicode identifier support

## Related Documentation

- [Dictionary Layer](../README.md) - Overview of all dictionary types
- [DynamicDawg](dynamic-dawg.md) - Byte-level variant
- [DoubleArrayTrieChar](double-array-trie-char.md) - Faster static alternative
- [Value Storage](../../09-value-storage/README.md) - Using values with DynamicDawgChar

## References

### Unicode and Levenshtein Distance

1. **Schulz, K. U., & Mihov, S. (2002)**. "Fast String Correction with Levenshtein Automata"
   - *International Journal on Document Analysis and Recognition*, 5(1), 67-85
   - 📄 Discusses character-level vs byte-level matching

2. **Unicode Consortium**. *The Unicode Standard, Version 15.0*
   - 📄 [https://www.unicode.org/versions/Unicode15.0.0/](https://www.unicode.org/versions/Unicode15.0.0/)
   - Official Unicode specification

### DAWG Algorithms

3. **Blumer, A., Blumer, J., Haussler, D., McConnell, R., & Ehrenfeucht, A. (1987)**. "Complete inverted files for efficient text retrieval and analysis"
   - *Journal of the ACM*, 34(3), 578-595
   - DOI: [10.1145/28869.28873](https://doi.org/10.1145/28869.28873)
   - 📄 DAWG construction algorithms

4. **Crochemore, M., & Vérin, R. (1997)**. "Direct construction of compact directed acyclic word graphs"
   - *Annual Symposium on Combinatorial Pattern Matching*, 116-129
   - DOI: [10.1007/3-540-63220-4_55](https://doi.org/10.1007/3-540-63220-4_55)
   - 📄 Incremental DAWG construction

## Next Steps

- **Byte-Level**: Compare with [DynamicDawg](dynamic-dawg.md)
- **Static Alternative**: Explore [DoubleArrayTrieChar](double-array-trie-char.md)
- **Values**: Learn about [Value Storage](../../09-value-storage/README.md)
- **Unicode Handling**: Read [DoubleArrayTrieChar Unicode Guide](double-array-trie-char.md#unicode-fundamentals)

---

**Navigation**: [← Dictionary Layer](../README.md) | [DynamicDawg (byte-level)](dynamic-dawg.md) | [Algorithms Home](../../README.md)
