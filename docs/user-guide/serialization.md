# Serialization Guide

**Version**: 0.9.1
**Last Updated**: 2026-06-19

This guide explains how to save and load dictionaries using liblevenshtein-rust's serialization support.

## Overview

Constructing large dictionaries can be time-consuming. Serialization allows you to:
- Build a dictionary once
- Save it to disk
- Load it quickly in subsequent runs
- Share dictionaries across applications
- Support multiple serialization formats

![Serialization formats: a dictionary is encoded as bincode, JSON, plain text, or protobuf, each optionally wrapped in gzip compression](../diagrams/serialization/serialization-formats.svg)

*Serialization formats: every backend can be written as bincode, JSON, text, or protobuf, optionally gzip-compressed.*

## Feature Flags

Serialization requires the `serialization` feature:

```toml
[dependencies]
liblevenshtein = {
    git = "https://github.com/universal-automata/liblevenshtein-rust",
    tag = "v0.9.1",
    features = ["serialization"]
}

# With compression support
liblevenshtein = {
    git = "https://github.com/universal-automata/liblevenshtein-rust",
    tag = "v0.9.1",
    features = ["serialization", "compression"]
}

# With Protobuf support
liblevenshtein = {
    git = "https://github.com/universal-automata/liblevenshtein-rust",
    tag = "v0.9.1",
    features = ["serialization", "protobuf"]
}
```

## Supported Formats

| Format | Feature Flag | Description | Best For |
|--------|-------------|-------------|----------|
| **Bincode** | `serialization` | Binary, fast, compact | Production use, Rust-to-Rust |
| **JSON** | `serialization` | Text-based, human-readable | Debugging, cross-language |
| **Plain Text** | `serialization` | Newline-delimited UTF-8 | Simple term lists, human-editable |
| **Protobuf** | `protobuf` | Binary, optimized, cross-language | Cross-language, optimized |
| **Gzip** | `compression` | Compressed (85% reduction) | Network transfer, storage |

## Basic Usage

### Saving a Dictionary

```rust
use liblevenshtein::prelude::*;
use liblevenshtein::serialization::{DictionarySerializer, Format};
use std::fs::File;

// Create a dictionary
let dict = DoubleArrayTrie::from_terms(vec![
    "test", "testing", "tested", "tester"
]);

// Save to file
let file = File::create("dictionary.bin")?;
dict.serialize(file, Format::Bincode)?;
```

### Loading a Dictionary

```rust
use liblevenshtein::prelude::*;
use liblevenshtein::serialization::{DictionaryDeserializer, Format};
use std::fs::File;

// Load from file
let file = File::open("dictionary.bin")?;
let dict: DoubleArrayTrie = DoubleArrayTrie::deserialize(file, Format::Bincode)?;

// Use the dictionary
let transducer = Transducer::new(dict, Algorithm::Standard);
for term in transducer.query("test", 1) {
    println!("{}", term);
}
```

## Format-Specific Usage

### Bincode (Binary, Fast)

**Characteristics:**
- Binary format
- Fast serialization/deserialization
- Compact size
- Rust-specific (not cross-language compatible)

```rust
use liblevenshtein::prelude::*;
use liblevenshtein::serialization::{DictionarySerializer, Format};
use std::fs::File;

let dict = DoubleArrayTrie::from_terms(vec!["test", "testing"]);

// Save
let file = File::create("dict.bincode")?;
dict.serialize(file, Format::Bincode)?;

// Load
let file = File::open("dict.bincode")?;
let dict: DoubleArrayTrie = DoubleArrayTrie::deserialize(file, Format::Bincode)?;
```

**When to use:** Default choice for Rust-to-Rust serialization.

### JSON (Human-Readable)

**Characteristics:**
- Text-based, human-readable
- Larger file size
- Slower than binary formats
- Cross-language compatible

```rust
use liblevenshtein::prelude::*;
use liblevenshtein::serialization::{DictionarySerializer, Format};
use std::fs::File;

let dict = DoubleArrayTrie::from_terms(vec!["test", "testing"]);

// Save
let file = File::create("dict.json")?;
dict.serialize(file, Format::Json)?;

// Load
let file = File::open("dict.json")?;
let dict: DoubleArrayTrie = DoubleArrayTrie::deserialize(file, Format::Json)?;
```

**When to use:** Debugging, inspection, cross-language interop.

### Plain Text (Simple Lists)

**Characteristics:**
- Newline-delimited UTF-8 text
- Human-readable and editable
- No structure preservation (rebuilds dictionary on load)
- Universal compatibility

```rust
use liblevenshtein::prelude::*;
use liblevenshtein::serialization::{DictionarySerializer, Format};
use std::fs::File;

let dict = DoubleArrayTrie::from_terms(vec!["test", "testing", "tested"]);

// Save
let file = File::create("terms.txt")?;
dict.serialize(file, Format::Text)?;

// The file contains:
// test
// testing
// tested

// Load (rebuilds dictionary from terms)
let file = File::open("terms.txt")?;
let dict: DoubleArrayTrie = DoubleArrayTrie::deserialize(file, Format::Text)?;
```

**When to use:** Simple term lists, manual editing, universal compatibility.

### Protobuf (Optimized, Cross-Language)

**Characteristics:**
- Binary, cross-language
- Multiple optimized variants
- 62% smaller than JSON (V2 format)
- Backend-specific optimizations

**Requires:** `protobuf` feature flag

```rust
use liblevenshtein::prelude::*;
use liblevenshtein::serialization::{DictionarySerializer, Format};
use std::fs::File;

let dict = DoubleArrayTrie::from_terms(vec!["test", "testing"]);

// Save with Protobuf V2 (optimized)
let file = File::create("dict.pb")?;
dict.serialize(file, Format::ProtobufV2)?;

// Load
let file = File::open("dict.pb")?;
let dict: DoubleArrayTrie = DoubleArrayTrie::deserialize(file, Format::ProtobufV2)?;
```

**Protobuf Format Variants:**
- `Format::Protobuf` - Standard protobuf format
- `Format::ProtobufV2` - Optimized format (62% smaller)
- `Format::ProtobufDat` - DoubleArrayTrie-specific optimization
- `Format::ProtobufSuffixAutomaton` - SuffixAutomaton-specific optimization

**When to use:** Cross-language applications, optimized storage, production systems.

### Gzip Compression

**Requires:** `compression` feature flag

**Characteristics:**
- Works with any format
- 85% file size reduction
- Slightly slower load/save
- Great for network transfer

```rust
use liblevenshtein::prelude::*;
use liblevenshtein::serialization::{DictionarySerializer, Format};
use std::fs::File;

let dict = DoubleArrayTrie::from_terms(vec!["test", "testing", "tested"]);

// Save with gzip compression
let file = File::create("dict.bincode.gz")?;
dict.serialize(file, Format::BincodeGz)?;

// Load compressed
let file = File::open("dict.bincode.gz")?;
let dict: DoubleArrayTrie = DoubleArrayTrie::deserialize(file, Format::BincodeGz)?;
```

**Compressed format variants:**
- `Format::BincodeGz` - Bincode with gzip
- `Format::JsonGz` - JSON with gzip
- `Format::TextGz` - Plain text with gzip

**When to use:** Network transfer, storage optimization, backups.

## Format Comparison

### File Size (100K terms dictionary)

| Format | Size | Relative | Compression |
|--------|------|----------|-------------|
| Bincode | 2.1 MB | 1.0× | - |
| BincodeGz | 320 KB | 0.15× | 85% reduction |
| JSON | 5.8 MB | 2.8× | - |
| JsonGz | 450 KB | 0.21× | 92% reduction |
| Text | 1.2 MB | 0.57× | - |
| TextGz | 180 KB | 0.09× | 85% reduction |
| ProtobufV2 | 2.2 MB | 1.05× | - |
| ProtobufDat | 1.9 MB | 0.90× | - |

### Serialization Speed (100K terms)

| Format | Save Time | Load Time |
|--------|-----------|-----------|
| Bincode | 12 ms | 8 ms |
| BincodeGz | 145 ms | 95 ms |
| JSON | 280 ms | 320 ms |
| Text | 45 ms | 180 ms (rebuild) |
| ProtobufV2 | 18 ms | 12 ms |

**Takeaway:** Use Bincode for speed, BincodeGz for size, Protobuf for cross-language.

## Advanced Usage

### Error Handling

```rust
use liblevenshtein::prelude::*;
use liblevenshtein::serialization::{DictionarySerializer, Format, SerializationError};
use std::fs::File;

fn save_dictionary(dict: &DoubleArrayTrie, path: &str) -> Result<(), SerializationError> {
    let file = File::create(path)
        .map_err(|e| SerializationError::Io(e))?;
    dict.serialize(file, Format::Bincode)?;
    Ok(())
}

fn load_dictionary(path: &str) -> Result<DoubleArrayTrie, SerializationError> {
    let file = File::open(path)
        .map_err(|e| SerializationError::Io(e))?;
    DoubleArrayTrie::deserialize(file, Format::Bincode)
}
```

### Automatic Format Detection

The CLI tool supports automatic format detection based on file extension:

```bash
# Save with auto-detected format
liblevenshtein convert -i terms.txt -o dict.bincode.gz

# Load with auto-detected format
liblevenshtein query dict.bincode.gz "test" 2
```

### Custom Serialization

For advanced use cases, you can manually serialize dictionary components:

```rust
use liblevenshtein::prelude::*;
use liblevenshtein::serialization::DictionarySerializer;

// Serialize to bytes
let dict = DoubleArrayTrie::from_terms(vec!["test"]);
let bytes = bincode::serialize(&dict)?;

// Deserialize from bytes
let dict: DoubleArrayTrie = bincode::deserialize(&bytes)?;
```

## CLI Tool Usage

The CLI tool provides convenient serialization operations:

### Convert Between Formats

```bash
# Text to Bincode
liblevenshtein convert -i words.txt -o dict.bincode

# Bincode to JSON (for inspection)
liblevenshtein convert -i dict.bincode -o dict.json

# Compress existing dictionary
liblevenshtein convert -i dict.bincode -o dict.bincode.gz
```

### Query Serialized Dictionaries

```bash
# Query directly from file
liblevenshtein query dict.bincode "test" 2

# Query compressed dictionary
liblevenshtein query dict.bincode.gz "test" 2 --algorithm transposition
```

### REPL with Persistence

```bash
# Start REPL with saved dictionary
liblevenshtein repl dict.bincode

# In REPL:
> query test 2
> insert testing
> save dict-updated.bincode
> exit
```

## Best Practices

### 1. Choose the Right Format

```rust
// Development/debugging: JSON
dict.serialize(file, Format::Json)?;

// Production (Rust-only): Bincode
dict.serialize(file, Format::Bincode)?;

// Production (cross-language): Protobuf V2
dict.serialize(file, Format::ProtobufV2)?;

// Network transfer: Compressed
dict.serialize(file, Format::BincodeGz)?;
```

### 2. Version Your Dictionaries

```rust
use serde::{Serialize, Deserialize};

#[derive(Serialize, Deserialize)]
struct VersionedDictionary {
    version: String,
    dict: DoubleArrayTrie,
}

let versioned = VersionedDictionary {
    version: "1.0.0".to_string(),
    dict,
};

// Save with version info
let file = File::create("dict-v1.0.0.bincode")?;
bincode::serialize_into(file, &versioned)?;
```

### 3. Validate After Loading

```rust
// Load dictionary
let dict: DoubleArrayTrie = DoubleArrayTrie::deserialize(file, Format::Bincode)?;

// Validate
if let Some(len) = dict.len() {
    if len == 0 {
        return Err("Empty dictionary loaded".into());
    }
}

// Test with known term
if !dict.contains("test") {
    return Err("Dictionary validation failed".into());
}
```

### 4. Handle Large Dictionaries

```rust
use std::io::BufWriter;

// Use buffered writer for large dictionaries
let file = File::create("large-dict.bincode")?;
let writer = BufWriter::new(file);
dict.serialize(writer, Format::Bincode)?;
```

## Performance Tips

1. **Use Bincode for speed**: Fastest serialization/deserialization
2. **Use compression for size**: 85% size reduction with minimal overhead
3. **Buffer I/O**: Use `BufReader`/`BufWriter` for large dictionaries
4. **Cache dictionaries**: Load once, reuse across multiple queries
5. **Consider format trade-offs**: Speed vs. size vs. compatibility

## Troubleshooting

### Dictionary Too Large

```bash
# Use compression
liblevenshtein convert -i large.txt -o large.bincode.gz

# Or use Protobuf V2
liblevenshtein convert -i large.txt -o large.pb --format protobuf-v2
```

### Cross-Language Compatibility

```bash
# Use Protobuf for cross-language
liblevenshtein convert -i words.txt -o dict.pb --format protobuf-v2
```

### Corrupted Files

```bash
# Validate by loading and checking
liblevenshtein query dict.bincode "" 0

# Rebuild from source if needed
liblevenshtein convert -i original-words.txt -o dict-rebuilt.bincode
```

## Related Documentation

- [Getting Started](getting-started.md) - Basic usage
- [Backends](backends.md) - Dictionary backend comparison
- [Features](features.md) - Full feature overview
- CLI Usage - CLI tool guide
- [Protobuf Design](../design/protobuf-serialization.md) - Protobuf format specification

## Examples

See `examples/serialization_basic.rs` in the repository:

```bash
cargo run --example serialization_basic --features serialization
```

---

[← Documentation Index](../README.md)
