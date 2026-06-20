# Protobuf Serialization Format

[← Documentation Index](../README.md)

## Overview

The protobuf serialization format provides cross-language compatibility for liblevenshtein dictionaries. It's supported across all implementations (Java, C++, Rust) and preserves the internal graph structure.

![Serialization formats: how Bincode, JSON, and Protobuf (V1 / optimized V2) encode the same dictionary, contrasting term-list versus graph-structure layouts and their relative sizes.](../diagrams/serialization/serialization-formats.svg)

## Format Specification

Defined in `proto/liblevenshtein.proto`:

```protobuf
message Dictionary {
    message Edge {
        uint64 source_id = 1;  // Source node ID
        uint32 label = 2;       // Character/byte label
        uint64 target_id = 3;   // Target node ID
    }

    repeated uint64 node_id = 1;         // All node IDs
    repeated uint64 final_node_id = 2;   // Terminal node IDs
    repeated Edge edge = 3;               // All edges
    uint64 root_id = 4;                   // Root node (usually 0)
    uint64 size = 5;                      // Number of terms
}
```

## Why Graph Structure?

Unlike bincode/JSON which serialize terms as strings, protobuf serializes the **actual trie/DAWG structure**:

### Advantages

1. **Cross-Language Compatibility**: Works with Java, C++, and Rust implementations
2. **Structure Preservation**: No rebuild needed - loads directly into memory
3. **DAWG Efficiency**: For minimized DAWGs with node sharing, much more compact than term lists
4. **Fast Deserialization**: No trie reconstruction needed

### Trade-offs

For simple tries without node sharing:
- **Bincode**: 109 bytes (just term strings)
- **Protobuf**: 262 bytes (full graph structure)

For DAWGs with significant node sharing:
- **Bincode**: Size grows with unique strings
- **Protobuf**: Size grows with unique nodes (can be much smaller!)

## Current Implementation

### Serialization

```rust
use liblevenshtein::prelude::*;

let dict = PathMapDictionary::from_iter(vec!["test", "testing"]);

// Serialize as protobuf
let mut file = std::fs::File::create("dict.pb")?;
ProtobufSerializer::serialize(&dict, file)?;
```

The serializer:
1. Traverses the dictionary structure (DFS)
2. Assigns sequential IDs to nodes
3. Records all edges with (source, label, target)
4. Marks final (terminal) nodes

### Deserialization

```rust
// Deserialize from protobuf
let file = std::fs::File::open("dict.pb")?;
let dict: PathMapDictionary = ProtobufSerializer::deserialize(file)?;
```

The deserializer:
1. Decodes protobuf message
2. Builds adjacency list from edges
3. Extracts terms via DFS from root
4. Reconstructs dictionary from terms

## Limitations & Future Optimizations

### Current Limitations

1. **No True DAWG Serialization**: Since the `Dictionary` trait doesn't expose node identity, we serialize as a trie structure (each path creates unique nodes). True DAWGs with node sharing require dictionary implementations to expose node IDs.

2. **Redundant node_id Field**: Node IDs are sequential (0, 1, 2, ...), so the `node_id` array could be omitted with minimal changes to consumers.

3. **Edge Packing**: Edges are stored as messages. Could pack as flat array: `[src1, lbl1, tgt1, src2, lbl2, tgt2, ...]`

### Proposed Optimizations

#### Option 1: Remove Redundant node_id Field

```protobuf
message DictionaryV2 {
    // REMOVED: repeated uint64 node_id = 1;
    repeated uint64 final_node_id = 1;  // Just finals needed
    repeated Edge edge = 2;
    uint64 root_id = 3;
    uint64 size = 4;

    // Consumers infer node IDs from edges
}
```

**Savings**: ~8 bytes per node (varint encoded)

#### Option 2: Packed Edge Format

```protobuf
message DictionaryV2 {
    repeated uint64 final_node_id = 1;
    // Pack as [src, lbl, tgt, src, lbl, tgt, ...]
    repeated uint64 edge_data = 2 [packed=true];
    uint64 root_id = 3;
    uint64 size = 4;
}
```

**Savings**: Message overhead per edge (~2 bytes each)

#### Option 3: Delta Encoding

For sequential node IDs, encode deltas:

```
Nodes: [0, 1, 2, 3, 4]
Deltas: [0, 1, 1, 1, 1]  // Smaller varints!
```

**Savings**: Varint efficiency for sequential IDs

### True DAWG Serialization

To serialize actual DAWGs with node sharing, we need:

1. **Dictionary Trait Extension**:
```rust
trait IdentifiableDictionary: Dictionary {
    fn node_id(&self, node: &Self::Node) -> u64;
}
```

2. **Modified Serialization**:
```rust
fn extract_graph_with_sharing<D: IdentifiableDictionary>(dict: &D) -> proto::Dictionary {
    let mut visited = HashMap::new();

    // Track which nodes we've seen
    fn traverse(node: &N, dict: &D, visited: &mut HashMap<u64, ()>) {
        let id = dict.node_id(node);
        if visited.contains_key(&id) {
            return;  // Already visited - this is node sharing!
        }
        visited.insert(id, ());

        for (label, child) in node.edges() {
            traverse(&child, dict, visited);
        }
    }
}
```

This would properly serialize DAWGs with shared nodes.

## Compatibility

The current format is **fully compatible** with:
- liblevenshtein-java
- liblevenshtein-cpp
- Future liblevenshtein implementations

Changes to the format (like proposed optimizations) should be:
1. Optional (via `DictionaryV2` message)
2. Backward compatible
3. Coordinated across all implementations

## Migration Guide

### Migrating from V1 to V2

The optimized V2 format provides 40-60% space savings while maintaining full compatibility. Here's how to migrate:

#### For New Projects

Use `OptimizedProtobufSerializer` from the start:

```rust
use liblevenshtein::prelude::*;

let dict = PathMapDictionary::from_iter(vec!["test", "testing"]);

// Serialize with V2 (optimized)
let mut file = std::fs::File::create("dict.pb")?;
OptimizedProtobufSerializer::serialize(&dict, file)?;

// Deserialize (automatically detects format)
let file = std::fs::File::open("dict.pb")?;
let dict: PathMapDictionary = OptimizedProtobufSerializer::deserialize(file)?;
```

#### For Existing V1 Files

Convert existing V1 protobuf files to V2:

```rust
use liblevenshtein::prelude::*;

// Read V1 format
let v1_file = std::fs::File::open("dict_v1.pb")?;
let dict: PathMapDictionary = ProtobufSerializer::deserialize(v1_file)?;

// Write V2 format
let v2_file = std::fs::File::create("dict_v2.pb")?;
OptimizedProtobufSerializer::serialize(&dict, v2_file)?;
```

#### Choosing Between V1 and V2

**Use V1 (`ProtobufSerializer`) when:**
- Cross-language compatibility with older liblevenshtein versions
- Other implementations haven't adopted V2 yet
- Compatibility is more important than file size

**Use V2 (`OptimizedProtobufSerializer`) when:**
- File size matters (60% smaller)
- Using only Rust implementation
- Other implementations support V2

#### Format Detection

Both serializers automatically detect the format during deserialization:

```rust
// Works with both V1 and V2 files
let file = std::fs::File::open("dict.pb")?;
let dict: PathMapDictionary = OptimizedProtobufSerializer::deserialize(file)?;
```

The `DictionaryContainer` message uses a `oneof` field to distinguish formats:

```protobuf
message DictionaryContainer {
    oneof format {
        Dictionary v1 = 1;      // Detected automatically
        DictionaryV2 v2 = 2;    // Detected automatically
    }
}
```

#### Performance Comparison

Based on benchmarks with 100 terms:

| Format | Size | vs V1 | vs Bincode |
|--------|------|-------|------------|
| Bincode | 1508 bytes | +32.3% | baseline |
| JSON | 1302 bytes | +14.2% | -13.7% |
| Protobuf V1 | 1140 bytes | baseline | -24.4% |
| Protobuf V2 | 454 bytes | **-60.2%** | **-69.9%** |

#### Migration Checklist

- [ ] Identify all `.pb` files using V1 format
- [ ] Decide on migration strategy (convert all at once vs gradual)
- [ ] Test deserialization with V2 serializer on existing V1 files
- [ ] Convert files using the code example above
- [ ] Update serialization code to use `OptimizedProtobufSerializer`
- [ ] Verify file sizes decreased by ~40-60%
- [ ] Update documentation/comments referencing the old format

## Benchmarks

```
Small dictionary (7 terms, trie structure):
- Bincode:  109 bytes
- JSON:     156 bytes
- Protobuf: 262 bytes

Large dictionary (1000 terms, DAWG structure):
- Bincode:  ~15 KB (term strings)
- Protobuf: ~8 KB (with node sharing) [projected]
```

## Usage Examples

See:
- `examples/serialization.rs` - Basic usage
- `src/commands/handlers/io.rs` - Feature-gated protobuf round-trip tests for
  PathMap, suffix automaton, and compressed protobuf dictionaries

## References

- Protocol Buffers: https://protobuf.dev/
- Varint Encoding: https://protobuf.dev/programming-guides/encoding/#varints
- DAWG Minimization: Schulz & Mihov (2002)
