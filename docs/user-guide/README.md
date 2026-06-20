# User Guide

Documentation for using liblevenshtein-rust in your projects.

## Getting Started

### [Getting Started](getting-started.md)
Quick start guide for new users:
- Installation instructions
- Basic usage examples
- Choosing algorithms and backends
- First fuzzy matching queries

## Core Guides

### [Algorithms](algorithms.md)
Deep dive into Levenshtein distance algorithms:
- Standard Levenshtein distance
- Transposition (Damerau-Levenshtein)
- Merge and Split operations
- Algorithm selection guide
- Distance threshold tuning

### [Dictionary Backends](backends.md)
Comprehensive backend comparison and selection guide:
- DoubleArrayTrie (recommended default)
- PathMap (dynamic updates)
- DAWG variants (space-efficient)
- SuffixAutomaton (substring matching)
- Unicode support with character-level variants
- Performance characteristics and trade-offs

### [Serialization](serialization.md)
Save and load dictionaries efficiently:
- Supported formats (Bincode, JSON, Text, Protobuf)
- Compression options (85% size reduction)
- Format comparison and selection
- CLI tool usage
- Best practices

## Feature Guides

### [Features](features.md)
Comprehensive overview of all library features:
- Levenshtein distance computation
- Fuzzy search capabilities
- Multiple dictionary backends
- Serialization options
- Cache eviction strategies
- SIMD acceleration

### [Code Completion Guide](code-completion.md)
Build code completion with liblevenshtein:
- Setting up dictionaries
- Configuring fuzzy matching
- Optimizing for IDE integration
- Performance tuning

### [Thread Safety](thread-safety.md)
Concurrent access patterns:
- Thread safety guarantees
- Dynamic dictionary updates
- Synchronization strategies
- Best practices for multi-threaded applications

## Additional Resources

- Main [README](../../README.md) - Project overview and quick examples
- [Developer Guide](../developer-guide/README.md) - Contributing and architecture
- [Design Documents](../design/README.md) - Technical specifications
- [Benchmarks](../benchmarks/README.md) - Performance measurements
