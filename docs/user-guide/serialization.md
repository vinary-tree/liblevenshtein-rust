# Binary Persistence Guide

**Version**: 0.9.1
**Last updated**: 2026-08-02

Large dictionaries require a compact persistence format. Liblevenshtein therefore supports
two dictionary formats: bincode for efficient Rust-to-Rust storage and Protocol Buffers for
portable binary interchange. Gzip can wrap either format. JSON, TOML, and newline-delimited
text are deliberately not dictionary persistence formats: their size and parse cost are not
appropriate for production dictionaries.

This restriction concerns persisted dictionaries. A program may still ingest a source word
list while constructing a dictionary; after construction, persist the resulting backend in a
supported binary format.

![Bincode and Protocol Buffers are the supported persistence formats; gzip may wrap either binary stream](../diagrams/serialization/serialization-formats.svg)

## Feature flags

| Capability | Cargo feature | Intended use |
|---|---|---|
| Bincode | `serialization` | Compact, high-throughput Rust storage |
| Protocol Buffers | `protobuf` | Portable binary interchange |
| Gzip wrapper | `compression` | Lower transfer or storage size |

```toml
[dependencies]
liblevenshtein = {
    git = "https://github.com/vinary-tree/liblevenshtein-rust",
    tag = "v0.9.1",
    features = ["serialization"]
}
```

Add `protobuf`, `compression`, or both when those capabilities are required. Enabling
`serialization` does not enable a text-format dependency.

The complete operation-model types (`OperationSet`, `OperationType`,
`OperationApplicability`, `SubstitutionSet`, and `SubstitutionPair`) deliberately do not
implement generic Serde traits. Bincode uses private versioned wire types internally. This
prevents downstream crates from silently treating JSON or TOML as an operation-set persistence
format while retaining the compact bincode API below. Other crate subsystems may use Serde for
non-dictionary configuration or WebAssembly bindings; that is not a persistence format.

## Bincode dictionaries

`BincodeSerializer` is the default for Rust applications. It works for every byte-oriented
dictionary backend, including backends that do not themselves implement Serde traits.

```rust
use libdictenstein::double_array_trie::DoubleArrayTrie;
use libdictenstein::serialization::{BincodeSerializer, DictionarySerializer};

let dictionary = DoubleArrayTrie::from_terms(vec!["test", "tested", "testing"]);

let mut bytes = Vec::new();
BincodeSerializer::serialize(&dictionary, &mut bytes)?;

let restored: DoubleArrayTrie = BincodeSerializer::deserialize(&bytes[..])?;
assert!(restored.contains("testing"));
# Ok::<(), libdictenstein::serialization::SerializationError>(())
```

Decoding is exact: a valid object followed by trailing bytes is rejected. This prevents a
caller from accidentally accepting a valid prefix while ignoring malformed or concatenated
data.

The lower-level `bincode_compat` module is available for types that directly implement
`Serialize` and `Deserialize`:

```rust
use libdictenstein::double_array_trie::DoubleArrayTrie;
use libdictenstein::serialization::bincode_compat;

let dictionary = DoubleArrayTrie::from_terms(vec!["café", "新しい"]);
let bytes = bincode_compat::serialize(&dictionary)?;
let restored: DoubleArrayTrie = bincode_compat::deserialize(&bytes)?;
assert!(restored.contains("café"));
# Ok::<(), Box<dyn std::error::Error>>(())
```

Prefer `BincodeSerializer` for ordinary dictionary storage. The compatibility module is a
Serde transport, not a separately versioned storage contract.

## Protocol Buffers dictionaries

Enable `protobuf` for a binary format whose schema can be implemented in other languages.
The general serializer preserves the dictionary's terms:

```rust
use libdictenstein::double_array_trie::DoubleArrayTrie;
use libdictenstein::serialization::{DictionarySerializer, ProtobufSerializer};

let dictionary = DoubleArrayTrie::from_terms(vec!["test", "tested", "testing"]);

let mut bytes = Vec::new();
ProtobufSerializer::serialize(&dictionary, &mut bytes)?;

let restored: DoubleArrayTrie = ProtobufSerializer::deserialize(&bytes[..])?;
assert!(restored.contains("tested"));
# Ok::<(), libdictenstein::serialization::SerializationError>(())
```

`OptimizedProtobufSerializer` is the compact general variant. The specialized
`DatProtobufSerializer` and `SuffixAutomatonProtobufSerializer` APIs preserve additional
backend structure. Their payloads are binary and self-identifying; they do not accept a
newline-text compatibility payload.

Protocol Buffers gives broad ecosystem support, but a schema alone does not guarantee that
two implementations enforce identical limits. Cross-language consumers should bound message
size, repeated-field counts, and decoded string bytes before admitting untrusted payloads.

## Gzip compression

With `compression`, `GzipSerializer<S>` composes with a supported binary serializer:

```rust
use libdictenstein::double_array_trie::DoubleArrayTrie;
use libdictenstein::serialization::{
    BincodeSerializer, DictionarySerializer, GzipSerializer,
};

let dictionary = DoubleArrayTrie::from_terms(vec!["test", "tested", "testing"]);

let mut compressed = Vec::new();
GzipSerializer::<BincodeSerializer>::serialize(&dictionary, &mut compressed)?;

let restored: DoubleArrayTrie =
    GzipSerializer::<BincodeSerializer>::deserialize(&compressed[..])?;
assert!(restored.contains("test"));
# Ok::<(), libdictenstein::serialization::SerializationError>(())
```

`GzipSerializer<ProtobufSerializer>` is available when both `compression` and `protobuf` are
enabled. Compression is a transport wrapper, not a third persistence schema. Apply compressed
and decompressed byte limits when reading untrusted data to prevent decompression bombs.

## Operation-set persistence

Generalized edit operations support the same two binary choices as dictionaries. The bincode
API uses a stable `LLEVOPS\0` envelope with version, flags, and declared payload length. The
protobuf API uses the versioned `OperationSetContainer` schema in
`proto/operation_set.proto`. Both preserve operation order, exact IEEE-754 weight bits, owned
diagnostic names, explicit applicability, raw-byte restrictions, and Unicode restrictions.
Listed substitutions are emitted in canonical order. The bincode version-1 layout is retained
by private wire structs, so removing public generic Serde did not change existing envelope
bytes.

```rust
use liblevenshtein::transducer::{
    OperationSet, OperationSetBinaryLimits, OperationSetBuilder,
};

let operations = OperationSetBuilder::new().with_standard_ops().build();
let bytes = operations.to_binary()?;

let mut limits = OperationSetBinaryLimits::default();
limits.max_operations = 16;
let restored = OperationSet::from_binary_with_limits(&bytes, limits)?;
assert_eq!(restored, operations);
# Ok::<(), Box<dyn std::error::Error>>(())
```

With `protobuf`, portable interchange uses a separate method so callers never guess a format:

```rust
use liblevenshtein::transducer::{OperationSet, OperationSetBuilder};

let operations = OperationSetBuilder::new().with_standard_ops().build();
let bytes = operations.to_protobuf()?;
let restored = OperationSet::from_protobuf(&bytes)?;
assert_eq!(restored, operations);
# Ok::<(), Box<dyn std::error::Error>>(())
```

Before `prost` allocates decoded vectors or strings, the protobuf decoder performs a
non-allocating wire scan that enforces payload, operation, per-operation pair, total-pair,
name, and aggregate restriction-text limits. It then rejects missing/unknown schema versions,
unknown applicability values, invalid bytes, inconsistent restriction tags, and every semantic
error checked by `OperationSet::validate`. Unknown protobuf fields are skipped for forward
compatibility and are not re-emitted. This differs intentionally from the bincode envelope,
which rejects any bytes outside its exact declared payload.

With `compression`, the following methods wrap exactly one gzip member around the selected
inner representation:

- `to_binary_gzip` / `from_binary_gzip`
- `to_protobuf_gzip` / `from_protobuf_gzip` (also requires `protobuf`)

The gzip decoders bound compressed input and decompressed output, verify the gzip checksum,
reject concatenated members or trailing bytes, and then delegate to the ordinary inner decoder.
Gzip is optional because bincode and protobuf are compact encodings but not compression
algorithms: repeated names, prefixes, and field patterns can still compress well, at the cost
of encode/decode CPU and loss of direct random access. Measure representative artifacts before
making gzip a storage default.

An operation's explicit applicability tag controls its behavior; its diagnostic name never
does. The compile-checked [`operation_set_persistence` example](../../examples/operation_set_persistence.rs)
round-trips one complete configuration through both formats.

## Command-line use

Dictionary format detection, conversion flags, and command examples belong to
the separate
[`liblevenshtein-cli` dictionary guide](https://github.com/vinary-tree/liblevenshtein-rust-cli/blob/master/docs/commands/dictionaries.md).
This page documents the reusable serialization API and wire-format behavior.

## Compatibility and safety

- Treat stored bytes as untrusted input unless their provenance is guaranteed.
- Bound input size before decoding and decompressed size while inflating gzip data.
- Keep the creating application version or schema version with long-lived artifacts.
- Decode into the expected backend and format; do not guess among unrelated schemas after a
  successful prefix decode.
- Rebuild artifacts when a format version is unsupported. Never reinterpret unknown bytes as
  a plaintext dictionary.
- Use atomic replacement when updating a dictionary file so interruption cannot leave a
  partially written artifact.

The operation-set envelope has its own explicit version. Dictionary bincode payloads follow
the `libdictenstein` compatibility contract; Protocol Buffers payloads follow their published
schema and embedded format markers.

## Choosing a format

Use bincode when all readers are Rust applications using compatible library versions. Use
Protocol Buffers when non-Rust readers, an explicit schema, or longer-lived interchange is
required. Add gzip only after measuring the size/latency trade-off on representative
dictionaries.

Do not select JSON, TOML, or another text encoding for production dictionary persistence.
