# Native binary persistence

Liblevenshtein.jl exposes the native Rust serializers as bounded byte APIs.
Serialization encodes an accepted dictionary term language, a suffix
automaton's indexed source texts, or a generalized edit-operation set into
bytes. The caller decides where to store those bytes. Decoding returns a
closeable native snapshot.

These operations require native API revision 32 and the
`BUILD_FEATURE_SERIALIZATION` bit. Protocol Buffers formats also require
`BUILD_FEATURE_PROTOBUF`; gzip formats require `BUILD_FEATURE_COMPRESSION`.
Operation-set persistence requires API revision 33.
The [native persistence design](https://github.com/vinary-tree/liblevenshtein-rust/blob/master/docs/design/protobuf-serialization.md)
defines the wire structures and compatibility roles.

## Dictionary formats

| Julia `format` | Native format | Compatibility |
|---|---|---|
| `:bincode_v1` | Fixed-integer, little-endian `Vec<String>` | Rust-native V1. |
| `:protobuf_v1` | Explicit graph nodes, edges, and terminals | Portable V1 interchange. |
| `:protobuf_v2` | Packed graph edges and terminal deltas | Compact native V2. |
| `:protobuf_dat_v1` | DAT-specific `LDT1` length-delimited terms | Native DAT reconstruction. |
| `:gzip_bincode_v1` | Gzip over bincode V1 | Compressed Rust-native V1. |
| `:gzip_protobuf_v1` | Gzip over Protobuf V1 | Compressed portable V1. |
| `:gzip_protobuf_v2` | Gzip over Protobuf V2 | Compressed native V2. |

The format is an explicit argument. These bytes do not contain an envelope
that negotiates a format automatically: retain the selected format alongside
the persisted bytes. V1 Protobuf is the choice for a V1-only cross-language
consumer. Native V2 and DAT formats have their own revisions and decoding
rules. All formats preserve accepted UTF-8 terms, including the empty term;
duplicates collapse because the encoder builds the native byte dictionary.
Mapped values and in-memory backend layout are not part of these term formats.

```julia
using Liblevenshtein

bytes = dictionary_bytes(["café", "cab", "cab"];
    format=:protobuf_v1, max_terms=3, max_term_bytes=8,
    max_total_term_bytes=16, max_payload_bytes=256)
write("terms.pb", bytes)

snapshot = dictionary_terms(read("terms.pb"); format=:protobuf_v1,
    max_terms=3, max_term_bytes=8,
    max_total_term_bytes=16, max_payload_bytes=256)
try
    collect(snapshot) # ["cab", "café"]
finally
    close(snapshot)
end
```

`bincode_dictionary_bytes` and `bincode_dictionary_terms` select bincode V1.
`protobuf_dictionary_bytes` and `protobuf_dictionary_terms` select V1 or V2
with `version=1` or `version=2`, and accept `gzip=true`. The `gzip_bincode_*`
and `dat_protobuf_dictionary_*` calls select their named formats. These are
convenience calls over `dictionary_bytes` and `dictionary_terms`.

## Value-preserving Bincode

`valued_dictionary_bytes(pairs; value_kind=:u64)` invokes Rust's
`BincodeSerializer::serialize_with_values` on a byte-domain dictionary. With
`unit_domain=:unicode`, it invokes `serialize_with_values_char` on a Unicode
dictionary. `value_kind=:bytes` selects native `Vec<u8>` values; `:u64`
selects native `u64` values. The returned wire type is respectively
`Vec<(String, Vec<u8>)>` or `Vec<(String, u64)>`, distinct from the term-only
`Vec<String>` format. The native generic serializer also supports other Rust
value types, but their concrete Serde schema must be supplied by a matching
typed consumer. The Julia boundary exposes the interoperable `u64` and raw
byte value domains; UTF-8 text values can be carried as raw bytes.

```julia
bytes = valued_dictionary_bytes(["cab" => UInt8[0, 255],
    "café" => UInt8[]]; value_kind=:bytes, unit_domain=:unicode,
    max_entries=2, max_term_bytes=8, max_total_term_bytes=16,
    max_value_bytes=2, max_total_value_bytes=2, max_payload_bytes=128)
entries = valued_dictionary_entries(bytes; value_kind=:bytes,
    unit_domain=:unicode, max_entries=2, max_term_bytes=8,
    max_total_term_bytes=16, max_value_bytes=2,
    max_total_value_bytes=2, max_payload_bytes=128)
try
    collect(entries) # ["cab" => UInt8[0, 255], "café" => UInt8[]]
finally
    close(entries)
end
```

Specify `value_kind` when encoding and decoding. The bytes contain no type or
unit-domain tag, so retain both choices alongside a persisted file. An empty
byte vector remains a present value. Input count, each term and value, both
aggregate byte totals, and payload bytes have caller-selected ceilings.
Malformed lengths, invalid UTF-8 terms, truncated records, trailing bytes,
and limit overruns fail before a decoded snapshot is published. Iteration
copies keys and values into Julia-owned objects under a lock.

## Suffix-automaton source formats

`suffix_source_bytes(texts; format=:bincode_v1)` preserves the *ordered source
texts* used to build the native suffix automaton. The other supported format
is `:protobuf_v1`. The API never enumerates all recognized substrings, whose
number can be much larger than the input. Duplicate source texts remain
distinct records.

```julia
bytes = suffix_source_bytes(["banana", "bandana"];
    format=:protobuf_v1, max_terms=2, max_term_bytes=16,
    max_total_term_bytes=32, max_payload_bytes=256)
sources = suffix_source_texts(bytes; format=:protobuf_v1,
    max_terms=2, max_term_bytes=16,
    max_total_term_bytes=32, max_payload_bytes=256)
try
    collect(sources) # ["banana", "bandana"]
finally
    close(sources)
end
```

## Generalized operation-set formats

`operation_set_bytes(operations; format=:binary_v1)` encodes a
`GeneralizedOperationSet` using the native versioned binary envelope. Other
formats are `:protobuf_v1`, `:gzip_binary_v1`, and `:gzip_protobuf_v1`.
The binary envelope carries a format version and rejects unsupported versions
or flags. Protobuf V1 uses its own versioned message. The selected codec must
still be retained alongside gzip data.

```julia
grammar = GeneralizedOperationSet([
    GeneralizedOperation(1, 1, 0, "match";
        applicability=APPLICABILITY_EQUAL),
    GeneralizedOperation(2, 1, 0.25, "digraph";
        restrictions=["ph" => "f"]),
])
bytes = operation_set_bytes(grammar; format=:protobuf_v1)
snapshot = operation_set_snapshot(bytes; format=:protobuf_v1)
try
    operations = collect(snapshot)
    restored = GeneralizedOperationSet(snapshot)
    @assert length(restored) == length(operations)
finally
    close(snapshot)
end
```

The snapshot retains native operation data; iteration copies names and
restriction pairs into Julia values. `SerializedByteRestriction` preserves
native raw-byte pairs, including non-UTF-8 bytes. Re-encode such a snapshot
with `operation_set_bytes(snapshot; format=...)`. Converting it to a Unicode
`GeneralizedOperationSet` rejects raw-byte pairs because they have no Unicode
scalar meaning. Each call accepts `max_payload_bytes`, `max_operations`,
`max_operation_name_bytes`, `max_restriction_pairs_per_operation`,
`max_total_restriction_pairs`, and `max_restriction_text_bytes`. Native decoders
check complete input, version, graph/envelope structure, gzip integrity, and
those ceilings before publishing a snapshot.

## Limits and ownership

Every operation accepts `max_terms`, `max_term_bytes`,
`max_total_term_bytes`, and `max_payload_bytes`. The count applies to input
records before dictionary deduplication. Byte ceilings apply to UTF-8 bytes,
not Unicode scalar counts. `max_payload_bytes` bounds encoded input/output;
for gzip decoding it also bounds the inflated binary message. The payload
ceiling must be at least eight bytes.

Before reconstruction, native preflight validates complete bincode lengths,
DAT term boundaries, Protobuf graph shape, UTF-8 terminal paths, suffix source
counts, and gzip integrity. Reachable graph cycles, mismatched counts,
truncation, and trailing compressed bytes fail without publishing a snapshot.
The returned byte vector is independently owned. A decoded snapshot owns its
terms after the input bytes are released. Iteration copies each Julia `String`
under a lock; those strings remain valid after `close(snapshot)`. Close the
snapshot deterministically when finished.

The C bridge's format IDs map one-to-one to the selected native Rust
serializers. Focused tests compare encoded bytes against those serializers in
the same build and exercise malformed data and resource ceilings. The suite
also decodes 13 committed byte fixtures produced by the tagged RC.5 Rust
serializers: seven dictionary formats, two suffix formats, and four operation-set
formats. Cross-build byte compatibility is promised only where the underlying
format contract states it; native Bincode and optimized Protobuf remain
versioned native data.
