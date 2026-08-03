# OperationSet Binary Persistence and Gzip Gate

[← Scientific ledger index](README.md)

**Date:** 2026-08-02
**Status:** complete
**Scope:** bincode, protobuf, and optional outer gzip for complete generalized operation sets

## Question and decision boundary

Bincode and protobuf are compact binary encodings, but neither is a compression algorithm.
The experiment asks whether gzip finds enough repeated structure to justify its CPU cost. It
does not ask whether gzip is a third format: decompression must yield exactly one supported
bincode envelope or protobuf message.

The policy decision is conservative:

- keep raw bincode and raw protobuf as the ordinary choices;
- keep gzip explicit and optional;
- do not enable gzip by default unless representative application artifacts demonstrate a
  worthwhile size/latency trade-off.

## Hypotheses

| ID | Hypothesis | Measurement | Decision rule |
|---|---|---|---|
| H1 | Both inner formats retain the complete semantic model. | Unit, integration, example, and property round trips; execution correspondence; exact weight bits. | Zero differences and zero decoder panics. |
| H2 | Gzip can exploit repetition that remains after binary encoding. | Raw and gzip byte counts for a 256-rule repetitive table. | At least 80% smaller for both formats. |
| H3 | Gzip is not uniformly beneficial. | Standard four-rule set size and encode/decode latency. | Any size regression or order-of-magnitude latency increase rejects default-on compression. |
| H4 | Resource admission precedes dangerous allocation/work. | Wire preflight properties, decompressed limit tests, Rocq/Dafny/Verus/SMT/TLA+ obligations. | Every tool passes; over-limit inputs cannot reach semantic admission. |

## Method

The benchmark corpus contains:

1. `standard`: the four ordinary Levenshtein operations.
2. `repetitive_256`: 256 equal-arity operations whose names share a long prefix.

The second corpus is intentionally compressible. It establishes whether gzip can help, not a
claim about every production operation table. The first corpus exposes fixed gzip overhead.

Command:

```text
cargo bench --features protobuf,compression \
  --bench operation_set_persistence_benchmarks -- --quick --noplot
```

Criterion used optimized code, 20 samples, and a two-second measurement window in the committed
benchmark configuration. Times below are interval centers from this quick gate and should be
re-measured on deployment hardware before selecting a policy.

## Results

### Encoded size

| Corpus | Bincode | Protobuf | Bincode + gzip | Protobuf + gzip |
|---|---:|---:|---:|---:|
| standard | 199 B | 92 B | 104 B | 94 B |
| repetitive_256 | 17,180 B | 12,803 B | 736 B | 705 B |

For the repetitive table, gzip reduced bincode by 95.7% and protobuf by 94.5%, so H2 passed.
For the standard table, gzip reduced bincode by 47.7% but made protobuf 2.2% larger.

### Encoding latency

| Corpus | Bincode | Protobuf | Bincode + gzip | Protobuf + gzip |
|---|---:|---:|---:|---:|
| standard | 191.6 ns | 237.2 ns | 10.26 µs | 8.17 µs |
| repetitive_256 | 4.33 µs | 17.14 µs | 72.67 µs | 67.20 µs |

On the standard set, the gzip compositions were approximately 53.5× and 34.4× slower than raw
bincode and protobuf. On the repetitive table, the multipliers fell to approximately 16.8×
and 3.9× because inner serialization became a larger share of the work.

### Decoding latency

| Corpus | Bincode | Protobuf | Bincode + gzip | Protobuf + gzip |
|---|---:|---:|---:|---:|
| standard | 341.6 ns | 472.2 ns | 4.49 µs | 4.60 µs |
| repetitive_256 | 25.09 µs | 36.00 µs | 37.56 µs | 43.78 µs |

Standard-set gzip decoding was approximately 13.1× slower for bincode and 9.7× slower for
protobuf. On the repetitive table, the multipliers were approximately 1.50× and 1.22×.

## Verification evidence

- `tests/operation_set_serialization.rs`: deterministic bincode, corruption/limit, execution,
  arbitrary-input properties, private-wire version-1 compatibility, and a compile-fail public
  API assertion excluding generic Serde.
- `tests/operation_set_protobuf.rs`: canonical protobuf, wire preflight, exact bits,
  unknown-field compatibility, execution, and arbitrary-input properties.
- `tests/operation_set_gzip.rs`: inner-byte correspondence, checksum/trailing-member rejection,
  decompressed limits, demonstrated compression, and property round trips.
- `examples/operation_set_persistence.rs`: compile-checked public API for both inner formats.
- `OperationSetSerialization.v`: assumption-free semantic, preflight, exact-bit, and gzip
  theorems.
- `DyckSerialization.dfy`: 14 obligations verified, 0 errors.
- `dyck_serialization.rs` (Verus): 10 obligations verified, 0 errors.
- `dyck_serialization.smt2`: 10 independent queries UNSAT in both Z3 and cvc5.
- `OperationSetPortableDecode.tla`: 709,936 distinct states checked; no invariant violation.

## Verdict

H1, H2, and H4 passed. H3 also passed in the intended negative sense: gzip is demonstrably not
uniformly beneficial. The committed API therefore keeps it an explicit outer wrapper. Raw
protobuf is already smaller than gzip-wrapped protobuf for the standard set, while highly
repetitive tables receive very large size reductions at measurable CPU cost.

Only the private bincode wire structs implement Serde. Public operation-model types expose no
generic serialization trait, so downstream JSON or TOML encoding is a compile-time error rather
than merely an undocumented possibility. The private wire layout is byte-for-byte compatible
with version 1 of the envelope.

This evidence does not justify random-access gzip dictionaries. Large dictionaries that need
partial loading should use a separately designed chunked/container index rather than treating
whole-stream gzip as transparent.
