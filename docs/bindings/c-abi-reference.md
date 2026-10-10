# The `llev_*` C ABI: functions, results, and lifetimes

This is the normative reference for liblevenshtein's project-owned C surface:
all **152 exported `llev_*` functions**, grouped by their header signature
families, preconditions, status and ownership contracts, and concurrency
rules. The [public header](../../include/liblevenshtein.h) gives each exact
prototype and per-call return contract; this guide explains the shared
protocols and the differences that matter to consumers. It is the **project
layer above the family canon**: everything about the
two-word `VtResource`, the base retain/release/`query_interface` protocol, and
the `vt.dictionary.v1` interface this ABI consumes is specified once in the
[interop ABI reference](https://github.com/vinary-tree/vinary-tree-interop/blob/master/docs/abi-reference.md) and
cited here rather than restated.

Companions: [binding corpus hub](README.md) ·
[resource-consumer internals](resource-consumer.md) (the safe-Rust layer under
this surface) · [snapshot-semantics theory](../theory/snapshot-semantics.md) ·
[binding trust model](../security/binding-trust-model.md) ·
[WASM topology](wasm-topology.md).

---

## 1. Terms

Interop-level terms (resource, vtable, retain/release, snapshot, paging) are
defined in the [canon's terms table](https://github.com/vinary-tree/vinary-tree-interop/blob/master/docs/abi-reference.md#1-terms).
The project-level terms this document adds:

| Term | Definition |
|---|---|
| transducer | An opaque `LlevTransducer`: a Levenshtein-automaton configuration (algorithm choice) holding one owned retain of a dictionary resource. Constructing it never copies the dictionary. |
| query cache | An opaque `LlevQueryCache`: an exclusive, synchronization-free, hard-bounded complete-result memo retaining one transducer. TinyLFU estimates reuse for admission; SIEVE selects victims. |
| cursor | An opaque `LlevQueryCursor`: one lazy query. It owns the retained immutable snapshot captured at query start plus all storage later exposed through batch leases. |
| batch | One bounded group of match descriptors transferred per boundary crossing (default capacity `LLEV_DEFAULT_MATCH_BATCH` = 256). |
| lease | The borrow state of a batch: from a successful `next_batch` until the matching `release_batch`, the descriptor array and term arenas belong to the caller and the cursor refuses to advance, reduce, or close. |
| generation | The `uint64_t` tag identifying the live lease. Strictly increasing per cursor, never zero, and required *exactly* at release — a stale generation cannot release a newer lease. |
| arena | Cursor-owned contiguous storage (`byte` and `u64` arenas) holding every term of the current batch back-to-back; descriptors point into it. Cleared-but-retained between batches, so steady-state batches allocate nothing. |
| reducer | A caller-supplied `LlevBatchReducer` callback invoked once per batch with borrowed descriptors — the allocation-minimizing expert path for managed languages. |
| unit domain | Which value space the dictionary's labels inhabit: bytes, Unicode scalars, or opaque `u64` tokens (`VtUnitDomain`, canon [§ 6.1](https://github.com/vinary-tree/vinary-tree-interop/blob/master/docs/abi-reference.md#61-vtunitdomain-and-vtvaluedomain)). |
| generalized automaton | An immutable runtime operation set with exact decimal cost scaling. Its online state retains a bounded finite-lookback row ring for one Unicode source. |
| universal automaton | An immutable standard, transposition, or merge-and-split specialization with an optional owned directional substitution policy. |
| online automaton | An exclusive state handle bound to one source and advanced by one domain-native target unit per call. It is independent of dictionary cursors. |
| phonetic result array | One caller-owned C array whose nested strings and detail arrays are freed together by its matching family-specific `*_free` function; never free individual nested members. |
| phonetic stream | An exclusive scanner or rewrite handle that buffers or delays input until `finish`; it is distinct from an immutable, reusable matcher configuration. |
| AOT bytes | Length-bearing, owned compiled phonetic bytes produced only when the phonetic AOT build feature is enabled. They are a versioned native format, not a stable cross-version wire ABI. |
| WallBreaker | An immutable, finite Unicode substring matcher with its own owned terms and result-cursor lease protocol; it is not a `vt.dictionary.v1` adapter. |

These opaque handles are owning C pointers, not generation-tagged identifiers.
A successful `free` consumes a handle; the caller must not pass that pointer to
another function or free it again. Such use after free is undefined behavior,
not a recoverable `LlevStatus`, even if a later allocation reuses the address.
This differs from a live cursor with an invalid or stale **batch lease
generation**, which is checked and returns the documented status. It also
differs from a retained resource or captured snapshot, whose independent
lifetime can outlast the handle from which it was obtained.

Throughout, $`n`$ is the number of matches a query yields, $`B`$ the batch
capacity, $`q`$ the query, $`k`$ the maximum edit distance, and
$`\deg(v)`$ a dictionary node's out-degree.

---

## 2. Where this surface sits

![Three-layer architecture: language facades over the four project C ABIs over the shared vinary-tree-interop resource plane, governed by bindings/api.json.](../diagrams/bindings/three-layer-architecture.svg)

The shared resource plane is a small capability graph: a two-word
`VtResource` discovers versioned dictionary, graph, snapshot-identity, and
WFST vtables; borrowed edges and arcs remain provider-owned for the duration
specified by their interface.

![Class diagram of the vinary-tree-interop ABI: VtResource and its base vtable negotiate dictionary, visit, graph, snapshot-identity, and scalar-WFST capability vtables plus their borrowed value types.](../diagrams/bindings/vt-structs-class.svg)

The 152 functions divide into eight groups:

| Group | Count | Functions |
|---|---|---|
| [Introspection](#4-introspection-4) | 4 | `llev_abi_version` · `llev_api_revision` · `llev_build_features` · `llev_last_error_message` |
| [Strings (legacy)](#5-string-helpers-3-legacy) | 3 | `llev_string_free` · `llev_string_array_free` · `llev_string_dup` |
| [Distances](#6-distance-functions-42) | 42 | Seven families × Unicode-scalar, byte, and u64-token domains × exact and thresholded calls |
| [Transducer + cursor](#7-transducer-and-cursor-11) | 11 | `llev_transducer_new` · `llev_transducer_snapshot` · `llev_transducer_free` · `llev_transducer_unit_domain` · `llev_transducer_query_utf8` · `llev_transducer_query_bytes` · `llev_transducer_query_u64` · `llev_query_cursor_next_batch` · `llev_query_cursor_release_batch` · `llev_query_cursor_reduce` · `llev_query_cursor_free` |
| [Bounded query cache](#7a-bounded-query-cache-8) | 8 | `llev_query_cache_new` · `llev_query_cache_clear` · `llev_query_cache_reset_stats` · `llev_query_cache_stats` · `llev_query_cache_free` · `llev_query_cache_query_utf8` · `llev_query_cache_query_bytes` · `llev_query_cache_query_u64` |
| [Standalone automata](#7b-standalone-automata-14) | 14 | Generalized and universal configuration, complete evaluation, online construction, advance, observation, and free functions |
| [WallBreaker](#7c-wallbreaker-8) | 8 | Owned matcher construction/query/free, leased cursor next/release/cancel/free, and Unicode-aware pattern splitting. |
| [Phonetic](#8-phonetic-surface-62) | 62 | Pattern and rule compilation, trusted file loading, compiled bytes, distance and syllable analysis, word/character/token grep, normalized dictionaries, incremental rewriting, expansion, and IPA feature relations. See the [family inventory](#family-inventory-and-feature-gates). |

Headers: [`include/liblevenshtein.h`](../../include/liblevenshtein.h)
(normative prototypes; common signature shapes are abbreviated below) over
[`include/liblevenshtein_abi.h`](../../include/liblevenshtein_abi.h)
(generated constants, enums, and POD types — regenerate via
`scripts/generate-bindings.py`, never edit numeric values by hand) over the
interop header. C++ consumers get the RAII wrapper
[`include/liblevenshtein.hpp`](../../include/liblevenshtein.hpp) (C++20:
`transducer` / `query_cursor` / `batch` types whose destructors settle leases
and handles automatically).

**Retired surface.** The pre-resource-ABI dictionary API — `LlevIndex`,
`llev_index_new`, `llev_index_insert`, `llev_index_query`,
`llev_index_free`, and every other `llev_index_*` symbol — is **removed**,
not deprecated. Dictionary construction and CRUD belong to libdictenstein
(`bindings/api.json` lists the dictionary types under
`forbiddenOwnedObjects`, and `scripts/check-bindings.py` rejects facades that
mention `llev_index_`). A module-level doc example still demonstrating the
retired API is ledgered as finding LLEV-B1 in the
[findings ledger](FINDINGS_LEDGER.md) and is being rewritten in this wave
(W3) to the `llev_transducer_new` flow shown in [§ 9](#9-a-complete-c-consumer).

Additive ABI evolution extends a size-delimited vtable under the same interface
identity. A breaking semantic or layout change receives a new identity so old
and new providers can coexist and be negotiated explicitly.

![ABI evolution timeline: additive fields preserve an interface identity and old struct size, while breaking changes fork a new identity that can coexist through query-interface negotiation.](../diagrams/bindings/abi-evolution-timeline.svg)

---

## 3. The status currency

Every fallible function returns `LlevStatus`, the project-level superset of
the interop `VtStatus`. All 13 values, pinned in
[`bindings/api.json`](../../bindings/api.json) and generated into
`liblevenshtein_abi.h` and `src/ffi/generated.rs`:

| Value | Name | Meaning | Returned when |
|---|---|---|---|
| 0 | `LLEV_STATUS_OK` | Success; every advertised output was written. | Any fallible function. |
| 1 | `LLEV_STATUS_END` | A stream finished. **A success value**, not an error: `boundary()` clears the error slot for it exactly as for `Ok`. | `llev_query_cursor_next_batch` on exhaustion; a reducer callback returns it to stop early. |
| 2 | `LLEV_STATUS_INVALID_ARGUMENT` | An argument was outside its domain. | Unknown algorithm/order/kind/variant values; malformed generalized operations; invalid scalars or policy units; a zero `max_matches`; a release with a stale, zero, or absent lease generation; pattern/rule-set parse failures. |
| 3 | `LLEV_STATUS_INVALID_UTF8` | A length-bearing text buffer was not valid UTF-8. | `utf8()` validation in every text-accepting entry point. |
| 4 | `LLEV_STATUS_NULL_POINTER` | A required pointer was NULL. | Argument preflight in every entry point; also mapped from a provider's `VtStatus` `NullPointer`. |
| 5 | `LLEV_STATUS_PANIC` | A Rust panic was caught at the boundary. The panic message is in the thread-local error slot. | Only from `boundary()`'s `catch_unwind` arm — no other code path constructs it. |
| 6 | `LLEV_STATUS_UNSUPPORTED` | The operation is not available in this build or from this provider. | Phonetic entry points without the `bindings-phonetic` feature; ordered streaming on byte/u64 domains; a `Bytes` value-domain provider; cached queries when snapshot identity is absent; mapped from provider `Unsupported`. |
| 7 | `LLEV_STATUS_IO_ERROR` | A storage-backed provider failed on I/O. | Mapped verbatim from provider `IoError`. |
| 8 | `LLEV_STATUS_CLOSED` | The provider reports its resource already torn down. | Mapped verbatim from provider `Closed`. |
| 9 | `LLEV_STATUS_LIMIT_EXCEEDED` | A provider or standalone-automaton resource limit was hit. | Mapped verbatim from provider `LimitExceeded`; source, target, retained-cell, step-work, operation, restriction, scaling, or allocation ceilings. |
| 10 | `LLEV_STATUS_PROVIDER_ERROR` | The dictionary provider misbehaved: an error status with no better mapping, malformed output despite `Ok`, a missing/incompatible interface, or an illegal `End` from an interface callback. | The consumer's whole-class response to invalid provider output (see the [trust model](../security/binding-trust-model.md)). |
| 11 | `LLEV_STATUS_BATCH_IN_USE` | A live lease blocks the requested state change. The refused object is **unchanged and still owned by the caller**. | `next_batch`/`reduce`/`free` on a leased cursor. |
| 12 | `LLEV_STATUS_DOMAIN_MISMATCH` | An entry point does not match its resource or policy unit domain. | `query_utf8` on a byte/u64 dictionary, or universal evaluation in a domain different from its owned substitution policy. |

### 3.1 Mapping from the interop `VtStatus`

Provider callbacks answer with the nine-value interop `VtStatus`
([canon § 3](https://github.com/vinary-tree/vinary-tree-interop/blob/master/docs/abi-reference.md#3-vtstatus--the-one-error-currency))
— but **on the Rust side the wire type is a raw `u32`**, not the enum. The
family's status wire rule (landed with LLEV-B6's fix, commit `e42485c`):
producers encode with `VtStatus::to_raw`, consumers decode with
`VtStatus::from_raw` at a single chokepoint (`status()` in
`src/bindings.rs`) **before any enum-typed use**, and an out-of-range value
decodes to `None` — treated as provider *misbehavior*
(`InvalidProviderOutput`, surfacing as `PROVIDER_ERROR`), never as undefined
behavior. The C header is unchanged by this rule: C enums are
integer-typed, so the ABI is byte-identical. Once decoded,
`map_binding_error` in `src/ffi/index.rs` lifts a recorded provider failure
into `LlevStatus`:

| Provider `VtStatus` | Resulting `LlevStatus` | Note |
|---|---|---|
| `InvalidArgument` (2) | `INVALID_ARGUMENT` (2) | preserved |
| `NullPointer` (3) | `NULL_POINTER` (4) | preserved in meaning; **renumbered** — never forward raw discriminants |
| `Unsupported` (4) | `UNSUPPORTED` (6) | preserved |
| `IoError` (5) | `IO_ERROR` (7) | preserved |
| `Closed` (6) | `CLOSED` (8) | preserved |
| `LimitExceeded` (7) | `LIMIT_EXCEEDED` (9) | preserved |
| `Ok` (0) | — | never an error; `Ok` cannot reach the mapper |
| `End` (1), `ProviderError` (8) | `PROVIDER_ERROR` (10) | `End` is **illegal from interface callbacks** (family contract pin F5) |
| any raw value outside 0..=8 | `PROVIDER_ERROR` (10) | refused at decode (`from_raw` → `None` → `InvalidProviderOutput`): unknown and future statuses degrade, they are never trusted — and never materialized as a Rust enum |

Structural failures detected by the consumer itself — null resource words map
to `NULL_POINTER`; an incompatible base ABI, a missing or incomplete
dictionary interface, and malformed provider output all map to
`PROVIDER_ERROR`; a unit-domain mismatch maps to `DOMAIN_MISMATCH`; the
reserved `Bytes` value domain maps to `UNSUPPORTED`.

The mapping's laws — totality over all inputs, `Ok`/`End` handling,
no-swallowing (an error never maps to a success), and
`Panic`-only-from-`catch_unwind` — receive their formal home in this wave
(W3) as the dual-solver SMT artifact
`docs/verification/smt/llev_status_mapping.smt2` (FV obligation #6, invariant
IDs LLEV-STAT-1 through LLEV-STAT-6, registered in
[`docs/verification/ABI_INVARIANTS.tsv`](../../docs/verification/ABI_INVARIANTS.tsv)
as its rows land). Until the rows appear there, the
[findings ledger](FINDINGS_LEDGER.md) tracks the obligation.

### 3.2 The error-message contract

`llev_last_error_message` returns a **thread-local, library-owned,
NUL-terminated** UTF-8 message:

- populated by every failing call on the same thread (including the caught
  panic's message);
- cleared (set to the empty string) by every call that returns `OK` or `END`;
- NUL bytes inside a message are sanitized to `\0` escapes;
- valid until the next `llev_*` call on the same thread; **never freed by the
  caller**.

Every fallible entry point runs inside `boundary()`
(`src/ffi/index.rs:124-148`): `catch_unwind` wraps the operation, success
clears the slot, failure stores the message, and a caught panic is downcast
to its message and surfaced as `PANIC`. No unwinding ever crosses this ABI —
the family containment law
([security model § 3](https://github.com/vinary-tree/vinary-tree-interop/blob/master/docs/security-model.md#3-the-panic-and-exception-containment-law)).

---

## 4. Introspection (4)

```c
uint32_t llev_abi_version(void);
uint32_t llev_api_revision(void);
uint64_t llev_build_features(void);
const char* llev_last_error_message(void);
```

| Function | Returns | Contract |
|---|---|---|
| `llev_abi_version` | `LLEV_ABI_VERSION` = 1 | The project ABI generation. A facade built for generation $`g`$ must refuse a library reporting a different generation. |
| `llev_api_revision` | `LLEV_API_REVISION` = 9 | The additive revision within the ABI generation ([evolution policy § 1](https://github.com/vinary-tree/vinary-tree-interop/blob/master/docs/abi-evolution.md#1-the-four-version-counters)). A facade needing revision $`r`$ refuses a library reporting less than $`r`$. Revision 5 adds generalized/universal automata; revision 6 adds the [finite Unicode WallBreaker surface](wallbreaker-unicode.md); revision 7 adds the Hamming, indel, and affine-gap standalone distance families; revision 8 adds the expanded [phonetic surface](#8-phonetic-surface-62); revision 9 adds [specialized Unicode traversals](#7d-specialized-unicode-traversals-7). |
| `llev_build_features` | bitset | `LLEV_BUILD_FEATURE_CORE` (1) is always set. `LLEV_BUILD_FEATURE_PHONETIC` (2) reports `bindings-phonetic`; `LLEV_BUILD_FEATURE_PHONETIC_AOT` (4) reports the additional `serialization` capability. Probe these before using optional phonetic operations. The AOT bit is set only when both features are compiled in. |
| `llev_last_error_message` | borrowed `const char*` | § 3.2. Never NULL; empty string when the last call on this thread succeeded. |

*Preconditions:* none — these are total. *Thread safety:* fully
thread-safe; the message pointer is per-thread state. *Complexity:*
$`\mathcal{O}(1)`$. *Statuses:* none returned (non-`LlevStatus` signatures);
these functions cannot fail.

---

## 5. String helpers (3, legacy)

```c
void llev_string_free(char* value);
void llev_string_array_free(char** values, size_t len);
char* llev_string_dup(const char* value);
```

Retained **only** for their independent allocation ABI (a stable
malloc/free-pair surface some facades use for round-tripping C strings). They
are unrelated to the cursor path — query terms are *borrowed* from leased
batches and must never be passed to `llev_string_free`.

| Function | Ownership | Statuses / failure signal |
|---|---|---|
| `llev_string_free` | Consumes a string previously *returned by this library's allocating helpers*; NULL is a safe no-op. Double-free is undefined behavior. | none (void) |
| `llev_string_array_free` | Consumes an array of `len` such strings plus the array itself; NULL array is a no-op; NULL elements are skipped. | none (void) |
| `llev_string_dup` | Returns a fresh NUL-terminated copy the caller must settle with `llev_string_free`. | returns NULL on NULL input, interior NUL, invalid UTF-8, or allocation failure |

*Thread safety:* thread-safe (pure allocation). *Complexity:*
$`\mathcal{O}(\lvert s \rvert)`$.

---

## 6. Distance functions (42)

```c
size_t llev_distance(const char* source, size_t source_len,
                     const char* target, size_t target_len);
size_t llev_distance_threshold(const char* source, size_t source_len,
                               const char* target, size_t target_len,
                               size_t threshold);
size_t llev_damerau_distance(const char* source, size_t source_len,
                             const char* target, size_t target_len);
size_t llev_damerau_distance_threshold(const char* source, size_t source_len,
                                       const char* target, size_t target_len,
                                       size_t threshold);
size_t llev_true_damerau_distance(const char* source, size_t source_len,
                                  const char* target, size_t target_len);
size_t llev_true_damerau_distance_threshold(const char* source,
                                            size_t source_len,
                                            const char* target,
                                            size_t target_len,
                                            size_t threshold);
size_t llev_merge_and_split_distance(const char* source, size_t source_len,
                                     const char* target, size_t target_len);
size_t llev_merge_and_split_distance_threshold(const char* source,
                                               size_t source_len,
                                               const char* target,
                                               size_t target_len,
                                               size_t threshold);

size_t llev_distance_bytes(const uint8_t* source, size_t source_len,
                           const uint8_t* target, size_t target_len);
size_t llev_distance_bytes_threshold(const uint8_t* source,
                                     size_t source_len,
                                     const uint8_t* target,
                                     size_t target_len,
                                     size_t threshold);
size_t llev_distance_u64(const uint64_t* source, size_t source_len,
                         const uint64_t* target, size_t target_len);
size_t llev_distance_u64_threshold(const uint64_t* source,
                                   size_t source_len,
                                   const uint64_t* target,
                                   size_t target_len,
                                   size_t threshold);

size_t llev_damerau_distance_bytes(const uint8_t* source,
                                   size_t source_len,
                                   const uint8_t* target,
                                   size_t target_len);
size_t llev_damerau_distance_bytes_threshold(const uint8_t* source,
                                             size_t source_len,
                                             const uint8_t* target,
                                             size_t target_len,
                                             size_t threshold);
size_t llev_damerau_distance_u64(const uint64_t* source,
                                 size_t source_len,
                                 const uint64_t* target,
                                 size_t target_len);
size_t llev_damerau_distance_u64_threshold(const uint64_t* source,
                                           size_t source_len,
                                           const uint64_t* target,
                                           size_t target_len,
                                           size_t threshold);

size_t llev_true_damerau_distance_bytes(const uint8_t* source,
                                        size_t source_len,
                                        const uint8_t* target,
                                        size_t target_len);
size_t llev_true_damerau_distance_bytes_threshold(const uint8_t* source,
                                                  size_t source_len,
                                                  const uint8_t* target,
                                                  size_t target_len,
                                                  size_t threshold);
size_t llev_true_damerau_distance_u64(const uint64_t* source,
                                      size_t source_len,
                                      const uint64_t* target,
                                      size_t target_len);
size_t llev_true_damerau_distance_u64_threshold(const uint64_t* source,
                                                size_t source_len,
                                                const uint64_t* target,
                                                size_t target_len,
                                                size_t threshold);

size_t llev_merge_and_split_distance_bytes(const uint8_t* source,
                                           size_t source_len,
                                           const uint8_t* target,
                                           size_t target_len);
size_t llev_merge_and_split_distance_bytes_threshold(const uint8_t* source,
                                                     size_t source_len,
                                                     const uint8_t* target,
                                                     size_t target_len,
                                                     size_t threshold);
size_t llev_merge_and_split_distance_u64(const uint64_t* source,
                                         size_t source_len,
                                         const uint64_t* target,
                                         size_t target_len);
size_t llev_merge_and_split_distance_u64_threshold(const uint64_t* source,
                                                   size_t source_len,
                                                   const uint64_t* target,
                                                   size_t target_len,
                                                   size_t threshold);
```

API revision 7 adds Hamming, indel, and affine-gap families in each domain,
with exact and thresholded forms (18 functions). Their exact spelling and
signatures are in the [public header](../../include/liblevenshtein.h):

| Family | Unsuffixed Unicode forms | Raw-byte and u64-token forms | Extra arguments |
|---|---|---|---|
| Hamming | `llev_hamming_distance`, `llev_hamming_distance_threshold` | `llev_hamming_distance_bytes`, `llev_hamming_distance_bytes_threshold`, `llev_hamming_distance_u64`, `llev_hamming_distance_u64_threshold` | threshold only in bounded form |
| Indel | `llev_indel_distance`, `llev_indel_distance_threshold` | `llev_indel_distance_bytes`, `llev_indel_distance_bytes_threshold`, `llev_indel_distance_u64`, `llev_indel_distance_u64_threshold` | threshold only in bounded form |
| Affine gap | `llev_affine_gap_distance`, `llev_affine_gap_distance_threshold` | `llev_affine_gap_distance_bytes`, `llev_affine_gap_distance_bytes_threshold`, `llev_affine_gap_distance_u64`, `llev_affine_gap_distance_u64_threshold` | `gap_open`, `gap_extend`, `substitution`; bounded form adds `threshold` last |

All 18 forms take the same four leading arguments as the corresponding
domain's existing calls: source pointer/length and target pointer/length.
Affine costs are nonnegative integers on one caller-chosen scale. A gap run of
length $`k`$ costs $`gap\_open + k\cdot gap\_extend`$; a substitution costs
`substitution`. Hamming is defined only for equal unit counts; indel allows
insertions and deletions but no one-step substitution. The affine kernel is
Gotoh dynamic programming; its bounded entry point currently evaluates the
exact score before comparing with `threshold`. Indel uses a bounded band.

These are pure functions over two length-bearing buffers. Unsuffixed functions
decode valid UTF-8 and count **Unicode scalar values**, not bytes. `_bytes`
functions accept arbitrary binary data. `_u64` functions compare aligned
`uint64_t` application tokens by value. They do not use `LlevStatus`; failure
is sentinel-coded so the hot path stays a single integer return:

| Sentinel | Meaning |
|---|---|
| `SIZE_MAX` | a NULL pointer with nonzero length, invalid UTF-8 in an unsuffixed call, or a misaligned u64 buffer |
| `SIZE_MAX - 1` | (threshold variants only) the exact distance exceeds `threshold` |
| `SIZE_MAX - 2` | Hamming lengths differ, or affine has no representable finite result (including arithmetic overflow or collision with the sentinel range) |

All three sentinels are reserved and never represent a successful distance.
Check for them before interpreting an exact result or comparing with a
threshold; affine scores can exceed the unit counts when costs exceed one.

| Function | Metric | Semantics |
|---|---|---|
| `llev_distance`(`_threshold`) | Levenshtein | insert · delete · substitute |
| `llev_damerau_distance`(`_threshold`) | **OSA** (optimal string alignment, "restricted Damerau") | adds adjacent transposition, but no substring may be edited twice — kept under its legacy name for ABI stability |
| `llev_true_damerau_distance`(`_threshold`) | unrestricted Damerau–Levenshtein | true metric with transposition; e.g. for `CA` → `ABC`: OSA gives 3, true Damerau gives 2 |
| `llev_merge_and_split_distance`(`_threshold`) | merge-and-split Levenshtein | adds symmetric one-to-two split and two-to-one merge operations at unit cost |
| `llev_hamming_distance`(`_threshold`) | Hamming on equal-length inputs | counts unequal unit positions; unequal lengths produce `SIZE_MAX - 2` |
| `llev_indel_distance`(`_threshold`) | insertion/deletion distance | no substitution operation; one replacement costs two edits |
| `llev_affine_gap_distance`(`_threshold`) | affine-gap weighted alignment | configurable substitution and gap-run costs; unrepresentable result produces `SIZE_MAX - 2` |

Every row also has `_bytes`, `_bytes_threshold`, `_u64`, and
`_u64_threshold` forms. `_threshold` is always the final suffix. API revision 4
added domain-explicit forms for the original four families; API revision 7
added three more families without changing the original symbols. The complete
recurrence and binding mapping
are in the [domain-preserving distance design](distance-domains.md).

*Preconditions:* each buffer is valid for its unit count when nonzero; u64
buffers are naturally aligned.
*Thread safety:* fully thread-safe and lock-free (no shared state; these do
not touch the error slot). *Complexity:* worst case
$`\mathcal{O}(\lvert s \rvert \cdot \lvert t \rvert)`$; the unbounded
Unicode Levenshtein path dispatches to Myers' bit-parallel algorithm for short
ASCII inputs ($`\le 64`$ bytes) and runtime SIMD lanes when applicable; byte Levenshtein also
uses Myers when its shorter operand fits one word. Standard, OSA, and
merge/split threshold variants run a banded dynamic program touching
$`\mathcal{O}\bigl((2k+1) \cdot \min(\lvert s \rvert, \lvert t \rvert)\bigr)`$
cells for threshold $`k`$. Unrestricted Damerau retains its full historical
matrix after its constant-time length lower-bound rejection. Hamming is
linear; affine-gap exact and bounded forms are quadratic in the worst case.

---

## 7. Transducer and cursor (11)

The heart of the ABI. The full object flow, end to end:

![Resource handoff sequence: obtain a dictionary resource, retain and negotiate in llev_transducer_new, capture the query-start snapshot, and let the cursor outlive the transducer and the source handle.](../diagrams/bindings/resource-handoff-sequence.svg)

### 7.1 `llev_transducer_new`

```c
LlevStatus llev_transducer_new(const VtResource* dictionary,
                               uint32_t algorithm,
                               LlevTransducer** out_transducer);
```

Retains a live dictionary resource and constructs an automaton configuration
around it. The resource is **borrowed at the call and retained inside**: the
function calls the provider's `retain` before any validation and releases it
again on every validation-failure path, so ownership of the caller's own
retain never moves (contract row
`ffi-borrowed-resource-retain-validate-release` in
[`UNSAFE_ABI_CONTRACTS.tsv`](../../docs/verification/UNSAFE_ABI_CONTRACTS.tsv)).
Validation enforces the base handshake (`struct_size`, `abi_version` = 1,
`reserved` = 0, all three base ops present), negotiates `vt.dictionary.v1`
at `minimum_version` = 1, and requires the interface ops the consumer needs
(`snapshot`, `root`, `node_is_final`, `node_edges`; `node_value_u64` exactly
when the value domain is `OPTIONAL_U64`).

![Interface-negotiation activity: the consumer validates and retains a copied VtResource, invokes the provider across the foreign trust boundary, then validates the returned size-delimited vtable before constructing a transducer or releasing on failure.](../diagrams/bindings/interface-negotiation-activity.svg)

- **Preconditions:** `dictionary` and `out_transducer` non-NULL; the resource
  obeys the interop contract for the whole life of the transducer.
- **Statuses:** `OK` · `NULL_POINTER` (either pointer NULL, or provider
  `NullPointer`) · `INVALID_ARGUMENT` (unknown `algorithm` value, or provider
  `InvalidArgument`) · `UNSUPPORTED` (a `Bytes`-value-domain provider, or
  provider `Unsupported`) · `IO_ERROR` / `CLOSED` / `LIMIT_EXCEEDED`
  (provider verbatim) · `PROVIDER_ERROR` (null resource vtable output,
  incompatible base ABI, missing/incomplete dictionary interface, or any
  malformed negotiation output) · `PANIC`.
- **Ownership:** on `OK`, `*out_transducer` is a caller-owned handle settled
  by exactly one `llev_transducer_free`. On failure, `*out_transducer` is
  untouched.
- **Algorithms:** `LLEV_ALGORITHM_STANDARD` (0) · `TRANSPOSITION` (1, OSA
  semantics) · `MERGE_AND_SPLIT` (2) · `DAMERAU_LEVENSHTEIN` (3,
  unrestricted).
- **Thread safety:** safe to call concurrently; the constructed transducer
  may be shared across threads for querying.
- **Complexity:** $`\mathcal{O}(1)`$ — a retain, one `query_interface`, and
  constant-size validation; the dictionary is never copied or walked.

### 7.2 `llev_transducer_snapshot`

```c
LlevStatus llev_transducer_snapshot(const LlevTransducer* transducer,
                                    LlevTransducer** out_transducer);
```

Captures the source revision visible at the call and returns a read-only
transducer pinned to that immutable provider snapshot. Every cursor created
from the returned handle shares the same validated compact graph, when the
provider implements `vt.dict.graph.v1`, or the same fallback immutable
node cache. Later mutations visible through the original transducer are not
visible through this handle. Calling the function on an already immutable
transducer is $`\mathcal{O}(1)`$.

- **Preconditions:** `transducer` and `out_transducer` are non-NULL and
  `transducer` is a live handle.
- **Statuses:** `OK` · `NULL_POINTER` · `INVALID_ARGUMENT` · `UNSUPPORTED` ·
  `IO_ERROR` · `CLOSED` · `LIMIT_EXCEEDED` (provider status mapping) ·
  `PROVIDER_ERROR` (a null or malformed snapshot, changed domains, or invalid
  optional graph) · `PANIC`.
- **Ownership:** on `OK`, `*out_transducer` is a new caller-owned handle
  settled by exactly one `llev_transducer_free`; the input remains owned by
  its caller. On failure, `*out_transducer` is untouched.
- **Thread safety:** safe to call concurrently. The returned handle is
  shareable for concurrent queries; each query still owns an independent
  cursor and fault channel.
- **Complexity:** $`\mathcal{O}(1)`$ for a memoized or already immutable
  provider snapshot. A provider's first compact-graph publication may perform
  its documented snapshot preparation outside this ABI's control; traversal
  never copies that graph per cursor.

### 7.3 `llev_transducer_free`

```c
void llev_transducer_free(LlevTransducer* transducer);
```

Releases the transducer's retain of the source resource (through the
provider's call gate, so a serialized provider never sees the final release
concurrently with a callback). NULL is a no-op. **Existing query cursors
remain valid** — each owns its own snapshot retain. Statuses: none (void);
infallible by construction. Complexity: $`\mathcal{O}(1)`$ plus the
provider's `release`.

### 7.4 `llev_transducer_unit_domain`

```c
LlevStatus llev_transducer_unit_domain(const LlevTransducer* transducer,
                                       VtUnitDomain* out_domain);
```

Reports which of the three query entry points this dictionary accepts.
**Statuses:** `OK` · `NULL_POINTER` · `PANIC`. Thread-safe (pure read).
$`\mathcal{O}(1)`$.

### 7.5 The three query starts

```c
LlevStatus llev_transducer_query_utf8(const LlevTransducer* transducer,
                                      const char* query, size_t query_len,
                                      size_t max_distance, uint32_t order,
                                      LlevQueryCursor** out_cursor);
LlevStatus llev_transducer_query_bytes(const LlevTransducer* transducer,
                                       const uint8_t* query, size_t query_len,
                                       size_t max_distance, uint32_t order,
                                       LlevQueryCursor** out_cursor);
LlevStatus llev_transducer_query_u64(const LlevTransducer* transducer,
                                     const uint64_t* query, size_t query_len,
                                     size_t max_distance, uint32_t order,
                                     LlevQueryCursor** out_cursor);
```

Each captures the provider's revision **now** — one $`\mathcal{O}(1)`$
`snapshot` callback plus one `root` read — and returns a lazy cursor pinned
to that revision forever (the query-start snapshot boundary; laws and proofs
in [snapshot semantics](../theory/snapshot-semantics.md)). Match traversal is
lazy. If the immutable provider advertises `vt.dict.graph.v1`, the first
consumer of that revision may additionally pay
$`\Theta(\lvert V\rvert + \lvert E\rvert)`$ once to validate and import its
compact graph; producer and consumer revision memos amortize that work across
later query starts.

- **Preconditions:** `transducer` and `out_cursor` non-NULL; the query buffer
  valid for its length when nonzero (`query_len` in bytes for UTF-8, elements
  otherwise; a NULL buffer with zero length is a legal empty query);
  `query_u64`'s buffer 8-byte aligned.
- **Domain gate:** the entry point must match the dictionary's unit domain
  (`DOMAIN_MISMATCH` otherwise). `query_utf8` requires `UNICODE_SCALAR`,
  `query_bytes` requires `BYTE`, `query_u64` requires `U64`.
- **Order:** `LLEV_QUERY_ORDER_TRAVERSAL` (0) streams in dictionary order
  with bounded state; `DISTANCE_THEN_TERM` (1) yields increasing distance
  (ties by term), buffering at most one distance layer — supported on the
  Unicode entry point only (`UNSUPPORTED` from bytes/u64).
- **Statuses:** `OK` · `NULL_POINTER` · `INVALID_UTF8` (`query_utf8` only) ·
  `INVALID_ARGUMENT` (unknown `order`, or provider `InvalidArgument`) ·
  `DOMAIN_MISMATCH` · `UNSUPPORTED` (ordered streaming on bytes/u64, or
  provider `Unsupported`) · `IO_ERROR` / `CLOSED` / `LIMIT_EXCEEDED`
  (provider verbatim, from the `snapshot`/`root` callbacks) ·
  `PROVIDER_ERROR` (null snapshot, snapshot that changes domains, or other
  malformed provider output) · `PANIC`.
- **Ownership:** on `OK`, `*out_cursor` is caller-owned, settled by exactly
  one successful `llev_query_cursor_free`. The cursor **may outlive** the
  transducer and the caller's own dictionary handle.
- **Thread safety:** concurrent query starts on one shared transducer are
  safe; callbacks to a non-reentrant provider serialize on its
  [call gate](resource-consumer.md#4-the-call-gate).
- **Complexity:** warm revision:
  $`\mathcal{O}(\lvert q \rvert)`$ to copy the query plus
  $`\mathcal{O}(1)`$ capture. Cold compact-graph revision:
  $`\mathcal{O}(\lvert q \rvert + \lvert V\rvert + \lvert E\rvert)`$ for the
  first validating import only. Providers without the optional graph retain
  the $`\mathcal{O}(\lvert q \rvert)`$ lazy callback path.

### 7A. Bounded query cache (8)

```c
LlevStatus llev_query_cache_new(const LlevTransducer* transducer,
                                size_t max_entries_per_order,
                                size_t max_weight_per_order,
                                LlevQueryCache** out_cache);
LlevStatus llev_query_cache_clear(LlevQueryCache* cache);
LlevStatus llev_query_cache_reset_stats(LlevQueryCache* cache);
LlevStatus llev_query_cache_stats(const LlevQueryCache* cache,
                                  LlevQueryCacheStats* out_stats);
void llev_query_cache_free(LlevQueryCache* cache);

LlevStatus llev_query_cache_query_utf8(LlevQueryCache* cache,
                                       const char* query, size_t query_len,
                                       size_t max_distance, uint32_t order,
                                       LlevQueryCursor** out_cursor);
LlevStatus llev_query_cache_query_bytes(LlevQueryCache* cache,
                                        const uint8_t* query, size_t query_len,
                                        size_t max_distance, uint32_t order,
                                        LlevQueryCursor** out_cursor);
LlevStatus llev_query_cache_query_u64(LlevQueryCache* cache,
                                      const uint64_t* query, size_t query_len,
                                      size_t max_distance, uint32_t order,
                                      LlevQueryCursor** out_cursor);
```

This API is an opt-in complete-result memo for repeated-query workloads. It
uses TinyLFU approximate-frequency admission and SIEVE victim selection under
hard entry and logical-weight bounds. Approximation changes residency only:
every miss drains the exact snapshot-consistent product walk before returning,
even if the candidate is too large or loses admission.

- **Construction:** `new` retains the transducer in $`\mathcal{O}(1)`$ time.
  Each hard limit applies independently to traversal and distance-then-term
  order shards. A zero entry or weight limit disables admission but not exact
  miss computation. On `OK`, `*out_cache` is settled by one `free`; failure
  leaves it untouched. Statuses: `OK` · `NULL_POINTER` · `PANIC`.
- **Query identity:** each query captures an immutable provider snapshot and
  requires `vt.snapshot.id.1`. A producer change clears both shards; a revision
  change clears stale residency before lookup. Missing identity is
  `UNSUPPORTED`, never a best-effort stale cache. The three functions retain
  the domain/order/status contracts of § 7.5 and add that identity failure.
- **Result ownership:** a hit returns a fresh `LlevQueryCursor` over shared
  immutable match storage. A miss returns the same cursor type after exact
  materialization. Every cursor may outlive the cache and follows the ordinary
  lease/reducer protocol.
- **Counters:** `stats` copies requests, hits, misses, admissions, rejections,
  evictions, resident entries, and logical weight across both shards in
  $`\mathcal{O}(1)`$. `clear` drops residency and policy history while
  preserving counters. `reset_stats` zeros counters while preserving residency
  and frequency history. All three return `OK` · `NULL_POINTER` · `PANIC`.
- **Thread safety:** one cache pointer is an exclusive mutable handle. The
  implementation contains no lock. Concurrent callers shard one cache per
  worker from the same shareable transducer; independent returned cursors are
  concurrently usable under their own contracts.
- **Complexity:** a hit is expected $`\mathcal{O}(\lvert q \rvert)`$ for exact
  key hashing/comparison plus an `Arc` clone and cursor creation. A miss adds
  the full query traversal and result materialization. Admission/eviction work
  is bounded by resident metadata and performs no provider callback while a
  lock is held because no cache lock exists.

The full policy rationale, literate transaction algorithm, language idioms,
and measurement guidance are in the
[bounded query-cache guide](query-cache.md).

### 7.6 The lease protocol

The batch path is a strict finite-state machine; every transition below is
exactly what `src/ffi/index.rs` implements:

![Cursor lease FSM: Idle alternates with Leased under next_batch/release_batch with exact-generation guards; End is sticky; free is refused with BATCH_IN_USE while a lease is live.](../diagrams/bindings/cursor-lease-state.svg)

```c
LlevStatus llev_query_cursor_next_batch(LlevQueryCursor* cursor,
                                        size_t max_matches,
                                        LlevMatchBatchView* out_batch);
LlevStatus llev_query_cursor_release_batch(LlevQueryCursor* cursor,
                                           uint64_t generation);
```

**`llev_query_cursor_next_batch`** advances the query until `max_matches`
descriptors are staged (or the stream ends), rebuilds the descriptor views
over the refilled arenas, and hands out one lease:

- `*out_batch` is always written: zeroed (`{NULL, 0, 0}`) first, then
  overwritten with `{matches, len, generation}` only on `OK`.
- On `OK`, `matches[0..len)` and every `term_data` they address are
  **borrowed** from cursor-owned storage. Per descriptor: `unit_domain`
  states the term encoding; UTF-8 terms have `term_len` scalar count and
  `byte_len` UTF-8 bytes; byte terms have `term_len == byte_len`; u64 terms
  have `term_data` 8-byte aligned and `byte_len == term_len * 8`; `has_id`
  is zero or one and gates `id`.
- The lease generation satisfies
  ```math
  g_{i+1} \;=\; \max\bigl((g_i + 1) \bmod 2^{64},\; 1\bigr) ,
  ```
  i.e. generations are strictly increasing per cursor within the `uint64_t`
  cycle and **never zero** — zero can never name a live lease.
- **Statuses:** `OK` · `END` (stream exhausted; **no lease taken**; the view
  stays zeroed; asking again yields `END` again) · `BATCH_IN_USE` (a live
  lease exists; cursor unchanged) · `INVALID_ARGUMENT` (`max_matches` = 0) ·
  `NULL_POINTER` · provider-fault mappings recorded during traversal
  (`INVALID_ARGUMENT` / `NULL_POINTER` / `UNSUPPORTED` / `IO_ERROR` /
  `CLOSED` / `LIMIT_EXCEEDED` / `PROVIDER_ERROR`) · `PANIC`.
- **Preconditions:** `cursor` and `out_batch` valid; the cursor **must not**
  be used from two threads at once (leases are single-owner state; see
  § 7.10).

**`llev_query_cursor_release_batch`** settles the lease:

- succeeds only for **the exact live generation**; a stale generation, zero,
  or a cursor with no live lease is refused with `INVALID_ARGUMENT` and
  changes nothing. After release the storage returns to the cursor and every
  borrowed pointer from that batch is invalid.
- **Statuses:** `OK` · `INVALID_ARGUMENT` · `NULL_POINTER` · `PANIC`.

The full protocol on the wire, including every refusal:

![Lease lifecycle sequence: OK with generation g₁, refusals of advance/free while leased and of stale or zero generations, exact-generation release, warm second batch with g₂ > g₁, clean End, and final free.](../diagrams/bindings/lease-lifecycle-sequence.svg)

While a lease is live, the library **neither mutates nor reallocates** the
descriptor array or the arenas (contract row `ffi-leased-batch-aliasing`;
Verus home `docs/verification/verus/ffi_batch_arena.rs`, landing this wave),
and the lease/generation machine's formal home is the wave-W3 TLA⁺ model
`docs/verification/tla/LlevBatchLease.tla` (invariant IDs LLEV-LEASE-1
through LLEV-LEASE-7) with anchor test
[`tests/ffi_resource_snapshot_semantics.rs`](../../tests/ffi_resource_snapshot_semantics.rs).

The literate batch loop every facade implements:

```text
procedure consume(cursor, B):                        ▷ B ≥ 1, default 256
    loop:
        status, view ← llev_query_cursor_next_batch(cursor, B)
        if status = END:        break                ▷ no lease was taken
        if status ≠ OK:         fail with llev_last_error_message()
        for i in 0 .. view.len − 1:                  ▷ borrowed, zero-copy
            process(view.matches[i])                 ▷ copy only what escapes
        status ← llev_query_cursor_release_batch(cursor, view.generation)
        if status ≠ OK:         fail                 ▷ exact generation required
    llev_query_cursor_free(cursor)                   ▷ OK: no lease is live
```

Boundary-crossing count for $`n`$ matches: $`\lceil n / B \rceil`$ lease
pairs plus one terminal `END` probe — never one crossing per match. Each
node expansion inside the traversal costs the provider
$`\lceil \deg(v) / 256 \rceil`$ crossings
([canon § 2](https://github.com/vinary-tree/vinary-tree-interop/blob/master/docs/abi-reference.md#2-prologue-what-kind-of-header-this-is)).

### 7.7 `llev_query_cursor_reduce`

```c
LlevStatus llev_query_cursor_reduce(LlevQueryCursor* cursor,
                                    size_t batch_size,
                                    LlevBatchReducer reducer,
                                    void* context,
                                    size_t* out_count);
```

The expert path: consume the remaining stream with **one callback per
reusable batch**, never creating a per-match host object. The callback
receives borrowed descriptors valid only for the duration of the call; the
internal lease is settled automatically around every invocation.

![Reducer flow: fill a batch, invoke the callback with borrowed views, continue on Ok, stop successfully on End, abort verbatim on any other status — with the lease auto-released in every arm.](../diagrams/bindings/reducer-flow-sequence.svg)

The literate reducer contract:

```text
procedure reduce(cursor, B, f, ctx):
    count ← 0
    loop:
        status ← fill one internal batch of ≤ B matches
        if status = END:  break                      ▷ stream exhausted
        r ← f(ctx, views, len)                       ▷ borrowed for this call only
        count ← count + len                          ▷ the batch already counted
        release the internal lease                   ▷ before r is inspected
        if r = END:       break                      ▷ EARLY STOP — a success
        if r ≠ OK:        return r verbatim          ▷ ABORT — cursor stays usable
    out_count ← count
    return OK
```

- **Statuses:** `OK` (completion **and** early stop — `END` is never
  returned by `reduce` itself) · `BATCH_IN_USE` (a lease from `next_batch`
  is live) · `NULL_POINTER` (cursor, reducer, or `out_count` NULL) ·
  `INVALID_ARGUMENT` (`batch_size` = 0) · traversal provider-fault mappings
  as in § 7.6 · **any status the reducer returned, verbatim** (the abort
  channel; the message slot then reads "batch reducer aborted the query") ·
  `PANIC`.
- After an abort the cursor is **not poisoned**: the lease was released and
  a later `next_batch`/`reduce` resumes exactly where the stream stopped.
- `*out_count` is written only on `OK` and counts every match delivered to
  the callback, including the final partial batch of an early stop.
- **Callback duties:** the reducer must not free the descriptors, must not
  re-enter the cursor (reducer reentrancy is undocumented behavior), must
  return one of the 13 published `LlevStatus` values (the abort channel
  forwards its return verbatim, so an out-of-range integer is a contract
  violation on this project-level wire), and — being `extern "C"` — must
  not unwind; signal failure through its return status. It executes on the
  calling thread.
- **Complexity:** $`\lceil n / B \rceil`$ callback crossings for the
  remaining $`n`$ matches.

### 7.8 `llev_query_cursor_free`

```c
LlevStatus llev_query_cursor_free(LlevQueryCursor* cursor);
```

Closes the cursor and releases its snapshot retain — unless a lease is live,
in which case it returns `BATCH_IN_USE` and **the cursor remains alive and
owned by the caller** (freeing storage a caller still borrows would be a
use-after-free factory; release the batch first). NULL is a no-op `OK`.

- **Statuses:** `OK` · `BATCH_IN_USE` · `PANIC`.
- **Ownership:** on `OK` the handle is consumed and must not be reused; on
  `BATCH_IN_USE` ownership is unchanged.
- **Complexity:** $`\mathcal{O}(1)`$ plus the provider's `release` (through
  the gate).

### 7.9 Cursor memory model

`LlevQueryCursor` owns: the retained snapshot provider, the safe-Rust batch,
the descriptor `views`, an `offsets` scratch vector, and the two term arenas.
`views`/`offsets` are preallocated to `LLEV_DEFAULT_MATCH_BATCH`; the arenas
start empty and warm up after the first batch (`clear()` retains capacity),
so a steady-state batch performs **zero** allocations once term volume
stabilizes. Descriptor `term_data` pointers are fixed up in a **second pass**
only after all arena writes for the batch completed — the realloc-safety
invariant (`ffi-leased-batch-aliasing`; Verus obligation LLEV-ARENA-1..3,
this wave). Details with the Rust types:
[resource-consumer.md](resource-consumer.md).

### 7.10 Thread-safety summary

| Object | Concurrent use |
|---|---|
| `LlevTransducer` | Shareable: concurrent `query_*`/`unit_domain` calls are safe. Callbacks to a non-reentrant provider serialize on that provider's gate (VT-GATE-1..3). |
| `LlevQueryCursor` | **Exclusive**: one thread at a time. The lease/generation state is deliberately unsynchronized single-owner state; interleave calls from two threads and the refusals in § 7.5 are no longer meaningful. Different cursors are fully independent, even over one dictionary. |
| `LlevGeneralizedAutomaton` / `LlevUniversalAutomaton` | Immutable configurations: shareable for concurrent construction of independent online states and complete evaluations. |
| generalized/universal online handles | **Exclusive**: one thread at a time. Advance mutates the committed prefix state; observation requires no mutation but must not race an advance. |
| `LlevPhoneticPattern` / `LlevPhoneticRuleSet` | Immutable after construction: shared concurrent reads (`matches`, `apply`, `size`, `len`) are safe. |
| error messages | Per-thread by construction (§ 3.2). |

---

## 7B. Standalone automata (14)

API revision 5 adds two dictionary-independent native machines. The
generalized family copies a runtime `LlevGeneralizedOperation` array and
evaluates exact fixed-point costs over Unicode scalars. The universal family
selects a standard, adjacent-transposition, or merge-and-split specialization
over bytes, Unicode scalars, or u64 tokens, with an optional owned directional
zero-cost substitution policy.

```c
LlevStatus llev_generalized_automaton_new(
    uint8_t max_distance, const LlevGeneralizedOperation* operations,
    size_t operation_count, LlevGeneralizedAutomaton** out_automaton);
void llev_generalized_automaton_free(LlevGeneralizedAutomaton* automaton);
LlevStatus llev_generalized_automaton_evaluate_utf8(
    const LlevGeneralizedAutomaton* automaton,
    const char* source, size_t source_len,
    const char* target, size_t target_len,
    const LlevAutomatonLimits* limits,
    LlevGeneralizedObservation* out_observation);
LlevStatus llev_generalized_online_new_utf8(
    const LlevGeneralizedAutomaton* automaton,
    const char* source, size_t source_len,
    const LlevAutomatonLimits* limits,
    LlevGeneralizedOnlineAutomaton** out_online);
LlevStatus llev_generalized_online_advance(
    LlevGeneralizedOnlineAutomaton* online, uint32_t scalar,
    LlevGeneralizedObservation* out_observation);
LlevStatus llev_generalized_online_observation(
    const LlevGeneralizedOnlineAutomaton* online,
    LlevGeneralizedObservation* out_observation);
void llev_generalized_online_free(LlevGeneralizedOnlineAutomaton* online);

LlevStatus llev_universal_automaton_new(
    uint8_t max_distance, uint32_t variant, uint32_t policy_unit_domain,
    const LlevUniversalEquivalence* equivalences, size_t equivalence_count,
    LlevUniversalAutomaton** out_automaton);
void llev_universal_automaton_free(LlevUniversalAutomaton* automaton);
LlevStatus llev_universal_automaton_evaluate(
    const LlevUniversalAutomaton* automaton, uint32_t unit_domain,
    const void* source_data, size_t source_len,
    const void* target_data, size_t target_len,
    const LlevAutomatonLimits* limits,
    LlevUniversalObservation* out_observation);
LlevStatus llev_universal_online_new(
    const LlevUniversalAutomaton* automaton, uint32_t unit_domain,
    const void* source_data, size_t source_len,
    const LlevAutomatonLimits* limits,
    LlevUniversalOnlineAutomaton** out_online);
LlevStatus llev_universal_online_advance(
    LlevUniversalOnlineAutomaton* online, uint64_t unit,
    LlevUniversalObservation* out_observation);
LlevStatus llev_universal_online_observation(
    const LlevUniversalOnlineAutomaton* online,
    LlevUniversalObservation* out_observation);
void llev_universal_online_free(LlevUniversalOnlineAutomaton* online);
```

Constructors copy borrowed configuration data and publish an output handle
only on `OK`. Null limits select the native defaults; explicit ceilings are
honored without clamping. Advance validates the next scalar or byte and target
limit before mutation. Generalized `current_row_nonempty == 0` is an exact
current-generation observation, not a permanent-death signal: retained older
rows can revive through a multi-target operation. Universal `alive == 0` is
permanent.

Generalized constructors return `INVALID_ARGUMENT` for malformed operations or
inexact/invalid weights and `LIMIT_EXCEEDED` for aggregate or arithmetic
ceilings. Universal constructors return `INVALID_ARGUMENT` for an unknown
variant, policy domain, invalid byte/Unicode value, or inconsistent empty
policy; evaluation adds `DOMAIN_MISMATCH` when an owned policy and input differ.
All fallible functions may return `NULL_POINTER` or `PANIC`; UTF-8 entry points
may return `INVALID_UTF8`; evaluation and advance may return
`LIMIT_EXCEEDED`. The exact data model, ownership pattern, liveness caveat,
complexities, and verification evidence are specified in the
[standalone-automata guide](standalone-automata.md).

---

## 7c. WallBreaker (8)

WallBreaker owns a finite Unicode term set and its own immutable substring
index. It cannot be synthesized from `vt.dictionary.v1`, whose traversal
contract does not expose exact substring occurrence. The eight public calls
are `llev_wallbreaker_new_utf8`, `llev_wallbreaker_free`,
`llev_wallbreaker_query_utf8`, `llev_wallbreaker_cursor_next_batch`,
`llev_wallbreaker_cursor_release_batch`, `llev_wallbreaker_cursor_cancel`,
`llev_wallbreaker_cursor_free`, and `llev_wallbreaker_split_utf8`.

Construction copies every length-bearing UTF-8 term; `(NULL, 0)` is the empty
term. `LlevWallBreakerLimits` specifies positive ceilings for terms, term
bytes, term/query scalars, candidate-clone bytes, result count, and result
bytes; distance has a separate maximum of eight. A failed constructor leaves
its output NULL. A successful query eagerly verifies complete results and
returns an independently owned cursor, so freeing the matcher does not
invalidate that cursor. Queries requiring substring materialization reject
over-budget candidate cloning before publishing any partial cursor.

The cursor publishes borrowed `LlevWallBreakerResultView` descriptors and
term bytes in a `LlevWallBreakerBatchView`. At most one generation is leased
at a time. Release that exact generation before advancing; a stale or double
release fails. If the next complete term exceeds the caller's batch-byte
ceiling, `LIMIT_EXCEEDED` does not advance the cursor. Cancellation forbids
future advances but preserves an outstanding lease until release or cursor
free. `llev_wallbreaker_split_utf8` projects native `PatternSplitter` pieces
with both UTF-8 byte and Unicode-scalar coordinates; `(NULL, 0)` is a sizing
call and reports required capacity without writing a partial array.

The fallible calls distinguish `NULL_POINTER`, `INVALID_ARGUMENT`,
`INVALID_UTF8`, `LIMIT_EXCEEDED`, and `PANIC` where their documented inputs
permit those failures; the cursor also reports `END` and `CLOSED` according
to its state. The [WallBreaker guide](wallbreaker-unicode.md) gives the
complete algorithm/limit model, example use, and the owned-cursor lifecycle.

---

## 7d. Specialized Unicode traversals (7)

API revision 9 adds three query constructors over the same query-start
dictionary snapshot as § 7. `llev_transducer_query_filtered_utf8` returns the
ordinary `LlevQueryCursor`: its `LlevValueFilterCallback` sees an optional u64
value at each accepted final node *before* the term is constructed. Return
zero to reject, one to admit, or two to abort with `INVALID_ARGUMENT`.
Results remain in dictionary traversal order and retain their provider IDs.
Keep the callback and its context valid until the cursor is freed. A caller
that moves the cursor between threads must make both safe to invoke on each
thread that advances it.

```c
LlevStatus llev_transducer_query_filtered_utf8(
    const LlevTransducer*, const char*, size_t, size_t,
    LlevValueFilterCallback, void*, LlevQueryCursor**);
LlevStatus llev_transducer_query_contextual_utf8(
    const LlevTransducer*, const char*, size_t, double, double,
    LlevContextualCostCallback, void*, LlevSpecializedCursor**);
LlevStatus llev_transducer_query_pruned_utf8(
    const LlevTransducer*, const char*, size_t, size_t,
    LlevPrefixCallback, void*, LlevSpecializedCursor**);
LlevStatus llev_specialized_cursor_next_batch(
    LlevSpecializedCursor*, size_t, LlevSpecializedBatchView*);
LlevStatus llev_specialized_cursor_release_batch(
    LlevSpecializedCursor*, uint64_t);
LlevStatus llev_specialized_cursor_reduce(
    LlevSpecializedCursor*, size_t, LlevSpecializedBatchReducer, void*, size_t*);
LlevStatus llev_specialized_cursor_free(LlevSpecializedCursor*);
```

Contextual callback operation codes are zero for substitution, one for
insertion, and two for deletion. A borrowed `LlevEditContextView` provides the
complete query, zero-based edit position, visited dictionary prefix, and
current edge scalar when present. Scalars are u32 Unicode values; the arrays
expire when the callback returns. NaN forbids an edit. A finite nonnegative
cost is allowed, but every positive cost must meet the declared strictly
positive `minimum_nonzero_cost`; violating it aborts with `INVALID_ARGUMENT`
so pruning cannot silently omit a match. The cursor reports a floating point
`cost`, with no prefix score.

The prefix callback operation codes are zero for structural unit comparison,
one for `enter`, two for `leave`, three for terminal membership, and four for
optional score. Enter/leave are balanced, including rejected subtrees and
early close. The returned cost is an integral edit distance represented as a
double. Results follow dictionary DFS order. Both specialized constructors
are Unicode-only; incompatible dictionary domains report `DOMAIN_MISMATCH`.

`LlevSpecializedCursor` owns reusable UTF-8 term storage and lends at most the
requested number of `LlevSpecializedMatch` descriptors per batch. A successful
`next_batch` produces a nonzero generation that must be released exactly
before advancing, reducing, or freeing; a live lease reports `BATCH_IN_USE`.
The reducer receives borrowed descriptors only during each callback. Return
`END` to stop successfully, or a published error status to abort. Foreign
callbacks execute synchronously on the advancing thread and must not unwind
across C. Their function and context pointers must remain valid until cursor
free; freeing a prefix cursor may invoke final `leave` callbacks.

---

## 8. Phonetic surface (62)

The phonetic algorithms are compiled with `bindings-phonetic`. Fallible
phonetic calls return `UNSUPPORTED` without that feature; the AOT byte calls
also require `serialization`. Probe `llev_build_features()` for
`LLEV_BUILD_FEATURE_PHONETIC` and, for compiled bytes,
`LLEV_BUILD_FEATURE_PHONETIC_AOT` before calling. Void release functions
are safe null no-ops even when an optional feature is disabled. All other
owned output handles and arrays must be released with the matching family
function, not `free(3)` or another family's release function.

### Family inventory and feature gates

| Family | Public functions | Gate and result ownership |
|---|---|---|
| Compiled patterns and dictionary-language product | `llev_phonetic_pattern_compile_regex`, `llev_phonetic_pattern_compile_llre`, `llev_phonetic_pattern_load_llre_file`, `llev_phonetic_pattern_free`, `llev_phonetic_pattern_size`, `llev_phonetic_pattern_matches`, `llev_transducer_query_pattern` | Phonetic; pattern owns native automaton, query publishes a normal snapshot cursor. |
| Rewrite rules | `llev_phonetic_rules_parse`, `llev_phonetic_rules_load_file`, `llev_phonetic_rules_builtin`, `llev_phonetic_rules_free`, `llev_phonetic_rules_len`, `llev_phonetic_rules_apply`, `llev_owned_string_free` | Phonetic; rules are immutable, rewritten text is an owned string. |
| Compiled native bytes | `llev_phonetic_rules_to_bytes`, `llev_phonetic_rules_from_bytes`, `llev_phonetic_pattern_to_bytes`, `llev_phonetic_pattern_from_bytes`, `llev_owned_bytes_free` | Phonetic AOT; exact owned bytes, or a new owned rule/pattern handle. |
| Articulation and syllables | `llev_phonetic_articulatory_distance`, `llev_phonetic_articulatory_edit_distance`, `llev_phonetic_syllable_count`, `llev_phonetic_syllable_boundaries` | Phonetic; scalar outputs or caller-owned boundary array. |
| Word-boundary grep | `llev_phonetic_grep_new`, `llev_phonetic_grep_free`, `llev_phonetic_grep_distance_config`, `llev_phonetic_grep_matches`, `llev_phonetic_grep_scan_line`, `llev_phonetic_grep_scan_text` | Phonetic; reusable matcher and caller-owned result array. |
| Normalized dictionary | `llev_phonetic_dictionary_new`, `llev_phonetic_dictionary_free`, `llev_phonetic_dictionary_query`, `llev_phonetic_candidates_free`, `llev_phonetic_dictionary_update` | Phonetic; mutable or compact native index, caller-owned nested candidate array. |
| Character-level online grep | `llev_phonetic_online_new`, `llev_phonetic_online_free`, `llev_phonetic_online_normalized_query`, `llev_phonetic_online_scan`, `llev_phonetic_online_matches_free`, `llev_phonetic_online_stream_new`, `llev_phonetic_online_stream_free`, `llev_phonetic_online_stream_feed`, `llev_phonetic_online_stream_finish` | Phonetic; reusable matcher, exclusive stream, owned match array. |
| Token-query grep | `llev_phonetic_token_new`, `llev_phonetic_token_free`, `llev_phonetic_token_scan`, `llev_phonetic_token_matches_free` | Phonetic; reusable query and one owned nested result graph. |
| Incremental rewrite | `llev_phonetic_transducer_new`, `llev_phonetic_transducer_free`, `llev_phonetic_transducer_feed`, `llev_phonetic_transducer_finish`, `llev_phonetic_transducer_reset`, `llev_phonetic_transducer_normalize` | Phonetic; exclusive stream with independently owned output strings. |
| Reverse expansion | `llev_phonetic_expand`, `llev_phonetic_expand_with_costs` | Phonetic; exhaustive and greedy cost-aware algorithms have distinct semantics. |
| IPA feature table | `llev_phonetic_features`, `llev_phonetic_chars_with_features`, `llev_phonetic_similar_chars`, `llev_phonetic_voicing_pair`, `llev_phonetic_feature_relation`, `llev_phonetic_expand_feature_based`, `llev_phonetic_feature_set_distance` | Phonetic; borrowed input scalars and caller-owned fixed-capacity result buffers. |

### 8.1 Owned strings

```c
void llev_owned_string_free(LlevOwnedString* value);
```

`LlevOwnedString` (`{char* data; size_t len;}`) carries heap-owned,
**not NUL-terminated** UTF-8 out of `llev_phonetic_rules_apply`. The free
zeroes the struct after releasing; NULL input and the empty
(`{NULL, 0}`) value are safe no-ops. Statuses: none (void).

### 8.2 Patterns

```c
LlevStatus llev_phonetic_pattern_compile_regex(const char* source, size_t source_len,
                                               LlevPhoneticPattern** out_pattern);
LlevStatus llev_phonetic_pattern_compile_llre(const char* source, size_t source_len,
                                              LlevPhoneticPattern** out_pattern);
void llev_phonetic_pattern_free(LlevPhoneticPattern* pattern);
LlevStatus llev_phonetic_pattern_size(const LlevPhoneticPattern* pattern,
                                      size_t* out_states, size_t* out_transitions);
LlevStatus llev_phonetic_pattern_matches(const LlevPhoneticPattern* pattern,
                                         const char* input, size_t input_len,
                                         uint8_t* out_matches);
```

A pattern is a reusable Unicode NFA compiled from a phonetic regular
expression or an **import-free** `.llre` document, subject to the public
state ceiling (`LANGUAGE_PRODUCT_MAX_STATES`; exceeding it is
`INVALID_ARGUMENT`, message included).

| Function | Statuses | Notes |
|---|---|---|
| `compile_regex` / `compile_llre` | `OK` · `NULL_POINTER` · `INVALID_UTF8` · `INVALID_ARGUMENT` (parse error, state ceiling, imports present) · `UNSUPPORTED` (feature off) · `PANIC` | On `OK`, caller owns the handle; one `llev_phonetic_pattern_free` settles it. |
| `pattern_free` | none (void) | NULL no-op. **Existing cursors retain their own copy of the pattern product** — freeing the pattern never invalidates a running `query_pattern` cursor. |
| `pattern_size` | `OK` · `NULL_POINTER` · `UNSUPPORTED` · `PANIC` | Writes NFA state and transition counts. $`\mathcal{O}(1)`$. |
| `pattern_matches` | `OK` · `NULL_POINTER` · `INVALID_UTF8` · `UNSUPPORTED` · `PANIC` | Complete-string acceptance; writes zero or one. $`\mathcal{O}(\lvert \text{input} \rvert \cdot \lvert \text{NFA} \rvert)`$ worst case. |

### 8.3 `llev_transducer_query_pattern`

```c
LlevStatus llev_transducer_query_pattern(const LlevTransducer* transducer,
                                         const LlevPhoneticPattern* pattern,
                                         uint8_t max_distance,
                                         LlevQueryCursor** out_cursor);
```

Starts a lazy query for dictionary terms within `max_distance` of the
pattern's **language** — the dictionary × language product. Unicode
dictionaries only (`DOMAIN_MISMATCH` otherwise). Captures a snapshot exactly
like the § 7.5 query starts and returns the same cursor type with the same
lease protocol. **Statuses:** `OK` · `NULL_POINTER` · `DOMAIN_MISMATCH` ·
`UNSUPPORTED` (feature off, or provider `Unsupported`) · `INVALID_ARGUMENT`
/ `IO_ERROR` / `CLOSED` / `LIMIT_EXCEEDED` (provider verbatim) ·
`PROVIDER_ERROR` · `PANIC`.

### 8.4 Rule sets

```c
LlevStatus llev_phonetic_rules_parse(const char* source, size_t source_len,
                                     LlevPhoneticRuleSet** out_rules);
LlevStatus llev_phonetic_rules_builtin(uint32_t kind, LlevPhoneticRuleSet** out_rules);
void llev_phonetic_rules_free(LlevPhoneticRuleSet* rules);
LlevStatus llev_phonetic_rules_len(const LlevPhoneticRuleSet* rules, size_t* out_len);
LlevStatus llev_phonetic_rules_apply(const LlevPhoneticRuleSet* rules,
                                     const char* input, size_t input_len,
                                     LlevOwnedString* out_text);
```

A rule set is a reusable `.llev` rewrite system applied to a fixed point
under the native fuel bound.

| Function | Statuses | Notes |
|---|---|---|
| `rules_parse` | `OK` · `NULL_POINTER` · `INVALID_UTF8` · `INVALID_ARGUMENT` (parse error, includes present) · `UNSUPPORTED` · `PANIC` | Import-free documents only. |
| `rules_builtin` | `OK` · `NULL_POINTER` · `INVALID_ARGUMENT` (unknown `kind`) · `UNSUPPORTED` · `PANIC` | `LLEV_PHONETIC_RULE_SET_ENGLISH_ORTHOGRAPHY` (0) · `ENGLISH_PHONETIC` (1). |
| `rules_free` | none (void) | NULL no-op. |
| `rules_len` | `OK` · `NULL_POINTER` · `UNSUPPORTED` · `PANIC` | Enabled-rule count. $`\mathcal{O}(1)`$. |
| `rules_apply` | `OK` · `NULL_POINTER` · `INVALID_UTF8` · `UNSUPPORTED` · `PANIC` | On `OK`, `*out_text` is caller-owned — settle with `llev_owned_string_free`. An empty result is the `{NULL, 0}` value (also safe to free). |

### 8.5 Trusted source files and compiled bytes

`llev_phonetic_rules_load_file` resolves `.llev` includes, and
`llev_phonetic_pattern_load_llre_file` resolves `.llre` imports from a trusted
local path. Both accept borrowed `LlevUtf8View` search directories; zero
directories select the native loader defaults. The caller sets a positive
aggregate path-byte ceiling, not a ceiling on the contents of files reached
transitively. Rules permit at most 64 search paths. Invalid UTF-8 paths,
malformed documents, missing files, import/include failures, and bound
violations return the corresponding `INVALID_UTF8`, `INVALID_ARGUMENT`,
`IO_ERROR`, or `LIMIT_EXCEEDED` status without publishing a partial handle.
These entry points are **not** a sandbox for adversarial filesystem paths.

The four AOT calls `llev_phonetic_{rules,pattern}_{to,from}_bytes` require
both the phonetic and serialization build features. `to_bytes` publishes one
`LlevOwnedBytes`; call `llev_owned_bytes_free` on its address, which releases
and clears it, making repeated release of that cleared *value* harmless.
`from_bytes` reads borrowed bytes and publishes a new owned rule/pattern
handle independent of those source bytes. Each input/output byte ceiling must
be positive and at most 16 MiB. Invalid magic/version, oversized data, or
malformed native payloads fail without a partial handle or owned byte buffer.
The compiled format is versioned native data, **not** a stable interchange
format across library versions. With serialization disabled, all four calls
return `UNSUPPORTED` without touching output storage; ordinary phonetic
matching and rewriting remain available. The
[serialization verification contract](../verification/PHONETIC_JULIA_BINDING_CONTRACT.md)
maps the output transaction and release laws to model-checked invariants and
real-ABI property tests.

### 8.6 Articulatory and syllable analysis

`llev_phonetic_articulatory_distance` compares two valid Unicode scalar
values using the native IPA feature table; an unknown but valid scalar has
the table's normal fallback cost. The optional
`LlevPhoneticFeatureWeights` contains seven finite, nonnegative `double`
costs; NULL selects native standard weights. Invalid scalar values or costs
return `INVALID_ARGUMENT` and leave the result unchanged.
`llev_phonetic_articulatory_edit_distance` applies these substitution costs
with unit insertion/deletion costs to two UTF-8 strings. Its positive
`max_cells` bounds each scalar length and their product before allocation;
exceeding it returns `LIMIT_EXCEEDED`, not a truncated distance.

`llev_phonetic_syllable_count` and
`llev_phonetic_syllable_boundaries` select English-orthographic (`ipa = 0`)
or IPA (`ipa = 1`) heuristics under a positive scalar ceiling. Empty input
has count zero. Boundaries are Unicode-scalar positions, not UTF-8 byte
offsets, and need not equal the count from the separate heuristic. For the
boundary call, `(out_positions = NULL, capacity = 0)` is a valid sizing call:
`out_count` reports the required number even if `LIMIT_EXCEEDED` prevents
publication. Other failures leave result outputs unchanged.

### 8.7 Word-boundary grep

`llev_phonetic_grep_new` compiles a reusable word matcher from UTF-8 source,
an optional cloned rule set, a distance, one published `LlevAlgorithm`, and
a zero-or-one case-insensitivity flag. The matcher may contain a local
`(?;N:...)` distance override; `llev_phonetic_grep_distance_config`
reports its effective and optional local values. The candidate membership
call `llev_phonetic_grep_matches` requires a positive candidate-byte ceiling
and returns both a match flag and a distance (zero when unmatched) on success.

`llev_phonetic_grep_scan_line` and `llev_phonetic_grep_scan_text` copy
`LlevPhoneticGrepMatch` descriptors into caller-provided storage. Their byte
offsets are zero-based and end-exclusive; line numbers are one-based, using
Rust's `str::lines` semantics for a complete document. A positive input-byte
ceiling is required. The `(NULL, 0)` sizing call and an undersized array
report the exact required count through `out_count` but write **no partial
descriptors**. `llev_phonetic_grep_free` consumes the matcher; copied
descriptors need no separate release.

### 8.8 Mutable and compact normalized dictionaries

`llev_phonetic_dictionary_new` copies every length-bearing UTF-8 term and
clones optional rewrite rules. Mode zero creates a mutable normalized
dictionary; mode one creates compact immutable term-ID payloads. NULL rules
select the native English Zompist normalization. The algorithm, mode,
positive term-count ceiling, and positive total-byte ceiling are validated
before publishing a handle. The empty term is a valid term and remains
queryable in **both** modes.

`llev_phonetic_dictionary_query` searches normalized space with a distance
bound and positive query-scalar/result-count ceilings. Successful candidates
arrive in native relevance order; each carries a copied original term,
distance, and normalized spelling. The entire result array and nested strings
are one ownership unit: pass the exact pointer/count pair to
`llev_phonetic_candidates_free`. A result ceiling failure leaves both output
pointer and count unchanged rather than publishing a prefix. Mode-zero
`llev_phonetic_dictionary_update` inserts or removes an owned term and
reports whether the set changed; compact mode returns `UNSUPPORTED` without
mutation. All dictionary handles are consumed by
`llev_phonetic_dictionary_free`.

### 8.9 Character-level and token-query grep

Character-level `llev_phonetic_online_new` creates a reusable matcher with
optional cloned rules, a positive pattern-scalar ceiling, and a Boolean
case-insensitivity flag. `llev_phonetic_online_normalized_query` returns an
owned string; `llev_phonetic_online_scan` returns an owned array of matches
with original-byte and Unicode-scalar spans plus copied original/normalized
text. Release the complete array with
`llev_phonetic_online_matches_free`, not with individual string frees.

For chunked input, `llev_phonetic_online_stream_new` clones the matcher into
an independent exclusive stream with a positive total-byte ceiling. Feed
each UTF-8 chunk through `llev_phonetic_online_stream_feed`, then call
`llev_phonetic_online_stream_finish` once with a positive match ceiling.
A rejected oversized feed does not change the accepted prefix. Even when
finish returns `LIMIT_EXCEEDED`, it consumes the stream and leaves output
pointer/count unchanged; a second feed or finish returns
`INVALID_ARGUMENT`. Free either handle independently.

Token-query `llev_phonetic_token_new` compiles native token-grep syntax under
a positive query-byte ceiling. `llev_phonetic_token_scan` applies positive
document-byte, match-count, and nested-detail ceilings. Returned
`LlevPhoneticTokenMatch` values include document byte spans, copied matched
text, total distance, and a nested array of token details with their own
spans, strings, and distances. The entire graph belongs to one exact
pointer/count pair; `llev_phonetic_token_matches_free` settles it. Limit or
parse failures do not publish a partial graph.

### 8.10 Incremental rewrite transducer

`llev_phonetic_transducer_new` clones optional rules (NULL means identity
rewriting) into an exclusive stream. Context-sensitive rules may delay
emission until later input or finish. `llev_phonetic_transducer_feed`
accepts UTF-8 chunks under positive input-scalar/output-byte ceilings and
publishes an independently owned `LlevOwnedString` for each successful
output. On output overflow it leaves that output untouched, resets buffered
context, and requires restarting the logical stream. A successful
`llev_phonetic_transducer_finish` flushes context and requires
`llev_phonetic_transducer_reset` before reuse. The separate
`llev_phonetic_transducer_normalize` call rewrites one complete string
without changing incremental state. `llev_phonetic_transducer_free` consumes
the handle; previously returned owned strings remain independent.

### 8.11 Reverse expansion and IPA features

`llev_phonetic_expand` exhaustively explores reverse-phonetic segmentations
as a regex pattern, while `llev_phonetic_expand_with_costs` uses a distinct
greedy maximum-rule-cost strategy and publishes both pattern and cost
transactionally. They are not aliases. The five positive fields of
`LlevPhoneticExpansionLimits` bound input scalars, rule count, total rule
units, explored nodes, and output bytes. Exhaustive expansion may reject an
intermediate-work bound even if the final deduplicated output would fit; it
never silently truncates. NULL rules select identity expansion. Returned
patterns are `LlevOwnedString` values freed with `llev_owned_string_free`.

IPA classification exposes a stable 42-bit `LlevPhoneticFeatureIndex` mask,
independent of Rust enum discriminants. `llev_phonetic_features` maps one
valid scalar to its mask; an unknown scalar gets zero, while a surrogate or
out-of-range scalar is `INVALID_ARGUMENT`.
`llev_phonetic_chars_with_features` selects table characters with all or
any requested bits; unknown bits are rejected. The related
`llev_phonetic_similar_chars` and
`llev_phonetic_expand_feature_based` copy native table results in their
documented order. Each accepts `(NULL, 0)` for sizing, reports required
capacity, and writes no partial array when too small.
`llev_phonetic_voicing_pair` sets an explicit found flag, so callers do not
mistake a missing counterpart for scalar zero.
`llev_phonetic_feature_relation` selects shared-feature similarity or
native zero-cost substitution; those are different predicates.
`llev_phonetic_feature_set_distance` applies optional finite, nonnegative
weights to two valid masks. The [Julia phonetic API guide](../../bindings/julia/Liblevenshtein/docs/src/phonetic.md)
provides worked examples while using these same native functions.

---

## 8B. Bounded dictionary and suffix persistence

API revision 32 adds native binary-format operations behind
`LLEV_BUILD_FEATURE_SERIALIZATION`. `llev_dictionary_serialize`
accepts borrowed `LlevUtf8Slice` terms, constructs the native byte-domain
dictionary language, and returns a `LlevOwnedBytes` buffer released through
`llev_owned_bytes_free`. `llev_dictionary_deserialize` accepts one
complete binary payload and returns an independent, immutable
`LlevDecodedDictionaryTerms` snapshot. Read its length and borrowed term views
with `llev_decoded_dictionary_terms_len` and
`llev_decoded_dictionary_term_at`; the views expire when
`llev_decoded_dictionary_terms_free` releases the snapshot. The source bytes may
be released immediately after a successful decode.

The format ID selects the serializer and its wire revision:

| Format ID | Native serializer | Feature gate | Compatibility role |
|---|---|---|---|
| `BINCODE_V1` = 1 | `BincodeSerializer` | Serialization | Fixed-integer, little-endian Rust format. |
| `PROTOBUF_V1` = 2 | `ProtobufSerializer` | Protobuf | Portable V1 graph interchange. |
| `PROTOBUF_V2` = 3 | `OptimizedProtobufSerializer` | Protobuf | Compact native V2 graph. |
| `GZIP_BINCODE_V1` = 4 | `GzipSerializer<BincodeSerializer>` | Compression | Gzip over bincode V1. |
| `GZIP_PROTOBUF_V1` = 5 | `GzipSerializer<ProtobufSerializer>` | Protobuf and compression | Gzip over portable V1. |
| `GZIP_PROTOBUF_V2` = 6 | `GzipSerializer<OptimizedProtobufSerializer>` | Protobuf and compression | Gzip over compact V2. |
| `PROTOBUF_DAT_V1` = 7 | `DatProtobufSerializer` | Protobuf | DAT-specific `LDT1` term payload. |

Unknown format IDs return `INVALID_ARGUMENT`; a recognized format whose
feature was omitted returns `UNSUPPORTED`. This path preserves the accepted UTF-8 term language,
including an empty term, but not dictionary values or the in-memory backend.
The resulting bytes match the selected native serializer's output for the same
accepted term set. V2 and DAT are not promised to be readable by V1-only
consumers. Callers persist the independently owned bytes with their usual file
or object-store operations.

`llev_suffix_source_serialize` and `llev_suffix_source_deserialize` select
suffix-automaton Bincode V1 or Protobuf V1 with format IDs 1 and 2. They
preserve the ordered indexed source texts, including duplicates. They do not
enumerate the accepted substring language. The decoder publishes the same
owned string-snapshot handle, typed as `LlevDecodedSourceTexts` in the C header;
release it with `llev_decoded_dictionary_terms_free`.

`LlevDictionaryLimits` requires ceilings for the term count, each term,
total term bytes, and payload bytes. The payload ceiling must be at least eight
bytes. Bincode lengths, Protobuf graph shape and terminal paths, DAT term
payloads, suffix source counts, gzip integrity, and compressed and inflated
sizes are validated before native reconstruction. Malformed payloads return `INVALID_ARGUMENT`;
exceeded ceilings return `LIMIT_EXCEEDED`. A failed call leaves the output
buffer empty or the output snapshot pointer null. Without the serialization
feature, the entry points return `UNSUPPORTED`.

Generalized edit-operation sets have a separate native persistence family:
`llev_operation_set_serialize` accepts the same borrowed operation descriptors
as the standalone generalized automaton. Format IDs 1 through 4 select the
versioned binary envelope, Protobuf V1, gzip binary, and gzip Protobuf. The
caller supplies `LlevOperationSetLimits`, covering payload bytes, operation
count, name bytes, pair counts, and aggregate restriction text. Decoding with
`llev_operation_set_deserialize` returns an owned `LlevDecodedOperationSet`.
`llev_decoded_operation_set_operation_at` and
`llev_decoded_operation_set_restriction_at` borrow views until
`llev_decoded_operation_set_free`. Restriction kind 1 preserves exact raw-byte
pairs, including non-UTF-8 values; kind 2 borrows UTF-8 string pairs.
`llev_decoded_operation_set_serialize` re-encodes the native snapshot without
losing byte pairs. Invalid versions, malformed envelopes, truncated or trailing
data, invalid Protobuf, corrupt gzip, and exceeded resource ceilings reject the
input before a handle is published. Every encoded buffer uses
`llev_owned_bytes_free`.

The value-preserving Bincode entry points
`llev_valued_dictionary_serialize` and
`llev_valued_dictionary_deserialize` bind native
`BincodeSerializer::serialize_with_values`,
`serialize_with_values_char`, and `deserialize_with_values`. Unit domains 1
and 2 select byte or Unicode dictionaries. Value kinds 1 and 2 select `u64`
or `Vec<u8>` records, which have distinct native wire types and differ from
term-only Bincode. `LlevValueLimits` bounds count, per-entry and aggregate
term/value bytes, and the entire payload. A decoded handle owns canonical
entries; inspect it with `llev_decoded_value_entries_len` and
`llev_decoded_value_entry_at`, then call
`llev_decoded_value_entries_free`. Empty raw-byte values remain present.
Borrowed views expire when the handle is freed. The format carries no value
kind or unit-domain tag, so persist those alongside the bytes.

## 9. A complete C consumer

The program below is the whole § 7 flow in one file: obtain a resource from
any conforming provider, construct, query through **both** the leased-batch
and reducer paths, handle every error with the thread-local message, and
settle every handle and retain on every exit path. It compile-checks clean —
byte-identical to this listing — under:

```sh
cc -std=c17 -Wall -Wextra -Werror -fsyntax-only \
   -I include -I vinary-tree-interop/include llev_consumer_example.c
```

```c
/*
 * llev_consumer_example.c — the complete liblevenshtein consumer flow.
 *
 * Obtains a dictionary VtResource from any conforming provider (the extern
 * constructor below is satisfied by libdictenstein's ldict_dictionary_resource
 * or by the minimal C provider in the interop ABI reference), constructs a
 * transducer, runs one query through the leased-batch path and one through
 * the reducer path, and settles every handle and retain on every exit path.
 *
 * Compile check: cc -std=c17 -Wall -Wextra -Werror -fsyntax-only \
 *                   -I include -I vinary-tree-interop/include \
 *                   llev_consumer_example.c
 */
#include <inttypes.h>
#include <stdio.h>

#include "liblevenshtein.h"

/* Any conforming provider: the caller receives one owned retain. */
extern VtResource make_dictionary(void);

/* Reducer callback: print each borrowed batch; stop early after 100 matches. */
static LlevStatus print_batch(void* context, const LlevMatch* matches,
                              size_t len) {
    size_t* printed = context;
    for (size_t i = 0; i < len; ++i) {
        const LlevMatch* m = &matches[i];
        printf("%.*s\tdistance=%zu", (int)m->byte_len,
               (const char*)m->term_data, m->distance);
        if (m->has_id == 1) {
            printf("\tid=%" PRIu64, m->id);
        }
        printf("\n");
        if (++*printed >= 100) {
            return LLEV_STATUS_END; /* early stop is a SUCCESS outcome */
        }
    }
    return LLEV_STATUS_OK;
}

int main(void) {
    int exit_code = 1;
    VtResource dictionary = make_dictionary();
    LlevTransducer* transducer = NULL;
    LlevQueryCursor* cursor = NULL;

    /* 1 · Retain the resource and negotiate vt.dictionary.v1. */
    LlevStatus status = llev_transducer_new(&dictionary,
                                            LLEV_ALGORITHM_TRANSPOSITION,
                                            &transducer);
    if (status != LLEV_STATUS_OK) {
        fprintf(stderr, "transducer: %s\n", llev_last_error_message());
        goto release_dictionary;
    }

    /* 2 · Capture the query-start snapshot and lease batches. */
    status = llev_transducer_query_utf8(transducer, "levenshtein", 11, 2,
                                        LLEV_QUERY_ORDER_TRAVERSAL, &cursor);
    if (status != LLEV_STATUS_OK) {
        fprintf(stderr, "query: %s\n", llev_last_error_message());
        goto free_transducer;
    }
    for (;;) {
        LlevMatchBatchView view = {0};
        status = llev_query_cursor_next_batch(cursor,
                                              LLEV_DEFAULT_MATCH_BATCH, &view);
        if (status == LLEV_STATUS_END) {
            break; /* stream exhausted; no lease taken */
        }
        if (status != LLEV_STATUS_OK) {
            fprintf(stderr, "batch: %s\n", llev_last_error_message());
            goto free_cursor;
        }
        for (size_t i = 0; i < view.len; ++i) {
            const LlevMatch* m = &view.matches[i];
            printf("%.*s\tdistance=%zu\n", (int)m->byte_len,
                   (const char*)m->term_data, m->distance);
        }
        /* Settle the lease with its exact generation before advancing. */
        status = llev_query_cursor_release_batch(cursor, view.generation);
        if (status != LLEV_STATUS_OK) {
            fprintf(stderr, "release: %s\n", llev_last_error_message());
            goto free_cursor;
        }
    }
    status = llev_query_cursor_free(cursor);
    cursor = NULL;
    if (status != LLEV_STATUS_OK) {
        fprintf(stderr, "close: %s\n", llev_last_error_message());
        goto free_transducer;
    }

    /* 3 · The same stream through the reducer path (a fresh snapshot). */
    status = llev_transducer_query_utf8(transducer, "levenshtein", 11, 2,
                                        LLEV_QUERY_ORDER_DISTANCE_THEN_TERM,
                                        &cursor);
    if (status != LLEV_STATUS_OK) {
        fprintf(stderr, "query: %s\n", llev_last_error_message());
        goto free_transducer;
    }
    size_t printed = 0;
    size_t reduced = 0;
    status = llev_query_cursor_reduce(cursor, LLEV_DEFAULT_MATCH_BATCH,
                                      print_batch, &printed, &reduced);
    if (status != LLEV_STATUS_OK) {
        fprintf(stderr, "reduce: %s\n", llev_last_error_message());
        goto free_cursor;
    }
    printf("reduced %zu matches\n", reduced);
    exit_code = 0;

    /* 4 · Teardown: cursors first, then the transducer, then the retain the
     *     provider handed us. Each release settles exactly one owned retain. */
free_cursor:
    if (cursor != NULL) {
        (void)llev_query_cursor_free(cursor);
    }
free_transducer:
    llev_transducer_free(transducer);
release_dictionary:
    if (dictionary.vtable != NULL && dictionary.vtable->release != NULL) {
        dictionary.vtable->release(dictionary.context);
    }
    return exit_code;
}
```

---

## 10. References

1. Vladimir I. Levenshtein. 1966. *Binary codes capable of correcting
   deletions, insertions, and reversals.* Soviet Physics Doklady 10(8),
   707-710. — The distance the § 6 and § 7 surfaces compute.
2. George E. Collins. 1960. *A method for overlapping and erasure of lists.*
   Communications of the ACM 3(12), 655-657.
   DOI: [10.1145/367487.367501](https://doi.org/10.1145/367487.367501).
   — The reference-counting discipline behind every retain this document
   mentions.
3. James R. Driscoll, Neil Sarnak, Daniel D. Sleator, and Robert E. Tarjan.
   1989. *Making data structures persistent.* Journal of Computer and System
   Sciences 38(1), 86-124.
   DOI: [10.1016/0022-0000(89)90034-2](<https://doi.org/10.1016/0022-0000(89)90034-2>).
   — Why query-start snapshot capture can be $`\mathcal{O}(1)`$
   (see [snapshot semantics](../theory/snapshot-semantics.md)).

<!--
DOI verification (2026-08-08): curl -sI --max-redirs 0 https://doi.org/<doi>
  10.1145/367487.367501        -> 302 (handle API responseCode 1)
  10.1016/0022-0000(89)90034-2 -> 302 (handle API responseCode 1)
Negative control 10.1145/9999999.9999999 -> 404 / responseCode 100.
Levenshtein 1966 is a Soviet Physics Doklady translation with no DOI
(canonical family citation form, plan Workstream C).
-->

---

*Family footer:* the canon under this document —
[interop ABI reference](https://github.com/vinary-tree/vinary-tree-interop/blob/master/docs/abi-reference.md) ·
[evolution policy](https://github.com/vinary-tree/vinary-tree-interop/blob/master/docs/abi-evolution.md) ·
[security model](https://github.com/vinary-tree/vinary-tree-interop/blob/master/docs/security-model.md). Sibling
project references —
[libdictenstein `ldict_*`](https://github.com/vinary-tree/libdictenstein/blob/master/docs/bindings/c-abi-reference.md) ·
[lling-llang `lling_*`](https://github.com/vinary-tree/lling-llang/blob/master/docs/api/c-abi-reference.md) ·
[duallity `duallity_*`](https://github.com/vinary-tree/duallity/blob/master/docs/architecture/06-resource-abi-and-bindings.md).
