# Julia phonetic binding verification contract

This document bounds the claims made for the additive revision-8 C/Julia
phonetic surface. The public Rust implementation remains the semantic oracle:
the binding owns no duplicate matcher, normalizer, feature table, or ranking
algorithm. An ABI handle is owned by exactly one Julia wrapper; supplied rules
are cloned into longer-lived native configurations; result text and arrays are
copied before native release. Native status errors become `NativeError`, while
Julia argument-domain errors become `ArgumentError`.

## Formal and executable correspondence

Three small TLA+ models describe *binding protocols*, not the complete
language recognized by every phonetic automaton:

- [`PhoneticBoundedResults.tla`](tla/PhoneticBoundedResults.tla) models
  all-or-nothing result publication and the word-grep required-capacity
  exception. TLC exhausted 120 distinct states. Registry laws
  `LLEV-PHON-BOUND-1..5` connect these invariants to real C ABI tests.
- [`PhoneticStreamingBoundary.tla`](tla/PhoneticStreamingBoundary.tla) models
  scanner versus rewrite-stream finish and overflow behavior. TLC exhausted
  45 distinct states. Registry laws `LLEV-PHON-STREAM-1..4` connect it to
  boundary tests.
- [`PhoneticAotBoundary.tla`](tla/PhoneticAotBoundary.tla) models the
  optional compiled-byte feature, output transaction, decode provenance,
  one release, and idempotent repeated free. TLC exhausted 16 distinct
  states. Registry laws `LLEV-PHON-AOT-1..5` connect it to boundary tests.

The executable differential suite is not a simulation of the model: it calls
the actual exported C functions and compares copied results with the public
Rust implementations. Each generative test has 32 bounded cases per run;
the Julia suite also exhausts 64 short spellings across mutable/compact
indexes, whole/chunked rewriting, and whole/chunked scanning. Failure cases
check unchanged output sentinels and required-capacity side channels.
Every one of the 14 modeled invariants has a generated real-ABI test in the
invariant registry. The serialization-disabled invariant requires its own
separately compiled feature variant; varying inputs within an enabled build
cannot test the absence of a compile-time feature.

| Family | Protocol model | Concrete properties and semantic authority |
|---|---|---|
| Articulatory distance and syllable heuristics | No stateful protocol beyond checked scalar/owned-output publication | `ffi_phonetic_analysis.rs::generated_articulatory_and_syllable_results_equal_native` varies inputs and compares C values to the public native functions; the Julia guide states Unicode-scalar and heuristic limits. A separate finite state-machine model of a pure deterministic arithmetic/table function would duplicate, rather than validate, its implementation. A mathematical recurrence proof is a distinct upstream phonetic-theory obligation and is **not** claimed here. |
| Word-boundary phonetic grep | Bounded results | `ffi_phonetic_analysis.rs::generated_word_grep_membership_equals_native` varies pattern and candidate; `generated_word_capacity_reports_exact_required_count_without_partial_payload` varies spans and capacity, checking all-or-none publication and the required-count side channel. |
| Mutable and compact normalized dictionaries | Bounded results | `ffi_phonetic_dictionary.rs::generated_mutable_and_compact_candidates_equal_native` varies terms, query, and edit bound, includes the empty spelling, compares both backends to public Rust, and checks insufficient result capacity leaves pointer/count sentinels unchanged; the Julia 64-case cross-surface property compares both backends and native relevance order. This campaign removed three native empty-term filters exposed by the property. |
| Character-level online grep and chunked scanner | Bounded results and streaming boundary | `ffi_phonetic_online.rs::generated_scan_and_chunked_stream_equal_native` varies pattern, text, and chunk split; `generated_rejected_feed_preserves_scanner_prefix` checks an oversized feed cannot mutate a subsequent accepted prefix; `generated_limited_finish_consumes_stream_without_publishing` checks a result-limited finish is atomic and single-use. Julia repeats whole/stream equality over 64 inputs. Native scan owns the matching semantics; the model constrains byte limits, finalization, and error publication. |
| Token-query grep | Bounded results | `ffi_phonetic_token.rs::generated_token_spans_and_details_equal_native` varies documents and compares every copied span, text, distance, and detail to native Rust; `generated_token_capacity_rejection_is_atomic` varies separators and checks both result and nested-detail ceilings leave pointer/count sentinels unchanged. |
| Incremental rewrite transducer | Streaming boundary | `ffi_phonetic_transducer.rs::generated_chunk_partitions_equal_whole_native_rewrite` varies input/split and checks exact native whole normalization. `generated_output_overflow_is_atomic_and_resets` checks a rejected feed neither publishes output nor contaminates the following stream. Julia repeats whole/chunked equivalence over 64 inputs. |
| Exhaustive and cost-aware reverse expansion | Bounded results | `ffi_phonetic_expansion.rs::generated_bounded_expansions_equal_native` varies inputs and compares both C outputs to the two distinct public native algorithms; it also checks the output-byte ceiling rejects without modifying the output sentinel. Bounded expansion explicitly errors rather than truncating. The model covers publication, while Rust unit tests retain the exact reverse-segmentation algorithm obligation. |
| IPA feature classification and relations | No stateful protocol; finite 42-bit feature domain | `ffi_phonetic_features.rs::generated_ipa_relations_equal_public_native` varies known and unknown scalars and checks cardinality and both native relations; fixed tests pin mask bits, voicing, expansions, and invalid masks. A state-transition model adds no information to this pure classification table. The stable bit mapping and native table are directly tested instead. |
| Trusted `.llev`/`.llre` file loaders | AOT transaction law by analogy; native loader graph is not reimplemented | `ffi_phonetic_files_aot.rs::file_loaders_resolve_includes_and_imports_like_native_rust` compares imported/included fixtures to native loaders; generated missing paths, invalid UTF-8, count and byte limits prove no partial handle. A finite model of arbitrary filesystem contents and import graph would be unsound without modeling filesystem and parser; the native loader is the authority. These APIs are documented for trusted local files, not a path sandbox. |
| Versioned AOT bytes | AOT boundary | `ffi_phonetic_files_aot.rs::generated_header_corruption_never_returns_a_partial_rule_handle` mutates magic/version and checks output sentinels are untouched; `generated_aot_roundtrip_survives_repeated_owned_byte_release` varies the decoded handle's input, checks exact native rewriting after freeing serialized bytes, and repeats release safely. A separately compiled no-serialization property varies input bytes and capacity while proving all four AOT entry points return `Unsupported` without publishing output. Fixed tests also compare byte-for-byte native serialization. |

## Limits of the claim

TLC explores complete *finite abstractions* of the ownership, result, and
streaming protocols. It does not prove unbounded algorithmic exactness,
termination of user-defined rewrite rules, all filesystem graphs, or binary
parser security. The public Rust differential oracle, bounded C properties,
Julia metamorphic tests, parser corruption tests, and native upstream tests
are separate layers of evidence. Future unbounded proofs may refine these
models, but neither a model check nor a 32-case property run is a universal
correctness certificate.
